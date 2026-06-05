#!/usr/bin/env python3
"""
Servidor local de transcripción de audio y video (español/multilingüe).
Levanta una web en http://127.0.0.1:5005. Todo el procesamiento es LOCAL.
"""

import base64
import json
import os
import subprocess
import tempfile
import threading
import webbrowser

from flask import Flask, request, Response, send_from_directory, stream_with_context

app = Flask(__name__, static_folder=None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_MODELS = {}
_MODELS_LOCK = threading.Lock()


def get_model(name: str):
    from faster_whisper import WhisperModel
    with _MODELS_LOCK:
        if name not in _MODELS:
            _MODELS[name] = WhisperModel(name, device="cpu", compute_type="int8")
        return _MODELS[name]


def fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _sse(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def extract_audio(video_path: str, audio_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
        capture_output=True, check=True, timeout=300,
    )


def extract_frame(video_path: str, timestamp: float):
    """Devuelve un data-URI JPEG del fotograma en `timestamp`, o None si falla."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-ss", str(timestamp), "-i", video_path,
             "-vframes", "1", "-vf", "scale=480:-2",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "pipe:1"],
            capture_output=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout:
            b64 = base64.b64encode(result.stdout).decode()
            return f"data:image/jpeg;base64,{b64}"
        return None
    except Exception:
        return None


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return {"error": "No se recibió ningún archivo de audio."}, 400

    f = request.files["audio"]
    model_name = request.form.get("model", "medium")
    language = request.form.get("language", "es").strip() or None
    if language == "auto":
        language = None

    suffix = os.path.splitext(f.filename or "audio")[1] or ".m4a"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name)
    tmp.close()

    def generate():
        try:
            yield _sse({"type": "status", "message": f"Cargando modelo '{model_name}'…"})
            model = get_model(model_name)

            yield _sse({"type": "status", "message": "Analizando el audio…"})
            segments, info = model.transcribe(
                tmp.name,
                language=language,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            yield _sse({
                "type": "info",
                "language": info.language,
                "language_probability": round(float(info.language_probability), 2),
                "duration": round(float(info.duration), 1),
            })

            for seg in segments:
                yield _sse({
                    "type": "segment",
                    "start": fmt_ts(seg.start),
                    "end": fmt_ts(seg.end),
                    "start_sec": round(float(seg.start), 2),
                    "text": seg.text.strip(),
                })

            yield _sse({"type": "done"})
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.route("/process-video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return {"error": "No se recibió ningún archivo de video."}, 400

    f = request.files["video"]
    model_name = request.form.get("model", "medium")
    language = request.form.get("language", "es").strip() or None
    if language == "auto":
        language = None

    suffix = os.path.splitext(f.filename or "video")[1] or ".mp4"
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp_video.name)
    tmp_video.close()

    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_audio.close()

    def generate():
        try:
            if not check_ffmpeg():
                yield _sse({"type": "error", "message": "ffmpeg no encontrado. Instálalo con: brew install ffmpeg"})
                return

            yield _sse({"type": "status", "message": "Extrayendo audio del video…"})
            try:
                extract_audio(tmp_video.name, tmp_audio.name)
            except subprocess.CalledProcessError:
                yield _sse({"type": "error", "message": "Error al extraer el audio. ¿El video tiene pista de audio?"})
                return

            yield _sse({"type": "status", "message": f"Cargando modelo '{model_name}'…"})
            model = get_model(model_name)

            yield _sse({"type": "status", "message": "Analizando el audio…"})
            segments, info = model.transcribe(
                tmp_audio.name,
                language=language,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            yield _sse({
                "type": "info",
                "language": info.language,
                "language_probability": round(float(info.language_probability), 2),
                "duration": round(float(info.duration), 1),
            })

            for seg in segments:
                screenshot = extract_frame(tmp_video.name, seg.start)
                event = {
                    "type": "segment",
                    "start": fmt_ts(seg.start),
                    "end": fmt_ts(seg.end),
                    "start_sec": round(float(seg.start), 2),
                    "text": seg.text.strip(),
                }
                if screenshot:
                    event["screenshot"] = screenshot
                yield _sse(event)

            yield _sse({"type": "done"})
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            for path in [tmp_video.name, tmp_audio.name]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


def open_browser():
    webbrowser.open("http://127.0.0.1:5005")


if __name__ == "__main__":
    print("\n  Transcriptor local en marcha:  http://127.0.0.1:5005")
    print("  (Para detenerlo: cierra esta ventana o pulsa Ctrl+C)\n")
    threading.Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=5005, threaded=True, debug=False)
