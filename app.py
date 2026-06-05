#!/usr/bin/env python3
"""
Servidor local de transcripción de audio y video (español/multilingüe).
Levanta una web en http://127.0.0.1:5005. Todo el procesamiento es LOCAL.
"""

import base64
import json
import math
import os
import subprocess
import tempfile
import threading
import webbrowser

from flask import Flask, request, Response, send_from_directory, stream_with_context

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = None  # sin límite de tamaño de upload

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


def get_video_duration(video_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.decode().strip())
    except Exception:
        pass
    return 0.0


def extract_audio_range(video_path: str, audio_path: str,
                        start_sec: float = 0.0, duration_sec: float = None) -> None:
    """Extrae audio del video original en el rango [start_sec, start_sec+duration_sec]."""
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", video_path]
    if duration_sec is not None:
        cmd += ["-t", str(duration_sec)]
    cmd += ["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())


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


def transcribe_audio(model, audio_path: str, language):
    """Transcribe con VAD; si falla, reintenta sin VAD."""
    try:
        segments, info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return segments, info
    except Exception:
        return model.transcribe(audio_path, language=language)


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
            segments, info = transcribe_audio(model, tmp.name, language)
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
    language   = request.form.get("language", "es").strip() or None
    chunk_mins = int(request.form.get("chunk_minutes", "0") or "0")
    if language == "auto":
        language = None

    suffix = os.path.splitext(f.filename or "video")[1] or ".mp4"
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp_video.name)
    tmp_video.close()

    def generate():
        audio_paths = []
        try:
            if not check_ffmpeg():
                yield _sse({"type": "error", "message": "ffmpeg no encontrado. Instálalo con: brew install ffmpeg"})
                return

            video_mb       = os.path.getsize(tmp_video.name) // 1024 // 1024
            total_duration = get_video_duration(tmp_video.name)

            # ── Calcular rangos de tiempo ────────────────────────────
            if chunk_mins > 0 and total_duration > 0:
                chunk_secs = chunk_mins * 60
                n_chunks   = math.ceil(total_duration / chunk_secs)
                offsets    = [i * chunk_secs for i in range(n_chunks)]
                yield _sse({"type": "status", "message": f"Video de {video_mb} MB ({fmt_ts(total_duration)}) → {n_chunks} partes de {chunk_mins} min"})
            else:
                chunk_secs = None
                offsets    = [0.0]
                n_chunks   = 1
                yield _sse({"type": "status", "message": f"Video de {video_mb} MB ({fmt_ts(total_duration)}). Cargando modelo…"})

            yield _sse({"type": "status", "message": f"Cargando modelo '{model_name}'…"})
            model = get_model(model_name)

            first_chunk = True

            # ── Procesar cada rango de tiempo ────────────────────────
            for idx, start_offset in enumerate(offsets):
                part_label = f"parte {idx + 1} de {n_chunks}" if n_chunks > 1 else "video"

                yield _sse({"type": "status", "message": f"Extrayendo audio de {part_label}…"})

                tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_audio.close()
                audio_paths.append(tmp_audio.name)

                try:
                    extract_audio_range(
                        tmp_video.name, tmp_audio.name,
                        start_sec=start_offset,
                        duration_sec=chunk_secs,
                    )
                except RuntimeError as exc:
                    lines = [l for l in str(exc).splitlines() if l.strip()]
                    yield _sse({"type": "error", "message": f"ffmpeg ({part_label}): {lines[-1] if lines else exc}"})
                    return

                if os.path.getsize(tmp_audio.name) < 1000:
                    yield _sse({"type": "error", "message": f"Audio vacío en {part_label}. ¿El video tiene pista de audio?"})
                    return

                yield _sse({"type": "status", "message": f"Transcribiendo {part_label}…"})
                segments, info = transcribe_audio(model, tmp_audio.name, language)

                if first_chunk:
                    yield _sse({
                        "type": "info",
                        "language": info.language,
                        "language_probability": round(float(info.language_probability), 2),
                        "duration": total_duration or round(float(info.duration), 1),
                        "total_parts": n_chunks,
                    })
                    first_chunk = False
                elif n_chunks > 1:
                    yield _sse({"type": "chunk", "part": idx + 1, "total": n_chunks})

                for seg in segments:
                    orig_start = seg.start + start_offset
                    orig_end   = seg.end   + start_offset
                    screenshot = extract_frame(tmp_video.name, orig_start)
                    event = {
                        "type":      "segment",
                        "start":     fmt_ts(orig_start),
                        "end":       fmt_ts(orig_end),
                        "start_sec": round(orig_start, 2),
                        "text":      seg.text.strip(),
                    }
                    if screenshot:
                        event["screenshot"] = screenshot
                    yield _sse(event)

                # liberar audio de esta parte inmediatamente
                try:
                    os.unlink(tmp_audio.name)
                    audio_paths.remove(tmp_audio.name)
                except OSError:
                    pass

            yield _sse({"type": "done"})

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            try:
                os.unlink(tmp_video.name)
            except OSError:
                pass
            for p in audio_paths:
                try:
                    os.unlink(p)
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
