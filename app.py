#!/usr/bin/env python3
"""
Servidor local de transcripción de audio (español/multilingüe).
Levanta una web en http://127.0.0.1:5005 donde cargas un audio y ves la
transcripción aparecer en pantalla. Todo el procesamiento es LOCAL.
"""

import io
import json
import os
import tempfile
import threading
import webbrowser

from flask import Flask, request, Response, send_from_directory, stream_with_context

app = Flask(__name__, static_folder=None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache de modelos ya cargados, para no recargarlos en cada petición.
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

    # Guardamos el audio en un archivo temporal.
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
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return Response(stream_with_context(generate()),
                    mimetype="application/x-ndjson")


def _sse(obj: dict) -> str:
    """Una línea NDJSON por evento."""
    return json.dumps(obj, ensure_ascii=False) + "\n"


def open_browser():
    webbrowser.open("http://127.0.0.1:5005")


if __name__ == "__main__":
    print("\n  Transcriptor local en marcha:  http://127.0.0.1:5005")
    print("  (Para detenerlo: cierra esta ventana o pulsa Ctrl+C)\n")
    threading.Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=5005, threaded=True, debug=False)
