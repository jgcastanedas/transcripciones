# Transcriptor de audio (local)

Una web que corre en tu propio Mac: cargas un audio y ves la transcripción en
pantalla, con marcas de tiempo. El audio **nunca sale de tu equipo**.

## Forma fácil (recomendada)

1. Doble clic en **`Iniciar transcriptor.command`**.
   - La primera vez macOS puede bloquearlo. Si pasa: clic derecho sobre el
     archivo → **Abrir** → **Abrir**. (O en Ajustes del Sistema → Privacidad y
     seguridad → "Abrir de todos modos".)
2. Se abre una ventana de Terminal, instala lo necesario (solo la 1ª vez) y
   abre el navegador en `http://127.0.0.1:5005`.
3. Arrastra el audio, elige modelo e idioma, y pulsa **Transcribir**.
4. Al terminar puedes **Copiar** o **Descargar .txt**.

Para cerrarlo: cierra la ventana de Terminal o pulsa `Ctrl+C`.

## Forma manual (si prefieres comandos)

Abre **Terminal**, ve a esta carpeta y ejecuta:

```bash
pip3 install -r requirements.txt
python3 app.py
```

Luego abre `http://127.0.0.1:5005` en el navegador.

## Notas

- **Primera vez**: cada modelo se descarga una sola vez. `medium` ≈ 1.5 GB,
  `large-v3` ≈ 3 GB. Después es rápido y funciona sin internet.
- **Modelos**: `tiny`/`base`/`small` son rápidos pero menos precisos;
  `medium` es buen equilibrio; `large-v3` es el más preciso (y más lento).
- **Formatos**: m4a, mp3, wav, ogg, mp4 y más (no necesita instalar ffmpeg).
- **Tiempo**: en CPU, ~13 min de audio con `medium` puede tardar varios
  minutos. Verás los segmentos aparecer en vivo mientras procesa.

## Archivos

- `app.py` — servidor local (Flask + faster-whisper).
- `index.html` — la interfaz web.
- `requirements.txt` — dependencias.
- `Iniciar transcriptor.command` — lanzador de doble clic para Mac.
