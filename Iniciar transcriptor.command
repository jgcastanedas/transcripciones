#!/bin/bash
# Doble clic para arrancar el transcriptor. Crea un entorno aislado,
# instala lo necesario la primera vez y abre la web en el navegador.

cd "$(dirname "$0")" || exit 1

echo "==============================================="
echo "  Transcriptor de audio local"
echo "==============================================="

# Elegir python3
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo "No encuentro Python 3. Instálalo desde https://www.python.org/downloads/ y vuelve a intentarlo."
  read -r -p "Pulsa Enter para cerrar."
  exit 1
fi

# Entorno virtual (solo se crea una vez)
if [ ! -d ".venv" ]; then
  echo "Preparando el entorno por primera vez…"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Verificando dependencias…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "Listo. Abriendo el navegador en http://127.0.0.1:5005"
echo "Para DETENER el transcriptor: cierra esta ventana o pulsa Ctrl+C."
echo ""

python app.py
