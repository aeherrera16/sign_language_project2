#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOCES_DIR="$ROOT/prototipo/voces"
mkdir -p "$VOCES_DIR"

echo "== Instalando Piper TTS =="
python3 -m pip install --user --upgrade piper-tts

BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES"

descargar() {
  local rel="$1"
  local out="$2"
  if [ -f "$VOCES_DIR/$out" ]; then
    echo "Ya existe: $out"
  else
    echo "Descargando: $out"
    curl -L --fail "$BASE/$rel" -o "$VOCES_DIR/$out"
  fi
}

# Voz femenina aproximada.
descargar "sharvard/medium/es_ES-sharvard-medium.onnx" "es_ES-sharvard-medium.onnx"
descargar "sharvard/medium/es_ES-sharvard-medium.onnx.json" "es_ES-sharvard-medium.onnx.json"

# Voz masculina aproximada.
descargar "davefx/medium/es_ES-davefx-medium.onnx" "es_ES-davefx-medium.onnx"
descargar "davefx/medium/es_ES-davefx-medium.onnx.json" "es_ES-davefx-medium.onnx.json"

echo
echo "Listo. Prueba:"
echo "  echo 'Hola, soy el traductor LSE' | piper --model '$VOCES_DIR/es_ES-sharvard-medium.onnx' --output_file /tmp/lse_voz.wav && aplay /tmp/lse_voz.wav"
