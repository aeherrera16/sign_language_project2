#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOCES_DIR="$ROOT/prototipo/voces"
mkdir -p "$VOCES_DIR"

echo "== Instalando Piper TTS =="
python3 -m pip install --user --upgrade piper-tts --break-system-packages

BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES"

descargar() {
  local rel="$1"
  local out="$2"
  local url="$BASE/$rel"
  if [[ "$rel" == http* ]]; then
    url="$rel"
  fi
  if [ -f "$VOCES_DIR/$out" ]; then
    echo "Ya existe: $out"
  else
    echo "Descargando: $out"
    curl -L --fail "$url" -o "$VOCES_DIR/$out"
  fi
}

# Solo se descarga la voz de alta calidad. Las demás están obsoletas.
echo "Descargando voz de alta calidad (es_MX-claude-high)..."
descargar "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx" "es_MX-claude-high.onnx"
descargar "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json" "es_MX-claude-high.onnx.json"

echo
echo "Listo. Prueba:"
echo "  echo 'Hola, soy el traductor LSE' | piper --model '$VOCES_DIR/es_MX-claude-high.onnx' --output_file /tmp/lse_voz.wav && aplay /tmp/lse_voz.wav"
