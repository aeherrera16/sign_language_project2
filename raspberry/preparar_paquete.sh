#!/bin/bash
# ============================================================
#  Prepara el paquete de distribución para Raspberry Pi
#  Se ejecuta en CI (GitHub Actions) para crear el artefacto.
#
#  Genera una carpeta TraductorLSE-RaspberryPi/ que contiene
#  todo lo necesario para correr en un Pi.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_DIR/dist/TraductorLSE-RaspberryPi"

echo "=== Preparando paquete Raspberry Pi ==="
echo "  Proyecto: $PROJECT_DIR"
echo "  Salida:   $OUTPUT_DIR"

# Limpiar y crear directorio de salida
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/prototipo/modelo"
mkdir -p "$OUTPUT_DIR/prototipo/datos"

# ============================================================
#  1. Copiar archivos del prototipo
# ============================================================
echo ""
echo "[1/4] Copiando archivos del prototipo..."

cp "$PROJECT_DIR/prototipo/menu.py"               "$OUTPUT_DIR/prototipo/"
cp "$PROJECT_DIR/prototipo/1_grabar_senas.py"      "$OUTPUT_DIR/prototipo/"
cp "$PROJECT_DIR/prototipo/2_entrenar_modelo.py"   "$OUTPUT_DIR/prototipo/"
cp "$PROJECT_DIR/prototipo/3_traductor.py"         "$OUTPUT_DIR/prototipo/"
cp "$PROJECT_DIR/prototipo/4_evaluar_iso25023.py"  "$OUTPUT_DIR/prototipo/"
cp "$PROJECT_DIR/prototipo/utils_silenciar.py"     "$OUTPUT_DIR/prototipo/"
cp "$PROJECT_DIR/prototipo/icon.png"               "$OUTPUT_DIR/prototipo/"

# Copiar modelo si existe
if [ -f "$PROJECT_DIR/prototipo/modelo/modelo.h5" ]; then
    cp "$PROJECT_DIR/prototipo/modelo/modelo.h5" "$OUTPUT_DIR/prototipo/modelo/"
    echo "  ✅ modelo.h5 incluido"
fi
if [ -f "$PROJECT_DIR/prototipo/modelo/encoder.pkl" ]; then
    cp "$PROJECT_DIR/prototipo/modelo/encoder.pkl" "$OUTPUT_DIR/prototipo/modelo/"
    echo "  ✅ encoder.pkl incluido"
fi
if [ -f "$PROJECT_DIR/prototipo/modelo/info.json" ]; then
    cp "$PROJECT_DIR/prototipo/modelo/info.json" "$OUTPUT_DIR/prototipo/modelo/"
fi

# Copiar datos de entrenamiento si existen
if [ -d "$PROJECT_DIR/prototipo/datos" ] && [ "$(ls -A "$PROJECT_DIR/prototipo/datos" 2>/dev/null)" ]; then
    cp -r "$PROJECT_DIR/prototipo/datos/"* "$OUTPUT_DIR/prototipo/datos/" 2>/dev/null || true
    echo "  ✅ Datos de entrenamiento incluidos"
fi

echo "  ✅ Archivos del prototipo copiados"

# ============================================================
#  2. Copiar script de inicio
# ============================================================
echo ""
echo "[2/4] Copiando scripts de inicio..."

cp "$PROJECT_DIR/Iniciar_LSE_RaspberryPi.sh" "$OUTPUT_DIR/"
chmod +x "$OUTPUT_DIR/Iniciar_LSE_RaspberryPi.sh"

echo "  ✅ Iniciar_LSE_RaspberryPi.sh incluido"

# ============================================================
#  3. Convertir modelo a TFLite (si TensorFlow está disponible)
# ============================================================
echo ""
echo "[3/4] Convirtiendo modelo a TFLite..."

if [ -f "$PROJECT_DIR/prototipo/modelo/modelo.h5" ]; then
    python3 -c "
import sys
try:
    import tensorflow as tf
    modelo = tf.keras.models.load_model('$PROJECT_DIR/prototipo/modelo/modelo.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('$OUTPUT_DIR/prototipo/modelo/modelo.tflite', 'wb') as f:
        f.write(tflite_model)
    print(f'  ✅ modelo.tflite generado ({len(tflite_model) / 1024:.0f} KB)')
except Exception as e:
    print(f'  ⚠️  No se pudo convertir a TFLite: {e}')
    print(f'  El Pi usará modelo.h5 como fallback')
" 2>/dev/null || echo "  ⚠️  TensorFlow no disponible para conversión"
else
    echo "  ⚠️  Sin modelo.h5 - el usuario deberá entrenar en el Pi"
fi

# ============================================================
#  4. Crear README específico para Raspberry Pi
# ============================================================
echo ""
echo "[4/4] Generando documentación..."

cat > "$OUTPUT_DIR/README_RaspberryPi.txt" << 'README'
============================================================
  TRADUCTOR LSE - Raspberry Pi
  Lengua de Señas Ecuatoriana
============================================================

REQUISITOS:
  - Raspberry Pi 4 (4GB RAM) o Pi 5
  - Raspberry Pi OS 64-bit instalado
  - Cámara USB o Pi Camera
  - Altavoz (USB o 3.5mm)
  - Pantalla HDMI

INSTALACIÓN (solo la primera vez):

  1. Copia esta carpeta completa al Raspberry Pi
     (por USB, git clone, o SSH/SCP)

  2. Abre una terminal en esta carpeta y ejecuta:

     chmod +x Iniciar_LSE_RaspberryPi.sh
     ./Iniciar_LSE_RaspberryPi.sh

  3. La primera vez tarda ~20-40 minutos
     (instala todo automáticamente)

  4. Después de la primera vez, la app se abre
     automáticamente al encender el Pi.

EJECUCIÓN POSTERIOR:
  - La app inicia sola al encender el Pi
  - O ejecuta: ./Iniciar_LSE_RaspberryPi.sh

SOLUCIÓN DE PROBLEMAS:
  - Sin cámara: verificar con "ls /dev/video*"
  - Sin audio: ejecutar "alsamixer" y subir volumen
  - Log de instalación: ver setup_raspberry.log
============================================================
README

echo "  ✅ README_RaspberryPi.txt generado"

# ============================================================
#  Resumen
# ============================================================
echo ""
echo "=== Paquete Raspberry Pi listo ==="
echo ""
echo "  Contenido:"
find "$OUTPUT_DIR" -type f | sort | while read f; do
    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "?")
    rel="${f#$OUTPUT_DIR/}"
    printf "    %-50s %s bytes\n" "$rel" "$size"
done
echo ""
echo "  Carpeta: $OUTPUT_DIR"
