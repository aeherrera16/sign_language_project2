#!/bin/bash
# ============================================================
#  Build script para crear ejecutable ARM64 (Raspberry Pi)
#  Se ejecuta DENTRO de un contenedor Docker ARM64 en CI.
#
#  Este script hace lo mismo que el job de Windows:
#  instala deps → construye con PyInstaller → genera artefacto
# ============================================================

set -e

echo ""
echo "=========================================="
echo "  BUILD: TraductorLSE ARM64 (Raspberry Pi)"
echo "=========================================="
echo "  Arquitectura: $(uname -m)"
echo "  Python: $(python3 --version)"
echo "  Fecha: $(date)"
echo ""

# ============================================================
#  1. Dependencias del sistema para compilar paquetes pip
# ============================================================
echo "[1/5] Instalando dependencias del sistema..."

apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    cmake \
    gfortran \
    libhdf5-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tk-dev \
    tcl-dev \
    > /dev/null 2>&1

echo "  ✅ Dependencias del sistema instaladas"

# ============================================================
#  2. Paquetes Python
# ============================================================
echo ""
echo "[2/5] Instalando paquetes Python..."
echo "  (Esto tarda varios minutos en ARM64)"

pip install --no-cache-dir --upgrade pip > /dev/null 2>&1

# Instalar numpy primero (muchos paquetes dependen de él)
echo "  → numpy..."
pip install --no-cache-dir numpy > /dev/null 2>&1

# PyInstaller
echo "  → pyinstaller..."
pip install --no-cache-dir pyinstaller > /dev/null 2>&1

# OpenCV headless (más ligero, sin GUI nativa)
echo "  → opencv..."
pip install --no-cache-dir opencv-python-headless > /dev/null 2>&1

# MediaPipe
echo "  → mediapipe..."
pip install --no-cache-dir mediapipe > /dev/null 2>&1

# TensorFlow (necesario para entrenar modelos en el Pi)
echo "  → tensorflow (esto tarda más)..."
pip install --no-cache-dir tensorflow > /dev/null 2>&1

# TTS
echo "  → pyttsx3..."
pip install --no-cache-dir pyttsx3 > /dev/null 2>&1

# scikit-learn
echo "  → scikit-learn..."
pip install --no-cache-dir scikit-learn > /dev/null 2>&1

echo "  ✅ Paquetes Python instalados"

# ============================================================
#  3. Limpiar paquetes innecesarios (reducir tamaño)
# ============================================================
echo ""
echo "[3/5] Limpiando paquetes innecesarios..."

pip uninstall -y tensorboard tensorboard-data-server tensorboard-plugin-wit 2>/dev/null || true

# Eliminar directorios de tensorboard del site-packages
SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
for pkg in tensorboard tensorboard_data_server tensorboard_plugin_wit; do
    if [ -d "$SITE_PKG/$pkg" ]; then
        rm -rf "$SITE_PKG/$pkg"
        echo "  Eliminado: $pkg"
    fi
done

echo "  ✅ Limpieza completada"

# ============================================================
#  4. Convertir modelo a TFLite (optimización para Pi)
# ============================================================
echo ""
echo "[4/5] Convirtiendo modelo a TFLite..."

if [ -f prototipo/modelo/modelo.h5 ]; then
    python3 -c "
import tensorflow as tf
modelo = tf.keras.models.load_model('prototipo/modelo/modelo.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('prototipo/modelo/modelo.tflite', 'wb') as f:
    f.write(tflite_model)
print(f'  ✅ modelo.tflite generado ({len(tflite_model) / 1024:.0f} KB)')
" 2>/dev/null || echo "  ⚠️ No se pudo convertir (se usará .h5)"
else
    echo "  ⚠️ Sin modelo.h5 — el usuario entrenará en el Pi"
fi

# ============================================================
#  5. Construir con PyInstaller
# ============================================================
echo ""
echo "[5/5] Construyendo ejecutable con PyInstaller..."
echo "  (Esto tarda varios minutos)"

pyinstaller build/traductor_lse_raspberry.spec --noconfirm --clean 2>&1 | tail -20

# Verificar que se creó
if [ ! -d "dist/TraductorLSE" ]; then
    echo "  ❌ ERROR: PyInstaller no generó la carpeta dist/TraductorLSE"
    exit 1
fi

# ============================================================
#  Copiar launcher y README al artefacto
# ============================================================
echo ""
echo "Copiando archivos adicionales..."

cp Iniciar_LSE_RaspberryPi.sh dist/TraductorLSE/
chmod +x dist/TraductorLSE/Iniciar_LSE_RaspberryPi.sh
chmod +x dist/TraductorLSE/TraductorLSE

# Crear README
cat > dist/TraductorLSE/README_RaspberryPi.txt << 'README'
============================================================
  TRADUCTOR LSE - Raspberry Pi
  Lengua de Señas Ecuatoriana
============================================================

REQUISITOS:
  - Raspberry Pi 4 (4GB RAM) o Pi 5
  - Raspberry Pi OS 64-bit
  - Cámara USB o Pi Camera
  - Altavoz (USB o 3.5mm)
  - Pantalla HDMI

USO:
  1. Copia esta carpeta al Raspberry Pi
  2. Abre una terminal en esta carpeta
  3. Ejecuta:

     chmod +x Iniciar_LSE_RaspberryPi.sh
     ./Iniciar_LSE_RaspberryPi.sh

  La primera vez instala espeak (~10 segundos).
  Después arranca directamente.

  O ejecuta el binario directamente:
     ./TraductorLSE
============================================================
README

# Renombrar carpeta final
mv dist/TraductorLSE dist/TraductorLSE-RaspberryPi

# ============================================================
#  Resumen
# ============================================================
echo ""
echo "=========================================="
echo "  ✅ BUILD COMPLETADO"
echo "=========================================="
echo ""
echo "  Contenido:"
ls -lh dist/TraductorLSE-RaspberryPi/ | head -20
echo ""
echo "  Tamaño total:"
du -sh dist/TraductorLSE-RaspberryPi/
echo ""
