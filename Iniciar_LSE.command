#!/bin/bash
# ============================================================
#  TRADUCTOR LSE - Lengua de Señas Ecuatoriana
#  Doble clic para abrir. Instala todo automáticamente.
# ============================================================

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Si no existe el entorno virtual, crearlo e instalar dependencias
if [ ! -f ".venv/bin/activate" ]; then
    echo ""
    echo "==============================================="
    echo "  Primera ejecución - Configurando..."
    echo "  Esto solo ocurre una vez."
    echo "==============================================="
    echo ""

    # Buscar Python 3.10 o el python3 disponible
    if command -v python3.10 &> /dev/null; then
        PY=python3.10
    elif command -v python3 &> /dev/null; then
        PY=python3
    else
        echo "❌ Python 3 no está instalado."
        echo "Instala Python 3.10 con: brew install python@3.10"
        read -p "Presiona ENTER para cerrar..."
        exit 1
    fi

    echo "[1/3] Creando entorno virtual con $PY..."
    $PY -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Error al crear el entorno virtual."
        read -p "Presiona ENTER para cerrar..."
        exit 1
    fi

    source .venv/bin/activate

    echo "[2/3] Actualizando pip..."
    pip install --upgrade pip > /dev/null 2>&1

    echo "[3/3] Instalando dependencias (puede tardar unos minutos)..."
    pip install opencv-python mediapipe tensorflow pyttsx3 scikit-learn
    if [ $? -ne 0 ]; then
        echo "❌ Error al instalar dependencias."
        read -p "Presiona ENTER para cerrar..."
        exit 1
    fi

    echo ""
    echo "==============================================="
    echo "  ✅ Configuración completada!"
    echo "  Abriendo la aplicación..."
    echo "==============================================="
    echo ""
else
    source .venv/bin/activate
fi

# Silenciar warnings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export MEDIAPIPE_DISABLE_GPU=1
export GLOG_minloglevel=3
export ABSL_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

# Abrir menú gráfico
python "$DIR/prototipo/menu.py"
