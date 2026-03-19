#!/bin/bash
# ============================================================
#  TRADUCTOR LSE - Configuración y Lanzamiento (Raspberry Pi)
#  Equivalente a Iniciar_LSE.bat de Windows.
#
#  Este script hace TODO automáticamente:
#  1. Instala dependencias del sistema (solo la primera vez)
#  2. Crea entorno virtual Python (solo la primera vez)
#  3. Instala paquetes Python (solo la primera vez)
#  4. Convierte el modelo a TFLite (solo la primera vez)
#  5. Lanza la aplicación
#
#  La primera ejecución tarda ~20-40 minutos (instalaciones).
#  Las siguientes ejecuciones inician en ~5 segundos.
# ============================================================

set -e

# Directorio del script (donde está el proyecto)
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

VENV_DIR="$DIR/.venv_pi"
MARKER="$VENV_DIR/.setup_completo"
LOG="$DIR/setup_raspberry.log"

# Colores para mensajes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

banner() {
    echo ""
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}  🤟  TRADUCTOR LSE - Raspberry Pi${NC}"
    echo -e "${BLUE}  Lengua de Señas Ecuatoriana${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo ""
}

info() {
    echo -e "${GREEN}  ✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}  ⚠️  $1${NC}"
}

error() {
    echo -e "${RED}  ❌ $1${NC}"
}

paso() {
    echo ""
    echo -e "${BLUE}  [$1] $2${NC}"
    echo -e "${BLUE}  $(printf '%.0s─' $(seq 1 50))${NC}"
}

# ============================================================
#  Verificar que estamos en Raspberry Pi (o Linux ARM)
# ============================================================
verificar_plataforma() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        error "Este script es solo para Raspberry Pi (Linux ARM)."
        error "Para Windows usa Iniciar_LSE.bat"
        exit 1
    fi
}

# ============================================================
#  PASO 1: Instalar dependencias del sistema
# ============================================================
instalar_sistema() {
    paso "1/5" "Instalando dependencias del sistema..."
    echo "  (Esto solo ocurre la primera vez)"
    echo ""

    # Actualizar repositorios
    echo "  Actualizando repositorios..."
    sudo apt-get update -qq >> "$LOG" 2>&1

    # Lista de paquetes necesarios
    PAQUETES=(
        # Python
        python3-pip
        python3-venv
        python3-dev
        # OpenCV dependencias
        libopencv-dev
        python3-opencv
        libatlas-base-dev
        libhdf5-dev
        # GUI (tkinter)
        python3-tk
        # TTS (text-to-speech)
        espeak
        espeak-data
        libespeak-dev
        # Audio
        alsa-utils
        # Cámara
        v4l-utils
        # Compilación (para paquetes pip que necesitan compilar)
        build-essential
        cmake
        gfortran
        libopenblas-dev
        libjpeg-dev
        libpng-dev
    )

    echo "  Instalando paquetes (puede tardar unos minutos)..."
    sudo apt-get install -y "${PAQUETES[@]}" >> "$LOG" 2>&1

    info "Dependencias del sistema instaladas"
}

# ============================================================
#  PASO 2: Crear entorno virtual Python
# ============================================================
crear_venv() {
    paso "2/5" "Creando entorno virtual Python..."

    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        info "Entorno virtual creado en $VENV_DIR"
    else
        info "Entorno virtual ya existe"
    fi

    # Activar
    source "$VENV_DIR/bin/activate"

    # Actualizar pip
    echo "  Actualizando pip..."
    pip install --upgrade pip >> "$LOG" 2>&1
    info "pip actualizado"
}

# ============================================================
#  PASO 3: Instalar paquetes Python
# ============================================================
instalar_python() {
    paso "3/5" "Instalando paquetes Python..."
    echo "  (opencv, mediapipe, tensorflow-lite, pyttsx3, scikit-learn)"
    echo "  Esto puede tardar 10-30 minutos en Raspberry Pi..."
    echo ""

    source "$VENV_DIR/bin/activate"

    # Instalar numpy primero (otras dependen de él)
    echo "  [a] numpy..."
    pip install numpy >> "$LOG" 2>&1
    info "numpy instalado"

    # OpenCV (usar la versión headless primero, más ligera)
    echo "  [b] opencv..."
    pip install opencv-python-headless >> "$LOG" 2>&1 || \
    pip install opencv-python >> "$LOG" 2>&1
    info "opencv instalado"

    # MediaPipe
    echo "  [c] mediapipe..."
    pip install mediapipe >> "$LOG" 2>&1
    info "mediapipe instalado"

    # TensorFlow Lite (MUCHO más ligero que TF completo para Pi)
    echo "  [d] tflite-runtime..."
    pip install tflite-runtime >> "$LOG" 2>&1 || {
        # Si tflite-runtime no está disponible, instalar TF completo como fallback
        warn "tflite-runtime no disponible, instalando tensorflow (más pesado)..."
        pip install tensorflow >> "$LOG" 2>&1
    }
    info "motor de inferencia instalado"

    # TTS
    echo "  [e] pyttsx3..."
    pip install pyttsx3 >> "$LOG" 2>&1
    info "pyttsx3 instalado"

    # scikit-learn
    echo "  [f] scikit-learn..."
    pip install scikit-learn >> "$LOG" 2>&1
    info "scikit-learn instalado"

    echo ""
    info "Todas las dependencias Python instaladas"
}

# ============================================================
#  PASO 4: Convertir modelo a TFLite (optimización para Pi)
# ============================================================
convertir_modelo() {
    paso "4/5" "Optimizando modelo para Raspberry Pi..."

    MODELO_H5="$DIR/prototipo/modelo/modelo.h5"
    MODELO_TFLITE="$DIR/prototipo/modelo/modelo.tflite"

    if [ -f "$MODELO_TFLITE" ]; then
        info "Modelo TFLite ya existe. Saltando conversión."
        return
    fi

    if [ ! -f "$MODELO_H5" ]; then
        warn "No hay modelo entrenado todavía."
        warn "Graba señas y entrena el modelo desde la app."
        return
    fi

    source "$VENV_DIR/bin/activate"

    echo "  Convirtiendo modelo.h5 → modelo.tflite..."

    python3 -c "
import os, sys
try:
    import tensorflow as tf
    modelo = tf.keras.models.load_model('$MODELO_H5')
    converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('$MODELO_TFLITE', 'wb') as f:
        f.write(tflite_model)
    print(f'  Modelo convertido: {len(tflite_model) / 1024:.0f} KB')
except Exception as e:
    print(f'  Aviso: No se pudo convertir a TFLite: {e}')
    print(f'  Se usará el modelo .h5 original (más lento)')
" 2>> "$LOG"

    if [ -f "$MODELO_TFLITE" ]; then
        info "Modelo optimizado para Pi (TFLite)"
    else
        warn "Se usará modelo original .h5 (funcionará, pero más lento)"
    fi
}

# ============================================================
#  PASO 5: Configurar inicio automático
# ============================================================
configurar_autostart() {
    paso "5/5" "Configurando inicio automático..."

    AUTOSTART_DIR="$HOME/.config/autostart"
    DESKTOP_FILE="$AUTOSTART_DIR/traductor_lse.desktop"

    mkdir -p "$AUTOSTART_DIR"

    cat > "$DESKTOP_FILE" << DESKTOP
[Desktop Entry]
Type=Application
Name=Traductor LSE
Comment=Traductor de Lengua de Señas Ecuatoriana
Exec=bash -c 'sleep 5 && cd $DIR && $VENV_DIR/bin/python prototipo/menu.py'
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
DESKTOP

    info "Inicio automático configurado"
    info "La app se abrirá sola al encender el Pi"
}

# ============================================================
#  Marcar setup como completado
# ============================================================
marcar_completo() {
    echo "$(date)" > "$MARKER"
    echo ""
    echo -e "${GREEN}=============================================${NC}"
    echo -e "${GREEN}  ✅ CONFIGURACIÓN COMPLETADA${NC}"
    echo -e "${GREEN}=============================================${NC}"
    echo -e "${GREEN}  La próxima vez que enciendas el Pi,${NC}"
    echo -e "${GREEN}  la app se abrirá automáticamente.${NC}"
    echo -e "${GREEN}=============================================${NC}"
    echo ""
}

# ============================================================
#  Verificar hardware (cámara y audio) — se ejecuta CADA vez
# ============================================================
verificar_hardware() {
    echo ""
    echo -e "${BLUE}  🔍 Verificando hardware...${NC}"
    echo -e "${BLUE}  $(printf '%.0s─' $(seq 1 50))${NC}"

    HARDWARE_OK=true

    # --- CÁMARA ---
    CAMARA_DETECTADA=false
    CAMARA_NOMBRE=""

    # Buscar dispositivos de video
    if ls /dev/video* 1>/dev/null 2>&1; then
        # Hay al menos un /dev/video*
        for dev in /dev/video*; do
            # Verificar que sea un dispositivo de captura real (no metadata)
            if v4l2-ctl --device="$dev" --all 2>/dev/null | grep -q "Video Capture"; then
                CAMARA_DETECTADA=true
                CAMARA_NOMBRE=$(v4l2-ctl --device="$dev" --info 2>/dev/null | grep "Card" | sed 's/.*: //' || echo "$dev")
                break
            fi
        done

        # Si v4l2-ctl no encontró captura, pero hay dispositivos, asumir que hay cámara
        if [ "$CAMARA_DETECTADA" = false ] && ls /dev/video* 1>/dev/null 2>&1; then
            CAMARA_DETECTADA=true
            CAMARA_NOMBRE=$(ls /dev/video* | head -1)
        fi
    fi

    # También verificar cámara oficial del Pi (CSI)
    if [ "$CAMARA_DETECTADA" = false ]; then
        if command -v libcamera-hello &>/dev/null; then
            if libcamera-hello --list-cameras 2>/dev/null | grep -q "Available"; then
                CAMARA_DETECTADA=true
                CAMARA_NOMBRE="Pi Camera (CSI)"
            fi
        fi
    fi

    if [ "$CAMARA_DETECTADA" = true ]; then
        info "📷 Cámara detectada: $CAMARA_NOMBRE"
    else
        error "📷 No se detectó ninguna cámara"
        echo -e "${YELLOW}     Conecta una cámara USB o Pi Camera y reinicia${NC}"
        echo -e "${YELLOW}     Verificar manualmente: ls /dev/video*${NC}"
        HARDWARE_OK=false
    fi

    # --- AUDIO / PARLANTE ---
    AUDIO_DETECTADO=false
    AUDIO_NOMBRE=""

    # Método 1: Verificar con aplay (ALSA)
    if command -v aplay &>/dev/null; then
        DISPOSITIVOS_AUDIO=$(aplay -l 2>/dev/null | grep "^card" || true)
        if [ -n "$DISPOSITIVOS_AUDIO" ]; then
            AUDIO_DETECTADO=true
            AUDIO_NOMBRE=$(echo "$DISPOSITIVOS_AUDIO" | head -1 | sed 's/card [0-9]*: //' | sed 's/\[.*//' | xargs)
        fi
    fi

    # Método 2: Verificar con PulseAudio
    if [ "$AUDIO_DETECTADO" = false ] && command -v pactl &>/dev/null; then
        SINKS=$(pactl list short sinks 2>/dev/null || true)
        if [ -n "$SINKS" ]; then
            AUDIO_DETECTADO=true
            AUDIO_NOMBRE=$(pactl list short sinks 2>/dev/null | head -1 | awk '{print $2}' || echo "PulseAudio")
        fi
    fi

    # Método 3: Verificar con PipeWire
    if [ "$AUDIO_DETECTADO" = false ] && command -v pw-cli &>/dev/null; then
        if pw-cli list-objects 2>/dev/null | grep -q "Audio/Sink"; then
            AUDIO_DETECTADO=true
            AUDIO_NOMBRE="PipeWire Audio"
        fi
    fi

    if [ "$AUDIO_DETECTADO" = true ]; then
        info "🔊 Audio detectado: $AUDIO_NOMBRE"

        # Intentar verificar que el volumen no esté en 0
        if command -v amixer &>/dev/null; then
            VOLUMEN=$(amixer get Master 2>/dev/null | grep -oP '\[\d+%\]' | head -1 || echo "")
            if [ -n "$VOLUMEN" ]; then
                if [ "$VOLUMEN" = "[0%]" ]; then
                    warn "🔇 Volumen en 0% — el TTS no se escuchará"
                    echo -e "${YELLOW}     Subir volumen: amixer set Master 80%${NC}"
                else
                    info "🔉 Volumen: $VOLUMEN"
                fi
            fi
        fi
    else
        error "🔊 No se detectó ningún dispositivo de audio"
        echo -e "${YELLOW}     Conecta un parlante USB o por jack 3.5mm${NC}"
        echo -e "${YELLOW}     Verificar manualmente: aplay -l${NC}"
        HARDWARE_OK=false
    fi

    # --- RESUMEN ---
    echo ""
    if [ "$HARDWARE_OK" = true ]; then
        info "✅ Hardware listo"
    else
        echo -e "${YELLOW}  ⚠️  Hay hardware faltante. La app puede no funcionar correctamente.${NC}"
        echo ""
        read -t 15 -p "  ¿Continuar de todas formas? [S/n] (auto-continúa en 15s): " RESPUESTA || RESPUESTA="s"
        echo ""
        RESPUESTA=${RESPUESTA:-s}
        if [[ ! "$RESPUESTA" =~ ^[sS]$ ]]; then
            error "Cancelado por el usuario. Conecta el hardware y vuelve a intentar."
            exit 1
        fi
        warn "Continuando sin hardware completo..."
    fi
}

# ============================================================
#  Lanzar la aplicación
# ============================================================
lanzar_app() {
    echo ""
    echo -e "${BLUE}  🚀 Iniciando Traductor LSE...${NC}"
    echo ""

    source "$VENV_DIR/bin/activate"

    # Variables de entorno para silenciar warnings
    export TF_CPP_MIN_LOG_LEVEL=3
    export TF_ENABLE_ONEDNN_OPTS=0
    export MEDIAPIPE_DISABLE_GPU=1
    export GLOG_minloglevel=3
    export ABSL_MIN_LOG_LEVEL=3
    export PYTHONWARNINGS=ignore

    cd "$DIR"
    python3 prototipo/menu.py
}

# ============================================================
#  MAIN: Orquestar todo
# ============================================================
main() {
    verificar_plataforma
    banner

    if [ -f "$MARKER" ]; then
        # Ya está configurado → verificar hardware y lanzar
        info "Sistema ya configurado."
        verificar_hardware
        lanzar_app
    else
        # Primera ejecución → instalar todo
        echo -e "${YELLOW}  Primera ejecución detectada.${NC}"
        echo -e "${YELLOW}  Se instalará todo automáticamente.${NC}"
        echo -e "${YELLOW}  Esto tarda ~20-40 minutos. No apagues el Pi.${NC}"
        echo ""
        echo "  Log detallado en: $LOG"
        echo ""

        instalar_sistema
        crear_venv
        instalar_python
        convertir_modelo
        configurar_autostart
        marcar_completo
        verificar_hardware
        lanzar_app
    fi
}

main "$@"
