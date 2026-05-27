#!/bin/bash
# ============================================================
#  TRADUCTOR LSE - Lanzador para Raspberry Pi
#  Equivalente a Iniciar_LSE.bat de Windows.
#
#  MODO 1 - EJECUTABLE (artefacto de GitHub Actions):
#    Si existe ./TraductorLSE (binario PyInstaller), lo lanza
#    directamente. Solo instala espeak si falta (~10 segundos).
#
#  MODO 2 - CÓDIGO FUENTE (git clone):
#    Si no hay ejecutable, instala Python + deps la primera vez
#    (~20-40 min) y ejecuta desde código fuente.
# ============================================================

set -e

# Directorio del script
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}  🤟  TRADUCTOR LSE - Raspberry Pi${NC}"
    echo -e "${BLUE}  Lengua de Señas Ecuatoriana${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo ""
}

info()  { echo -e "${GREEN}  ✅ $1${NC}"; }
warn()  { echo -e "${YELLOW}  ⚠️  $1${NC}"; }
error() { echo -e "${RED}  ❌ $1${NC}"; }

paso() {
    echo ""
    echo -e "${BLUE}  [$1] $2${NC}"
    echo -e "${BLUE}  $(printf '%.0s─' $(seq 1 50))${NC}"
}

# ============================================================
#  Verificar plataforma
# ============================================================
verificar_plataforma() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        error "Este script es solo para Raspberry Pi (Linux)."
        error "Para Windows usa Iniciar_LSE.bat"
        exit 1
    fi
}

# ============================================================
#  Instalar espeak (único requisito del sistema para TTS)
#  Solo se ejecuta si no está instalado (~10 segundos)
# ============================================================
instalar_espeak() {
    if ! command -v espeak &>/dev/null; then
        paso "🔧" "Instalando espeak (síntesis de voz)..."
        sudo apt-get update -qq > /dev/null 2>&1
        sudo apt-get install -y espeak espeak-data > /dev/null 2>&1
        info "espeak instalado"
    fi
}

# ============================================================
#  Verificar hardware (cámara y audio) — CADA ejecución
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
        for dev in /dev/video*; do
            # Verificar que sea captura real (no metadata)
            if command -v v4l2-ctl &>/dev/null; then
                if v4l2-ctl --device="$dev" --all 2>/dev/null | grep -q "Video Capture"; then
                    CAMARA_DETECTADA=true
                    CAMARA_NOMBRE=$(v4l2-ctl --device="$dev" --info 2>/dev/null | grep "Card" | sed 's/.*: //' || echo "$dev")
                    break
                fi
            else
                # Sin v4l2-ctl, asumir que /dev/video* es cámara
                CAMARA_DETECTADA=true
                CAMARA_NOMBRE="$dev"
                break
            fi
        done

        # Fallback: si hay /dev/video* pero v4l2-ctl no confirmó
        if [ "$CAMARA_DETECTADA" = false ]; then
            CAMARA_DETECTADA=true
            CAMARA_NOMBRE=$(ls /dev/video* | head -1)
        fi
    fi

    # Pi Camera (CSI)
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
        echo -e "${YELLOW}     Conecta una cámara USB o Pi Camera${NC}"
        HARDWARE_OK=false
    fi

    # --- AUDIO / PARLANTE ---
    AUDIO_DETECTADO=false
    AUDIO_NOMBRE=""

    # Método 1: ALSA
    if command -v aplay &>/dev/null; then
        DISPOSITIVOS_AUDIO=$(aplay -l 2>/dev/null | grep "^card" || true)
        if [ -n "$DISPOSITIVOS_AUDIO" ]; then
            AUDIO_DETECTADO=true
            AUDIO_NOMBRE=$(echo "$DISPOSITIVOS_AUDIO" | head -1 | sed 's/card [0-9]*: //' | sed 's/\[.*//' | xargs)
        fi
    fi

    # Método 2: PulseAudio
    if [ "$AUDIO_DETECTADO" = false ] && command -v pactl &>/dev/null; then
        SINKS=$(pactl list short sinks 2>/dev/null || true)
        if [ -n "$SINKS" ]; then
            AUDIO_DETECTADO=true
            AUDIO_NOMBRE=$(pactl list short sinks 2>/dev/null | head -1 | awk '{print $2}' || echo "PulseAudio")
        fi
    fi

    # Método 3: PipeWire
    if [ "$AUDIO_DETECTADO" = false ] && command -v pw-cli &>/dev/null; then
        if pw-cli list-objects 2>/dev/null | grep -q "Audio/Sink"; then
            AUDIO_DETECTADO=true
            AUDIO_NOMBRE="PipeWire Audio"
        fi
    fi

    if [ "$AUDIO_DETECTADO" = true ]; then
        info "🔊 Audio detectado: $AUDIO_NOMBRE"

        # Verificar volumen
        if command -v amixer &>/dev/null; then
            VOLUMEN=$(amixer get Master 2>/dev/null | grep -oP '\[\d+%\]' | head -1 || echo "")
            if [ -n "$VOLUMEN" ]; then
                if [ "$VOLUMEN" = "[0%]" ]; then
                    warn "🔇 Volumen en 0% — subir con: amixer set Master 80%"
                else
                    info "🔉 Volumen: $VOLUMEN"
                fi
            fi
        fi
    else
        error "🔊 No se detectó dispositivo de audio"
        echo -e "${YELLOW}     Conecta un parlante USB o por jack 3.5mm${NC}"
        HARDWARE_OK=false
    fi

    # --- RESUMEN ---
    echo ""
    if [ "$HARDWARE_OK" = true ]; then
        info "Hardware listo ✅"
    else
        echo -e "${YELLOW}  ⚠️  Hardware faltante. La app puede no funcionar bien.${NC}"
        echo ""
        read -t 15 -p "  ¿Continuar de todas formas? [S/n] (auto en 15s): " RESPUESTA || RESPUESTA="s"
        echo ""
        RESPUESTA=${RESPUESTA:-s}
        if [[ ! "$RESPUESTA" =~ ^[sS]$ ]]; then
            error "Cancelado. Conecta el hardware y vuelve a intentar."
            exit 1
        fi
        warn "Continuando sin hardware completo..."
    fi
}

# ============================================================
#  Configurar inicio automático (solo la primera vez)
# ============================================================
configurar_autostart() {
    AUTOSTART_DIR="$HOME/.config/autostart"
    DESKTOP_FILE="$AUTOSTART_DIR/traductor_lse.desktop"

    if [ -f "$DESKTOP_FILE" ]; then
        return  # Ya configurado
    fi

    paso "🔄" "Configurando inicio automático..."

    mkdir -p "$AUTOSTART_DIR"

    # Determinar comando de ejecución
    if [ -f "$DIR/TraductorLSE" ]; then
        EXEC_CMD="bash -c 'sleep 5 && cd $DIR && ./TraductorLSE'"
    else
        EXEC_CMD="bash -c 'sleep 5 && cd $DIR && ./Iniciar_LSE_RaspberryPi.sh'"
    fi

    cat > "$DESKTOP_FILE" << DESKTOP
[Desktop Entry]
Type=Application
Name=Traductor LSE
Comment=Traductor de Lengua de Señas Ecuatoriana
Exec=$EXEC_CMD
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
DESKTOP

    info "Inicio automático configurado"
    info "La app se abrirá sola al encender el Pi"
}

# ============================================================
#  MODO 1: Lanzar ejecutable PyInstaller (artefacto)
# ============================================================
lanzar_ejecutable() {
    info "Ejecutable encontrado → lanzando directo"

    # Variables de entorno
    export TF_CPP_MIN_LOG_LEVEL=3
    export TF_ENABLE_ONEDNN_OPTS=0
    export MEDIAPIPE_DISABLE_GPU=1
    export GLOG_minloglevel=3
    export ABSL_MIN_LOG_LEVEL=3
    export PYTHONWARNINGS=ignore

    echo ""
    echo -e "${BLUE}  🚀 Iniciando Traductor LSE...${NC}"
    echo ""

    cd "$DIR"
    ./TraductorLSE
}

# ============================================================
#  MODO 2: Ejecutar desde código fuente (fallback)
#  Para quienes clonan el repo en vez de usar el artefacto
# ============================================================
VENV_DIR="$DIR/.venv_pi"
MARKER="$VENV_DIR/.setup_completo"
LOG="$DIR/setup_raspberry.log"

instalar_sistema() {
    paso "1/4" "Instalando dependencias del sistema..."
    sudo apt-get update -qq >> "$LOG" 2>&1
    sudo apt-get install -y \
        python3-pip python3-venv python3-dev \
        libopencv-dev python3-opencv libatlas-base-dev libhdf5-dev \
        python3-tk espeak espeak-data libespeak-dev \
        alsa-utils v4l-utils \
        build-essential cmake gfortran libopenblas-dev libjpeg-dev libpng-dev \
        >> "$LOG" 2>&1
    info "Dependencias del sistema instaladas"
}

crear_venv() {
    paso "2/4" "Creando entorno virtual Python..."
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip >> "$LOG" 2>&1
    info "Entorno virtual listo"
}

instalar_python() {
    paso "3/4" "Instalando paquetes Python..."
    echo "  Esto puede tardar 10-30 minutos en Raspberry Pi..."
    source "$VENV_DIR/bin/activate"
    pip install numpy >> "$LOG" 2>&1
    pip install opencv-python-headless >> "$LOG" 2>&1 || pip install opencv-python >> "$LOG" 2>&1
    pip install mediapipe >> "$LOG" 2>&1
    pip install tflite-runtime >> "$LOG" 2>&1 || pip install tensorflow >> "$LOG" 2>&1
    pip install pyttsx3 >> "$LOG" 2>&1
    pip install scikit-learn >> "$LOG" 2>&1
    pip install firebase-admin >> "$LOG" 2>&1
    info "Paquetes Python instalados"
}

lanzar_desde_fuente() {
    echo ""
    echo -e "${BLUE}  🚀 Iniciando Traductor LSE (modo directo)...${NC}"
    echo ""
    source "$VENV_DIR/bin/activate"
    export TF_CPP_MIN_LOG_LEVEL=3
    export TF_ENABLE_ONEDNN_OPTS=0
    export MEDIAPIPE_DISABLE_GPU=1
    export GLOG_minloglevel=3
    export ABSL_MIN_LOG_LEVEL=3
    export PYTHONWARNINGS=ignore
    cd "$DIR"
    # Modo directo: va al traductor sin menú, con auto-reinicio
    python3 prototipo/modo_traductor.py --loop
}

setup_desde_fuente() {
    echo -e "${YELLOW}  Ejecutable no encontrado.${NC}"
    echo -e "${YELLOW}  Ejecutando desde código fuente...${NC}"

    if [ -f "$MARKER" ]; then
        info "Dependencias ya instaladas."
        verificar_hardware
        lanzar_desde_fuente
    else
        echo -e "${YELLOW}  Primera ejecución. Instalando todo (~20-40 min)...${NC}"
        echo "  Log: $LOG"
        echo ""
        instalar_sistema
        crear_venv
        instalar_python
        paso "4/4" "Finalizando..."
        echo "$(date)" > "$MARKER"
        info "Configuración completada"
        configurar_autostart
        verificar_hardware
        lanzar_desde_fuente
    fi
}

# ============================================================
#  MAIN
# ============================================================
main() {
    verificar_plataforma
    banner

    if [ -f "$DIR/TraductorLSE" ]; then
        # MODO 1: Ejecutable PyInstaller presente → rápido
        instalar_espeak
        configurar_autostart
        verificar_hardware
        lanzar_ejecutable
    else
        # MODO 2: Sin ejecutable → instalar desde fuente
        setup_desde_fuente
    fi
}

main "$@"
