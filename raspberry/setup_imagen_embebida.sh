#!/bin/bash
# ============================================================
#  SETUP: Imagen Embebida del Traductor LSE
#  Para Raspberry Pi 5 (8GB) con Raspberry Pi OS Lite 64-bit
#
#  Este script convierte tu Raspberry Pi en un DISPOSITIVO
#  DEDICADO que arranca directamente al Traductor LSE.
#
#  USO:
#    sudo bash raspberry/setup_imagen_embebida.sh
#
#  EJECUTAR DESDE: La raíz del proyecto (sign_language_project2/)
# ============================================================

set -euo pipefail

# ============================================================
#  CONFIGURACIÓN
# ============================================================
APP_DIR="/opt/traductor_lse"
REAL_USER="${SUDO_USER:-pi}"
REAL_HOME=$(eval echo "~${REAL_USER}")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="/tmp/setup_traductor_lse.log"
STEP=0
TOTAL_STEPS=9

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ============================================================
#  FUNCIONES AUXILIARES
# ============================================================
banner() {
    clear
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                  ║${NC}"
    echo -e "${CYAN}║   🤟  TRADUCTOR LSE — Setup Imagen Embebida     ║${NC}"
    echo -e "${CYAN}║   Raspberry Pi 5 — Sistema Dedicado              ║${NC}"
    echo -e "${CYAN}║                                                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

paso() {
    STEP=$((STEP + 1))
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  [${STEP}/${TOTAL_STEPS}] $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

info()    { echo -e "  ${GREEN}✅ $1${NC}"; }
warn()    { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
error()   { echo -e "  ${RED}❌ $1${NC}"; }
detalle() { echo -e "  ${CYAN}   → $1${NC}"; }

fallo() {
    error "$1"
    echo ""
    echo -e "${RED}  El log completo está en: ${LOG_FILE}${NC}"
    echo -e "${RED}  Revisa las últimas líneas con: tail -30 ${LOG_FILE}${NC}"
    echo ""
    exit 1
}

# ============================================================
#  VERIFICACIONES PREVIAS
# ============================================================
verificar_requisitos() {
    paso "Verificando requisitos"

    # Verificar que se ejecuta como root
    if [ "$(id -u)" -ne 0 ]; then
        error "Este script debe ejecutarse con sudo"
        echo "  Uso: sudo bash raspberry/setup_imagen_embebida.sh"
        exit 1
    fi
    info "Ejecutando como root (sudo)"

    # Verificar que estamos en Linux ARM64
    if [ "$(uname -s)" != "Linux" ]; then
        fallo "Este script es solo para Raspberry Pi (Linux)"
    fi

    ARCH=$(uname -m)
    if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "armv7l" ]; then
        fallo "Arquitectura no soportada: $ARCH (necesita aarch64)"
    fi
    info "Plataforma: Linux $ARCH"

    # Verificar el usuario real
    if [ -z "$REAL_USER" ] || [ "$REAL_USER" = "root" ]; then
        # Buscar un usuario no-root
        REAL_USER=$(ls /home/ | head -1)
        REAL_HOME="/home/$REAL_USER"
    fi
    info "Usuario del sistema: $REAL_USER"
    info "Home: $REAL_HOME"

    # Verificar que el proyecto existe
    if [ ! -f "$PROJECT_DIR/prototipo/menu.py" ]; then
        fallo "No se encontró el proyecto en $PROJECT_DIR"
    fi
    info "Proyecto encontrado en: $PROJECT_DIR"

    # Verificar espacio en disco
    ESPACIO_LIBRE=$(df / --output=avail -BG 2>/dev/null | tail -1 | tr -d ' G' || echo "unknown")
    if [ "$ESPACIO_LIBRE" != "unknown" ] && [ "$ESPACIO_LIBRE" -lt 8 ]; then
        warn "Espacio libre: ${ESPACIO_LIBRE}GB (se recomiendan mínimo 8GB)"
    else
        info "Espacio en disco: ${ESPACIO_LIBRE}GB disponibles"
    fi

    # Verificar conexión a internet
    if ping -c 1 -W 3 google.com &>/dev/null; then
        info "Conexión a internet: OK"
    else
        fallo "Sin conexión a internet. Conéctate por ethernet o WiFi primero."
    fi
}

# ============================================================
#  PASO 2: DEPENDENCIAS DEL SISTEMA
# ============================================================
instalar_dependencias_sistema() {
    paso "Instalando dependencias del sistema"
    echo "  Esto puede tardar unos minutos..."

    {
        apt-get update -qq
        
        # Paquetes esenciales
        apt-get install -y --no-install-recommends \
            python3-pip python3-venv python3-dev python3-tk \
            python3-numpy python3-opencv \
            build-essential cmake gfortran pkg-config \
            libatlas-base-dev libhdf5-dev libopenblas-dev \
            libjpeg-dev libpng-dev libgl1 libglib2.0-0 \
            libsm6 libxext6 libxrender1 \
            tk-dev tcl-dev

        # Audio (para espeak/pyttsx3)
        apt-get install -y --no-install-recommends \
            espeak espeak-data libespeak-dev \
            alsa-utils pulseaudio

        # Video (para cámara)
        apt-get install -y --no-install-recommends \
            v4l-utils

        # X11 mínimo (para tkinter y OpenCV GUI en modo kiosk)
        apt-get install -y --no-install-recommends \
            xserver-xorg x11-xserver-utils xinit \
            matchbox-window-manager \
            xdotool unclutter

    } >> "$LOG_FILE" 2>&1 || fallo "Error instalando paquetes del sistema"

    info "Todas las dependencias del sistema instaladas"
}

# ============================================================
#  PASO 3: COPIAR APLICACIÓN
# ============================================================
copiar_aplicacion() {
    paso "Copiando aplicación a $APP_DIR"

    # Crear directorio
    mkdir -p "$APP_DIR"

    # Copiar archivos del prototipo
    cp -r "$PROJECT_DIR/prototipo/"* "$APP_DIR/"
    info "Archivos del prototipo copiados"

    # Verificar archivos críticos
    local ARCHIVOS_CRITICOS=(
        "menu.py"
        "3_traductor.py"
        "1_grabar_senas.py"
        "2_entrenar_modelo.py"
        "utils_silenciar.py"
    )

    for archivo in "${ARCHIVOS_CRITICOS[@]}"; do
        if [ -f "$APP_DIR/$archivo" ]; then
            detalle "$archivo ✓"
        else
            warn "Falta: $archivo"
        fi
    done

    # Verificar modelo
    if [ -f "$APP_DIR/modelo/modelo.h5" ]; then
        local TAMANO=$(du -sh "$APP_DIR/modelo/modelo.h5" | cut -f1)
        info "Modelo encontrado: modelo.h5 ($TAMANO)"
    else
        warn "No hay modelo entrenado (modelo.h5)"
        warn "El usuario deberá entrenar el modelo desde la app"
    fi

    # Verificar datos
    if [ -d "$APP_DIR/datos" ]; then
        local NUM_SENAS=$(ls -d "$APP_DIR/datos/"*/ 2>/dev/null | wc -l)
        info "Datos de señas: $NUM_SENAS señas encontradas"
    else
        warn "No hay datos de señas (se podrán grabar desde la app)"
    fi

    # Asignar permisos al usuario
    chown -R "$REAL_USER:$REAL_USER" "$APP_DIR"
    info "Permisos asignados a $REAL_USER"
}

# ============================================================
#  PASO 4: ENTORNO VIRTUAL PYTHON
# ============================================================
crear_entorno_python() {
    paso "Creando entorno virtual Python"

    # Crear venv
    if [ ! -d "$APP_DIR/.venv" ]; then
        sudo -u "$REAL_USER" python3 -m venv "$APP_DIR/.venv" --system-site-packages
        info "Entorno virtual creado"
    else
        info "Entorno virtual ya existe"
    fi

    # Activar y actualizar pip
    source "$APP_DIR/.venv/bin/activate"
    pip install --upgrade pip >> "$LOG_FILE" 2>&1
    info "pip actualizado"
}

# ============================================================
#  PASO 5: DEPENDENCIAS PYTHON
# ============================================================
instalar_dependencias_python() {
    paso "Instalando dependencias Python (esto tarda 10-30 min)"

    source "$APP_DIR/.venv/bin/activate"

    # numpy (probablemente ya vino con --system-site-packages)
    echo -e "  ${CYAN}→ numpy...${NC}"
    pip install numpy >> "$LOG_FILE" 2>&1 || true
    info "numpy OK"

    # OpenCV
    echo -e "  ${CYAN}→ opencv-python...${NC}"
    pip install opencv-python-headless >> "$LOG_FILE" 2>&1 || \
        pip install opencv-python >> "$LOG_FILE" 2>&1 || \
        warn "opencv-python desde pip falló, usando sistema (python3-opencv)"
    info "OpenCV OK"

    # MediaPipe
    echo -e "  ${CYAN}→ mediapipe (puede tardar)...${NC}"
    pip install mediapipe >> "$LOG_FILE" 2>&1 || {
        warn "mediapipe estándar falló, intentando alternativas..."
        pip install mediapipe-rpi4 >> "$LOG_FILE" 2>&1 || \
        pip install mediapipe==0.10.9 >> "$LOG_FILE" 2>&1 || \
        fallo "No se pudo instalar mediapipe. Ver log: $LOG_FILE"
    }
    info "MediaPipe OK"

    # TensorFlow
    echo -e "  ${CYAN}→ tensorflow (esto es lo que más tarda)...${NC}"
    pip install tensorflow >> "$LOG_FILE" 2>&1 || {
        warn "tensorflow completo falló, intentando tflite-runtime..."
        pip install tflite-runtime >> "$LOG_FILE" 2>&1 || \
        fallo "No se pudo instalar tensorflow ni tflite-runtime"
    }
    info "TensorFlow OK"

    # pyttsx3 (Text-to-Speech)
    echo -e "  ${CYAN}→ pyttsx3...${NC}"
    pip install pyttsx3 >> "$LOG_FILE" 2>&1 || fallo "No se pudo instalar pyttsx3"
    info "pyttsx3 OK"

    # scikit-learn
    echo -e "  ${CYAN}→ scikit-learn...${NC}"
    pip install scikit-learn >> "$LOG_FILE" 2>&1 || fallo "No se pudo instalar scikit-learn"
    info "scikit-learn OK"

    deactivate
    info "Todas las dependencias Python instaladas"
}

# ============================================================
#  PASO 6: VERIFICAR QUE LA APP FUNCIONA
# ============================================================
verificar_app() {
    paso "Verificando que la aplicación funciona"

    source "$APP_DIR/.venv/bin/activate"

    # Test 1: Imports básicos
    echo -e "  ${CYAN}→ Verificando imports...${NC}"
    sudo -u "$REAL_USER" "$APP_DIR/.venv/bin/python3" -c "
import sys
sys.path.insert(0, '$APP_DIR')
errores = []

# Test imports uno por uno
try:
    import cv2
    print('  ✅ cv2 (OpenCV):', cv2.__version__)
except Exception as e:
    errores.append(f'cv2: {e}')
    print(f'  ❌ cv2: {e}')

try:
    import mediapipe as mp
    print('  ✅ mediapipe:', mp.__version__)
except Exception as e:
    errores.append(f'mediapipe: {e}')
    print(f'  ❌ mediapipe: {e}')

try:
    import tensorflow as tf
    print('  ✅ tensorflow:', tf.__version__)
except Exception:
    try:
        import tflite_runtime
        print('  ✅ tflite_runtime')
    except Exception as e:
        errores.append(f'tensorflow/tflite: {e}')
        print(f'  ❌ tensorflow/tflite: {e}')

try:
    import pyttsx3
    print('  ✅ pyttsx3')
except Exception as e:
    errores.append(f'pyttsx3: {e}')
    print(f'  ❌ pyttsx3: {e}')

try:
    import sklearn
    print('  ✅ scikit-learn:', sklearn.__version__)
except Exception as e:
    errores.append(f'sklearn: {e}')
    print(f'  ❌ scikit-learn: {e}')

try:
    import tkinter
    print('  ✅ tkinter')
except Exception as e:
    errores.append(f'tkinter: {e}')
    print(f'  ❌ tkinter: {e}')

if errores:
    print(f'\n  ⚠️ {len(errores)} errores encontrados')
    sys.exit(1)
else:
    print('\n  ✅ Todos los imports correctos')
" 2>>"$LOG_FILE" || {
        warn "Algunos imports fallaron. Revisa el log."
        warn "Continuando de todas formas..."
    }

    # Test 2: Verificar espeak
    if command -v espeak &>/dev/null; then
        info "espeak disponible"
    else
        warn "espeak no encontrado"
    fi

    deactivate
    info "Verificación completada"
}

# ============================================================
#  PASO 7: CONFIGURAR MODO KIOSK (ARRANQUE AUTOMÁTICO)
# ============================================================
configurar_kiosk() {
    paso "Configurando arranque automático (modo kiosk)"

    # --- 7a: Script de inicio X11 ---
    detalle "Creando script de inicio X11..."

    cat > "$APP_DIR/kiosk_xinit.sh" << 'XINIT_SCRIPT'
#!/bin/bash
# ============================================================
#  Traductor LSE - Sesión X11 Kiosk
#  Este script se ejecuta dentro de xinit
# ============================================================

# Desactivar salvapantallas y DPMS
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

# Ocultar cursor del ratón después de 0.5 segundos sin mover
unclutter -idle 0.5 -root &

# Window manager mínimo (necesario para tkinter)
# -use_titlebar no: sin barra de título
# -use_cursor no: sin cursor decorado
matchbox-window-manager -use_cursor no -use_titlebar no &

# Esperar a que el WM esté listo
sleep 1

# Variables de entorno para silenciar warnings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export MEDIAPIPE_DISABLE_GPU=1
export GLOG_minloglevel=3
export ABSL_MIN_LOG_LEVEL=3
export PYTHONWARNINGS=ignore

# Activar entorno virtual y lanzar la app
cd /opt/traductor_lse
source .venv/bin/activate

# Bucle infinito: si la app se cierra, la reinicia
while true; do
    python3 menu.py 2>/tmp/traductor_lse_error.log
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        # Salida limpia: el usuario cerró la app
        # Esperar 3 segundos y reiniciar
        sleep 3
    else
        # Error: esperar un poco más y reintentar
        sleep 5
    fi
done
XINIT_SCRIPT

    chmod +x "$APP_DIR/kiosk_xinit.sh"
    chown "$REAL_USER:$REAL_USER" "$APP_DIR/kiosk_xinit.sh"
    info "Script X11 kiosk creado"

    # --- 7b: Script de arranque desde TTY ---
    detalle "Creando script de arranque TTY..."

    cat > "$APP_DIR/kiosk_arranque.sh" << 'ARRANQUE_SCRIPT'
#!/bin/bash
# ============================================================
#  Traductor LSE - Arranque automático desde TTY1
#  Se ejecuta desde .bash_profile cuando el usuario inicia sesión
#  en la consola física (tty1), NO en SSH.
# ============================================================

# Solo ejecutar en tty1 (consola física, no SSH)
CURRENT_TTY=$(tty 2>/dev/null)
if [ "$CURRENT_TTY" = "/dev/tty1" ]; then
    echo ""
    echo "  🤟 Iniciando Traductor LSE..."
    echo "  (Para acceder por SSH: ssh $(whoami)@$(hostname -I | awk '{print $1}'))"
    echo ""
    sleep 2
    
    # Iniciar X11 con nuestra sesión kiosk
    xinit /opt/traductor_lse/kiosk_xinit.sh -- :0 -nocursor 2>/tmp/xinit_error.log
fi
ARRANQUE_SCRIPT

    chmod +x "$APP_DIR/kiosk_arranque.sh"
    chown "$REAL_USER:$REAL_USER" "$APP_DIR/kiosk_arranque.sh"
    info "Script de arranque TTY creado"

    # --- 7c: Configurar autologin en tty1 ---
    detalle "Configurando autologin en consola..."

    # Crear override de getty para autologin
    mkdir -p /etc/systemd/system/getty@tty1.service.d/
    cat > /etc/systemd/system/getty@tty1.service.d/autologin.conf << AUTOLOGIN
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin ${REAL_USER} --noclear %I \$TERM
AUTOLOGIN

    info "Autologin configurado para $REAL_USER en tty1"

    # --- 7d: Agregar a .bash_profile ---
    detalle "Configurando .bash_profile..."

    # Backup del .bash_profile
    if [ -f "$REAL_HOME/.bash_profile" ]; then
        cp "$REAL_HOME/.bash_profile" "$REAL_HOME/.bash_profile.bak"
    fi

    # Agregar el arranque kiosk (solo si no está ya)
    if ! grep -q "kiosk_arranque.sh" "$REAL_HOME/.bash_profile" 2>/dev/null; then
        cat >> "$REAL_HOME/.bash_profile" << 'PROFILE'

# ============================================================
# TRADUCTOR LSE - Arranque automático en modo kiosk
# Solo se activa en tty1 (consola física), no por SSH
# Para desactivar: comenta o borra estas líneas
# ============================================================
if [ -f /opt/traductor_lse/kiosk_arranque.sh ]; then
    source /opt/traductor_lse/kiosk_arranque.sh
fi
PROFILE
        chown "$REAL_USER:$REAL_USER" "$REAL_HOME/.bash_profile"
    fi
    info ".bash_profile configurado"

    info "Modo kiosk completamente configurado"
}

# ============================================================
#  PASO 8: OPTIMIZAR ARRANQUE (BOOT RÁPIDO Y SILENCIOSO)
# ============================================================
optimizar_arranque() {
    paso "Optimizando arranque (rápido y silencioso)"

    # --- 8a: Ocultar mensajes de boot ---
    detalle "Ocultando mensajes de boot..."

    # Modificar cmdline.txt para boot silencioso
    local CMDLINE_FILE=""
    if [ -f /boot/firmware/cmdline.txt ]; then
        CMDLINE_FILE="/boot/firmware/cmdline.txt"
    elif [ -f /boot/cmdline.txt ]; then
        CMDLINE_FILE="/boot/cmdline.txt"
    fi

    if [ -n "$CMDLINE_FILE" ]; then
        # Backup
        cp "$CMDLINE_FILE" "${CMDLINE_FILE}.bak"

        # Leer contenido actual
        CMDLINE=$(cat "$CMDLINE_FILE")

        # Agregar opciones de silencio si no están
        if ! echo "$CMDLINE" | grep -q "quiet"; then
            CMDLINE="$CMDLINE quiet"
        fi
        if ! echo "$CMDLINE" | grep -q "splash"; then
            CMDLINE="$CMDLINE splash"
        fi
        if ! echo "$CMDLINE" | grep -q "loglevel=0"; then
            CMDLINE="$CMDLINE loglevel=0"
        fi
        if ! echo "$CMDLINE" | grep -q "logo.nologo"; then
            CMDLINE="$CMDLINE logo.nologo"
        fi
        if ! echo "$CMDLINE" | grep -q "vt.global_cursor_default=0"; then
            CMDLINE="$CMDLINE vt.global_cursor_default=0"
        fi

        # Cambiar console=tty1 a console=tty3 (oculta mensajes)
        CMDLINE=$(echo "$CMDLINE" | sed 's/console=tty1/console=tty3/g')

        echo "$CMDLINE" > "$CMDLINE_FILE"
        info "cmdline.txt optimizado"
    else
        warn "No se encontró cmdline.txt"
    fi

    # --- 8b: Configurar config.txt ---
    detalle "Optimizando config.txt..."

    local CONFIG_FILE=""
    if [ -f /boot/firmware/config.txt ]; then
        CONFIG_FILE="/boot/firmware/config.txt"
    elif [ -f /boot/config.txt ]; then
        CONFIG_FILE="/boot/config.txt"
    fi

    if [ -n "$CONFIG_FILE" ]; then
        # Backup
        cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"

        # Agregar optimizaciones si no están
        if ! grep -q "# TRADUCTOR LSE" "$CONFIG_FILE"; then
            cat >> "$CONFIG_FILE" << 'BOOTCONFIG'

# ============================================================
# TRADUCTOR LSE - Optimizaciones de arranque
# ============================================================
# Desactivar splash de Raspberry Pi
disable_splash=1
# Arranque más rápido
boot_delay=0
# Forzar salida HDMI (para que siempre detecte pantalla)
hdmi_force_hotplug=1
# Activar cámara
start_x=1
gpu_mem=128
BOOTCONFIG
        fi
        info "config.txt optimizado"
    else
        warn "No se encontró config.txt"
    fi

    # --- 8c: Desactivar servicios innecesarios ---
    detalle "Desactivando servicios innecesarios..."

    local SERVICIOS_DESACTIVAR=(
        "bluetooth"
        "avahi-daemon"
        "triggerhappy"
        "ModemManager"
    )

    for servicio in "${SERVICIOS_DESACTIVAR[@]}"; do
        if systemctl is-enabled "$servicio" &>/dev/null; then
            systemctl disable "$servicio" >> "$LOG_FILE" 2>&1 || true
            detalle "Desactivado: $servicio"
        fi
    done

    info "Arranque optimizado"
}

# ============================================================
#  PASO 9: CREAR SCRIPT DE UTILIDADES
# ============================================================
crear_utilidades() {
    paso "Creando utilidades de mantenimiento"

    # Script para desactivar modo kiosk (debug)
    cat > "$APP_DIR/desactivar_kiosk.sh" << 'DESKIOSK'
#!/bin/bash
# Desactiva el modo kiosk para poder usar el Pi normalmente
echo "Desactivando modo kiosk..."
sed -i 's|source /opt/traductor_lse/kiosk_arranque.sh|# source /opt/traductor_lse/kiosk_arranque.sh|' ~/.bash_profile
echo "✅ Kiosk desactivado. Reinicia con: sudo reboot"
echo "Para reactivar: bash /opt/traductor_lse/activar_kiosk.sh"
DESKIOSK
    chmod +x "$APP_DIR/desactivar_kiosk.sh"

    # Script para activar modo kiosk
    cat > "$APP_DIR/activar_kiosk.sh" << 'AKIOSK'
#!/bin/bash
# Reactiva el modo kiosk
echo "Activando modo kiosk..."
sed -i 's|# source /opt/traductor_lse/kiosk_arranque.sh|source /opt/traductor_lse/kiosk_arranque.sh|' ~/.bash_profile
echo "✅ Kiosk activado. Reinicia con: sudo reboot"
AKIOSK
    chmod +x "$APP_DIR/activar_kiosk.sh"

    # Script para ver logs de errores
    cat > "$APP_DIR/ver_errores.sh" << 'VERROR'
#!/bin/bash
echo "=== Últimos errores del Traductor LSE ==="
echo ""
echo "--- Error de la app ---"
cat /tmp/traductor_lse_error.log 2>/dev/null || echo "(sin errores)"
echo ""
echo "--- Error de X11 ---"
cat /tmp/xinit_error.log 2>/dev/null || echo "(sin errores)"
echo ""
echo "--- Log de setup ---"
tail -20 /tmp/setup_traductor_lse.log 2>/dev/null || echo "(sin log)"
VERROR
    chmod +x "$APP_DIR/ver_errores.sh"

    chown -R "$REAL_USER:$REAL_USER" "$APP_DIR"

    info "Utilidades creadas en $APP_DIR/"
    detalle "desactivar_kiosk.sh — Para desactivar arranque automático"
    detalle "activar_kiosk.sh   — Para reactivar arranque automático"
    detalle "ver_errores.sh     — Para ver logs de errores"
}

# ============================================================
#  RESUMEN FINAL
# ============================================================
resumen_final() {
    echo ""
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                  ║${NC}"
    echo -e "${GREEN}║   ✅  SETUP COMPLETADO EXITOSAMENTE             ║${NC}"
    echo -e "${GREEN}║                                                  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}  Lo que se configuró:${NC}"
    echo -e "  ● Dependencias del sistema instaladas"
    echo -e "  ● Aplicación copiada a ${APP_DIR}"
    echo -e "  ● Python + dependencias en ${APP_DIR}/.venv"
    echo -e "  ● Arranque automático en modo kiosk"
    echo -e "  ● Boot rápido y silencioso"
    echo ""
    echo -e "${BOLD}  Qué pasará al reiniciar:${NC}"
    echo -e "  1. Pantalla negra durante ~10-15 segundos"
    echo -e "  2. El Traductor LSE aparece automáticamente"
    echo -e "  3. No verás terminal, escritorio, ni login"
    echo ""
    echo -e "${BOLD}  Acceso por SSH (para mantenimiento):${NC}"
    IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo -e "  ${CYAN}ssh ${REAL_USER}@${IP_ADDR}${NC}"
    echo ""
    echo -e "${BOLD}  Utilidades disponibles (por SSH):${NC}"
    echo -e "  ${CYAN}bash /opt/traductor_lse/desactivar_kiosk.sh${NC}  → Desactivar arranque auto"
    echo -e "  ${CYAN}bash /opt/traductor_lse/activar_kiosk.sh${NC}    → Reactivar arranque auto"
    echo -e "  ${CYAN}bash /opt/traductor_lse/ver_errores.sh${NC}      → Ver logs de errores"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  ¿Reiniciar ahora para activar el modo kiosk?${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    read -p "  ¿Reiniciar? [S/n]: " RESPUESTA
    RESPUESTA=${RESPUESTA:-s}
    if [[ "$RESPUESTA" =~ ^[sS]$ ]]; then
        echo ""
        echo -e "  ${GREEN}Reiniciando en 5 segundos...${NC}"
        echo -e "  ${CYAN}(Desconecta teclado/monitor si quieres la experiencia final)${NC}"
        sleep 5
        reboot
    else
        echo ""
        echo -e "  Para reiniciar manualmente: ${CYAN}sudo reboot${NC}"
        echo ""
    fi
}

# ============================================================
#  MAIN
# ============================================================
main() {
    banner

    echo -e "  ${BOLD}Este script convertirá tu Raspberry Pi en un${NC}"
    echo -e "  ${BOLD}dispositivo dedicado del Traductor LSE.${NC}"
    echo ""
    echo -e "  ${YELLOW}⏱️  Tiempo estimado: 20-40 minutos${NC}"
    echo -e "  ${YELLOW}📡 Se requiere conexión a internet${NC}"
    echo ""
    read -p "  ¿Continuar? [S/n]: " RESPUESTA
    RESPUESTA=${RESPUESTA:-s}
    if [[ ! "$RESPUESTA" =~ ^[sS]$ ]]; then
        echo "  Cancelado."
        exit 0
    fi

    # Iniciar log
    echo "=== Setup Traductor LSE ===" > "$LOG_FILE"
    echo "Fecha: $(date)" >> "$LOG_FILE"
    echo "Arquitectura: $(uname -m)" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    verificar_requisitos
    instalar_dependencias_sistema
    copiar_aplicacion
    crear_entorno_python
    instalar_dependencias_python
    verificar_app
    configurar_kiosk
    optimizar_arranque
    crear_utilidades
    resumen_final
}

main "$@"
