#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 🤟 INSTALADOR - TRADUCTOR LSE PARA RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Detectar usuario
if [ "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
else
    REAL_USER="$(whoami)"
fi
USER_HOME=$(eval echo ~$REAL_USER)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$USER_HOME/TraductorLSE"
DESKTOP_DIR="$USER_HOME/Desktop"
APPLICATIONS_DIR="$USER_HOME/.local/share/applications"
ICONS_DIR="$USER_HOME/.local/share/icons"

echo -e "${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                       ║"
echo "║   🤟 INSTALADOR - TRADUCTOR LSE                                       ║"
echo "║                                                                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Verificar root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ Ejecuta con: sudo ./instalar.sh${NC}"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 1: Dependencias
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[1/5] 📦 Instalando dependencias...${NC}"

apt update

# Instalar paquetes base (compatibles con Debian Trixie/Bookworm)
apt install -y python3 python3-pip python3-venv python3-tk python3-pil python3-pil.imagetk \
    espeak-ng v4l-utils || {
    echo -e "${YELLOW}Intentando paquetes alternativos...${NC}"
    apt install -y python3 python3-pip python3-venv python3-tk espeak v4l-utils
}

# Instalar libatlas si está disponible (opcional para numpy)
apt install -y libatlas-base-dev 2>/dev/null || apt install -y libopenblas-dev 2>/dev/null || true

echo -e "${GREEN}✅ Dependencias instaladas${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 2: Copiar archivos
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[2/5] 📂 Instalando aplicación...${NC}"

mkdir -p "$INSTALL_DIR"
cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
chown -R "$REAL_USER:$REAL_USER" "$INSTALL_DIR"

echo -e "${GREEN}✅ Archivos copiados${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 3: Entorno Python (usando Python 3.11 para compatibilidad con MediaPipe)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[3/5] 🐍 Configurando Python...${NC}"

# Verificar versión de Python
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "   Python detectado: $PYTHON_VERSION"

# MediaPipe no soporta Python 3.13, intentar instalar Python 3.11
if [[ "$PYTHON_VERSION" == "3.13" ]] || [[ "$PYTHON_VERSION" > "3.12" ]]; then
    echo -e "${YELLOW}   Python $PYTHON_VERSION no es compatible con MediaPipe${NC}"
    echo -e "${YELLOW}   Instalando Python 3.11...${NC}"
    apt install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null || {
        echo -e "${RED}   No se pudo instalar Python 3.11${NC}"
        echo -e "${YELLOW}   Intentando con paquetes del sistema...${NC}"
    }
    PYTHON_CMD="python3.11"
else
    PYTHON_CMD="python3"
fi

# Verificar que el Python elegido existe
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${YELLOW}   Usando Python del sistema...${NC}"
    PYTHON_CMD="python3"
fi

echo -e "   Usando: $PYTHON_CMD"

# Crear entorno virtual
sudo -u "$REAL_USER" $PYTHON_CMD -m venv "$INSTALL_DIR/venv"
sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip

# Instalar dependencias con piwheels (optimizado para Raspberry Pi)
echo -e "${BLUE}   Instalando paquetes Python (esto puede tardar unos minutos)...${NC}"
sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install \
    --extra-index-url https://www.piwheels.org/simple \
    numpy opencv-python-headless Pillow scikit-learn

# Intentar instalar MediaPipe
echo -e "${BLUE}   Instalando MediaPipe...${NC}"
sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install mediapipe 2>/dev/null || {
    echo -e "${YELLOW}   MediaPipe no disponible, instalando desde GitHub...${NC}"
    sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install \
        mediapipe-rpi4 2>/dev/null || \
    sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install \
        --extra-index-url https://google-coral.github.io/py-repo/ \
        mediapipe 2>/dev/null || {
        echo -e "${RED}   ⚠️ MediaPipe no se pudo instalar${NC}"
        echo -e "${YELLOW}   La aplicación usará modo básico${NC}"
    }
}

# Intentar tensorflow o tflite-runtime
echo -e "${BLUE}   Instalando TensorFlow Lite...${NC}"
sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install tflite-runtime 2>/dev/null || \
    sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install tensorflow 2>/dev/null || {
        echo -e "${YELLOW}   ⚠️ TensorFlow no instalado, se usará modelo alternativo${NC}"
    }

echo -e "${GREEN}✅ Python configurado${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 4: Crear icono
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[4/5] 🎨 Creando icono...${NC}"

mkdir -p "$ICONS_DIR"

cat > "$ICONS_DIR/traductor-lse.svg" << 'ICON'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#e94560"/>
      <stop offset="100%" style="stop-color:#0f3460"/>
    </linearGradient>
  </defs>
  <circle cx="50" cy="50" r="48" fill="url(#g)"/>
  <text x="50" y="60" font-size="40" fill="white" text-anchor="middle">🤟</text>
</svg>
ICON

chown "$REAL_USER:$REAL_USER" "$ICONS_DIR/traductor-lse.svg"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 5: Crear lanzador
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[5/5] 🖥️ Creando accesos directos...${NC}"

mkdir -p "$APPLICATIONS_DIR"

# Script lanzador
cat > "$INSTALL_DIR/lanzar.sh" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
exec "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/iniciar.py"
EOF
chmod +x "$INSTALL_DIR/lanzar.sh"

# .desktop para menú
cat > "$APPLICATIONS_DIR/traductor-lse.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Traductor LSE
GenericName=Sign Language Translator
Comment=Traductor de Lengua de Señas Ecuatoriana
Exec=$INSTALL_DIR/lanzar.sh
Icon=$ICONS_DIR/traductor-lse.svg
Terminal=false
Categories=Utility;Accessibility;Education;
StartupNotify=true
EOF

# Copiar al escritorio
if [ -d "$DESKTOP_DIR" ]; then
    cp "$APPLICATIONS_DIR/traductor-lse.desktop" "$DESKTOP_DIR/"
    chmod +x "$DESKTOP_DIR/traductor-lse.desktop"
    chown "$REAL_USER:$REAL_USER" "$DESKTOP_DIR/traductor-lse.desktop"
fi

chown "$REAL_USER:$REAL_USER" "$APPLICATIONS_DIR/traductor-lse.desktop"
update-desktop-database "$APPLICATIONS_DIR" 2>/dev/null || true

echo -e "\n${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║   ✅ ¡INSTALACIÓN COMPLETADA!                                        ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Cómo usar:${NC}"
echo ""
echo "   1. Buscar 'Traductor LSE' en el menú de aplicaciones"
echo "   2. O hacer doble clic en el icono del escritorio"
echo "   3. O ejecutar: $INSTALL_DIR/lanzar.sh"
echo ""
echo -e "${BLUE}Primero entrena algunas señas, luego usa el traductor.${NC}"
echo ""
