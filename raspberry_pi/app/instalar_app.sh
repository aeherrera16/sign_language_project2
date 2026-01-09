#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 🤟 INSTALADOR DE APLICACIÓN - TRADUCTOR LSE PARA RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════════
#
# Instala el Traductor LSE como una aplicación nativa de escritorio:
#   - Crea icono en el menú de aplicaciones
#   - Configura auto-inicio opcional
#   - Crea acceso directo en el escritorio
#
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

INSTALL_DIR="$USER_HOME/TraductorLSE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="$USER_HOME/Desktop"
APPLICATIONS_DIR="$USER_HOME/.local/share/applications"
AUTOSTART_DIR="$USER_HOME/.config/autostart"
ICONS_DIR="$USER_HOME/.local/share/icons"

echo -e "${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                       ║"
echo "║   🤟 INSTALADOR - TRADUCTOR LSE                                       ║"
echo "║      Aplicación de Escritorio para Raspberry Pi                       ║"
echo "║                                                                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# Verificar root
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ Ejecuta con: sudo ./instalar_app.sh${NC}"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 1: Instalar dependencias del sistema
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[1/6] 📦 Instalando dependencias del sistema...${NC}"

apt update
apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-tk \
    python3-pil \
    python3-pil.imagetk \
    espeak \
    v4l-utils \
    libatlas-base-dev \
    libjasper-dev \
    libhdf5-dev \
    libqt4-test

echo -e "${GREEN}✅ Dependencias instaladas${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 2: Crear directorios
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[2/6] 📂 Creando estructura de la aplicación...${NC}"

mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/model"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$APPLICATIONS_DIR"
mkdir -p "$ICONS_DIR"
mkdir -p "$AUTOSTART_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 3: Copiar archivos
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[3/6] 📦 Copiando archivos de la aplicación...${NC}"

# Copiar la aplicación
cp "$SCRIPT_DIR/traductor_lse_app.py" "$INSTALL_DIR/"

# Copiar modelo si existe
if [ -d "$SCRIPT_DIR/../model" ]; then
    cp -r "$SCRIPT_DIR/../model/"* "$INSTALL_DIR/model/"
elif [ -d "$SCRIPT_DIR/model" ]; then
    cp -r "$SCRIPT_DIR/model/"* "$INSTALL_DIR/model/"
fi

echo -e "${GREEN}✅ Archivos copiados${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 4: Crear entorno virtual
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[4/6] 🐍 Configurando entorno Python...${NC}"

sudo -u "$REAL_USER" python3 -m venv "$INSTALL_DIR/venv"
sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip

sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install \
    numpy \
    opencv-python-headless \
    mediapipe \
    Pillow

# Intentar instalar tflite-runtime
sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install tflite-runtime 2>/dev/null || \
    echo -e "${YELLOW}   ⚠️ tflite-runtime no disponible, instalando tensorflow...${NC}" && \
    sudo -u "$REAL_USER" "$INSTALL_DIR/venv/bin/pip" install tensorflow

echo -e "${GREEN}✅ Entorno Python configurado${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 5: Crear icono SVG
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[5/6] 🎨 Creando icono de la aplicación...${NC}"

cat > "$ICONS_DIR/traductor-lse.svg" << 'ICON_SVG'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#e94560;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0f3460;stop-opacity:1" />
    </linearGradient>
  </defs>
  <circle cx="50" cy="50" r="48" fill="url(#grad1)"/>
  <g fill="white" transform="translate(20, 15) scale(0.6)">
    <!-- Mano estilizada -->
    <path d="M50 10 C40 10 35 20 35 35 L35 55 C35 60 38 65 45 65 L55 65 C62 65 65 60 65 55 L65 35 C65 20 60 10 50 10 Z"/>
    <circle cx="35" cy="25" r="8"/>
    <circle cx="65" cy="25" r="8"/>
    <circle cx="30" cy="45" r="6"/>
    <circle cx="70" cy="45" r="6"/>
    <rect x="45" y="65" width="10" height="25" rx="5"/>
  </g>
  <text x="50" y="92" font-family="Arial" font-size="12" fill="white" text-anchor="middle" font-weight="bold">LSE</text>
</svg>
ICON_SVG

# Convertir a PNG si está disponible inkscape o rsvg-convert
if command -v rsvg-convert &> /dev/null; then
    rsvg-convert -w 128 -h 128 "$ICONS_DIR/traductor-lse.svg" > "$ICONS_DIR/traductor-lse.png"
elif command -v inkscape &> /dev/null; then
    inkscape "$ICONS_DIR/traductor-lse.svg" -o "$ICONS_DIR/traductor-lse.png" -w 128 -h 128 2>/dev/null || true
fi

echo -e "${GREEN}✅ Icono creado${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 6: Crear archivo .desktop
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[6/6] 🖥️ Creando acceso en menú de aplicaciones...${NC}"

# Crear script lanzador
cat > "$INSTALL_DIR/lanzar.sh" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
exec "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/traductor_lse_app.py"
EOF
chmod +x "$INSTALL_DIR/lanzar.sh"

# Archivo .desktop para el menú de aplicaciones
cat > "$APPLICATIONS_DIR/traductor-lse.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Traductor LSE
GenericName=Sign Language Translator
Comment=Traductor de Lengua de Señas Ecuatoriana en tiempo real
Exec=$INSTALL_DIR/lanzar.sh
Icon=$ICONS_DIR/traductor-lse.svg
Terminal=false
Categories=Utility;Accessibility;Education;
Keywords=sign;language;translator;deaf;accessibility;señas;
StartupNotify=true
StartupWMClass=TraductorLSE
EOF

# Copiar al escritorio
if [ -d "$DESKTOP_DIR" ]; then
    cp "$APPLICATIONS_DIR/traductor-lse.desktop" "$DESKTOP_DIR/"
    chmod +x "$DESKTOP_DIR/traductor-lse.desktop"
fi

# Actualizar cache de iconos
update-desktop-database "$APPLICATIONS_DIR" 2>/dev/null || true
gtk-update-icon-cache -f -t "$ICONS_DIR" 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════════════════
# Preguntar sobre auto-inicio
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}¿Deseas que la aplicación inicie automáticamente al encender?${NC}"
read -p "(s/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Ss]$ ]]; then
    cat > "$AUTOSTART_DIR/traductor-lse.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Traductor LSE
Exec=$INSTALL_DIR/lanzar.sh
Hidden=false
X-GNOME-Autostart-enabled=true
Comment=Inicia el traductor de lengua de señas automáticamente
EOF
    chown "$REAL_USER:$REAL_USER" "$AUTOSTART_DIR/traductor-lse.desktop"
    echo -e "${GREEN}✅ Auto-inicio configurado${NC}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Ajustar permisos
# ═══════════════════════════════════════════════════════════════════════════════
chown -R "$REAL_USER:$REAL_USER" "$INSTALL_DIR"
chown "$REAL_USER:$REAL_USER" "$APPLICATIONS_DIR/traductor-lse.desktop"
chown "$REAL_USER:$REAL_USER" "$ICONS_DIR/traductor-lse.svg"
[ -f "$ICONS_DIR/traductor-lse.png" ] && chown "$REAL_USER:$REAL_USER" "$ICONS_DIR/traductor-lse.png"
[ -f "$DESKTOP_DIR/traductor-lse.desktop" ] && chown "$REAL_USER:$REAL_USER" "$DESKTOP_DIR/traductor-lse.desktop"

# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                       ║"
echo "║   ✅ ¡APLICACIÓN INSTALADA CORRECTAMENTE!                            ║"
echo "║                                                                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}📂 Ubicación:${NC} $INSTALL_DIR"
echo ""
echo -e "${YELLOW}🚀 CÓMO ABRIR:${NC}"
echo ""
echo "   1. Buscar 'Traductor LSE' en el menú de aplicaciones"
echo ""
echo "   2. Hacer doble clic en el icono del escritorio"
echo ""
echo "   3. Desde terminal: $INSTALL_DIR/lanzar.sh"
echo ""
echo -e "${PURPLE}🤟 ¡Conecta una cámara y comienza a traducir!${NC}"
echo ""
