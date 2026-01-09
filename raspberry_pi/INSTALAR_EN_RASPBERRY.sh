#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 🤟 INSTALADOR AUTOMÁTICO TODO-EN-UNO - TRADUCTOR LSE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Este script instala AUTOMÁTICAMENTE todo lo necesario:
#   ✅ Python y dependencias del sistema
#   ✅ TensorFlow Lite (optimizado para Raspberry Pi)
#   ✅ MediaPipe, OpenCV, NumPy
#   ✅ La aplicación de escritorio
#   ✅ Iconos y accesos directos
#
# NO REQUIERE INTERVENCIÓN MANUAL - Todo se instala automáticamente
#
# Uso: sudo ./instalar_completo.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

echo -e "${PURPLE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   🤟 INSTALADOR AUTOMÁTICO - TRADUCTOR LSE                           ║
║                                                                       ║
║      📦 Instalación TODO-EN-UNO                                       ║
║      🚀 Sin configuración manual necesaria                            ║
║      ⚡ Todo se instala automáticamente                               ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# Verificar root
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ Este script requiere permisos de root${NC}"
    echo -e "${YELLOW}Ejecuta: sudo ./instalar_completo.sh${NC}"
    exit 1
fi

echo -e "${CYAN}👤 Instalando para usuario: $REAL_USER${NC}"
echo -e "${CYAN}📂 Directorio de instalación: $INSTALL_DIR${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 1: Actualizar repositorios
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[1/8] 🔄 Actualizando repositorios del sistema...${NC}"
apt-get update -qq
echo -e "${GREEN}✅ Repositorios actualizados${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 2: Instalar Python y dependencias del sistema
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[2/8] 📦 Instalando Python y dependencias del sistema...${NC}"

DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-tk \
    python3-pil \
    python3-pil.imagetk \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqt5gui5 \
    libqt5webkit5 \
    libqt5test5 \
    gfortran \
    espeak \
    espeak-ng \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    v4l-utils \
    git \
    wget \
    curl \
    > /dev/null 2>&1

echo -e "${GREEN}✅ Dependencias del sistema instaladas${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 3: Crear estructura de directorios
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[3/8] 📂 Creando estructura de directorios...${NC}"

mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/app"
mkdir -p "$INSTALL_DIR/model"
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$USER_HOME/.local/share/applications"
mkdir -p "$USER_HOME/.local/share/icons"
mkdir -p "$USER_HOME/.config/autostart"

echo -e "${GREEN}✅ Directorios creados${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 4: Crear entorno virtual Python
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[4/8] 🐍 Configurando entorno virtual Python...${NC}"

sudo -u "$REAL_USER" python3 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

# Actualizar pip
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

echo -e "${GREEN}✅ Entorno virtual creado${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 5: Instalar dependencias Python (AUTOMÁTICO)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[5/8] 📚 Instalando librerías Python (esto puede tardar 5-10 min)...${NC}"

# Instalar dependencias básicas primero
echo -e "${CYAN}   → Instalando NumPy, Pillow...${NC}"
pip install --no-cache-dir \
    numpy==1.24.3 \
    Pillow==10.0.1 \
    > /dev/null 2>&1

# OpenCV optimizado para Raspberry Pi
echo -e "${CYAN}   → Instalando OpenCV...${NC}"
pip install --no-cache-dir opencv-python-headless==4.8.1.78 > /dev/null 2>&1

# MediaPipe para detección de manos
echo -e "${CYAN}   → Instalando MediaPipe...${NC}"
pip install --no-cache-dir mediapipe==0.10.9 > /dev/null 2>&1

# TensorFlow Lite Runtime (ligero, optimizado para Raspberry Pi)
echo -e "${CYAN}   → Instalando TensorFlow Lite...${NC}"
# Intentar instalar tflite-runtime primero (más ligero)
if ! pip install --no-cache-dir tflite-runtime==2.14.0 > /dev/null 2>&1; then
    echo -e "${YELLOW}   ⚠️  tflite-runtime no disponible, instalando TensorFlow completo...${NC}"
    # Si falla, instalar TensorFlow completo
    pip install --no-cache-dir tensorflow==2.15.0 > /dev/null 2>&1
fi

# Scikit-learn para preprocesamiento
echo -e "${CYAN}   → Instalando Scikit-learn...${NC}"
pip install --no-cache-dir scikit-learn==1.3.2 > /dev/null 2>&1

# Otras utilidades
echo -e "${CYAN}   → Instalando utilidades adicionales...${NC}"
pip install --no-cache-dir \
    scipy==1.11.4 \
    joblib==1.3.2 \
    > /dev/null 2>&1

echo -e "${GREEN}✅ Todas las librerías Python instaladas correctamente${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 6: Copiar archivos de la aplicación
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[6/8] 📦 Instalando aplicación de escritorio...${NC}"

# Copiar archivos de la aplicación
if [ -f "$SCRIPT_DIR/app/traductor_lse_app.py" ]; then
    cp "$SCRIPT_DIR/app/traductor_lse_app.py" "$INSTALL_DIR/app/"
    echo -e "${GREEN}   ✓ Aplicación copiada${NC}"
fi

# Copiar archivos del entrenador si existen
if [ -f "$SCRIPT_DIR/../TraductorLSE-Standalone/trainer/entrenar_modelo.py" ]; then
    mkdir -p "$INSTALL_DIR/trainer"
    cp "$SCRIPT_DIR/../TraductorLSE-Standalone/trainer/entrenar_modelo.py" "$INSTALL_DIR/trainer/"
    echo -e "${GREEN}   ✓ Módulo de entrenamiento copiado${NC}"
elif [ -f "$SCRIPT_DIR/trainer/entrenar_modelo.py" ]; then
    mkdir -p "$INSTALL_DIR/trainer"
    cp "$SCRIPT_DIR/trainer/entrenar_modelo.py" "$INSTALL_DIR/trainer/"
    echo -e "${GREEN}   ✓ Módulo de entrenamiento copiado${NC}"
fi

# Copiar modelo si existe
if [ -d "$SCRIPT_DIR/../TraductorLSE-Standalone/model" ]; then
    cp -r "$SCRIPT_DIR/../TraductorLSE-Standalone/model/"* "$INSTALL_DIR/model/" 2>/dev/null || true
    echo -e "${GREEN}   ✓ Modelo preentrenado copiado${NC}"
elif [ -d "$SCRIPT_DIR/model" ]; then
    cp -r "$SCRIPT_DIR/model/"* "$INSTALL_DIR/model/" 2>/dev/null || true
    echo -e "${GREEN}   ✓ Modelo preentrenado copiado${NC}"
fi

# Copiar datos si existen
if [ -d "$SCRIPT_DIR/../TraductorLSE-Standalone/data" ]; then
    cp -r "$SCRIPT_DIR/../TraductorLSE-Standalone/data/"* "$INSTALL_DIR/data/" 2>/dev/null || true
fi

echo -e "${GREEN}✅ Aplicación instalada${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 7: Crear aplicación principal (lanzador con menú)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[7/8] 🎨 Creando lanzador de aplicación...${NC}"

cat > "$INSTALL_DIR/traductor_lse.py" << 'EOFAPP'
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🤟 TRADUCTOR LSE - APLICACIÓN DE ESCRITORIO
═══════════════════════════════════════════════════════════════════════════════
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os
from pathlib import Path

INSTALL_DIR = Path(__file__).parent
MODEL_DIR = INSTALL_DIR / "model"
DATA_DIR = INSTALL_DIR / "data"

COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'accent': '#e94560',
    'success': '#00d26a',
    'text': '#ffffff',
    'text_secondary': '#a0a0a0',
}

class TraductorLSEApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traductor LSE")
        self.root.geometry("600x500")
        self.root.configure(bg=COLORS['bg_dark'])
        self._create_ui()
        self._update_status()
    
    def _create_ui(self):
        # Título
        title_frame = tk.Frame(self.root, bg=COLORS['bg_dark'], pady=30)
        title_frame.pack(fill=tk.X)
        
        tk.Label(title_frame, text="🤟", font=('Helvetica', 60),
                bg=COLORS['bg_dark'], fg=COLORS['accent']).pack()
        
        tk.Label(title_frame, text="Traductor LSE", font=('Helvetica', 28, 'bold'),
                bg=COLORS['bg_dark'], fg=COLORS['text']).pack()
        
        tk.Label(title_frame, text="Lengua de Señas Ecuatoriana", font=('Helvetica', 12),
                bg=COLORS['bg_dark'], fg=COLORS['text_secondary']).pack()
        
        # Botones
        btn_frame = tk.Frame(self.root, bg=COLORS['bg_dark'], padx=60)
        btn_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Botón Traductor
        traductor_btn = tk.Button(
            btn_frame,
            text="🎯 INICIAR TRADUCTOR",
            font=('Helvetica', 16, 'bold'),
            bg=COLORS['success'],
            fg='white',
            activebackground='#00b85c',
            activeforeground='white',
            relief=tk.FLAT,
            cursor='hand2',
            pady=20,
            command=self.abrir_traductor
        )
        traductor_btn.pack(fill=tk.X, pady=10)
        
        tk.Label(btn_frame, text="Traduce señas en tiempo real",
                font=('Helvetica', 10), bg=COLORS['bg_dark'],
                fg=COLORS['text_secondary']).pack()
        
        # Botón Entrenador
        entrenar_btn = tk.Button(
            btn_frame,
            text="🎓 ENTRENAR MODELO",
            font=('Helvetica', 16, 'bold'),
            bg=COLORS['accent'],
            fg='white',
            activebackground='#ff6b7a',
            activeforeground='white',
            relief=tk.FLAT,
            cursor='hand2',
            pady=20,
            command=self.abrir_entrenador
        )
        entrenar_btn.pack(fill=tk.X, pady=(30, 10))
        
        tk.Label(btn_frame, text="Captura y entrena nuevas señas",
                font=('Helvetica', 10), bg=COLORS['bg_dark'],
                fg=COLORS['text_secondary']).pack()
        
        # Estado
        status_frame = tk.Frame(self.root, bg=COLORS['bg_medium'], padx=20, pady=15)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Label(status_frame, text="ESTADO DEL MODELO", font=('Helvetica', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text']).pack(anchor=tk.W)
        
        self.status_label = tk.Label(status_frame, text="Verificando...",
                                    font=('Helvetica', 11), bg=COLORS['bg_medium'],
                                    fg=COLORS['text_secondary'], justify=tk.LEFT)
        self.status_label.pack(anchor=tk.W, pady=5)
    
    def _update_status(self):
        model_exists = (MODEL_DIR / "best_model.h5").exists() or (MODEL_DIR / "model.tflite").exists()
        labels_exists = (MODEL_DIR / "labels.pkl").exists()
        
        if model_exists and labels_exists:
            import pickle
            with open(MODEL_DIR / "labels.pkl", 'rb') as f:
                labels = pickle.load(f)
            
            self.status_label.config(
                text=f"✅ Modelo listo: {len(labels)} señas\n   ({', '.join(labels[:5])}{'...' if len(labels) > 5 else ''})",
                fg=COLORS['success']
            )
        else:
            self.status_label.config(
                text="⚠️ No hay modelo entrenado\n   Usa el Entrenador para crear uno",
                fg=COLORS['accent']
            )
    
    def abrir_traductor(self):
        if not (MODEL_DIR / "labels.pkl").exists():
            messagebox.showwarning(
                "Sin Modelo",
                "No hay modelo entrenado.\n\nUsa el Entrenador primero."
            )
            return
        
        self.root.withdraw()
        traductor_path = INSTALL_DIR / "app" / "traductor_lse_app.py"
        subprocess.run([sys.executable, str(traductor_path)])
        self.root.deiconify()
        self._update_status()
    
    def abrir_entrenador(self):
        self.root.withdraw()
        entrenador_path = INSTALL_DIR / "trainer" / "entrenar_modelo.py"
        if not entrenador_path.exists():
            messagebox.showerror("Error", "Módulo de entrenamiento no encontrado")
            self.root.deiconify()
            return
        subprocess.run([sys.executable, str(entrenador_path)])
        self.root.deiconify()
        self._update_status()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    app = TraductorLSEApp()
    app.run()
EOFAPP

chmod +x "$INSTALL_DIR/traductor_lse.py"

# Crear script lanzador
cat > "$INSTALL_DIR/lanzar.sh" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
exec "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/traductor_lse.py"
EOF
chmod +x "$INSTALL_DIR/lanzar.sh"

echo -e "${GREEN}✅ Lanzador creado${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 8: Crear iconos y acceso directo
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[8/8] 🎨 Creando accesos directos...${NC}"

# Crear icono SVG
cat > "$USER_HOME/.local/share/icons/traductor-lse.svg" << 'ICON_SVG'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#e94560;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0f3460;stop-opacity:1" />
    </linearGradient>
  </defs>
  <circle cx="50" cy="50" r="48" fill="url(#grad1)"/>
  <g fill="white" transform="translate(20, 15) scale(0.6)">
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

# Archivo .desktop
cat > "$USER_HOME/.local/share/applications/traductor-lse.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Traductor LSE
GenericName=Sign Language Translator
Comment=Traductor de Lengua de Señas Ecuatoriana
Exec=$INSTALL_DIR/lanzar.sh
Icon=$USER_HOME/.local/share/icons/traductor-lse.svg
Terminal=false
Categories=Utility;Accessibility;Education;
Keywords=sign;language;translator;accessibility;
StartupNotify=true
EOF

# Copiar al escritorio
if [ -d "$USER_HOME/Desktop" ]; then
    cp "$USER_HOME/.local/share/applications/traductor-lse.desktop" "$USER_HOME/Desktop/"
    chmod +x "$USER_HOME/Desktop/traductor-lse.desktop"
fi

# Ajustar permisos
chown -R "$REAL_USER:$REAL_USER" "$INSTALL_DIR"
chown "$REAL_USER:$REAL_USER" "$USER_HOME/.local/share/applications/traductor-lse.desktop"
chown "$REAL_USER:$REAL_USER" "$USER_HOME/.local/share/icons/traductor-lse.svg"
[ -f "$USER_HOME/Desktop/traductor-lse.desktop" ] && chown "$REAL_USER:$REAL_USER" "$USER_HOME/Desktop/traductor-lse.desktop"

# Actualizar cache
update-desktop-database "$USER_HOME/.local/share/applications" 2>/dev/null || true

echo -e "${GREEN}✅ Accesos directos creados${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ✅ ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!                           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${CYAN}📂 Ubicación: $INSTALL_DIR${NC}"
echo ""
echo -e "${YELLOW}🚀 CÓMO USAR:${NC}"
echo ""
echo -e "   ${GREEN}1.${NC} Busca 'Traductor LSE' en el menú de aplicaciones"
echo -e "   ${GREEN}2.${NC} O haz clic en el icono del escritorio"
echo -e "   ${GREEN}3.${NC} O ejecuta desde terminal: $INSTALL_DIR/lanzar.sh"
echo ""
echo -e "${PURPLE}🤟 ¡Todo listo! Conecta una cámara y empieza a usar la aplicación.${NC}"
echo ""
echo -e "${CYAN}📝 Notas:${NC}"
echo -e "   • NO necesitas instalar nada más"
echo -e "   • Todas las dependencias ya están incluidas"
echo -e "   • El modelo se entrena desde la propia aplicación"
echo ""
