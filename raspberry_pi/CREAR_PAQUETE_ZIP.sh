#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 📦 EMPAQUETADOR - TRADUCTOR LSE PARA RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════════
#
# Crea un archivo ZIP con TODO lo necesario para instalar en Raspberry Pi:
#   ✅ Instalador automático
#   ✅ Código de la aplicación
#   ✅ Modelo entrenado (si existe)
#   ✅ Documentación
#
# Uso: ./empaquetar_raspberry.sh
#
# Genera: TraductorLSE-RaspberryPi-COMPLETO.zip
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEMP_DIR="/tmp/TraductorLSE-Package"
OUTPUT_FILE="$PROJECT_ROOT/TraductorLSE-RaspberryPi-COMPLETO.zip"

echo -e "${CYAN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   📦 EMPAQUETADOR - TRADUCTOR LSE                                     ║
║      Para Raspberry Pi 4                                              ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# Limpiar y crear directorio temporal
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[1/5] 🧹 Preparando espacio de trabajo...${NC}"

rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR/raspberry_pi"
mkdir -p "$TEMP_DIR/raspberry_pi/app"
mkdir -p "$TEMP_DIR/raspberry_pi/trainer"
mkdir -p "$TEMP_DIR/raspberry_pi/model"
mkdir -p "$TEMP_DIR/raspberry_pi/data"

echo -e "${GREEN}✅ Directorio temporal creado${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# Copiar instalador y documentación
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[2/5] 📄 Copiando instalador y documentación...${NC}"

# Instalador automático
cp "$SCRIPT_DIR/instalar_completo.sh" "$TEMP_DIR/raspberry_pi/"
chmod +x "$TEMP_DIR/raspberry_pi/instalar_completo.sh"

# Documentación
cp "$SCRIPT_DIR/LEEME_INSTALACION.md" "$TEMP_DIR/raspberry_pi/README.md"

# Crear archivo de instrucciones rápidas
cat > "$TEMP_DIR/raspberry_pi/INICIO_RAPIDO.txt" << 'EOF'
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   🤟 TRADUCTOR LSE - INICIO RÁPIDO                                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

📋 REQUISITOS:
   • Raspberry Pi 4 (4GB+ RAM)
   • Raspberry Pi OS (Bullseye/Bookworm)
   • Conexión a internet
   • Cámara USB o Raspberry Pi Camera

🚀 INSTALACIÓN (2 PASOS):

1. Copia esta carpeta "raspberry_pi" a tu Raspberry Pi

2. Abre terminal y ejecuta:
   
   cd raspberry_pi
   chmod +x instalar_completo.sh
   sudo ./instalar_completo.sh

⏱️ Tiempo: 10-15 minutos (todo automático)

✅ QUÉ SE INSTALA AUTOMÁTICAMENTE:
   • Python 3.11
   • TensorFlow Lite
   • MediaPipe
   • OpenCV
   • NumPy, Scikit-learn
   • Aplicación de escritorio
   • Iconos y accesos directos

🎯 DESPUÉS DE INSTALAR:

   Busca "Traductor LSE" en tu menú de aplicaciones
   o ejecuta: ~/TraductorLSE/lanzar.sh

🎓 PRIMERA VEZ:

   1. Abre la aplicación
   2. Selecciona "ENTRENAR MODELO"
   3. Captura algunas señas (30-40 muestras cada una)
   4. Presiona "Entrenar"
   5. ¡Listo para traducir!

📖 Documentación completa: README.md

EOF

echo -e "${GREEN}✅ Documentos copiados${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# Copiar código de la aplicación
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[3/5] 💻 Copiando aplicación de escritorio...${NC}"

# Aplicación principal del traductor
if [ -f "$SCRIPT_DIR/app/traductor_lse_app.py" ]; then
    cp "$SCRIPT_DIR/app/traductor_lse_app.py" "$TEMP_DIR/raspberry_pi/app/"
    echo -e "${GREEN}   ✓ Traductor copiado${NC}"
else
    echo -e "${YELLOW}   ⚠️  traductor_lse_app.py no encontrado${NC}"
fi

# Módulo de entrenamiento
TRAINER_SOURCE=""
if [ -f "$PROJECT_ROOT/TraductorLSE-Standalone/trainer/entrenar_modelo.py" ]; then
    TRAINER_SOURCE="$PROJECT_ROOT/TraductorLSE-Standalone/trainer/entrenar_modelo.py"
elif [ -f "$SCRIPT_DIR/trainer/entrenar_modelo.py" ]; then
    TRAINER_SOURCE="$SCRIPT_DIR/trainer/entrenar_modelo.py"
fi

if [ -n "$TRAINER_SOURCE" ]; then
    cp "$TRAINER_SOURCE" "$TEMP_DIR/raspberry_pi/trainer/"
    echo -e "${GREEN}   ✓ Módulo de entrenamiento copiado${NC}"
else
    echo -e "${YELLOW}   ⚠️  entrenar_modelo.py no encontrado${NC}"
fi

echo -e "${GREEN}✅ Aplicación copiada${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# Copiar modelo y datos (si existen)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[4/5] 🧠 Copiando modelo entrenado (si existe)...${NC}"

# Buscar modelo en varias ubicaciones
MODEL_COPIED=false

if [ -d "$PROJECT_ROOT/TraductorLSE-Standalone/model" ]; then
    if [ -f "$PROJECT_ROOT/TraductorLSE-Standalone/model/labels.pkl" ]; then
        cp -r "$PROJECT_ROOT/TraductorLSE-Standalone/model/"* "$TEMP_DIR/raspberry_pi/model/" 2>/dev/null || true
        MODEL_COPIED=true
        echo -e "${GREEN}   ✓ Modelo copiado desde TraductorLSE-Standalone${NC}"
    fi
fi

if [ "$MODEL_COPIED" = false ] && [ -d "$SCRIPT_DIR/model" ]; then
    if [ -f "$SCRIPT_DIR/model/labels.pkl" ]; then
        cp -r "$SCRIPT_DIR/model/"* "$TEMP_DIR/raspberry_pi/model/" 2>/dev/null || true
        MODEL_COPIED=true
        echo -e "${GREEN}   ✓ Modelo copiado desde raspberry_pi${NC}"
    fi
fi

if [ "$MODEL_COPIED" = false ]; then
    echo -e "${YELLOW}   ℹ️  No se encontró modelo preentrenado (se puede entrenar después)${NC}"
fi

# Copiar datos de entrenamiento si existen
if [ -d "$PROJECT_ROOT/TraductorLSE-Standalone/data" ]; then
    if [ -f "$PROJECT_ROOT/TraductorLSE-Standalone/data/training_data.pkl" ]; then
        cp -r "$PROJECT_ROOT/TraductorLSE-Standalone/data/"* "$TEMP_DIR/raspberry_pi/data/" 2>/dev/null || true
        echo -e "${GREEN}   ✓ Datos de entrenamiento copiados${NC}"
    fi
fi

echo -e "${GREEN}✅ Datos copiados${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# Crear archivo ZIP
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[5/5] 🗜️  Creando archivo ZIP...${NC}"

# Eliminar ZIP anterior si existe
[ -f "$OUTPUT_FILE" ] && rm "$OUTPUT_FILE"

# Crear ZIP
cd "$TEMP_DIR"
zip -r "$OUTPUT_FILE" raspberry_pi/ -q

# Limpiar
rm -rf "$TEMP_DIR"

# Calcular tamaño
SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo -e "${GREEN}✅ Archivo creado${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ✅ ¡PAQUETE CREADO EXITOSAMENTE!                                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${CYAN}📦 Archivo: ${NC}$(basename "$OUTPUT_FILE")"
echo -e "${CYAN}📂 Ubicación: ${NC}$OUTPUT_FILE"
echo -e "${CYAN}💾 Tamaño: ${NC}$SIZE"
echo ""
echo -e "${YELLOW}🚀 SIGUIENTES PASOS:${NC}"
echo ""
echo -e "   ${GREEN}1.${NC} Transfiere el archivo ZIP a tu Raspberry Pi:"
echo -e "      ${CYAN}scp \"$OUTPUT_FILE\" pi@raspberrypi.local:~/${NC}"
echo ""
echo -e "   ${GREEN}2.${NC} En el Raspberry Pi, descomprime:"
echo -e "      ${CYAN}unzip $(basename "$OUTPUT_FILE")${NC}"
echo ""
echo -e "   ${GREEN}3.${NC} Ejecuta el instalador:"
echo -e "      ${CYAN}cd raspberry_pi${NC}"
echo -e "      ${CYAN}sudo ./instalar_completo.sh${NC}"
echo ""
echo -e "${PURPLE}🤟 ¡Todo listo para instalar en tu Raspberry Pi!${NC}"
echo ""
