#!/bin/bash
# ============================================================
#  LANZADOR RÁPIDO - Traductor LSE
#  Uso: ./ejecutar.sh          (menú interactivo)
#       ./ejecutar.sh grabar   (directo al grabador)
#       ./ejecutar.sh entrenar (directo al entrenador)
#       ./ejecutar.sh traducir (directo al traductor)
#       ./ejecutar.sh evaluar  (directo a métricas)
#       ./ejecutar.sh flujo    (grabar → entrenar → traducir)
# ============================================================

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv/bin/activate"
PROTO="$DIR/prototipo"

# Colores
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Activar entorno virtual
if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo -e "${RED}❌ No se encontró el entorno virtual en $VENV${NC}"
    exit 1
fi

# Silenciar warnings de TensorFlow, MediaPipe y absl
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export MEDIAPIPE_DISABLE_GPU=1
export GLOG_minloglevel=3
export ABSL_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

grabar() {
    local nombre="$1"
    local cantidad="${2:-30}"
    
    if [ -z "$nombre" ]; then
        read -p "Nombre de la seña: " nombre
    fi
    
    if [ -z "$nombre" ]; then
        echo -e "${RED}❌ Nombre vacío${NC}"
        return 1
    fi
    
    echo -e "${GREEN}📹 Grabando seña: ${BOLD}${nombre}${NC} ${GREEN}(${cantidad} secuencias)${NC}"
    echo -e "${GREEN}   Presiona Q en la ventana cuando termines${NC}\n"
    python "$PROTO/1_grabar_senas.py" --nombre "$nombre" --cantidad "$cantidad"
}

entrenar() {
    echo -e "${GREEN}🧠 Entrenando modelo...${NC}\n"
    python "$PROTO/2_entrenar_modelo.py"
}

traducir() {
    echo -e "${GREEN}🔊 Iniciando traductor...${NC}\n"
    python "$PROTO/3_traductor.py"
}

evaluar() {
    echo -e "${GREEN}📊 Evaluando métricas ISO...${NC}\n"
    python "$PROTO/4_evaluar_iso25023.py"
}

estado() {
    echo -e "\n${CYAN}📂 Estado del proyecto:${NC}"
    echo -e "${BOLD}Señas grabadas:${NC}"
    for d in "$PROTO/datos"/*/; do
        if [ -d "$d" ]; then
            nombre=$(basename "$d")
            archivos=$(ls "$d"/*.json 2>/dev/null | wc -l | tr -d ' ')
            echo -e "  ${GREEN}✓${NC} $nombre ($archivos archivos)"
        fi
    done
    echo ""
    if [ -f "$PROTO/modelo/modelo.h5" ]; then
        echo -e "${BOLD}Modelo:${NC} ${GREEN}✅ Entrenado${NC}"
        if [ -f "$PROTO/modelo/info.json" ]; then
            python -c "import json; d=json.load(open('$PROTO/modelo/info.json')); acc=d.get('accuracy_test', d.get('accuracy',0)); print(f'  Clases: {d[\"clases\"]}'); print(f'  Accuracy (test): {acc:.1%}'); gap=d.get('gap_overfitting',0); print(f'  Gap overfitting: {gap:.1%}') if gap > 0 else None"
        fi
    else
        echo -e "${BOLD}Modelo:${NC} ${RED}❌ No entrenado${NC}"
    fi
    echo ""
}

flujo_completo() {
    echo -e "\n${YELLOW}🚀 FLUJO COMPLETO: Grabar → Entrenar → Traducir${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    # Pedir nombre UNA sola vez
    read -p "Nombre de la seña a grabar: " nombre_sena
    if [ -z "$nombre_sena" ]; then
        echo -e "${RED}❌ Nombre vacío${NC}"
        return 1
    fi
    
    echo -e "\n${BOLD}━━━ PASO 1/3: GRABAR ━━━${NC}"
    grabar "$nombre_sena" 30
    
    echo -e "\n${BOLD}━━━ PASO 2/3: ENTRENAR ━━━${NC}"
    entrenar
    
    echo -e "\n${BOLD}━━━ PASO 3/3: TRADUCIR ━━━${NC}"
    traducir
    
    echo -e "\n${GREEN}✅ Flujo completo terminado${NC}\n"
}

grabar_varias() {
    echo -e "\n${YELLOW}📹 GRABAR MÚLTIPLES SEÑAS${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    echo -e "Escribe los nombres de las señas separados por espacio."
    echo -e "Ejemplo: PRESIDENTE ECUADOR DECIR HOY\n"
    
    read -p "Señas a grabar: " -a senas
    
    if [ ${#senas[@]} -eq 0 ]; then
        echo -e "${RED}❌ No se ingresaron señas${NC}"
        return 1
    fi
    
    total=${#senas[@]}
    actual=0
    
    for sena in "${senas[@]}"; do
        actual=$((actual + 1))
        echo -e "\n${CYAN}━━━ Seña $actual/$total: ${BOLD}$sena${NC} ${CYAN}━━━${NC}"
        grabar "$sena" 30
    done
    
    echo -e "\n${GREEN}✅ Todas las señas grabadas ($total)${NC}"
    echo -e "${BOLD}¿Entrenar modelo con los nuevos datos?${NC}"
    read -p "[S/n]: " respuesta
    if [ "$respuesta" != "n" ] && [ "$respuesta" != "N" ]; then
        entrenar
    fi
}

menu() {
    while true; do
        clear
        echo -e "${CYAN}${BOLD}"
        echo "  ╔══════════════════════════════════════════╗"
        echo "  ║       🤟 TRADUCTOR LSE - PROTOTIPO       ║"
        echo "  ╠══════════════════════════════════════════╣"
        echo "  ║                                          ║"
        echo "  ║   1. 📹  Grabar UNA seña                 ║"
        echo "  ║   2. 📹  Grabar VARIAS señas             ║"
        echo "  ║   3. 🧠  Entrenar modelo                 ║"
        echo "  ║   4. 🔊  Iniciar traductor               ║"
        echo "  ║   5. 📊  Evaluar métricas ISO            ║"
        echo "  ║   6. 📂  Ver estado del proyecto         ║"
        echo "  ║   7. 🚀  Flujo completo (1→3→4)         ║"
        echo "  ║   0. 🚪  Salir                           ║"
        echo "  ║                                          ║"
        echo "  ╚══════════════════════════════════════════╝"
        echo -e "${NC}"
        
        read -p "  Opción: " opcion
        
        case $opcion in
            1) grabar ;;
            2) grabar_varias ;;
            3) entrenar ;;
            4) traducir ;;
            5) evaluar ;;
            6) estado ;;
            7) flujo_completo ;;
            0) echo -e "\n${GREEN}👋 ¡Hasta luego!${NC}\n"; exit 0 ;;
            *) echo -e "\n${RED}❌ Opción no válida${NC}"; sleep 1 ;;
        esac
        
        echo ""
        read -p "Presiona ENTER para volver al menú..."
    done
}

# === EJECUCIÓN ===
case "${1}" in
    grabar|1)    grabar "$2" "$3" ;;
    varias)      grabar_varias ;;
    entrenar|2)  entrenar ;;
    traducir|3)  traducir ;;
    evaluar|4)   evaluar ;;
    estado|5)    estado ;;
    flujo|6)     flujo_completo ;;
    terminal)    menu ;;
    *)           echo -e "${GREEN}Abriendo menú gráfico...${NC}"; python "$PROTO/menu.py" ;;
esac
