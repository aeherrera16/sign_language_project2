#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# 🚀 PUBLICAR IMAGEN EN DOCKER HUB
# ════════════════════════════════════════════════════════════════════════════════
#
# Este script construye y publica la imagen del Traductor LSE en Docker Hub
# para que pueda ser descargada fácilmente en cualquier Raspberry Pi.
#
# REQUISITOS:
#   1. Cuenta en Docker Hub (https://hub.docker.com)
#   2. Docker Desktop instalado con buildx habilitado
#   3. Haber iniciado sesión: docker login
#
# USO:
#   ./publish_to_dockerhub.sh <tu-usuario-dockerhub>
#
# EJEMPLO:
#   ./publish_to_dockerhub.sh miusuario
#   -> Publicará: miusuario/traductor-lse:latest
#
# ════════════════════════════════════════════════════════════════════════════════

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Verificar argumento
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Debes proporcionar tu usuario de Docker Hub${NC}"
    echo ""
    echo "Uso: $0 <usuario-dockerhub>"
    echo "Ejemplo: $0 miusuario"
    exit 1
fi

DOCKERHUB_USER="$1"
IMAGE_NAME="traductor-lse"
FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}"
VERSION="2.0.0"

echo -e "${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   🐳 PUBLICAR EN DOCKER HUB                                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${CYAN}📦 Imagen: ${FULL_IMAGE}${NC}"
echo -e "${CYAN}🏷️  Versión: ${VERSION}${NC}"
echo ""

# Verificar login en Docker Hub
echo -e "${BLUE}🔐 Verificando sesión en Docker Hub...${NC}"
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo -e "${YELLOW}⚠️  No has iniciado sesión en Docker Hub${NC}"
    echo "Ejecuta: docker login"
    docker login
fi

# Crear builder multiplataforma si no existe
echo -e "\n${BLUE}🔧 Configurando builder multiplataforma...${NC}"
if ! docker buildx ls | grep -q "multiarch"; then
    docker buildx create --name multiarch --use
fi
docker buildx use multiarch

# Construir y publicar para múltiples arquitecturas
echo -e "\n${BLUE}🏗️  Construyendo imagen para ARM64 y AMD64...${NC}"
echo -e "${YELLOW}   (Esto puede tardar 10-20 minutos la primera vez)${NC}"
echo ""

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag "${FULL_IMAGE}:latest" \
    --tag "${FULL_IMAGE}:${VERSION}" \
    --push \
    .

# Verificar publicación
echo -e "\n${GREEN}✅ ¡Imagen publicada exitosamente!${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📦 Tu imagen está disponible en:${NC}"
echo -e "   ${GREEN}https://hub.docker.com/r/${FULL_IMAGE}${NC}"
echo ""
echo -e "${CYAN}🚀 Para instalar en Raspberry Pi, ejecuta:${NC}"
echo -e "   ${YELLOW}docker run -d --name traductor-lse \\${NC}"
echo -e "   ${YELLOW}  -p 80:80 \\${NC}"
echo -e "   ${YELLOW}  --device /dev/video0:/dev/video0 \\${NC}"
echo -e "   ${YELLOW}  -v traductor-data:/app/data \\${NC}"
echo -e "   ${YELLOW}  ${FULL_IMAGE}:latest${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
