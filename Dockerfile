# ════════════════════════════════════════════════════════════════════════════════
# Dockerfile Multi-Stage para Traductor LSE
# Optimizado para Raspberry Pi 4 (ARM64)
# ════════════════════════════════════════════════════════════════════════════════
# 
# Este Dockerfile crea una imagen completa con:
#   - Backend (Python/FastAPI)
#   - Frontend (React pre-compilado)
#   - Modelo de IA incluido
#   - Nginx como servidor web
#
# Para construir para ARM64 (Raspberry Pi):
#   docker buildx build --platform linux/arm64 -t usuario/traductor-lse:latest --push .
#
# ════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STAGE 1: Construir Frontend
# ─────────────────────────────────────────────────────────────────────────────────
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copiar archivos de dependencias
COPY frontend/react_app/package*.json ./

# Instalar dependencias
RUN npm ci --silent

# Copiar código fuente
COPY frontend/react_app/ ./

# Construir para producción
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────────
# STAGE 2: Imagen Final (Backend + Frontend + Nginx)
# ─────────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

LABEL maintainer="Traductor LSE Team"
LABEL description="Sistema de Traducción de Lengua de Señas Ecuatoriana"
LABEL version="2.0.0"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    espeak-ng \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear directorios de la aplicación
WORKDIR /app
RUN mkdir -p /app/backend /app/frontend /app/data/gestures /app/model /app/uploads /app/logs

# ─────────────────────────────────────────────────────────────────────────────────
# Instalar dependencias Python
# ─────────────────────────────────────────────────────────────────────────────────
COPY backend/requirements.txt /app/backend/

# Dependencias optimizadas para producción
RUN pip install --no-cache-dir \
    fastapi==0.95.2 \
    uvicorn[standard]==0.20.0 \
    numpy==1.26.4 \
    opencv-python-headless==4.8.1.78 \
    mediapipe==0.10.9 \
    pillow==10.0.1 \
    httpx==0.24.1 \
    python-dotenv==1.0.0 \
    aiofiles==23.2.1 \
    python-multipart==0.0.6 \
    scikit-learn==1.3.2

# ─────────────────────────────────────────────────────────────────────────────────
# Copiar código del backend
# ─────────────────────────────────────────────────────────────────────────────────
COPY backend/ /app/backend/

# Copiar modelo entrenado (si existe)
COPY backend/model/ /app/model/

# Copiar datos de gestos (si existen)
COPY backend/data/ /app/data/

# ─────────────────────────────────────────────────────────────────────────────────
# Copiar frontend compilado desde stage anterior
# ─────────────────────────────────────────────────────────────────────────────────
COPY --from=frontend-builder /app/frontend/dist /app/frontend

# ─────────────────────────────────────────────────────────────────────────────────
# Configurar Nginx
# ─────────────────────────────────────────────────────────────────────────────────
RUN rm -f /etc/nginx/sites-enabled/default

COPY <<EOF /etc/nginx/sites-available/traductor-lse
server {
    listen 80;
    server_name _;
    
    # Servir frontend estático
    location / {
        root /app/frontend;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }
    
    # Proxy API al backend
    location /api/ {
        rewrite ^/api/(.*) /\$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        client_max_body_size 50M;
    }
    
    # Proxy directo para endpoints sin /api
    location ~ ^/(capture|recognize|gestures|training|health|system)/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        client_max_body_size 50M;
    }
}
EOF

RUN ln -s /etc/nginx/sites-available/traductor-lse /etc/nginx/sites-enabled/

# ─────────────────────────────────────────────────────────────────────────────────
# Configurar Supervisor (para ejecutar Nginx + Backend juntos)
# ─────────────────────────────────────────────────────────────────────────────────
COPY <<EOF /etc/supervisor/conf.d/traductor-lse.conf
[supervisord]
nodaemon=true
user=root

[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/app/logs/nginx.log
stderr_logfile=/app/logs/nginx_error.log

[program:backend]
command=/usr/local/bin/uvicorn main:app --host 127.0.0.1 --port 8000
directory=/app/backend
autostart=true
autorestart=true
stdout_logfile=/app/logs/backend.log
stderr_logfile=/app/logs/backend_error.log
environment=PYTHONPATH="/app/backend"
EOF

# ─────────────────────────────────────────────────────────────────────────────────
# Script de inicio
# ─────────────────────────────────────────────────────────────────────────────────
COPY <<EOF /app/start.sh
#!/bin/bash
echo "🤟 Iniciando Traductor LSE..."
echo "   Versión: 2.0.0"
echo "   Puerto: 80 (HTTP)"
echo ""

# Mostrar IP para acceso remoto
IP=\$(hostname -I | awk '{print \$1}')
echo "🌐 Accede desde:"
echo "   - Local: http://localhost"
echo "   - Red:   http://\$IP"
echo ""

# Iniciar supervisor (que a su vez inicia nginx y backend)
exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
EOF

RUN chmod +x /app/start.sh

# ─────────────────────────────────────────────────────────────────────────────────
# Puertos y CMD
# ─────────────────────────────────────────────────────────────────────────────────
EXPOSE 80

# Volúmenes para datos persistentes
VOLUME ["/app/data", "/app/model", "/app/uploads"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

CMD ["/app/start.sh"]
