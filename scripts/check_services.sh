#!/usr/bin/env bash
# Comprueba status rápido de backend y frontend (docker-compose must be installed)
set -e

echo "Comprobando contenedores docker (si está usando docker-compose)..."
if command -v docker >/dev/null 2>&1; then
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | sed -n '1,200p'
else
  echo "Docker no instalado o no en PATH"
fi

echo
echo "Comprobando endpoints locales (si está ejecutando localmente, fuera de docker):"
curl -sS http://127.0.0.1:8000/health || true
echo
curl -sS http://127.0.0.1:5173/ | sed -n '1,6p' || true

exit 0
