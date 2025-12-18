#!/usr/bin/env bash
# start_all.sh — arranca backend (uvicorn) y frontend (vite) en background
# Uso: ./scripts/start_all.sh [start|stop|status]

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT_DIR/.venv"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend/react_app"
LOG_DIR="$ROOT_DIR/logs"
PIDS_FILE="$LOG_DIR/pids.json"

mkdir -p "$LOG_DIR"

function start() {
  echo "Iniciando servicios..."

  # activar venv si existe
  if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
  else
    echo "WARNING: virtualenv no encontrado en $VENV — intenta crear uno con: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  fi

  # Backend
  echo "Arrancando backend (uvicorn) en puerto 8000..."
  nohup bash -c "cd \"$BACKEND_DIR\" && source \"$ROOT_DIR/.venv/bin/activate\" && uvicorn main:app --host 0.0.0.0 --port 8000 --reload" > "$LOG_DIR/backend_out.log" 2>&1 &
  backend_pid=$!
  echo "backend_pid=$backend_pid"

  # Frontend
  echo "Arrancando frontend (vite) en puerto 5173..."
  nohup bash -c "cd \"$FRONTEND_DIR\" && npm run dev" > "$LOG_DIR/frontend_out.log" 2>&1 &
  frontend_pid=$!
  echo "frontend_pid=$frontend_pid"

  # Guardar pids
  cat > "$PIDS_FILE" <<EOF
{
  "backend_pid": $backend_pid,
  "frontend_pid": $frontend_pid
}
EOF

  echo "Servicios iniciados. Logs: $LOG_DIR"
}

function stop() {
  if [ -f "$PIDS_FILE" ]; then
    backend_pid=$(jq -r '.backend_pid' "$PIDS_FILE" 2>/dev/null || echo "")
    frontend_pid=$(jq -r '.frontend_pid' "$PIDS_FILE" 2>/dev/null || echo "")

    if [ -n "$backend_pid" ] && kill -0 "$backend_pid" 2>/dev/null; then
      echo "Deteniendo backend pid=$backend_pid"
      kill "$backend_pid" || true
    fi
    if [ -n "$frontend_pid" ] && kill -0 "$frontend_pid" 2>/dev/null; then
      echo "Deteniendo frontend pid=$frontend_pid"
      kill "$frontend_pid" || true
    fi

    rm -f "$PIDS_FILE"
    echo "Servicios detenidos."
  else
    echo "No hay servicios registrados en $PIDS_FILE"
  fi
}

function status() {
  if [ -f "$PIDS_FILE" ]; then
    echo "Contenido de $PIDS_FILE:"
    cat "$PIDS_FILE"
  else
    echo "No hay servicios iniciados (no existe $PIDS_FILE)"
  fi
  echo "Backend log: $LOG_DIR/backend_out.log"
  echo "Frontend log: $LOG_DIR/frontend_out.log"
}

case "${1-}" in
  ''|start)
    start
    ;;
  stop)
    stop
    ;;
  status)
    status
    ;;
  *)
    echo "Uso: $0 [start|stop|status]"
    exit 1
    ;;
esac
