Docker setup

This repository contains Dockerfiles and a docker-compose.yml to run the stack locally.

Services included:
- localai (local LLM alternative)
- n8n (optional)
- backend (FastAPI + uvicorn)
- frontend (Vite dev server)

Quick start (build and run):

# build and start in foreground
docker compose up --build

# start in background
docker compose up -d --build

# stop
docker compose down

Notes:
- The backend exposes port 8000. The frontend exposes 5173.
- Sqlite DB files and uploads are mounted as volumes to preserve data across restarts.
- If you prefer to run services locally (no Docker), you can still use `scripts/start_all.sh`.
