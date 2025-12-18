Arrancar el proyecto - Frontend + Backend
=========================================

Este README explica los pasos mínimos para instalar dependencias y arrancar el backend (FastAPI + uvicorn) y el frontend (Vite + React) en tu máquina macOS (zsh).

Requisitos previos
- Python 3.10+
- Node.js 18+ y npm
- (Opcional) Ollama instalado si quieres análisis LLM local

1) Clonar y situarse en el repo
```bash
cd /path/to/sign_language_project2
```

2) Crear y activar virtualenv (Python)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Instalar dependencias frontend
```bash
cd frontend/react_app
npm install
cd ../..  # volver al root del repo
```

4) Iniciar backend y frontend (opción A - manual)
- Backend (en un terminal):
```bash
source .venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- Frontend (en otro terminal):
```bash
cd frontend/react_app
npm run dev
```
La UI estará disponible en `http://localhost:5173/`.

5) Iniciar ambos con script (opción B - automático)
El repo incluye `scripts/start_all.sh` que arranca backend y frontend en background y guarda logs/pids en `logs/`.

- Start:
```bash
./scripts/start_all.sh start
```
- Stop:
```bash
./scripts/start_all.sh stop
```
- Status:
```bash
./scripts/start_all.sh status
```

Logs
- Backend: `logs/backend_out.log`
- Frontend: `logs/frontend_out.log`
- PIDs: `logs/pids.json`

Verificaciones rápidas
- Health del backend:
```bash
curl http://localhost:8000/health
```
- Frontend:
Abrir `http://localhost:5173/` en el navegador.

Endpoints útiles
- POST /api/gestures -> crear una seña (body JSON: {name, description})
- POST /api/capture/save -> subir una captura (multipart/form-data) fields: image (file), gesture_id (string), metadata (string)

Notas
- El almacenamiento por defecto es SQLite (`data/gestures/db.sqlite`) y archivos en `uploads/gestures`.
- En producción, reemplazar SQLite por PostgreSQL y usar un almacenamiento de objetos para los archivos.

Siguientes pasos
- Integrar el botón "Guardar captura" en la UI para enviar imágenes etiquetadas al backend.
- Añadir una interfaz de revisión de capturas para filtrar y aceptar muestras para entrenamiento.

Si quieres, puedo ahora integrar el botón de guardado en `CapturaIA.jsx` y hacer una prueba end-to-end.
