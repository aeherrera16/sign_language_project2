"""
Backend API para Traductor LSE
Sistema de reconocimiento de Lengua de Señas Ecuatoriana con:
- Segmentación semántica pixel a pixel
- Integración con IA generativa (DeepSeek/OpenAI)
- MediaPipe para tracking de manos
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from routers import gestures, recognition, training, capture
from routers import gestures_db

app = FastAPI(
    title="Traductor LSE API",
    description="API para reconocimiento de Lengua de Señas Ecuatoriana en tiempo real con IA",
    version="3.0.0"
)

# CORS para permitir requests desde el frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear carpetas necesarias
os.makedirs("data/gestures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("model/checkpoints", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Montar carpeta de uploads
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Incluir routers
app.include_router(gestures.router, prefix="/api/gestures", tags=["Gestures"])
app.include_router(recognition.router, prefix="/api/recognize", tags=["Recognition"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(capture.router, prefix="/api/capture", tags=["🎨 Capture + AI"])
app.include_router(gestures_db.router)

@app.get("/")
async def root():
    return {
        "message": "🤟 Traductor LSE API",
        "version": "2.0.0",
        "features": [
            "Segmentación semántica pixel a pixel",
            "Integración con IA generativa",
            "Reconocimiento en tiempo real",
            "Base de datos expandible"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "segmentation": "active",
            "mediapipe": "active",
            "llm": "active"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
