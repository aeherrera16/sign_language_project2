# Pydantic models para validación de datos
from pydantic import BaseModel
from typing import List, Optional

class GestureCreate(BaseModel):
    """Modelo para crear una nueva seña"""
    name: str
    samples: int = 100

class GestureResponse(BaseModel):
    """Respuesta con información de una seña"""
    name: str
    samples: int
    path: str

class PredictionResult(BaseModel):
    """Resultado de una predicción"""
    gesture: str
    confidence: float

class RecognitionResponse(BaseModel):
    """Respuesta completa de reconocimiento"""
    success: bool
    gesture: Optional[str]
    confidence: float
    top_predictions: List[PredictionResult]
    num_hands: int
    llm_context: Optional[dict] = None

class TrainingStatus(BaseModel):
    """Estado del entrenamiento"""
    is_training: bool
    progress: int
    message: str
    history: Optional[dict] = None
