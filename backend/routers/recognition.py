"""
Router para reconocimiento en tiempo real
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
from typing import Optional

from services.segmentation_service import segmentation_service
from services.llm_service import llm_service

router = APIRouter()

# Cargar modelo al iniciar
model = None
labels = None

def load_model():
    """Carga el modelo entrenado"""
    global model, labels
    
    model_path = "model/best_model.h5"
    labels_path = "model/labels.pkl"
    
    if os.path.exists(model_path) and os.path.exists(labels_path):
        model = tf.keras.models.load_model(model_path)
        with open(labels_path, "rb") as f:
            labels = pickle.load(f)
        return True
    return False

@router.on_event("startup")
async def startup_event():
    """Cargar modelo al inicio"""
    if load_model():
        print("✓ Modelo cargado exitosamente")
    else:
        print("⚠ No se encontró modelo entrenado")

@router.post("/predict")
async def predict_gesture(
    image: UploadFile = File(...),
    use_llm: bool = True
):
    """
    Reconoce una seña desde una imagen
    Usa segmentación semántica + landmarks + LLM para contextualización
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo no disponible. Entrena el modelo primero."
            )
        
        # Leer imagen
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")
        
        # Procesar con segmentación
        result = segmentation_service.process_frame(img)
        
        if not result["has_hands"]:
            return {
                "success": False,
                "message": "No se detectaron manos",
                "gesture": None,
                "confidence": 0.0
            }
        
        # Preparar landmarks para el modelo
        landmarks_data = np.array(result["landmarks"]["landmarks"])
        
        # Si hay 2 manos, concatenar; si hay 1, rellenar con ceros
        if len(landmarks_data) == 2:
            input_vector = np.concatenate(landmarks_data)
        else:
            input_vector = np.concatenate([landmarks_data[0], np.zeros(63)])
        
        # Asegurar la forma correcta (126 para 2 manos)
        if input_vector.shape[0] != 126:
            input_vector = np.pad(input_vector, (0, 126 - input_vector.shape[0]))
        
        # Predicción
        input_data = np.expand_dims(input_vector, axis=0)
        prediction = model.predict(input_data, verbose=0)[0]
        
        # Top 5 predicciones
        top_indices = np.argsort(prediction)[-5:][::-1]
        top_predictions = [
            {
                "gesture": labels[idx],
                "confidence": float(prediction[idx])
            }
            for idx in top_indices
        ]
        
        # Mejor predicción
        best_idx = top_indices[0]
        gesture = labels[best_idx]
        confidence = float(prediction[best_idx])
        
        response = {
            "success": True,
            "gesture": gesture,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "num_hands": result["landmarks"]["num_hands"]
        }
        
        # Usar LLM para mejorar contexto (opcional)
        if use_llm and confidence < 0.9:
            llm_analysis = await llm_service.analyze_gesture_context(
                gesture,
                confidence,
                []
            )
            response["llm_context"] = llm_analysis
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-video")
async def predict_from_video(
    video: UploadFile = File(...)
):
    """
    Reconoce señas desde un video
    Procesa frame por frame
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo no disponible"
            )
        
        # Guardar video temporalmente
        temp_path = f"uploads/temp_{video.filename}"
        with open(temp_path, "wb") as f:
            f.write(await video.read())
        
        cap = cv2.VideoCapture(temp_path)
        results = []
        
        frame_count = 0
        while cap.isOpened() and frame_count < 100:  # Limitar a 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar cada 5 frames
            if frame_count % 5 == 0:
                result = segmentation_service.process_frame(frame)
                
                if result["has_hands"]:
                    landmarks_data = np.array(result["landmarks"]["landmarks"])
                    
                    if len(landmarks_data) == 2:
                        input_vector = np.concatenate(landmarks_data)
                    else:
                        input_vector = np.concatenate([landmarks_data[0], np.zeros(63)])
                    
                    if input_vector.shape[0] == 126:
                        input_data = np.expand_dims(input_vector, axis=0)
                        prediction = model.predict(input_data, verbose=0)[0]
                        
                        best_idx = np.argmax(prediction)
                        results.append({
                            "frame": frame_count,
                            "gesture": labels[best_idx],
                            "confidence": float(prediction[best_idx])
                        })
            
            frame_count += 1
        
        cap.release()
        os.remove(temp_path)
        
        return {
            "success": True,
            "total_frames": frame_count,
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info")
async def get_model_info():
    """Obtiene información sobre el modelo actual"""
    if model is None:
        return {
            "loaded": False,
            "message": "No hay modelo entrenado"
        }
    
    return {
        "loaded": True,
        "num_gestures": len(labels) if labels else 0,
        "gestures": labels if labels else [],
        "model_summary": str(model.summary()) if model else None
    }
