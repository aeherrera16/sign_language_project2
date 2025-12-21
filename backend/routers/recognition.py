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
from typing import Optional, List

from services.hand_segmentation_service import hand_segmentation_service
from services.llm_service import llm_service

router = APIRouter()

class TranslationRequest(tf.keras.utils.Sequence): # Actually, let's use a simple pydantic model if possible, but recognition.py doesn't use pydantic yet. Wait, I'll just use a POST with a Dict.
    pass

from pydantic import BaseModel

class SequenceRequest(BaseModel):
    gestures: List[str]

@router.post("/translate-sequence")
async def translate_sequence(request: SequenceRequest):
    """
    Traduce una secuencia de señas a español natural usando el LLM.
    Incluye un fallback simple si el LLM no está disponible.
    """
    try:
        if not request.gestures:
            return {"success": True, "translation": ""}
        
        # Intentar traducción inteligente
        try:
            translation = await llm_service.translate_to_text(request.gestures)
            
            # Si el servicio LLM devolvió un error de conexión (string que empieza con ❌ o Error)
            if translation.startswith("❌") or translation.startswith("Error"):
                raise Exception(translation)
                
            return {
                "success": True, 
                "original": request.gestures,
                "translation": translation,
                "method": "llm"
            }
        except Exception as llm_err:
            print(f"⚠ Fallback de traducción activa: {llm_err}")
            # Fallback: Unir con espacios y capitalizar
            fallback_text = " ".join(request.gestures).capitalize() + "."
            return {
                "success": True,
                "original": request.gestures,
                "translation": fallback_text,
                "method": "fallback",
                "warning": str(llm_err)
            }
            
    except Exception as e:
        print(f"❌ Error crítico en translate_sequence: {e}")
        return {"success": False, "translation": "Error interno del servidor"}

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

@router.post("/reload")
async def reload_model_endpoint():
    """Recarga el modelo en memoria (útil tras entrenar)"""
    if load_model():
        return {"success": True, "message": "Modelo recargado exitosamente"}
    return {"success": False, "message": "No se encontró el archivo del modelo"}


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
        
        print("📸 Recibida petición de predicción...")
        
        # Leer imagen
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")
        
        # Procesar con detección de manos
        hands_info = hand_segmentation_service.detect_hands(img)
        
        if not hands_info or hands_info['num_hands'] == 0:
            return {
                "success": False,
                "message": "No se detectaron manos",
                "gesture": None,
                "confidence": 0.0
            }
        
        # Preparar landmarks para el modelo
        hands_data = []
        for hand in hands_info['hands']:
            hands_data.append({
                "index": hand.get('index', 0),
                "score": float(hand['confidence']),
                "label": hand.get('handedness', 'unknown'),
                "landmarks": hand['landmarks'].tolist()
            })
            
        # Extraer landmarks y aplanar: [21, 3] -> [63]
        landmarks_list = [h['landmarks'].flatten() for h in hands_info['hands']]
        
        # Si hay 2 manos, concatenar; si hay 1, rellenar con ceros
        if len(landmarks_list) >= 2:
            input_vector = np.concatenate(landmarks_list[:2])
        else:
            input_vector = np.concatenate([landmarks_list[0], np.zeros(63)])
        
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
            "num_hands": hands_info['num_hands'],
            "hands": hands_data # Incluimos los landmarks procesados
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
                hands_info = hand_segmentation_service.detect_hands(frame)
                
                if hands_info and hands_info['num_hands'] > 0:
                    landmarks_list = [h['landmarks'].flatten() for h in hands_info['hands']]
                    
                    if len(landmarks_list) >= 2:
                        input_vector = np.concatenate(landmarks_list[:2])
                    else:
                        input_vector = np.concatenate([landmarks_list[0], np.zeros(63)])
                    
                    if input_vector.shape[0] == 126:
                        input_data = np.expand_dims(input_vector, axis=0)
                        
                        # Solo predecir si tenemos vector completo
                        if input_data.shape[1] == 126:
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
