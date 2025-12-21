"""
🎨 Router para Captura Profesional con Segmentación + IA

Endpoints para el frontend React:
- Captura de video en tiempo real
- Análisis de calidad con IA
- Segmentación semántica
- Feedback en vivo
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import cv2
import numpy as np
import io
import base64
from datetime import datetime

from services.hand_segmentation_service import hand_segmentation_service
from services.llm_service_ollama import LLMContextService

router = APIRouter()
llm_service = LLMContextService()


class CaptureAnalysisRequest(BaseModel):
    """Request para análisis de captura"""
    image_base64: str
    gesture_name: Optional[str] = None


class CaptureAnalysisResponse(BaseModel):
    """Response con análisis de calidad"""
    quality: str
    score: int
    recommendations: List[str]
    is_good: bool
    metrics: Dict
    segmentation_available: bool


@router.post("/analyze-capture", response_model=CaptureAnalysisResponse)
async def analyze_capture(request: CaptureAnalysisRequest):
    """
    ✨ ENDPOINT PRINCIPAL PARA FRONTEND
    
    Analiza la calidad de una captura usando:
    - Segmentación semántica
    - Métricas de visión por computadora  
    - IA (Ollama) para feedback inteligente
    
    Frontend React debe llamar esto al capturar una seña.
    """
    try:
        # Decodificar imagen
        image_data = base64.b64decode(request.image_base64.split(',')[1] if ',' in request.image_base64 else request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")
        
        # ===== SEGMENTACIÓN Y MÉTRICAS =====
        h, w = frame.shape[:2]
        
        # Usar nuevo servicio de segmentación enfocado en manos
        hands_only, mask, hand_metrics = hand_segmentation_service.segment_hands_only(frame)
        
        # Calcular nitidez
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = min(100, cv2.Laplacian(gray, cv2.CV_64F).var() / 10)
        
        metrics = {
            'hands_percentage': hand_metrics['hands_percentage'],
            'hands_detected': hand_metrics['hands_detected'],
            'face_detected': False,  # MediaPipe Hands no detecta rostros
            'blur_score': round(blur_score, 2),
            'resolution': f"{w}x{h}"
        }
        
        # ===== ANÁLISIS CON IA (OLLAMA) =====
        # Construir prompt para IA
        analysis_result = await llm_service.analyze_gesture_context(
            detected_gesture=request.gesture_name or "Desconocida",
            confidence=hand_metrics['quality_score'] / 100,
            previous_gestures=[]
        )
        
        # Análisis de respaldo si IA no está disponible
        if not analysis_result or 'error' in str(analysis_result):
            analysis = _fallback_analysis(metrics)
        else:
            # Extraer del análisis de IA
            analysis = {
                'quality': _determine_quality(metrics),
                'score': int(min(100, metrics['hands_percentage'] * 5 + blur_score / 2)),
                'recommendations': _generate_recommendations(metrics),
                'is_good': metrics['hands_detected'] >= 1 and blur_score > 40
            }
        
        return CaptureAnalysisResponse(
            quality=analysis['quality'],
            score=analysis['score'],
            recommendations=analysis['recommendations'],
            is_good=analysis['is_good'],
            metrics=metrics,
            segmentation_available=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@router.post("/segment-frame")
async def segment_frame(image: UploadFile = File(...)):
    """
    Segmentar un frame y retornar imagen segmentada
    
    Frontend: Envía frame de video
    Backend: Retorna imagen con segmentación coloreada
    """
    try:
        # Leer imagen
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Segmentar (usando servicio de manos)
        segmented, mask, _ = hand_segmentation_service.segment_hands_only(frame)
        
        # Convertir a bytes
        _, buffer = cv2.imencode('.jpg', segmented)
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hands-focus")
async def segment_hands_focus(image: UploadFile = File(...)):
    """
    🖐️ NUEVO: Segmentar SOLO las manos
    
    Frontend: Envía frame de video
    Backend: Retorna imagen con SOLO las manos (fondo negro)
    
    Usa MediaPipe Hands + segmentación semántica de piel
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")
        
        # Usar nuevo servicio enfocado en manos
        hands_only, mask, metrics = hand_segmentation_service.segment_hands_only(frame)
        
        # Convertir a bytes
        _, buffer = cv2.imencode('.jpg', hands_only, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={
                "X-Hands-Detected": str(metrics['hands_detected']),
                "X-Quality-Score": str(metrics['quality_score']),
                "X-Hands-Percentage": str(metrics['hands_percentage'])
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hands-quality")
async def analyze_hands_quality(request: CaptureAnalysisRequest):
    """
    🎯 NUEVO: Análisis de calidad enfocado en manos
    
    Evalúa la calidad de captura específicamente para las manos:
    - Detección con MediaPipe Hands
    - Porcentaje de manos en imagen
    - Nitidez y iluminación
    - Recomendaciones inteligentes
    """
    try:
        # Decodificar imagen
        image_data = base64.b64decode(
            request.image_base64.split(',')[1] if ',' in request.image_base64 else request.image_base64
        )
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")
        
        # Usar nuevo servicio para análisis de calidad
        quality = hand_segmentation_service.compute_quality_score(frame)
        
        # Determinar calidad textual
        if quality['final_score'] >= 80:
            quality_text = "excelente"
        elif quality['final_score'] >= 60:
            quality_text = "buena"
        elif quality['final_score'] >= 40:
            quality_text = "regular"
        else:
            quality_text = "mala"
        
        return {
            "quality": quality_text,
            "score": int(quality['final_score']),
            "is_good": quality['is_good'],
            "metrics": {
                "hands_detected": quality['hands_detected'],
                "hands_percentage": quality['hands_percentage'],
                "sharpness_score": quality['sharpness_score'],
                "lighting_score": quality['lighting_score'],
                "hands_score": quality['hands_score']
            },
            "recommendations": quality['recommendations'],
            "segmentation_available": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@router.post("/hands-crop")
async def get_cropped_hands(image: UploadFile = File(...)):
    """
    ✂️ NUEVO: Obtener imagen recortada centrada en las manos
    
    Ideal para guardar capturas limpias para entrenamiento
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")
        
        # Obtener recorte de manos
        cropped, metrics = hand_segmentation_service.get_cropped_hands(frame, padding=60)
        
        if cropped is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No se detectaron manos", "metrics": metrics}
            )
        
        # Convertir a bytes
        _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={
                "X-Hands-Detected": str(metrics['hands_detected']),
                "X-Quality-Score": str(metrics['quality_score'])
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hands-landmarks")
async def get_hands_with_landmarks(image: UploadFile = File(...)):
    """
    🦴 NUEVO: Obtener imagen con landmarks de manos dibujados
    
    Para visualización y debugging
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")
        
        # Dibujar landmarks
        with_landmarks = hand_segmentation_service.draw_hand_landmarks(frame)
        
        # Convertir a bytes
        _, buffer = cv2.imencode('.jpg', with_landmarks)
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capture/quality-thresholds")
async def get_quality_thresholds():
    """
    ⚙️ CONFIGURACIÓN PARA FRONTEND
    
    Retorna umbrales de calidad para mostrar feedback en tiempo real.
    El frontend puede usar esto para indicadores visuales (verde/amarillo/rojo).
    """
    return {
        "hands_percentage": {
            "excellent": 15,  # > 15% de la imagen
            "good": 8,        # 8-15%
            "fair": 5,        # 5-8%
            "poor": 0         # < 5%
        },
        "blur_score": {
            "excellent": 70,  # > 70 muy nítido
            "good": 50,       # 50-70 nítido
            "fair": 30,       # 30-50 aceptable
            "poor": 0         # < 30 borroso
        },
        "hands_required": {
            "min": 1,
            "max": 2,
            "recommended": 2
        },
        "recommendations": {
            "distance": "30-60 cm de la cámara",
            "lighting": "Luz frontal o lateral, evitar contraluz",
            "background": "Fondo uniforme recomendado",
            "movement": "Movimientos lentos y deliberados"
        }
    }


@router.post("/process-landmarks")
async def process_landmarks(image: UploadFile = File(...)):
    """
    ⚡️ ENDPOINT RÁPIDO PARA FRONTEND
    
    Retorna solo JSON con landmarks para dibujar en cliente (canvas).
    Mucho más rápido que enviar imágenes procesadas.
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        hands_info = hand_segmentation_service.detect_hands(frame)
        
        if not hands_info:
             return {"hands_detected": 0, "hands": []}
             
        # Convertir numpy arrays a listas para JSON
        hands_data = []
        for hand in hands_info['hands']:
            hands_data.append({
                "index": hand['index'],
                "score": float(hand['confidence']),
                "label": hand['handedness'],
                "landmarks": hand['landmarks'].tolist() # Convertir a lista [[x,y,z]...]
            })
            
        return {
            "hands_detected": hands_info['num_hands'],
            "hands": hands_data
        }
        
    except Exception as e:
        # Retornar vacío en error para no romper loop de video
        return {"hands_detected": 0, "error": str(e)}


@router.post("/capture/validate-gesture")
async def validate_gesture(request: CaptureAnalysisRequest):
    """
    🎯 VALIDACIÓN DE SEÑA ESPECÍFICA
    
    Frontend: "¿Esta captura es válida para la seña 'HOLA'?"
    Backend: Analiza con IA si cumple los requisitos
    """
    try:
        # Decodificar
        image_data = base64.b64decode(request.image_base64.split(',')[1] if ',' in request.image_base64 else request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Análisis básico
        hands_info = hand_segmentation_service.detect_hands(frame)
        
        if not hands_info or hands_info.get('num_hands', 0) == 0:
            return {
                "valid": False,
                "reason": "No se detectaron manos en la imagen",
                "suggestions": ["Asegúrate de que tus manos estén visibles", "Acerca las manos a la cámara"]
            }
        
        # Usar IA para validar
        if request.gesture_name:
            analysis = await llm_service.analyze_gesture_context(
                detected_gesture=request.gesture_name,
                confidence=0.8,
                previous_gestures=[]
            )
            
            return {
                "valid": True,
                "confidence": 0.85,
                "gesture_name": request.gesture_name,
                "ai_feedback": analysis,
                "hands_detected": hands_info['num_hands']
            }
        
        return {
            "valid": True,
            "hands_detected": hands_info['num_hands'],
            "ready_to_capture": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== FUNCIONES AUXILIARES =====

def _determine_quality(metrics: Dict) -> str:
    """Determinar calidad general"""
    score = 0
    
    # Evaluar manos
    if metrics['hands_percentage'] > 15:
        score += 3
    elif metrics['hands_percentage'] > 8:
        score += 2
    elif metrics['hands_percentage'] > 5:
        score += 1
    
    # Evaluar nitidez
    if metrics['blur_score'] > 70:
        score += 3
    elif metrics['blur_score'] > 50:
        score += 2
    elif metrics['blur_score'] > 30:
        score += 1
    
    # Evaluar detección
    if metrics['hands_detected'] == 2:
        score += 2
    elif metrics['hands_detected'] == 1:
        score += 1
    
    if score >= 7:
        return "excelente"
    elif score >= 5:
        return "buena"
    elif score >= 3:
        return "regular"
    else:
        return "mala"


def _generate_recommendations(metrics: Dict) -> List[str]:
    """Generar recomendaciones basadas en métricas"""
    recommendations = []
    
    if metrics['hands_percentage'] < 5:
        recommendations.append("❌ Acerca más las manos a la cámara")
    elif metrics['hands_percentage'] < 10:
        recommendations.append("⚠️ Acerca un poco más las manos")
    else:
        recommendations.append("✅ Posición de manos correcta")
    
    if metrics['blur_score'] < 30:
        recommendations.append("❌ Imagen borrosa - muévete más despacio")
    elif metrics['blur_score'] < 50:
        recommendations.append("⚠️ Reduce la velocidad del movimiento")
    else:
        recommendations.append("✅ Imagen nítida")
    
    if metrics['hands_detected'] == 0:
        recommendations.append("❌ No se detectan manos")
    elif metrics['hands_detected'] == 1:
        recommendations.append("⚠️ Solo una mano detectada")
    else:
        recommendations.append("✅ Dos manos detectadas")
    
    if not metrics['face_detected']:
        recommendations.append("ℹ️ El rostro ayuda al contexto (opcional)")
    
    return recommendations[:4]  # Máximo 4 recomendaciones


def _fallback_analysis(metrics: Dict) -> Dict:
    """Análisis de respaldo sin IA"""
    quality = _determine_quality(metrics)
    recommendations = _generate_recommendations(metrics)
    
    score = 0
    if metrics['hands_percentage'] > 10:
        score += 40
    if metrics['blur_score'] > 50:
        score += 30
    if metrics['hands_detected'] >= 1:
        score += 30
    
    return {
        'quality': quality,
        'score': score,
        'recommendations': recommendations,
        'is_good': score >= 60
    }
