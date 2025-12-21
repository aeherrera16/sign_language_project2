"""
Router para gestión de señas (gestures)
Endpoints para capturar, listar y gestionar señas
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import base64
from datetime import datetime
from typing import List

from services.hand_segmentation_service import hand_segmentation_service

router = APIRouter()

@router.post("/capture")
async def capture_gesture(
    gesture_name: str = Form(...),
    video: UploadFile = File(...)
):
    """
    Captura una nueva seña desde video/imagen
    Aplica segmentación semántica y extrae landmarks
    """
    try:
        # Leer el archivo subido
        contents = await video.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen")
        
        # 1. Segmentar manos para verificar calidad
        hands_only, mask, metrics = hand_segmentation_service.segment_hands_only(image)
        
        if metrics["hands_detected"] == 0:
            return {
                "success": False,
                "error": "No se detectaron manos"
            }
            
        # 2. Obtener landmarks completos
        hands_info = hand_segmentation_service.detect_hands(image)
        if not hands_info:
             return JSONResponse(
                status_code=400,
                content={"error": "No se pudieron extraer landmarks"}
            )
        
        # Guardar datos
        gesture_dir = f"data/gestures/{gesture_name}"
        os.makedirs(gesture_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar imagen segmentada
        cv2.imwrite(
            f"{gesture_dir}/segmented_{timestamp}.jpg",
            hands_only
        )
        
        # Guardar máscara
        cv2.imwrite(
            f"{gesture_dir}/mask_{timestamp}.jpg",
            mask
        )
        
        # 7. Guardar landmarks como numpy
        all_landmarks = [h['landmarks'] for h in hands_info['hands']]
        np.save(
            f"{gesture_dir}/landmarks_{timestamp}.npy",
            np.array(all_landmarks)
        )
        
        # Codificar imagen segmentada a base64 para feedback instantáneo
        _, buffer = cv2.imencode('.jpg', hands_only, [cv2.IMWRITE_JPEG_QUALITY, 70])
        segmented_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "gesture_name": gesture_name,
            "timestamp": timestamp,
            "num_hands": hands_info["num_hands"],
            "segmented_image": f"data:image/jpeg;base64,{segmented_b64}",
            "message": f"Seña '{gesture_name}' capturada exitosamente"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_gestures():
    """Lista todas las señas capturadas"""
    try:
        gestures_dir = "data/gestures"
        
        if not os.path.exists(gestures_dir):
            return {"gestures": []}
        
        gestures = []
        for gesture_name in os.listdir(gestures_dir):
            gesture_path = os.path.join(gestures_dir, gesture_name)
            
            if os.path.isdir(gesture_path):
                # Contar muestras
                samples = [f for f in os.listdir(gesture_path) if f.startswith("landmarks_")]
                
                gestures.append({
                    "name": gesture_name,
                    "samples": len(samples),
                    "path": gesture_path
                })
        
        return {
            "total": len(gestures),
            "gestures": gestures
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{gesture_name}")
async def delete_gesture(gesture_name: str):
    """Elimina una seña del dataset"""
    try:
        gesture_path = f"data/gestures/{gesture_name}"
        
        if not os.path.exists(gesture_path):
            raise HTTPException(status_code=404, detail=f"Seña '{gesture_name}' no encontrada")
        
        import shutil
        shutil.rmtree(gesture_path)
        
        return {
            "success": True,
            "message": f"Seña '{gesture_name}' eliminado"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-capture")
async def batch_capture_gesture(
    gesture_name: str = Form(...),
    num_samples: int = Form(100)
):
    """
    Captura múltiples muestras de una seña en tiempo real
    (Requiere cámara en el servidor - Útil para testing local)
    """
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="No se pudo acceder a la cámara")
        
        gesture_dir = f"data/gestures/{gesture_name}"
        os.makedirs(gesture_dir, exist_ok=True)
        
        captured = 0
        timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            hands_only, mask, metrics = hand_segmentation_service.segment_hands_only(frame)
            
            if metrics["hands_detected"] > 0:
                # Obtener landmarks para guardar
                hands_info = hand_segmentation_service.detect_hands(frame)
                
                if hands_info:
                    # Guardar imagen segmentada
                    cv2.imwrite(
                        f"{gesture_dir}/segmented_{timestamp_base}_{captured}.jpg",
                        hands_only
                    )
                    
                    # Guardar landmarks
                    all_landmarks = [h['landmarks'] for h in hands_info['hands']]
                    np.save(
                        f"{gesture_dir}/landmarks_{timestamp_base}_{captured}.npy",
                        np.array(all_landmarks)
                    )
                    
                    captured += 1
        
        cap.release()
        
        return {
            "success": True,
            "gesture_name": gesture_name,
            "samples_captured": captured,
            "message": f"{captured} muestras capturadas de '{gesture_name}'"
        }
        
    except Exception as e:
        if 'cap' in locals():
            cap.release()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cap' in locals():
            cap.release()
