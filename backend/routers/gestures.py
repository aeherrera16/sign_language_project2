"""
Router para gestión de señas (gestures)
Endpoints para capturar, listar y gestionar señas
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
from datetime import datetime
from typing import List

from services.segmentation_service import segmentation_service

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
        
        # Procesar con segmentación semántica
        result = segmentation_service.process_frame(image)
        
        if not result["has_hands"]:
            return JSONResponse(
                status_code=400,
                content={"error": "No se detectaron manos en la imagen"}
            )
        
        # Guardar datos
        gesture_dir = f"data/gestures/{gesture_name}"
        os.makedirs(gesture_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar imagen segmentada
        cv2.imwrite(
            f"{gesture_dir}/segmented_{timestamp}.jpg",
            result["segmented_image"]
        )
        
        # Guardar máscara
        cv2.imwrite(
            f"{gesture_dir}/mask_{timestamp}.jpg",
            result["mask"]
        )
        
        # Guardar landmarks como numpy
        landmarks_data = np.array(result["landmarks"]["landmarks"])
        np.save(
            f"{gesture_dir}/landmarks_{timestamp}.npy",
            landmarks_data
        )
        
        return {
            "success": True,
            "gesture_name": gesture_name,
            "timestamp": timestamp,
            "num_hands": result["landmarks"]["num_hands"],
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
            "message": f"Seña '{gesture_name}' eliminada"
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
    (Requiere cámara en el servidor)
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
            result = segmentation_service.process_frame(frame)
            
            if result["has_hands"]:
                # Guardar
                cv2.imwrite(
                    f"{gesture_dir}/segmented_{timestamp_base}_{captured}.jpg",
                    result["segmented_image"]
                )
                
                landmarks_data = np.array(result["landmarks"]["landmarks"])
                np.save(
                    f"{gesture_dir}/landmarks_{timestamp_base}_{captured}.npy",
                    landmarks_data
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
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cap' in locals():
            cap.release()
