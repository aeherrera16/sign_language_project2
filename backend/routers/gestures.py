"""
Router para gestión de señas (gestures)
Endpoints para capturar, listar y gestionar señas

CON MULTIPLICACIÓN AUTOMÁTICA x100 para mejor entrenamiento
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import base64
from datetime import datetime
from typing import List

from services.hand_segmentation_service import hand_segmentation_service

router = APIRouter()


def augment_single_landmark(landmarks: np.ndarray, aug_type: int) -> np.ndarray:
    """
    Aplica una variación a los landmarks para Data Augmentation.
    
    Args:
        landmarks: Array shape (2, 21, 3)
        aug_type: Tipo de augmentación (0-8)
    
    Returns:
        Landmarks aumentados
    """
    augmented = landmarks.copy().astype(np.float64)
    
    if aug_type == 0:
        # Ruido pequeño
        noise = np.random.normal(0, 2.0, augmented.shape)
        augmented += noise
        
    elif aug_type == 1:
        # Ruido medio
        noise = np.random.normal(0, 5.0, augmented.shape)
        augmented += noise
        
    elif aug_type == 2:
        # Escalado pequeño (acercar)
        scale = np.random.uniform(1.02, 1.08)
        augmented[:, :, :2] *= scale
        
    elif aug_type == 3:
        # Escalado pequeño (alejar)
        scale = np.random.uniform(0.92, 0.98)
        augmented[:, :, :2] *= scale
        
    elif aug_type == 4:
        # Traslación X
        shift = np.random.uniform(-10, 10)
        augmented[:, :, 0] += shift
        
    elif aug_type == 5:
        # Traslación Y
        shift = np.random.uniform(-10, 10)
        augmented[:, :, 1] += shift
        
    elif aug_type == 6:
        # Rotación pequeña (-5° a +5°)
        angle = np.random.uniform(-0.087, 0.087)  # radianes
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for h in range(2):
            for p in range(21):
                x, y = augmented[h, p, 0], augmented[h, p, 1]
                augmented[h, p, 0] = x * cos_a - y * sin_a
                augmented[h, p, 1] = x * sin_a + y * cos_a
                
    elif aug_type == 7:
        # Ruido + escalado
        noise = np.random.normal(0, 1.5, augmented.shape)
        scale = np.random.uniform(0.97, 1.03)
        augmented = (augmented + noise) * scale
        
    else:  # aug_type == 8
        # Todo combinado suave
        noise = np.random.normal(0, 1.0, augmented.shape)
        scale = np.random.uniform(0.98, 1.02)
        shift_x = np.random.uniform(-5, 5)
        shift_y = np.random.uniform(-5, 5)
        augmented += noise
        augmented[:, :, :2] *= scale
        augmented[:, :, 0] += shift_x
        augmented[:, :, 1] += shift_y
    
    return augmented.astype(np.float64)


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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
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
        
        # 7. Guardar landmarks como numpy - SIEMPRE 2 manos (126 features)
        all_landmarks = [h['landmarks'] for h in hands_info['hands']]
        
        # IMPORTANTE: Siempre guardar exactamente 2 manos para consistencia con el modelo
        while len(all_landmarks) < 2:
            all_landmarks.append(np.zeros((21, 3)))  # Rellenar segunda mano con ceros
        
        # Tomar solo las primeras 2 manos si hay más
        all_landmarks = all_landmarks[:2]
        landmarks_array = np.array(all_landmarks)
        
        # ═══════════════════════════════════════════════════════════════════════
        # MULTIPLICACIÓN AUTOMÁTICA x20: Guardar original + 19 variaciones
        # (El entrenamiento aplica x100 adicional en memoria para total x2000)
        # ═══════════════════════════════════════════════════════════════════════
        samples_saved = 0
        
        # 1. Guardar original
        np.save(f"{gesture_dir}/landmarks_{timestamp}.npy", landmarks_array)
        samples_saved += 1
        
        # 2. Guardar 19 variaciones aumentadas (cicla por los 9 tipos de augmentación)
        for aug_idx in range(19):
            aug_type = aug_idx % 9  # Ciclar entre 0-8 tipos de augmentación
            augmented = augment_single_landmark(landmarks_array, aug_type)
            aug_timestamp = f"{timestamp}_aug{aug_idx:02d}"
            np.save(f"{gesture_dir}/landmarks_{aug_timestamp}.npy", augmented)
            samples_saved += 1
        
        # Codificar imagen segmentada a base64 para feedback instantáneo
        _, buffer = cv2.imencode('.jpg', hands_only, [cv2.IMWRITE_JPEG_QUALITY, 70])
        segmented_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "gesture_name": gesture_name,
            "timestamp": timestamp,
            "num_hands": hands_info["num_hands"],
            "samples_saved": samples_saved,  # Ahora guarda 10 muestras por captura
            "segmented_image": f"data:image/jpeg;base64,{segmented_b64}",
            "message": f"Seña '{gesture_name}' capturada exitosamente (x{samples_saved} muestras)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capture-sequence")
async def capture_gesture_sequence(request: Request):
    """
    🎬 NUEVO: Captura una secuencia de frames para señas DINÁMICAS (con movimiento)
    
    Recibe múltiples frames (frame_0, frame_1, ...) y extrae landmarks de cada uno.
    Guarda la secuencia temporal completa para entrenar modelos LSTM/Transformer.
    
    Esto permite reconocer señas que requieren movimiento, no solo posiciones estáticas.
    """
    try:
        form = await request.form()
        
        gesture_name = form.get('gesture_name')
        sequence_length = int(form.get('sequence_length', 30))
        
        if not gesture_name:
            raise HTTPException(status_code=400, detail="Se requiere gesture_name")
        
        # Directorio para guardar
        gesture_dir = f"data/gestures/{gesture_name}"
        os.makedirs(gesture_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Procesar cada frame de la secuencia
        sequence_landmarks = []
        valid_frames = 0
        preview_image = None
        
        for i in range(sequence_length):
            frame_key = f"frame_{i}"
            if frame_key not in form:
                continue
            
            frame_file = form[frame_key]
            contents = await frame_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                # Frame inválido, agregar zeros para mantener la secuencia
                sequence_landmarks.append(np.zeros((2, 21, 3)))
                continue
            
            # Detectar manos en este frame
            hands_info = hand_segmentation_service.detect_hands(image)
            
            if hands_info and hands_info['num_hands'] > 0:
                # Extraer landmarks
                frame_landmarks = []
                for hand in hands_info['hands']:
                    frame_landmarks.append(hand['landmarks'])
                
                # Asegurar siempre 2 manos (rellenar con zeros si falta una)
                while len(frame_landmarks) < 2:
                    frame_landmarks.append(np.zeros((21, 3)))
                
                sequence_landmarks.append(np.array(frame_landmarks[:2]))
                valid_frames += 1
                
                # Guardar un frame del medio como preview
                if i == sequence_length // 2:
                    hands_only, _, _ = hand_segmentation_service.segment_hands_only(image)
                    _, buffer = cv2.imencode('.jpg', hands_only, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    preview_image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            else:
                # No se detectaron manos, agregar zeros
                sequence_landmarks.append(np.zeros((2, 21, 3)))
        
        # Verificar que tengamos suficientes frames válidos (al menos 50%)
        if valid_frames < sequence_length * 0.5:
            return {
                "success": False,
                "error": f"Solo {valid_frames}/{sequence_length} frames válidos. Intenta de nuevo con manos más visibles."
            }
        
        # Convertir a numpy array: shape (num_frames, 2, 21, 3)
        sequence_array = np.array(sequence_landmarks)
        
        # Guardar la secuencia completa
        sequence_filename = f"sequence_{timestamp}.npy"
        np.save(f"{gesture_dir}/{sequence_filename}", sequence_array)
        
        # También guardar metadata
        metadata = {
            "gesture_name": gesture_name,
            "timestamp": timestamp,
            "sequence_length": len(sequence_landmarks),
            "valid_frames": valid_frames,
            "shape": list(sequence_array.shape),
            "type": "dynamic"  # Marcador de que es una seña dinámica
        }
        
        import json
        with open(f"{gesture_dir}/sequence_meta_{timestamp}.json", "w") as f:
            json.dump(metadata, f)
        
        return {
            "success": True,
            "gesture_name": gesture_name,
            "timestamp": timestamp,
            "valid_frames": valid_frames,
            "total_frames": len(sequence_landmarks),
            "preview_image": preview_image,
            "message": f"Secuencia de '{gesture_name}' guardada ({valid_frames} frames válidos)"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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
                # Contar muestras estáticas
                static_samples = [f for f in os.listdir(gesture_path) if f.startswith("landmarks_")]
                # Contar secuencias dinámicas (incluyendo las del scraper 'conadis_')
                dynamic_samples = [f for f in os.listdir(gesture_path) 
                                 if (f.startswith("sequence_") or f.startswith("conadis_")) 
                                 and f.endswith(".npy")]
                
                total_count = len(static_samples) + len(dynamic_samples)
                gestures.append({
                    "name": gesture_name,
                    "samples": total_count, # Mostrar total para que el frontend no diga '0 img'
                    "sequences": len(dynamic_samples),
                    "total": total_count,
                    "type": "dynamic" if len(dynamic_samples) > len(static_samples) else "static",
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
                    
                    # Guardar landmarks - SIEMPRE 2 manos
                    all_landmarks = [h['landmarks'] for h in hands_info['hands']]
                    while len(all_landmarks) < 2:
                        all_landmarks.append(np.zeros((21, 3)))
                    all_landmarks = all_landmarks[:2]
                    
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

