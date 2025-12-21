import cv2
import os
import numpy as np
import mediapipe as mp
import sys

# Añadir root al path para importar servicios
sys.path.append(os.getcwd())
try:
    from backend.services.hand_segmentation_service import HandSegmentationService
except ImportError:
    # Fallback si ejecuta desde scripts/
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from backend.services.hand_segmentation_service import HandSegmentationService

def regenerate():
    print("🔄 Iniciando regeneración de landmarks...")
    DATA_DIR = "backend/data/gestures"
    if not os.path.exists(DATA_DIR):
        print(f"No data dir at {DATA_DIR}")
        return

    service = HandSegmentationService()
    
    total_processed = 0
    total_errors = 0

    for gesture_name in os.listdir(DATA_DIR):
        gesture_path = os.path.join(DATA_DIR, gesture_name)
        if not os.path.isdir(gesture_path):
            continue
            
        print(f"📁 Procesando carpeta: {gesture_name}")
        images = [f for f in os.listdir(gesture_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(gesture_path, img_name)
            npy_name = f"landmarks_{os.path.splitext(img_name)[0]}.npy"
            # O usar timestamp si el nombre es capture_TIMESTAMP.jpg
            # Simplemente usemos el mismo nombre base.
            npy_path = os.path.join(gesture_path, npy_name)
            
            # Si ya existe, saltar (o sobrescribir si queremos forzar)
            # if os.path.exists(npy_path): continue

            try:
                # Leer imagen
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # Detectar manos
                hands_info = service.detect_hands(image)
                
                if hands_info["num_hands"] > 0:
                    # Extraer landmarks
                    all_landmarks = [h['landmarks'] for h in hands_info['hands']]
                    np.save(npy_path, np.array(all_landmarks))
                    total_processed += 1
                    print(f"  ✅ Generado: {npy_name}", end='\r')
                else:
                    # No manos detectadas en la foto guardada
                    # print(f"  ⚠️ No manos en: {img_name}")
                    total_errors += 1
            except Exception as e:
                print(f"  ❌ Error en {img_name}: {e}")
        
        print(f"\nTerminado {gesture_name}")

    print(f"\n🎉 Finalizado. Procesados: {total_processed}, Sin manos: {total_errors}")

if __name__ == "__main__":
    regenerate()
