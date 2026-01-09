#!/usr/bin/env python3
"""
Regenera landmarks para que todos tengan formato de 2 manos (126 features)
Esto arregla landmarks antiguos que solo tenían 1 mano.
"""

import os
import numpy as np
from pathlib import Path

GESTURES_DIR = Path("backend/data/gestures")

def fix_landmarks():
    """Regenera todos los landmarks para tener 2 manos"""
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    for gesture_dir in GESTURES_DIR.iterdir():
        if not gesture_dir.is_dir():
            continue
        
        gesture_name = gesture_dir.name
        print(f"\n📁 Procesando: {gesture_name}")
        
        for file in gesture_dir.glob("landmarks_*.npy"):
            try:
                landmarks = np.load(file)
                original_shape = landmarks.shape
                
                # Verificar si necesita arreglo
                if landmarks.ndim == 3 and landmarks.shape[0] == 2:
                    # Ya tiene 2 manos, está bien
                    skipped_count += 1
                    continue
                
                # Arreglar: convertir a formato (2, 21, 3)
                if landmarks.ndim == 3 and landmarks.shape[0] == 1:
                    # Solo 1 mano, agregar segunda con ceros
                    fixed_landmarks = np.zeros((2, 21, 3))
                    fixed_landmarks[0] = landmarks[0]
                    np.save(file, fixed_landmarks)
                    fixed_count += 1
                    print(f"   ✅ {file.name}: {original_shape} → (2, 21, 3)")
                    
                elif landmarks.ndim == 2 and landmarks.shape == (21, 3):
                    # Una mano sin la dimensión batch
                    fixed_landmarks = np.zeros((2, 21, 3))
                    fixed_landmarks[0] = landmarks
                    np.save(file, fixed_landmarks)
                    fixed_count += 1
                    print(f"   ✅ {file.name}: {original_shape} → (2, 21, 3)")
                    
                else:
                    print(f"   ⚠️ {file.name}: Formato desconocido {original_shape}")
                    error_count += 1
                    
            except Exception as e:
                print(f"   ❌ {file.name}: Error - {e}")
                error_count += 1
    
    print(f"\n{'='*50}")
    print(f"✅ Arreglados: {fixed_count}")
    print(f"⏭️ Saltados (ya OK): {skipped_count}")
    print(f"❌ Errores: {error_count}")
    print(f"{'='*50}")
    
    return fixed_count, skipped_count, error_count


if __name__ == "__main__":
    print("🔧 REGENERANDO LANDMARKS A FORMATO 2 MANOS")
    print("="*50)
    fix_landmarks()
    print("\n✅ Listo! Ahora puedes reentrenar el modelo.")
