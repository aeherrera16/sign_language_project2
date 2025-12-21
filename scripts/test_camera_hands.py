#!/usr/bin/env python3
"""
Script de Prueba de Captura de Manos con Cámara en Tiempo Real

Este script te permite probar la detección de manos y la integración completa
del sistema antes de comenzar el entrenamiento del modelo.

USO:
    cd /Users/anahy/sign_language_project2
    source .venv/bin/activate
    python3 scripts/test_camera_hands.py

CONTROLES:
    - 'q' = Salir
    - 's' = Guardar captura actual
    - 'm' = Cambiar modo (hands-focus / landmarks / skin)
    - SPACE = Congelar/descongelar frame
"""

import cv2
import numpy as np
import sys
import os
import json
import requests
from datetime import datetime

# Agregar path del backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.hand_segmentation_service import HandSegmentationService

# Colores para UI
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def get_quality_color(score):
    """Retorna color BGR basado en score"""
    if score >= 80:
        return GREEN
    elif score >= 60:
        return YELLOW
    elif score >= 40:
        return (0, 165, 255)  # Orange
    return RED

def draw_text_with_bg(img, text, pos, font_scale=0.6, color=WHITE, bg_color=BLACK):
    """Dibujar texto con fondo"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 4), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

def main():
    print("=" * 60)
    print("  PRUEBA DE CAPTURA DE MANOS - Sign Language Project")
    print("=" * 60)
    print()
    print("Iniciando cámara...")
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        print("Intenta con un índice diferente (1, 2, etc.)")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Inicializar servicio de detección
    print("Inicializando servicio de segmentación de manos...")
    service = HandSegmentationService()
    print("¡Listo!")
    print()
    print("CONTROLES:")
    print("  q = Salir")
    print("  s = Guardar captura")
    print("  m = Cambiar modo")
    print("  SPACE = Congelar/descongelar")
    print()
    
    mode = 'hands-focus'  # 'hands-focus', 'landmarks', 'quality'
    modes = ['hands-focus', 'landmarks', 'quality']
    mode_idx = 0
    
    frozen = False
    last_frame = None
    capture_count = 0
    
    # Estadísticas
    detections = 0
    total_frames = 0
    
    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                print("Error leyendo frame")
                break
            frame = cv2.flip(frame, 1)  # Mirror
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()
        
        total_frames += 1
        
        # Procesar según modo
        display_right = np.zeros_like(frame)
        metrics = {}
        
        try:
            if mode == 'hands-focus':
                # Segmentar solo las manos
                hands_only, mask, metrics = service.segment_hands_only(frame)
                display_right = hands_only
                
            elif mode == 'landmarks':
                # Dibujar landmarks
                display_right = service.draw_hand_landmarks(frame)
                info = service.detect_hands(frame)
                if info:
                    metrics = {
                        'hands_detected': info['num_hands'],
                        'hands_info': [{'handedness': h['handedness'], 'confidence': h['confidence']} 
                                      for h in info['hands']]
                    }
                else:
                    metrics = {'hands_detected': 0}
                    
            elif mode == 'quality':
                # Análisis completo de calidad
                metrics = service.compute_quality_score(frame)
                hands_only, mask, _ = service.segment_hands_only(frame)
                display_right = hands_only
        
        except Exception as e:
            print(f"Error procesando: {e}")
            metrics = {'hands_detected': 0, 'error': str(e)}
        
        # Actualizar estadísticas
        if metrics.get('hands_detected', 0) > 0:
            detections += 1
        
        # === DIBUJAR UI ===
        
        # Título del modo
        draw_text_with_bg(frame, f"ORIGINAL - Modo: {mode.upper()}", (10, 25), 0.7)
        draw_text_with_bg(display_right, "PROCESADO", (10, 25), 0.7)
        
        # Información de manos detectadas
        hands = metrics.get('hands_detected', 0)
        color = GREEN if hands > 0 else RED
        draw_text_with_bg(frame, f"Manos: {hands}", (10, 55), 0.6, color)
        
        # Mostrar confidence de cada mano
        y_pos = 80
        for i, hand in enumerate(metrics.get('hands_info', [])):
            hand_type = hand.get('handedness', 'Unknown')
            conf = hand.get('confidence', 0)
            text = f"  {hand_type}: {conf:.1%}"
            draw_text_with_bg(frame, text, (10, y_pos), 0.5, GREEN)
            y_pos += 22
        
        # Mostrar métricas adicionales según modo
        if mode == 'hands-focus':
            pct = metrics.get('hands_percentage', 0)
            score = metrics.get('quality_score', 0)
            draw_text_with_bg(frame, f"Cobertura: {pct:.1f}%", (10, y_pos), 0.5)
            y_pos += 22
            draw_text_with_bg(frame, f"Calidad: {score:.0f}/100", (10, y_pos), 0.5, 
                             get_quality_color(score))
            
        elif mode == 'quality':
            final_score = metrics.get('final_score', 0)
            is_good = metrics.get('is_good', False)
            color = GREEN if is_good else RED
            draw_text_with_bg(frame, f"Score Final: {final_score:.0f}/100", (10, y_pos), 0.6, color)
            y_pos += 25
            
            for rec in metrics.get('recommendations', [])[:3]:
                draw_text_with_bg(frame, rec[:40], (10, y_pos), 0.4)
                y_pos += 18
        
        # Estadísticas globales
        accuracy = (detections / total_frames * 100) if total_frames > 0 else 0
        draw_text_with_bg(frame, f"Detection Rate: {accuracy:.1f}%", (10, 460), 0.5)
        
        # Estado de frozen
        if frozen:
            draw_text_with_bg(frame, "CONGELADO", (280, 25), 0.8, YELLOW)
        
        # Combinar frames lado a lado
        combined = np.hstack([frame, display_right])
        
        # Mostrar
        cv2.imshow('Test Captura de Manos - Sign Language Project', combined)
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nSaliendo...")
            break
            
        elif key == ord('s'):
            # Guardar captura
            capture_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Guardar imagen original
            orig_path = f"/tmp/capture_{timestamp}_original.jpg"
            cv2.imwrite(orig_path, last_frame)
            
            # Guardar imagen procesada
            proc_path = f"/tmp/capture_{timestamp}_processed.jpg"
            cv2.imwrite(proc_path, display_right)
            
            # Guardar métricas
            metrics_path = f"/tmp/capture_{timestamp}_metrics.json"
            with open(metrics_path, 'w') as f:
                # Convertir numpy types a python types
                clean_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean_metrics[k] = float(v)
                    else:
                        clean_metrics[k] = v
                json.dump(clean_metrics, f, indent=2)
            
            print(f"\n✅ Captura #{capture_count} guardada:")
            print(f"   Original: {orig_path}")
            print(f"   Procesado: {proc_path}")
            print(f"   Métricas: {metrics_path}")
            
            # También intentar guardar al backend
            try:
                with open(orig_path, 'rb') as f:
                    r = requests.post('http://localhost:8000/api/capture/save',
                                    files={'image': (f'capture_{timestamp}.jpg', f, 'image/jpeg')},
                                    data={'gesture_id': '', 'metadata': json.dumps(clean_metrics)},
                                    timeout=5)
                if r.status_code == 200:
                    print(f"   Backend: Guardado con ID {r.json().get('id')}")
            except Exception as e:
                print(f"   Backend: No disponible ({e})")
            
        elif key == ord('m'):
            # Cambiar modo
            mode_idx = (mode_idx + 1) % len(modes)
            mode = modes[mode_idx]
            print(f"\n🔄 Modo cambiado a: {mode}")
            
        elif key == ord(' '):
            # Toggle frozen
            frozen = not frozen
            if frozen:
                print("\n⏸️ Frame congelado")
            else:
                print("\n▶️ Captura reanudada")
    
    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    
    # Resumen final
    print()
    print("=" * 60)
    print("  RESUMEN DE PRUEBA")
    print("=" * 60)
    print(f"  Frames procesados: {total_frames}")
    print(f"  Detecciones de manos: {detections}")
    print(f"  Tasa de detección: {accuracy:.1f}%")
    print(f"  Capturas guardadas: {capture_count}")
    print()
    
    if accuracy > 80:
        print("  ✅ ¡Excelente! La detección funciona muy bien.")
        print("     Puedes proceder con el entrenamiento del modelo.")
    elif accuracy > 50:
        print("  ⚠️ La detección es aceptable pero podría mejorar.")
        print("     Considera mejorar la iluminación o el ángulo de la cámara.")
    else:
        print("  ❌ La tasa de detección es baja.")
        print("     Verifica la iluminación, distancia a la cámara y fondo.")
    print()

if __name__ == "__main__":
    main()
