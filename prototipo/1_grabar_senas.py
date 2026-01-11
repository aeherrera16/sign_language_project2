#!/usr/bin/env python3
"""
GRABADOR DE SECUENCIAS - MediaPipe + LSTM
Basado en técnicas de: Morfín-Chávez (2023), Sincan & Keles (2020)

Captura secuencias de 30 frames de landmarks para entrenar modelo LSTM.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
from datetime import datetime

# === CONFIGURACIÓN ===
FRAMES_SECUENCIA = 30      # ~1 segundo a 30fps
LANDMARKS_MANO = 21        # MediaPipe: 21 puntos por mano
COORDS = 3                 # x, y, z
FEATURES = LANDMARKS_MANO * COORDS * 2   # 126 (2 manos)

DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")

# === MEDIAPIPE ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


def extraer_landmarks(frame):
    """
    Extrae 126 features de las manos (2 × 21 × 3).
    Normaliza coordenadas relativas a la muñeca.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    features = np.zeros(FEATURES)
    
    if result.multi_hand_landmarks:
        for idx, hand_lm in enumerate(result.multi_hand_landmarks[:2]):
            # Normalizar respecto a la muñeca (punto 0)
            wrist = hand_lm.landmark[0]
            
            for i, lm in enumerate(hand_lm.landmark):
                base = idx * 63 + i * 3
                # Coordenadas relativas a la muñeca
                features[base] = lm.x - wrist.x
                features[base + 1] = lm.y - wrist.y
                features[base + 2] = lm.z - wrist.z
    
    return features, result


def guardar_datos(nombre_sena, secuencias):
    """Guarda secuencias en formato JSON."""
    os.makedirs(os.path.join(DIR_DATOS, nombre_sena), exist_ok=True)
    
    archivo = os.path.join(
        DIR_DATOS, nombre_sena, 
        f"seq_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    
    with open(archivo, 'w') as f:
        json.dump({
            "sena": nombre_sena,
            "frames": FRAMES_SECUENCIA,
            "features": FEATURES,
            "secuencias": [s.tolist() for s in secuencias]
        }, f)
    
    print(f"✅ {len(secuencias)} secuencias guardadas")


def main():
    print("\n" + "="*60)
    print("  GRABADOR DE SEÑAS DINÁMICAS")
    print("  Técnica: MediaPipe Landmarks + Secuencias Temporales")
    print("="*60)
    
    nombre = input("\nNombre de la seña: ").strip().upper()
    if not nombre:
        print("❌ Nombre inválido")
        return
    
    print(f"\n🎯 Grabando: {nombre}")
    print(f"   Frames por secuencia: {FRAMES_SECUENCIA}")
    print(f"   Features por frame: {FEATURES}")
    print("\n[G] Grabar | [Q] Guardar y salir\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    secuencias = []
    buffer = []
    grabando = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        features, result = extraer_landmarks(frame)
        
        # Dibujar manos
        hay_mano = False
        if result.multi_hand_landmarks:
            hay_mano = True
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                    mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
                )
        
        # Grabación
        if grabando:
            buffer.append(features)
            progreso = len(buffer) / FRAMES_SECUENCIA
            
            # Barra de progreso
            cv2.rectangle(frame, (20, 440), (620, 465), (50,50,50), -1)
            cv2.rectangle(frame, (20, 440), (int(20 + 600*progreso), 465), (0,255,0), -1)
            cv2.putText(frame, f"Grabando: {len(buffer)}/{FRAMES_SECUENCIA}", 
                       (230, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            if len(buffer) >= FRAMES_SECUENCIA:
                secuencias.append(np.array(buffer))
                buffer = []
                grabando = False
                print(f"  ✓ Secuencia {len(secuencias)} completada")
        
        # UI
        cv2.rectangle(frame, (0, 0), (640, 50), (40,40,40), -1)
        cv2.putText(frame, f"Sena: {nombre}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, f"Total: {len(secuencias)}", (450, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # Indicador de mano
        color = (0,255,0) if hay_mano else (0,0,255)
        cv2.circle(frame, (620, 25), 12, color, -1)
        
        if not grabando:
            cv2.putText(frame, "[G] Grabar  [Q] Salir", (200, 475),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        
        cv2.imshow('Grabador LSE', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g') and hay_mano and not grabando:
            grabando = True
            buffer = []
            print(f"  ⏺ Grabando secuencia {len(secuencias)+1}...")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    if secuencias:
        guardar_datos(nombre, secuencias)
        print(f"\n✅ Total: {len(secuencias)} secuencias de '{nombre}'")
    else:
        print("\n⚠️ No se grabaron secuencias")


if __name__ == "__main__":
    main()
