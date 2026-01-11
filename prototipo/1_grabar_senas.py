#!/usr/bin/env python3
"""
GRABADOR DE SEÑAS - Prototipo LSE
Graba secuencias de landmarks para entrenar el modelo.

Controles:
    G - Grabar secuencia (1 segundo)
    Q - Guardar y salir
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
from datetime import datetime

# Configuración
FRAMES_POR_SECUENCIA = 30
DATOS_DIR = os.path.join(os.path.dirname(__file__), "datos")

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)


def extraer_landmarks(frame):
    """Extrae 126 valores (2 manos × 21 puntos × 3 coords)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    landmarks = np.zeros(126)
    
    if results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks[:2]):
            for i, lm in enumerate(hand.landmark):
                base = idx * 63 + i * 3
                landmarks[base:base+3] = [lm.x, lm.y, lm.z]
    
    return landmarks, results


def guardar(nombre, secuencias):
    """Guarda secuencias en JSON."""
    os.makedirs(os.path.join(DATOS_DIR, nombre), exist_ok=True)
    archivo = os.path.join(DATOS_DIR, nombre, f"{datetime.now():%Y%m%d_%H%M%S}.json")
    
    with open(archivo, 'w') as f:
        json.dump({
            "sena": nombre,
            "secuencias": [s.tolist() for s in secuencias]
        }, f)
    
    print(f"✅ Guardadas {len(secuencias)} secuencias")


def main():
    print("\n" + "="*50)
    print("  GRABADOR DE SEÑAS LSE")
    print("="*50)
    
    nombre = input("\nNombre de la seña: ").strip().upper()
    if not nombre:
        return
    
    print(f"\nGrabando: {nombre}")
    print("[G] Grabar | [Q] Salir\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    secuencias = []
    buffer = []
    grabando = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        landmarks, results = extraer_landmarks(frame)
        
        # Dibujar manos
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        
        # Grabación
        if grabando:
            buffer.append(landmarks)
            prog = len(buffer) / FRAMES_POR_SECUENCIA
            cv2.rectangle(frame, (20, 440), (int(20 + 600*prog), 470), (0,255,0), -1)
            
            if len(buffer) >= FRAMES_POR_SECUENCIA:
                secuencias.append(np.array(buffer))
                buffer = []
                grabando = False
                print(f"  ✓ Secuencia {len(secuencias)}")
        
        # UI
        cv2.rectangle(frame, (0, 0), (640, 50), (40,40,40), -1)
        cv2.putText(frame, f"{nombre} - {len(secuencias)} secuencias", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        hay_mano = results.multi_hand_landmarks is not None
        color = (0,255,0) if hay_mano else (0,0,255)
        cv2.circle(frame, (620, 25), 15, color, -1)
        
        cv2.imshow('Grabador LSE', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g') and hay_mano and not grabando:
            grabando = True
            buffer = []
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if secuencias:
        guardar(nombre, secuencias)


if __name__ == "__main__":
    main()
