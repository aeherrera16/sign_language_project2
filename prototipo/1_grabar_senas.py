#!/usr/bin/env python3
"""
GRABADOR DE SEÑAS - AUTOMÁTICO
Graba automáticamente cuando detecta mano estable.
NO requiere presionar teclas.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
from datetime import datetime
import time

# === CONFIGURACIÓN ===
FRAMES_SECUENCIA = 30
LANDMARKS_MANO = 21
COORDS = 3
FEATURES = LANDMARKS_MANO * COORDS * 2

DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)


def extraer_landmarks(frame):
    """Extrae landmarks de las manos."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    features = np.zeros(FEATURES)
    num_manos = 0
    
    if result.multi_hand_landmarks:
        num_manos = len(result.multi_hand_landmarks)
        for idx, hand_lm in enumerate(result.multi_hand_landmarks[:2]):
            wrist = hand_lm.landmark[0]
            for i, lm in enumerate(hand_lm.landmark):
                base = idx * 63 + i * 3
                features[base] = lm.x - wrist.x
                features[base + 1] = lm.y - wrist.y
                features[base + 2] = lm.z - wrist.z
    
    return features, result, num_manos


def guardar_datos(nombre_sena, secuencias):
    """Guarda secuencias en JSON."""
    os.makedirs(os.path.join(DIR_DATOS, nombre_sena), exist_ok=True)
    archivo = os.path.join(DIR_DATOS, nombre_sena, f"seq_{datetime.now():%Y%m%d_%H%M%S}.json")
    
    with open(archivo, 'w') as f:
        json.dump({
            "sena": nombre_sena,
            "frames": FRAMES_SECUENCIA,
            "features": FEATURES,
            "secuencias": [s.tolist() for s in secuencias]
        }, f)
    
    print(f"✅ {len(secuencias)} secuencias guardadas en {archivo}")


def main():
    print("\n" + "="*60)
    print("  GRABADOR AUTOMÁTICO DE SEÑAS")
    print("="*60)
    
    nombre = input("\nNombre de la seña: ").strip().upper()
    if not nombre:
        print("❌ Nombre inválido")
        return
    
    try:
        meta = int(input("¿Cuántas secuencias quieres grabar? [30]: ") or "30")
    except:
        meta = 30
    
    print(f"\n🎯 Seña: {nombre}")
    print(f"📊 Meta: {meta} secuencias")
    print("\n" + "="*60)
    print("  INSTRUCCIONES:")
    print("  1. Muestra tu mano haciendo la seña")
    print("  2. Cuando el círculo esté VERDE, la grabación es automática")
    print("  3. Baja la mano entre grabaciones (pausa de 1 seg)")
    print("  4. Presiona Q en la VENTANA para terminar")
    print("="*60)
    
    input("\nPresiona ENTER para comenzar...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    secuencias = []
    buffer = []
    grabando = False
    pausa_hasta = 0
    
    print("\n🎥 Cámara iniciada. Muestra tu mano...")
    
    while len(secuencias) < meta:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        features, result, num_manos = extraer_landmarks(frame)
        
        hay_mano = num_manos > 0
        ahora = time.time()
        en_pausa = ahora < pausa_hasta
        
        # Dibujar manos
        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=4),
                    mp_draw.DrawingSpec(color=(0,200,0), thickness=2)
                )
        
        # === LÓGICA DE GRABACIÓN AUTOMÁTICA ===
        if hay_mano and not en_pausa:
            if not grabando:
                grabando = True
                buffer = []
                print(f"  ⏺ Grabando secuencia {len(secuencias)+1}...")
            
            buffer.append(features)
            progreso = len(buffer) / FRAMES_SECUENCIA
            
            # Barra de progreso
            cv2.rectangle(frame, (20, 430), (620, 460), (50,50,50), -1)
            cv2.rectangle(frame, (20, 430), (int(20 + 600*progreso), 460), (0,255,0), -1)
            cv2.putText(frame, f"GRABANDO: {len(buffer)}/{FRAMES_SECUENCIA}", 
                       (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            if len(buffer) >= FRAMES_SECUENCIA:
                secuencias.append(np.array(buffer))
                buffer = []
                grabando = False
                pausa_hasta = ahora + 1.0  # Pausa de 1 segundo
                print(f"  ✓ Secuencia {len(secuencias)}/{meta} completada")
        
        elif not hay_mano:
            if grabando and len(buffer) < FRAMES_SECUENCIA:
                # Perdió la mano durante grabación - resetear
                buffer = []
                grabando = False
        
        elif en_pausa:
            # Mostrar pausa
            tiempo_restante = pausa_hasta - ahora
            cv2.rectangle(frame, (150, 200), (490, 280), (0,100,200), -1)
            cv2.putText(frame, "PAUSA", (260, 235), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f"Baja la mano... {tiempo_restante:.1f}s", (180, 265),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # === UI ===
        # Header
        cv2.rectangle(frame, (0, 0), (640, 70), (40,40,40), -1)
        cv2.putText(frame, f"Sena: {nombre}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"Progreso: {len(secuencias)}/{meta}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Porcentaje total
        pct = len(secuencias) / meta * 100
        cv2.rectangle(frame, (400, 20), (630, 50), (60,60,60), -1)
        cv2.rectangle(frame, (400, 20), (int(400 + 230*(len(secuencias)/meta)), 50), (0,200,0), -1)
        cv2.putText(frame, f"{pct:.0f}%", (500, 42), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Estado
        if hay_mano and not en_pausa:
            cv2.circle(frame, (620, 90), 20, (0,255,0), -1)
            cv2.putText(frame, "DETECTANDO", (520, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        elif en_pausa:
            cv2.circle(frame, (620, 90), 20, (0,165,255), -1)
        else:
            cv2.circle(frame, (620, 90), 20, (0,0,255), -1)
            cv2.putText(frame, "SIN MANO", (530, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        cv2.imshow('Grabador LSE - Automatico', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    if secuencias:
        guardar_datos(nombre, secuencias)
        print(f"\n✅ COMPLETADO: {len(secuencias)} secuencias de '{nombre}'")
    else:
        print("\n⚠️ No se grabaron secuencias")


if __name__ == "__main__":
    main()
