#!/usr/bin/env python3
"""
=============================================================================
MÓDULO 1: GRABACIÓN DE SEÑAS DINÁMICAS
=============================================================================
Este script graba secuencias de landmarks de manos para entrenar el modelo LSTM.

USO:
    python 1_grabar_senas.py

CONTROLES:
    - Escribe el nombre de la seña cuando se solicite
    - Presiona 'g' para comenzar a grabar una secuencia
    - Presiona 'q' para salir y guardar
    
Cada seña se graba como una secuencia de 30 frames (≈1 segundo).
Se recomienda grabar 30-50 secuencias por palabra.
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
from datetime import datetime

# Configuración
SECUENCIA_FRAMES = 30  # Número de frames por secuencia (≈1 segundo a 30fps)
DATOS_DIR = os.path.join(os.path.dirname(__file__), "datos")

# Señas para noticias (vocabulario inicial)
VOCABULARIO_NOTICIAS = [
    "PRESIDENTE",
    "GOBIERNO", 
    "PAIS",
    "ECUADOR",
    "DECIR",
    "ANUNCIAR",
    "AÑO",
    "DINERO",
    "POBREZA",
    "TRABAJO",
    "SUBIR",
    "BAJAR",
    "BUENO",
    "MALO",
    "HOY",
]

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def extraer_landmarks(frame):
    """Extrae los landmarks de las manos del frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Vector de 126 valores (2 manos × 21 puntos × 3 coordenadas)
    landmarks = np.zeros(126)
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= 2:  # Máximo 2 manos
                break
            for i, lm in enumerate(hand_landmarks.landmark):
                base = idx * 63 + i * 3
                landmarks[base] = lm.x
                landmarks[base + 1] = lm.y
                landmarks[base + 2] = lm.z
    
    return landmarks, results

def dibujar_manos(frame, results):
    """Dibuja los landmarks de las manos en el frame."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
    return frame

def guardar_secuencias(nombre_sena, secuencias):
    """Guarda las secuencias en un archivo JSON."""
    sena_dir = os.path.join(DATOS_DIR, nombre_sena)
    os.makedirs(sena_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = os.path.join(sena_dir, f"secuencias_{timestamp}.json")
    
    datos = {
        "sena": nombre_sena,
        "frames_por_secuencia": SECUENCIA_FRAMES,
        "num_secuencias": len(secuencias),
        "secuencias": [seq.tolist() for seq in secuencias]
    }
    
    with open(archivo, 'w') as f:
        json.dump(datos, f)
    
    print(f"✅ Guardadas {len(secuencias)} secuencias en {archivo}")
    return archivo

def main():
    print("=" * 60)
    print("   GRABADOR DE SEÑAS DINÁMICAS PARA LSE")
    print("=" * 60)
    print("\nVocabulario disponible para noticias:")
    for i, sena in enumerate(VOCABULARIO_NOTICIAS, 1):
        print(f"  {i:2}. {sena}")
    
    print("\n" + "-" * 60)
    nombre_sena = input("Escribe el nombre de la seña a grabar: ").strip().upper()
    
    if not nombre_sena:
        print("❌ Nombre inválido")
        return
    
    print(f"\n🎯 Grabando seña: {nombre_sena}")
    print("=" * 60)
    print("\nCONTROLES:")
    print("  [G] - Iniciar grabación de secuencia (mantén la seña)")
    print("  [Q] - Guardar y salir")
    print("-" * 60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: No se puede abrir la cámara")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    secuencias_grabadas = []
    secuencia_actual = []
    grabando = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Espejo
        landmarks, results = extraer_landmarks(frame)
        frame = dibujar_manos(frame, results)
        
        # Estado de grabación
        if grabando:
            secuencia_actual.append(landmarks)
            progreso = len(secuencia_actual)
            
            # Barra de progreso
            barra_ancho = int((progreso / SECUENCIA_FRAMES) * 300)
            cv2.rectangle(frame, (170, 430), (170 + barra_ancho, 460), (0, 255, 0), -1)
            cv2.rectangle(frame, (170, 430), (470, 460), (255, 255, 255), 2)
            cv2.putText(frame, f"Grabando: {progreso}/{SECUENCIA_FRAMES}", 
                       (180, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Si completó la secuencia
            if progreso >= SECUENCIA_FRAMES:
                secuencias_grabadas.append(np.array(secuencia_actual))
                secuencia_actual = []
                grabando = False
                print(f"  ✓ Secuencia {len(secuencias_grabadas)} grabada")
        
        # Información en pantalla
        cv2.rectangle(frame, (0, 0), (640, 80), (50, 50, 50), -1)
        cv2.putText(frame, f"Sena: {nombre_sena}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Secuencias grabadas: {len(secuencias_grabadas)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detección de mano
        mano_detectada = results.multi_hand_landmarks is not None
        color_estado = (0, 255, 0) if mano_detectada else (0, 0, 255)
        estado = "MANO DETECTADA" if mano_detectada else "NO HAY MANO"
        cv2.putText(frame, estado, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_estado, 2)
        
        # Instrucciones
        if not grabando:
            cv2.putText(frame, "Presiona [G] para grabar | [Q] para salir", 
                       (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('Grabador de Senas LSE', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g') and not grabando and mano_detectada:
            grabando = True
            secuencia_actual = []
            print(f"  ⏺ Iniciando grabación de secuencia {len(secuencias_grabadas) + 1}...")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Guardar secuencias
    if secuencias_grabadas:
        guardar_secuencias(nombre_sena, secuencias_grabadas)
        print(f"\n✅ Total: {len(secuencias_grabadas)} secuencias de '{nombre_sena}'")
    else:
        print("\n⚠️ No se grabaron secuencias")

if __name__ == "__main__":
    main()
