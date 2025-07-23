# -*- coding: utf-8 -*-
import os
# Configurar variables de entorno ANTES de cualquier import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import cv2
import sys
import mediapipe as mp
import numpy as np

# ========================
# Funciones para extraer landmarks SOLO DE MANOS
# ========================
def extract_hand_landmarks(results):
    """Extrae landmarks solo de las manos (126 dimensiones)"""
    if results.multi_hand_landmarks:
        hands_landmarks = []
        for hand in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            # Normalizar relativo a la muñeca
            wrist = landmarks[0]
            landmarks -= wrist
            # Escalar relativo al tamaño de la mano
            scale = np.linalg.norm(landmarks[12])
            if scale > 0:
                landmarks /= scale
            hands_landmarks.append(landmarks.flatten())
        
        if len(hands_landmarks) == 2:
            return np.concatenate(hands_landmarks)  # 2 manos: 2 * 63 = 126
        else:
            # Una mano detectada, completar con ceros para la segunda mano
            return np.concatenate([hands_landmarks[0], np.zeros(63)])  # 1 mano + ceros = 126
    
    # No se detectaron manos
    return None

# ========================
# Argumento: nombre del gesto
# ========================
if len(sys.argv) < 2:
    print("❗ Debes proporcionar el nombre del gesto como argumento.")
    sys.exit(1)

gesture_name = sys.argv[1]
save_dir = f"data/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)

# ========================
# Inicializar MediaPipe (SOLO MANOS)
# ========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========================
# Inicializar camara
# ========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la camara.")
    sys.exit(1)

print(f"🎥 Grabando gesto: {gesture_name}")
print("📋 Instrucciones:")
print("   • Realiza el gesto claramente frente a la cámara")
print("   • Mantén las manos dentro del cuadro")
print("   • Presiona ESC para salir antes de completar")
print("   • Se grabarán automáticamente 100 muestras")
print("")

count = 0

# ========================
# Loop de captura (SOLO MANOS)
# ========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al capturar el frame.")
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar solo las manos
    results_hands = hands.process(rgb)
    landmarks_hand = extract_hand_landmarks(results_hands)

    # Solo guardar si se detectan manos
    if landmarks_hand is not None:
        filename = os.path.join(save_dir, f"{gesture_name}_{count}.npy")
        print(f"💾 Guardando muestra {count + 1}/100: {filename} (shape: {landmarks_hand.shape})")
        np.save(filename, landmarks_hand)
        count += 1

        # Dibujar landmarks de las manos
        if results_hands.multi_hand_landmarks:
            for handLms in results_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

    # Mostrar información en pantalla
    cv2.putText(image, f"Gesto: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Muestras: {count}/100", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    if landmarks_hand is not None:
        cv2.putText(image, "✅ Manos detectadas", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "❌ No se detectan manos", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.putText(image, "Presiona ESC para salir", (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("LSE Ecuador - Grabando Gesto", image)

    # Salida por ESC o si se completan 100 muestras
    if cv2.waitKey(1) & 0xFF == 27 or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Grabación completada: {count} muestras guardadas")
print(f"📁 Ubicación: {save_dir}")
print("🎯 Datos listos para entrenamiento")