import numpy as np
import mediapipe as mp

def extract_hand_landmarks(results):
    """Extrae landmarks de las manos"""
    if results.multi_hand_landmarks:
        hand_landmarks = []
        for hand_lms in results.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                hand_landmarks.extend([lm.x, lm.y, lm.z])
        
        # Asegurar que siempre tengamos 126 features (2 manos * 21 puntos * 3 coordenadas)
        while len(hand_landmarks) < 126:
            hand_landmarks.extend([0.0, 0.0, 0.0])
        
        return np.array(hand_landmarks[:126])
    else:
        return np.zeros(126)

def extract_face_landmarks(results):
    """Extrae landmarks de la cara (placeholder)"""
    return np.zeros(468 * 3)  # MediaPipe face mesh tiene 468 puntos
