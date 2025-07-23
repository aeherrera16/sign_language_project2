# -*- coding: utf-8 -*-
import numpy as np

def extract_hand_landmarks(results):
    """
    Extrae y normaliza los puntos de referencia (landmarks) de las manos desde el resultado de MediaPipe.
    - Centra los puntos en la muneca (landmark 0).
    - Escala usando la distancia al landmark 12 (dedo medio).
    - Devuelve un vector de 126 elementos si hay 2 manos, o 63 + ceros si solo hay una.
    """
    if results is not None and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        hands_landmarks = []

        for hand in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])  # 21 puntos
            wrist = landmarks[0]
            landmarks -= wrist  # Centrado
            scale = np.linalg.norm(landmarks[12])  # Escala usando dedo medio
            if scale > 0:
                landmarks /= scale
            hands_landmarks.append(landmarks.flatten())  # 63 elementos por mano

        if len(hands_landmarks) == 2:
            return np.concatenate(hands_landmarks)  # 126 elementos (2 manos)
        else:
            return np.concatenate([hands_landmarks[0], np.zeros(63)])  # 1 mano + relleno

    return None

def extract_face_landmarks(results):
    """
    Extrae y normaliza los puntos de referencia faciales desde el resultado de MediaPipe.
    - Centra los puntos en el landmark 1 (nariz).
    - Escala usando la distancia al landmark 9 (mejilla aproximadamente).
    - Devuelve un vector de 1404 elementos (468 puntos x 3 coords).
    """
    if results is not None and hasattr(results, 'multi_face_landmarks') and results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])  # 468 puntos
        center = landmarks[1]
        landmarks -= center  # Centrado
        scale = np.linalg.norm(landmarks[9])  # Escala (distancia nariz -> mejilla)
        if scale > 0:
            landmarks /= scale
        return landmarks.flatten()  # 1404 elementos

    return None

def extraer_landmarks(results_hands, results_face):
    """
    Extrae landmarks combinados de manos y cara para el sistema de reconocimiento.
    
    Args:
        results_hands: Resultados de deteccion de manos de MediaPipe
        results_face: Resultados de deteccion facial de MediaPipe
    
    Returns:
        numpy array con landmarks combinados o None si no se detectan features
    """
    # Extraer landmarks de manos
    hand_landmarks = extract_hand_landmarks(results_hands)
    
    # Extraer landmarks de cara
    face_landmarks = extract_face_landmarks(results_face)
    
    # Verificar que tenemos al menos datos de manos
    if hand_landmarks is not None:
        if face_landmarks is not None:
            # Concatenar landmarks de manos y cara
            combined_landmarks = np.concatenate([hand_landmarks, face_landmarks])
        else:
            # Solo manos, rellenar con ceros para la cara
            combined_landmarks = np.concatenate([hand_landmarks, np.zeros(1404)])
        
        return combined_landmarks
    
    return None
