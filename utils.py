import numpy as np

def extract_hand_landmarks(results):
    if results.multi_hand_landmarks:
        hands_landmarks = []
        for hand in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            wrist = landmarks[0]
            landmarks -= wrist
            scale = np.linalg.norm(landmarks[12])
            if scale > 0:
                landmarks /= scale
            hands_landmarks.append(landmarks.flatten())
        
        if len(hands_landmarks) == 2:
            return np.concatenate(hands_landmarks)
        else:
            return np.concatenate([hands_landmarks[0], np.zeros(63)])  # 21 pts * 3 coords = 63
    return None

def extract_face_landmarks(results):
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
        # Normalizar: centrar en landmark 1 (nariz) y escalar por distancia a landmark 9 (por ejemplo)
        center = landmarks[1]
        landmarks -= center
        scale = np.linalg.norm(landmarks[9])
        if scale > 0:
            landmarks /= scale
        return landmarks.flatten()
    return None
