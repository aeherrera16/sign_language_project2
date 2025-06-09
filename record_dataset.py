import cv2
import os
import sys
import mediapipe as mp
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
            return np.concatenate(hands_landmarks)  # 2 manos -> 2*63 = 126 elementos
        else:
            return np.concatenate([hands_landmarks[0], np.zeros(63)])  # 1 mano + ceros para segunda mano
    return None

def extract_face_landmarks(results):
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
        center = landmarks[1]  # centrar en punto 1 (nariz)
        landmarks -= center
        scale = np.linalg.norm(landmarks[9])
        if scale > 0:
            landmarks /= scale
        return landmarks.flatten()  # 468*3=1404 elementos
    return None

if len(sys.argv) < 2:
    print("Debes proporcionar el nombre del gesto como argumento.")
    sys.exit(1)

gesture_name = sys.argv[1]
save_dir = f"data/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(rgb)
    results_face = face_mesh.process(rgb)

    landmarks_hand = extract_hand_landmarks(results_hands)
    landmarks_face = extract_face_landmarks(results_face)

    if landmarks_hand is not None or landmarks_face is not None:
        if landmarks_hand is None:
            data_to_save = landmarks_face
        elif landmarks_face is None:
            data_to_save = landmarks_hand
        else:
            # Asegurar que ambos son vectores planos
            landmarks_hand = landmarks_hand.flatten()
            landmarks_face = landmarks_face.flatten()
            data_to_save = np.hstack((landmarks_hand, landmarks_face))  # concatena en 1D

        filename = os.path.join(save_dir, f"{gesture_name}_{count}.npy")
        print(f"Guardando {filename} con forma {data_to_save.shape}")
        np.save(filename, data_to_save)
        count += 1

        # Dibujar landmarks
        if results_hands.multi_hand_landmarks:
            for handLms in results_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
        if results_face.multi_face_landmarks:
            for faceLms in results_face.multi_face_landmarks:
                mp_draw.draw_landmarks(image, faceLms, mp_face_mesh.FACEMESH_TESSELATION)

    cv2.putText(image, f"{gesture_name}: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Grabando gesto", image)

    if cv2.waitKey(1) & 0xFF == 27 or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()
