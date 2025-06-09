import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
import time
import threading
from utils import extract_hand_landmarks, extract_face_landmarks

# Inicializar voz
engine = pyttsx3.init()
engine.setProperty('rate', 100)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Cargar modelo y etiquetas
model = tf.keras.models.load_model("model/gesture_model.h5")
with open("model/labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.8, min_tracking_confidence=0.8)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                             min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

# Inicializar cámara
cap = cv2.VideoCapture(0)

last_spoken_time = 0
spoken_gesture = None
current_gesture = None
gesture_counter = 0
required_frames = 5
min_time_between_phrases = 2.0  # segundos para no hablar seguido

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar con Mediapipe
    hands_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    hand_landmarks = extract_hand_landmarks(hands_results)
    face_landmarks = extract_face_landmarks(face_results)

    # Combinar entrada
    if hand_landmarks is not None and face_landmarks is not None:
        input_vector = np.concatenate([hand_landmarks, face_landmarks])
    elif hand_landmarks is not None:
        input_vector = np.concatenate([hand_landmarks, np.zeros(1404)])
    elif face_landmarks is not None:
        input_vector = np.concatenate([np.zeros(126), face_landmarks])
    else:
        input_vector = None

    if input_vector is not None and input_vector.shape[0] == 1530:
        input_data = np.expand_dims(input_vector, axis=0)
        prediction = model.predict(input_data, verbose=0)[0]

        gesture_index = np.argmax(prediction)
        gesture = labels[gesture_index]
        confidence = np.max(prediction)

        cv2.putText(image, f"{gesture} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if confidence >= 0.9:
            if gesture == current_gesture:
                gesture_counter += 1
            else:
                current_gesture = gesture
                gesture_counter = 1

            # Verificar si se debe hablar
            if gesture_counter >= required_frames:
                current_time = time.time()
                if gesture != spoken_gesture and (current_time - last_spoken_time > min_time_between_phrases):
                    threading.Thread(target=speak, args=(gesture,), daemon=True).start()
                    spoken_gesture = gesture
                    last_spoken_time = current_time
                    gesture_counter = 0

    # Dibujar landmarks si están presentes
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_TESSELATION)

    cv2.imshow("Reconocimiento en tiempo real", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
