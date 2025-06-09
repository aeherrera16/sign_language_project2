import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from utils import extract_hand_landmarks

model = tf.keras.models.load_model("model/gesture_model.h5")
with open("model/labels.pkl", "rb") as f:
    labels = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    hand_landmarks = extract_hand_landmarks(results)
    face_landmarks = extract_face_landmarks(results)

    if hand_landmarks is not None and face_landmarks is not None:
        input_vector = np.hstack((hand_landmarks, face_landmarks))
    elif hand_landmarks is not None:
        input_vector = np.hstack((hand_landmarks, np.zeros(1404)))  # relleno rostro con ceros
    elif face_landmarks is not None:
        input_vector = np.hstack((np.zeros(126), face_landmarks))   # relleno manos con ceros
    else:
        input_vector = None

    if input_vector is not None:
        prediction = model.predict(np.expand_dims(input_vector, axis=0))[0]

    cv2.imshow("Reconocimiento en tiempo real", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
