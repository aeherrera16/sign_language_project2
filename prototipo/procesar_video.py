#!/usr/bin/env python3
"""
PROCESADOR DE VIDEO - Extrae una secuencia de entrenamiento desde un video
============================================================================
Para aprovechar un diccionario de señas ya grabado en video (un clip por
seña) en vez de grabar todo de nuevo con la webcam.

Usa la MISMA extracción de landmarks (manos + pose + rostro, coordenadas
relativas) que 1_grabar_senas.py, así que las secuencias resultantes son
100% compatibles con el modelo entrenado. La diferencia es que NO aplica
el filtro de "zona activa"/"mano en reposo" (pensado para descartar ruido
en captura en vivo) — se asume que el video ya está recortado a solo la
seña, así que se usan todos los frames donde MediaPipe detecta una mano.

Soporta señas dinámicas: si el video dura más o menos que 2 segundos,
la secuencia se remuestrea a los 30 frames que espera el modelo (igual
que 1_grabar_senas.py con normalizar_frames).

Uso:
    python3 procesar_video.py --video /ruta/al/video.mp4 --nombre TODOS
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import argparse
import cv2
import numpy as np
import json
from datetime import datetime

from utils_silenciar import init_mediapipe_holistic
mp, mp_holistic, mp_draw, holistic = init_mediapipe_holistic(
    detection_conf=0.4, tracking_conf=0.3, model_complexity=0
)

FRAMES_SECUENCIA = 30
LANDMARKS_MANO = 21
COORDS = 3
FEATURES_MANOS = LANDMARKS_MANO * COORDS * 2          # 126
FEATURES_POSE = 5 * COORDS                            # 15
FEATURES_ROSTRO = 6 * COORDS                          # 18
FEATURES = FEATURES_MANOS + FEATURES_POSE + FEATURES_ROSTRO  # 159

DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")

POSE_INDICES = [
    mp_holistic.PoseLandmark.NOSE.value,
    mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
    mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
    mp_holistic.PoseLandmark.LEFT_ELBOW.value,
    mp_holistic.PoseLandmark.RIGHT_ELBOW.value,
]
POSE_LEFT_SHOULDER_IDX = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
POSE_RIGHT_SHOULDER_IDX = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value

# Cejas + boca (referencia relativa a la nariz) — mismos índices que
# 1_grabar_senas.py/3_traductor.py.
FACE_NOSE_TIP_IDX = 1
FACE_INDICES = [65, 295, 105, 334, 61, 291]


def _features_manos(left_lm, right_lm):
    features = np.zeros(FEATURES_MANOS)
    if left_lm is not None:
        wrist = left_lm.landmark[0]
        for i, lm in enumerate(left_lm.landmark):
            base = i * 3
            features[base] = lm.x - wrist.x
            features[base + 1] = lm.y - wrist.y
            features[base + 2] = lm.z - wrist.z
    if right_lm is not None:
        wrist = right_lm.landmark[0]
        for i, lm in enumerate(right_lm.landmark):
            base = 63 + i * 3
            features[base] = lm.x - wrist.x
            features[base + 1] = lm.y - wrist.y
            features[base + 2] = lm.z - wrist.z
    return features


def _features_pose(result):
    features = np.zeros(FEATURES_POSE)
    pose_lm = result.pose_landmarks
    if pose_lm is None:
        return features
    lm_list = pose_lm.landmark
    ls = lm_list[POSE_LEFT_SHOULDER_IDX]
    rs = lm_list[POSE_RIGHT_SHOULDER_IDX]
    cx, cy, cz = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2, (ls.z + rs.z) / 2
    for i, idx in enumerate(POSE_INDICES):
        lm = lm_list[idx]
        base = i * 3
        features[base] = lm.x - cx
        features[base + 1] = lm.y - cy
        features[base + 2] = lm.z - cz
    return features


def _features_rostro(result):
    features = np.zeros(FEATURES_ROSTRO)
    face_lm = result.face_landmarks
    if face_lm is None:
        return features
    lm_list = face_lm.landmark
    nose = lm_list[FACE_NOSE_TIP_IDX]
    for i, idx in enumerate(FACE_INDICES):
        lm = lm_list[idx]
        base = i * 3
        features[base] = lm.x - nose.x
        features[base + 1] = lm.y - nose.y
        features[base + 2] = lm.z - nose.z
    return features


def normalizar_frames(frames_lista, objetivo):
    n = len(frames_lista)
    if n == objetivo:
        return frames_lista
    idx = np.clip(np.round(np.linspace(0, n - 1, objetivo)).astype(int), 0, n - 1)
    return [frames_lista[i] for i in idx]


def guardar_datos(nombre_sena, secuencias):
    os.makedirs(os.path.join(DIR_DATOS, nombre_sena), exist_ok=True)
    archivo = os.path.join(DIR_DATOS, nombre_sena, f"seq_video_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(archivo, 'w') as f:
        json.dump({
            "sena": nombre_sena,
            "frames": FRAMES_SECUENCIA,
            "features": FEATURES,
            "secuencias": [s.tolist() for s in secuencias]
        }, f)
    print(f"✅ {len(secuencias)} secuencia(s) guardada(s) en {archivo}")


def procesar_video(ruta_video, nombre_sena):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"📹 Video: {ruta_video}")
    print(f"   {total_frames} frames a {fps:.1f} fps (~{total_frames/fps:.1f}s)")

    frames_capturados = []
    frames_con_mano = 0
    idx_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx_frame += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        left_lm = result.left_hand_landmarks
        right_lm = result.right_hand_landmarks

        # Se descartan solo los frames sin ninguna mano detectada (intro/outro
        # del video sin señar); no se aplica el filtro de zona activa/reposo
        # porque el video ya viene recortado a la seña.
        if left_lm is None and right_lm is None:
            continue

        frames_con_mano += 1
        features = np.concatenate([
            _features_manos(left_lm, right_lm),
            _features_pose(result),
            _features_rostro(result),
        ])
        frames_capturados.append(features)

    cap.release()

    print(f"   {frames_con_mano}/{idx_frame} frames con mano detectada")

    if len(frames_capturados) < 5:
        raise RuntimeError(
            f"Muy pocos frames con mano detectada ({len(frames_capturados)}). "
            "Verifica que el video muestre la seña con claridad."
        )

    normalizados = normalizar_frames(frames_capturados, FRAMES_SECUENCIA)
    secuencia = np.array(normalizados, dtype=np.float32)
    guardar_datos(nombre_sena, [secuencia])


def main():
    parser = argparse.ArgumentParser(description="Extrae una secuencia de entrenamiento desde un video")
    parser.add_argument('--video', required=True, help='Ruta al archivo de video')
    parser.add_argument('--nombre', required=True, help='Nombre de la seña (mayúsculas)')
    args = parser.parse_args()
    procesar_video(args.video, args.nombre.upper())


if __name__ == "__main__":
    main()
