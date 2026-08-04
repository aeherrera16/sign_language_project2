#!/usr/bin/env python3
"""
BENCHMARK DE FPS — MediaPipe Holistic
Corre esto en la computadora Y en la Raspberry Pi antes de regrabar todo
el dataset, para confirmar que el FPS real alcanza para señas dinámicas.

Uso:
    python3 bench_holistic_fps.py --segundos 15 --complexity 0
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import argparse
import time
import cv2

from utils_silenciar import init_mediapipe_holistic


def main(segundos, complexity, detection_conf, tracking_conf, fuente):
    print(f"\n🔬 Benchmark Holistic — fuente={fuente}, model_complexity={complexity}, {segundos}s")
    mp, mp_holistic, mp_draw, holistic = init_mediapipe_holistic(
        detection_conf=detection_conf, tracking_conf=tracking_conf, model_complexity=complexity
    )

    # fuente puede ser un índice de cámara USB (ej. "0") o una URL (ej. RTSP de la cámara WiFi)
    cap = cv2.VideoCapture(int(fuente) if fuente.isdigit() else fuente)
    if fuente.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara/fuente de video.")

    # Warm-up
    for _ in range(5):
        cap.read()

    frames = 0
    t_inicio = time.time()
    t_fin = t_inicio + segundos

    while time.time() < t_fin:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic.process(rgb)
        frames += 1

    elapsed = time.time() - t_inicio
    fps = frames / elapsed

    cap.release()
    holistic.close()

    print(f"\n📊 Resultado: {frames} frames en {elapsed:.1f}s → {fps:.1f} FPS")
    if fps < 10:
        print("   ⚠️  FPS bajo — puede sentirse lento/entrecortado para señas dinámicas.")
        print("   💡 Si esto corre en Raspberry Pi, considera bajar la resolución de captura")
        print("      o aumentar LSE_INTERVALO_PREDICCION para predecir con menos frecuencia.")
    elif fps < 18:
        print("   ⚠️  FPS moderado — debería funcionar pero con cierto retraso perceptible.")
    else:
        print("   ✅ FPS bueno para reconocimiento en tiempo real.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de FPS para MediaPipe Holistic")
    parser.add_argument("--segundos", type=int, default=15, help="Duración del benchmark")
    parser.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2],
                        help="model_complexity de Holistic (0=lite, 1=full, 2=heavy)")
    parser.add_argument("--detection-conf", type=float, default=0.5)
    parser.add_argument("--tracking-conf", type=float, default=0.5)
    parser.add_argument("--fuente", type=str, default="0",
                        help="Índice de cámara USB (ej. 0) o URL de video (ej. rtsp://192.168.100.1:8080/?action=stream)")
    args = parser.parse_args()
    main(args.segundos, args.complexity, args.detection_conf, args.tracking_conf, args.fuente)
