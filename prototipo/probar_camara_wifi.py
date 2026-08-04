#!/usr/bin/env python3
"""
PROBAR CÁMARA WIFI (DV01 / Viidure, chip Anyka AX3292)

Conecta tu computadora a la red WiFi que crea la cámara (la misma que usa
la app Viidure para emparejar) ANTES de correr este script.

Prueba combinaciones conocidas de IP/ruta RTSP para cámaras de acción
chinas con chips similares (Anyka/Novatek/Hisilicon/SigmaStar), ya que no
hay una URL confirmada para este modelo exacto — esto es exploratorio.
"""

import cv2
import time

# IP real confirmada de la cámara (router de su red WiFi propia): 192.168.100.1
# Puertos confirmados abiertos en distintos escaneos: 8080 y 8081 (RTSP 554 cerrado)
IP = "192.168.100.1"
CANDIDATOS = [
    f"http://{IP}:8080/",
    f"http://{IP}:8081/",
    f"http://{IP}:8080/?action=stream",
    f"http://{IP}:8081/?action=stream",
    f"http://{IP}:8080/videostream.cgi",
    f"http://{IP}:8081/videostream.cgi",
    f"http://{IP}:8080/11",
    f"http://{IP}:8081/11",
    f"http://{IP}:8080/livestream",
    f"http://{IP}:8081/livestream",
    f"rtsp://{IP}:8080/live",
    f"rtsp://{IP}:8081/live",
]


def probar(url, timeout_seg=5):
    print(f"\nProbando: {url}")
    cap = cv2.VideoCapture(url)
    inicio = time.time()
    while time.time() - inicio < timeout_seg:
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  ✅ FUNCIONA — frame recibido, tamaño {frame.shape}")
            cap.release()
            return True
        time.sleep(0.2)
    print("  ❌ No respondió")
    cap.release()
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("  Asegúrate de estar conectado a la WiFi de la cámara")
    print("  (la misma que usa la app Viidure)")
    print("=" * 60)
    encontrado = False
    for url in CANDIDATOS:
        if probar(url):
            encontrado = True
            print(f"\n🎯 Esta funcionó: {url}")
            break
    if not encontrado:
        print("\n⚠️  Ninguna combinación conocida funcionó.")
        print("   Puede que esta cámara no exponga RTSP/HTTP directo, o use otra ruta.")
        print("   Alternativa: revisa la configuración de red de tu Mac (System Settings")
        print("   > Wi-Fi > detalles de la red de la cámara > Router/IP) para confirmar")
        print("   la IP real de la cámara, y prueba esa IP con las mismas rutas.")
