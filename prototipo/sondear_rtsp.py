#!/usr/bin/env python3
"""
Sondeo RTSP en la cámara DV01 (puerto 8080 confirmado como servidor RTSP
real). Primero manda OPTIONS para ver el banner/Server del software (puede
identificar la librería RTSP exacta usada), luego prueba rutas adicionales
específicas de este modelo/marca.
"""

import socket

IP = "192.168.100.1"
PUERTO = 8080

RUTAS_EXTRA = [
    "/3292", "/AX3292", "/DV01", "/dv01",
    "/01", "/02", "/cam1", "/cam0",
    "/trackID=0", "/track1",
    "/wifi_camera", "/wind",
    "/h264.sdp", "/video.sdp", "/stream.sdp",
    "/playlist.m3u8",  # por si acaso es HLS disfrazado
]


def enviar(metodo, ruta, extra_headers=""):
    url = f"rtsp://{IP}:{PUERTO}{ruta}"
    req = (
        f"{metodo} {url if ruta != '*' else '*'} RTSP/1.0\r\n"
        f"CSeq: 1\r\n"
        f"User-Agent: sondeo-python\r\n"
        f"{extra_headers}"
        f"\r\n"
    )
    try:
        s = socket.create_connection((IP, PUERTO), timeout=2)
        s.sendall(req.encode())
        s.settimeout(2)
        resp = s.recv(4096).decode(errors="replace")
        s.close()
        return resp
    except Exception as e:
        return f"(error: {e})"


if __name__ == "__main__":
    print("=== OPTIONS a la raíz (busca el banner del servidor) ===")
    resp = enviar("OPTIONS", "/")
    print(resp)
    print()

    print("=== OPTIONS a '*' (algunos servidores solo responden así) ===")
    resp2 = enviar("OPTIONS", "*")
    print(resp2)
    print()

    print("=== Probando rutas adicionales específicas de esta marca ===")
    for ruta in RUTAS_EXTRA:
        resp = enviar("DESCRIBE", ruta, "Accept: application/sdp\r\n")
        primera = resp.splitlines()[0] if resp and not resp.startswith("(error") else resp
        marca = "✅✅✅ ENCONTRADO" if "200" in primera else ("⚠️ revisar" if "404" not in primera else "❌")
        print(f"{marca}  {ruta:25s} -> {primera}")
        if marca != "❌":
            print("    " + resp.replace("\r\n", "\n    "))
