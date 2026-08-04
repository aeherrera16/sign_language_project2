#!/usr/bin/env python3
"""
GRABADOR DE SEÑAS - AUTOMÁTICO CON FILTRADO
Graba automáticamente cuando detecta mano estable.
Incluye filtrado de zona activa y posición natural.
NO requiere presionar teclas.

Captura holística (MediaPipe Holistic): manos + pose superior + rostro,
para incluir marcadores no manuales (cejas, boca, postura) relevantes en LSE.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import json
from datetime import datetime
import time

# Importar MediaPipe SIN warnings (usa redirección de stderr a nivel OS)
from utils_silenciar import init_mediapipe_holistic
# detection_conf=0.4: debe coincidir con 3_traductor.py para que el modelo
# entrene con la misma distribución de detecciones que va a ver en inferencia
# (más permisivo porque la cámara apunta a la persona de enfrente y no se le
# puede pedir que ajuste cómo señaliza para evitar manos de canto/perfil).
mp, mp_holistic, mp_draw, holistic = init_mediapipe_holistic(
    detection_conf=0.4, tracking_conf=0.3, model_complexity=0
)

# === CONFIGURACIÓN ===
FRAMES_SECUENCIA = 30
LANDMARKS_MANO = 21
COORDS = 3
FEATURES_MANOS = LANDMARKS_MANO * COORDS * 2          # 126
FEATURES_POSE = 5 * COORDS                            # 15
FEATURES_ROSTRO = 6 * COORDS                           # 18
FEATURES = FEATURES_MANOS + FEATURES_POSE + FEATURES_ROSTRO  # 159

# Índices del modelo BlazePose (mp_holistic.PoseLandmark)
POSE_INDICES = [
    mp_holistic.PoseLandmark.NOSE.value,
    mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
    mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
    mp_holistic.PoseLandmark.LEFT_ELBOW.value,
    mp_holistic.PoseLandmark.RIGHT_ELBOW.value,
]
POSE_LEFT_SHOULDER_IDX = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
POSE_RIGHT_SHOULDER_IDX = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value

# Índices del face mesh canónico de MediaPipe (468 puntos)
FACE_NOSE_TIP_IDX = 1
FACE_INDICES = [
    105,  # ceja izquierda
    334,  # ceja derecha
    61,   # comisura boca izquierda
    291,  # comisura boca derecha
    13,   # labio superior (centro)
    14,   # labio inferior (centro)
]

DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")

# === ZONA ACTIVA DE SEÑAS (ROI) ===
ROI_X_MIN = 0.10
ROI_X_MAX = 0.90
ROI_Y_MIN = 0.05
ROI_Y_MAX = 0.75

# === FILTRO DE POSICIÓN NATURAL ===
UMBRAL_MANO_CAIDA_Y = 0.70
# Bajado de 0.015 — señas con dedos superpuestos/juntos (p.ej. la letra P
# del alfabeto dactilológico) tienen spread naturalmente bajo y se estaban
# confundiendo con una mano en reposo real.
UMBRAL_MOVIMIENTO_DEDOS = 0.008

# Configuración de cámara
MODO_WEARABLE = True # True si la cámara está en el pecho (no voltea la imagen)

# Fuente de cámara: índice USB (ej. "0") o URL de video (ej. RTSP de una
# cámara WiFi: "rtsp://192.168.100.1:8080/?action=stream"). Configurable
# por variable de entorno para no tocar código al cambiar de cámara.
CAMARA_FUENTE = os.environ.get("LSE_CAMARA_FUENTE", "0")

# === GRABACIÓN POR TIEMPO FIJO (en vez de auto-disparo al detectar mano) ===
# El auto-disparo cortaba la seña a mitad de gesto o empezaba tarde — mismo
# problema reportado en Fernández (2026, ESPOCH), que lo resolvió con
# sesiones cronometradas. Aquí se combina con nuestros filtros de zona
# activa/reposo como control de calidad DESPUÉS de capturar, no como gatillo.
DURACION_MUESTRA = 2.0          # segundos de grabación real por secuencia
PAUSA_PREPARACION = 2.0         # cuenta atrás antes de empezar a grabar
PAUSA_ENTRE_MUESTRAS = 1.0      # descanso después de cada secuencia
MIN_FRAMES_CAPTURADOS = 10      # si se capturan menos, se descarta (cámara lenta/glitch)
MIN_PROPORCION_MANO_VALIDA = 0.5  # al menos 50% del tiempo debe haber mano válida

# =============================================================================
# FUNCIONES DE FILTRADO (compartidas con el traductor)
# =============================================================================

def calcular_centro_mano(hand_landmarks):
    """Calcula el centro de todos los landmarks de una mano."""
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return np.mean(xs), np.mean(ys)


def calcular_spread_dedos(hand_landmarks):
    """Calcula la extensión de los dedos. Manos relajadas = bajo spread."""
    puntas = [4, 8, 12, 16, 20]
    muñeca = hand_landmarks.landmark[0]

    distancias = []
    for p in puntas:
        lm = hand_landmarks.landmark[p]
        dx = lm.x - muñeca.x
        dy = lm.y - muñeca.y
        distancias.append(np.sqrt(dx**2 + dy**2))

    return np.std(distancias)


def esta_en_zona_activa(hand_landmarks):
    """Verifica si la mano está dentro de la zona activa de señas."""
    cx, cy = calcular_centro_mano(hand_landmarks)
    return (ROI_X_MIN <= cx <= ROI_X_MAX and ROI_Y_MIN <= cy <= ROI_Y_MAX)


def es_mano_en_reposo(hand_landmarks):
    """Detecta si la mano está en posición natural de reposo."""
    _, cy = calcular_centro_mano(hand_landmarks)
    spread = calcular_spread_dedos(hand_landmarks)

    if cy > UMBRAL_MANO_CAIDA_Y and spread < UMBRAL_MOVIMIENTO_DEDOS:
        return True
    return False


HOLD_FRAMES_MANO = 4  # ~150-200ms a 20-25 FPS: debe coincidir con 3_traductor.py

class SostenedorManos:
    """
    Sostiene brevemente los landmarks de una mano cuando MediaPipe la pierde
    por 1-2 frames (p.ej. mano de canto durante un giro de muñeca, como en
    GRACIAS). La cámara apunta a la persona de enfrente, no se le puede pedir
    que ajuste su forma de señalizar — la mitigación es de software, y debe
    ser idéntica a la de 3_traductor.py para que el modelo entrene con la
    misma distribución que va a ver en inferencia.
    """
    def __init__(self, hold_frames=HOLD_FRAMES_MANO):
        self.hold_frames = hold_frames
        self._ultimo = {"left": None, "right": None}
        self._edad = {"left": 0, "right": 0}

    def actualizar(self, result):
        salida = {}
        for lado, actual in (("left", result.left_hand_landmarks), ("right", result.right_hand_landmarks)):
            if actual is not None:
                self._ultimo[lado] = actual
                self._edad[lado] = 0
                salida[lado] = actual
            elif self._ultimo[lado] is not None and self._edad[lado] < self.hold_frames:
                self._edad[lado] += 1
                salida[lado] = self._ultimo[lado]
            else:
                self._ultimo[lado] = None
                salida[lado] = None
        return salida


_sostenedor_manos = SostenedorManos()


def filtrar_manos(manos):
    """
    Evalúa las manos (ya con sostén breve aplicado, ver SostenedorManos).
    Una mano es válida si:
    1. Está en la zona activa
    2. No está en posición de reposo

    Retorna: dict {"left": bool, "right": bool}
    """
    validas = {"left": False, "right": False}

    for lado in ("left", "right"):
        hand_lm = manos.get(lado)
        if hand_lm is None:
            continue
        if not esta_en_zona_activa(hand_lm):
            continue
        if es_mano_en_reposo(hand_lm):
            continue
        validas[lado] = True

    return validas


# =============================================================================
# EXTRACCIÓN DE LANDMARKS CON FILTRADO
# =============================================================================

def _features_manos(manos, validas):
    """slot0 = mano izquierda, slot1 = mano derecha (fijo, Holistic ya las separa)."""
    features = np.zeros(FEATURES_MANOS)

    if validas["left"] and manos.get("left"):
        hand_lm = manos["left"]
        wrist = hand_lm.landmark[0]
        for i, lm in enumerate(hand_lm.landmark):
            base = i * 3
            features[base] = lm.x - wrist.x
            features[base + 1] = lm.y - wrist.y
            features[base + 2] = lm.z - wrist.z

    if validas["right"] and manos.get("right"):
        hand_lm = manos["right"]
        wrist = hand_lm.landmark[0]
        for i, lm in enumerate(hand_lm.landmark):
            base = 63 + i * 3
            features[base] = lm.x - wrist.x
            features[base + 1] = lm.y - wrist.y
            features[base + 2] = lm.z - wrist.z

    return features


def _features_pose(result):
    """Postura superior (nariz, hombros, codos), relativa al punto medio de hombros."""
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
    """Marcadores no manuales (cejas, boca), relativos a la punta de la nariz."""
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


def extraer_landmarks(frame, result=None):
    """Extrae el vector de features (manos + pose + rostro) con filtrado integrado."""
    if result is None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

    manos = _sostenedor_manos.actualizar(result)
    validas = filtrar_manos(manos)
    num_manos_validas = int(validas["left"]) + int(validas["right"])

    features = np.concatenate([
        _features_manos(manos, validas),
        _features_pose(result),
        _features_rostro(result),
    ])

    return features, result, num_manos_validas, validas, manos


def dibujar_roi(frame):
    """Dibuja la zona activa de señas."""
    h, w = frame.shape[:2]
    x1 = int(ROI_X_MIN * w)
    y1 = int(ROI_Y_MIN * h)
    x2 = int(ROI_X_MAX * w)
    y2 = int(ROI_Y_MAX * h)

    color_roi = (100, 100, 100)
    for x in range(x1, x2, 20):
        cv2.line(frame, (x, y1), (min(x+10, x2), y1), color_roi, 1)
        cv2.line(frame, (x, y2), (min(x+10, x2), y2), color_roi, 1)
    for y in range(y1, y2, 20):
        cv2.line(frame, (x1, y), (x1, min(y+10, y2)), color_roi, 1)
        cv2.line(frame, (x2, y), (x2, min(y+10, y2)), color_roi, 1)

    cv2.putText(frame, "ZONA DE SENAS", (x1 + 5, y1 + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_roi, 1)


def normalizar_frames(frames_lista, objetivo):
    """
    Reescala una lista de frames capturados (de duración variable) a
    exactamente `objetivo` frames, por remuestreo de índices: recorta si
    hay más de los necesarios, duplica frames cercanos si hay menos.
    Mismo criterio usado en Fernández (2026, ESPOCH) para normalizar
    secuencias de duración variable a un tamaño fijo de entrada del modelo.
    """
    n = len(frames_lista)
    if n == objetivo:
        return frames_lista
    idx = np.clip(np.round(np.linspace(0, n - 1, objetivo)).astype(int), 0, n - 1)
    return [frames_lista[i] for i in idx]


def guardar_datos(nombre_sena, secuencias):
    """Guarda secuencias en JSON."""
    os.makedirs(os.path.join(DIR_DATOS, nombre_sena), exist_ok=True)
    archivo = os.path.join(DIR_DATOS, nombre_sena, f"seq_{datetime.now():%Y%m%d_%H%M%S}.json")

    with open(archivo, 'w') as f:
        json.dump({
            "sena": nombre_sena,
            "frames": FRAMES_SECUENCIA,
            "features": FEATURES,
            "secuencias": [s.tolist() for s in secuencias]
        }, f)

    print(f"✅ {len(secuencias)} secuencias guardadas en {archivo}")


def main(nombre=None, meta=40):
    """
    Args:
        nombre: Nombre de la seña (si None, pregunta interactivamente)
        meta: Cantidad de secuencias a grabar (default: 40)
              El entrenador genera 14 variantes por cada una → ~600 muestras de entrenamiento

    Grabación por TIEMPO FIJO (no auto-disparo al detectar mano): cada
    secuencia tiene una cuenta atrás de preparación, una ventana de
    grabación de duración fija, y una pausa de descanso. Esto evita cortar
    la seña a mitad de gesto (problema reportado al usar auto-disparo).
    Los filtros de zona activa/reposo actúan como control de calidad
    DESPUÉS de capturar: si la mano no estuvo en posición válida la
    mayor parte del tiempo, la secuencia se descarta y se reintenta.
    """
    print("\n" + "="*60)
    print("  GRABADOR DE SEÑAS (TIEMPO FIJO)")
    print("="*60)

    # Si no se pasa nombre, preguntar
    if not nombre:
        nombre = input("\nNombre de la seña: ").strip().upper()
    else:
        nombre = nombre.strip().upper()

    if not nombre:
        print("❌ Nombre inválido")
        return

    print(f"\n🎯 Seña: {nombre}")
    print(f"📊 Meta: {meta} secuencias reales → ~{meta * 15} en entrenamiento (14x augmentation)")
    print(f"  Cuenta atrás de {PAUSA_PREPARACION:.0f}s, grabación de {DURACION_MUESTRA:.0f}s, pausa de {PAUSA_ENTRE_MUESTRAS:.0f}s")
    print(f"  Presiona Q para terminar")

    cap = cv2.VideoCapture(int(CAMARA_FUENTE) if CAMARA_FUENTE.isdigit() else CAMARA_FUENTE)
    if CAMARA_FUENTE.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la cámara.\n"
            "Verifica que la cámara esté conectada y no esté siendo usada por otra aplicación."
        )

    # Verificar que se puede leer un frame
    ret, _ = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(
            "La cámara se abrió pero no se pudo leer un frame.\n"
            "Verifica los permisos de la cámara."
        )

    secuencias = []
    buffer_crudo = []  # lista de (features, num_manos_validas) durante la ventana de grabación
    estado = "preparando"  # preparando -> grabando -> pausa -> preparando...
    tiempo_estado_inicio = time.time()

    print("\n🎥 Cámara iniciada. Prepárate para la primera seña...")

    while len(secuencias) < meta:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        if not MODO_WEARABLE:
            frame = cv2.flip(frame, 1)

        # Procesar con MediaPipe Holistic (siempre, para feedback visual)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)
        features, result, num_manos_validas, validas, manos = extraer_landmarks(frame, result)

        ahora = time.time()
        transcurrido = ahora - tiempo_estado_inicio
        total_detectadas = int(manos.get("left") is not None) + int(manos.get("right") is not None)

        # Dibujar zona activa y manos con colores (en todo momento)
        dibujar_roi(frame)
        for lado, hand_lm in (("left", manos.get("left")), ("right", manos.get("right"))):
            if hand_lm is None:
                continue
            if validas[lado]:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_holistic.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=4),
                    mp_draw.DrawingSpec(color=(0,200,0), thickness=2)
                )
            else:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_holistic.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,0,150), thickness=1, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0,0,100), thickness=1)
                )

        # === MÁQUINA DE ESTADOS: preparando -> grabando -> pausa ===
        if estado == "preparando":
            restante = PAUSA_PREPARACION - transcurrido
            cv2.rectangle(frame, (120, 190), (520, 290), (0,100,200), -1)
            cv2.putText(frame, "PREPARATE", (200, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f"Grabando en {max(restante, 0):.1f}s", (190, 265),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            if restante <= 0:
                estado = "grabando"
                tiempo_estado_inicio = ahora
                buffer_crudo = []
                print(f"  ⏺ Grabando secuencia {len(secuencias)+1}...")

        elif estado == "grabando":
            buffer_crudo.append((features, num_manos_validas))
            progreso = min(transcurrido / DURACION_MUESTRA, 1.0)
            cv2.rectangle(frame, (20, 430), (620, 460), (50,50,50), -1)
            cv2.rectangle(frame, (20, 430), (int(20 + 600*progreso), 460), (0,0,255), -1)
            cv2.putText(frame, f"GRABANDO: {transcurrido:.1f}/{DURACION_MUESTRA:.0f}s",
                       (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if transcurrido >= DURACION_MUESTRA:
                n_capturados = len(buffer_crudo)
                n_con_mano = sum(1 for _, n in buffer_crudo if n > 0)
                proporcion_valida = (n_con_mano / n_capturados) if n_capturados else 0.0

                if n_capturados < MIN_FRAMES_CAPTURADOS or proporcion_valida < MIN_PROPORCION_MANO_VALIDA:
                    print(f"  ⚠ Descartada (mano válida {proporcion_valida:.0%} del tiempo, {n_capturados} frames) — repite")
                else:
                    feats_crudos = [f for f, _ in buffer_crudo]
                    feats_normalizados = normalizar_frames(feats_crudos, FRAMES_SECUENCIA)
                    secuencias.append(np.array(feats_normalizados))
                    print(f"  ✓ Secuencia {len(secuencias)}/{meta} completada "
                          f"({n_capturados} frames capturados → {FRAMES_SECUENCIA}, mano válida {proporcion_valida:.0%})")

                estado = "pausa"
                tiempo_estado_inicio = ahora

        elif estado == "pausa":
            restante = PAUSA_ENTRE_MUESTRAS - transcurrido
            cv2.rectangle(frame, (150, 200), (490, 280), (60,60,60), -1)
            cv2.putText(frame, "DESCANSA", (220, 235),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f"Siguiente en {max(restante, 0):.1f}s", (190, 265),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            if restante <= 0:
                estado = "preparando"
                tiempo_estado_inicio = ahora

        # === UI ===
        cv2.rectangle(frame, (0, 0), (640, 70), (40,40,40), -1)
        cv2.putText(frame, f"Sena: {nombre}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"Progreso: {len(secuencias)}/{meta}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        info_manos = f"Manos: {num_manos_validas}/{total_detectadas}"
        cv2.putText(frame, info_manos, (400, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        pct = len(secuencias) / meta * 100
        cv2.rectangle(frame, (400, 20), (630, 50), (60,60,60), -1)
        cv2.rectangle(frame, (400, 20), (int(400 + 230*(len(secuencias)/meta)), 50), (0,200,0), -1)
        cv2.putText(frame, f"{pct:.0f}%", (500, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        color_estado = {"preparando": (0,165,255), "grabando": (0,0,255), "pausa": (100,100,100)}[estado]
        cv2.circle(frame, (620, 90), 20, color_estado, -1)
        cv2.putText(frame, estado.upper(), (515, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_estado, 1)

        cv2.imshow('Grabador LSE - Tiempo Fijo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

    if secuencias:
        guardar_datos(nombre, secuencias)
        print(f"\n✅ COMPLETADO: {len(secuencias)} secuencias de '{nombre}'")

        # === Sincronizar con la nube automáticamente ===
        try:
            from sync_cloud import SyncCloud
            sync = SyncCloud()
            if sync.conectar():
                print("\n☁️  Subiendo datos de señas a la nube...")
                sync.subir_datos_senas()
            else:
                print("\n☁️  Sin conexión. Los datos se sincronizarán después.")
        except ImportError:
            pass  # firebase-admin no instalado, silencioso
        except Exception as e:
            print(f"\n☁️  Error de sync (no crítico): {e}")
    else:
        print("\n⚠️ No se grabaron secuencias")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Grabador de señas LSE')
    parser.add_argument('--nombre', '-n', type=str, default=None, help='Nombre de la seña')
    parser.add_argument('--cantidad', '-c', type=int, default=40,
                        help='Secuencias a grabar (default: 40). El entrenador genera 14 variantes por cada una.')
    args = parser.parse_args()
    main(nombre=args.nombre, meta=args.cantidad)
