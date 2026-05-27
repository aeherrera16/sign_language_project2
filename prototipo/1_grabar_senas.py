#!/usr/bin/env python3
"""
GRABADOR DE SEÑAS - AUTOMÁTICO CON FILTRADO
Graba automáticamente cuando detecta mano estable.
Incluye filtrado de zona activa y posición natural.
NO requiere presionar teclas.
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
from utils_silenciar import init_mediapipe
mp, mp_hands, mp_draw, hands = init_mediapipe(
    max_hands=4, detection_conf=0.5, tracking_conf=0.3, static_mode=False
)

# === CONFIGURACIÓN ===
FRAMES_SECUENCIA = 30
LANDMARKS_MANO = 21
COORDS = 3
FEATURES = LANDMARKS_MANO * COORDS * 2

DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")

# === ZONA ACTIVA DE SEÑAS (ROI) ===
ROI_X_MIN = 0.10
ROI_X_MAX = 0.90
ROI_Y_MIN = 0.05
ROI_Y_MAX = 0.75

# === FILTRO DE POSICIÓN NATURAL ===
UMBRAL_MANO_CAIDA_Y = 0.70
UMBRAL_MOVIMIENTO_DEDOS = 0.015


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


def filtrar_manos(result):
    """
    Filtra las manos detectadas. Solo permite las que estén:
    1. En la zona activa
    2. No en posición de reposo
    3. Máximo 2 manos (las más cercanas al centro)
    
    Retorna: lista de índices de manos válidas
    """
    if not result.multi_hand_landmarks:
        return []
    
    manos_validas = []
    
    for idx, hand_lm in enumerate(result.multi_hand_landmarks):
        cx, cy = calcular_centro_mano(hand_lm)
        
        if not esta_en_zona_activa(hand_lm):
            continue
        
        if es_mano_en_reposo(hand_lm):
            continue
        
        dist_centro = abs(cx - 0.5)
        manos_validas.append((idx, dist_centro))
    
    # Máximo 2 manos, las más cercanas al centro
    if len(manos_validas) > 2:
        manos_validas.sort(key=lambda x: x[1])
        manos_validas = manos_validas[:2]
    
    return [m[0] for m in manos_validas]


# =============================================================================
# EXTRACCIÓN DE LANDMARKS CON FILTRADO
# =============================================================================

def extraer_landmarks(frame, result=None):
    """Extrae landmarks de las manos con filtrado integrado."""
    if result is None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
    
    features = np.zeros(FEATURES)
    
    indices_validos = filtrar_manos(result)
    num_manos_validas = len(indices_validos)
    
    if result.multi_hand_landmarks and indices_validos:
        validos = indices_validos[:2]

        # Señas con 2 manos: ordenar por lateralidad (Left→slot0, Right→slot1)
        # Garantiza orden consistente frame a frame — MediaPipe puede alternar el orden
        if len(validos) == 2 and result.multi_handedness:
            validos = sorted(
                validos,
                key=lambda i: 0 if result.multi_handedness[i].classification[0].label == "Left" else 1
            )

        for slot, idx in enumerate(validos):
            hand_lm = result.multi_hand_landmarks[idx]
            wrist = hand_lm.landmark[0]
            for i, lm in enumerate(hand_lm.landmark):
                base = slot * 63 + i * 3
                features[base] = lm.x - wrist.x
                features[base + 1] = lm.y - wrist.y
                features[base + 2] = lm.z - wrist.z

    return features, result, num_manos_validas, indices_validos


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


def main(nombre=None, meta=15):
    """
    Args:
        nombre: Nombre de la seña (si None, pregunta interactivamente)
        meta: Cantidad de secuencias a grabar (default: 15)
              El entrenador genera 14 variantes por cada una → ~210 muestras de entrenamiento
    """
    print("\n" + "="*60)
    print("  GRABADOR AUTOMÁTICO DE SEÑAS (CON FILTRADO)")
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
    print(f"  Grabación automática | Presiona Q para terminar")
    
    cap = cv2.VideoCapture(0)
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
    buffer = []
    grabando = False
    pausa_hasta = 0
    
    print("\n🎥 Cámara iniciada. Muestra tu mano en la zona activa...")
    
    while len(secuencias) < meta:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Procesar con MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        # Extraer con filtrado
        features, result, num_manos_validas, indices_validos = extraer_landmarks(frame, result)
        
        hay_mano = num_manos_validas > 0
        ahora = time.time()
        en_pausa = ahora < pausa_hasta
        total_detectadas = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        
        # Dibujar zona activa
        dibujar_roi(frame)
        
        # Dibujar manos con colores
        if result.multi_hand_landmarks:
            for idx, hand_lm in enumerate(result.multi_hand_landmarks):
                if idx in indices_validos:
                    # Mano válida: VERDE
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=4),
                        mp_draw.DrawingSpec(color=(0,200,0), thickness=2)
                    )
                else:
                    # Mano ignorada: ROJO tenue
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,0,150), thickness=1, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0,0,100), thickness=1)
                    )
                    cx, cy = calcular_centro_mano(hand_lm)
                    h_frame, w_frame = frame.shape[:2]
                    px, py = int(cx * w_frame), int(cy * h_frame)
                    cv2.putText(frame, "IGNORADA", (px - 30, py - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)
        
        # === LÓGICA DE GRABACIÓN AUTOMÁTICA ===
        if hay_mano and not en_pausa:
            if not grabando:
                grabando = True
                buffer = []
                print(f"  ⏺ Grabando secuencia {len(secuencias)+1}...")
            
            buffer.append(features)
            progreso = len(buffer) / FRAMES_SECUENCIA
            
            # Barra de progreso
            cv2.rectangle(frame, (20, 430), (620, 460), (50,50,50), -1)
            cv2.rectangle(frame, (20, 430), (int(20 + 600*progreso), 460), (0,255,0), -1)
            cv2.putText(frame, f"GRABANDO: {len(buffer)}/{FRAMES_SECUENCIA}", 
                       (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            if len(buffer) >= FRAMES_SECUENCIA:
                secuencias.append(np.array(buffer))
                buffer = []
                grabando = False
                pausa_hasta = ahora + 1.0
                print(f"  ✓ Secuencia {len(secuencias)}/{meta} completada")
        
        elif not hay_mano:
            if grabando and len(buffer) < FRAMES_SECUENCIA:
                buffer = []
                grabando = False
        
        elif en_pausa:
            tiempo_restante = pausa_hasta - ahora
            cv2.rectangle(frame, (150, 200), (490, 280), (0,100,200), -1)
            cv2.putText(frame, "PAUSA", (260, 235), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f"Baja la mano... {tiempo_restante:.1f}s", (180, 265),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # === UI ===
        # Header
        cv2.rectangle(frame, (0, 0), (640, 70), (40,40,40), -1)
        cv2.putText(frame, f"Sena: {nombre}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"Progreso: {len(secuencias)}/{meta}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Info de filtrado
        info_manos = f"Manos: {num_manos_validas}/{total_detectadas}"
        cv2.putText(frame, info_manos, (400, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        # Porcentaje total
        pct = len(secuencias) / meta * 100
        cv2.rectangle(frame, (400, 20), (630, 50), (60,60,60), -1)
        cv2.rectangle(frame, (400, 20), (int(400 + 230*(len(secuencias)/meta)), 50), (0,200,0), -1)
        cv2.putText(frame, f"{pct:.0f}%", (500, 42), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Estado
        if hay_mano and not en_pausa:
            cv2.circle(frame, (620, 90), 20, (0,255,0), -1)
            cv2.putText(frame, "DETECTANDO", (520, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        elif en_pausa:
            cv2.circle(frame, (620, 90), 20, (0,165,255), -1)
        elif total_detectadas > 0:
            cv2.circle(frame, (620, 90), 20, (0,0,200), -1)
            cv2.putText(frame, "IGNORADA", (530, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1)
        else:
            cv2.circle(frame, (620, 90), 20, (0,0,255), -1)
            cv2.putText(frame, "SIN MANO", (530, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        cv2.imshow('Grabador LSE - Automatico', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
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
    parser.add_argument('--cantidad', '-c', type=int, default=15,
                        help='Secuencias a grabar (default: 15). El entrenador genera 14 variantes por cada una.')
    args = parser.parse_args()
    main(nombre=args.nombre, meta=args.cantidad)
