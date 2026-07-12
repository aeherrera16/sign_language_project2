#!/usr/bin/env python3
"""
TRADUCTOR LSE - Con generación de oraciones naturales
Detecta fin de frase cuando bajas las manos.
Convierte palabras clave en oraciones con sentido.

MEJORAS:
- Zona activa de señas (ROI): ignora manos fuera del área de señas
- Filtro de posición natural: descarta manos en reposo/colgando
- Selección de señante principal: si hay múltiples personas, usa la más centrada
- Ignora si hay más de 2 manos válidas en la zona activa
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import sys

# Protección para PyInstaller windowed (console=False) donde stdout/stderr son None
class _NullWriter:
    def write(self, text): pass
    def flush(self): pass
    def fileno(self): raise OSError("No fd")
    def isatty(self): return False
    def __bool__(self): return True

if sys.stdout is None:
    sys.stdout = _NullWriter()
if sys.stderr is None:
    sys.stderr = _NullWriter()

import cv2
import numpy as np
import json
import pickle
import time
import shutil
import tempfile
from collections import deque
import threading
import queue

# Importar MediaPipe y TensorFlow SIN warnings
from utils_silenciar import init_mediapipe_holistic, init_tensorflow
# detection_conf bajo (0.4): la cámara apunta a la persona de enfrente (no se
# le puede pedir que ajuste cómo señaliza), así que hay que ser más permisivo
# para captar manos parcialmente visibles/de canto durante giros de muñeca.
mp, mp_holistic, mp_draw, holistic = init_mediapipe_holistic(detection_conf=0.4, tracking_conf=0.5, model_complexity=0)
tf = init_tensorflow()

TTS_OK = True

DIR_BASE = os.path.dirname(__file__)


def _leer_version():
    """Lee prototipo/VERSION si existe (lo escribe el workflow de release)."""
    try:
        with open(os.path.join(DIR_BASE, "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


_version_actual = _leer_version()
TITULO_VENTANA = f"Traductor LSE v{_version_actual}" if _version_actual else "Traductor LSE"
VOICE_CONFIG = os.path.join(DIR_BASE, ".voz_config.json")
DIR_VOCES = os.path.join(DIR_BASE, "voces")
PIPER_VOICES = {
    "neural": {
        "nombre": "Voz Humana (Alta Calidad)",
        "model": os.path.join(DIR_VOCES, "es_MX-claude-high.onnx"),
        "config": os.path.join(DIR_VOCES, "es_MX-claude-high.onnx.json"),
    },
}

def _env_float(nombre, defecto):
    try:
        return float(os.environ.get(nombre, defecto))
    except (TypeError, ValueError):
        return defecto

def _env_int(nombre, defecto):
    try:
        return int(os.environ.get(nombre, defecto))
    except (TypeError, ValueError):
        return defecto

def _env_bool(nombre, defecto):
    valor = os.environ.get(nombre)
    if valor is None:
        return defecto
    return valor.strip().lower() in {"1", "true", "yes", "on", "si", "sí"}

def _voz_preferida():
    voz = os.environ.get("LSE_VOZ", "").strip().lower()
    if not voz and os.path.exists(VOICE_CONFIG):
        try:
            with open(VOICE_CONFIG, encoding="utf-8") as f:
                voz = json.load(f).get("voz", "")
        except Exception:
            voz = ""
    return voz if voz in {"neural", "robot"} else "neural"

def _get_piper_bin():
    import shutil
    if shutil.which("piper"): return "piper"
    local_bin = os.path.expanduser("~/.local/bin/piper")
    if os.path.exists(local_bin): return local_bin
    return None

def _piper_disponible(voz):
    cfg = PIPER_VOICES.get(voz)
    return bool(
        cfg and _get_piper_bin() and
        os.path.exists(cfg["model"]) and
        os.path.exists(cfg["config"])
    )

def describir_voz_actual():
    voz = _voz_preferida()
    if voz == "neural" and _piper_disponible(voz):
        return "Voz Humana (Piper)"
    return "Voz Robótica (espeak)"

def hablar_texto(texto):
    if not texto: return
    texto_voz = preparar_texto_para_voz(texto)
    print(f"\n🔊 HABLANDO: {texto}")
    import subprocess
    try:
        if sys.platform == 'darwin':
            # macOS: 'say' viene instalado por defecto, voz española.
            # OJO: el nombre real de la voz lleva tilde ("Mónica"). Si se
            # pasa "Monica" sin tilde, `say` no encuentra coincidencia exacta
            # y cae SILENCIOSAMENTE (exit 0, sin error) a la voz por defecto
            # del sistema — que normalmente es en inglés, y se nota como un
            # acento raro leyendo texto en español.
            subprocess.run(['say', '-v', 'Paulina', '-r', str(TTS_VELOCIDAD), texto_voz],
                           stderr=subprocess.DEVNULL)
        elif sys.platform == 'win32':
            # Windows: SAPI5 vía pyttsx3, viene con Windows (no requiere instalar
            # nada aparte, ideal para el .exe empaquetado). La calidad/idioma de
            # la voz depende de las voces instaladas en esa PC — si no hay voz en
            # español, sonará con acento raro (mismo caso que macOS arriba), pero
            # nunca falla silenciosamente ni requiere binarios externos.
            import pyttsx3
            engine = pyttsx3.init()
            for voz in engine.getProperty('voices'):
                if 'spanish' in voz.name.lower() or 'español' in voz.name.lower() or '_es' in voz.id.lower():
                    engine.setProperty('voice', voz.id)
                    break
            engine.setProperty('rate', TTS_VELOCIDAD)
            engine.setProperty('volume', min(1.0, TTS_VOLUMEN / 200))
            engine.say(texto_voz)
            engine.runAndWait()
        elif _voz_preferida() == "neural" and _piper_disponible("neural"):
            cfg = PIPER_VOICES["neural"]
            piper_bin = _get_piper_bin()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            try:
                subprocess.run(
                    [piper_bin, "--model", cfg["model"], "--config", cfg["config"],
                     "--length_scale", str(PIPER_LENGTH_SCALE),
                     "--output_file", wav_path],
                    input=texto_voz,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20
                )
                player = "aplay" if shutil.which("aplay") else "paplay"
                subprocess.run([player, wav_path], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=20)
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
        else:
            # Voz robótica de fallback: masculina por defecto.
            voz_espeak = "es+m3"
            subprocess.run(['espeak', '-v', voz_espeak, '-s', str(TTS_VELOCIDAD),
                            '-a', str(TTS_VOLUMEN), texto_voz],
                          stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️ Error de voz: {e}")


# === CONFIGURACIÓN ===
DIR_MODELO = os.path.join(DIR_BASE, "modelo")
RELOAD_FLAG = os.path.join(DIR_MODELO, ".reload_model")  # Escrito por modo_traductor.py cuando hay nuevo modelo
ESTADO_TRADUCTOR = os.path.join(DIR_BASE, ".traductor_estado.json")
FRAME_TRADUCTOR = os.path.join(DIR_BASE, ".traductor_frame.jpg")
FRAMES = 30
FEATURES_MANOS = 126
FEATURES_POSE = 15
FEATURES_ROSTRO = 18
FEATURES = FEATURES_MANOS + FEATURES_POSE + FEATURES_ROSTRO  # 159

# Índices del modelo BlazePose (mp_holistic.PoseLandmark) y del face mesh canónico
POSE_INDICES = [
    mp_holistic.PoseLandmark.NOSE.value,
    mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
    mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
    mp_holistic.PoseLandmark.LEFT_ELBOW.value,
    mp_holistic.PoseLandmark.RIGHT_ELBOW.value,
]
POSE_LEFT_SHOULDER_IDX = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
POSE_RIGHT_SHOULDER_IDX = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
FACE_NOSE_TIP_IDX = 1
FACE_INDICES = [105, 334, 61, 291, 13, 14]  # ceja izq/der, comisura boca izq/der, labio sup/inf

# Fuente de cámara: índice USB (ej. "0") o URL de video (ej. RTSP de una
# cámara WiFi: "rtsp://192.168.100.1:8080/?action=stream"). Configurable
# por variable de entorno para no tocar código al cambiar de cámara.
CAMARA_FUENTE = os.environ.get("LSE_CAMARA_FUENTE", "0")

UMBRAL_CONFIANZA = _env_float("LSE_UMBRAL_CONFIANZA", 0.85)
MARGEN_MINIMO = _env_float("LSE_MARGEN_MINIMO", 0.35)
CLASES_SILENCIO = {"NINGUNA", "NONE", "SILENCIO"}  # Se detectan pero no se hablan ni muestran
COOLDOWN = _env_float("LSE_COOLDOWN", 0.5)
TIEMPO_SIN_MANOS_PARA_FIN = _env_float("LSE_TIEMPO_FIN_FRASE", 1.0)
# Tiempo máximo sin manos antes de limpiar el buffer acumulado.
# Valor alto (1.5s) evita que un breve parpadeo de detección (muy frecuente
# con RTSP WiFi a ~10fps) resetee el buffer y obligue a empezar desde cero.
TIEMPO_LIMPIEZA_BUFFER = _env_float("LSE_TIEMPO_LIMPIEZA_BUFFER", 1.5)
# 5 confirmaciones seguidas (antes 3) + margen más amplio (antes 0.25):
# valores probados y afinados en vivo en la Raspberry Pi para evitar que
# el traductor "invente" señas con una sola lectura ruidosa.
CONFIRMACIONES_REQUERIDAS = _env_int("LSE_CONFIRMACIONES", 8)
INTERVALO_PREDICCION = _env_float("LSE_INTERVALO_PREDICCION", 0.12)
# Tiempo mínimo desde que aparece una mano antes de intentar la primera
# predicción — evita que el sistema "se lance a adivinar" en el instante en
# que ve una mano, dándole tiempo al señante a empezar a formar la seña.
# No exige quietud (compatible con señas dinámicas/conversación continua),
# solo retrasa el primer intento de predicción.
TIEMPO_MINIMO_SENA = _env_float("LSE_TIEMPO_MINIMO_SENA", 0.6)
TTS_VELOCIDAD = _env_int("LSE_TTS_VELOCIDAD", 140)
TTS_VOLUMEN = _env_int("LSE_TTS_VOLUMEN", 180)
PIPER_LENGTH_SCALE = _env_float("LSE_PIPER_LENGTH_SCALE", 1.0)
INTERVALO_FRAME_REMOTO = _env_float("LSE_INTERVALO_FRAME", 0.25)

# === FILTRO DE POSICIÓN NATURAL ===
UMBRAL_MANO_CAIDA_Y = 0.70
# Debe coincidir con 1_grabar_senas.py.
UMBRAL_MOVIMIENTO_DEDOS = 0.008

# === DIFUMINADO DE FONDO ===
BLUR_STRENGTH = 21          # Reducido de 35→21 para rendimiento en RPi (debe ser impar)
BLUR_ACTIVO_DEFAULT = _env_bool("LSE_BLUR", False)  # Tecla [B] alterna en tiempo real

# === SISTEMA DE GENERACIÓN DE ORACIONES (100% OFFLINE) ===
# Vocabulario acotado a las clases realmente entrenadas en el modelo actual
# (ver modelo/info.json). El diccionario de noticias original se retiró:
# el modelo ya no produce esas clases, era código muerto.

# Vocabulario: palabra de seña → texto natural (fallback palabra por palabra
# cuando la secuencia no coincide con ningún patrón completo)
VOCABULARIO = {
    "HOLA": "hola",
    "BIENVENIDO": "bienvenido",
    "ESCUELA": "la escuela",
    "PRUEBA": "prueba",
    "TRADUCTOR": "el traductor",
    "LENGUA": "lengua",
    "SEÑAS": "señas",
    "TODOS": "todos",
    "APRENDER": "aprender",
    "COSAS": "cosas",
    "MAS": "más",
    "MI": "mi",
    "NOMBRE": "nombre",
    "ANAHY": "Anahy",
    "CAMILA": "Camila",
    # Letras del alfabeto dactilológico (deletreo)
    "E": "e",
    "S": "s",
    "P": "p",
}

# Patrones de frases completas: secuencia exacta de señas → oración natural
PATRONES = [
    # Deletreo de siglas: E-S-P-E seguidas se lee como la sigla completa.
    (["E", "S", "P", "E"], "ESPE"),

    # Frase completa de bienvenida/demo del prototipo funcional.
    # Señar en este orden: HOLA, BIENVENIDO, ESCUELA, E, S, P, E, PRUEBA,
    # TRADUCTOR, LENGUA, SEÑAS, TODOS. ("ecuatoriana" es texto fijo de la
    # plantilla — ECUADOR no es una seña entrenada.)
    (["HOLA", "BIENVENIDO", "ESCUELA", "E", "S", "P", "E", "PRUEBA",
      "TRADUCTOR", "LENGUA", "SEÑAS", "TODOS"],
     "Hola, bienvenido a la Escuela Politécnica del Ejército, ESPE, esta es "
     "una prueba del traductor de lengua de señas ecuatoriana. Bienvenido a todos"),

    # Presentaciones
    (["HOLA", "NOMBRE", "ANAHY"], "Hola, mi nombre es Anahy"),
    (["HOLA", "NOMBRE", "CAMILA"], "Hola, mi nombre es Camila"),
    (["HOLA", "NOMBRE", "MI", "ANAHY"], "Hola, mi nombre es Anahy"),
    (["HOLA", "NOMBRE", "MI", "CAMILA"], "Hola, mi nombre es Camila"),

    # Combinaciones parciales útiles
    (["LENGUA", "SEÑAS"], "lengua de señas"),
    (["APRENDER", "LENGUA", "SEÑAS"], "aprender lengua de señas"),
]

# Ajustes fonéticos solo para TTS. No cambian el texto mostrado en pantalla.
PRONUNCIACION_TTS = {
    "BanEcuador": "Banco Ecuador",
    "EEUU": "Estados Unidos",
    "ANÑOS": "años",
}

def preparar_texto_para_voz(texto):
    resultado = texto
    for original, pronunciable in PRONUNCIACION_TTS.items():
        resultado = resultado.replace(original, pronunciable)
    return resultado


# Conversión de números a texto
NUMEROS_TEXTO = {
    "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
    "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve",
    "10": "diez", "11": "once", "12": "doce", "13": "trece", "14": "catorce",
    "15": "quince", "16": "dieciséis", "17": "diecisiete", "18": "dieciocho",
    "19": "diecinueve", "20": "veinte", "21": "veintiuno", "22": "veintidós",
    "23": "veintitrés", "24": "veinticuatro", "25": "veinticinco",
    "26": "veintiséis", "27": "veintisiete", "28": "veintiocho", "29": "veintinueve",
    "30": "treinta", "40": "cuarenta", "50": "cincuenta", "60": "sesenta",
    "70": "setenta", "80": "ochenta", "90": "noventa", "100": "cien",
}

def numero_a_texto(numero):
    """Convierte un número a texto en español."""
    if numero in NUMEROS_TEXTO:
        return NUMEROS_TEXTO[numero]
    
    try:
        num = int(numero)
        if num < 100:
            decenas = (num // 10) * 10
            unidades = num % 10
            if unidades == 0:
                return NUMEROS_TEXTO.get(str(decenas), numero)
            elif decenas == 20:
                return NUMEROS_TEXTO.get(str(num), f"veinti{NUMEROS_TEXTO.get(str(unidades), '')}")
            else:
                return f"{NUMEROS_TEXTO.get(str(decenas), '')} y {NUMEROS_TEXTO.get(str(unidades), '')}"
        elif num >= 1000 and num < 10000:
            return str(num)  # Años como número
    except:
        pass
    
    return numero

def combinar_numeros(palabras):
    """Combina números consecutivos: ['2', '8', 'AÑO'] → ['28', 'AÑO']"""
    resultado = []
    numero_actual = ""
    
    for palabra in palabras:
        if palabra.isdigit():
            numero_actual += palabra
        else:
            if numero_actual:
                resultado.append(numero_actual)
                numero_actual = ""
            resultado.append(palabra)
    
    if numero_actual:
        resultado.append(numero_actual)
    
    return resultado

def buscar_patron(palabras):
    """Busca un patrón que coincida con las palabras."""
    for patron, resultado in PATRONES:
        if palabras == patron:
            return resultado
        # Buscar coincidencia parcial al inicio
        if len(palabras) >= len(patron) and palabras[:len(patron)] == patron:
            resto = palabras[len(patron):]
            if resto:
                return resultado + " " + generar_oracion(resto)
            return resultado
    return None

def generar_oracion(palabras):
    """
    Convierte lista de señas en oración natural.
    100% offline - usa reglas predefinidas.
    """
    if not palabras:
        return ""
    
    # Combinar números consecutivos
    palabras = combinar_numeros(palabras)
    
    # Buscar patrón predefinido
    patron = buscar_patron(palabras)
    if patron:
        return patron.capitalize()
    
    # Si no hay patrón, construir palabra por palabra
    oracion = []
    i = 0
    
    while i < len(palabras):
        palabra = palabras[i]
        siguiente = palabras[i + 1] if i + 1 < len(palabras) else None
        
        if palabra.isdigit():
            oracion.append(numero_a_texto(palabra))
        elif palabra in VOCABULARIO:
            texto = VOCABULARIO[palabra]
            oracion.append(texto)
            
            # Añadir conector "de" si corresponde
            if siguiente and siguiente in ["ECUADOR", "PAIS", "QUITO", "GUAYAQUIL"]:
                oracion.append("de")
        else:
            oracion.append(palabra.lower())
        
        i += 1
    
    resultado = " ".join(oracion)
    resultado = " ".join(resultado.split())  # Limpiar espacios
    
    if resultado:
        resultado = resultado[0].upper() + resultado[1:]
    
    return resultado


# =============================================================================
# FUNCIONES DE FILTRADO DE MANOS
# =============================================================================

def calcular_centro_mano(hand_landmarks):
    """Calcula el centro (promedio) de todos los landmarks de una mano."""
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return np.mean(xs), np.mean(ys)


def calcular_spread_dedos(hand_landmarks):
    """Calcula la extensión de los dedos (std de distancias punta-muñeca)."""
    puntas = [4, 8, 12, 16, 20]
    muñeca = hand_landmarks.landmark[0]
    distancias = []
    for p in puntas:
        lm = hand_landmarks.landmark[p]
        dx = lm.x - muñeca.x
        dy = lm.y - muñeca.y
        distancias.append(np.sqrt(dx**2 + dy**2))
    return np.std(distancias)


def es_mano_en_reposo(hand_landmarks):
    """Detecta si la mano está caída (posición natural de reposo)."""
    _, cy = calcular_centro_mano(hand_landmarks)
    spread = calcular_spread_dedos(hand_landmarks)
    if cy > UMBRAL_MANO_CAIDA_Y and spread < UMBRAL_MOVIMIENTO_DEDOS:
        return True
    return False


HOLD_FRAMES_MANO = _env_int("LSE_HOLD_FRAMES_MANO", 4)  # subir a 20 en Pi con camara WiFi (~10fps)

class SostenedorManos:
    """
    Sostiene brevemente los landmarks de una mano cuando MediaPipe la pierde
    por 1-2 frames (p.ej. mano de canto durante un giro de muñeca, como en
    GRACIAS). La cámara apunta a la persona de enfrente, así que no se puede
    pedir que ajuste su forma de señalizar — la mitigación tiene que ser de
    software. Sin esto, esos frames quedarían en cero y abrirían un hueco en
    medio de la ventana temporal que ve el LSTM.
    No sustituye una mano realmente ausente: pasado HOLD_FRAMES se zera igual.
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


def filtrar_manos(manos):
    """
    Evalúa las manos (ya con sostén breve aplicado, ver SostenedorManos):
    - Descarta manos en posición de reposo (caídas)
    Retorna: dict {"left": bool, "right": bool}, info de debug
    """
    total = int(manos.get("left") is not None) + int(manos.get("right") is not None)
    validas = {"left": False, "right": False}

    for lado in ("left", "right"):
        hand_lm = manos.get(lado)
        if hand_lm is None or es_mano_en_reposo(hand_lm):
            continue
        validas[lado] = True

    n_validas = int(validas["left"]) + int(validas["right"])
    razon = "ok" if n_validas else ("reposo" if total else "sin_manos")
    return validas, {"total": total, "validas": n_validas, "razon": razon}


def manos_validas_landmarks(manos, validas):
    """Devuelve la lista de objetos landmark de las manos marcadas como válidas."""
    out = []
    if validas.get("left") and manos.get("left"):
        out.append(manos["left"])
    if validas.get("right") and manos.get("right"):
        out.append(manos["right"])
    return out


def obtener_bbox_mano(hand_landmarks, h, w, margen=40):
    """Obtiene el bounding box de una mano con margen extra."""
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x1 = max(0, min(xs) - margen)
    y1 = max(0, min(ys) - margen)
    x2 = min(w, max(xs) + margen)
    y2 = min(h, max(ys) + margen)
    return x1, y1, x2, y2


def difuminar_fondo(frame, manos, validas, activo=True):
    """
    Difumina el fondo y deja las manos nítidas.
    - Sin manos o activo=False: devuelve frame original (sin coste de blur)
    - Con manos: solo blurrea cuando hay algo que mostrar
    """
    manos_validas = manos_validas_landmarks(manos, validas)
    if not activo or not manos_validas:
        return frame

    h, w = frame.shape[:2]
    blurred = cv2.GaussianBlur(frame, (BLUR_STRENGTH, BLUR_STRENGTH), 0)
    resultado = blurred.copy()

    for hand_lm in manos_validas:
        x1, y1, x2, y2 = obtener_bbox_mano(hand_lm, h, w, margen=70)
        resultado[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        cv2.rectangle(resultado, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return resultado


# =============================================================================
# CLASE TRADUCTOR
# =============================================================================

def obtener_dispositivo_audio():
    """Detecta el nombre del dispositivo de audio actual (Bluetooth o Cable) en Raspberry Pi."""
    import subprocess
    if sys.platform != 'linux':
        return "Audio predeterminado"
    
    # 1. Chequear si hay un dispositivo Bluetooth conectado
    try:
        bt_str = subprocess.check_output(['bluetoothctl', 'devices', 'Connected'], stderr=subprocess.STDOUT, text=True).strip()
        if bt_str:
            nombre = " ".join(bt_str.split()[2:])
            return f"BT: {nombre}"
    except Exception:
        pass

    # 2. Fallback ALSA (Cable)
    try:
        al_str = subprocess.check_output(['aplay', '-l'], stderr=subprocess.STDOUT, text=True)
        for linea in al_str.split('\n'):
            if linea.startswith('card'):
                nombre = linea.split(':')[1].split('[')[0].strip()
                return f"Cable: {nombre}"
    except Exception:
        pass

    return "Audio del sistema"


class TraductorLSE:
    def __init__(self):
        # Cola de voz: una sola frase se reproduce a la vez. Antes cada
        # frase lanzaba su propio hilo, y si la siguiente seña se detectaba
        # mientras la anterior todavía sonaba, los dos audios competían por
        # el mismo dispositivo y se cortaban entre sí. Con la cola, las
        # frases se hablan en orden, sin solaparse, sin congelar el video.
        self._cola_voz = queue.Queue()
        threading.Thread(target=self._procesar_cola_voz, daemon=True).start()

        self.modelo = None
        self.encoder = None
        self.buffer = deque(maxlen=FRAMES)
        self.palabras = []
        self.ultima_deteccion = 0
        self.ultima_prediccion = 0
        self.ultima_sena = None
        self.ultimo_tiempo_con_mano = time.time()
        self.frase_completa = False
        # Para confirmación de señas
        self.sena_candidata = None
        self.confirmaciones = 0
        # buffer_listo: la ventana de FRAMES está llena y se puede predecir.
        # Ya no se exige que la mano esté quieta — el filtro de ruido es
        # la confirmación por N predicciones consecutivas + la clase NINGUNA.
        self.ultimo_features = None
        self.buffer_listo = False
        self.tiempo_primera_mano = None
        # Info de filtrado para debug en pantalla
        self.filtro_info = {"total": 0, "validas": 0, "rechazadas": [], "razon": ""}
        # Sostiene brevemente landmarks de mano perdidos por perfiles/cantos
        self.sostenedor_manos = SostenedorManos()
        # Audio activo detectado
        self.audio_dispositivo = obtener_dispositivo_audio()
        self.voz_dispositivo = describir_voz_actual()
        # Blur toggle
        self.blur_activo = BLUR_ACTIVO_DEFAULT
        # Hot-reload del modelo
        self._ultimo_check_reload = 0
        self._notif_recarga = 0
        # Monitor remoto para panel web/Telegram
        self._ultimo_estado_remoto = 0
        self._ultimo_frame_remoto = 0
        self.ultima_confianza = 0.0
        self.ultimo_evento = "Iniciando traductor"
        self.ultima_oracion = ""
        self.top_predicciones = []
        # TFLite vs SavedModel vs Keras
        self._use_tflite    = False
        self._use_savedmodel = False
        self._sm_infer      = None
        self._interp        = None
        self._in_idx        = None
        self._out_idx       = None
        
    def _cargar_tflite(self, tflite_path):
        """Carga el intérprete TFLite. Usa tflite-runtime si está disponible (más ligero)."""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        interp = tflite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        self._interp         = interp
        self._in_idx         = interp.get_input_details()[0]['index']
        self._out_idx        = interp.get_output_details()[0]['index']
        self._use_tflite     = True

    def recargar_modelo_si_hay_nuevo(self):
        """Comprueba la flag de recarga cada 5s. Swap atómico del modelo sin parar el video."""
        ahora = time.time()
        if ahora - self._ultimo_check_reload < 5:
            return False
        self._ultimo_check_reload = ahora

        if not os.path.exists(RELOAD_FLAG):
            return False

        try:
            os.remove(RELOAD_FLAG)
            if not self.cargar_modelo():
                return False
            self._notif_recarga = time.time()
            print(f"🔄 Modelo recargado: {list(self.encoder.classes_)}")
            return True
        except Exception as e:
            print(f"⚠️ Error recargando modelo: {e}")
            return False

    def cargar_modelo(self):
        tflite_path = os.path.join(DIR_MODELO, "modelo.tflite")
        sm_path     = os.path.join(DIR_MODELO, "modelo_savedmodel")
        h5_path     = os.path.join(DIR_MODELO, "modelo.h5")

        self._use_tflite     = False
        self._use_savedmodel = False
        self.modelo          = None

        if os.path.exists(tflite_path):
            # TFLite: ~50 MB RAM, 3x más rápido en ARM — preferido para RPi
            try:
                self._cargar_tflite(tflite_path)
                print("✅ Modelo TFLite cargado (optimizado para Raspberry Pi)")
            except Exception as e:
                print(f"⚠️ TFLite falló ({e}), probando SavedModel...")
                os.path.exists(tflite_path) and os.remove(tflite_path)  # no volver a intentarlo

        if not self._use_tflite and os.path.isdir(sm_path):
            # SavedModel: compatible con TF 2.14+ sin problemas de versión Keras
            try:
                import tensorflow as tf_load
                sm = tf_load.saved_model.load(sm_path)
                self._sm_infer       = sm.signatures['serving_default']
                self._use_savedmodel = True
                print("✅ Modelo SavedModel cargado (compatible Raspberry Pi)")
            except Exception as e:
                print(f"⚠️ SavedModel falló ({e}), probando .h5...")

        if not self._use_tflite and not self._use_savedmodel:
            if not os.path.exists(h5_path):
                print("❌ No hay modelo. Ejecuta: python 2_entrenar_modelo.py")
                return False
            import keras
            self.modelo = keras.models.load_model(h5_path, compile=False)
            print("✅ Modelo Keras (.h5) cargado")

        with open(os.path.join(DIR_MODELO, "encoder.pkl"), 'rb') as f:
            self.encoder = pickle.load(f)
        print(f"   Clases: {list(self.encoder.classes_)}")
        return True
    
    def extraer_landmarks_filtrado(self, result, validas, manos):
        """
        Extrae el vector de features (manos + pose + rostro) usando solo las
        manos válidas (ya filtradas; pueden venir sostenidas brevemente por
        SostenedorManos si MediaPipe las perdió 1-2 frames). Debe ser idéntico
        al grabador (1_grabar_senas.py) para que el modelo vea los mismos datos.
        slot0 = mano izquierda, slot1 = mano derecha (fijo, Holistic ya las separa).
        Pose y rostro se toman directo de `result` (sin sostén, no presentan
        este problema de perfiles/cantos).
        """
        features_manos = np.zeros(FEATURES_MANOS)

        if validas.get("left") and manos.get("left"):
            hand_lm = manos["left"]
            wrist = hand_lm.landmark[0]
            for i, lm in enumerate(hand_lm.landmark):
                base = i * 3
                features_manos[base] = lm.x - wrist.x
                features_manos[base + 1] = lm.y - wrist.y
                features_manos[base + 2] = lm.z - wrist.z

        if validas.get("right") and manos.get("right"):
            hand_lm = manos["right"]
            wrist = hand_lm.landmark[0]
            for i, lm in enumerate(hand_lm.landmark):
                base = 63 + i * 3
                features_manos[base] = lm.x - wrist.x
                features_manos[base + 1] = lm.y - wrist.y
                features_manos[base + 2] = lm.z - wrist.z

        features_pose = np.zeros(FEATURES_POSE)
        if result.pose_landmarks:
            lm_list = result.pose_landmarks.landmark
            ls = lm_list[POSE_LEFT_SHOULDER_IDX]
            rs = lm_list[POSE_RIGHT_SHOULDER_IDX]
            cx, cy, cz = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2, (ls.z + rs.z) / 2
            for i, idx in enumerate(POSE_INDICES):
                lm = lm_list[idx]
                base = i * 3
                features_pose[base] = lm.x - cx
                features_pose[base + 1] = lm.y - cy
                features_pose[base + 2] = lm.z - cz

        features_rostro = np.zeros(FEATURES_ROSTRO)
        if result.face_landmarks:
            lm_list = result.face_landmarks.landmark
            nose = lm_list[FACE_NOSE_TIP_IDX]
            for i, idx in enumerate(FACE_INDICES):
                lm = lm_list[idx]
                base = i * 3
                features_rostro[base] = lm.x - nose.x
                features_rostro[base + 1] = lm.y - nose.y
                features_rostro[base + 2] = lm.z - nose.z

        return np.concatenate([features_manos, features_pose, features_rostro])
    
    def predecir(self):
        if len(self.buffer) < FRAMES:
            return None, 0.0

        seq = np.array(list(self.buffer), dtype=np.float32)

        if self._use_tflite:
            self._interp.set_tensor(self._in_idx, np.expand_dims(seq, 0))
            self._interp.invoke()
            pred = self._interp.get_tensor(self._out_idx)[0]
        elif self._use_savedmodel:
            import tensorflow as tf_inf
            result = self._sm_infer(tf_inf.constant(np.expand_dims(seq, 0)))
            pred = list(result.values())[0].numpy()[0]
        else:
            # Llamar al modelo directamente en vez de .predict(): .predict()
            # tiene overhead de retrazado pensado para lotes grandes, no para
            # una sola muestra por llamada en tiempo real.
            pred = self.modelo(np.expand_dims(seq, 0), training=False).numpy()[0]

        orden = np.argsort(pred)[::-1]
        self.top_predicciones = [
            {
                "sena": str(self.encoder.inverse_transform([int(i)])[0]),
                "confianza": float(pred[i])
            }
            for i in orden[:3]
        ]
        sorted_pred = pred[orden]
        conf   = sorted_pred[0]
        margen = sorted_pred[0] - sorted_pred[1] if len(sorted_pred) > 1 else sorted_pred[0]

        if conf < UMBRAL_CONFIANZA or margen < MARGEN_MINIMO:
            return None, conf

        idx = np.argmax(pred)
        return self.encoder.inverse_transform([idx])[0], conf
    
    def hablar(self, texto):
        if TTS_OK and texto:
            # Encolar es instantáneo — no congela el video. La reproducción
            # real ocurre en el hilo worker (_procesar_cola_voz), una frase
            # a la vez.
            self._cola_voz.put(texto)

    def _procesar_cola_voz(self):
        """Worker en segundo plano: habla una frase a la vez, en orden, sin solapar audio."""
        while True:
            texto = self._cola_voz.get()
            try:
                hablar_texto(texto)
            except Exception as e:
                print(f"⚠️ Error de voz: {e}")
            finally:
                self._cola_voz.task_done()

    def guardar_estado_remoto(self, ahora, hay_mano_valida):
        """Publica un resumen liviano para verlo desde el celular sin tocar la cámara."""
        if ahora - self._ultimo_estado_remoto < 0.5:
            return
        self._ultimo_estado_remoto = ahora
        data = {
            "activo": True,
            "timestamp": ahora,
            "manos_detectadas": self.filtro_info.get("total", 0),
            "manos_validas": self.filtro_info.get("validas", 0),
            "mano_estable": bool(self.buffer_listo),
            "hay_mano": bool(hay_mano_valida),
            "sena_candidata": self.sena_candidata,
            "confirmaciones": self.confirmaciones,
            "confirmaciones_requeridas": CONFIRMACIONES_REQUERIDAS,
            "palabras": list(self.palabras),
            "oracion_preview": generar_oracion(self.palabras) if self.palabras else "",
            "ultima_sena": self.ultima_sena,
            "ultima_confianza": float(self.ultima_confianza),
            "top_predicciones": list(self.top_predicciones),
            "ultima_oracion": self.ultima_oracion,
            "evento": self.ultimo_evento,
            "audio": self.audio_dispositivo,
            "voz": self.voz_dispositivo,
            "blur": self.blur_activo,
        }
        tmp = ESTADO_TRADUCTOR + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, ESTADO_TRADUCTOR)
        except Exception:
            pass

    def guardar_frame_remoto(self, frame, ahora):
        """Guarda una captura reducida de la salida del traductor para el panel/Telegram."""
        if ahora - self._ultimo_frame_remoto < INTERVALO_FRAME_REMOTO:
            return
        self._ultimo_frame_remoto = ahora
        try:
            preview = cv2.resize(frame, (480, 360))
            tmp = FRAME_TRADUCTOR + ".tmp.jpg"
            cv2.imwrite(tmp, preview, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            os.replace(tmp, FRAME_TRADUCTOR)
        except Exception:
            pass
    
    # dibujar_roi eliminado - ya no se usa el recuadro ROI
    
    def ejecutar(self):
        if not self.cargar_modelo():
            return
        
        print("\n" + "="*60)
        print("  TRADUCTOR LSE - ORACIONES NATURALES")
        print("="*60)
        print("  Solo tus 2 manos | Blur con [B]")
        print("  [C] Limpiar | [D] Debug | [Q] Salir")
        print("-"*60)
        
        if CAMARA_FUENTE.isdigit():
            cap = cv2.VideoCapture(int(CAMARA_FUENTE))
        else:
            # Para streams de red (RTSP) limitar timeout a 5s por intento
            # en vez del default de 30s — evita que el arranque parezca colgado
            cap = cv2.VideoCapture(CAMARA_FUENTE, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        if CAMARA_FUENTE.isdigit():
            # Ajustes específicos de cámara USB — no aplican a streams de red
            # (RTSP/HTTP), donde el formato lo negocia ffmpeg con la cámara.
            # Ya NO se fuerza CAP_PROP_BUFFERSIZE=1 ni FOURCC=MJPG: en algunos
            # drivers/webcams de Windows eso obliga a una decodificación extra
            # por software y se sentía menos fluido que 1_grabar_senas.py, que
            # nunca tocó estos parámetros — se deja que la cámara use su modo
            # nativo, igual que el grabador.
            cap.set(cv2.CAP_PROP_FPS, 30)
            # Resolución reducida (antes 640x480): menos píxeles que procesar
            # en MediaPipe Holistic por frame, para PCs sin GPU donde el
            # procesamiento de pose+rostro+manos se sentía trabado.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not cap.isOpened():
            raise RuntimeError(
                "No se pudo abrir la cámara.\n"
                "Verifica que la cámara esté conectada y no esté siendo usada por otra aplicación."
            )
        
        # Warm-up de la cámara para evitar timeouts iniciales
        time.sleep(0.5)
        for _ in range(5):
            ret, _ = cap.read()
            if ret: break
            time.sleep(0.2)
        if not ret:
            raise RuntimeError("La cámara está conectada pero no envía imágenes (timeout). Desconéctala del USB y vuelve a conectarla.")
        
        cv2.namedWindow(TITULO_VENTANA, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(TITULO_VENTANA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass  # No hay gestor de ventanas disponible (modo headless)
        
        mostrar_debug = False
        self.ultimo_evento = "Traductor activo"
        reintentos_camara = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                reintentos_camara += 1
                if reintentos_camara > 15:
                    print("\n❌ Error: Se perdió la conexión con la cámara de forma permanente.")
                    break
                time.sleep(0.1)
                continue
            reintentos_camara = 0

            frame = cv2.flip(frame, 1)
            ahora = time.time()

            # Chequear si hay un modelo nuevo listo (no interrumpe el video)
            self.recargar_modelo_si_hay_nuevo()
            
            # Procesar con MediaPipe Holistic
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            # === SOSTÉN BREVE DE MANOS (perfiles/cantos de 1-2 frames) ===
            manos = self.sostenedor_manos.actualizar(result)

            # === FILTRADO DE MANOS ===
            validas, self.filtro_info = filtrar_manos(manos)
            hay_mano_valida = validas["left"] or validas["right"]

            # Extraer features solo de manos válidas (+ pose y rostro siempre)
            features = self.extraer_landmarks_filtrado(result, validas, manos)

            # === DIFUMINAR FONDO (solo cuando hay manos y blur activo) ===
            frame = difuminar_fondo(frame, manos, validas, activo=self.blur_activo)

            # Dibujar landmarks de manos válidas (VERDE)
            for hand_lm in manos_validas_landmarks(manos, validas):
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_holistic.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                    mp_draw.DrawingSpec(color=(0,200,0), thickness=2)
                )
            
            # === LÓGICA DE DETECCIÓN ===
            if hay_mano_valida:
                if self.tiempo_primera_mano is None:
                    self.tiempo_primera_mano = ahora
                self.ultimo_tiempo_con_mano = ahora
                self.ultimo_features = features.copy()
                self.buffer.append(features)
                self.buffer_listo = len(self.buffer) >= FRAMES
                tiempo_con_mano = ahora - self.tiempo_primera_mano

                # Predecir en ventana deslizante, sin exigir que la mano esté quieta:
                # una conversación real en LSE es continua. El filtro de falsos positivos
                # es la confirmación por N predicciones consecutivas sobre ventanas
                # solapadas (ver CONFIRMACIONES_REQUERIDAS) + la clase NINGUNA entrenada
                # para posturas de transición. El cooldown solo evita repetir palabras
                # ya aceptadas, no bloquea el análisis de la seña actual.
                # TIEMPO_MINIMO_SENA: además, no se intenta ni la primera predicción
                # hasta pasado ese tiempo desde que apareció la mano — evita que se
                # lance a adivinar en el instante en que detecta cualquier mano.
                if (self.buffer_listo and tiempo_con_mano >= TIEMPO_MINIMO_SENA
                        and ahora - self.ultima_prediccion > INTERVALO_PREDICCION):
                    self.ultima_prediccion = ahora
                    sena, conf = self.predecir()

                    if sena is None and mostrar_debug and conf > 0:
                        print(f"  ✗ Rechazado ({conf:.0%}) — confianza baja o modelo dudoso")
                    if conf > 0:
                        self.ultima_confianza = float(conf)

                    if sena and conf >= UMBRAL_CONFIANZA:
                        if sena == self.sena_candidata:
                            self.confirmaciones += 1
                        else:
                            self.sena_candidata = sena
                            self.confirmaciones = 1
                        
                        if self.confirmaciones >= CONFIRMACIONES_REQUERIDAS:
                            es_silencio = sena.upper() in CLASES_SILENCIO
                            if es_silencio:
                                # Seña de "no sé" — resetear sin agregar ni hablar
                                self.ultima_deteccion = ahora
                            puede_agregar = sena != self.ultima_sena
                            if not es_silencio and puede_agregar:
                                self.palabras.append(sena)
                                self.ultima_sena = sena
                                self.ultima_deteccion = ahora
                                self.ultima_confianza = float(conf)
                                self.ultimo_evento = f"Seña detectada: {sena} ({conf:.0%})"
                                print(f"  ✓ {sena} ({conf:.0%}) [Confirmado {self.confirmaciones}x]")
                                # No hablar por palabra individual — solo al completar
                                # la frase entera (ver bloque "fin de frase" más abajo).
                            self.sena_candidata = None
                            self.confirmaciones = 0
            
            else:
                # Sin manos - verificar fin de frase
                tiempo_sin_manos = ahora - self.ultimo_tiempo_con_mano

                # KEY FIX: limpiar buffer rápido para evitar predicciones
                # con frames viejos cuando las manos vuelven a aparecer brevemente
                if tiempo_sin_manos > TIEMPO_LIMPIEZA_BUFFER and self.ultimo_features is not None:
                    self.buffer.clear()
                    self.ultimo_features = None
                    self.buffer_listo = False
                    self.tiempo_primera_mano = None
                    self.sena_candidata = None
                    self.confirmaciones = 0

                if tiempo_sin_manos > TIEMPO_SIN_MANOS_PARA_FIN and self.palabras:
                    oracion = generar_oracion(self.palabras)
                    self.ultima_oracion = oracion
                    self.ultimo_evento = f"Frase enviada a voz: {oracion}"
                    self.hablar(oracion)
                    self.palabras = []
                    self.ultima_sena = None
                    self.buffer.clear()
                    self.ultimo_features = None
                    self.buffer_listo = False
                    self.tiempo_primera_mano = None
            
            # === UI ===
            h, w = frame.shape[:2]
            
            # Header semi-transparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (20,20,20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "TRADUCTOR LSE", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            # Manos detectadas
            n_validas = self.filtro_info.get('validas', 0)
            manos_txt = f"Manos: {n_validas}"
            cv2.putText(frame, manos_txt, (w-180, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            
            # Indicador de estado
            if hay_mano_valida:
                if self.buffer_listo:
                    cv2.circle(frame, (w-30, 25), 12, (0,255,0), -1)
                else:
                    cv2.circle(frame, (w-30, 25), 12, (0,255,255), -1)
                
                # Progreso de confirmación
                if self.sena_candidata and self.confirmaciones > 0:
                    progreso = f"{self.sena_candidata} ({self.confirmaciones}/{CONFIRMACIONES_REQUERIDAS})"
                    cv2.putText(frame, progreso, (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
            else:
                cv2.circle(frame, (w-30, 25), 12, (100,100,100), -1)
                
                if self.palabras:
                    tiempo_restante = TIEMPO_SIN_MANOS_PARA_FIN - (ahora - self.ultimo_tiempo_con_mano)
                    if tiempo_restante > 0:
                        cv2.putText(frame, f"Hablando en {tiempo_restante:.1f}s", (w-220, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
            
            # Debug (si activado)
            if mostrar_debug:
                cv2.rectangle(frame, (0, 90), (350, 180), (0,0,0), -1)
                cv2.putText(frame, f"Manos: {self.filtro_info.get('total',0)} det | {n_validas} valid", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                cv2.putText(frame, f"Confianza min: {UMBRAL_CONFIANZA:.0%}", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(frame, f"Confirmaciones: {CONFIRMACIONES_REQUERIDAS}x", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(frame, f"Buffer listo: {self.buffer_listo}", (10, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if self.buffer_listo else (0,0,255), 1)
            
            # Barra inferior: palabras detectadas
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h-70), (w, h), (20,20,20), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
            
            texto_palabras = " → ".join(self.palabras) if self.palabras else "Muestra señas con tus manos"
            cv2.putText(frame, texto_palabras, (10, h-42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            if self.palabras:
                oracion_preview = generar_oracion(self.palabras)
                cv2.putText(frame, oracion_preview, (10, h-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            
            # Controles (discreto)
            blur_txt = "ON" if self.blur_activo else "OFF"
            cv2.putText(frame, f"[Q] Salir  [C] Limpiar  [D] Debug  [B] Blur:{blur_txt}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,150), 1)

            # Notificación de modelo recargado (3 segundos)
            if ahora - self._notif_recarga < 3:
                cv2.putText(frame, "🔄 Modelo actualizado", (w//2 - 120, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)
            
            # Info de Audio (arriba a la derecha debajo de las manos)
            cv2.putText(frame, f"Salida: {self.audio_dispositivo}", (w-200, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

            self.guardar_estado_remoto(ahora, hay_mano_valida)
            self.guardar_frame_remoto(frame, ahora)
            
            cv2.imshow(TITULO_VENTANA, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.palabras = []
                self.ultima_sena = None
                self.buffer.clear()
                self.ultimo_features = None
                self.buffer_listo = False
                self.tiempo_primera_mano = None
                self.ultimo_evento = "Subtítulos limpiados"
                print("🗑️ Limpiado")
            elif key == ord('d'):
                mostrar_debug = not mostrar_debug
                print(f"🔧 Debug: {'ON' if mostrar_debug else 'OFF'}")
            elif key == ord('b'):
                self.blur_activo = not self.blur_activo
                print(f"🎨 Blur: {'ON' if self.blur_activo else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        try:
            with open(ESTADO_TRADUCTOR, "w", encoding="utf-8") as f:
                json.dump({"activo": False, "timestamp": time.time(), "evento": "Traductor cerrado"}, f)
        except Exception:
            pass
        print("\n👋 Traductor cerrado correctamente\n")


if __name__ == "__main__":
    TraductorLSE().ejecutar()
