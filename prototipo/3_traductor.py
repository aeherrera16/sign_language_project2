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
from collections import deque

# Importar MediaPipe y TensorFlow SIN warnings
from utils_silenciar import init_mediapipe, init_tensorflow
mp, mp_hands, mp_draw, hands = init_mediapipe(max_hands=2, detection_conf=0.5, tracking_conf=0.5)
tf = init_tensorflow()

TTS_OK = True

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

def hablar_texto(texto):
    if not texto: return
    print(f"\n🔊 HABLANDO: {texto}")
    import subprocess
    try:
        if sys.platform == 'darwin':
            # macOS: 'say' viene instalado por defecto, voz española
            subprocess.run(['say', '-v', 'Monica', '-r', str(TTS_VELOCIDAD), texto],
                           stderr=subprocess.DEVNULL)
        else:
            # Linux / Raspberry Pi: más lento y claro para conversación presencial.
            subprocess.run(['espeak', '-v', 'es+f3', '-s', str(TTS_VELOCIDAD),
                            '-a', str(TTS_VOLUMEN), texto],
                          stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️ Error de voz: {e}")


# === CONFIGURACIÓN ===
DIR_MODELO = os.path.join(os.path.dirname(__file__), "modelo")
RELOAD_FLAG = os.path.join(DIR_MODELO, ".reload_model")  # Escrito por modo_traductor.py cuando hay nuevo modelo
ESTADO_TRADUCTOR = os.path.join(os.path.dirname(__file__), ".traductor_estado.json")
FRAME_TRADUCTOR = os.path.join(os.path.dirname(__file__), ".traductor_frame.jpg")
FRAMES = 30
FEATURES = 126
UMBRAL_CONFIANZA = _env_float("LSE_UMBRAL_CONFIANZA", 0.82)
MARGEN_MINIMO = _env_float("LSE_MARGEN_MINIMO", 0.25)
CLASES_SILENCIO = {"NINGUNA", "NONE", "SILENCIO"}  # Se detectan pero no se hablan ni muestran
COOLDOWN = _env_float("LSE_COOLDOWN", 0.9)
TIEMPO_SIN_MANOS_PARA_FIN = _env_float("LSE_TIEMPO_FIN_FRASE", 1.4)
CONFIRMACIONES_REQUERIDAS = _env_int("LSE_CONFIRMACIONES", 2)
UMBRAL_ESTABILIDAD = 0.05     # Más tolerante al movimiento
TIEMPO_ESTABLE_REQUERIDO = _env_float("LSE_TIEMPO_ESTABLE", 0.2)
TTS_VELOCIDAD = _env_int("LSE_TTS_VELOCIDAD", 105)
TTS_VOLUMEN = _env_int("LSE_TTS_VOLUMEN", 180)
INTERVALO_FRAME_REMOTO = _env_float("LSE_INTERVALO_FRAME", 0.25)

# === FILTRO DE POSICIÓN NATURAL ===
UMBRAL_MANO_CAIDA_Y = 0.70
UMBRAL_MOVIMIENTO_DEDOS = 0.015

# === DIFUMINADO DE FONDO ===
BLUR_STRENGTH = 21          # Reducido de 35→21 para rendimiento en RPi (debe ser impar)
BLUR_ACTIVO_DEFAULT = True  # Tecla [B] para alternar en tiempo real

# === SISTEMA DE GENERACIÓN DE ORACIONES (100% OFFLINE) ===
# Vocabulario y patrones basados en noticias ecuatorianas

# Vocabulario: palabra de seña → texto natural
VOCABULARIO = {
    # === PERSONAS Y CARGOS ===
    "PRESIDENTE": "el presidente",
    "NOBOA": "Noboa",
    "CORREA": "Correa",
    "GOBIERNO": "el gobierno",
    "MINISTRO": "el ministro",
    "JUEZ": "el juez",
    "ALCALDE": "el alcalde",
    "CONCEJAL": "el concejal",
    "LIDER": "el líder",
    "CANDIDATO": "el candidato",
    "PERSONA": "la persona",
    "CIUDADANO": "el ciudadano",
    "FAMILIA": "la familia",
    "NIÑO": "el niño",
    "MUJER": "la mujer",
    "HOMBRE": "el hombre",
    "MADRE": "la madre",
    "PADRE": "el padre",
    "CANTANTE": "el cantante",
    "JUGADOR": "el jugador",
    "TECNICO": "el técnico",
    
    # === INSTITUCIONES ===
    "CORTE": "la Corte",
    "CONSTITUCIONAL": "Constitucional",
    "BANCO": "el banco",
    "BANECUADOR": "BanEcuador",
    "EMPRESA": "la empresa",
    "POLICIA": "la policía",
    "EJERCITO": "el ejército",
    "ASAMBLEA": "la Asamblea",
    "TRIBUNAL": "el tribunal",
    "HOSPITAL": "el hospital",
    "ESCUELA": "la escuela",
    "UNIVERSIDAD": "la universidad",
    
    # === LUGARES ===
    "ECUADOR": "Ecuador",
    "PAIS": "el país",
    "QUITO": "Quito",
    "GUAYAQUIL": "Guayaquil",
    "MANABI": "Manabí",
    "ESMERALDAS": "Esmeraldas",
    "CUENCA": "Cuenca",
    "COLOMBIA": "Colombia",
    "ESTADOS_UNIDOS": "Estados Unidos",
    "EEUU": "Estados Unidos",
    "VENEZUELA": "Venezuela",
    "SIRIA": "Siria",
    "HONDURAS": "Honduras",
    "CIUDAD": "la ciudad",
    "CALLE": "la calle",
    
    # === ECONOMÍA ===
    "DINERO": "el dinero",
    "DOLAR": "dólares",
    "DOLARES": "dólares",
    "MILLON": "millón",
    "MILLONES": "millones",
    "CREDITO": "el crédito",
    "PRESTAMO": "el préstamo",
    "UTILIDAD": "la utilidad",
    "GANANCIA": "la ganancia",
    "ECONOMIA": "la economía",
    "INVERSION": "la inversión",
    
    # === ENERGÍA ===
    "ENERGIA": "la energía",
    "ELECTRICIDAD": "la electricidad",
    "ELECTRICO": "eléctrico",
    "APAGON": "apagón",
    "LUZ": "la luz",
    "GENERADOR": "el generador",
    "EMBALSE": "el embalse",
    
    # === POLÍTICA ===
    "FALLO": "el fallo",
    "DECISION": "la decisión",
    "LEY": "la ley",
    "VOTO": "el voto",
    "ELECCION": "la elección",
    "PROTESTA": "la protesta",
    "CONCESION": "la concesión",
    "SECTOR": "el sector",
    
    # === SUCESOS ===
    "DETENIDO": "fue detenido",
    "ARRESTADO": "fue arrestado",
    "ASESINADO": "fue asesinado",
    "MUERTO": "murió",
    "ACCIDENTE": "accidente",
    "MASACRE": "masacre",
    "CRIMEN": "crimen",
    "DROGA": "droga",
    "DEPORTADO": "deportado",
    
    # === DEPORTES ===
    "FUTBOL": "fútbol",
    "EQUIPO": "el equipo",
    "BARCELONA": "Barcelona",
    "EMELEC": "Emelec",
    "LIGA": "Liga",
    "PARTIDO": "el partido",
    "COPA": "la Copa",
    "MUNDIAL": "el Mundial",
    "GOL": "gol",
    "CAMPEON": "campeón",
    
    # === VERBOS ===
    "DECIR": "dijo",
    "DIJO": "dijo",
    "ANUNCIAR": "anunció",
    "ANUNCIO": "anunció",
    "CERRAR": "cerró",
    "CERRO": "cerró",
    "ABRIR": "abrió",
    "DETENER": "detuvo",
    "MORIR": "murió",
    "MURIO": "murió",
    "MATAR": "mató",
    "ATACAR": "atacó",
    "PROTESTAR": "protestó",
    "RECHAZAR": "rechazó",
    "APROBAR": "aprobó",
    "ENTREGAR": "entregó",
    "OTORGAR": "otorgó",
    "PEDIR": "pidió",
    "INICIAR": "inició",
    "TERMINAR": "terminó",
    "GANAR": "ganó",
    "PERDER": "perdió",
    "SUBIR": "subió",
    "BAJAR": "bajó",
    "AUMENTAR": "aumentó",
    "REDUCIR": "redujo",
    "MEJORAR": "mejoró",
    "EMPEORAR": "empeoró",
    "TENER": "tiene",
    "TIENE": "tiene",
    "SER": "es",
    "ESTAR": "está",
    "HABER": "hay",
    "HAY": "hay",
    
    # === ADJETIVOS ===
    "BUENO": "bueno",
    "MALO": "malo",
    "NUEVO": "nuevo",
    "GRANDE": "grande",
    "MUCHO": "mucho",
    "POCO": "poco",
    "MAS": "más",
    "MENOS": "menos",
    "PRIVADO": "privado",
    "PUBLICO": "público",
    
    # === TIEMPO ===
    "HOY": "hoy",
    "AYER": "ayer",
    "MAÑANA": "mañana",
    "AÑO": "año",
    "ANÑO": "año",
    "ANÑOS": "años",
    "MES": "mes",
    "SEMANA": "semana",
    "DIA": "día",
    "ENERO": "enero",
    "FEBRERO": "febrero",
    "MARZO": "marzo",
    "DICIEMBRE": "diciembre",
    
    # === OTROS ===
    "NOTICIA": "la noticia",
    "PROBLEMA": "el problema",
    "SOLUCION": "la solución",
    "SEGURIDAD": "la seguridad",
    "TRABAJO": "el trabajo",
    "EMPLEO": "el empleo",
    "POBREZA": "la pobreza",
    "SALUD": "la salud",
    "EDUCACION": "la educación",
    "CALOR": "el calor",
    "TEMPERATURA": "la temperatura",
}

# Patrones de frases completas
PATRONES = [
    # === PRESIDENTE Y GOBIERNO ===
    (["PRESIDENTE", "ECUADOR"], "El presidente de Ecuador"),
    (["PRESIDENTE", "NOBOA"], "El presidente Noboa"),
    (["PRESIDENTE", "DECIR"], "El presidente dijo que"),
    (["PRESIDENTE", "ANUNCIAR"], "El presidente anunció que"),
    (["GOBIERNO", "ECUADOR"], "El gobierno de Ecuador"),
    (["GOBIERNO", "ANUNCIAR"], "El gobierno anunció que"),
    (["GOBIERNO", "ENTREGAR"], "El gobierno entregó"),
    (["GOBIERNO", "INICIAR"], "El gobierno inició"),
    
    # === CORTE Y JUSTICIA ===
    (["CORTE", "CONSTITUCIONAL"], "La Corte Constitucional"),
    (["FALLO", "CORTE"], "El fallo de la Corte"),
    (["CORTE", "DECIR"], "La Corte dijo que"),
    (["JUEZ", "DECIR"], "El juez declaró que"),
    
    # === ECONOMÍA Y BANCOS ===
    (["BANECUADOR", "CERRAR"], "BanEcuador cerró"),
    (["BANCO", "OTORGAR"], "El banco otorgó"),
    (["ECONOMIA", "MEJORAR"], "La economía mejoró"),
    (["ECONOMIA", "EMPEORAR"], "La economía empeoró"),
    (["CREDITO", "AUMENTAR"], "Los créditos aumentaron"),
    (["TRABAJO", "SUBIR"], "El empleo aumentó"),
    (["TRABAJO", "BAJAR"], "El empleo disminuyó"),
    (["POBREZA", "BAJAR"], "La pobreza bajó"),
    (["POBREZA", "SUBIR"], "La pobreza subió"),
    
    # === ENERGÍA ===
    (["ENERGIA", "PROBLEMA"], "Hay problemas con la energía"),
    (["APAGON", "HOY"], "Hay apagones hoy"),
    (["LUZ", "CORTAR"], "Cortaron la luz"),
    (["EMPRESA", "ELECTRICO"], "La empresa eléctrica"),
    (["EMBALSE", "BAJAR"], "El embalse bajó"),
    
    # === INTERNACIONAL ===
    (["ESTADOS_UNIDOS", "PROTESTAR"], "Hay protestas en Estados Unidos"),
    (["EEUU", "PROTESTAR"], "Hay protestas en Estados Unidos"),
    (["EEUU", "DEPORTAR"], "Estados Unidos deportó"),
    (["COLOMBIA", "DETENER"], "En Colombia detuvieron"),
    (["VENEZUELA", "PROBLEMA"], "Hay problemas en Venezuela"),
    
    # === SUCESOS Y CRIMEN ===
    (["PERSONA", "DETENIDO"], "Una persona fue detenida"),
    (["PERSONA", "MUERTO"], "Una persona murió"),
    (["LIDER", "DETENIDO"], "El líder fue detenido"),
    (["MASACRE", "MANABI"], "Hubo una masacre en Manabí"),
    (["ACCIDENTE", "MUERTO"], "Murió en un accidente"),
    
    # === POLÍTICA ===
    (["ECUADOR", "RECHAZAR"], "Ecuador rechazó"),
    (["ASAMBLEA", "APROBAR"], "La Asamblea aprobó"),
    (["PROTESTA", "CALLE"], "Hay protestas en las calles"),
    (["ELECCION", "VOTO"], "En las elecciones votaron"),
    
    # === DEPORTES ===
    (["BARCELONA", "GANAR"], "Barcelona ganó"),
    (["BARCELONA", "PERDER"], "Barcelona perdió"),
    (["EMELEC", "GANAR"], "Emelec ganó"),
    (["EMELEC", "PERDER"], "Emelec perdió"),
    (["FUTBOL", "PARTIDO"], "En el partido de fútbol"),
    (["MUNDIAL", "FUTBOL"], "El Mundial de fútbol"),
    (["JUGADOR", "MORIR"], "El jugador murió"),
    (["TECNICO", "MORIR"], "El técnico murió"),
    
    # === CLIMA ===
    (["GUAYAQUIL", "CALOR"], "En Guayaquil hace calor"),
    (["TEMPERATURA", "SUBIR"], "La temperatura subió"),
    
    # === TIEMPO ===
    (["AÑO"], "este año"),
    (["HOY"], "hoy"),
    (["AYER"], "ayer"),
]


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


def filtrar_manos(result):
    """
    Filtra las manos detectadas:
    - Descarta manos en posición de reposo (caídas)
    - MediaPipe ya limita a 2 manos máximo
    Retorna: lista de índices válidos, info de debug
    """
    if not result.multi_hand_landmarks:
        return [], {"total": 0, "validas": 0, "razon": "sin_manos"}
    
    total = len(result.multi_hand_landmarks)
    validas = []
    
    for idx, hand_lm in enumerate(result.multi_hand_landmarks):
        if es_mano_en_reposo(hand_lm):
            continue
        validas.append(idx)
    
    return validas, {"total": total, "validas": len(validas), "razon": "ok" if validas else "reposo"}


def obtener_bbox_mano(hand_landmarks, h, w, margen=40):
    """Obtiene el bounding box de una mano con margen extra."""
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x1 = max(0, min(xs) - margen)
    y1 = max(0, min(ys) - margen)
    x2 = min(w, max(xs) + margen)
    y2 = min(h, max(ys) + margen)
    return x1, y1, x2, y2


def difuminar_fondo(frame, result, indices_validos, activo=True):
    """
    Difumina el fondo y deja las manos nítidas.
    - Sin manos o activo=False: devuelve frame original (sin coste de blur)
    - Con manos: solo blurrea cuando hay algo que mostrar
    """
    if not activo or not indices_validos:
        return frame

    h, w = frame.shape[:2]
    blurred = cv2.GaussianBlur(frame, (BLUR_STRENGTH, BLUR_STRENGTH), 0)
    resultado = blurred.copy()

    for idx in indices_validos:
        hand_lm = result.multi_hand_landmarks[idx]
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
        self.modelo = None
        self.encoder = None
        self.buffer = deque(maxlen=FRAMES)
        self.palabras = []
        self.ultima_deteccion = 0
        self.ultima_sena = None
        self.ultimo_tiempo_con_mano = time.time()
        self.frase_completa = False
        # Para confirmación de señas
        self.sena_candidata = None
        self.confirmaciones = 0
        # Para detección de estabilidad
        self.ultimo_features = None
        self.tiempo_estable_inicio = None
        self.mano_estable = False
        # Info de filtrado para debug en pantalla
        self.filtro_info = {"total": 0, "validas": 0, "rechazadas": [], "razon": ""}
        # Audio activo detectado
        self.audio_dispositivo = obtener_dispositivo_audio()
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
    
    def extraer_landmarks_filtrado(self, frame, result, indices_validos):
        """
        Extrae landmarks SOLO de las manos válidas (ya filtradas).
        Las manos fuera de la zona activa o en posición natural son ignoradas.
        """
        features = np.zeros(FEATURES)
        
        if not result.multi_hand_landmarks or not indices_validos:
            return features

        validos = indices_validos[:2]

        # Señas con 2 manos: ordenar por lateralidad (Left→slot0, Right→slot1)
        # Debe ser idéntico al grabador para que el modelo vea los mismos datos
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

        return features
    
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
            pred = self.modelo.predict(np.expand_dims(seq, 0), verbose=0)[0]

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
            import threading
            # Habla en un hilo aparte para no congelar tu video
            threading.Thread(target=hablar_texto, args=(texto,), daemon=True).start()

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
            "mano_estable": bool(self.mano_estable),
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
        
        cap = cv2.VideoCapture(0)
        # Resolución optimizada para FPS ultra-rápidos en Raspberry Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            raise RuntimeError(
                "No se pudo abrir la cámara.\n"
                "Verifica que la cámara esté conectada y no esté siendo usada por otra aplicación."
            )
        
        cv2.namedWindow('Traductor LSE', cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty('Traductor LSE', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass  # No hay gestor de ventanas disponible (modo headless)
        
        mostrar_debug = False
        self.ultimo_evento = "Traductor activo"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ahora = time.time()

            # Chequear si hay un modelo nuevo listo (no interrumpe el video)
            self.recargar_modelo_si_hay_nuevo()
            
            # Procesar con MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            # === FILTRADO DE MANOS ===
            indices_validos, self.filtro_info = filtrar_manos(result)
            hay_mano_valida = len(indices_validos) > 0
            
            # Extraer features solo de manos válidas
            features = self.extraer_landmarks_filtrado(frame, result, indices_validos)
            
            # === DIFUMINAR FONDO (solo cuando hay manos y blur activo) ===
            frame = difuminar_fondo(frame, result, indices_validos, activo=self.blur_activo)
            
            # Dibujar landmarks de manos válidas (VERDE)
            if result.multi_hand_landmarks:
                for idx in indices_validos:
                    hand_lm = result.multi_hand_landmarks[idx]
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                        mp_draw.DrawingSpec(color=(0,200,0), thickness=2)
                    )
            
            # === LÓGICA DE DETECCIÓN ===
            if hay_mano_valida:
                self.ultimo_tiempo_con_mano = ahora
                
                # Verificar estabilidad
                if self.ultimo_features is not None:
                    movimiento = np.mean(np.abs(features - self.ultimo_features))
                    if movimiento < UMBRAL_ESTABILIDAD:
                        if self.tiempo_estable_inicio is None:
                            self.tiempo_estable_inicio = ahora
                        elif ahora - self.tiempo_estable_inicio >= TIEMPO_ESTABLE_REQUERIDO:
                            self.mano_estable = True
                    else:
                        self.tiempo_estable_inicio = None
                        self.mano_estable = False
                        self.sena_candidata = None
                        self.confirmaciones = 0
                
                self.ultimo_features = features.copy()
                self.buffer.append(features)
                
                # Predecir solo si mano estable
                if self.mano_estable and len(self.buffer) >= FRAMES and ahora - self.ultima_deteccion > COOLDOWN:
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
                            elif sena != self.ultima_sena:
                                self.palabras.append(sena)
                                self.ultima_sena = sena
                                self.ultima_deteccion = ahora
                                self.ultima_confianza = float(conf)
                                self.ultimo_evento = f"Seña detectada: {sena} ({conf:.0%})"
                                print(f"  ✓ {sena} ({conf:.0%}) [Confirmado {self.confirmaciones}x]")
                            self.sena_candidata = None
                            self.confirmaciones = 0
            
            else:
                # Sin manos - verificar fin de frase
                tiempo_sin_manos = ahora - self.ultimo_tiempo_con_mano

                # KEY FIX: limpiar buffer rápido para evitar predicciones
                # con frames viejos cuando las manos vuelven a aparecer brevemente
                if tiempo_sin_manos > 0.3 and self.ultimo_features is not None:
                    self.buffer.clear()
                    self.ultimo_features = None
                    self.tiempo_estable_inicio = None
                    self.mano_estable = False
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
                    self.tiempo_estable_inicio = None
                    self.mano_estable = False
            
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
                if self.mano_estable:
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
                cv2.putText(frame, f"Estable: {self.mano_estable}", (10, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if self.mano_estable else (0,0,255), 1)
            
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
            
            cv2.imshow('Traductor LSE', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.palabras = []
                self.ultima_sena = None
                self.buffer.clear()
                self.ultimo_features = None
                self.tiempo_estable_inicio = None
                self.mano_estable = False
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
        hands.close()
        try:
            with open(ESTADO_TRADUCTOR, "w", encoding="utf-8") as f:
                json.dump({"activo": False, "timestamp": time.time(), "evento": "Traductor cerrado"}, f)
        except Exception:
            pass
        print("\n👋 Traductor cerrado correctamente\n")


if __name__ == "__main__":
    TraductorLSE().ejecutar()
