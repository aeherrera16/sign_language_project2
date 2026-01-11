#!/usr/bin/env python3
"""
TRADUCTOR LSE - Con generación de oraciones naturales
Detecta fin de frase cuando bajas las manos.
Convierte palabras clave en oraciones con sentido.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
import pickle
import time
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# TTS - Configuración para español
try:
    import pyttsx3
    tts = pyttsx3.init()
    tts.setProperty('rate', 140)  # Velocidad más lenta para mejor claridad
    
    # Buscar voz en español (varias variantes)
    voces_espanol = ['spanish', 'español', 'es_', 'es-', 'monica', 'jorge', 'paulina', 'diego']
    voz_encontrada = False
    
    for voice in tts.getProperty('voices'):
        voice_lower = voice.name.lower()
        for esp in voces_espanol:
            if esp in voice_lower:
                tts.setProperty('voice', voice.id)
                voz_encontrada = True
                print(f"🔊 Voz TTS: {voice.name}")
                break
        if voz_encontrada:
            break
    
    if not voz_encontrada:
        print("⚠️ No se encontró voz en español. Usando voz por defecto.")
    
    TTS_OK = True
except Exception as e:
    TTS_OK = False
    print(f"⚠️ TTS no disponible: {e}")

# === CONFIGURACIÓN ===
DIR_MODELO = os.path.join(os.path.dirname(__file__), "modelo")
FRAMES = 30
FEATURES = 126
UMBRAL_CONFIANZA = 0.80  # Aumentado de 0.65 a 0.80
COOLDOWN = 2.5           # Aumentado de 1.5 a 2.5 segundos
TIEMPO_SIN_MANOS_PARA_FIN = 2.0
CONFIRMACIONES_REQUERIDAS = 2  # Debe detectar la misma seña 2 veces

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

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
    "BANCO": "el banco",
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
        # Para confirmación
        self.sena_candidata = None
        self.confirmaciones = 0
        
    def cargar_modelo(self):
        modelo_path = os.path.join(DIR_MODELO, "modelo.h5")
        if not os.path.exists(modelo_path):
            print("\n❌ No hay modelo. Ejecuta:")
            print("   python 2_entrenar_modelo.py")
            return False
        
        self.modelo = tf.keras.models.load_model(modelo_path)
        with open(os.path.join(DIR_MODELO, "encoder.pkl"), 'rb') as f:
            self.encoder = pickle.load(f)
        
        print(f"✅ Modelo cargado: {list(self.encoder.classes_)}")
        return True
    
    def extraer_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        features = np.zeros(FEATURES)
        num_manos = 0
        
        if result.multi_hand_landmarks:
            num_manos = len(result.multi_hand_landmarks)
            for idx, hand_lm in enumerate(result.multi_hand_landmarks[:2]):
                wrist = hand_lm.landmark[0]
                for i, lm in enumerate(hand_lm.landmark):
                    base = idx * 63 + i * 3
                    features[base] = lm.x - wrist.x
                    features[base + 1] = lm.y - wrist.y
                    features[base + 2] = lm.z - wrist.z
        
        return features, result, num_manos
    
    def predecir(self):
        if len(self.buffer) < FRAMES:
            return None, 0.0
        
        seq = np.array(list(self.buffer))
        pred = self.modelo.predict(np.expand_dims(seq, 0), verbose=0)[0]
        idx = np.argmax(pred)
        
        return self.encoder.inverse_transform([idx])[0], pred[idx]
    
    def hablar(self, texto):
        if TTS_OK and texto:
            print(f"\n🔊 HABLANDO: {texto}")
            tts.say(texto)
            tts.runAndWait()
    
    def ejecutar(self):
        if not self.cargar_modelo():
            return
        
        print("\n" + "="*60)
        print("  TRADUCTOR LSE - ORACIONES NATURALES")
        print("="*60)
        print("\nInstrucciones:")
        print("  1. Haz señas frente a la cámara")
        print("  2. BAJA las manos cuando termines la frase")
        print("  3. El sistema convertirá a oración y hablará")
        print("\n  [C] Limpiar | [Q] Salir")
        print("-"*60)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            features, result, num_manos = self.extraer_landmarks(frame)
            ahora = time.time()
            
            # Dibujar manos
            hay_mano = num_manos > 0
            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                        mp_draw.DrawingSpec(color=(0,200,0), thickness=2)
                    )
            
            # === LÓGICA DE DETECCIÓN ===
            if hay_mano:
                self.ultimo_tiempo_con_mano = ahora
                self.buffer.append(features)
                
                # Intentar predecir
                if len(self.buffer) >= FRAMES and ahora - self.ultima_deteccion > COOLDOWN:
                    sena, conf = self.predecir()
                    
                    if sena and conf >= UMBRAL_CONFIANZA:
                        if sena != self.ultima_sena:
                            self.palabras.append(sena)
                            self.ultima_sena = sena
                            self.ultima_deteccion = ahora
                            print(f"  ✓ {sena} ({conf:.0%})")
            
            else:
                # Sin manos - verificar si es fin de frase
                tiempo_sin_manos = ahora - self.ultimo_tiempo_con_mano
                
                if tiempo_sin_manos > TIEMPO_SIN_MANOS_PARA_FIN and self.palabras:
                    # FIN DE FRASE - generar oración y hablar
                    oracion = generar_oracion(self.palabras)
                    self.hablar(oracion)
                    
                    # Resetear
                    self.palabras = []
                    self.ultima_sena = None
                    self.buffer.clear()
            
            # === UI ===
            h, w = frame.shape[:2]
            
            # Header
            cv2.rectangle(frame, (0, 0), (w, 50), (40,40,40), -1)
            cv2.putText(frame, "TRADUCTOR LSE", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            # Estado
            if hay_mano:
                cv2.circle(frame, (w-30, 25), 15, (0,255,0), -1)
            else:
                cv2.circle(frame, (w-30, 25), 15, (0,0,255), -1)
                if self.palabras:
                    tiempo_restante = TIEMPO_SIN_MANOS_PARA_FIN - (ahora - self.ultimo_tiempo_con_mano)
                    if tiempo_restante > 0:
                        cv2.putText(frame, f"Fin en {tiempo_restante:.1f}s", (w-150, 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)
            
            # Palabras detectadas
            cv2.rectangle(frame, (0, h-80), (w, h-40), (50,50,50), -1)
            texto_palabras = " → ".join(self.palabras) if self.palabras else "[Esperando señas...]"
            cv2.putText(frame, f"Palabras: {texto_palabras}", (10, h-55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            # Preview de oración
            cv2.rectangle(frame, (0, h-40), (w, h), (30,30,30), -1)
            if self.palabras:
                oracion_preview = generar_oracion(self.palabras)
                cv2.putText(frame, f"Oracion: {oracion_preview}", (10, h-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            else:
                cv2.putText(frame, "Baja las manos para terminar la frase", (10, h-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
            
            # Instrucción de salida visible
            cv2.putText(frame, "Presiona Q para SALIR", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255), 1)
            
            cv2.imshow('Traductor LSE', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.palabras = []
                self.ultima_sena = None
                self.buffer.clear()
                print("🗑️ Limpiado")
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\n👋 Traductor cerrado correctamente\n")


if __name__ == "__main__":
    TraductorLSE().ejecutar()
