#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🤟 TRADUCTOR LSE - MODO AUTÓNOMO PARA RASPBERRY PI 4
═══════════════════════════════════════════════════════════════════════════════

Aplicación 100% autónoma que:
- Inicia automáticamente al encender la Raspberry Pi
- Enciende la cámara inmediatamente
- Comienza a traducir sin intervención del usuario
- Anuncia por voz lo que detecta
- No requiere teclado, mouse ni monitor

IDEAL PARA:
- Dispositivos dedicados tipo "appliance"
- Uso por personas con discapacidad auditiva
- Funcionamiento offline completo

═══════════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import subprocess
import time
import sys
import signal
import logging
from pathlib import Path
from collections import deque
from datetime import datetime
import threading
import os

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════════

# Configurar logging para archivos
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"traductor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rutas del modelo
SCRIPT_DIR = Path(__file__).parent
MODEL_PATHS = [
    SCRIPT_DIR / "model",                 # Modelo empaquetado junto al script
    SCRIPT_DIR.parent / "backend" / "model",  # Ubicación del proyecto principal
    Path("/home/pi/traductor-lse/model"),     # Ubicación de instalación estándar
]

# Configuración de reconocimiento
CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.60,      # Umbral de confianza
    'STABILITY_FRAMES': 4,             # Frames consecutivos para confirmar
    'SILENCE_TIMEOUT': 2.5,            # Segundos sin gestos para sintetizar frase
    'COOLDOWN_TIME': 1.5,              # Segundos antes de reconocer mismo gesto
    'CAMERA_RETRY_SECONDS': 5,         # Reintentar cámara cada X segundos
    'MAX_CAMERA_RETRIES': 20,          # Máximo intentos de conexión de cámara
    'FRAME_WIDTH': 640,
    'FRAME_HEIGHT': 480,
    'ANNOUNCE_ON_START': True,         # Anunciar inicio por voz
    'ANNOUNCE_GESTURES': True,         # Anunciar cada gesto detectado
    'SHOW_DISPLAY': False,             # Para Raspberry Pi sin monitor: False
}

# Colores para visualización (si hay display)
COLORS = {
    'GREEN': (0, 255, 0),
    'RED': (0, 0, 255),
    'BLUE': (255, 0, 0),
    'YELLOW': (0, 255, 255),
    'PURPLE': (255, 0, 255),
    'WHITE': (255, 255, 255),
}


# ════════════════════════════════════════════════════════════════════════════════
# CLASE TTS (Text-to-Speech) - MEJORADA
# ════════════════════════════════════════════════════════════════════════════════

class TextToSpeech:
    """Sintetizador de voz con soporte para Raspberry Pi/Linux/macOS"""
    
    def __init__(self, rate: int = 150):
        self.rate = rate
        self.is_macos = sys.platform == 'darwin'
        self.is_linux = sys.platform.startswith('linux')
        self.lock = threading.Lock()
        self._test_audio()
    
    def _test_audio(self):
        """Prueba que el sistema de audio funcione"""
        if self.is_linux:
            # Intentar configurar audio en Raspberry Pi
            try:
                subprocess.run(['amixer', 'set', 'Master', '90%'], 
                             capture_output=True, timeout=5)
            except:
                pass
    
    def speak(self, text: str, block: bool = False):
        """Reproduce texto como voz"""
        if not text:
            return
        
        with self.lock:
            try:
                if self.is_macos:
                    # macOS: usar comando 'say'
                    voices = ['Mónica', 'Paulina', 'Juan']
                    for voice in voices:
                        result = subprocess.run(
                            ['say', '-v', voice, '-r', str(self.rate), text],
                            capture_output=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            break
                    else:
                        # Fallback sin voz específica
                        subprocess.run(['say', text], capture_output=True)
                else:
                    # Linux/Raspberry Pi: usar espeak
                    cmd = ['espeak', '-ves', '-s', str(self.rate), text]
                    if block:
                        subprocess.run(cmd, capture_output=True, timeout=30)
                    else:
                        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                        
            except subprocess.TimeoutExpired:
                logger.warning("TTS timeout")
            except Exception as e:
                logger.error(f"Error TTS: {e}")
    
    def speak_async(self, text: str):
        """Habla en segundo plano"""
        thread = threading.Thread(target=self.speak, args=(text,))
        thread.daemon = True
        thread.start()


# ════════════════════════════════════════════════════════════════════════════════
# CLASE RECONOCEDOR DE GESTOS
# ════════════════════════════════════════════════════════════════════════════════

class GestureRecognizer:
    """
    Reconocedor de gestos idéntico al backend web.
    Soporta tanto TFLite (más eficiente) como H5 (fallback).
    """
    
    def __init__(self):
        self.model = None
        self.labels = []
        self.loaded = False
        self.use_tflite = False
        self.interpreter = None
        
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def find_model_dir(self) -> Path:
        """Busca el directorio del modelo"""
        for path in MODEL_PATHS:
            if path.exists() and (path / "labels.pkl").exists():
                logger.info(f"📂 Modelo encontrado en: {path}")
                return path
        return None
    
    def load_model(self) -> bool:
        """Carga el modelo de reconocimiento"""
        try:
            model_dir = self.find_model_dir()
            if not model_dir:
                logger.error("❌ No se encontró el directorio del modelo")
                return False
            
            labels_path = model_dir / "labels.pkl"
            tflite_path = model_dir / "model.tflite"
            h5_path = model_dir / "best_model.h5"
            
            # Cargar etiquetas
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
            
            # Intentar TFLite primero (más eficiente para RPi)
            if tflite_path.exists():
                try:
                    # Intentar tflite_runtime primero
                    try:
                        import tflite_runtime.interpreter as tflite
                        logger.info("📦 Usando tflite_runtime")
                    except ImportError:
                        import tensorflow as tf
                        tflite = tf.lite
                        logger.info("📦 Usando tf.lite")
                    
                    self.interpreter = tflite.Interpreter(model_path=str(tflite_path))
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    self.use_tflite = True
                    logger.info("✅ Modelo TFLite cargado")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error cargando TFLite: {e}, intentando H5...")
                    self.use_tflite = False
            
            # Fallback a H5
            if not self.use_tflite and h5_path.exists():
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(h5_path))
                logger.info("✅ Modelo H5 cargado")
            
            if not self.use_tflite and self.model is None:
                logger.error("❌ No se pudo cargar ningún modelo")
                return False
            
            self.loaded = True
            logger.info(f"✅ {len(self.labels)} señas disponibles: {', '.join(self.labels)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            return False
    
    def detect_hands(self, frame) -> dict:
        """Detecta manos en el frame"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            return {'num_hands': 0, 'hands': []}
        
        hands_info = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([int(lm.x * w), int(lm.y * h), lm.z])
            
            landmarks = np.array(landmarks)
            handedness = None
            confidence = 0.5
            
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label
                confidence = results.multi_handedness[idx].classification[0].score
            
            hands_info.append({
                'landmarks': landmarks,
                'handedness': handedness,
                'confidence': confidence
            })
        
        return {'num_hands': len(hands_info), 'hands': hands_info}
    
    def predict(self, hands_info: dict) -> tuple:
        """Predice el gesto"""
        if not self.loaded or hands_info['num_hands'] == 0:
            return None, 0.0
        
        try:
            # Extraer y aplanar landmarks
            landmarks_list = [h['landmarks'].flatten() for h in hands_info['hands']]
            
            # Concatenar o rellenar con ceros
            if len(landmarks_list) >= 2:
                input_vector = np.concatenate(landmarks_list[:2])
            else:
                input_vector = np.concatenate([landmarks_list[0], np.zeros(63)])
            
            # Asegurar 126 features
            if input_vector.shape[0] != 126:
                input_vector = np.pad(input_vector, (0, max(0, 126 - input_vector.shape[0])))[:126]
            
            input_data = np.expand_dims(input_vector, axis=0).astype(np.float32)
            
            # Predecir
            if self.use_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                prediction = self.model.predict(input_data, verbose=0)[0]
            
            best_idx = np.argmax(prediction)
            confidence = float(prediction[best_idx])
            gesture = self.labels[best_idx] if best_idx < len(self.labels) else None
            
            return gesture, confidence
            
        except Exception as e:
            logger.error(f"Error predicción: {e}")
            return None, 0.0
    
    def draw_landmarks(self, frame, hands_info: dict):
        """Dibuja landmarks si hay display"""
        for hand in hands_info.get('hands', []):
            landmarks = hand['landmarks']
            
            for point in landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 5, COLORS['GREEN'], -1)
            
            connections = [
                [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12],
                [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [0, 17]
            ]
            
            for conn in connections:
                for i in range(len(conn) - 1):
                    pt1 = (int(landmarks[conn[i]][0]), int(landmarks[conn[i]][1]))
                    pt2 = (int(landmarks[conn[i+1]][0]), int(landmarks[conn[i+1]][1]))
                    cv2.line(frame, pt1, pt2, COLORS['BLUE'], 2)


# ════════════════════════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL - MODO AUTÓNOMO
# ════════════════════════════════════════════════════════════════════════════════

class TraductorAutonomo:
    """
    Aplicación principal que funciona de forma completamente autónoma.
    Diseñada para arrancar automáticamente sin intervención del usuario.
    """
    
    def __init__(self):
        self.running = True
        self.recognizer = GestureRecognizer()
        self.tts = TextToSpeech(rate=170)
        
        # Estado
        self.current_gesture = None
        self.stability_count = 0
        self.last_spoken = None
        self.last_spoken_time = 0
        self.gesture_buffer = []
        self.last_gesture_time = time.time()
        
        # Estadísticas
        self.start_time = time.time()
        self.gestures_detected = 0
        self.fps_history = deque(maxlen=30)
        
        # Manejar señales para cierre limpio
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Maneja señales de terminación"""
        logger.info(f"📴 Señal recibida ({signum}), cerrando...")
        self.running = False
    
    def iniciar(self) -> bool:
        """Inicializa todos los componentes"""
        logger.info("═" * 60)
        logger.info("🤟 TRADUCTOR LSE - MODO AUTÓNOMO")
        logger.info("═" * 60)
        logger.info(f"📅 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📝 Log: {LOG_FILE}")
        
        # Anunciar inicio
        if CONFIG['ANNOUNCE_ON_START']:
            self.tts.speak("Iniciando traductor de lengua de señas", block=True)
        
        # Cargar modelo
        if not self.recognizer.load_model():
            self.tts.speak("Error: No se pudo cargar el modelo de reconocimiento")
            return False
        
        # Anunciar listo
        if CONFIG['ANNOUNCE_ON_START']:
            self.tts.speak("Traductor listo. Muestre una seña frente a la cámara.", block=True)
        
        return True
    
    def conectar_camara(self) -> cv2.VideoCapture:
        """Conecta a la cámara con reintentos automáticos"""
        for intento in range(CONFIG['MAX_CAMERA_RETRIES']):
            logger.info(f"📷 Intentando conectar a cámara (intento {intento + 1}/{CONFIG['MAX_CAMERA_RETRIES']})")
            
            # Probar diferentes índices de cámara
            for cam_id in [0, 1, 2]:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Configurar cámara
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['FRAME_WIDTH'])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['FRAME_HEIGHT'])
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir latencia
                        
                        logger.info(f"✅ Cámara {cam_id} conectada")
                        return cap
                    cap.release()
            
            if intento < CONFIG['MAX_CAMERA_RETRIES'] - 1:
                logger.warning(f"⏳ Reintentando en {CONFIG['CAMERA_RETRY_SECONDS']} segundos...")
                time.sleep(CONFIG['CAMERA_RETRY_SECONDS'])
        
        logger.error("❌ No se pudo conectar a ninguna cámara")
        self.tts.speak("Error: No se detectó cámara. Verifique la conexión.")
        return None
    
    def ejecutar(self):
        """Bucle principal de ejecución"""
        cap = self.conectar_camara()
        if cap is None:
            return
        
        logger.info("🎥 Traducción en tiempo real iniciada")
        last_frame_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️ Frame perdido, reintentando...")
                    time.sleep(0.1)
                    continue
                
                # Voltear horizontalmente (espejo)
                frame = cv2.flip(frame, 1)
                
                # Procesar frame
                self._procesar_frame(frame)
                
                # Calcular FPS
                current_time = time.time()
                fps = 1.0 / max(current_time - last_frame_time, 0.001)
                self.fps_history.append(fps)
                last_frame_time = current_time
                
                # Verificar timeout para sintetizar frase
                if (len(self.gesture_buffer) > 0 and 
                    current_time - self.last_gesture_time > CONFIG['SILENCE_TIMEOUT']):
                    self._sintetizar_frase()
                
                # Mostrar visualización (si está habilitado)
                if CONFIG['SHOW_DISPLAY']:
                    avg_fps = sum(self.fps_history) / max(len(self.fps_history), 1)
                    self._dibujar_ui(frame, avg_fps)
                    cv2.imshow('Traductor LSE', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Limitar FPS para no sobrecargar CPU
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"❌ Error en bucle principal: {e}")
        finally:
            cap.release()
            if CONFIG['SHOW_DISPLAY']:
                cv2.destroyAllWindows()
            self._mostrar_estadisticas()
            self.tts.speak("Traductor detenido")
    
    def _procesar_frame(self, frame):
        """Procesa un frame para reconocimiento"""
        hands_info = self.recognizer.detect_hands(frame)
        
        if CONFIG['SHOW_DISPLAY']:
            self.recognizer.draw_landmarks(frame, hands_info)
        
        if hands_info['num_hands'] == 0:
            self.stability_count = 0
            self.current_gesture = None
            return
        
        gesture, confidence = self.recognizer.predict(hands_info)
        
        if gesture and confidence >= CONFIG['CONFIDENCE_THRESHOLD']:
            if gesture == self.current_gesture:
                self.stability_count += 1
            else:
                self.current_gesture = gesture
                self.stability_count = 1
            
            if self.stability_count >= CONFIG['STABILITY_FRAMES']:
                current_time = time.time()
                
                if (self.last_spoken != gesture or 
                    current_time - self.last_spoken_time > CONFIG['COOLDOWN_TIME']):
                    
                    self.last_spoken = gesture
                    self.last_spoken_time = current_time
                    self.last_gesture_time = current_time
                    self.gesture_buffer.append(gesture)
                    self.gestures_detected += 1
                    
                    logger.info(f"✅ Detectado: {gesture} ({confidence*100:.1f}%)")
                    
                    if CONFIG['ANNOUNCE_GESTURES']:
                        self.tts.speak_async(gesture)
        else:
            self.stability_count = max(0, self.stability_count - 1)
    
    def _sintetizar_frase(self):
        """Sintetiza la frase acumulada"""
        if not self.gesture_buffer:
            return
        
        phrase = " ".join(self.gesture_buffer)
        phrase = phrase.capitalize() + "."
        
        logger.info(f"💬 Frase completa: {phrase}")
        self.tts.speak(phrase, block=True)
        
        self.gesture_buffer.clear()
    
    def _dibujar_ui(self, frame, fps):
        """Dibuja interfaz visual"""
        h, w = frame.shape[:2]
        
        # Fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h-80), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Info superior
        cv2.putText(frame, "TRADUCTOR LSE - MODO AUTONOMO", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['GREEN'], 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Detectados: {self.gestures_detected}", 
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['YELLOW'], 1)
        
        # Última detección
        if self.last_spoken:
            cv2.putText(frame, f"Ultima seña: {self.last_spoken}", (10, h-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['PURPLE'], 2)
        
        # Buffer
        if self.gesture_buffer:
            buffer_text = " ".join(self.gesture_buffer[-5:])
            cv2.putText(frame, f"Frase: {buffer_text}", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['WHITE'], 1)
    
    def _mostrar_estadisticas(self):
        """Muestra estadísticas finales"""
        duracion = time.time() - self.start_time
        logger.info("═" * 60)
        logger.info("📊 ESTADÍSTICAS DE SESIÓN")
        logger.info("═" * 60)
        logger.info(f"   Duración: {duracion/60:.1f} minutos")
        logger.info(f"   Gestos detectados: {self.gestures_detected}")
        if duracion > 0:
            logger.info(f"   Gestos/minuto: {self.gestures_detected/(duracion/60):.1f}")
        logger.info("═" * 60)


# ════════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════════

def main():
    """Punto de entrada principal"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   🤟  TRADUCTOR DE LENGUA DE SEÑAS ECUATORIANA                   ║
    ║       Versión Autónoma para Raspberry Pi                         ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Detectar si hay display disponible
    if os.environ.get('DISPLAY'):
        logger.info("🖥️ Display detectado, habilitando visualización")
        CONFIG['SHOW_DISPLAY'] = True
    else:
        logger.info("📟 Sin display, modo solo audio")
        CONFIG['SHOW_DISPLAY'] = False
    
    # Crear y ejecutar traductor
    traductor = TraductorAutonomo()
    
    if traductor.iniciar():
        traductor.ejecutar()
    else:
        logger.error("❌ No se pudo inicializar el traductor")
        sys.exit(1)


if __name__ == "__main__":
    main()
