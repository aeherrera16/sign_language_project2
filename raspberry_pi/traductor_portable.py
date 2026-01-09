#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🤟 TRADUCTOR PORTÁTIL DE LENGUA DE SEÑAS ECUATORIANA - RASPBERRY PI
═══════════════════════════════════════════════════════════════════════════════

Aplicación standalone para Raspberry Pi 4 que:
- Captura video de cámara USB
- Reconoce señas en tiempo real (IGUAL QUE LA VERSIÓN WEB)
- Reproduce traducciones por bocina con espeak/pyttsx3
- Funciona 100% offline

USO:
    python traductor_portable.py
    python traductor_portable.py --camera 0 --confidence 0.65 --no-display

═══════════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import argparse
import subprocess
import time
import sys
import logging
from pathlib import Path
from collections import deque
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas - Buscar modelo en ubicación relativa
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Intentar varias ubicaciones del modelo
MODEL_PATHS = [
    PROJECT_DIR / "backend" / "model",  # Cuando está en el proyecto principal
    SCRIPT_DIR / "model",                # Cuando se copia junto al script
    Path("model"),                       # Directorio actual
]

# Configuración de reconocimiento
CONFIDENCE_THRESHOLD = 0.65
STABILITY_FRAMES = 4  # Frames consecutivos para confirmar
SILENCE_TIMEOUT = 2.0  # Segundos sin gestos para auto-traducir
COOLDOWN_TIME = 1.5  # Segundos antes de reconocer el mismo gesto

# Colores para visualización
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_PURPLE = (255, 0, 255)

# ════════════════════════════════════════════════════════════════════════════════
# CLASE TTS (Text-to-Speech)
# ════════════════════════════════════════════════════════════════════════════════

class TextToSpeech:
    """Sintetizador de voz para macOS y Raspberry Pi"""
    
    def __init__(self, rate: int = 150):
        self.rate = rate
        self.engine = None
        self.is_macos = sys.platform == 'darwin'
        self.is_linux = sys.platform.startswith('linux')
        self._try_pyttsx3()
    
    def _try_pyttsx3(self):
        """Intenta inicializar pyttsx3"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            
            # Buscar voz en español
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower() or 'mónica' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            
            logger.info("✅ TTS: Usando pyttsx3")
        except:
            self.engine = None
            if self.is_macos:
                logger.info("ℹ️ TTS: Usando 'say' de macOS")
            else:
                logger.info("ℹ️ TTS: Usando espeak")
    
    def speak(self, text: str, block: bool = False):
        """Reproduce texto como voz"""
        if not text:
            return
        
        try:
            if self.engine and block:
                self.engine.say(text)
                self.engine.runAndWait()
            elif self.is_macos:
                # macOS: usar comando 'say' con voz en español
                subprocess.Popen(
                    ['say', '-v', 'Mónica', text],  # Mónica es la voz española en macOS
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Linux/Raspberry Pi: usar espeak
                subprocess.Popen(
                    ['espeak', '-ves', '-s', str(self.rate), text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
        except Exception as e:
            logger.error(f"Error TTS: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# CLASE RECONOCEDOR - IGUAL QUE EL BACKEND WEB
# ════════════════════════════════════════════════════════════════════════════════

class GestureRecognizer:
    """
    Reconocedor de gestos usando el MISMO método que el backend web.
    
    Diferencia clave: Usa coordenadas de PIXEL para landmarks (x * width, y * height)
    y concatena 2 manos = 126 features (63 por mano)
    """
    
    def __init__(self):
        self.model = None
        self.labels = []
        self.loaded = False
        self.model_dir = None
        
        # MediaPipe - MISMA configuración que backend
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def find_model_dir(self) -> Path:
        """Encuentra el directorio del modelo"""
        for path in MODEL_PATHS:
            if path.exists() and (path / "labels.pkl").exists():
                return path
        return None
    
    def load_model(self) -> bool:
        """Carga el modelo (TFLite o H5)"""
        try:
            self.model_dir = self.find_model_dir()
            if not self.model_dir:
                logger.error("❌ No se encontró directorio del modelo")
                return False
            
            logger.info(f"📂 Usando modelo de: {self.model_dir}")
            
            tflite_path = self.model_dir / "model.tflite"
            h5_path = self.model_dir / "best_model.h5"
            labels_path = self.model_dir / "labels.pkl"
            
            # Cargar labels
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
            
            # Intentar TFLite primero (más eficiente para RPi)
            if tflite_path.exists():
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    import tensorflow as tf
                    tflite = tf.lite
                
                self.interpreter = tflite.Interpreter(model_path=str(tflite_path))
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.use_tflite = True
                logger.info("✅ Usando modelo TFLite")
            elif h5_path.exists():
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(h5_path))
                self.use_tflite = False
                logger.info("✅ Usando modelo Keras (H5)")
            else:
                logger.error("❌ No se encontró modelo .tflite ni .h5")
                return False
            
            self.loaded = True
            logger.info(f"✅ Modelo cargado: {len(self.labels)} señas: {self.labels}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            return False
    
    def detect_hands(self, frame) -> dict:
        """
        Detecta manos y extrae landmarks.
        IGUAL que hand_segmentation_service.detect_hands()
        
        Returns:
            Dict con 'num_hands' y 'hands' (lista con landmarks en coords de pixel)
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            return {'num_hands': 0, 'hands': []}
        
        hands_info = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extraer landmarks en COORDENADAS DE PIXEL (igual que backend)
            landmarks = []
            for lm in hand_landmarks.landmark:
                # IMPORTANTE: x * width, y * height, z
                landmarks.append([int(lm.x * w), int(lm.y * h), lm.z])
            
            landmarks = np.array(landmarks)
            
            # Obtener tipo de mano y confianza
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
        
        return {
            'num_hands': len(hands_info),
            'hands': hands_info
        }
    
    def predict(self, hands_info: dict) -> tuple:
        """
        Predice el gesto EXACTAMENTE igual que el backend web.
        
        Método:
        1. Aplanar landmarks: [21, 3] -> [63]
        2. Si hay 2 manos: concatenar = [126]
        3. Si hay 1 mano: rellenar con ceros = [63 + 63 ceros] = [126]
        """
        if not self.loaded or hands_info['num_hands'] == 0:
            return None, 0.0
        
        try:
            # Extraer y aplanar landmarks (IGUAL que backend líneas 148-159)
            landmarks_list = [h['landmarks'].flatten() for h in hands_info['hands']]
            
            # Concatenar o rellenar con ceros
            if len(landmarks_list) >= 2:
                input_vector = np.concatenate(landmarks_list[:2])
            else:
                # UNA MANO: rellenar segunda con ceros (CLAVE!)
                input_vector = np.concatenate([landmarks_list[0], np.zeros(63)])
            
            # Asegurar 126 features
            if input_vector.shape[0] != 126:
                input_vector = np.pad(input_vector, (0, 126 - input_vector.shape[0]))
            
            # Preparar entrada
            input_data = np.expand_dims(input_vector, axis=0).astype(np.float32)
            
            # Predecir
            if self.use_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                prediction = self.model.predict(input_data, verbose=0)[0]
            
            # Obtener mejor predicción
            best_idx = np.argmax(prediction)
            confidence = float(prediction[best_idx])
            gesture = self.labels[best_idx] if best_idx < len(self.labels) else None
            
            return gesture, confidence
            
        except Exception as e:
            logger.error(f"Error predicción: {e}")
            return None, 0.0
    
    def draw_landmarks(self, frame, hands_info: dict):
        """Dibuja landmarks en el frame"""
        h, w = frame.shape[:2]
        
        for hand in hands_info.get('hands', []):
            landmarks = hand['landmarks']
            
            # Dibujar puntos
            for point in landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 5, COLOR_GREEN, -1)
            
            # Dibujar conexiones
            connections = [
                [0, 1, 2, 3, 4],      # Pulgar
                [0, 5, 6, 7, 8],      # Índice
                [5, 9, 10, 11, 12],   # Medio
                [9, 13, 14, 15, 16],  # Anular
                [13, 17, 18, 19, 20], # Meñique
                [0, 17]               # Palma
            ]
            
            for conn in connections:
                for i in range(len(conn) - 1):
                    pt1 = (int(landmarks[conn[i]][0]), int(landmarks[conn[i]][1]))
                    pt2 = (int(landmarks[conn[i+1]][0]), int(landmarks[conn[i+1]][1]))
                    cv2.line(frame, pt1, pt2, COLOR_BLUE, 2)

# ════════════════════════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

class TraductorPortable:
    """Aplicación principal del traductor portátil"""
    
    def __init__(self, camera_id: int = 0, confidence: float = 0.65, display: bool = True):
        self.camera_id = camera_id
        self.confidence_threshold = confidence
        self.display = display
        
        # Componentes
        self.recognizer = GestureRecognizer()
        self.tts = TextToSpeech()
        
        # Estado
        self.current_gesture = None
        self.stability_count = 0
        self.last_spoken = None
        self.last_spoken_time = 0
        self.gesture_buffer = []
        self.last_gesture_time = time.time()
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
    
    def initialize(self) -> bool:
        """Inicializa todos los componentes"""
        logger.info("═" * 60)
        logger.info("🤟 TRADUCTOR PORTÁTIL DE LENGUA DE SEÑAS")
        logger.info("═" * 60)
        
        # Cargar modelo
        if not self.recognizer.load_model():
            self.tts.speak("Error: No se pudo cargar el modelo")
            return False
        
        # Anunciar inicio
        self.tts.speak("Traductor de lengua de señas listo", block=True)
        
        return True
    
    def run(self):
        """Bucle principal de la aplicación"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir la cámara {self.camera_id}")
            self.tts.speak("Error: No se pudo abrir la cámara")
            return
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("📷 Cámara iniciada")
        logger.info("Presiona 'q' para salir, 'r' para reiniciar buffer")
        logger.info("═" * 60)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Voltear horizontalmente (espejo)
                frame = cv2.flip(frame, 1)
                
                # Procesar frame
                self.process_frame(frame)
                
                # Calcular FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_frame_time + 0.001)
                self.fps_history.append(fps)
                self.last_frame_time = current_time
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                
                # Verificar timeout para auto-traducción
                if (len(self.gesture_buffer) > 0 and 
                    current_time - self.last_gesture_time > SILENCE_TIMEOUT):
                    self.synthesize_phrase()
                
                # Mostrar visualización
                if self.display:
                    self.draw_ui(frame, avg_fps)
                    cv2.imshow('Traductor LSE', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.gesture_buffer.clear()
                        self.tts.speak("Buffer limpiado")
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Detenido por usuario")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tts.speak("Traductor detenido")
    
    def process_frame(self, frame):
        """Procesa un frame y reconoce gestos"""
        # Detectar manos (igual que backend)
        hands_info = self.recognizer.detect_hands(frame)
        
        # Dibujar landmarks
        self.recognizer.draw_landmarks(frame, hands_info)
        
        if hands_info['num_hands'] == 0:
            self.stability_count = 0
            self.current_gesture = None
            return
        
        # Predecir (igual que backend)
        gesture, confidence = self.recognizer.predict(hands_info)
        
        if gesture and confidence >= self.confidence_threshold:
            # Estabilidad
            if gesture == self.current_gesture:
                self.stability_count += 1
            else:
                self.current_gesture = gesture
                self.stability_count = 1
            
            # Confirmar gesto
            if self.stability_count >= STABILITY_FRAMES:
                current_time = time.time()
                
                # Verificar cooldown
                if (self.last_spoken != gesture or 
                    current_time - self.last_spoken_time > COOLDOWN_TIME):
                    
                    # Registrar gesto
                    self.last_spoken = gesture
                    self.last_spoken_time = current_time
                    self.last_gesture_time = current_time
                    
                    # Agregar al buffer
                    self.gesture_buffer.append(gesture)
                    
                    # Hablar
                    self.tts.speak(gesture)
                    
                    logger.info(f"✅ {gesture} ({confidence*100:.1f}%)")
        else:
            self.stability_count = max(0, self.stability_count - 1)
    
    def synthesize_phrase(self):
        """Sintetiza la frase completa del buffer"""
        if not self.gesture_buffer:
            return
        
        # Crear frase
        phrase = " ".join(self.gesture_buffer)
        phrase = phrase.capitalize() + "."
        
        logger.info(f"💬 Frase: {phrase}")
        self.tts.speak(phrase, block=True)
        
        # Limpiar buffer
        self.gesture_buffer.clear()
    
    def draw_ui(self, frame, fps):
        """Dibuja la interfaz sobre el frame"""
        h, w = frame.shape[:2]
        
        # Fondo semi-transparente para texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Título
        cv2.putText(frame, "TRADUCTOR LSE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)
        
        # Señas disponibles
        gestures_text = f"Senas: {', '.join(self.recognizer.labels[:5])}"
        if len(self.recognizer.labels) > 5:
            gestures_text += f"... (+{len(self.recognizer.labels)-5})"
        cv2.putText(frame, gestures_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Última detección
        if self.last_spoken:
            cv2.putText(frame, f"Ultima: {self.last_spoken}", (10, h-70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PURPLE, 2)
        
        # Buffer de gestos
        if self.gesture_buffer:
            buffer_text = " ".join(self.gesture_buffer[-5:])  # Últimos 5
            cv2.putText(frame, f"Buffer: {buffer_text}", (10, h-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)
        
        # Instrucciones
        cv2.putText(frame, "Q=Salir  R=Reiniciar", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

# ════════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Traductor Portátil de Lengua de Señas Ecuatoriana'
    )
    parser.add_argument('--camera', type=int, default=0,
                        help='ID de cámara (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.65,
                        help='Umbral de confianza (default: 0.65)')
    parser.add_argument('--no-display', action='store_true',
                        help='Ejecutar sin ventana de visualización')
    
    args = parser.parse_args()
    
    # Crear y ejecutar traductor
    traductor = TraductorPortable(
        camera_id=args.camera,
        confidence=args.confidence,
        display=not args.no_display
    )
    
    if traductor.initialize():
        traductor.run()
    else:
        logger.error("❌ No se pudo inicializar el traductor")
        exit(1)


if __name__ == "__main__":
    main()
