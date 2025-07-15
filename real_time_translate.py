import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
import time
import threading
import os
import sys
from utils import extract_hand_landmarks, extract_face_landmarks
from sklearn.preprocessing import StandardScaler
from collections import deque

# Obtener ruta de recursos (útil para ejecutables con PyInstaller)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Configuración del motor de voz
engine = pyttsx3.init()
engine.setProperty('rate', 120)  # Velocidad mejorada
voices = engine.getProperty('voices')
if voices:
    # Intentar usar voz en español si está disponible
    for voice in voices:
        if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
            engine.setProperty('voice', voice.id)
            break

def speak(text):
    """Función mejorada para texto a voz"""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error en síntesis de voz: {e}")

class GesturePredictor:
    def __init__(self, model_path, labels_path, scaler_path=None):
        """Inicializar el predictor de gestos"""
        
        # Verificar archivos
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Etiquetas no encontradas: {labels_path}")
        
        # Cargar modelo y etiquetas
        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)
        
        # Cargar normalizador si existe
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print("✅ Normalizador cargado")
        
        # Buffer para suavizar predicciones
        self.prediction_buffer = deque(maxlen=5)
        self.confidence_threshold = 0.6  # Umbral de confianza más alto
        
        print(f"✅ Modelo cargado con {len(self.labels)} clases")
    
    def predict(self, input_vector):
        """Predecir gesto con suavizado"""
        
        if input_vector is None or input_vector.shape[0] != 1530:
            return None, 0.0
        
        # Normalizar si hay scaler disponible
        if self.scaler:
            input_vector = self.scaler.transform(input_vector.reshape(1, -1)).flatten()
        
        # Predicción
        input_data = np.expand_dims(input_vector, axis=0)
        prediction = self.model.predict(input_data, verbose=0)[0]
        
        # Agregar al buffer
        self.prediction_buffer.append(prediction)
        
        # Promedio de predicciones para suavizar
        if len(self.prediction_buffer) >= 3:
            avg_prediction = np.mean(list(self.prediction_buffer), axis=0)
        else:
            avg_prediction = prediction
        
        gesture_index = np.argmax(avg_prediction)
        confidence = avg_prediction[gesture_index]
        gesture = self.labels[gesture_index]
        
        return gesture, confidence

# Verificar rutas de archivos
model_path = resource_path("model/gesture_model.h5")
labels_path = resource_path("model/labels.pkl")
scaler_path = resource_path("model/scaler.pkl")

# Inicializar predictor
try:
    predictor = GesturePredictor(model_path, labels_path, scaler_path)
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    sys.exit(1)

# Inicializar MediaPipe con configuración optimizada
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,
    min_detection_confidence=0.7,  # Reducido para mejor detección
    min_tracking_confidence=0.5    # Reducido para mejor seguimiento
)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1,
    min_detection_confidence=0.5,  # Reducido para mejor detección
    min_tracking_confidence=0.5    # Reducido para mejor seguimiento
)

mp_draw = mp.solutions.drawing_utils

# Iniciar cámara con mejor configuración
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    sys.exit(1)

# Configurar resolución óptima
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🎥 Reconocimiento en tiempo real mejorado iniciado")
print("📋 Controles:")
print("  - ESC: Salir")
print("  - ESPACIO: Pausar/Reanudar")
print("  - 'r': Reiniciar buffer de predicciones")

# Variables de control mejoradas
last_spoken_time = 0
spoken_gesture = None
current_gesture = None
gesture_counter = 0
required_frames = 3  # Reducido para mejor respuesta
min_time_between_phrases = 2.0  # Tiempo entre frases
paused = False
fps_counter = 0
fps_start_time = time.time()

# Estadísticas en tiempo real
stats = {
    'predictions': 0,
    'confident_predictions': 0,
    'gestures_spoken': 0,
    'start_time': time.time()
}

# Loop principal mejorado
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al leer de la cámara.")
            break

        # Calcular FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not paused:
            # Procesar con MediaPipe
            hands_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            # Extraer landmarks
            hand_landmarks = extract_hand_landmarks(hands_results)
            face_landmarks = extract_face_landmarks(face_results)

            # Crear vector de entrada
            input_vector = None
            if hand_landmarks is not None and face_landmarks is not None:
                input_vector = np.concatenate([hand_landmarks, face_landmarks])
            elif hand_landmarks is not None:
                input_vector = np.concatenate([hand_landmarks, np.zeros(1404)])
            elif face_landmarks is not None:
                input_vector = np.concatenate([np.zeros(126), face_landmarks])

            # Predicción mejorada
            if input_vector is not None:
                gesture, confidence = predictor.predict(input_vector)
                stats['predictions'] += 1
                
                if gesture and confidence >= predictor.confidence_threshold:
                    stats['confident_predictions'] += 1
                    
                    # Mostrar predicción
                    cv2.putText(image, f"{gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(image, f"Confianza: {confidence:.3f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Control de repetición y habla mejorado
                    if gesture == current_gesture:
                        gesture_counter += 1
                    else:
                        current_gesture = gesture
                        gesture_counter = 1

                    if gesture_counter >= required_frames:
                        current_time = time.time()
                        if (gesture != spoken_gesture and 
                            (current_time - last_spoken_time > min_time_between_phrases)):
                            
                            # Hablar en hilo separado
                            threading.Thread(
                                target=speak, 
                                args=(gesture,), 
                                daemon=True
                            ).start()
                            
                            spoken_gesture = gesture
                            last_spoken_time = current_time
                            gesture_counter = 0
                            stats['gestures_spoken'] += 1
                            
                            print(f"🗣️ Hablando: {gesture} (confianza: {confidence:.3f})")
                else:
                    # Predicción con baja confianza
                    if gesture:
                        cv2.putText(image, f"{gesture} (?)", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                        cv2.putText(image, f"Confianza baja: {confidence:.3f}", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Dibujar landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmark in hands_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image, hand_landmark, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        # Dibujar rostro (opcional, comentar si es muy lento)
        # if face_results.multi_face_landmarks:
        #     for face_landmark in face_results.multi_face_landmarks:
        #         mp_draw.draw_landmarks(
        #             image, face_landmark, mp_face.FACEMESH_TESSELATION,
        #             mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
        #         )

        # Mostrar información en pantalla
        runtime = time.time() - stats['start_time']
        info_text = [
            f"FPS: {fps if 'fps' in locals() else 0:.1f}",
            f"Tiempo: {int(runtime)}s",
            f"Predicciones: {stats['predictions']}",
            f"Confiables: {stats['confident_predictions']}",
            f"Habladas: {stats['gestures_spoken']}",
            f"Umbral: {predictor.confidence_threshold:.2f}",
            "PAUSADO" if paused else "ACTIVO"
        ]
        
        for i, text in enumerate(info_text):
            color = (0, 0, 255) if paused else (255, 255, 255)
            cv2.putText(image, text, (10, image.shape[0] - 20 - i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Mostrar ventana
        cv2.imshow("Reconocimiento en tiempo real", image)
        
        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # ESPACIO
            paused = not paused
            print(f"{'⏸️ Pausado' if paused else '▶️ Reanudado'}")
        elif key == ord('r'):  # R
            predictor.prediction_buffer.clear()
            print("🔄 Buffer de predicciones reiniciado")

except KeyboardInterrupt:
    print("\n⏹️ Interrumpido por usuario")

finally:
    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar estadísticas finales
    total_time = time.time() - stats['start_time']
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"  - Tiempo total: {total_time:.1f}s")
    print(f"  - Predicciones totales: {stats['predictions']}")
    print(f"  - Predicciones confiables: {stats['confident_predictions']}")
    print(f"  - Gestos hablados: {stats['gestures_spoken']}")
    if stats['predictions'] > 0:
        print(f"  - Tasa de confianza: {stats['confident_predictions']/stats['predictions']*100:.1f}%")
    print("🛑 Programa finalizado.")
