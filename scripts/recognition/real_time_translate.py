#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSE ECUADOR - RECONOCIMIENTO EN TIEMPO REAL
Traductor de señas ecuatorianas con sintesis de voz optimizada
Optimizado para trabajar solo con landmarks de manos (126 dimensiones)
"""

import os
import sys
import time
import warnings
import threading
import queue
import pickle

# Configurar entorno antes de importar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

# Suprimir warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Configurar TensorFlow con minima verbosidad
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(0)

print("LSE ECUADOR - Iniciando sistema...")

# Verificar que el modelo existe
if not os.path.exists("model/gesture_model.h5"):
    print("ERROR: Modelo no encontrado")
    print("Ejecuta: python configuracion_rapida.py")
    input("Presiona Enter para salir...")
    sys.exit(1)

# Cargar modelo y etiquetas
try:
    print("Cargando modelo...")
    model = tf.keras.models.load_model("model/gesture_model.h5", compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    with open("model/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    
    print(f"Modelo cargado: {len(labels)} gestos")
    print(f"Gestos: {', '.join(labels)}")
    
except Exception as e:
    print(f"Error cargando modelo: {e}")
    sys.exit(1)

# Configurar MediaPipe
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe inicializado")
    
except Exception as e:
    print(f"Error MediaPipe: {e}")
    sys.exit(1)

# Configurar sintesis de voz
try:
    import pyttsx3
    engine = pyttsx3.init()
    
    # Configurar voz en español
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
            engine.setProperty('voice', voice.id)
            break
    
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    print(f"Voz inicializada: {len(voices)} voces")
    
except ImportError:
    print("pyttsx3 no disponible - funcionara sin voz")
    engine = None
except Exception as e:
    print(f"Error inicializando voz: {e}")
    engine = None

# Variables globales para control de voz
last_spoken_time = 0
last_gesture = ""
speech_queue = queue.Queue()
speech_thread_running = False

def speech_worker():
    """Hilo trabajador para sintesis de voz"""
    global speech_thread_running
    speech_thread_running = True
    
    while speech_thread_running:
        try:
            text = speech_queue.get(timeout=1)
            if text is None:  # Señal para terminar
                break
            if engine:
                engine.say(text)
                engine.runAndWait()
                print(f"Voz completada: '{text}'")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error de voz: {e}")

def speak_async(text):
    """Enviar texto a la cola de sintesis de voz"""
    try:
        speech_queue.put(text, block=False)
    except queue.Full:
        pass  # Ignorar si la cola esta llena

# Iniciar hilo de voz
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def extract_landmarks(image):
    """Extraer landmarks de las manos detectadas UNICAMENTE (126 dimensiones)"""
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        landmarks = []
        if results.multi_hand_landmarks:
            hands_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer landmarks de cada mano (21 puntos × 3 coordenadas = 63 por mano)
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.extend([lm.x, lm.y, lm.z])
                hands_landmarks.append(hand_data)
            
            # Si tenemos 2 manos, concatenar ambas (126 dimensiones)
            if len(hands_landmarks) == 2:
                landmarks = hands_landmarks[0] + hands_landmarks[1]
            # Si tenemos 1 mano, completar con ceros para la segunda (126 dimensiones)
            elif len(hands_landmarks) == 1:
                landmarks = hands_landmarks[0] + [0.0] * 63
        
        # Si no hay manos detectadas, devolver vector de ceros
        if len(landmarks) == 0:
            landmarks = [0.0] * 126
        
        # Asegurar exactamente 126 dimensiones
        landmarks = landmarks[:126]
        while len(landmarks) < 126:
            landmarks.append(0.0)
        
        return np.array(landmarks).reshape(1, -1)
        
    except Exception as e:
        print(f"Error landmarks: {e}")
        return np.zeros((1, 126))

def predict_gesture(landmarks):
    """Predecir gesto basado en landmarks"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = model.predict(landmarks, verbose=0)
        
        max_confidence = np.max(predictions[0])
        gesture_index = np.argmax(predictions[0])
        gesture = labels[gesture_index]
        
        return gesture, max_confidence
        
    except Exception as e:
        print(f"Error prediccion: {e}")
        return None, 0.0

def should_speak_gesture(gesture, confidence):
    """Determinar si se debe pronunciar el gesto"""
    global last_spoken_time, last_gesture
    
    current_time = time.time()
    min_confidence = 0.35  # Umbral de confianza
    min_time_between_speech = 2.0  # Segundos entre pronunciaciones
    
    # Verificar confianza minima
    if confidence < min_confidence:
        return False
    
    # Verificar tiempo desde ultima pronunciacion
    if current_time - last_spoken_time < min_time_between_speech:
        return False
    
    # Verificar si es un gesto diferente o ha pasado suficiente tiempo
    if gesture != last_gesture or (current_time - last_spoken_time) > 5.0:
        last_spoken_time = current_time
        last_gesture = gesture
        return True
    
    return False

def main():
    """Funcion principal de reconocimiento"""
    print("\nLSE ECUADOR - RECONOCIMIENTO EN TIEMPO REAL")
    print("=" * 50)
    print("   Controles:")
    print("   - ESC: Salir")
    print("   - ESPACIO: Activar/Desactivar voz")
    print("   - 'r': Reiniciar deteccion")
    print("   - 'h': Mostrar ayuda")
    print(f"   Gestos disponibles: {', '.join(labels)}")
    print("=" * 50)
    
    # Inicializar camara con manejo de errores mejorado
    print("Inicializando camara...")
    cap = None
    
    # Probar diferentes backends de camara
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        try:
            print(f"Probando backend: {backend}")
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                # Probar leer un frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Camara inicializada con backend: {backend}")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                cap = None
        except Exception as e:
            print(f"Error con backend {backend}: {e}")
            if cap:
                cap.release()
            cap = None
    
    if cap is None or not cap.isOpened():
        print("ERROR: No se puede acceder a ninguna camara")
        print("Verifica que:")
        print("- La camara este conectada")
        print("- No este siendo usada por otra aplicacion")
        print("- Los drivers esten instalados")
        input("Presiona Enter para salir...")
        return
    
    # Configurar camara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Verificar configuracion
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camara configurada: {width}x{height} @ {fps}fps")
    
    voice_enabled = True
    frame_count = 0
    detection_active = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error leyendo frame, reintentando...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Voltear la imagen horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)
            
            # Crear copia para procesamiento
            display_frame = frame.copy()
            
            gesture = None
            confidence = 0.0
            
            if detection_active:
                # Extraer landmarks
                landmarks = extract_landmarks(frame)
                
                # Predecir gesto
                gesture, confidence = predict_gesture(landmarks)
            
            # Procesar imagen para visualizacion de landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Dibujar landmarks de las manos
            hands_detected = False
            if results.multi_hand_landmarks:
                hands_detected = True
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Dibujar landmarks con diferentes colores para cada mano
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Verde para primera mano, azul para segunda
                    
                    # Dibujar conexiones
                    mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=color, thickness=2)
                    )
                    
                    # Dibujar numero de mano
                    h, w, _ = display_frame.shape
                    hand_center = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    cx, cy = int(hand_center.x * w), int(hand_center.y * h)
                    cv2.putText(display_frame, f"Mano {i+1}", (cx-30, cy-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Crear panel de informacion
            info_height = 160
            info_panel = np.zeros((info_height, display_frame.shape[1], 3), dtype=np.uint8)
            
            # Titulo
            cv2.putText(info_panel, "LSE ECUADOR - TIEMPO REAL", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Estado de deteccion
            status_color = (0, 255, 0) if hands_detected else (0, 0, 255)
            status_text = f"Manos detectadas: {'SI' if hands_detected else 'NO'}"
            cv2.putText(info_panel, status_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            # Mostrar gesto y confianza
            if gesture and confidence > 0.1:
                color = (0, 255, 0) if confidence > 0.35 else (0, 165, 255)
                cv2.putText(info_panel, f"Gesto detectado: {gesture.upper()}", (10, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(info_panel, f"Confianza: {confidence:.2f} ({confidence*100:.1f}%)", (10, 95), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Pronunciar si cumple condiciones
                if voice_enabled and should_speak_gesture(gesture, confidence):
                    speak_async(gesture)
                    cv2.putText(info_panel, "PRONUNCIANDO...", (10, 115), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(info_panel, "Ningun gesto detectado", (10, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Estado de voz
            voice_status = "ACTIVADA" if voice_enabled else "DESACTIVADA"
            voice_color = (0, 255, 0) if voice_enabled else (0, 0, 255)
            cv2.putText(info_panel, f"Voz: {voice_status}", (10, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, voice_color, 1)
            
            # Controles
            cv2.putText(info_panel, "ESC:Salir SPACE:Voz R:Reset H:Ayuda D:Det.On/Off", 
                       (display_frame.shape[1]//3, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Gestos disponibles
            gestures_text = f"Gestos: {' | '.join(labels)}"
            cv2.putText(info_panel, gestures_text, (display_frame.shape[1]//3, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Combinar frame con panel de informacion
            combined_frame = np.vstack([display_frame, info_panel])
            
            # Mostrar frame
            cv2.imshow("LSE Ecuador - Reconocimiento en Tiempo Real", combined_frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # ESPACIO
                voice_enabled = not voice_enabled
                status = "ACTIVADA" if voice_enabled else "DESACTIVADA"
                print(f"Voz {status}")
            elif key == ord('r') or key == ord('R'):  # R
                print("Reiniciando deteccion...")
                global last_spoken_time, last_gesture
                last_spoken_time = 0
                last_gesture = ""
            elif key == ord('d') or key == ord('D'):  # D
                detection_active = not detection_active
                status = "ACTIVADA" if detection_active else "DESACTIVADA"
                print(f"Deteccion {status}")
            elif key == ord('h') or key == ord('H'):  # H
                print("\n" + "="*50)
                print("AYUDA LSE ECUADOR:")
                print("- Mantén las manos dentro del cuadro")
                print("- Gestos claros y bien iluminados")
                print("- Espera 2 segundos entre gestos")
                print(f"- Gestos disponibles: {', '.join(labels)}")
                print("- Confianza mínima: 35%")
                print("- ESC: Salir")
                print("- ESPACIO: Activar/Desactivar voz")
                print("- R: Reiniciar detección")
                print("- D: Activar/Desactivar detección")
                print("- H: Mostrar esta ayuda")
                print("="*50)
    
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario")
    except Exception as e:
        print(f"\nError durante reconocimiento: {e}")
    
    finally:
        # Limpiar recursos
        cap.release()
        cv2.destroyAllWindows()
        
        # Detener hilo de voz
        global speech_thread_running
        speech_thread_running = False
        speech_queue.put(None)  # Señal para terminar
        
        try:
            speech_thread.join(timeout=2)
        except:
            pass
        
        print("Recursos liberados")
        print("Gracias por usar LSE Ecuador!")

if __name__ == "__main__":
    main()
