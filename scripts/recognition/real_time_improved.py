#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# Configurar TensorFlow para evitar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

"""
 RECONOCIMIENTO DE LENGUAJE DE SEÑAS EN TIEMPO REAL MEJORADO
==============================================================
Sistema avanzado con metricas de confianza, suavizado de predicciones y evaluacion en tiempo real
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import time
from collections import deque
import pyttsx3
import threading
from utils import extraer_landmarks

class SignLanguageRealTime:
    def __init__(self):
        print(" Inicializando reconocedor mejorado...")
        
        # Configuracion de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Inicializacion de detectores
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
        )
        
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        # Cargar modelo y etiquetas
        self.cargar_modelo()
        
        # Sistema de suavizado de predicciones
        self.prediction_buffer = deque(maxlen=10)  # Buffer mas grande
        self.confidence_buffer = deque(maxlen=10)
        self.last_prediction = None
        self.prediction_count = {}
        
        # Configuracion de texto a voz
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        
        # Metricas en tiempo real
        self.total_predictions = 0
        self.confident_predictions = 0
        self.start_time = time.time()
        
        # Configuracion de visualizacion
        self.colors = {
            'hands': (0, 255, 0),
            'face': (255, 0, 0),
            'text': (255, 255, 255),
            'confidence': (0, 255, 255),
            'metrics': (255, 165, 0)
        }
        
        print(" Sistema inicializado correctamente")
    
    def cargar_modelo(self):
        """Cargar modelo y etiquetas"""
        try:
            # Cargar modelo
            self.model = tf.keras.models.load_model('model/best_model.h5')
            print(" Modelo cargado desde model/best_model.h5")
            
            # Cargar etiquetas
            with open('model/labels.pkl', 'rb') as f:
                self.labels = pickle.load(f)
            print(f" {len(self.labels)} etiquetas cargadas")
            
        except Exception as e:
            print(f" Error cargando modelo: {e}")
            print("Intentando cargar modelo alternativo...")
            try:
                # Intentar modelo alternativo
                self.model = tf.keras.models.load_model('gesture_model.h5')
                
                # Cargar labels.txt como alternativa
                with open('labels.txt', 'r', encoding='utf-8') as f:
                    self.labels = [line.strip() for line in f.readlines()]
                print(" Modelo alternativo cargado correctamente")
                
            except Exception as e2:
                print(f" Error critico: {e2}")
                exit(1)
    
    def predecir_gesto(self, landmarks):
        """Realizar prediccion con metricas mejoradas"""
        try:
            # Realizar prediccion
            prediccion = self.model.predict(landmarks.reshape(1, -1), verbose=0)
            clase_predicha = np.argmax(prediccion)
            confianza = float(np.max(prediccion))
            
            self.total_predictions += 1
            
            # Filtro de confianza
            if confianza < 0.7:  # Umbral mas alto
                return None, confianza
            
            self.confident_predictions += 1
            
            # Obtener etiqueta
            if clase_predicha < len(self.labels):
                gesto = self.labels[clase_predicha]
                
                # Agregar a buffers
                self.prediction_buffer.append(gesto)
                self.confidence_buffer.append(confianza)
                
                return gesto, confianza
            
        except Exception as e:
            print(f" Error en prediccion: {e}")
        
        return None, 0.0
    
    def obtener_prediccion_suavizada(self):
        """Obtener prediccion suavizada usando votacion ponderada"""
        if len(self.prediction_buffer) < 3:
            return None, 0.0
        
        # Contar votos con pesos por confianza
        votos_ponderados = {}
        for gesto, confianza in zip(self.prediction_buffer, self.confidence_buffer):
            if gesto in votos_ponderados:
                votos_ponderados[gesto] += confianza
            else:
                votos_ponderados[gesto] = confianza
        
        # Obtener el gesto con mayor peso
        if votos_ponderados:
            mejor_gesto = max(votos_ponderados, key=votos_ponderados.get)
            confianza_promedio = votos_ponderados[mejor_gesto] / list(self.prediction_buffer).count(mejor_gesto)
            
            # Verificar consistencia (al menos 40% de apariciones)
            apariciones = list(self.prediction_buffer).count(mejor_gesto)
            porcentaje = apariciones / len(self.prediction_buffer)
            
            if porcentaje >= 0.4:  # Al menos 40% de consistencia
                return mejor_gesto, confianza_promedio
        
        return None, 0.0
    
    def hablar_gesto(self, gesto):
        """Pronunciar gesto usando TTS en hilo separado"""
        def speak():
            try:
                self.tts.say(gesto)
                self.tts.runAndWait()
            except:
                pass
        
        threading.Thread(target=speak, daemon=True).start()
    
    def dibujar_landmarks(self, image, results_hands, results_face):
        """Dibujar landmarks mejorados"""
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=self.colors['hands'], thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=self.colors['hands'], thickness=2)
                )
        
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                # Solo dibujar algunos puntos clave de la cara
                for idx in [10, 151, 9, 175, 0]:  # Puntos importantes
                    if idx < len(face_landmarks.landmark):
                        x = int(face_landmarks.landmark[idx].x * image.shape[1])
                        y = int(face_landmarks.landmark[idx].y * image.shape[0])
                        cv2.circle(image, (x, y), 2, self.colors['face'], -1)
    
    def dibujar_info(self, image, gesto_actual, confianza, fps):
        """Dibujar informacion mejorada en pantalla"""
        h, w = image.shape[:2]
        
        # Fondo semitransparente para informacion
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 160), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Informacion principal
        y_offset = 35
        
        # Gesto actual
        if gesto_actual:
            cv2.putText(image, f"Gesto: {gesto_actual}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
            y_offset += 25
            
            # Barra de confianza
            cv2.putText(image, f"Confianza: {confianza:.1%}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['confidence'], 2)
            
            # Barra visual de confianza
            bar_w = 200
            bar_h = 10
            bar_x, bar_y = 20, y_offset + 10
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_w * confianza), bar_y + bar_h), 
                         self.colors['confidence'], -1)
            y_offset += 35
        else:
            cv2.putText(image, "Sin deteccion confiable", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            y_offset += 30
        
        # Metricas en tiempo real
        tiempo_transcurrido = time.time() - self.start_time
        precision_tiempo_real = (self.confident_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
        
        cv2.putText(image, f"FPS: {fps:.1f} | Precision: {precision_tiempo_real:.1f}%", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['metrics'], 1)
        y_offset += 20
        
        cv2.putText(image, f"Predicciones: {self.confident_predictions}/{self.total_predictions}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['metrics'], 1)
        
        # Instrucciones
        cv2.putText(image, "Presiona 'q' para salir, 's' para capturar", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def ejecutar(self):
        """Bucle principal de reconocimiento"""
        print("🎥 Iniciando camara...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(" Error: No se pudo abrir la camara")
            return
        
        print(" Camara iniciada. Presiona 'q' para salir")
        
        # Variables para FPS
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(" Error leyendo frame")
                    break
                
                # Voltear imagen horizontalmente
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Deteccion de landmarks
                results_hands = self.hands.process(rgb_frame)
                results_face = self.face_mesh.process(rgb_frame)
                
                # Extraer landmarks y predecir
                gesto_actual = None
                confianza = 0.0
                
                landmarks = extraer_landmarks(results_hands, results_face)
                if landmarks is not None:
                    gesto_pred, conf_pred = self.predecir_gesto(landmarks)
                    
                    # Obtener prediccion suavizada
                    gesto_actual, confianza = self.obtener_prediccion_suavizada()
                    
                    # Hablar si es una nueva prediccion confiable
                    if (gesto_actual and gesto_actual != self.last_prediction and 
                        confianza > 0.8):
                        self.hablar_gesto(gesto_actual)
                        self.last_prediction = gesto_actual
                
                # Calcular FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Dibujar landmarks e informacion
                self.dibujar_landmarks(frame, results_hands, results_face)
                self.dibujar_info(frame, gesto_actual, confianza, current_fps)
                
                # Mostrar frame
                cv2.imshow('Reconocimiento de Lenguaje de Senas Mejorado', frame)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Capturar screenshot
                    filename = f"captura_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Captura guardada: {filename}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido por usuario")
        
        finally:
            # Mostrar estadisticas finales
            tiempo_total = time.time() - self.start_time
            print(f"\n📊 ESTADISTICAS FINALES:")
            print(f"     Tiempo total: {tiempo_total:.1f}s")
            print(f"    Predicciones totales: {self.total_predictions}")
            print(f"    Predicciones confiables: {self.confident_predictions}")
            print(f"   📈 Precision promedio: {(self.confident_predictions/self.total_predictions*100):.1f}%" if self.total_predictions > 0 else "   📈 Sin predicciones")
            print(f"    FPS promedio: {self.total_predictions/tiempo_total:.1f}")
            
            cap.release()
            cv2.destroyAllWindows()
            print(" Recursos liberados correctamente")

def main():
    """Funcion principal"""
    print("=" * 60)
    print(" RECONOCIMIENTO DE LENGUAJE DE SEÑAS EN TIEMPO REAL")
    print("   Sistema Mejorado con Metricas Avanzadas")
    print("=" * 60)
    
    try:
        recognizer = SignLanguageRealTime()
        recognizer.ejecutar()
    except Exception as e:
        print(f" Error critico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
