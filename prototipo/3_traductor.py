#!/usr/bin/env python3
"""
=============================================================================
MÓDULO 3: TRADUCTOR LSE CON SUBTÍTULOS Y VOZ
=============================================================================
Prototipo funcional que traduce señas de la LSE a texto y voz.

USO:
    python 3_traductor.py

CONTROLES:
    [ESPACIO] - Convertir subtítulos a voz
    [C]       - Limpiar subtítulos
    [R]       - Reiniciar detección
    [Q]       - Salir

REQUISITOS:
    - Modelo entrenado en carpeta 'modelo/'
    - Ejecutar primero 1_grabar_senas.py y 2_entrenar_modelo.py
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
import pickle
import time
from collections import deque
from datetime import datetime

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Text-to-Speech
try:
    import pyttsx3
    TTS_DISPONIBLE = True
except ImportError:
    TTS_DISPONIBLE = False
    print("⚠️ pyttsx3 no instalado. Instalar con: pip install pyttsx3")

# Configuración
MODELO_DIR = os.path.join(os.path.dirname(__file__), "modelo")
SECUENCIA_FRAMES = 30
LANDMARKS_SIZE = 126
UMBRAL_CONFIANZA = 0.7
COOLDOWN_DETECCION = 1.5  # Segundos entre detecciones

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# TTS
if TTS_DISPONIBLE:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    # Buscar voz en español
    for voice in voices:
        if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

class TraductorLSE:
    def __init__(self):
        self.modelo = None
        self.etiquetas = None
        self.senas_info = None
        self.secuencia_actual = deque(maxlen=SECUENCIA_FRAMES)
        self.subtitulos = []
        self.ultima_deteccion = 0
        self.ultima_sena = None
        self.cargar_modelo()
    
    def cargar_modelo(self):
        """Carga el modelo entrenado."""
        modelo_path = os.path.join(MODELO_DIR, 'modelo_lstm.h5')
        etiquetas_path = os.path.join(MODELO_DIR, 'etiquetas.pkl')
        senas_path = os.path.join(MODELO_DIR, 'senas.json')
        
        if not os.path.exists(modelo_path):
            print("❌ No se encontró el modelo. Ejecuta primero:")
            print("   1. python 1_grabar_senas.py")
            print("   2. python 2_entrenar_modelo.py")
            return False
        
        print("🔄 Cargando modelo...")
        self.modelo = tf.keras.models.load_model(modelo_path)
        
        with open(etiquetas_path, 'rb') as f:
            self.etiquetas = pickle.load(f)
        
        with open(senas_path, 'r') as f:
            self.senas_info = json.load(f)
        
        print(f"✅ Modelo cargado. Señas: {self.senas_info['senas']}")
        return True
    
    def extraer_landmarks(self, frame):
        """Extrae landmarks de las manos."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        landmarks = np.zeros(LANDMARKS_SIZE)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx >= 2:
                    break
                for i, lm in enumerate(hand_landmarks.landmark):
                    base = idx * 63 + i * 3
                    landmarks[base] = lm.x
                    landmarks[base + 1] = lm.y
                    landmarks[base + 2] = lm.z
        
        return landmarks, results
    
    def dibujar_manos(self, frame, results):
        """Dibuja landmarks en el frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        return frame
    
    def predecir(self):
        """Realiza predicción con la secuencia actual."""
        if len(self.secuencia_actual) < SECUENCIA_FRAMES:
            return None, 0.0
        
        secuencia = np.array(list(self.secuencia_actual))
        secuencia = np.expand_dims(secuencia, axis=0)
        
        prediccion = self.modelo.predict(secuencia, verbose=0)[0]
        clase_idx = np.argmax(prediccion)
        confianza = prediccion[clase_idx]
        
        sena = self.etiquetas.inverse_transform([clase_idx])[0]
        
        return sena, confianza
    
    def hablar(self, texto):
        """Convierte texto a voz."""
        if TTS_DISPONIBLE and texto:
            print(f"🔊 Hablando: {texto}")
            engine.say(texto)
            engine.runAndWait()
    
    def dibujar_interfaz(self, frame):
        """Dibuja la interfaz en el frame."""
        h, w = frame.shape[:2]
        
        # Panel superior (info)
        cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
        cv2.putText(frame, "TRADUCTOR LSE - Prototipo Funcional", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Buffer: {len(self.secuencia_actual)}/{SECUENCIA_FRAMES}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Panel de subtítulos (inferior)
        cv2.rectangle(frame, (0, h-100), (w, h), (30, 30, 30), -1)
        cv2.rectangle(frame, (5, h-95), (w-5, h-35), (50, 50, 50), -1)
        
        # Texto de subtítulos
        texto_subtitulos = " ".join(self.subtitulos) if self.subtitulos else "[Esperando señas...]"
        
        # Truncar si es muy largo
        if len(texto_subtitulos) > 50:
            texto_subtitulos = "..." + texto_subtitulos[-47:]
        
        cv2.putText(frame, texto_subtitulos, (15, h-55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Controles
        cv2.putText(frame, "[ESPACIO] Hablar | [C] Limpiar | [Q] Salir", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Última seña detectada
        if self.ultima_sena:
            cv2.putText(frame, f"Ultima: {self.ultima_sena}", 
                       (w-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def ejecutar(self):
        """Bucle principal del traductor."""
        if self.modelo is None:
            return
        
        print("\n" + "=" * 60)
        print("   TRADUCTOR LSE EN TIEMPO REAL")
        print("=" * 60)
        print("\nCONTROLES:")
        print("  [ESPACIO] - Convertir subtítulos a voz")
        print("  [C]       - Limpiar subtítulos")
        print("  [Q]       - Salir")
        print("-" * 60)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se puede abrir la cámara")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎥 Cámara iniciada. Mostrando ventana...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks, results = self.extraer_landmarks(frame)
            frame = self.dibujar_manos(frame, results)
            
            # Agregar landmarks al buffer
            mano_detectada = results.multi_hand_landmarks is not None
            
            if mano_detectada:
                self.secuencia_actual.append(landmarks)
                
                # Verificar si podemos hacer predicción
                tiempo_actual = time.time()
                if (len(self.secuencia_actual) >= SECUENCIA_FRAMES and 
                    tiempo_actual - self.ultima_deteccion > COOLDOWN_DETECCION):
                    
                    sena, confianza = self.predecir()
                    
                    if sena and confianza >= UMBRAL_CONFIANZA:
                        # Evitar repetir la misma seña consecutivamente
                        if sena != self.ultima_sena or tiempo_actual - self.ultima_deteccion > 3.0:
                            self.subtitulos.append(sena)
                            self.ultima_sena = sena
                            self.ultima_deteccion = tiempo_actual
                            print(f"✓ Detectado: {sena} ({confianza*100:.1f}%)")
                            
                            # Indicador visual
                            cv2.rectangle(frame, (0, 60), (640, 100), (0, 200, 0), -1)
                            cv2.putText(frame, f"DETECTADO: {sena}", (20, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Dibujar interfaz
            frame = self.dibujar_interfaz(frame)
            
            cv2.imshow('Traductor LSE', frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Espacio - Hablar
                texto = " ".join(self.subtitulos)
                if texto:
                    self.hablar(texto)
            elif key == ord('c'):  # C - Limpiar
                self.subtitulos = []
                self.ultima_sena = None
                print("🗑️ Subtítulos limpiados")
            elif key == ord('r'):  # R - Reiniciar
                self.secuencia_actual.clear()
                print("🔄 Buffer reiniciado")
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\n👋 ¡Hasta luego!")

def main():
    print("=" * 60)
    print("   TRADUCTOR DE LENGUA DE SEÑAS ECUATORIANA")
    print("   Prototipo Funcional - Objetivo 1")
    print("=" * 60)
    
    traductor = TraductorLSE()
    traductor.ejecutar()

if __name__ == "__main__":
    main()
