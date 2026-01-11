#!/usr/bin/env python3
"""
TRADUCTOR LSE - Prototipo Funcional
Reconoce señas y las muestra como subtítulos + voz.

Controles:
    ESPACIO - Hablar los subtítulos
    C       - Limpiar subtítulos
    Q       - Salir
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
import time
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# TTS
try:
    import pyttsx3
    tts = pyttsx3.init()
    tts.setProperty('rate', 150)
except:
    tts = None
    print("⚠️ Sin TTS. Instalar: pip install pyttsx3")

# Configuración
MODELO_DIR = os.path.join(os.path.dirname(__file__), "modelo")
FRAMES = 30
LANDMARKS = 126
UMBRAL = 0.7
COOLDOWN = 1.5

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)


class Traductor:
    def __init__(self):
        self.modelo = None
        self.clases = None
        self.buffer = deque(maxlen=FRAMES)
        self.subtitulos = []
        self.ultima_deteccion = 0
        self.ultima_sena = None
        
        self.cargar_modelo()
    
    def cargar_modelo(self):
        modelo_path = os.path.join(MODELO_DIR, "modelo.h5")
        clases_path = os.path.join(MODELO_DIR, "clases.pkl")
        
        if not os.path.exists(modelo_path):
            print("❌ No hay modelo. Ejecuta:")
            print("   1. python 1_grabar_senas.py")
            print("   2. python 2_entrenar_modelo.py")
            return False
        
        self.modelo = tf.keras.models.load_model(modelo_path)
        with open(clases_path, 'rb') as f:
            self.clases = pickle.load(f)
        
        print(f"✅ Modelo cargado: {list(self.clases.classes_)}")
        return True
    
    def extraer_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        landmarks = np.zeros(LANDMARKS)
        
        if results.multi_hand_landmarks:
            for idx, hand in enumerate(results.multi_hand_landmarks[:2]):
                for i, lm in enumerate(hand.landmark):
                    base = idx * 63 + i * 3
                    landmarks[base:base+3] = [lm.x, lm.y, lm.z]
        
        return landmarks, results
    
    def predecir(self):
        if len(self.buffer) < FRAMES:
            return None, 0
        
        seq = np.array(list(self.buffer))
        pred = self.modelo.predict(np.expand_dims(seq, 0), verbose=0)[0]
        idx = np.argmax(pred)
        
        return self.clases.inverse_transform([idx])[0], pred[idx]
    
    def hablar(self, texto):
        if tts and texto:
            print(f"🔊 {texto}")
            tts.say(texto)
            tts.runAndWait()
    
    def ejecutar(self):
        if not self.modelo:
            return
        
        print("\n" + "="*50)
        print("  TRADUCTOR LSE")
        print("="*50)
        print("[ESPACIO] Hablar | [C] Limpiar | [Q] Salir\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks, results = self.extraer_landmarks(frame)
            
            # Dibujar manos
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                
                self.buffer.append(landmarks)
                
                # Predecir
                ahora = time.time()
                if len(self.buffer) >= FRAMES and ahora - self.ultima_deteccion > COOLDOWN:
                    sena, conf = self.predecir()
                    
                    if sena and conf >= UMBRAL and sena != self.ultima_sena:
                        self.subtitulos.append(sena)
                        self.ultima_sena = sena
                        self.ultima_deteccion = ahora
                        print(f"✓ {sena} ({conf*100:.0f}%)")
            
            # UI - Header
            cv2.rectangle(frame, (0, 0), (640, 40), (40,40,40), -1)
            cv2.putText(frame, "TRADUCTOR LSE", (10, 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # UI - Subtítulos
            cv2.rectangle(frame, (0, 430), (640, 480), (30,30,30), -1)
            texto = " ".join(self.subtitulos[-6:]) if self.subtitulos else "[Esperando...]"
            cv2.putText(frame, texto, (10, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow('Traductor LSE', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.hablar(" ".join(self.subtitulos))
            elif key == ord('c'):
                self.subtitulos = []
                self.ultima_sena = None
                print("🗑️ Limpiado")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Traductor().ejecutar()
