#!/usr/bin/env python3
"""
TRADUCTOR LSE - Reconocimiento en Tiempo Real
Muestra subtítulos y convierte a voz.

Técnica: MediaPipe landmarks → LSTM → Clasificación → TTS
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

# TTS
try:
    import pyttsx3
    tts = pyttsx3.init()
    tts.setProperty('rate', 150)
    # Buscar voz en español
    for voice in tts.getProperty('voices'):
        if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
            tts.setProperty('voice', voice.id)
            break
    TTS_OK = True
except:
    TTS_OK = False
    print("⚠️ TTS no disponible. Instalar: pip install pyttsx3")

# === CONFIGURACIÓN ===
DIR_MODELO = os.path.join(os.path.dirname(__file__), "modelo")
FRAMES = 30
FEATURES = 126
UMBRAL_CONFIANZA = 0.70
COOLDOWN = 1.5  # Segundos entre detecciones

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


class TraductorLSE:
    def __init__(self):
        self.modelo = None
        self.encoder = None
        self.clases = []
        self.buffer = deque(maxlen=FRAMES)
        self.subtitulos = []
        self.ultima_deteccion = 0
        self.ultima_sena = None
        
    def cargar_modelo(self):
        """Carga el modelo entrenado."""
        modelo_path = os.path.join(DIR_MODELO, "modelo.h5")
        encoder_path = os.path.join(DIR_MODELO, "encoder.pkl")
        info_path = os.path.join(DIR_MODELO, "info.json")
        
        if not os.path.exists(modelo_path):
            print("\n❌ No hay modelo entrenado.")
            print("   Ejecuta primero:")
            print("   1. python 1_grabar_senas.py")
            print("   2. python 2_entrenar_modelo.py")
            return False
        
        print("🔄 Cargando modelo...")
        self.modelo = tf.keras.models.load_model(modelo_path)
        
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)
        
        with open(info_path) as f:
            info = json.load(f)
            self.clases = info['clases']
        
        print(f"✅ Modelo cargado")
        print(f"   Señas: {self.clases}")
        print(f"   Accuracy: {info.get('accuracy', 'N/A'):.1%}")
        return True
    
    def extraer_landmarks(self, frame):
        """Extrae landmarks normalizados."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        features = np.zeros(FEATURES)
        
        if result.multi_hand_landmarks:
            for idx, hand_lm in enumerate(result.multi_hand_landmarks[:2]):
                wrist = hand_lm.landmark[0]
                for i, lm in enumerate(hand_lm.landmark):
                    base = idx * 63 + i * 3
                    features[base] = lm.x - wrist.x
                    features[base + 1] = lm.y - wrist.y
                    features[base + 2] = lm.z - wrist.z
        
        return features, result
    
    def predecir(self):
        """Realiza predicción con el buffer actual."""
        if len(self.buffer) < FRAMES:
            return None, 0.0
        
        seq = np.array(list(self.buffer))
        seq = np.expand_dims(seq, axis=0)
        
        pred = self.modelo.predict(seq, verbose=0)[0]
        idx = np.argmax(pred)
        conf = pred[idx]
        
        return self.encoder.inverse_transform([idx])[0], conf
    
    def hablar(self, texto):
        """Convierte texto a voz."""
        if TTS_OK and texto:
            print(f"🔊 Hablando: {texto}")
            tts.say(texto)
            tts.runAndWait()
    
    def ejecutar(self):
        """Loop principal del traductor."""
        if not self.cargar_modelo():
            return
        
        print("\n" + "="*60)
        print("  TRADUCTOR LSE - TIEMPO REAL")
        print("="*60)
        print("\nControles:")
        print("  [ESPACIO] → Convertir subtítulos a voz")
        print("  [C]       → Limpiar subtítulos")
        print("  [Q]       → Salir")
        print("-"*60)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            features, result = self.extraer_landmarks(frame)
            
            # Dibujar manos
            hay_mano = False
            if result.multi_hand_landmarks:
                hay_mano = True
                for hand_lm in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                        mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
                    )
                
                # Agregar al buffer
                self.buffer.append(features)
                
                # Intentar predecir
                ahora = time.time()
                if (len(self.buffer) >= FRAMES and 
                    ahora - self.ultima_deteccion > COOLDOWN):
                    
                    sena, conf = self.predecir()
                    
                    if sena and conf >= UMBRAL_CONFIANZA:
                        if sena != self.ultima_sena:
                            # Métricas ISO/IEC 25023
                            tiempo_respuesta = (ahora - self.ultima_deteccion) * 1000  # ms
                            
                            self.subtitulos.append(sena)
                            self.ultima_sena = sena
                            self.ultima_deteccion = ahora
                            
                            # Log con métricas
                            print(f"✓ {sena} | Confianza: {conf:.0%} | Tiempo: {tiempo_respuesta:.0f}ms")
            
            # === UI ===
            h, w = frame.shape[:2]
            
            # Header
            cv2.rectangle(frame, (0, 0), (w, 50), (40,40,40), -1)
            cv2.putText(frame, "TRADUCTOR LSE", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            # Buffer indicator
            buf_pct = len(self.buffer) / FRAMES
            cv2.rectangle(frame, (w-110, 15), (w-10, 35), (60,60,60), -1)
            cv2.rectangle(frame, (w-110, 15), (int(w-110 + 100*buf_pct), 35), (0,255,0), -1)
            
            # Última seña detectada
            if self.ultima_sena:
                cv2.putText(frame, f"Ultima: {self.ultima_sena}", (200, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            # Subtítulos
            cv2.rectangle(frame, (0, h-70), (w, h), (30,30,30), -1)
            texto = " ".join(self.subtitulos[-8:]) if self.subtitulos else "[Esperando señas...]"
            cv2.putText(frame, texto, (10, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # Controles
            cv2.putText(frame, "[ESPACIO] Hablar | [C] Limpiar | [Q] Salir", 
                       (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            
            cv2.imshow('Traductor LSE', frame)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                texto = " ".join(self.subtitulos)
                if texto:
                    self.hablar(texto)
            elif key == ord('c'):
                self.subtitulos = []
                self.ultima_sena = None
                self.buffer.clear()
                print("🗑️ Subtítulos limpiados")
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\n👋 ¡Hasta luego!")


if __name__ == "__main__":
    TraductorLSE().ejecutar()
