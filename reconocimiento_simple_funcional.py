#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECONOCIMIENTO SIMPLE LSE ECUADOR CON VOZ
Versión completa con síntesis de voz
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import pyttsx3

# Configurar entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ReconocimientoConVoz:
    def __init__(self):
        print("🚀 Iniciando LSE Ecuador - Reconocimiento con Voz")
        
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Inicializar síntesis de voz
        try:
            self.engine = pyttsx3.init()
            # Configurar voz en español
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            
            # Configurar velocidad y volumen
            self.engine.setProperty('rate', 150)  # Velocidad
            self.engine.setProperty('volume', 0.8)  # Volumen
            
            print("🔊 Síntesis de voz iniciada")
        except Exception as e:
            print(f"⚠️ Error con síntesis de voz: {e}")
            self.engine = None
        
        # Control de voz
        self.ultimo_gesto_hablado = ""
        self.tiempo_ultimo_audio = 0
        self.audio_thread = None
        
    def hablar(self, texto):
        """Reproducir texto con síntesis de voz"""
        if not self.engine:
            return
            
        # Evitar repetir muy rápido
        tiempo_actual = time.time()
        if tiempo_actual - self.tiempo_ultimo_audio < 2.0:
            return
            
        if texto != self.ultimo_gesto_hablado:
            self.ultimo_gesto_hablado = texto
            self.tiempo_ultimo_audio = tiempo_actual
            
            # Ejecutar en hilo separado para no bloquear video
            if self.audio_thread and self.audio_thread.is_alive():
                return
                
            self.audio_thread = threading.Thread(target=self._hablar_hilo, args=(texto,))
            self.audio_thread.daemon = True
            self.audio_thread.start()
    
    def _hablar_hilo(self, texto):
        """Función auxiliar para hablar en hilo separado"""
        try:
            print(f"🔊 Diciendo: {texto}")
            self.engine.say(texto)
            self.engine.runAndWait()
        except Exception as e:
            print(f"⚠️ Error de audio: {e}")
    
    def detectar_gesto_simple(self, landmarks):
        """Detectar gesto basado en posiciones de landmarks"""
        if not landmarks:
            return "SIN_DETECCION"
        
        # Obtener puntos clave
        puntos = landmarks.landmark
        
        # Punta de dedos
        punta_pulgar = puntos[4]
        punta_indice = puntos[8]
        punta_medio = puntos[12]
        punta_anular = puntos[16]
        punta_meñique = puntos[20]
        
        # Base de dedos
        base_indice = puntos[6]
        base_medio = puntos[10]
        base_anular = puntos[14]
        base_meñique = puntos[18]
        
        # Detectar mano (izquierda o derecha)
        muñeca = puntos[0]
        
        # Detectar dedos extendidos
        dedos_arriba = []
        
        # Pulgar (diferente lógica según la mano)
        if punta_pulgar.x > puntos[3].x:  # Mano derecha
            dedos_arriba.append(1)
        else:
            dedos_arriba.append(0)
            
        # Otros dedos
        for punta, base in [(punta_indice, base_indice), (punta_medio, base_medio), 
                           (punta_anular, base_anular), (punta_meñique, base_meñique)]:
            if punta.y < base.y:
                dedos_arriba.append(1)
            else:
                dedos_arriba.append(0)
        
        # Clasificar gesto
        total_dedos = sum(dedos_arriba)
        
        if total_dedos == 5:
            return "HOLA"
        elif total_dedos == 0:
            return "ADIOS"
        elif dedos_arriba == [1, 0, 0, 0, 0]:
            return "SI"
        elif dedos_arriba == [0, 1, 0, 0, 0]:
            return "NO"
        elif total_dedos >= 3:
            return "GRACIAS"
        else:
            return "PROCESANDO..."
    
    def ejecutar(self):
        """Ejecutar reconocimiento con voz"""
        print("📹 Iniciando cámara...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: No se puede acceder a la cámara")
            input("Presiona Enter para salir...")
            return
        
        print("✅ Cámara iniciada correctamente")
        print("🖐️ Muestra tu mano frente a la cámara")
        print("🔊 Las señas se reproducirán con voz")
        print("⚠️ Presiona 'q' para salir")
        
        gesto_anterior = ""
        contador_estable = 0
        
        # Colores para la interfaz
        COLOR_VERDE = (0, 255, 0)
        COLOR_AZUL = (255, 100, 0)
        COLOR_BLANCO = (255, 255, 255)
        COLOR_ROJO = (0, 0, 255)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Voltear imagen horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Convertir color para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            results = self.hands.process(rgb_frame)
            
            gesto_actual = "SIN_MANO"
            
            # Dibujar landmarks y detectar gestos
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks con estilo mejorado
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=COLOR_AZUL, thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=COLOR_VERDE, thickness=2))
                    
                    # Detectar gesto
                    gesto_actual = self.detectar_gesto_simple(hand_landmarks)
            
            # Estabilizar detección (evitar parpadeo)
            if gesto_actual == gesto_anterior:
                contador_estable += 1
            else:
                contador_estable = 0
                gesto_anterior = gesto_actual
            
            # Mostrar gesto si es estable
            if contador_estable > 8 and gesto_actual not in ["SIN_MANO", "PROCESANDO..."]:
                gesto_mostrar = gesto_actual
                # Reproducir con voz
                self.hablar(gesto_actual)
                color_texto = COLOR_VERDE
            else:
                gesto_mostrar = "DETECTANDO..." if gesto_actual != "SIN_MANO" else "SIN_MANO"
                color_texto = COLOR_BLANCO
            
            # Crear overlay de información
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Título principal
            cv2.putText(frame, "LSE ECUADOR - RECONOCIMIENTO CON VOZ", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_VERDE, 2)
            
            # Gesto detectado
            cv2.putText(frame, f"Gesto: {gesto_mostrar}", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_texto, 3)
            
            # Instrucciones
            cv2.putText(frame, "Presiona 'q' para salir", 
                       (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLANCO, 2)
            
            # Indicador de estado
            estado_color = COLOR_VERDE if results.multi_hand_landmarks else COLOR_ROJO
            cv2.circle(frame, (frame.shape[1] - 30, 30), 15, estado_color, -1)
            
            # Lista de gestos disponibles
            gestos_texto = ["HOLA (mano abierta)", "ADIOS (puño)", "SI (pulgar)", "NO (indice)", "GRACIAS (3+ dedos)"]
            for i, gesto_info in enumerate(gestos_texto):
                cv2.putText(frame, gesto_info, 
                           (10, frame.shape[0] - 100 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_AZUL, 1)
            
            # Mostrar frame
            cv2.imshow('LSE Ecuador - Reconocimiento con Voz', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Limpiar recursos
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        if self.engine:
            self.engine.stop()
        
        print("✅ Reconocimiento finalizado")

def main():
    """Función principal"""
    try:
        reconocedor = ReconocimientoConVoz()
        reconocedor.ejecutar()
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("Presiona Enter para salir...")

if __name__ == "__main__":
    main()
