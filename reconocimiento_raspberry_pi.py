#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECONOCIMIENTO LSE OPTIMIZADO PARA RASPBERRY PI
Versión ultra-ligera con configuración específica para RPi
"""

import os
import sys
import cv2
import numpy as np
import threading
import time

# Configuración específica para Raspberry Pi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

# Importar MediaPipe después de configurar
try:
    import mediapipe as mp
    print("✅ MediaPipe cargado")
except ImportError:
    print("❌ Error: MediaPipe no disponible")
    sys.exit(1)

# Síntesis de voz opcional (puede fallar en RPi)
try:
    import pyttsx3
    VOZ_DISPONIBLE = True
    print("✅ Síntesis de voz disponible")
except ImportError:
    VOZ_DISPONIBLE = False
    print("⚠️ Síntesis de voz no disponible")

class ReconocimientoRaspberryPi:
    def __init__(self):
        print("🚀 LSE Ecuador - Versión Raspberry Pi")
        
        # Configuración optimizada para RPi
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Solo 1 mano para mejor rendimiento
            min_detection_confidence=0.5,  # Menos estricto
            min_tracking_confidence=0.3,   # Menos estricto
            model_complexity=0  # Modelo más simple
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Configurar voz si está disponible
        self.engine = None
        if VOZ_DISPONIBLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 120)  # Más lento
                self.engine.setProperty('volume', 0.9)
                print("🔊 Voz configurada")
            except:
                self.engine = None
                print("⚠️ Voz no configurada")
        
        # Control de audio
        self.ultimo_gesto = ""
        self.tiempo_ultimo_audio = 0
        
        # Configuración de video optimizada
        self.ancho_frame = 320  # Resolución baja
        self.alto_frame = 240
        self.fps_objetivo = 10  # FPS bajo para RPi
        
    def hablar(self, texto):
        """Reproducir texto con voz (si está disponible)"""
        if not self.engine:
            return
            
        tiempo_actual = time.time()
        if tiempo_actual - self.tiempo_ultimo_audio < 3.0:  # 3 segundos entre audios
            return
            
        if texto != self.ultimo_gesto:
            self.ultimo_gesto = texto
            self.tiempo_ultimo_audio = tiempo_actual
            
            # Ejecutar en hilo separado
            def hablar_hilo():
                try:
                    print(f"🔊 Diciendo: {texto}")
                    self.engine.say(texto)
                    self.engine.runAndWait()
                except:
                    pass
                    
            hilo = threading.Thread(target=hablar_hilo)
            hilo.daemon = True
            hilo.start()
    
    def detectar_gesto_rapido(self, landmarks):
        """Detección rápida optimizada para RPi"""
        if not landmarks:
            return "SIN_DETECCION"
        
        try:
            puntos = landmarks.landmark
            
            # Solo verificar puntas principales
            punta_pulgar = puntos[4]
            punta_indice = puntos[8]
            punta_medio = puntos[12]
            punta_anular = puntos[16]
            punta_meñique = puntos[20]
            
            # Bases
            base_indice = puntos[6]
            base_medio = puntos[10]
            base_anular = puntos[14]
            base_meñique = puntos[18]
            
            # Contar dedos extendidos (lógica simplificada)
            dedos = 0
            
            # Pulgar
            if punta_pulgar.x > puntos[3].x:
                dedos += 1
                
            # Otros dedos
            if punta_indice.y < base_indice.y:
                dedos += 1
            if punta_medio.y < base_medio.y:
                dedos += 1
            if punta_anular.y < base_anular.y:
                dedos += 1
            if punta_meñique.y < base_meñique.y:
                dedos += 1
            
            # Clasificación simple
            if dedos == 5:
                return "HOLA"
            elif dedos == 0:
                return "ADIOS"
            elif dedos == 1:
                if punta_pulgar.x > puntos[3].x and punta_indice.y > base_indice.y:
                    return "SI"
                elif punta_indice.y < base_indice.y:
                    return "NO"
            elif dedos >= 3:
                return "GRACIAS"
            else:
                return "DETECTANDO"
                
        except:
            return "ERROR"
    
    def ejecutar(self):
        """Ejecutar reconocimiento optimizado para RPi"""
        print("📹 Configurando cámara para Raspberry Pi...")
        
        # Configuración de cámara optimizada
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: No se puede acceder a la cámara")
            print("💡 Consejos para RPi:")
            print("   - Verificar que la cámara esté habilitada: sudo raspi-config")
            print("   - Verificar conexión de la cámara")
            print("   - Intentar con: sudo modprobe bcm2835-v4l2")
            input("Presiona Enter para salir...")
            return
        
        # Configurar resolución baja para mejor rendimiento
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho_frame)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto_frame)
        cap.set(cv2.CAP_PROP_FPS, self.fps_objetivo)
        
        print(f"✅ Cámara iniciada: {self.ancho_frame}x{self.alto_frame} @ {self.fps_objetivo}fps")
        print("🖐️ Muestra UNA mano frente a la cámara")
        print("🔊 Audio:", "Disponible" if self.engine else "No disponible")
        print("⚠️ Presiona 'q' para salir")
        
        # Variables de control
        gesto_anterior = ""
        contador_estable = 0
        frame_count = 0
        tiempo_inicio = time.time()
        
        # Colores
        VERDE = (0, 255, 0)
        AZUL = (255, 100, 0)
        BLANCO = (255, 255, 255)
        ROJO = (0, 0, 255)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ No se puede leer de la cámara")
                    break
                
                frame_count += 1
                
                # Procesar solo cada 2 frames para mejor rendimiento
                if frame_count % 2 != 0:
                    cv2.imshow('LSE Ecuador - Raspberry Pi', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Voltear imagen
                frame = cv2.flip(frame, 1)
                
                # Convertir a RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Procesar con MediaPipe
                results = self.hands.process(rgb_frame)
                
                gesto_actual = "SIN_MANO"
                
                # Procesar detección
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Dibujar landmarks simples
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=AZUL, thickness=1, circle_radius=1),
                            self.mp_draw.DrawingSpec(color=VERDE, thickness=1))
                        
                        # Detectar gesto
                        gesto_actual = self.detectar_gesto_rapido(hand_landmarks)
                
                # Estabilizar detección
                if gesto_actual == gesto_anterior:
                    contador_estable += 1
                else:
                    contador_estable = 0
                    gesto_anterior = gesto_actual
                
                # Mostrar gesto estable
                if contador_estable > 6 and gesto_actual not in ["SIN_MANO", "DETECTANDO", "ERROR"]:
                    gesto_mostrar = gesto_actual
                    self.hablar(gesto_actual)
                    color_texto = VERDE
                else:
                    gesto_mostrar = "DETECTANDO..." if gesto_actual != "SIN_MANO" else "SIN_MANO"
                    color_texto = BLANCO
                
                # Interfaz simple
                cv2.putText(frame, f"LSE RPi: {gesto_mostrar}", 
                           (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2)
                
                cv2.putText(frame, "q=salir", 
                           (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLANCO, 1)
                
                # Estado de detección
                estado_color = VERDE if results.multi_hand_landmarks else ROJO
                cv2.circle(frame, (frame.shape[1] - 15, 15), 8, estado_color, -1)
                
                # Mostrar FPS cada 30 frames
                if frame_count % 30 == 0:
                    tiempo_actual = time.time()
                    fps_real = 30 / (tiempo_actual - tiempo_inicio)
                    print(f"📊 FPS: {fps_real:.1f}")
                    tiempo_inicio = tiempo_actual
                
                # Mostrar frame
                cv2.imshow('LSE Ecuador - Raspberry Pi', frame)
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"❌ Error durante ejecución: {e}")
        
        finally:
            # Limpiar recursos
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            
            if self.engine:
                try:
                    self.engine.stop()
                except:
                    pass
            
            print("✅ Reconocimiento finalizado")

def main():
    """Función principal"""
    print("🍓 LSE ECUADOR - RASPBERRY PI EDITION")
    print("=" * 50)
    
    try:
        reconocedor = ReconocimientoRaspberryPi()
        reconocedor.ejecutar()
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        input("Presiona Enter para salir...")

if __name__ == "__main__":
    main()
