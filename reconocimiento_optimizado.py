#!/usr/bin/env python3
"""
🎯 RECONOCIMIENTO OPTIMIZADO CON MEJOR PRECISIÓN
===============================================
Versión mejorada con configuraciones optimizadas para mejor reconocimiento
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os
from collections import deque, Counter
import time

def main():
    print("🎯 INICIANDO RECONOCIMIENTO OPTIMIZADO...")
    print("=" * 50)
    
    # 1. CARGAR MODELO
    print("📦 Cargando modelo...")
    try:
        if os.path.exists('model/best_model.h5'):
            model = tf.keras.models.load_model('model/best_model.h5')
            with open('model/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
            print(f"✅ Modelo cargado: {len(labels)} clases")
        else:
            print("❌ No se encontró el modelo. Ejecuta 'python train_model.py' primero")
            return
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # 2. INICIALIZAR MEDIAPIPE CON CONFIGURACIÓN OPTIMIZADA
    print("🤖 Inicializando MediaPipe optimizado...")
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    # Configuración más estricta para mejor calidad
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,   # Más alto para mejor calidad
        min_tracking_confidence=0.7     # Más alto para mejor calidad
    )
    
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    
    print("✅ MediaPipe inicializado con configuración optimizada")
    
    # 3. INICIALIZAR CÁMARA
    print("🎥 Inicializando cámara...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se puede acceder a la cámara")
        return
    
    # Configurar resolución y FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("✅ Cámara inicializada")
    
    # 4. SISTEMA DE SUAVIZADO AVANZADO
    prediction_buffer = deque(maxlen=15)  # Buffer más grande
    confidence_buffer = deque(maxlen=15)
    last_stable_prediction = None
    stability_counter = 0
    
    # 5. FUNCIÓN DE EXTRACCIÓN MEJORADA
    def extraer_landmarks_robusto(results_hands, results_face):
        """Extracción robusta y normalizada de landmarks"""
        landmarks = []
        
        # Manos (126 elementos) con normalización mejorada
        if results_hands and hasattr(results_hands, 'multi_hand_landmarks') and results_hands.multi_hand_landmarks:
            hands_data = []
            for hand in results_hands.multi_hand_landmarks[:2]:
                hand_points = []
                
                # Convertir a array numpy para normalización
                hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
                
                # Normalización centrada en la muñeca
                wrist = hand_landmarks[0]
                hand_landmarks = hand_landmarks - wrist
                
                # Escalar por la distancia al dedo medio
                scale = np.linalg.norm(hand_landmarks[12])
                if scale > 0:
                    hand_landmarks = hand_landmarks / scale
                
                # Aplanar
                hand_points = hand_landmarks.flatten().tolist()
                hands_data.extend(hand_points)
            
            # Asegurar 126 elementos (2 manos)
            while len(hands_data) < 126:
                hands_data.append(0.0)
            landmarks.extend(hands_data[:126])
        else:
            landmarks.extend([0.0] * 126)
        
        # Cara (1404 elementos) con normalización
        if results_face and hasattr(results_face, 'multi_face_landmarks') and results_face.multi_face_landmarks:
            face = results_face.multi_face_landmarks[0]
            
            # Convertir a array numpy
            face_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark])
            
            # Normalización centrada
            center = face_landmarks[1]  # Punto de referencia nasal
            face_landmarks = face_landmarks - center
            
            # Escalar
            scale = np.linalg.norm(face_landmarks[10])
            if scale > 0:
                face_landmarks = face_landmarks / scale
            
            face_points = face_landmarks.flatten().tolist()
            landmarks.extend(face_points[:1404])
            
            # Completar si falta
            while len(landmarks) < 1530:
                landmarks.append(0.0)
        else:
            landmarks.extend([0.0] * 1404)
        
        return np.array(landmarks[:1530])
    
    # 6. FUNCIÓN DE PREDICCIÓN MEJORADA
    def predecir_con_suavizado(landmarks):
        """Predicción con suavizado temporal avanzado"""
        nonlocal last_stable_prediction, stability_counter
        
        # Realizar predicción
        prediccion = model.predict(landmarks.reshape(1, -1), verbose=0)
        clase_predicha = np.argmax(prediccion)
        confianza = float(np.max(prediccion))
        
        # Filtro de confianza más estricto
        if confianza < 0.8:  # Umbral alto para mayor precisión
            return None, 0.0, "Confianza baja"
        
        gesto = labels[clase_predicha]
        
        # Agregar a buffers
        prediction_buffer.append(gesto)
        confidence_buffer.append(confianza)
        
        # Análisis de estabilidad
        if len(prediction_buffer) >= 10:
            # Contar ocurrencias en el buffer
            counter = Counter(list(prediction_buffer)[-10:])  # Últimas 10 predicciones
            most_common = counter.most_common(1)[0]
            most_common_gesture, count = most_common
            
            # Si un gesto aparece al menos 7 de las últimas 10 veces
            if count >= 7:
                avg_confidence = np.mean([conf for pred, conf in zip(prediction_buffer, confidence_buffer) 
                                        if pred == most_common_gesture][-count:])
                
                # Verificar estabilidad
                if most_common_gesture == last_stable_prediction:
                    stability_counter += 1
                else:
                    stability_counter = 1
                    last_stable_prediction = most_common_gesture
                
                # Solo mostrar si es estable (al menos 3 frames consecutivos)
                if stability_counter >= 3:
                    return most_common_gesture, avg_confidence, "ESTABLE"
        
        return None, confianza, "Estabilizando..."
    
    # 7. LOOP PRINCIPAL OPTIMIZADO
    print("\n🎯 SISTEMA OPTIMIZADO LISTO")
    print("=" * 50)
    print("💡 CONSEJOS PARA MEJOR RECONOCIMIENTO:")
    print("   1. 🖐️ Haz gestos claros y definidos")
    print("   2. ⏱️ Mantén la seña por 3-4 segundos")
    print("   3. 💡 Buena iluminación uniforme")
    print("   4. 🎨 Fondo contrastante y simple")
    print("   5. 📏 Distancia de 60-80cm de la cámara")
    print("   6. 🔄 Evita movimientos bruscos")
    print()
    print("Controles:")
    print("   'q' = Salir")
    print("   'r' = Reiniciar buffers")
    print("   'h' = Mostrar ayuda")
    print("=" * 50)
    
    frame_count = 0
    detecciones = 0
    predicciones_exitosas = 0
    predicciones_estables = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error capturando video")
            break
        
        frame_count += 1
        
        # Flip horizontal para efecto espejo
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        results_hands = hands.process(frame_rgb)
        results_face = face_mesh.process(frame_rgb)
        
        # Dibujar landmarks de manos con mejor visualización
        if results_hands.multi_hand_landmarks:
            detecciones += 1
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
        
        # Intentar predicción
        prediccion_text = "Sin detección de manos"
        confianza_text = ""
        status_text = ""
        color = (128, 128, 128)  # Gris por defecto
        
        try:
            landmarks = extraer_landmarks_robusto(results_hands, results_face)
            
            if np.any(landmarks[:126]):  # Si hay datos de manos
                gesto, confianza, status = predecir_con_suavizado(landmarks)
                
                if gesto:
                    predicciones_exitosas += 1
                    if status == "ESTABLE":
                        predicciones_estables += 1
                        color = (0, 255, 0)  # Verde - predicción estable
                        prediccion_text = f"SEÑA: {gesto.upper()}"
                        confianza_text = f"Confianza: {confianza:.2f}"
                        status_text = "✅ ESTABLE"
                    else:
                        color = (0, 255, 255)  # Amarillo - estabilizando
                        prediccion_text = f"Detectando: {gesto}"
                        confianza_text = f"Confianza: {confianza:.2f}"
                        status_text = status
                else:
                    prediccion_text = "Confianza insuficiente"
                    status_text = status
                    color = (0, 165, 255)  # Naranja
            else:
                prediccion_text = "Coloca las manos en el centro"
                color = (128, 128, 128)
                
        except Exception as e:
            prediccion_text = f"Error: {str(e)[:25]}"
            color = (0, 0, 255)
        
        # Dibujar información en pantalla con mejor diseño
        overlay = frame.copy()
        
        # Fondo semitransparente para texto
        cv2.rectangle(overlay, (10, 10), (w-10, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Texto principal
        cv2.putText(frame, prediccion_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        if confianza_text:
            cv2.putText(frame, confianza_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        if status_text:
            cv2.putText(frame, status_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Estadísticas en la parte inferior
        stats_y = h - 100
        cv2.rectangle(frame, (10, stats_y), (w-10, h-10), (0, 0, 0), -1)
        cv2.putText(frame, f"Frames: {frame_count}", (20, stats_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Detecciones: {detecciones}", (20, stats_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Predicciones: {predicciones_exitosas}", (20, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Estables: {predicciones_estables}", (20, stats_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Buffer: {len(prediction_buffer)}/15", (220, stats_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "'q'=Salir, 'r'=Reset, 'h'=Ayuda", (220, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mostrar frame
        cv2.imshow('LSE Ecuador - Reconocimiento Optimizado', frame)
        
        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("🔄 Reiniciando buffers...")
            prediction_buffer.clear()
            confidence_buffer.clear()
            last_stable_prediction = None
            stability_counter = 0
        elif key == ord('h'):
            print("\n💡 AYUDA PARA MEJOR RECONOCIMIENTO:")
            print("=" * 40)
            print("1. Posición: Manos en el centro de la pantalla")
            print("2. Distancia: 60-80cm de la cámara")
            print("3. Iluminación: Uniforme, sin sombras")
            print("4. Fondo: Simple y contrastante")
            print("5. Gesto: Claro y mantenido por 3-4 segundos")
            print("6. Si falla: Verifica que el gesto esté en el dataset")
            print("=" * 40)
    
    # Limpiar recursos
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face_mesh.close()
    
    # Estadísticas finales
    print(f"\n📊 SESIÓN TERMINADA:")
    print(f"   Frames procesados: {frame_count}")
    print(f"   Detecciones de manos: {detecciones}")
    print(f"   Predicciones realizadas: {predicciones_exitosas}")
    print(f"   Predicciones estables: {predicciones_estables}")
    
    if detecciones > 0:
        precision_rate = (predicciones_estables / detecciones) * 100
        print(f"   Tasa de precisión: {precision_rate:.1f}%")
        
        if precision_rate < 30:
            print("\n🚨 BAJA PRECISIÓN DETECTADA:")
            print("💡 RECOMENDACIONES:")
            print("   1. Ejecuta 'python refuerzo_rapido.py' para mejorar el dataset")
            print("   2. Asegúrate de hacer gestos incluidos en el modelo")
            print("   3. Verifica las condiciones de iluminación")
            print("   4. Reentrena el modelo con más datos")

if __name__ == "__main__":
    main()
