#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# Configurar TensorFlow para evitar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

"""
RECONOCIMIENTO LSE SIMPLIFICADO Y ROBUSTO
===========================================
Version simplificada que garantiza el funcionamiento
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os

def main():
    print("INICIANDO RECONOCIMIENTO LSE SIMPLIFICADO...")
    print("=" * 50)
    
    # 1. CARGAR MODELO
    print("Cargando modelo...")
    try:
        if os.path.exists('model/best_model.h5'):
            model = tf.keras.models.load_model('model/best_model.h5')
            with open('model/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
            print(f"Modelo cargado: {len(labels)} clases")
        else:
            print("No se encontro el modelo. Ejecuta 'python scripts/core/train_model.py' primero")
            return
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return
    
    # 2. INICIALIZAR MEDIAPIPE
    print("Inicializando MediaPipe...")
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    # Configuracion mas permisiva para mejor deteccion
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,  # Mas bajo para mejor deteccion
        min_tracking_confidence=0.3    # Mas bajo para mejor deteccion
    )
    
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,  # Mas bajo para mejor deteccion
        min_tracking_confidence=0.3    # Mas bajo para mejor deteccion
    )
    
    print(" MediaPipe inicializado")
    
    # 3. INICIALIZAR CAMARA
    print("🎥 Inicializando camara...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" No se puede acceder a la camara")
        return
    
    # Configurar resolucion
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(" Camara inicializada")
    
    # 4. FUNCION DE EXTRACCION SIMPLIFICADA
    def extraer_landmarks_simple(results_hands, results_face):
        """Extraccion simplificada y robusta de landmarks"""
        landmarks = []
        
        # Manos (126 elementos)
        if results_hands and hasattr(results_hands, 'multi_hand_landmarks') and results_hands.multi_hand_landmarks:
            hands_data = []
            for hand in results_hands.multi_hand_landmarks[:2]:  # Maximo 2 manos
                hand_points = []
                for lm in hand.landmark:
                    hand_points.extend([lm.x, lm.y, lm.z])
                hands_data.extend(hand_points)
            
            # Asegurar 126 elementos (2 manos)
            while len(hands_data) < 126:
                hands_data.append(0.0)
            landmarks.extend(hands_data[:126])
        else:
            landmarks.extend([0.0] * 126)  # Sin manos detectadas
        
        # Cara (1404 elementos) - simplificado
        if results_face and hasattr(results_face, 'multi_face_landmarks') and results_face.multi_face_landmarks:
            face = results_face.multi_face_landmarks[0]
            face_points = []
            for lm in face.landmark:
                face_points.extend([lm.x, lm.y, lm.z])
            landmarks.extend(face_points[:1404])
            
            # Completar si falta
            while len(landmarks) < 1530:
                landmarks.append(0.0)
        else:
            landmarks.extend([0.0] * 1404)  # Sin cara detectada
        
        return np.array(landmarks[:1530])  # Asegurar tamano correcto
    
    # 5. LOOP PRINCIPAL
    print("\n SISTEMA LISTO - Haz senas frente a la camara")
    print(" Consejos:")
    print("   - Manten buena iluminacion")
    print("   - Fondo contrastante")
    print("   - Manos visibles en el centro")
    print("   - Presiona 'q' para salir")
    print("   - Presiona 'r' para reiniciar deteccion")
    
    frame_count = 0
    detecciones = 0
    predicciones_exitosas = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Error capturando video")
            break
        
        frame_count += 1
        
        # Flip horizontal para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        results_hands = hands.process(frame_rgb)
        results_face = face_mesh.process(frame_rgb)
        
        # Dibujar landmarks de manos
        if results_hands.multi_hand_landmarks:
            detecciones += 1
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
        
        # Intentar prediccion
        prediccion_text = "Sin deteccion"
        confianza_text = ""
        
        try:
            landmarks = extraer_landmarks_simple(results_hands, results_face)
            
            if np.any(landmarks[:126]):  # Si hay datos de manos
                # Realizar prediccion
                prediccion = model.predict(landmarks.reshape(1, -1), verbose=0)
                clase_predicha = np.argmax(prediccion)
                confianza = float(np.max(prediccion))
                
                if confianza > 0.3:  # Umbral mas bajo para ver mas predicciones
                    predicciones_exitosas += 1
                    gesto = labels[clase_predicha]
                    prediccion_text = f"Sena: {gesto}"
                    confianza_text = f"Confianza: {confianza:.2f}"
                    
                    # Color basado en confianza
                    if confianza > 0.7:
                        color = (0, 255, 0)  # Verde - alta confianza
                    elif confianza > 0.5:
                        color = (0, 255, 255)  # Amarillo - media confianza
                    else:
                        color = (0, 165, 255)  # Naranja - baja confianza
                else:
                    prediccion_text = "Confianza muy baja"
                    color = (0, 0, 255)  # Rojo
            else:
                prediccion_text = "No se detectan manos"
                color = (128, 128, 128)  # Gris
                
        except Exception as e:
            prediccion_text = f"Error: {str(e)[:30]}"
            color = (0, 0, 255)
        
        # Mostrar informacion en pantalla
        cv2.putText(frame, prediccion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if confianza_text:
            cv2.putText(frame, confianza_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Estadisticas
        cv2.putText(frame, f"Frames: {frame_count}", (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Detecciones: {detecciones}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Predicciones: {predicciones_exitosas}", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar frame
        cv2.imshow('LSE Ecuador - Reconocimiento Simplificado', frame)
        
        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print(" Reiniciando contadores...")
            frame_count = 0
            detecciones = 0
            predicciones_exitosas = 0
    
    # Limpiar recursos
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face_mesh.close()
    
    print(f"\n📊 SESION TERMINADA:")
    print(f"   Frames procesados: {frame_count}")
    print(f"   Detecciones de manos: {detecciones}")
    print(f"   Predicciones exitosas: {predicciones_exitosas}")
    
    if detecciones == 0:
        print("\n🚨 CONSEJOS PARA MEJORAR LA DETECCION:")
        print("   1.  Mejora la iluminacion")
        print("   2.  Manten las manos bien visibles")
        print("   3.  Usa un fondo contrastante")
        print("   4.  Manten distancia adecuada de la camara")
    elif predicciones_exitosas == 0:
        print("\n🚨 CONSEJOS PARA MEJORAR LAS PREDICCIONES:")
        print("   1.  Haz gestos mas claros y definidos")
        print("   2.  Manten la sena por mas tiempo")
        print("   3. 📚 Usa gestos que esten en el dataset")
        print("   4.  Entrena mas el modelo si es necesario")

if __name__ == "__main__":
    main()
