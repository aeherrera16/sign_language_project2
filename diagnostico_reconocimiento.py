#!/usr/bin/env python3
"""
🔧 DIAGNÓSTICO DEL SISTEMA DE RECONOCIMIENTO
===========================================
Script para diagnosticar problemas de reconocimiento de señas
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os
import sys

def verificar_modelo():
    """Verificar que el modelo existe y funciona"""
    print("🔍 VERIFICANDO MODELO...")
    print("=" * 50)
    
    # Verificar archivos del modelo
    archivos_modelo = [
        'model/best_model.h5',
        'model/labels.pkl',
        'gesture_model.h5',
        'labels.txt'
    ]
    
    modelo_encontrado = False
    labels_encontradas = False
    
    for archivo in archivos_modelo:
        if os.path.exists(archivo):
            print(f"✅ {archivo} - ENCONTRADO")
            if 'model' in archivo.lower() and archivo.endswith('.h5'):
                modelo_encontrado = True
            if 'label' in archivo.lower():
                labels_encontradas = True
        else:
            print(f"❌ {archivo} - NO ENCONTRADO")
    
    if not modelo_encontrado:
        print("\n🚨 ERROR CRÍTICO: No se encontró ningún modelo (.h5)")
        print("💡 SOLUCIÓN: Ejecuta 'python train_model.py' primero")
        return False
    
    if not labels_encontradas:
        print("\n🚨 ERROR CRÍTICO: No se encontraron las etiquetas")
        print("💡 SOLUCIÓN: Verifica que existan labels.pkl o labels.txt")
        return False
    
    # Intentar cargar el modelo
    try:
        if os.path.exists('model/best_model.h5'):
            model = tf.keras.models.load_model('model/best_model.h5')
            print(f"✅ Modelo cargado: {model.input_shape}")
            
            with open('model/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
            print(f"✅ Etiquetas cargadas: {len(labels)} clases")
            
        elif os.path.exists('gesture_model.h5'):
            model = tf.keras.models.load_model('gesture_model.h5')
            print(f"✅ Modelo alternativo cargado: {model.input_shape}")
            
            with open('labels.txt', 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines()]
            print(f"✅ Etiquetas alternativas cargadas: {len(labels)} clases")
        
        print("\n📋 PRIMERAS 10 CLASES:")
        for i, label in enumerate(labels[:10]):
            print(f"   {i}: {label}")
            
        return True, model, labels
        
    except Exception as e:
        print(f"\n❌ ERROR CARGANDO MODELO: {e}")
        return False
        
def verificar_camara():
    """Verificar que la cámara funciona"""
    print("\n🎥 VERIFICANDO CÁMARA...")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No se puede acceder a la cámara")
        print("💡 SOLUCIONES:")
        print("   1. Verifica que la cámara esté conectada")
        print("   2. Cierra otras aplicaciones que usen la cámara")
        print("   3. Reinicia el sistema")
        return False
    
    # Verificar resolución
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ Cámara disponible: {width}x{height} @ {fps}FPS")
    
    # Tomar una foto de prueba
    ret, frame = cap.read()
    if ret:
        print("✅ Captura de imagen exitosa")
        cv2.imwrite('test_camera.jpg', frame)
        print("💾 Imagen guardada como 'test_camera.jpg'")
    else:
        print("❌ Error capturando imagen")
        cap.release()
        return False
    
    cap.release()
    return True

def verificar_mediapipe():
    """Verificar MediaPipe"""
    print("\n🤖 VERIFICANDO MEDIAPIPE...")
    print("=" * 50)
    
    try:
        mp_hands = mp.solutions.hands
        mp_face = mp.solutions.face_mesh
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe Hands inicializado")
        print("✅ MediaPipe Face Mesh inicializado")
        
        # Probar con imagen de prueba si existe
        if os.path.exists('test_camera.jpg'):
            image = cv2.imread('test_camera.jpg')
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results_hands = hands.process(image_rgb)
            results_face = face_mesh.process(image_rgb)
            
            if results_hands.multi_hand_landmarks:
                print(f"✅ Manos detectadas: {len(results_hands.multi_hand_landmarks)}")
            else:
                print("⚠️ No se detectaron manos en la imagen de prueba")
                
            if results_face.multi_face_landmarks:
                print(f"✅ Cara detectada: {len(results_face.multi_face_landmarks)}")
            else:
                print("⚠️ No se detectó cara en la imagen de prueba")
        
        hands.close()
        face_mesh.close()
        return True
        
    except Exception as e:
        print(f"❌ Error con MediaPipe: {e}")
        return False

def verificar_landmarks():
    """Verificar extracción de landmarks"""
    print("\n📐 VERIFICANDO EXTRACCIÓN DE LANDMARKS...")
    print("=" * 50)
    
    try:
        from utils import extraer_landmarks
        print("✅ Función extraer_landmarks importada")
        
        # Crear datos de prueba simulados
        test_hands = None
        test_face = None
        
        landmarks = extraer_landmarks(test_hands, test_face)
        print(f"✅ Landmarks extraídos: {len(landmarks)} características")
        
        if len(landmarks) > 0:
            print(f"   📊 Rango de valores: {np.min(landmarks):.3f} - {np.max(landmarks):.3f}")
            return True
        else:
            print("❌ No se extrajeron landmarks")
            return False
            
    except Exception as e:
        print(f"❌ Error extrayendo landmarks: {e}")
        return False

def test_reconocimiento_completo():
    """Prueba completa del sistema de reconocimiento"""
    print("\n🚀 PRUEBA COMPLETA DEL SISTEMA...")
    print("=" * 50)
    
    # Verificaciones previas
    modelo_ok = verificar_modelo()
    if not modelo_ok:
        return False
        
    camara_ok = verificar_camara()
    if not camara_ok:
        return False
        
    mediapipe_ok = verificar_mediapipe()
    if not mediapipe_ok:
        return False
        
    landmarks_ok = verificar_landmarks()
    if not landmarks_ok:
        return False
    
    print("\n✅ TODAS LAS VERIFICACIONES PASARON")
    print("\n🎯 INICIANDO PRUEBA EN VIVO...")
    print("💡 Haz una seña frente a la cámara")
    print("💡 Presiona 'q' para salir")
    
    # Inicializar sistema
    try:
        mp_hands = mp.solutions.hands
        mp_face = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Cargar modelo
        if os.path.exists('model/best_model.h5'):
            model = tf.keras.models.load_model('model/best_model.h5')
            with open('model/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
        else:
            model = tf.keras.models.load_model('gesture_model.h5')
            with open('labels.txt', 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines()]
        
        cap = cv2.VideoCapture(0)
        
        detecciones = 0
        predicciones = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = hands.process(frame_rgb)
            results_face = face_mesh.process(frame_rgb)
            
            # Dibujar landmarks
            if results_hands.multi_hand_landmarks:
                detecciones += 1
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Intentar predicción
                try:
                    from utils import extraer_landmarks
                    landmarks = extraer_landmarks(results_hands, results_face)
                    
                    if len(landmarks) > 0:
                        prediccion = model.predict(landmarks.reshape(1, -1), verbose=0)
                        clase_predicha = np.argmax(prediccion)
                        confianza = float(np.max(prediccion))
                        
                        if confianza > 0.5:
                            predicciones += 1
                            gesto = labels[clase_predicha]
                            
                            # Mostrar predicción
                            text = f"{gesto}: {confianza:.2f}"
                            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                except Exception as e:
                    cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Mostrar estadísticas
            cv2.putText(frame, f"Detecciones: {detecciones}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Predicciones: {predicciones}", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Diagnóstico LSE', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()
        
        print(f"\n📊 RESULTADOS DE LA PRUEBA:")
        print(f"   Detecciones de manos: {detecciones}")
        print(f"   Predicciones realizadas: {predicciones}")
        
        if detecciones == 0:
            print("\n🚨 PROBLEMA: No se detectaron manos")
            print("💡 SOLUCIONES:")
            print("   1. Asegúrate de tener buena iluminación")
            print("   2. Mantén las manos visibles en el centro")
            print("   3. Fondo contrastante (no del color de tu piel)")
            
        elif predicciones == 0:
            print("\n🚨 PROBLEMA: Se detectaron manos pero no hay predicciones")
            print("💡 SOLUCIONES:")
            print("   1. Verifica que el modelo esté entrenado")
            print("   2. Haz gestos más claros y definidos")
            print("   3. Mantén la seña por más tiempo")
            
        else:
            print("\n✅ SISTEMA FUNCIONANDO CORRECTAMENTE")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN PRUEBA COMPLETA: {e}")
        return False

def main():
    """Función principal de diagnóstico"""
    print("🔧 DIAGNÓSTICO DEL SISTEMA LSE ECUADOR")
    print("=" * 60)
    print("Este script verificará todos los componentes del sistema")
    print("=" * 60)
    
    # Ejecutar todas las verificaciones
    print("\n🔍 INICIANDO DIAGNÓSTICO COMPLETO...")
    
    # Verificar si existe el modelo entrenado
    if not os.path.exists('model/best_model.h5') and not os.path.exists('gesture_model.h5'):
        print("\n🚨 PROBLEMA ENCONTRADO: No hay modelo entrenado")
        print("💡 SOLUCIÓN INMEDIATA:")
        print("   1. Ejecuta: python train_model.py")
        print("   2. Espera a que termine el entrenamiento")
        print("   3. Vuelve a ejecutar este diagnóstico")
        return
    
    # Ejecutar prueba completa
    test_reconocimiento_completo()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNÓSTICO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()
