# -*- coding: utf-8 -*-
import os
# Configurar TensorFlow para evitar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import importlib
import subprocess

def test_import(module_name, description=""):
    """Prueba la importacion de un modulo"""
    try:
        importlib.import_module(module_name)
        print(f" {module_name:<20} - {description}")
        return True
    except ImportError as e:
        print(f" {module_name:<20} - {description} | Error: {e}")
        return False

def test_specific_functionality():
    """Prueba funcionalidades especificas"""
    print("\n🔧 PRUEBAS DE FUNCIONALIDAD:")
    
    # Test OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(" Camara                - Acceso correcto")
            cap.release()
        else:
            print("⚠️ Camara                - No se pudo acceder")
    except Exception as e:
        print(f" Camara                - Error: {e}")
    
    # Test MediaPipe
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        print(" MediaPipe Hands       - Inicializacion correcta")
    except Exception as e:
        print(f" MediaPipe Hands       - Error: {e}")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f" TensorFlow            - Version {tf.__version__}")
        
        # Test GPU si esta disponible
        if tf.config.list_physical_devices('GPU'):
            print(" GPU                   - Detectada y disponible")
        else:
            print("ℹ️ GPU                   - No detectada (usando CPU)")
    except Exception as e:
        print(f" TensorFlow            - Error: {e}")
    
    # Test Text-to-Speech
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print(f" Text-to-Speech        - {len(voices) if voices else 0} voces disponibles")
    except Exception as e:
        print(f" Text-to-Speech        - Error: {e}")

def get_system_info():
    """Obtiene informacion del sistema"""
    print("\n💻 INFORMACION DEL SISTEMA:")
    
    print(f"  Python: {sys.version}")
    print(f"  Plataforma: {sys.platform}")
    
    try:
        import platform
        print(f"  SO: {platform.system()} {platform.release()}")
        print(f"  Arquitectura: {platform.machine()}")
    except:
        pass

def install_missing_packages():
    """Instala paquetes faltantes"""
    print("\n INSTALACION DE PAQUETES FALTANTES:")
    
    missing_packages = []
    
    # Lista de paquetes requeridos
    required_packages = [
        ('opencv-python', 'cv2'),
        ('mediapipe', 'mediapipe'),
        ('tensorflow', 'tensorflow'),
        ('scikit-learn', 'sklearn'),
        ('numpy', 'numpy'),
        ('pyttsx3', 'pyttsx3'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pandas', 'pandas')
    ]
    
    for package_name, import_name in required_packages:
        if not test_import(import_name, f"Requerido para el proyecto"):
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n📋 Paquetes faltantes: {', '.join(missing_packages)}")
        response = input("\nDeseas instalar los paquetes faltantes? (s/n): ").lower().strip()
        
        if response in ['s', 'si', 'si', 'y', 'yes']:
            for package in missing_packages:
                try:
                    print(f"📥 Instalando {package}...")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f" {package} instalado correctamente")
                except subprocess.CalledProcessError as e:
                    print(f" Error instalando {package}: {e}")
            
            print("\n Re-ejecutando pruebas...")
            main()
        else:
            print("ℹ️ Instalacion cancelada")
    else:
        print(" Todos los paquetes estan instalados")

def main():
    """Funcion principal"""
    print("🧪 PRUEBA DE DEPENDENCIAS DEL PROYECTO")
    print("=" * 60)
    
    # Informacion del sistema
    get_system_info()
    
    print("\n📚 PRUEBAS DE IMPORTACION:")
    
    # Dependencias principales
    modules_to_test = [
        ('cv2', 'OpenCV - Procesamiento de video'),
        ('mediapipe', 'MediaPipe - Deteccion de landmarks'),
        ('numpy', 'NumPy - Computacion numerica'),
        ('tensorflow', 'TensorFlow - Machine Learning'),
        ('sklearn', 'Scikit-learn - Utilidades ML'),
        ('pyttsx3', 'pyttsx3 - Text-to-Speech'),
        ('matplotlib', 'Matplotlib - Visualizacion'),
        ('matplotlib.pyplot', 'Matplotlib PyPlot'),
        ('seaborn', 'Seaborn - Visualizacion estadistica'),
        ('pandas', 'Pandas - Manipulacion de datos'),
        ('pickle', 'Pickle - Serializacion'),
        ('json', 'JSON - Manipulacion de datos'),
        ('threading', 'Threading - Multihilo'),
        ('tkinter', 'Tkinter - GUI'),
        ('os', 'OS - Sistema operativo'),
        ('sys', 'Sys - Sistema'),
        ('time', 'Time - Tiempo'),
        ('datetime', 'DateTime - Fecha y hora'),
        ('collections', 'Collections - Estructuras de datos')
    ]
    
    successful_imports = 0
    total_imports = len(modules_to_test)
    
    for module, description in modules_to_test:
        if test_import(module, description):
            successful_imports += 1
    
    # Pruebas de funcionalidad
    test_specific_functionality()
    
    # Resumen
    print(f"\n📊 RESUMEN:")
    print(f"  Importaciones exitosas: {successful_imports}/{total_imports}")
    print(f"  Porcentaje de exito: {successful_imports/total_imports*100:.1f}%")
    
    if successful_imports == total_imports:
        print("\n Todas las dependencias estan correctamente instaladas!")
        print(" El proyecto esta listo para ejecutarse")
    else:
        print(f"\n⚠️ Faltan {total_imports - successful_imports} dependencias")
        install_missing_packages()

if __name__ == "__main__":
    main()
