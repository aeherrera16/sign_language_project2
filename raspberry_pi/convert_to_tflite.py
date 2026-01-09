#!/usr/bin/env python3
"""
Convierte el modelo Keras (.h5) a TensorFlow Lite (.tflite)
Optimizado para Raspberry Pi 4

Uso:
    python convert_to_tflite.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Rutas - Buscar modelo en ubicación del proyecto principal
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent  # Subir un nivel desde raspberry_pi/
MODEL_DIR = PROJECT_DIR / "backend" / "model"
H5_PATH = MODEL_DIR / "best_model.h5"
TFLITE_PATH = MODEL_DIR / "model.tflite"

def convert_model():
    print("═" * 60)
    print("🔄 CONVERSIÓN DE MODELO A TFLITE")
    print("═" * 60)
    
    # Verificar modelo H5
    if not H5_PATH.exists():
        print(f"❌ No se encontró el modelo: {H5_PATH}")
        sys.exit(1)
    
    print(f"📂 Modelo origen: {H5_PATH}")
    print(f"📂 Destino: {TFLITE_PATH}")
    print()
    
    try:
        # Suprimir warnings de TensorFlow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        import tensorflow as tf
        import numpy as np
        print(f"✅ TensorFlow versión: {tf.__version__}")
        
        # Cargar modelo
        print("\n📥 Cargando modelo Keras...")
        model = tf.keras.models.load_model(str(H5_PATH))
        
        # Mostrar resumen
        print("\n📊 Arquitectura del modelo:")
        model.summary()
        
        # Método alternativo: Crear modelo sin BatchNorm training mode
        print("\n🔄 Preparando modelo para conversión...")
        
        # Obtener la forma de entrada del modelo
        input_shape = model.input_shape[1:]  # Ignorar batch dimension
        print(f"   Forma de entrada: {input_shape}")
        
        # Crear función concreta para conversión
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, input_shape[0]], dtype=tf.float32)])
        def serving_fn(x):
            return model(x, training=False)
        
        # Obtener la función concreta
        concrete_func = serving_fn.get_concrete_function()
        
        print("\n🔄 Convirtiendo a TFLite...")
        
        # Convertir usando la función concreta
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Optimizaciones para Raspberry Pi
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convertir
        tflite_model = converter.convert()
        
        # Guardar
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        
        # Verificar tamaño
        h5_size = H5_PATH.stat().st_size / 1024  # KB
        tflite_size = TFLITE_PATH.stat().st_size / 1024  # KB
        reduction = (1 - tflite_size / h5_size) * 100
        
        print("\n" + "═" * 60)
        print("✅ CONVERSIÓN COMPLETADA")
        print("═" * 60)
        print(f"   Modelo H5:     {h5_size:.1f} KB")
        print(f"   Modelo TFLite: {tflite_size:.1f} KB")
        print(f"   Reducción:     {reduction:.1f}%")
        print()
        print(f"📱 Modelo listo para Raspberry Pi: {TFLITE_PATH}")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow no está instalado")
        print("   Instala con: pip install tensorflow")
        return False
        
    except Exception as e:
        print(f"❌ Error en conversión: {e}")
        print("\n💡 Alternativa: El modelo H5 también funciona en la RPi con TensorFlow completo")
        print("   El script traductor_portable.py intentará usar H5 como fallback.")
        return False


def verify_tflite():
    """Verifica que el modelo TFLite funcione"""
    if not TFLITE_PATH.exists():
        print("❌ Modelo TFLite no existe")
        return False
    
    try:
        import tensorflow as tf
        import numpy as np
        
        print("\n🔍 Verificando modelo TFLite...")
        
        # Cargar intérprete
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
        interpreter.allocate_tensors()
        
        # Obtener detalles
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Entrada: {input_details[0]['shape']} ({input_details[0]['dtype']})")
        print(f"   Salida:  {output_details[0]['shape']} ({output_details[0]['dtype']})")
        
        # Prueba con datos aleatorios
        input_shape = input_details[0]['shape']
        test_input = np.random.random(input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   Predicción de prueba: {output[0][:3]}... (primeros 3 valores)")
        print("✅ Modelo TFLite verificado correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando TFLite: {e}")
        return False


if __name__ == "__main__":
    success = convert_model()
    if success:
        verify_tflite()
