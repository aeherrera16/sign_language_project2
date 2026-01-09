#!/usr/bin/env python3
"""
=============================================================================
MÓDULO 2: ENTRENAMIENTO DEL MODELO LSTM
=============================================================================
Este script entrena un modelo LSTM para reconocer señas dinámicas.

USO:
    python 2_entrenar_modelo.py

REQUISITOS:
    - Tener secuencias grabadas en la carpeta 'datos/'
    - Al menos 2 señas diferentes con 10+ secuencias cada una
=============================================================================
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Configuración
DATOS_DIR = os.path.join(os.path.dirname(__file__), "datos")
MODELO_DIR = os.path.join(os.path.dirname(__file__), "modelo")
SECUENCIA_FRAMES = 30
LANDMARKS_SIZE = 126  # 2 manos × 21 puntos × 3 coordenadas

def cargar_datos():
    """Carga todas las secuencias grabadas."""
    X = []  # Secuencias
    y = []  # Etiquetas
    
    if not os.path.exists(DATOS_DIR):
        print("❌ No existe la carpeta de datos")
        return None, None
    
    senas = [d for d in os.listdir(DATOS_DIR) if os.path.isdir(os.path.join(DATOS_DIR, d))]
    
    if not senas:
        print("❌ No hay señas grabadas")
        return None, None
    
    print(f"\n📂 Señas encontradas: {len(senas)}")
    
    for sena in senas:
        sena_dir = os.path.join(DATOS_DIR, sena)
        archivos = [f for f in os.listdir(sena_dir) if f.endswith('.json')]
        
        total_secuencias = 0
        for archivo in archivos:
            with open(os.path.join(sena_dir, archivo), 'r') as f:
                datos = json.load(f)
                for secuencia in datos['secuencias']:
                    # Asegurar que la secuencia tiene el tamaño correcto
                    secuencia = np.array(secuencia)
                    if len(secuencia) == SECUENCIA_FRAMES:
                        X.append(secuencia)
                        y.append(sena)
                        total_secuencias += 1
        
        print(f"  ✓ {sena}: {total_secuencias} secuencias")
    
    return np.array(X), np.array(y)

def crear_modelo(num_clases):
    """Crea el modelo LSTM."""
    model = Sequential([
        # Capa LSTM 1
        LSTM(64, return_sequences=True, input_shape=(SECUENCIA_FRAMES, LANDMARKS_SIZE)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capa LSTM 2
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capa LSTM 3
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capas densas
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        
        # Capa de salida
        Dense(num_clases, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("=" * 60)
    print("   ENTRENAMIENTO DE MODELO LSTM PARA LSE")
    print("=" * 60)
    
    # Cargar datos
    X, y = cargar_datos()
    
    if X is None or len(X) == 0:
        print("\n❌ No hay datos suficientes para entrenar")
        print("   Usa primero: python 1_grabar_senas.py")
        return
    
    # Verificar mínimo de datos
    clases_unicas = np.unique(y)
    if len(clases_unicas) < 2:
        print(f"\n❌ Se necesitan al menos 2 señas diferentes")
        print(f"   Actualmente solo hay: {clases_unicas}")
        return
    
    print(f"\n📊 Total de secuencias: {len(X)}")
    print(f"📊 Total de clases: {len(clases_unicas)}")
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📈 Datos de entrenamiento: {len(X_train)}")
    print(f"📈 Datos de prueba: {len(X_test)}")
    
    # Crear modelo
    model = crear_modelo(len(clases_unicas))
    model.summary()
    
    # Callbacks
    os.makedirs(MODELO_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(MODELO_DIR, 'mejor_modelo.h5'),
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    print("-" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluar
    print("\n" + "=" * 60)
    print("   RESULTADOS")
    print("=" * 60)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Precisión final: {accuracy * 100:.2f}%")
    print(f"✅ Pérdida final: {loss:.4f}")
    
    # Guardar modelo y etiquetas
    model.save(os.path.join(MODELO_DIR, 'modelo_lstm.h5'))
    
    with open(os.path.join(MODELO_DIR, 'etiquetas.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Guardar lista de señas
    with open(os.path.join(MODELO_DIR, 'senas.json'), 'w') as f:
        json.dump({
            'senas': list(label_encoder.classes_),
            'num_clases': len(clases_unicas),
            'precision': float(accuracy)
        }, f, indent=2)
    
    print(f"\n💾 Modelo guardado en: {MODELO_DIR}/")
    print(f"   - modelo_lstm.h5")
    print(f"   - etiquetas.pkl")
    print(f"   - senas.json")
    
    # Convertir a TFLite para Raspberry Pi
    print("\n🔄 Convirtiendo a TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(os.path.join(MODELO_DIR, 'modelo.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    print(f"   - modelo.tflite (para Raspberry Pi)")
    print("\n✅ ¡Entrenamiento completado!")

if __name__ == "__main__":
    main()
