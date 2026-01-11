#!/usr/bin/env python3
"""
ENTRENADOR DE MODELO - Prototipo LSE
Entrena un modelo LSTM con las señas grabadas.
"""

import os
import json
import numpy as np
import pickle
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuración
DATOS_DIR = os.path.join(os.path.dirname(__file__), "datos")
MODELO_DIR = os.path.join(os.path.dirname(__file__), "modelo")
FRAMES = 30
LANDMARKS = 126


def cargar_datos():
    """Carga todas las secuencias grabadas."""
    X, y = [], []
    
    for sena_dir in glob(os.path.join(DATOS_DIR, "*")):
        if not os.path.isdir(sena_dir):
            continue
        
        sena = os.path.basename(sena_dir)
        
        for archivo in glob(os.path.join(sena_dir, "*.json")):
            with open(archivo) as f:
                datos = json.load(f)
                for seq in datos["secuencias"]:
                    if len(seq) == FRAMES:
                        X.append(seq)
                        y.append(sena)
        
        print(f"  {sena}: {sum(1 for label in y if label == sena)} muestras")
    
    return np.array(X), np.array(y)


def crear_modelo(num_clases):
    """Crea modelo LSTM simple."""
    return Sequential([
        LSTM(64, return_sequences=True, input_shape=(FRAMES, LANDMARKS)),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_clases, activation='softmax')
    ])


def main():
    print("\n" + "="*50)
    print("  ENTRENADOR DE MODELO LSE")
    print("="*50 + "\n")
    
    # Cargar datos
    X, y = cargar_datos()
    
    if len(X) == 0:
        print("❌ No hay datos. Ejecuta primero: python 1_grabar_senas.py")
        return
    
    clases = np.unique(y)
    if len(clases) < 2:
        print(f"❌ Necesitas al menos 2 señas. Solo tienes: {clases}")
        return
    
    print(f"\n📊 {len(X)} secuencias, {len(clases)} clases")
    
    # Preparar datos
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    y_cat = to_categorical(y_enc)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    
    # Entrenar
    modelo = crear_modelo(len(clases))
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("\n🚀 Entrenando...")
    modelo.fit(X_train, y_train, validation_data=(X_test, y_test),
               epochs=50, batch_size=16, verbose=1)
    
    # Evaluar
    loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Precisión: {acc*100:.1f}%")
    
    # Guardar
    os.makedirs(MODELO_DIR, exist_ok=True)
    modelo.save(os.path.join(MODELO_DIR, "modelo.h5"))
    
    with open(os.path.join(MODELO_DIR, "clases.pkl"), 'wb') as f:
        pickle.dump(encoder, f)
    
    print(f"💾 Modelo guardado en {MODELO_DIR}/")


if __name__ == "__main__":
    main()
