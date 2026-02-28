#!/usr/bin/env python3
"""
ENTRENADOR LSTM - Reconocimiento de Señas Dinámicas
Basado en: Sincan & Keles (2020), Basnin et al. (2021)

Arquitectura: LSTM multicapa para secuencias temporales de landmarks.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import json
import numpy as np
import pickle
from glob import glob

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURACIÓN ===
DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")
DIR_MODELO = os.path.join(os.path.dirname(__file__), "modelo")

FRAMES = 30
FEATURES = 126


def cargar_datos():
    """Carga todas las secuencias de todas las señas."""
    X, y = [], []
    
    print("\n📂 Cargando datos...")
    
    for sena_dir in sorted(glob(os.path.join(DIR_DATOS, "*"))):
        if not os.path.isdir(sena_dir):
            continue
        
        sena = os.path.basename(sena_dir)
        count = 0
        
        for archivo in glob(os.path.join(sena_dir, "*.json")):
            with open(archivo) as f:
                datos = json.load(f)
                for seq in datos["secuencias"]:
                    seq = np.array(seq)
                    if seq.shape == (FRAMES, FEATURES):
                        X.append(seq)
                        y.append(sena)
                        count += 1
        
        print(f"   {sena}: {count} secuencias")
    
    return np.array(X), np.array(y)


def crear_modelo(num_clases):
    """
    Arquitectura LSTM basada en papers:
    - Sincan & Keles (2020): CNN + LSTM → 95%
    - Basnin et al. (2021): CNN + LSTM → 88.5%
    """
    model = Sequential([
        # LSTM 1: Captura patrones temporales iniciales
        LSTM(64, return_sequences=True, input_shape=(FRAMES, FEATURES)),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM 2: Patrones más complejos
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM 3: Representación final
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Clasificador
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_clases, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    print("\n" + "="*60)
    print("  ENTRENADOR DE MODELO LSTM")
    print("  Técnica: Secuencias temporales de MediaPipe landmarks")
    print("="*60)
    
    # Cargar datos
    X, y = cargar_datos()
    
    if len(X) == 0:
        print("\n❌ No hay datos. Ejecuta: python 1_grabar_senas.py")
        return
    
    clases = np.unique(y)
    if len(clases) < 2:
        print(f"\n❌ Necesitas mínimo 2 señas. Solo tienes: {list(clases)}")
        return
    
    print(f"\n📊 Resumen:")
    print(f"   Secuencias: {len(X)}")
    print(f"   Clases: {len(clases)} → {list(clases)}")
    print(f"   Shape: {X.shape}")
    
    # Codificar etiquetas
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    y_cat = to_categorical(y_enc)
    
    # Dividir datos (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Crear modelo
    modelo = crear_modelo(len(clases))
    modelo.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
    ]
    
    # Entrenar
    print("\n🚀 Entrenando...")
    print("-"*60)
    
    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluar en TRAIN y TEST para detectar overfitting
    loss_test, acc_test = modelo.evaluate(X_test, y_test, verbose=0)
    loss_train, acc_train = modelo.evaluate(X_train, y_train, verbose=0)
    gap = acc_train - acc_test
    
    print(f"\n✅ Accuracy TEST (datos no vistos):  {acc_test*100:.2f}%")
    print(f"   Accuracy TRAIN (datos vistos):    {acc_train*100:.2f}%")
    print(f"   Gap (overfitting):                {gap*100:.2f}%")
    
    if gap > 0.15:
        print(f"   ⚠️ Gap alto: el modelo memoriza más de lo que aprende")
        print(f"   💡 Graba más secuencias por seña para mejorar")
    
    # Reporte detallado
    y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\n📋 Reporte de clasificación (TEST set):")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))
    
    # Guardar modelo
    os.makedirs(DIR_MODELO, exist_ok=True)
    
    modelo.save(os.path.join(DIR_MODELO, "modelo.h5"))
    
    with open(os.path.join(DIR_MODELO, "encoder.pkl"), 'wb') as f:
        pickle.dump(encoder, f)
    
    # Guardar info con métricas honestas
    with open(os.path.join(DIR_MODELO, "info.json"), 'w') as f:
        json.dump({
            "clases": list(encoder.classes_),
            "accuracy_test": float(acc_test),
            "accuracy_train": float(acc_train),
            "gap_overfitting": float(gap),
            "frames": FRAMES,
            "features": FEATURES,
            "muestras_train": int(len(X_train)),
            "muestras_test": int(len(X_test)),
            "nota": "accuracy_test es la métrica real (datos no vistos)"
        }, f, indent=2)
    
    print(f"\n💾 Modelo guardado en: {DIR_MODELO}/")
    print("   - modelo.h5")
    print("   - encoder.pkl")
    print("   - info.json")


if __name__ == "__main__":
    main()
