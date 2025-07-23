#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTRENAMIENTO CON DATASET OPTIMIZADO (SOLO MANOS)
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def load_optimized_dataset(data_path="data_hands_only"):
    """Carga dataset optimizado"""
    X, y = [], []
    
    for gesture_folder in os.listdir(data_path):
        gesture_path = os.path.join(data_path, gesture_folder)
        
        if os.path.isdir(gesture_path):
            for file in os.listdir(gesture_path):
                if file.endswith('.npy'):
                    try:
                        landmarks = np.load(os.path.join(gesture_path, file))
                        if landmarks.shape[0] == 126:  # Verificar que sean 126 features
                            X.append(landmarks)
                            y.append(gesture_folder)
                    except:
                        continue
    
    return np.array(X), np.array(y)

def create_optimized_model(input_shape=(126,), num_classes=205):
    """Modelo optimizado para landmarks de manos únicamente"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Capas específicamente diseñadas para landmarks de manos
        tf.keras.layers.Dense(256, activation='relu', name='hands_embedding'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation='relu', name='gesture_features'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu', name='final_features'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Capa de salida
        tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model

def main():
    """Entrenamiento principal"""
    print("🚀 ENTRENANDO MODELO OPTIMIZADO (SOLO MANOS)")
    
    # Cargar datos
    X, y = load_optimized_dataset()
    print(f"📊 Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} features")
    
    # Encodear etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    
    # Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )
    
    # Crear modelo
    model = create_optimized_model(num_classes=len(le.classes_))
    
    # Entrenar
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
        ]
    )
    
    # Evaluar
    test_loss, test_acc, test_top3 = model.evaluate(X_test, y_test)
    print(f"🎯 Precisión en test: {test_acc:.4f}")
    print(f"🎯 Top-3 precisión: {test_top3:.4f}")
    
    # Guardar modelo
    model.save('model/optimized_hands_only_model.h5')
    
    with open('model/optimized_labels.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("✅ Modelo optimizado guardado")

if __name__ == "__main__":
    main()
