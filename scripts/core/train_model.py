# -*- coding: utf-8 -*-
import os
# Configurar TensorFlow para evitar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
import json
from datetime import datetime

def cargar_datos():
    X, y = [], []

    if not os.path.exists("data"):
        raise FileNotFoundError(" La carpeta 'data' no existe. Graba gestos primero.")

    gestures = sorted(os.listdir("data"))
    if not gestures:
        raise ValueError(" No hay gestos en la carpeta 'data'. Graba gestos primero.")

    print("📂 Gestos encontrados:", gestures)

    for label, gesture in enumerate(gestures):
        gesture_path = os.path.join("data", gesture)
        if not os.path.isdir(gesture_path):
            continue
            
        gesture_samples = 0
        for file in os.listdir(gesture_path):
            if file.endswith(".npy"):
                try:
                    sample = np.load(os.path.join(gesture_path, file))
                    # Verificar que el sample tenga 126 dimensiones (solo manos)
                    if sample.shape == (126,):
                        X.append(sample)
                        y.append(label)
                        gesture_samples += 1
                    else:
                        print(f"⚠️ Archivo ignorado por forma inválida: {file} ({sample.shape}) - Se esperan 126 dimensiones")
                except Exception as e:
                    print(f"⚠️ Error al procesar {file}: {e}")
        
        print(f"  {gesture}: {gesture_samples} muestras")

    if not X:
        raise ValueError(" No se encontraron muestras validas para entrenamiento.")

    return np.array(X), np.array(y), gestures

def construir_modelo_mejorado(num_clases):
    """Modelo optimizado para landmarks de MANOS ÚNICAMENTE (126 dimensiones)"""
    
    model = tf.keras.Sequential([
        # Capa de entrada con normalizacion (SOLO MANOS: 126 dimensiones)
        tf.keras.layers.Input(shape=(126,)),
        tf.keras.layers.BatchNormalization(),
        
        # Primera capa densa (reducida para menor complejidad)
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Segunda capa densa
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Tercera capa densa
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        
        # Capa de salida
        tf.keras.layers.Dense(num_clases, activation='softmax')
    ])

    # Optimizador mejorado
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    
    return model

def data_augmentation(X, y, augment_factor=2):
    """Aumenta los datos aplicando pequenas variaciones"""
    
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        # Datos originales
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        # Crear variaciones
        for _ in range(augment_factor):
            # Agregar ruido gaussiano pequeno
            noise = np.random.normal(0, 0.01, X[i].shape)
            augmented_sample = X[i] + noise
            
            X_augmented.append(augmented_sample)
            y_augmented.append(y[i])
    
    return np.array(X_augmented), np.array(y_augmented)

def main():
    print(" ENTRENAMIENTO DEL MODELO MEJORADO")
    print("=" * 50)
    
    # Cargar datos
    X, y, gestures = cargar_datos()
    print(f"\n📊 Dataset cargado:")
    print(f"  - Muestras totales: {len(X)}")
    print(f"  - Clases: {len(gestures)}")
    print(f"  - Forma de entrada: {X.shape}")
    
    # Verificar distribucion de clases
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📈 Distribucion por clase:")
    for i, (gesture, count) in enumerate(zip(gestures, counts)):
        print(f"  {gesture}: {count} muestras")
    
    # Aplicar aumento de datos si hay pocas muestras
    min_samples = min(counts)
    if min_samples < 50:
        print(f"\n Aplicando aumento de datos (minimo: {min_samples} muestras)")
        X, y = data_augmentation(X, y, augment_factor=1)
        print(f"  - Nuevas muestras totales: {len(X)}")
    
    # Normalizacion de datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\n📂 Division de datos:")
    print(f"  - Entrenamiento: {len(X_train)} muestras")
    print(f"  - Prueba: {len(X_test)} muestras")

    os.makedirs("model", exist_ok=True)

    # Construir modelo mejorado
    model = construir_modelo_mejorado(num_clases=len(gestures))
    
    print(f"\n🏗️ Arquitectura del modelo:")
    model.summary()

    # Callbacks mejorados
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        "model/best_model.h5", 
        monitor="val_accuracy", 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )

    print(f"\n Iniciando entrenamiento...")
    
    # Entrenamiento
    start_time = datetime.now()
    
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"\n Tiempo de entrenamiento: {training_time:.2f} segundos")

    # Guardar modelo final
    model.save("model/gesture_model.h5")
    print(" Modelo guardado en model/gesture_model.h5")

    # Guardar scaler
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(" Normalizador guardado en model/scaler.pkl")

    # Guardar etiquetas
    with open("model/labels.pkl", "wb") as f:
        pickle.dump(gestures, f)
    print(" Etiquetas guardadas en model/labels.pkl")

    # Guardar historial de entrenamiento
    history_dict = history.history.copy()
    history_dict['training_time'] = training_time
    history_dict['timestamp'] = datetime.now().isoformat()
    
    with open("model/training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    print(" Historial guardado en model/training_history.json")

    # Evaluacion final
    print(f"\n🧪 Evaluacion final:")
    loss, acc, top_k_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  - Perdida: {loss:.4f}")
    print(f"  - Precision: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - Top-K Precision: {top_k_acc:.4f}")

    # Predicciones y reporte detallado
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(f"\n📋 Reporte de clasificacion:")
    print(classification_report(y_test, y_pred, target_names=gestures))
    
    # Guardar metricas finales
    final_metrics = {
        'timestamp': datetime.now().isoformat(),
        'training_time': training_time,
        'final_loss': float(loss),
        'final_accuracy': float(acc),
        'final_top_k_accuracy': float(top_k_acc),
        'num_epochs': len(history.history['loss']),
        'num_samples_train': len(X_train),
        'num_samples_test': len(X_test),
        'num_classes': len(gestures),
        'class_names': gestures,
        'data_augmented': bool(min_samples < 50)
    }
    
    # Convertir tipos numpy a tipos nativos de Python para JSON
    final_metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in final_metrics.items()}
    
    with open("model/final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n Entrenamiento completado exitosamente!")
    print(f"📁 Archivos guardados en la carpeta 'model/'")

if __name__ == "__main__":
    main()
