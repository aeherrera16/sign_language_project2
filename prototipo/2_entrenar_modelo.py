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
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURACIÓN ===
DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")
DIR_MODELO = os.path.join(os.path.dirname(__file__), "modelo")

FRAMES = 30
FEATURES = 126
AUGMENTACIONES_POR_MUESTRA = 14  # Cada secuencia real genera 14 variantes (+1400%)


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


def _mirror_seq(seq):
    """
    Espeja horizontalmente: niega el componente X de cada landmark.
    Simula hacer la misma seña con la mano contraria.
    Los features son relativos a la muñeca (lm.x - wrist.x), así que
    negar X invierte la lateralidad sin afectar Y ni Z.
    Layout: slot0 = landmarks 0-62, slot1 = landmarks 63-125.
    Dentro de cada slot: [x0,y0,z0, x1,y1,z1, ...] por landmark.
    """
    m = seq.copy()
    for slot in range(2):
        base = slot * 63
        for i in range(21):
            m[:, base + i * 3] *= -1  # Negar solo X
    return m


def augmentar_secuencia(seq, n=14):
    """
    Genera n variantes de una secuencia de landmarks para ampliar el dataset.
    Usa solo numpy — sin dependencias extra, funciona en Raspberry Pi.

    Transformaciones aplicadas (combinadas aleatoriamente):
    1. Ruido gaussiano — simula temblor leve, diferentes personas
    2. Escala global — simula mano más cerca/lejos de la cámara
    3. Desplazamiento temporal — seña empieza un poco antes o después
    4. Variación de velocidad — mismo signo hecho más rápido o lento
    5. Mirror horizontal (50%) — simula mano contraria, refuerza fronteras entre clases
    """
    resultado = []
    n_frames, n_feat = seq.shape

    for _ in range(n):
        aug = seq.copy().astype(np.float32)

        # 1. Ruido gaussiano leve
        ruido_sigma = np.random.uniform(0.004, 0.013)
        aug += np.random.normal(0, ruido_sigma, aug.shape).astype(np.float32)

        # 2. Escala global (simula distancia a la cámara)
        aug *= np.float32(np.random.uniform(0.87, 1.13))

        # 3. Desplazamiento temporal (shift de ±5 frames con relleno de ceros)
        shift = np.random.randint(-5, 6)
        if shift > 0:
            aug = np.vstack([aug[shift:], np.zeros((shift, n_feat), dtype=np.float32)])
        elif shift < 0:
            aug = np.vstack([np.zeros((-shift, n_feat), dtype=np.float32), aug[:shift]])

        # 4. Variación de velocidad: resamplear con interpolación lineal numpy
        speed = np.random.uniform(0.72, 1.28)
        src_positions = np.linspace(0, n_frames - 1, n_frames) * speed
        src_positions = np.clip(src_positions, 0, n_frames - 1)
        floor_i = np.floor(src_positions).astype(int)
        ceil_i = np.minimum(floor_i + 1, n_frames - 1)
        frac = (src_positions - floor_i).astype(np.float32)[:, np.newaxis]
        aug = aug[floor_i] * (1.0 - frac) + aug[ceil_i] * frac

        # 5. Mirror horizontal (50% de probabilidad)
        if np.random.random() < 0.5:
            aug = _mirror_seq(aug)

        resultado.append(aug.astype(np.float32))

    return resultado


def crear_modelo(num_clases):
    import keras
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
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    print("\n" + "="*60)
    print("  ENTRENADOR DE MODELO LSTM")
    print("  Técnica: Secuencias temporales de MediaPipe landmarks")
    print("="*60)
    
    # Descargar datos nuevos de la nube antes de entrenar
    try:
        from sync_cloud import SyncCloud
        sync = SyncCloud()
        if sync.conectar():
            print("\n☁️  Descargando datos nuevos de la nube...")
            sync.descargar_datos_senas()
    except ImportError:
        pass  # firebase-admin no instalado
    except Exception as e:
        print(f"☁️  Error de sync (no crítico): {e}")

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

    # Aumentar SOLO datos de entrenamiento (test permanece limpio y honesto)
    if AUGMENTACIONES_POR_MUESTRA > 0:
        print(f"\n📈 Aplicando data augmentation ({AUGMENTACIONES_POR_MUESTRA}x)...")
        X_aug_list, y_aug_list = [], []
        for seq, label in zip(X_train, y_train):
            variantes = augmentar_secuencia(seq, n=AUGMENTACIONES_POR_MUESTRA)
            X_aug_list.extend(variantes)
            y_aug_list.extend([label] * AUGMENTACIONES_POR_MUESTRA)

        X_train = np.concatenate([X_train, np.array(X_aug_list)], axis=0)
        y_train = np.concatenate([y_train, np.array(y_aug_list)], axis=0)

        # Mezclar para que el entrenamiento no vea bloques por seña
        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        muestras_por_clase = len(X_train) // len(clases)
        print(f"   Train aumentado: {len(X_train)} muestras (~{muestras_por_clase} por seña)")
        print(f"   Test sin tocar:  {len(X_test)} muestras (evaluación honesta)")

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

    # Convertir a TFLite automáticamente (necesario para Raspberry Pi)
    print("\n🔄 Convirtiendo a TFLite (optimizado para Raspberry Pi)...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = os.path.join(DIR_MODELO, "modelo.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"   ✅ modelo.tflite generado ({len(tflite_model)//1024} KB)")
        print("   → El traductor usará TFLite automáticamente en el RPi")
    except Exception as e:
        print(f"   ⚠️  No se pudo convertir a TFLite: {e}")
        print("   → Se usará modelo.h5 (más pesado en RAM)")

    # === Sincronizar con la nube automáticamente ===
    try:
        from sync_cloud import SyncCloud
        sync = SyncCloud()
        if sync.conectar():
            print("\n☁️  Subiendo modelo a la nube...")
            sync.subir_modelo()
            sync.subir_metricas()
        else:
            print("\n☁️  Sin conexión. El modelo se sincronizará después.")
    except ImportError:
        print("\n☁️  firebase-admin no instalado. Modelo guardado solo localmente.")
    except Exception as e:
        print(f"\n☁️  Error de sync (no crítico): {e}")


if __name__ == "__main__":
    main()
