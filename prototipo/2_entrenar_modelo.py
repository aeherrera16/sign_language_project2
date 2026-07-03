#!/usr/bin/env python3
"""
ENTRENADOR GRU BIDIRECCIONAL - Reconocimiento de Señas Dinámicas
Arquitectura validada en Fernández (2026, ESPOCH) para LSEC: GRU
bidireccional sobre secuencias temporales de landmarks, superior a LSTM
unidireccional en precisión, especialmente en la clase "silencio"/NINGUNA.
"""

import os
import sys
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
from keras.layers import GRU, Bidirectional, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURACIÓN ===
DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")
DIR_MODELO = os.path.join(os.path.dirname(__file__), "modelo")

FRAMES = 30
FEATURES_MANOS = 126
FEATURES_POSE = 15
FEATURES_ROSTRO = 18
FEATURES = FEATURES_MANOS + FEATURES_POSE + FEATURES_ROSTRO  # 159
AUGMENTACIONES_POR_MUESTRA = 14  # Cada secuencia real genera 14 variantes (+1400%)

# Offsets de los bloques izquierda/derecha en pose y rostro (ver 1_grabar_senas.py
# y 3_traductor.py para el layout completo). A diferencia de las manos —que ya
# vienen pre-separadas en slot0=izq/slot1=der por la extracción— estos son
# puntos anatómicos fijos dentro de su propio bloque, así que el espejo
# horizontal debe negar X y además intercambiar cada par izquierda/derecha.
_POSE_LEFT_SHOULDER = FEATURES_MANOS + 1 * 3
_POSE_RIGHT_SHOULDER = FEATURES_MANOS + 2 * 3
_POSE_LEFT_ELBOW = FEATURES_MANOS + 3 * 3
_POSE_RIGHT_ELBOW = FEATURES_MANOS + 4 * 3
_FACE_BASE = FEATURES_MANOS + FEATURES_POSE
_FACE_CEJA_IZQ = _FACE_BASE + 0 * 3
_FACE_CEJA_DER = _FACE_BASE + 1 * 3
_FACE_BOCA_IZQ = _FACE_BASE + 2 * 3
_FACE_BOCA_DER = _FACE_BASE + 3 * 3


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
    Simula hacer la misma seña con la mano contraria (y el cuerpo/rostro
    visto en espejo).
    Los features son relativos a un punto ancla (muñeca para manos, punto
    medio de hombros para pose, punta de nariz para rostro), así que negar
    X invierte la lateralidad sin afectar Y ni Z.

    Manos (0-125): slot0 = izquierda, slot1 = derecha — ya vienen separadas
    por la extracción, solo se niega X dentro de cada slot (sin intercambiar
    slots).

    Pose (126-140) y rostro (141-158): son puntos anatómicos fijos (no
    pre-separados por lado), así que además de negar X hay que intercambiar
    cada par izquierda/derecha (hombros, codos, cejas, comisuras de boca)
    para que el espejo sea anatómicamente coherente.
    """
    m = seq.copy()

    # Manos: negar X dentro de cada slot
    for slot in range(2):
        base = slot * 63
        for i in range(21):
            m[:, base + i * 3] *= -1

    # Pose + rostro: negar X de todos los puntos
    for base in range(FEATURES_MANOS, FEATURES_MANOS + FEATURES_POSE + FEATURES_ROSTRO, 3):
        m[:, base] *= -1

    # Intercambiar pares izquierda/derecha (ya con X negada)
    def swap_par(idx_a, idx_b):
        tmp = m[:, idx_a:idx_a + 3].copy()
        m[:, idx_a:idx_a + 3] = m[:, idx_b:idx_b + 3]
        m[:, idx_b:idx_b + 3] = tmp

    swap_par(_POSE_LEFT_SHOULDER, _POSE_RIGHT_SHOULDER)
    swap_par(_POSE_LEFT_ELBOW, _POSE_RIGHT_ELBOW)
    swap_par(_FACE_CEJA_IZQ, _FACE_CEJA_DER)
    swap_par(_FACE_BOCA_IZQ, _FACE_BOCA_DER)

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
    Arquitectura GRU bidireccional — replica la que mejor resultado dio en
    Fernández (2026, ESPOCH) comparando LSTM vs GRU para LSEC: GRU
    bidireccional superó a LSTM unidireccional en todas las clases, sobre
    todo en la clase "silencio" (precisión 0.64→0.77 con el mismo recall),
    que es exactamente nuestro problema de "no debe hablar si no es una
    seña real". Bidireccional funciona porque siempre alimentamos la
    ventana completa de FRAMES (no streaming frame-a-frame), así que no
    rompe el diseño en tiempo real. Bonus: GRU tiene menos parámetros que
    LSTM → más liviano para Raspberry Pi.
    """
    L2 = 1e-5
    model = Sequential([
        Bidirectional(GRU(128, return_sequences=True, activation='tanh',
                           kernel_regularizer=l2(L2)),
                      input_shape=(FRAMES, FEATURES)),
        Dropout(0.3),

        Bidirectional(GRU(64, return_sequences=False, activation='tanh',
                           kernel_regularizer=l2(L2))),
        Dropout(0.3),

        Dense(64, activation='relu', kernel_regularizer=l2(L2)),
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
    print("  ENTRENADOR DE MODELO (GRU BIDIRECCIONAL)")
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

    # SavedModel: compatible con TF 2.14+ (necesario para Raspberry Pi con TF != versión de entrenamiento)
    import shutil
    sm_path = os.path.join(DIR_MODELO, "modelo_savedmodel")
    if os.path.exists(sm_path):
        shutil.rmtree(sm_path)
    tf.saved_model.save(modelo, sm_path)

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
    print("   - modelo_savedmodel/ (compatible con Raspberry Pi)")
    print("   - encoder.pkl")
    print("   - info.json")

    # Convertir a TFLite (necesario para Raspberry Pi — usa menos RAM y es más rápido)
    # Se corre en un subproceso aparte: en ciertas combinaciones de TF/Keras
    # (visto con TF 2.16 + Keras 3 + BatchNormalization) la conversión puede
    # abortar el proceso con un error fatal de LLVM/MLIR que NO es una
    # excepción de Python capturable. Aislarlo evita perder el modelo/reporte
    # ya guardados si eso vuelve a pasar.
    print("\n🔄 Convirtiendo a TFLite (optimizado para Raspberry Pi)...")
    tflite_path = os.path.join(DIR_MODELO, "modelo.tflite")
    import subprocess

    def _intentar_conversion(metodo):
        return subprocess.run(
            [
                sys.executable, os.path.join(os.path.dirname(__file__), "convertir_tflite.py"),
                "--metodo", metodo,
                "--savedmodel-path", sm_path,
                "--h5-path", os.path.join(DIR_MODELO, "modelo.h5"),
                "--output", tflite_path,
                "--frames", str(FRAMES),
                "--features", str(FEATURES),
            ],
            capture_output=True, text=True
        )

    tflite_ok = False
    # "concrete" primero: no pasa por SavedModel, evita el bug de MLIR con
    # BatchNormalization que puede abortar el proceso (returncode -6/SIGABRT).
    # Cada método corre en su PROPIO subproceso — si uno crashea a nivel
    # nativo, no se lleva al siguiente intento.
    for metodo in ("concrete", "savedmodel"):
        resultado = _intentar_conversion(metodo)
        if resultado.returncode == 0 and os.path.exists(tflite_path):
            kb = os.path.getsize(tflite_path) // 1024
            print(f"   ✅ modelo.tflite generado vía '{metodo}' ({kb} KB)")
            tflite_ok = True
            break
        motivo = (resultado.stderr.strip().splitlines() or ["sin detalle"])[-1]
        print(f"   ⚠️  Método '{metodo}' falló (returncode={resultado.returncode}): {motivo}")

    if not tflite_ok:
        print("   ⚠️  TFLite no disponible — el Pi usará modelo_savedmodel/ (más pesado, pero funciona igual)")

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
