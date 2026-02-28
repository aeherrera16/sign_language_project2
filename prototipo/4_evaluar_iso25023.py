#!/usr/bin/env python3
"""
EVALUACIÓN ISO/IEC 25023 - Métricas de Calidad HONESTAS
Usa validación cruzada (k-fold) para métricas realistas.
NO evalúa sobre los mismos datos de entrenamiento.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
import time
from glob import glob

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DIR_MODELO = os.path.join(os.path.dirname(__file__), "modelo")
DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")
FRAMES = 30
FEATURES = 126


def cargar_datos():
    """Carga todos los datos disponibles."""
    X, y = [], []
    detalles = {}
    
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
        
        detalles[sena] = count
    
    return np.array(X), np.array(y), detalles


def crear_modelo(num_clases):
    """Modelo LSTM idéntico al de entrenamiento."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(FRAMES, FEATURES)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
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


def evaluar_con_holdout(X, y, encoder):
    """
    Evaluación con hold-out (30% test) - método principal.
    Entrena un modelo NUEVO y evalúa en datos que NUNCA vio.
    """
    y_enc = encoder.transform(y)
    y_cat = to_categorical(y_enc)
    num_clases = len(encoder.classes_)
    
    # División estratificada: 70% train, 30% test
    X_train, X_test, y_train, y_test, _, y_test_enc = train_test_split(
        X, y_cat, y_enc, test_size=0.30, random_state=42, stratify=y_enc
    )
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Entrenar modelo desde cero
    modelo = crear_modelo(num_clases)
    modelo.fit(X_train, y_train, epochs=80, batch_size=16, verbose=0,
               validation_data=(X_test, y_test))
    
    # Predecir sobre DATOS NO VISTOS
    y_pred_proba = modelo.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test_enc, y_pred)
    confianzas = np.max(y_pred_proba, axis=1)
    
    return accuracy, y_test_enc, y_pred, confianzas


def evaluar_con_kfold(X, y, encoder, k=3):
    """
    Evaluación con K-Fold Cross Validation.
    Más robusta: evalúa en CADA fold como datos no vistos.
    """
    y_enc = encoder.transform(y)
    num_clases = len(encoder.classes_)
    
    # Usar min(k, muestras por clase) para evitar errores
    min_por_clase = min(np.bincount(y_enc))
    k_real = min(k, min_por_clase)
    
    if k_real < 2:
        print(f"   ⚠️ Pocas muestras por clase (mín: {min_por_clase}). Usando hold-out.")
        return None
    
    skf = StratifiedKFold(n_splits=k_real, shuffle=True, random_state=42)
    
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_enc)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = to_categorical(y_enc[train_idx], num_classes=num_clases)
        y_test_enc = y_enc[test_idx]
        
        modelo = crear_modelo(num_clases)
        modelo.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        
        y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test_enc, y_pred)
        accuracies.append(acc)
        print(f"   Fold {fold+1}/{k_real}: {acc:.2%}")
    
    return accuracies


def evaluar_modelo():
    """Evaluación completa ISO/IEC 25023 con métricas honestas."""
    
    print("\n" + "="*60)
    print("  EVALUACIÓN ISO/IEC 25023 - MÉTRICAS HONESTAS")
    print("  (Evalúa en datos NO vistos durante entrenamiento)")
    print("="*60)
    
    # Verificar modelo guardado
    modelo_path = os.path.join(DIR_MODELO, "modelo.h5")
    if not os.path.exists(modelo_path):
        print("\n❌ No hay modelo. Entrena primero con: python 2_entrenar_modelo.py")
        return
    
    # Cargar encoder
    with open(os.path.join(DIR_MODELO, "encoder.pkl"), 'rb') as f:
        encoder = pickle.load(f)
    
    # Cargar datos
    print("\n🔄 Cargando datos...")
    X, y, detalles = cargar_datos()
    
    if len(X) == 0:
        print("❌ No hay datos de evaluación")
        return
    
    print(f"\n📂 Datos disponibles:")
    for sena, count in detalles.items():
        print(f"   {sena}: {count} secuencias")
    print(f"   TOTAL: {len(X)} secuencias, {len(detalles)} clases")
    
    total_secuencias = len(X)
    
    # === MÉTRICA 1: PRECISIÓN CON HOLD-OUT (30% test) ===
    print("\n" + "-"*60)
    print("📐 MÉTRICA 1: PRECISIÓN (Hold-out 70/30)")
    print("   Modelo entrenado desde cero, evaluado en 30% no visto")
    print("-"*60)
    
    accuracy, y_true, y_pred, confianzas = evaluar_con_holdout(X, y, encoder)
    
    print(f"\n✅ Accuracy (datos no vistos): {accuracy:.2%}")
    
    print("\n📋 Reporte por clase:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("📊 Matriz de confusión:")
    print(f"   {'':>12}", end="")
    for c in encoder.classes_:
        print(f"  {c:>8}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"   {encoder.classes_[i]:>12}", end="")
        for val in row:
            print(f"  {val:>8}", end="")
        print()
    
    # === MÉTRICA 1b: CROSS VALIDATION (más robusta) ===
    print("\n" + "-"*60)
    print("📐 MÉTRICA 1b: VALIDACIÓN CRUZADA (3-Fold)")
    print("   Cada fold evalúa en datos que NUNCA vio")
    print("-"*60)
    
    kfold_accs = evaluar_con_kfold(X, y, encoder, k=3)
    
    if kfold_accs:
        cv_mean = np.mean(kfold_accs)
        cv_std = np.std(kfold_accs)
        print(f"\n✅ Accuracy promedio CV: {cv_mean:.2%} ± {cv_std:.2%}")
    else:
        cv_mean = accuracy
        cv_std = 0.0
    
    # === MÉTRICA 2: TIEMPO DE RESPUESTA ===
    print("\n" + "-"*60)
    print("⏱️ MÉTRICA 2: TIEMPO DE RESPUESTA")
    print("-"*60)
    
    # Cargar modelo guardado para medir tiempo real
    modelo_guardado = tf.keras.models.load_model(modelo_path)
    
    tiempos = []
    for i in range(min(50, len(X))):
        seq = np.expand_dims(X[i], axis=0)
        inicio = time.perf_counter()
        _ = modelo_guardado.predict(seq, verbose=0)
        fin = time.perf_counter()
        tiempos.append((fin - inicio) * 1000)
    
    tiempo_prom = np.mean(tiempos)
    tiempo_max = np.max(tiempos)
    tiempo_min = np.min(tiempos)
    
    print(f"\n✅ Tiempo promedio: {tiempo_prom:.2f} ms")
    print(f"   Tiempo mínimo:  {tiempo_min:.2f} ms")
    print(f"   Tiempo máximo:  {tiempo_max:.2f} ms")
    
    # === MÉTRICA 3: CONFIANZA ===
    print("\n" + "-"*60)
    print("📊 MÉTRICA 3: DISTRIBUCIÓN DE CONFIANZA")
    print("-"*60)
    
    conf_prom = float(np.mean(confianzas))
    conf_min = float(np.min(confianzas))
    conf_max = float(np.max(confianzas))
    
    print(f"\n✅ Confianza promedio: {conf_prom:.2%}")
    print(f"   Confianza mínima:  {conf_min:.2%}")
    print(f"   Confianza máxima:  {conf_max:.2%}")
    
    # Predicciones por debajo del umbral
    umbral = 0.96
    bajo_umbral = np.sum(confianzas < umbral)
    print(f"   Bajo umbral ({umbral:.0%}): {bajo_umbral}/{len(confianzas)} ({bajo_umbral/len(confianzas):.0%})")
    
    # === MÉTRICA 4: ANÁLISIS DE OVERFITTING ===
    print("\n" + "-"*60)
    print("🔍 MÉTRICA 4: ANÁLISIS DE OVERFITTING")
    print("-"*60)
    
    # Evaluar en datos de entrenamiento (para comparar)
    y_enc_all = encoder.transform(y)
    y_pred_all = np.argmax(modelo_guardado.predict(X, verbose=0), axis=1)
    acc_train_all = accuracy_score(y_enc_all, y_pred_all)
    
    gap = acc_train_all - accuracy
    print(f"\n   Accuracy en TODO el dataset:     {acc_train_all:.2%}")
    print(f"   Accuracy en datos NO vistos:     {accuracy:.2%}")
    print(f"   Gap (overfitting):               {gap:.2%}")
    
    if gap > 0.15:
        print(f"   ⚠️ Gap alto ({gap:.0%}): El modelo está memorizando, NO generalizando")
        print(f"   💡 Solución: Grabar MÁS secuencias por seña (mín. 50-100)")
    elif gap > 0.05:
        print(f"   ⚠️ Gap moderado: Podría mejorar con más datos")
    else:
        print(f"   ✅ Gap bajo: El modelo generaliza bien")
    
    # === RESUMEN ===
    print("\n" + "="*60)
    print("  RESUMEN ISO/IEC 25023")
    print("="*60)
    
    # Accuracy real a reportar: el menor entre holdout y CV
    acc_final = min(accuracy, cv_mean) if kfold_accs else accuracy
    
    print(f"""
┌────────────────────────────────────────────────────────────┐
│  MÉTRICA                       │  VALOR                    │
├────────────────────────────────────────────────────────────┤
│  Exactitud (datos no vistos)   │  {acc_final:.2%}                    │
│  Exactitud (cross-validation)  │  {cv_mean:.2%} ± {cv_std:.2%}           │
│  Gap de Overfitting            │  {gap:.2%}                    │
│  Tiempo de Respuesta (prom)    │  {tiempo_prom:.2f} ms               │
│  Confianza Promedio            │  {conf_prom:.2%}                    │
│  Clases Reconocidas            │  {len(encoder.classes_)}                          │
│  Muestras Totales              │  {total_secuencias}                        │
└────────────────────────────────────────────────────────────┘
    """)
    
    # Disclaimer
    if total_secuencias < 200:
        print(f"⚠️ NOTA: Con solo {total_secuencias} muestras, las métricas pueden")
        print(f"   ser inestables. Se recomienda mín. 50-100 secuencias por seña.")
    
    # Guardar reporte
    reporte = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metodo": "hold-out 70/30 + cross-validation",
        "metricas": {
            "accuracy_holdout": float(accuracy),
            "accuracy_cv_mean": float(cv_mean),
            "accuracy_cv_std": float(cv_std),
            "accuracy_train_completo": float(acc_train_all),
            "gap_overfitting": float(gap),
            "tiempo_respuesta_ms": float(tiempo_prom),
            "confianza_promedio": float(conf_prom),
            "confianza_minima": float(conf_min),
        },
        "clases": list(encoder.classes_),
        "muestras_total": int(total_secuencias),
        "muestras_por_clase": {k: int(v) for k, v in detalles.items()},
        "nota": "Métricas calculadas en datos NO vistos durante entrenamiento"
    }
    
    with open(os.path.join(DIR_MODELO, "evaluacion_iso25023.json"), 'w') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Reporte guardado: modelo/evaluacion_iso25023.json")


if __name__ == "__main__":
    evaluar_modelo()
