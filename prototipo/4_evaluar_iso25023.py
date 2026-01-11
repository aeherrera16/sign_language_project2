#!/usr/bin/env python3
"""
EVALUACIÓN ISO/IEC 25023 - Métricas de Calidad
Genera reporte de precisión y tiempo de respuesta.
"""

import os
import json
import numpy as np
import pickle
import time
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mediapipe as mp
import cv2

DIR_MODELO = os.path.join(os.path.dirname(__file__), "modelo")
DIR_DATOS = os.path.join(os.path.dirname(__file__), "datos")
FRAMES = 30
FEATURES = 126


def cargar_datos_test():
    """Carga datos para evaluación."""
    X, y = [], []
    
    for sena_dir in sorted(glob(os.path.join(DIR_DATOS, "*"))):
        if not os.path.isdir(sena_dir):
            continue
        
        sena = os.path.basename(sena_dir)
        
        for archivo in glob(os.path.join(sena_dir, "*.json")):
            with open(archivo) as f:
                datos = json.load(f)
                for seq in datos["secuencias"]:
                    seq = np.array(seq)
                    if seq.shape == (FRAMES, FEATURES):
                        X.append(seq)
                        y.append(sena)
    
    return np.array(X), np.array(y)


def evaluar_modelo():
    """Evalúa el modelo según ISO/IEC 25023."""
    
    print("\n" + "="*60)
    print("  EVALUACIÓN ISO/IEC 25023")
    print("  Métricas de Calidad del Prototipo LSE")
    print("="*60)
    
    # Verificar modelo
    modelo_path = os.path.join(DIR_MODELO, "modelo.h5")
    if not os.path.exists(modelo_path):
        print("\n❌ No hay modelo. Entrena primero con: python 2_entrenar_modelo.py")
        return
    
    # Cargar modelo
    print("\n🔄 Cargando modelo...")
    modelo = tf.keras.models.load_model(modelo_path)
    
    with open(os.path.join(DIR_MODELO, "encoder.pkl"), 'rb') as f:
        encoder = pickle.load(f)
    
    # Cargar datos
    print("🔄 Cargando datos de prueba...")
    X, y = cargar_datos_test()
    
    if len(X) == 0:
        print("❌ No hay datos de prueba")
        return
    
    y_encoded = encoder.transform(y)
    
    print(f"\n📊 Datos: {len(X)} secuencias, {len(encoder.classes_)} clases")
    
    # === MÉTRICA 1: PRECISIÓN (Accuracy) ===
    print("\n" + "-"*60)
    print("📐 MÉTRICA 1: PRECISIÓN FUNCIONAL")
    print("-"*60)
    
    y_pred_proba = modelo.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_encoded, y_pred)
    print(f"\n✅ Accuracy Global: {accuracy:.2%}")
    
    print("\n📋 Reporte por clase:")
    print(classification_report(y_encoded, y_pred, target_names=encoder.classes_))
    
    # === MÉTRICA 2: TIEMPO DE RESPUESTA ===
    print("-"*60)
    print("⏱️ MÉTRICA 2: TIEMPO DE RESPUESTA")
    print("-"*60)
    
    tiempos = []
    for i in range(min(50, len(X))):  # Medir 50 predicciones
        seq = np.expand_dims(X[i], axis=0)
        
        inicio = time.perf_counter()
        _ = modelo.predict(seq, verbose=0)
        fin = time.perf_counter()
        
        tiempos.append((fin - inicio) * 1000)  # ms
    
    tiempo_promedio = np.mean(tiempos)
    tiempo_max = np.max(tiempos)
    tiempo_min = np.min(tiempos)
    
    print(f"\n✅ Tiempo promedio: {tiempo_promedio:.2f} ms")
    print(f"   Tiempo mínimo: {tiempo_min:.2f} ms")
    print(f"   Tiempo máximo: {tiempo_max:.2f} ms")
    
    # === MÉTRICA 3: CONFIANZA ===
    print("\n" + "-"*60)
    print("📊 MÉTRICA 3: DISTRIBUCIÓN DE CONFIANZA")
    print("-"*60)
    
    confianzas = np.max(y_pred_proba, axis=1)
    print(f"\n✅ Confianza promedio: {np.mean(confianzas):.2%}")
    print(f"   Confianza mínima: {np.min(confianzas):.2%}")
    print(f"   Confianza máxima: {np.max(confianzas):.2%}")
    
    # === RESUMEN ISO/IEC 25023 ===
    print("\n" + "="*60)
    print("  RESUMEN ISO/IEC 25023")
    print("="*60)
    
    print(f"""
┌─────────────────────────────────────────────────────────┐
│  MÉTRICA                    │  VALOR                   │
├─────────────────────────────────────────────────────────┤
│  Exactitud Funcional        │  {accuracy:.2%}                   │
│  Tiempo de Respuesta (prom) │  {tiempo_promedio:.2f} ms                │
│  Confianza Promedio         │  {np.mean(confianzas):.2%}                   │
│  Clases Reconocidas         │  {len(encoder.classes_)}                       │
│  Muestras Evaluadas         │  {len(X)}                      │
└─────────────────────────────────────────────────────────┘
    """)
    
    # Guardar reporte
    reporte = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metricas": {
            "accuracy": float(accuracy),
            "tiempo_respuesta_ms": float(tiempo_promedio),
            "confianza_promedio": float(np.mean(confianzas)),
        },
        "clases": list(encoder.classes_),
        "muestras": int(len(X))
    }
    
    with open(os.path.join(DIR_MODELO, "evaluacion_iso25023.json"), 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"💾 Reporte guardado: modelo/evaluacion_iso25023.json")


if __name__ == "__main__":
    evaluar_modelo()
