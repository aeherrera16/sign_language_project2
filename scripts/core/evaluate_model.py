# -*- coding: utf-8 -*-
import os
# Configurar TensorFlow para evitar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import os
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

def load_data():
    """Carga los datos del dataset"""
    X, y = [], []
    
    if not os.path.exists("data"):
        raise FileNotFoundError(" La carpeta 'data' no existe.")
    
    gestures = sorted(os.listdir("data"))
    if not gestures:
        raise ValueError(" No hay gestos en la carpeta 'data'.")
    
    print(f"📂 Cargando {len(gestures)} gestos: {gestures}")
    
    for label, gesture in enumerate(gestures):
        gesture_path = os.path.join("data", gesture)
        if not os.path.isdir(gesture_path):
            continue
            
        gesture_samples = 0
        for file in os.listdir(gesture_path):
            if file.endswith(".npy"):
                try:
                    sample = np.load(os.path.join(gesture_path, file))
                    if sample.shape == (1530,):
                        X.append(sample)
                        y.append(label)
                        gesture_samples += 1
                    else:
                        print(f"⚠️ Archivo ignorado por forma invalida: {file} ({sample.shape})")
                except Exception as e:
                    print(f"⚠️ Error al procesar {file}: {e}")
        
        print(f"  {gesture}: {gesture_samples} muestras")
    
    if not X:
        raise ValueError(" No se encontraron muestras validas.")
    
    return np.array(X), np.array(y), gestures

def evaluate_model_comprehensive(model_path="model/gesture_model.h5", 
                                labels_path="model/labels.pkl"):
    """Evaluacion comprensiva del modelo"""
    
    # Cargar modelo y etiquetas
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Etiquetas no encontradas: {labels_path}")
    
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    
    # Cargar datos
    X, y, gesture_names = load_data()
    
    # Dividir datos (mismo split que entrenamiento)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\n📊 Datos de evaluacion:")
    print(f"  - Total muestras: {len(X)}")
    print(f"  - Entrenamiento: {len(X_train)}")
    print(f"  - Prueba: {len(X_test)}")
    print(f"  - Clases: {len(labels)}")
    
    # Predicciones
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Metricas basicas
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Guardar metricas
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_samples': len(X),
        'num_train': len(X_train),
        'num_test': len(X_test),
        'num_classes': len(labels),
        'class_names': labels
    }
    
    os.makedirs("evaluation", exist_ok=True)
    
    # Guardar metricas en JSON
    with open("evaluation/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n METRICAS GENERALES:")
    print(f"  Precision: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Reporte detallado por clase
    print(f"\n📋 REPORTE POR CLASE:")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizaciones
    create_visualizations(cm, labels, y_test, y_pred_proba, report)
    
    # Analisis de errores
    analyze_errors(y_test, y_pred, y_pred_proba, labels, X_test)
    
    return metrics, report

def create_visualizations(cm, labels, y_test, y_pred_proba, report):
    """Crear visualizaciones de evaluacion"""
    
    plt.style.use('default')
    
    # 1. Matriz de confusion
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusion', fontsize=16, fontweight='bold')
    plt.xlabel('Prediccion', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('evaluation/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Metricas por clase
    classes = list(report.keys())[:-3]  # Excluir 'accuracy', 'macro avg', 'weighted avg'
    precisions = [report[cls]['precision'] for cls in classes]
    recalls = [report[cls]['recall'] for cls in classes]
    f1s = [report[cls]['f1-score'] for cls in classes]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    x_pos = np.arange(len(classes))
    
    # Precision
    bars1 = ax1.bar(x_pos, precisions, alpha=0.8, color='skyblue')
    ax1.set_title('Precision por Clase', fontweight='bold')
    ax1.set_ylabel('Precision')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(precisions):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Recall
    bars2 = ax2.bar(x_pos, recalls, alpha=0.8, color='lightcoral')
    ax2.set_title('Recall por Clase', fontweight='bold')
    ax2.set_ylabel('Recall')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(recalls):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # F1-Score
    bars3 = ax3.bar(x_pos, f1s, alpha=0.8, color='lightgreen')
    ax3.set_title('F1-Score por Clase', fontweight='bold')
    ax3.set_ylabel('F1-Score')
    ax3.set_xlabel('Clases')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(classes, rotation=45, ha='right')
    ax3.set_ylim(0, 1)
    for i, v in enumerate(f1s):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('evaluation/metrics_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribucion de confianza
    max_probs = np.max(y_pred_proba, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Distribucion de Confianza en Predicciones', fontweight='bold')
    plt.xlabel('Confianza Maxima')
    plt.ylabel('Frecuencia')
    plt.axvline(np.mean(max_probs), color='red', linestyle='--', 
                label=f'Media: {np.mean(max_probs):.3f}')
    plt.axvline(np.median(max_probs), color='orange', linestyle='--', 
                label=f'Mediana: {np.median(max_probs):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Visualizaciones guardadas en la carpeta 'evaluation/'")

def analyze_errors(y_test, y_pred, y_pred_proba, labels, X_test):
    """Analisis detallado de errores"""
    
    incorrect_indices = np.where(y_test != y_pred)[0]
    
    if len(incorrect_indices) == 0:
        print(" No hay errores! El modelo es perfecto en el conjunto de prueba.")
        return
    
    print(f"\n ANALISIS DE ERRORES ({len(incorrect_indices)} errores):")
    
    error_analysis = {}
    
    for idx in incorrect_indices:
        true_label = labels[y_test[idx]]
        pred_label = labels[y_pred[idx]]
        confidence = y_pred_proba[idx][y_pred[idx]]
        
        error_key = f"{true_label} -> {pred_label}"
        if error_key not in error_analysis:
            error_analysis[error_key] = []
        error_analysis[error_key].append(confidence)
    
    # Mostrar errores mas frecuentes
    sorted_errors = sorted(error_analysis.items(), 
                          key=lambda x: len(x[1]), reverse=True)
    
    print("\n🔍 Errores mas frecuentes:")
    for i, (error_type, confidences) in enumerate(sorted_errors[:10]):
        avg_conf = np.mean(confidences)
        print(f"  {i+1:2d}. {error_type:20s} - {len(confidences):2d} veces (confianza promedio: {avg_conf:.3f})")
    
    # Guardar analisis de errores
    error_report = {
        'total_errors': len(incorrect_indices),
        'error_rate': len(incorrect_indices) / len(y_test),
        'error_breakdown': {k: len(v) for k, v in error_analysis.items()}
    }
    
    with open("evaluation/error_analysis.json", "w") as f:
        json.dump(error_report, f, indent=2)

def plot_training_history():
    """Grafica el historial de entrenamiento si existe"""
    
    history_path = "model/training_history.json"
    if not os.path.exists(history_path):
        print("⚠️ No se encontro historial de entrenamiento")
        return
    
    with open(history_path, "r") as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Precision
    ax1.plot(history['accuracy'], label='Entrenamiento', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Validacion', linewidth=2)
    ax1.set_title('Precision del Modelo', fontweight='bold')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Perdida
    ax2.plot(history['loss'], label='Entrenamiento', linewidth=2)
    ax2.plot(history['val_loss'], label='Validacion', linewidth=2)
    ax2.set_title('Perdida del Modelo', fontweight='bold')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Perdida')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📈 Historial de entrenamiento graficado")

def generate_report():
    """Genera un reporte completo en texto"""
    
    # Leer metricas
    if os.path.exists("evaluation/metrics.json"):
        with open("evaluation/metrics.json", "r") as f:
            metrics = json.load(f)
    else:
        print("⚠️ No se encontraron metricas guardadas")
        return
    
    # Leer analisis de errores
    if os.path.exists("evaluation/error_analysis.json"):
        with open("evaluation/error_analysis.json", "r") as f:
            errors = json.load(f)
    else:
        errors = None
    
    # Generar reporte
    report = f"""
# REPORTE DE EVALUACION DEL MODELO
Generado: {metrics['timestamp']}

## RESUMEN GENERAL
- Total de muestras: {metrics['num_samples']}
- Muestras de entrenamiento: {metrics['num_train']}
- Muestras de prueba: {metrics['num_test']}
- Numero de clases: {metrics['num_classes']}

## METRICAS PRINCIPALES
- Precision: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}

## CLASES
{', '.join(metrics['class_names'])}

"""
    
    if errors:
        report += f"""
## ANALISIS DE ERRORES
- Total de errores: {errors['total_errors']}
- Tasa de error: {errors['error_rate']:.4f} ({errors['error_rate']*100:.2f}%)

### Errores mas comunes:
"""
        sorted_errors = sorted(errors['error_breakdown'].items(), 
                              key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors[:5]:
            report += f"- {error_type}: {count} veces\n"
    
    report += """
## ARCHIVOS GENERADOS
- confusion_matrix.png: Matriz de confusion
- metrics_by_class.png: Metricas por clase
- confidence_distribution.png: Distribucion de confianza
- training_history.png: Historial de entrenamiento
- metrics.json: Metricas en formato JSON
- error_analysis.json: Analisis de errores detallado
"""
    
    with open("evaluation/report.md", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("📄 Reporte completo guardado en evaluation/report.md")

def main():
    """Funcion principal"""
    print("🔍 EVALUACION COMPRENSIVA DEL MODELO")
    print("=" * 50)
    
    try:
        # Evaluacion principal
        metrics, report = evaluate_model_comprehensive()
        
        # Historial de entrenamiento
        plot_training_history()
        
        # Generar reporte
        generate_report()
        
        print("\n Evaluacion completada exitosamente!")
        print("📁 Revisa la carpeta 'evaluation/' para ver todos los resultados")
        
    except Exception as e:
        print(f" Error durante la evaluacion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
