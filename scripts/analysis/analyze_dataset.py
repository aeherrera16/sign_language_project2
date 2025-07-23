# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import json
from datetime import datetime

def analyze_dataset():
    """Analiza el dataset de gestos y genera reportes"""
    
    if not os.path.exists("data"):
        print(" La carpeta 'data' no existe.")
        return
    
    print("🔍 ANALISIS DEL DATASET")
    print("=" * 50)
    
    # Obtener informacion basica
    gestures = sorted([g for g in os.listdir("data") if os.path.isdir(os.path.join("data", g))])
    
    if not gestures:
        print(" No se encontraron gestos en la carpeta 'data'.")
        return
    
    dataset_info = {
        'timestamp': datetime.now().isoformat(),
        'total_gestures': len(gestures),
        'gesture_details': {},
        'statistics': {}
    }
    
    # Analizar cada gesto
    total_samples = 0
    samples_per_gesture = []
    file_sizes = []
    
    print(f"\n📂 Analizando {len(gestures)} gestos:")
    
    for gesture in gestures:
        gesture_path = os.path.join("data", gesture)
        files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
        
        gesture_samples = len(files)
        total_samples += gesture_samples
        samples_per_gesture.append(gesture_samples)
        
        # Analizar tamanos de archivos y formas
        valid_samples = 0
        invalid_samples = 0
        sample_shapes = []
        
        for file in files:
            try:
                file_path = os.path.join(gesture_path, file)
                sample = np.load(file_path)
                sample_shapes.append(sample.shape)
                
                # Verificar si la forma es correcta
                if sample.shape == (1530,):
                    valid_samples += 1
                else:
                    invalid_samples += 1
                
                # Tamano del archivo
                file_size = os.path.getsize(file_path)
                file_sizes.append(file_size)
                
            except Exception as e:
                invalid_samples += 1
                print(f"  ⚠️ Error en {file}: {e}")
        
        # Guardar informacion del gesto
        dataset_info['gesture_details'][gesture] = {
            'total_files': gesture_samples,
            'valid_samples': valid_samples,
            'invalid_samples': invalid_samples,
            'shapes': Counter([str(shape) for shape in sample_shapes])
        }
        
        print(f"  {gesture:20s}: {valid_samples:3d} validas, {invalid_samples:2d} invalidas")
    
    # Estadisticas generales
    dataset_info['statistics'] = {
        'total_samples': total_samples,
        'min_samples_per_gesture': min(samples_per_gesture),
        'max_samples_per_gesture': max(samples_per_gesture),
        'mean_samples_per_gesture': np.mean(samples_per_gesture),
        'std_samples_per_gesture': np.std(samples_per_gesture),
        'total_size_mb': sum(file_sizes) / (1024 * 1024),
        'avg_file_size_kb': np.mean(file_sizes) / 1024
    }
    
    print(f"\n📊 ESTADISTICAS GENERALES:")
    print(f"  - Total de muestras: {total_samples}")
    print(f"  - Minimo por gesto: {dataset_info['statistics']['min_samples_per_gesture']}")
    print(f"  - Maximo por gesto: {dataset_info['statistics']['max_samples_per_gesture']}")
    print(f"  - Promedio por gesto: {dataset_info['statistics']['mean_samples_per_gesture']:.1f}")
    print(f"  - Desviacion estandar: {dataset_info['statistics']['std_samples_per_gesture']:.1f}")
    print(f"  - Tamano total: {dataset_info['statistics']['total_size_mb']:.2f} MB")
    
    # Crear visualizaciones
    create_dataset_visualizations(gestures, samples_per_gesture, dataset_info)
    
    # Guardar reporte
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/dataset_analysis.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Recomendaciones
    generate_recommendations(dataset_info)
    
    return dataset_info

def create_dataset_visualizations(gestures, samples_per_gesture, dataset_info):
    """Crear visualizaciones del dataset"""
    
    plt.style.use('default')
    os.makedirs("analysis", exist_ok=True)
    
    # 1. Distribucion de muestras por gesto
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(gestures)), samples_per_gesture, alpha=0.7, color='skyblue')
    plt.title('Distribucion de Muestras por Gesto', fontsize=16, fontweight='bold')
    plt.xlabel('Gestos')
    plt.ylabel('Numero de Muestras')
    plt.xticks(range(len(gestures)), gestures, rotation=45, ha='right')
    
    # Agregar valores en las barras
    for i, v in enumerate(samples_per_gesture):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    # Lineas de referencia
    mean_samples = np.mean(samples_per_gesture)
    plt.axhline(mean_samples, color='red', linestyle='--', alpha=0.7, 
                label=f'Promedio: {mean_samples:.1f}')
    plt.axhline(50, color='orange', linestyle='--', alpha=0.7, 
                label='Minimo recomendado: 50')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/samples_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histograma de distribucion
    plt.figure(figsize=(10, 6))
    plt.hist(samples_per_gesture, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Histograma de Muestras por Gesto', fontweight='bold')
    plt.xlabel('Numero de Muestras')
    plt.ylabel('Frecuencia')
    plt.axvline(mean_samples, color='red', linestyle='--', 
                label=f'Promedio: {mean_samples:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/samples_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Mapa de calor de balance del dataset
    balance_data = []
    for gesture in gestures:
        samples = dataset_info['gesture_details'][gesture]['valid_samples']
        if samples < 30:
            balance_data.append(0)  # Insuficiente
        elif samples < 50:
            balance_data.append(1)  # Bajo
        elif samples < 80:
            balance_data.append(2)  # Bueno
        else:
            balance_data.append(3)  # Excelente
    
    # Reorganizar para visualizacion
    rows = int(np.ceil(len(gestures) / 10))
    cols = min(len(gestures), 10)
    
    balance_matrix = np.full((rows, cols), -1)
    for i, value in enumerate(balance_data):
        row = i // cols
        col = i % cols
        balance_matrix[row, col] = value
    
    plt.figure(figsize=(12, max(3, rows)))
    colors = ['red', 'orange', 'yellow', 'green', 'white']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    im = plt.imshow(balance_matrix, cmap=cmap, vmin=-1, vmax=3)
    
    # Etiquetas
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(gestures):
                text = gestures[idx][:8]  # Truncar nombres largos
                color = 'white' if balance_matrix[i, j] <= 0 else 'black'
                plt.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    plt.title('Balance del Dataset por Gesto', fontweight='bold')
    
    # Leyenda personalizada
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='< 30 (Insuficiente)'),
        Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='30-49 (Bajo)'),
        Rectangle((0,0),1,1, facecolor='yellow', alpha=0.7, label='50-79 (Bueno)'),
        Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='≥80 (Excelente)')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('analysis/dataset_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Visualizaciones guardadas en la carpeta 'analysis/'")

def generate_recommendations(dataset_info):
    """Generar recomendaciones para mejorar el dataset"""
    
    print(f"\n RECOMENDACIONES:")
    
    recommendations = []
    
    # Verificar balance del dataset
    min_samples = dataset_info['statistics']['min_samples_per_gesture']
    max_samples = dataset_info['statistics']['max_samples_per_gesture']
    std_samples = dataset_info['statistics']['std_samples_per_gesture']
    
    if std_samples > 20:
        recommendations.append("📊 Dataset desbalanceado: Considera equilibrar el numero de muestras por gesto")
        print("  - Dataset desbalanceado detectado")
    
    if min_samples < 50:
        recommendations.append(f"📈 Aumentar datos: {min_samples} muestras es insuficiente. Recomendado: minimo 50 por gesto")
        print(f"  - Aumentar muestras (minimo actual: {min_samples})")
    
    # Verificar gestos con pocas muestras
    low_sample_gestures = []
    for gesture, details in dataset_info['gesture_details'].items():
        if details['valid_samples'] < 50:
            low_sample_gestures.append(f"{gesture} ({details['valid_samples']})")
    
    if low_sample_gestures:
        recommendations.append(f" Gestos con pocas muestras: {', '.join(low_sample_gestures[:5])}")
        print(f"  - Gestos con pocas muestras: {len(low_sample_gestures)}")
    
    # Verificar muestras invalidas
    total_invalid = sum(details['invalid_samples'] for details in dataset_info['gesture_details'].values())
    if total_invalid > 0:
        recommendations.append(f"🔧 Limpiar datos: {total_invalid} muestras invalidas encontradas")
        print(f"  - Limpiar {total_invalid} muestras invalidas")
    
    # Recomendaciones de entrenamiento
    total_samples = dataset_info['statistics']['total_samples']
    if total_samples < 500:
        recommendations.append(" Dataset pequeno: Considera usar tecnicas de aumento de datos")
        print("  - Aplicar aumento de datos")
    
    if dataset_info['statistics']['std_samples_per_gesture'] > 30:
        recommendations.append("⚖️ Usar stratified split para mantener balance en entrenamiento/prueba")
        print("  - Usar division estratificada")
    
    # Guardar recomendaciones
    with open("analysis/recommendations.txt", "w", encoding='utf-8') as f:
        f.write("RECOMENDACIONES PARA MEJORAR EL DATASET\n")
        f.write("=" * 50 + "\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
    
    if not recommendations:
        print("   El dataset esta en buen estado")

def clean_invalid_samples():
    """Limpiar muestras invalidas del dataset"""
    
    print("\n🧹 LIMPIEZA DE MUESTRAS INVALIDAS")
    print("=" * 40)
    
    if not os.path.exists("data"):
        print(" La carpeta 'data' no existe.")
        return
    
    gestures = sorted([g for g in os.listdir("data") if os.path.isdir(os.path.join("data", g))])
    
    total_removed = 0
    
    for gesture in gestures:
        gesture_path = os.path.join("data", gesture)
        files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
        
        removed_count = 0
        
        for file in files:
            file_path = os.path.join(gesture_path, file)
            try:
                sample = np.load(file_path)
                if sample.shape != (1530,):
                    os.remove(file_path)
                    removed_count += 1
                    print(f"  🗑️ Removido: {file} (forma: {sample.shape})")
            except Exception as e:
                os.remove(file_path)
                removed_count += 1
                print(f"  🗑️ Removido: {file} (error: {e})")
        
        if removed_count > 0:
            total_removed += removed_count
            print(f"  {gesture}: {removed_count} archivos removidos")
    
    if total_removed == 0:
        print(" No se encontraron muestras invalidas")
    else:
        print(f"\n Limpieza completada: {total_removed} archivos removidos")

def main():
    """Funcion principal"""
    print("🔬 ANALISIS Y OPTIMIZACION DEL DATASET")
    print("=" * 60)
    
    # Analisis del dataset
    dataset_info = analyze_dataset()
    
    if dataset_info:
        # Preguntar si limpiar datos invalidos
        print("\n" + "="*50)
        response = input("Deseas limpiar muestras invalidas? (s/n): ").lower().strip()
        
        if response in ['s', 'si', 'si', 'y', 'yes']:
            clean_invalid_samples()
            print("\n Re-analizando despues de la limpieza...")
            analyze_dataset()
        
        print(f"\n📁 Resultados guardados en la carpeta 'analysis/'")

if __name__ == "__main__":
    main()
