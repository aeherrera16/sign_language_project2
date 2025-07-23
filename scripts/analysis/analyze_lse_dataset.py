#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS COMPLETO DEL DATASET LSE ECUADOR
Identifica problemas de calidad y sugiere mejoras
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import json
from pathlib import Path

class LSEDatasetAnalyzer:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.gestures_info = {}
        self.landmark_stats = {}
        
    def analyze_dataset_structure(self):
        """Analiza la estructura del dataset"""
        print("🔍 ANALIZANDO ESTRUCTURA DEL DATASET")
        print("=" * 50)
        
        if not os.path.exists(self.data_path):
            print(f"❌ Error: No se encuentra el directorio {self.data_path}")
            return None
        
        # Contar carpetas y archivos por gesto
        gesture_counts = {}
        total_samples = 0
        
        for gesture_folder in os.listdir(self.data_path):
            gesture_path = os.path.join(self.data_path, gesture_folder)
            
            if os.path.isdir(gesture_path):
                sample_count = len([f for f in os.listdir(gesture_path) 
                                  if f.endswith(('.npy', '.pkl', '.json', '.jpg', '.png'))])
                gesture_counts[gesture_folder] = sample_count
                total_samples += sample_count
        
        print(f"📊 Total de gestos: {len(gesture_counts)}")
        print(f"📊 Total de muestras: {total_samples}")
        print(f"📊 Promedio por gesto: {total_samples/len(gesture_counts):.1f}")
        
        # Identificar gestos con pocas muestras
        min_samples = min(gesture_counts.values()) if gesture_counts else 0
        max_samples = max(gesture_counts.values()) if gesture_counts else 0
        
        print(f"📊 Mínimo de muestras: {min_samples}")
        print(f"📊 Máximo de muestras: {max_samples}")
        
        # Gestos críticos (menos de 50 muestras)
        critical_gestures = {k: v for k, v in gesture_counts.items() if v < 50}
        
        if critical_gestures:
            print(f"\n⚠️  GESTOS CRÍTICOS (< 50 muestras): {len(critical_gestures)}")
            for gesture, count in sorted(critical_gestures.items(), key=lambda x: x[1]):
                print(f"   {gesture}: {count} muestras")
        
        # Gestos con buena cantidad (> 80 muestras)
        good_gestures = {k: v for k, v in gesture_counts.items() if v >= 80}
        
        if good_gestures:
            print(f"\n✅ GESTOS CON BUENA CANTIDAD (≥ 80 muestras): {len(good_gestures)}")
            for gesture, count in sorted(good_gestures.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {gesture}: {count} muestras")
        
        self.gestures_info = gesture_counts
        return gesture_counts
    
    def analyze_landmark_files(self):
        """Analiza archivos de landmarks existentes"""
        print("\n🔍 ANALIZANDO ARCHIVOS DE LANDMARKS")
        print("=" * 50)
        
        landmark_issues = []
        feature_sizes = Counter()
        
        for gesture_name, sample_count in self.gestures_info.items():
            gesture_path = os.path.join(self.data_path, gesture_name)
            
            if os.path.isdir(gesture_path):
                files = os.listdir(gesture_path)
                npy_files = [f for f in files if f.endswith('.npy')]
                
                if npy_files:
                    # Analizar algunos archivos de ejemplo
                    sample_files = npy_files[:5]  # Primeros 5 archivos
                    
                    for file in sample_files:
                        try:
                            file_path = os.path.join(gesture_path, file)
                            data = np.load(file_path)
                            
                            feature_sizes[data.shape[0] if data.ndim == 1 else data.size] += 1
                            
                            # Verificar calidad
                            if np.any(np.isnan(data)):
                                landmark_issues.append(f"{gesture_name}/{file}: Contiene NaN")
                            
                            if np.all(data == 0):
                                landmark_issues.append(f"{gesture_name}/{file}: Todo ceros")
                            
                            # Verificar rango de valores
                            if np.max(data) > 10 or np.min(data) < -10:
                                landmark_issues.append(f"{gesture_name}/{file}: Valores fuera de rango")
                                
                        except Exception as e:
                            landmark_issues.append(f"{gesture_name}/{file}: Error cargando - {e}")
        
        print(f"📊 Tamaños de features encontrados:")
        for size, count in feature_sizes.most_common():
            print(f"   {size} features: {count} archivos")
        
        if landmark_issues:
            print(f"\n⚠️  PROBLEMAS ENCONTRADOS: {len(landmark_issues)}")
            for issue in landmark_issues[:20]:  # Mostrar primeros 20
                print(f"   {issue}")
            if len(landmark_issues) > 20:
                print(f"   ... y {len(landmark_issues) - 20} más")
        else:
            print("✅ No se encontraron problemas evidentes en los landmarks")
    
    def suggest_improvements(self):
        """Sugiere mejoras específicas"""
        print("\n💡 SUGERENCIAS DE MEJORA")
        print("=" * 50)
        
        # Sugerencias basadas en el análisis
        suggestions = []
        
        critical_count = len([v for v in self.gestures_info.values() if v < 50])
        if critical_count > 0:
            suggestions.append(f"🎯 PRIORIDAD ALTA: {critical_count} gestos necesitan más muestras")
        
        low_count = len([v for v in self.gestures_info.values() if v < 80])
        if low_count > 0:
            suggestions.append(f"📈 Mejorar dataset: {low_count} gestos con pocas muestras")
        
        # Sugerencias técnicas
        suggestions.extend([
            "🔧 Extraer SOLO landmarks de manos (eliminar puntos faciales)",
            "🎯 Implementar augmentación de datos para gestos críticos",
            "📊 Usar transfer learning con datasets públicos (WLASL/MS-ASL)",
            "🧹 Limpiar datos: eliminar muestras con landmarks de baja calidad",
            "⚖️ Balancear dataset: igualar número de muestras por gesto"
        ])
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    
    def create_improvement_plan(self):
        """Crea un plan de mejora específico"""
        print("\n📋 PLAN DE MEJORA ESPECÍFICO")
        print("=" * 50)
        
        # Gestos que necesitan más datos
        critical_gestures = [k for k, v in self.gestures_info.items() if v < 50]
        
        if critical_gestures:
            print("🎯 FASE 1: Capturar más datos para gestos críticos")
            for gesture in critical_gestures[:10]:  # Primeros 10
                needed = 80 - self.gestures_info[gesture]
                print(f"   • {gesture}: necesita +{needed} muestras")
        
        print("\n🔧 FASE 2: Mejoras técnicas")
        print("   • Extraer landmarks solo de manos (126 features)")
        print("   • Implementar normalización mejorada")
        print("   • Aplicar augmentación de datos")
        
        print("\n📊 FASE 3: Transfer Learning")
        print("   • Descargar dataset WLASL o MS-ASL")
        print("   • Entrenar modelo base con datos públicos")
        print("   • Fine-tuning con datos LSE Ecuador")
        
        print("\n✅ FASE 4: Validación")
        print("   • Evaluar precisión mejorada")
        print("   • Pruebas en tiempo real")
        print("   • Ajustes finales")
    
    def generate_report(self):
        """Genera reporte completo"""
        print("📄 GENERANDO REPORTE COMPLETO...")
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_gestures": len(self.gestures_info),
            "total_samples": sum(self.gestures_info.values()),
            "critical_gestures": [k for k, v in self.gestures_info.items() if v < 50],
            "good_gestures": [k for k, v in self.gestures_info.items() if v >= 80],
            "gesture_counts": self.gestures_info
        }
        
        # Guardar reporte
        with open("analysis/dataset_analysis_detailed.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("✅ Reporte guardado en: analysis/dataset_analysis_detailed.json")
        
        return report
    
    def run_complete_analysis(self):
        """Ejecuta análisis completo"""
        print("🚀 INICIANDO ANÁLISIS COMPLETO DEL DATASET LSE ECUADOR")
        print("=" * 60)
        
        # Crear directorio de análisis si no existe
        os.makedirs("analysis", exist_ok=True)
        
        # Ejecutar análisis
        self.analyze_dataset_structure()
        self.analyze_landmark_files()
        self.suggest_improvements()
        self.create_improvement_plan()
        
        # Generar reporte
        report = self.generate_report()
        
        print("\n🎉 ANÁLISIS COMPLETADO")
        print("=" * 30)
        print("📊 Revisa el reporte detallado en: analysis/dataset_analysis_detailed.json")
        
        return report

def main():
    """Función principal"""
    analyzer = LSEDatasetAnalyzer()
    report = analyzer.run_complete_analysis()
    
    # Mostrar resumen final
    print(f"\n📈 RESUMEN EJECUTIVO:")
    print(f"   • Total gestos: {report['total_gestures']}")
    print(f"   • Total muestras: {report['total_samples']}")
    print(f"   • Gestos críticos: {len(report['critical_gestures'])}")
    print(f"   • Gestos buenos: {len(report['good_gestures'])}")
    
    critical_ratio = len(report['critical_gestures']) / report['total_gestures'] * 100
    
    if critical_ratio > 30:
        print(f"\n⚠️  ATENCIÓN: {critical_ratio:.1f}% de gestos tienen pocas muestras")
        print("   Recomendación: Priorizar captura de más datos")
    elif critical_ratio > 15:
        print(f"\n🔄 MODERADO: {critical_ratio:.1f}% de gestos necesitan mejora")
        print("   Recomendación: Combinar captura + transfer learning")
    else:
        print(f"\n✅ BUENO: Solo {critical_ratio:.1f}% de gestos críticos")
        print("   Recomendación: Enfocarse en optimización técnica")

if __name__ == "__main__":
    main()
