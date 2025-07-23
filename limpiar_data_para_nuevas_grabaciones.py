#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMPIADOR DE DATOS LSE ECUADOR
Elimina toda la data actual para permitir grabación de nuevas señas
"""

import os
import shutil
import sys

def print_header():
    """Imprimir encabezado"""
    print("=" * 60)
    print("    LSE ECUADOR - LIMPIADOR DE DATOS")
    print("    Eliminar data actual y preparar para nuevas grabaciones")
    print("=" * 60)

def check_data_folders():
    """Verificar qué datos existen actualmente"""
    print("\nDATA ACTUAL ENCONTRADA:")
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print("❌ No existe carpeta 'data'")
        return []
    
    folders_found = []
    total_files = 0
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            npy_files = [f for f in os.listdir(item_path) if f.endswith('.npy')]
            if npy_files:
                print(f"   📁 {item}: {len(npy_files)} archivos .npy")
                folders_found.append((item, len(npy_files)))
                total_files += len(npy_files)
    
    if folders_found:
        print(f"\n📊 TOTAL: {len(folders_found)} carpetas, {total_files} archivos")
    else:
        print("✅ No hay datos para eliminar")
    
    return folders_found

def backup_current_data():
    """Crear backup de datos actuales"""
    print("\n🔄 CREANDO BACKUP...")
    
    if not os.path.exists("data"):
        print("❌ No hay datos para respaldar")
        return False
    
    # Crear carpeta de backup con timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_data_{timestamp}"
    
    try:
        shutil.copytree("data", backup_dir)
        print(f"✅ Backup creado: {backup_dir}")
        return True
    except Exception as e:
        print(f"❌ Error creando backup: {e}")
        return False

def clean_data_folders():
    """Eliminar todas las carpetas de datos"""
    print("\n🗑️ ELIMINANDO DATOS ACTUALES...")
    
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print("✅ No hay carpeta 'data' que eliminar")
        return True
    
    try:
        # Eliminar carpeta completa
        shutil.rmtree(data_dir)
        print("✅ Carpeta 'data' eliminada completamente")
        
        # Recrear carpeta vacía
        os.makedirs(data_dir)
        print("✅ Carpeta 'data' recreada vacía")
        
        return True
        
    except Exception as e:
        print(f"❌ Error eliminando datos: {e}")
        return False

def clean_model_files():
    """Eliminar modelo entrenado actual"""
    print("\n🧠 ELIMINANDO MODELO ACTUAL...")
    
    model_files = [
        "model/gesture_model.h5",
        "model/labels.pkl", 
        "model/final_metrics.json",
        "model/training_history.json",
        "model/training_metrics.json"
    ]
    
    files_removed = 0
    
    for file_path in model_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   ✅ Eliminado: {file_path}")
                files_removed += 1
            except Exception as e:
                print(f"   ❌ Error eliminando {file_path}: {e}")
        else:
            print(f"   ℹ️  No existe: {file_path}")
    
    if files_removed > 0:
        print(f"✅ {files_removed} archivos de modelo eliminados")
    else:
        print("ℹ️  No había modelos que eliminar")

def setup_for_recording():
    """Preparar el sistema para nueva grabación"""
    print("\n📹 PREPARANDO PARA NUEVA GRABACIÓN...")
    
    # Crear estructura básica de carpetas
    gestos_base = ["hola", "adios", "gracias", "si", "no"]
    
    print("Creando carpetas para gestos básicos:")
    for gesto in gestos_base:
        folder_path = os.path.join("data", gesto)
        os.makedirs(folder_path, exist_ok=True)
        print(f"   📁 {folder_path}")
    
    print("✅ Sistema preparado para grabación")

def show_next_steps():
    """Mostrar pasos siguientes"""
    print("\n" + "="*60)
    print("🎯 PRÓXIMOS PASOS:")
    print("="*60)
    print()
    print("1. 📹 GRABAR NUEVAS SEÑAS:")
    print("   python main_interface.py")
    print("   → Clic en 'Grabar Dataset'")
    print("   → Graba 30+ ejemplos de cada seña")
    print()
    print("2. 🧠 ENTRENAR NUEVO MODELO:")
    print("   python main_interface.py") 
    print("   → Clic en 'Entrenar Modelo'")
    print()
    print("3. ✅ VERIFICAR SISTEMA:")
    print("   python verificacion_sistema_completo.py")
    print()
    print("4. 🎮 PROBAR RECONOCIMIENTO:")
    print("   python scripts/recognition/real_time_translate.py")
    print()
    print("💡 RECORDATORIO SEÑAS LSE ECUADOR:")
    print("   🖐️  HOLA: Mano abierta hacia adelante")
    print("   👋 ADIOS: Mano lateral (izq-der)")
    print("   🙏 GRACIAS: Mano al pecho")
    print("   👍 SÍ: Puño vertical (arriba-abajo)")
    print("   👎 NO: Dedo horizontal (izq-der)")

def main():
    """Función principal"""
    print_header()
    
    # Verificar datos actuales
    folders_found = check_data_folders()
    
    if not folders_found:
        print("\n✅ No hay datos que eliminar. Sistema ya está limpio.")
        setup_for_recording()
        show_next_steps()
        return
    
    # Confirmar eliminación
    print(f"\n⚠️  ADVERTENCIA:")
    print(f"   Se eliminarán {len(folders_found)} carpetas de datos")
    print(f"   y el modelo entrenado actual")
    print(f"\n❓ ¿Estás seguro de eliminar toda la data actual?")
    print(f"   (Se creará un backup automático)")
    
    while True:
        respuesta = input("\n[S]í / [N]o / [B]ackup solo: ").strip().lower()
        
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            break
        elif respuesta in ['n', 'no']:
            print("\n❌ Operación cancelada")
            return
        elif respuesta in ['b', 'backup']:
            print("\n📦 Creando solo backup...")
            if backup_current_data():
                print("✅ Backup creado exitosamente")
            return
        else:
            print("❌ Respuesta no válida. Usa S/N/B")
    
    print("\n🚀 INICIANDO LIMPIEZA COMPLETA...")
    
    # Paso 1: Crear backup
    print("\n" + "="*50)
    backup_success = backup_current_data()
    
    # Paso 2: Limpiar datos
    print("\n" + "="*50)
    data_success = clean_data_folders()
    
    # Paso 3: Limpiar modelo
    print("\n" + "="*50)
    clean_model_files()
    
    # Paso 4: Preparar para grabación
    print("\n" + "="*50)
    setup_for_recording()
    
    # Mostrar resultado
    print("\n" + "="*60)
    if data_success:
        print("🎉 LIMPIEZA COMPLETADA EXITOSAMENTE")
        print("="*60)
        print("✅ Datos anteriores eliminados")
        if backup_success:
            print("✅ Backup creado")
        print("✅ Sistema preparado para nueva grabación")
        print("✅ Carpetas básicas creadas")
    else:
        print("❌ LIMPIEZA INCOMPLETA")
        print("   Revisa los errores mostrados arriba")
    
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
    
    input("\nPresiona Enter para salir...")
