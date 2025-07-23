#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMPIEZA EXHAUSTIVA FINAL DEL PROYECTO LSE ECUADOR
Eliminar todos los archivos innecesarios, backups viejos y scripts de limpieza
"""

import os
import shutil
from datetime import datetime

def crear_backup_final():
    """Crear backup final antes de la limpieza exhaustiva"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_limpieza_final_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Archivos críticos a respaldar
    archivos_criticos = [
        "main_interface_elegante.py",
        "main_interface.py", 
        "verificacion_sistema_completo.py",
        "utils.py",
        "configuracion_rapida.py",
        "limpiar_data_para_nuevas_grabaciones.py"
    ]
    
    for archivo in archivos_criticos:
        if os.path.exists(archivo):
            shutil.copy2(archivo, backup_dir)
    
    return backup_dir

def identificar_archivos_innecesarios():
    """Identificar todos los archivos que se pueden eliminar"""
    
    # Scripts de limpieza temporales (este mismo incluido)
    scripts_limpieza = [
        "cleanup_duplicates.py",
        "cleanup_markdown.py", 
        "generar_reporte_limpieza.py",
        "limpieza_exhaustiva_final.py"  # Este mismo script
    ]
    
    # Archivos de verificación duplicados o redundantes
    verificacion_redundantes = [
        "verificacion_funcionalidades.py"  # Mantenemos solo verificacion_sistema_completo.py
    ]
    
    # Archivos de texto/documentación redundantes
    docs_redundantes = [
        "INICIO_RAPIDO.txt",  # Info ya está en README.md
        "REPORTE_LIMPIEZA_FINAL.md"  # Temporal
    ]
    
    # Backups múltiples (mantener solo el más reciente)
    backups_antiguos = []
    backups_encontrados = []
    
    for item in os.listdir('.'):
        if os.path.isdir(item) and 'backup_' in item:
            backups_encontrados.append(item)
    
    # Ordenar backups por fecha y mantener solo los 2 más recientes
    backups_encontrados.sort()
    if len(backups_encontrados) > 2:
        backups_antiguos = backups_encontrados[:-2]  # Eliminar todos excepto los 2 últimos
    
    return scripts_limpieza, verificacion_redundantes, docs_redundantes, backups_antiguos

def limpiar_directorio_evaluation():
    """Limpiar directorio evaluation si está vacío o tiene archivos temporales"""
    eval_dir = "evaluation"
    if os.path.exists(eval_dir):
        try:
            contenido = os.listdir(eval_dir)
            if len(contenido) == 0:
                os.rmdir(eval_dir)
                return f"📁 {eval_dir}/ (directorio vacío)"
            else:
                print(f"  📁 {eval_dir}/ contiene {len(contenido)} archivos - conservado")
                return None
        except:
            return None
    return None

def main():
    """Función principal de limpieza exhaustiva"""
    print("=" * 70)
    print("🧹 LIMPIEZA EXHAUSTIVA FINAL - LSE ECUADOR")
    print("=" * 70)
    
    # Crear backup final
    backup_final = crear_backup_final()
    print(f"📦 Backup final creado: {backup_final}")
    
    # Identificar archivos a eliminar
    scripts_limpieza, verificacion_redundantes, docs_redundantes, backups_antiguos = identificar_archivos_innecesarios()
    
    print(f"\n🎯 ARCHIVOS A ELIMINAR:")
    
    total_eliminados = 0
    
    # Eliminar scripts de limpieza
    print(f"\n📜 Scripts de limpieza temporales:")
    for archivo in scripts_limpieza:
        if os.path.exists(archivo):
            print(f"  ❌ {archivo}")
            if archivo != "limpieza_exhaustiva_final.py":  # No eliminar este script aún
                os.remove(archivo)
                total_eliminados += 1
    
    # Eliminar verificaciones redundantes  
    print(f"\n🔍 Scripts de verificación redundantes:")
    for archivo in verificacion_redundantes:
        if os.path.exists(archivo):
            print(f"  ❌ {archivo}")
            os.remove(archivo)
            total_eliminados += 1
    
    # Eliminar documentación redundante
    print(f"\n📄 Documentación redundante:")
    for archivo in docs_redundantes:
        if os.path.exists(archivo):
            print(f"  ❌ {archivo}")
            os.remove(archivo)
            total_eliminados += 1
    
    # Eliminar backups antiguos
    print(f"\n📦 Backups antiguos:")
    for backup in backups_antiguos:
        if os.path.exists(backup):
            print(f"  ❌ {backup}/")
            shutil.rmtree(backup)
            total_eliminados += 1
    
    # Limpiar directorio evaluation
    print(f"\n📁 Directorios vacíos:")
    eval_eliminado = limpiar_directorio_evaluation()
    if eval_eliminado:
        print(f"  ❌ {eval_eliminado}")
        total_eliminados += 1
    
    print(f"\n" + "=" * 70)
    print(f"✅ LIMPIEZA EXHAUSTIVA COMPLETADA")
    print(f"=" * 70)
    print(f"🗑️ Total archivos/directorios eliminados: {total_eliminados}")
    print(f"📦 Backup de seguridad: {backup_final}")
    
    print(f"\n📋 ARCHIVOS PRINCIPALES CONSERVADOS:")
    archivos_principales = [
        "main_interface_elegante.py",
        "main_interface.py",
        "verificacion_sistema_completo.py", 
        "utils.py",
        "configuracion_rapida.py",
        "limpiar_data_para_nuevas_grabaciones.py",
        "README.md",
        "GUIA_GRABACION_NUEVAS_SENAS.md",
        "DOCUMENTACION_CONSOLIDADA.md"
    ]
    
    for archivo in archivos_principales:
        if os.path.exists(archivo):
            print(f"  ✅ {archivo}")
    
    print(f"\n📁 DIRECTORIOS CONSERVADOS:")
    directorios = ["scripts/", "data/", "model/", "venv310/"]
    for directorio in directorios:
        if os.path.exists(directorio):
            print(f"  ✅ {directorio}")
    
    print(f"\n🚀 PROYECTO LIMPIO Y LISTO PARA USAR:")
    print(f"  • Ejecutar: python main_interface_elegante.py")
    print(f"  • Verificar: python verificacion_sistema_completo.py")
    
    # Auto-eliminar este script al final
    print(f"\n🗑️ Auto-eliminando script de limpieza...")
    
    return total_eliminados

if __name__ == "__main__":
    eliminados = main()
    
    # Auto-eliminar este script
    import sys
    script_actual = sys.argv[0]
    if os.path.exists(script_actual):
        os.remove(script_actual)
        print(f"  ❌ {script_actual} (auto-eliminado)")
        eliminados += 1
    
    print(f"\n🎉 LIMPIEZA FINAL COMPLETADA - {eliminados} elementos eliminados")
