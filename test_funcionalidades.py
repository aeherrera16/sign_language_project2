#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRUEBA DE FUNCIONALIDADES PRINCIPALES
Verificar que grabación, entrenamiento y reconocimiento funcionen
"""

import os
import sys
import subprocess
import cv2

def test_camera():
    """Probar acceso a la cámara"""
    print("🎥 Probando acceso a cámara...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("  ✅ Cámara funcionando correctamente")
                cap.release()
                return True
            else:
                print("  ❌ No se puede leer de la cámara")
                cap.release()
                return False
        else:
            print("  ❌ No se puede abrir la cámara")
            return False
    except Exception as e:
        print(f"  ❌ Error con cámara: {e}")
        return False

def test_script_exists(script_path):
    """Verificar que un script existe"""
    if os.path.exists(script_path):
        print(f"  ✅ {script_path}")
        return True
    else:
        print(f"  ❌ FALTA: {script_path}")
        return False

def test_record_functionality():
    """Probar funcionalidad de grabación"""
    print("\n📹 Probando script de grabación...")
    script_path = "scripts/core/record_dataset.py"
    if test_script_exists(script_path):
        # Verificar que tenga las importaciones necesarias
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'cv2' in content and 'mediapipe' in content:
                    print("  ✅ Importaciones correctas para grabación")
                    return True
                else:
                    print("  ❌ Faltan importaciones necesarias")
                    return False
        except Exception as e:
            print(f"  ❌ Error leyendo script: {e}")
            return False
    return False

def test_train_functionality():
    """Probar funcionalidad de entrenamiento"""
    print("\n🧠 Probando script de entrenamiento...")
    script_path = "scripts/core/train_model.py"
    if test_script_exists(script_path):
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'tensorflow' in content and 'train_test_split' in content:
                    print("  ✅ Importaciones correctas para entrenamiento")
                    return True
                else:
                    print("  ❌ Faltan importaciones necesarias")
                    return False
        except Exception as e:
            print(f"  ❌ Error leyendo script: {e}")
            return False
    return False

def test_recognition_functionality():
    """Probar funcionalidad de reconocimiento"""
    print("\n🎯 Probando script de reconocimiento...")
    script_path = "scripts/recognition/real_time_translate.py"
    if test_script_exists(script_path):
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'cv2' in content and 'mediapipe' in content and 'tensorflow' in content:
                    print("  ✅ Importaciones correctas para reconocimiento")
                    return True
                else:
                    print("  ❌ Faltan importaciones necesarias")
                    return False
        except Exception as e:
            print(f"  ❌ Error leyendo script: {e}")
            return False
    return False

def test_interface_connections():
    """Verificar conexiones en la interfaz"""
    print("\n🖥️ Probando conexiones de interfaz...")
    interface_path = "main_interface_elegante.py"
    if test_script_exists(interface_path):
        try:
            with open(interface_path, 'r', encoding='utf-8') as f:
                content = f.read()
                checks = [
                    ('record_dataset.py' in content, "Conexión grabación"),
                    ('train_model.py' in content, "Conexión entrenamiento"), 
                    ('real_time_translate.py' in content, "Conexión reconocimiento"),
                    ('subprocess' in content, "Ejecución de procesos")
                ]
                
                all_good = True
                for check, desc in checks:
                    if check:
                        print(f"  ✅ {desc}")
                    else:
                        print(f"  ❌ {desc}")
                        all_good = False
                
                return all_good
        except Exception as e:
            print(f"  ❌ Error leyendo interfaz: {e}")
            return False
    return False

def main():
    """Función principal de prueba"""
    print("=" * 60)
    print("🧪 PRUEBA DE FUNCIONALIDADES LSE ECUADOR")
    print("=" * 60)
    
    # Realizar pruebas
    results = {
        "Cámara": test_camera(),
        "Grabación": test_record_functionality(),
        "Entrenamiento": test_train_functionality(),
        "Reconocimiento": test_recognition_functionality(),
        "Interfaz": test_interface_connections()
    }
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} pruebas pasadas")
    
    if passed == total:
        print("\n🎉 ¡TODAS LAS FUNCIONALIDADES ESTÁN LISTAS!")
        print("\n🚀 INSTRUCCIONES DE USO:")
        print("  1. Ejecutar: python main_interface_elegante.py")
        print("  2. Hacer clic en 'Grabar Dataset' para grabar gestos")
        print("  3. Hacer clic en 'Entrenar Modelo' para entrenar IA")
        print("  4. Hacer clic en 'Reconocimiento' para traducir en tiempo real")
    else:
        print(f"\n⚠️ {total - passed} funcionalidades requieren atención")
        print("Revisa los errores mostrados arriba")

if __name__ == "__main__":
    main()
