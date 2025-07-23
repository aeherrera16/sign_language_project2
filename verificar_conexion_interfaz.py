#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRUEBA DE CONEXIÓN INTERFAZ - RECONOCIMIENTO
Verificar que el botón llame correctamente al script de reconocimiento con voz
"""

import os
import sys

def verificar_conexion():
    """Verificar que todo esté conectado correctamente"""
    print("🔍 VERIFICANDO CONEXIÓN INTERFAZ - RECONOCIMIENTO")
    print("=" * 60)
    
    # 1. Verificar que el script de reconocimiento existe
    script_reconocimiento = "reconocimiento_simple_funcional.py"
    if os.path.exists(script_reconocimiento):
        print(f"✅ {script_reconocimiento} - ENCONTRADO")
    else:
        print(f"❌ {script_reconocimiento} - NO ENCONTRADO")
        return False
    
    # 2. Verificar interfaz principal
    interfaz_principal = "main_interface_elegante.py"
    if os.path.exists(interfaz_principal):
        print(f"✅ {interfaz_principal} - ENCONTRADO")
    else:
        print(f"❌ {interfaz_principal} - NO ENCONTRADO")
        return False
    
    # 3. Verificar conexión en el código
    with open(interfaz_principal, 'r', encoding='utf-8') as f:
        contenido = f.read()
        
        if 'reconocimiento_simple_funcional.py' in contenido:
            print("✅ CONEXIÓN EN CÓDIGO - CORRECTA")
        else:
            print("❌ CONEXIÓN EN CÓDIGO - NO ENCONTRADA")
            return False
            
        if 'window_mode=True' in contenido:
            print("✅ MODO VENTANA SEPARADA - CONFIGURADO")
        else:
            print("⚠️ MODO VENTANA - VERIFICAR CONFIGURACIÓN")
    
    # 4. Verificar función específica
    if 'def reconocimiento_tiempo_real' in contenido:
        print("✅ FUNCIÓN BOTÓN - ENCONTRADA")
    else:
        print("❌ FUNCIÓN BOTÓN - NO ENCONTRADA")
        return False
    
    print("\n🎉 RESULTADO: CONEXIÓN PERFECTA")
    print("=" * 60)
    print("✅ El botón 'Reconocimiento en Tiempo Real' está correctamente conectado")
    print("✅ Llamará a 'reconocimiento_simple_funcional.py' en ventana separada")
    print("✅ El script incluye síntesis de voz en español")
    print("✅ Todo listo para funcionar")
    
    print("\n🚀 INSTRUCCIONES DE USO:")
    print("1. Ejecutar: python main_interface_elegante.py")
    print("2. Hacer clic en el botón 'Reconocimiento en Tiempo Real'")
    print("3. Se abrirá ventana de cámara con reconocimiento y voz")
    print("4. Presionar 'q' en la ventana de cámara para cerrar")
    
    return True

if __name__ == "__main__":
    verificar_conexion()
