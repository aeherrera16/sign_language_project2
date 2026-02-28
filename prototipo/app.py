#!/usr/bin/env python3
"""
MENÚ PRINCIPAL - Traductor LSE
Interfaz de menú en terminal.
"""

import os
import subprocess
import sys

DIR = os.path.dirname(os.path.abspath(__file__))

def limpiar():
    os.system('clear' if os.name != 'nt' else 'cls')

def mostrar_menu():
    limpiar()
    print("\n" + "="*50)
    print("       🤟 TRADUCTOR LSE - PROTOTIPO")
    print("="*50)
    print("\n  Selecciona una opción:\n")
    print("    1. 📹 Grabar Señas")
    print("    2. 🧠 Entrenar Modelo")
    print("    3. 🔊 Iniciar Traductor")
    print("    4. 📊 Evaluar Métricas ISO")
    print("    5. 🚪 Salir")
    print("\n" + "-"*50)

def ejecutar_script(script):
    """Ejecuta un script."""
    script_path = os.path.join(DIR, script)
    print(f"\n▶ Ejecutando {script}...\n")
    print("-"*50)
    
    try:
        subprocess.run([sys.executable, script_path])
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelado por el usuario")
    
    input("\n\nPresiona ENTER para volver al menú...")

def main():
    while True:
        mostrar_menu()
        
        try:
            opcion = input("  Opción [1-5]: ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        
        if opcion == '1':
            ejecutar_script('1_grabar_senas.py')
        elif opcion == '2':
            ejecutar_script('2_entrenar_modelo.py')
        elif opcion == '3':
            ejecutar_script('3_traductor.py')
        elif opcion == '4':
            ejecutar_script('4_evaluar_iso25023.py')
        elif opcion == '5':
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("\n❌ Opción no válida")
            input("Presiona ENTER...")

if __name__ == "__main__":
    main()
