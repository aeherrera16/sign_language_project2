#!/usr/bin/env python3
"""
🎯 REFUERZO RÁPIDO DE GESTOS CRÍTICOS
=====================================
Script para mejorar rápidamente la precisión del modelo
"""

import os
import subprocess
import sys

def main():
    print("🎯 REFUERZO RÁPIDO DE GESTOS CRÍTICOS")
    print("=" * 50)
    print("Este script te ayudará a mejorar la precisión del modelo")
    print("reforzando los gestos con menos muestras.")
    print()
    
    # Gestos críticos identificados
    gestos_criticos = [
        ("yo", 21, 79),           # Necesita +79 muestras
        ("nosotros", 26, 74),     # Necesita +74 muestras  
        ("el", 37, 63),           # Necesita +63 muestras
        ("z", 40, 60),            # Necesita +60 muestras
        ("Jueves", 43, 57),       # Necesita +57 muestras
        ("Azul", 44, 56),         # Necesita +56 muestras
        ("Blanco", 45, 55),       # Necesita +55 muestras
        ("Morado", 45, 55),       # Necesita +55 muestras
    ]
    
    print("🔴 GESTOS QUE NECESITAN REFUERZO URGENTE:")
    print("-" * 50)
    for gesto, actual, necesita in gestos_criticos:
        print(f"   {gesto:<12}: {actual:>3} muestras → necesita +{necesita}")
    
    print("\n💡 RECOMENDACIÓN:")
    print("   - Graba 10-15 muestras por sesión")
    print("   - Haz 4-6 sesiones por gesto")
    print("   - Usa diferentes ángulos y velocidades")
    print("   - Asegúrate de hacer el gesto correctamente")
    
    while True:
        print("\n🎬 ¿Qué gesto quieres reforzar?")
        print("0. Salir")
        for i, (gesto, actual, necesita) in enumerate(gestos_criticos, 1):
            print(f"{i}. {gesto} (actual: {actual}, necesita: +{necesita})")
        
        try:
            choice = input("\nElige una opción (0-8): ").strip()
            
            if choice == "0":
                print("👋 ¡Hasta luego!")
                break
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(gestos_criticos):
                gesto_elegido = gestos_criticos[choice_num - 1][0]
                
                print(f"\n📹 ¿Cuántas sesiones quieres grabar para '{gesto_elegido}'?")
                print("   Recomendado: 3-5 sesiones de 10-15 muestras cada una")
                
                sesiones = input("Número de sesiones (recomendado 4): ").strip()
                if not sesiones:
                    sesiones = "4"
                
                try:
                    num_sesiones = int(sesiones)
                    print(f"\n🚀 ¡Perfecto! Vas a grabar {num_sesiones} sesiones de '{gesto_elegido}'")
                    print("\n💡 CONSEJOS IMPORTANTES:")
                    print("   1. 🖐️ Haz el gesto EXACTAMENTE como debe ser")
                    print("   2. ⏱️ Mantén el gesto por 2-3 segundos")
                    print("   3. 📏 Mantén distancia adecuada de la cámara")
                    print("   4. 💡 Asegúrate de tener buena iluminación")
                    print("   5. 🎨 Usa fondo contrastante")
                    
                    input("\n🎬 Presiona ENTER cuando estés listo para empezar...")
                    
                    for i in range(1, num_sesiones + 1):
                        print(f"\n📹 SESIÓN {i} de {num_sesiones}")
                        print(f"Preparándote para grabar '{gesto_elegido}'...")
                        input("Presiona ENTER para iniciar la grabación...")
                        
                        # Ejecutar script de grabación
                        try:
                            subprocess.run([sys.executable, "record_dataset.py", gesto_elegido], 
                                         check=True)
                            print(f"✅ Sesión {i} completada")
                        except subprocess.CalledProcessError as e:
                            print(f"❌ Error en la sesión {i}: {e}")
                            continue
                        except KeyboardInterrupt:
                            print("\n⏹️ Grabación interrumpida por el usuario")
                            break
                    
                    print(f"\n🎉 ¡Todas las sesiones de '{gesto_elegido}' completadas!")
                    print("\n🔄 ¿Quieres reentrenar el modelo ahora? (recomendado)")
                    retrain = input("Reentrenar modelo? (s/n): ").strip().lower()
                    
                    if retrain in ['s', 'si', 'yes', 'y']:
                        print("\n🧠 Reentrenando modelo con los nuevos datos...")
                        try:
                            subprocess.run([sys.executable, "train_model.py"], check=True)
                            print("✅ Modelo reentrenado exitosamente")
                            print("🎯 La precisión debería haber mejorado")
                        except subprocess.CalledProcessError as e:
                            print(f"❌ Error reentrenando: {e}")
                    
                except ValueError:
                    print("❌ Número de sesiones inválido")
                    continue
            else:
                print("❌ Opción inválida")
                continue
                
        except ValueError:
            print("❌ Opción inválida")
            continue
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break

if __name__ == "__main__":
    main()
