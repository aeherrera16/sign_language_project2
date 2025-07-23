#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 REFUERZO RAPIDO DE GESTOS CRITICOS
=====================================
Script para mejorar rapidamente la precision del modelo
"""

import os
import subprocess
import sys

def main():
    print("REFUERZO RAPIDO DE GESTOS CRITICOS")
    print("=" * 50)
    print("Este script te ayudara a mejorar la precision del modelo")
    print("reforzando los gestos con menos muestras.")
    print()
    
    # Gestos criticos identificados
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
    
    print("GESTOS QUE NECESITAN REFUERZO URGENTE:")
    print("-" * 50)
    for gesto, actual, necesita in gestos_criticos:
        print(f"   {gesto:<12}: {actual:>3} muestras -> necesita +{necesita}")
    
    print("\nRECOMENDACION:")
    print("   - Graba 10-15 muestras por sesion")
    print("   - Haz 4-6 sesiones por gesto")
    print("   - Usa diferentes angulos y velocidades")
    print("   - Asegurate de hacer el gesto correctamente")
    
    while True:
        print("\nQue gesto quieres reforzar?")
        print("0. Salir")
        for i, (gesto, actual, necesita) in enumerate(gestos_criticos, 1):
            print(f"{i}. {gesto} (actual: {actual}, necesita: +{necesita})")
        
        try:
            choice = input("\nElige una opcion (0-8): ").strip()
            
            if choice == "0":
                print("Hasta luego!")
                break
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(gestos_criticos):
                gesto_elegido = gestos_criticos[choice_num - 1][0]
                
                print(f"\nCuantas sesiones quieres grabar para '{gesto_elegido}'?")
                print("   Recomendado: 3-5 sesiones de 10-15 muestras cada una")
                
                sesiones = input("Numero de sesiones (recomendado 4): ").strip()
                if not sesiones:
                    sesiones = "4"
                
                try:
                    num_sesiones = int(sesiones)
                    print(f"\nPerfecto! Vas a grabar {num_sesiones} sesiones de '{gesto_elegido}'")
                    print("\nCONSEJOS IMPORTANTES:")
                    print("   1. Haz el gesto EXACTAMENTE como debe ser")
                    print("   2. Manten el gesto por 2-3 segundos")
                    print("   3. Manten distancia adecuada de la camara")
                    print("   4. Asegurate de tener buena iluminacion")
                    print("   5. Usa fondo contrastante")
                    
                    input("\nPresiona ENTER cuando estes listo para empezar...")
                    
                    for i in range(1, num_sesiones + 1):
                        print(f"\nSESION {i} de {num_sesiones}")
                        print(f"Preparandote para grabar '{gesto_elegido}'...")
                        input("Presiona ENTER para iniciar la grabacion...")
                        
                        # Ejecutar script de grabacion
                        try:
                            subprocess.run([sys.executable, "scripts/core/record_dataset.py", gesto_elegido], 
                                         check=True)
                            print(f"Sesion {i} completada")
                        except subprocess.CalledProcessError as e:
                            print(f"Error en la sesion {i}: {e}")
                            continue
                        except KeyboardInterrupt:
                            print("\nGrabacion interrumpida por el usuario")
                            break
                    
                    print(f"\nTodas las sesiones de '{gesto_elegido}' completadas!")
                    print("\nQuieres reentrenar el modelo ahora? (recomendado)")
                    retrain = input("Reentrenar modelo? (s/n): ").strip().lower()
                    
                    if retrain in ['s', 'si', 'yes', 'y']:
                        print("\nReentrenando modelo con los nuevos datos...")
                        try:
                            subprocess.run([sys.executable, "scripts/core/train_model.py"], check=True)
                            print("Modelo reentrenado exitosamente")
                            print("La precision deberia haber mejorado")
                        except subprocess.CalledProcessError as e:
                            print(f"Error reentrenando: {e}")
                    
                except ValueError:
                    print("Numero de sesiones invalido")
                    continue
            else:
                print("Opcion invalida")
                continue
                
        except ValueError:
            print("Opcion invalida")
            continue
        except KeyboardInterrupt:
            print("\nHasta luego!")
            break

if __name__ == "__main__":
    main()
