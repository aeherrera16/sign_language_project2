# -*- coding: utf-8 -*-
"""
🇪🇨 ASISTENTE INTELIGENTE PARA EXPANSION DE DATASET LSE
Este script te guia paso a paso para expandir tu dataset de lengua de senas ecuatoriana
"""

import os
import json
import sys
import subprocess
from datetime import datetime
from collections import defaultdict

class LSEDatasetExpansion:
    def __init__(self):
        self.gestures_ecuatorianos = {
            "🏙️ Geografia": [
                "Costa", "Sierra", "Oriente", "Galapagos",
                "Quito", "Guayaquil", "Cuenca", "Ambato", "Machala",
                "Portoviejo", "Loja", "Riobamba", "Ibarra", "Manta",
                "Mitad_del_Mundo", "Cotopaxi", "Chimborazo", "Banos", "Otavalo"
            ],
            "🍽️ Gastronomia": [
                "encebollado", "fritada", "hornado", "cuy", "locro", "fanesca",
                "bolon", "tigrillo", "corviche", "empanadas_de_viento",
                "humitas", "tamales", "llapingachos", "chicha", "colada_morada"
            ],
            "💬 Expresiones": [
                "chevere", "bacan", "jama", "chuta", "achachay", "atatay",
                "nano", "nana", "pana", "causa", "longo", "chulla",
                "yapa", "guambra", "taita"
            ],
            "👔 Profesiones": [
                "doctor", "enfermera", "profesor", "ingeniero", "abogado",
                "contador", "arquitecto", "dentista", "agricultor", "pescador",
                "artesano", "comerciante", "chofer", "cocinero", "policia"
            ],
            "🏢 Instituciones": [
                "hospital", "clinica", "banco", "registro_civil", "municipio",
                "universidad", "colegio", "escuela", "mercado", "farmacia"
            ]
        }
        
        self.critical_gestures = [
            ("yo", 21), ("nosotros", 26), ("el", 37), ("z", 40), ("Jueves", 43),
            ("Azul", 44), ("Blanco", 45), ("Morado", 45), ("q", 51), ("Martes", 52)
        ]

    def analyze_current_dataset(self):
        """Analiza el estado actual del dataset"""
        if not os.path.exists("data"):
            print(" No se encontro la carpeta 'data'")
            return None
        
        gesture_counts = {}
        for folder in os.listdir("data"):
            if os.path.isdir(os.path.join("data", folder)):
                npy_files = [f for f in os.listdir(os.path.join("data", folder)) if f.endswith('.npy')]
                gesture_counts[folder] = len(npy_files)
        
        return gesture_counts

    def show_status(self):
        """Muestra el estado actual del dataset"""
        counts = self.analyze_current_dataset()
        if not counts:
            return
        
        total_gestures = len(counts)
        total_samples = sum(counts.values())
        avg_samples = total_samples / total_gestures if total_gestures > 0 else 0
        
        print("🇪🇨 ESTADO ACTUAL DEL DATASET LSE")
        print("=" * 50)
        print(f"📂 Total de gestos: {total_gestures}")
        print(f"📋 Total de muestras: {total_samples:,}")
        print(f"📊 Promedio por gesto: {avg_samples:.1f}")
        
        # Categorizar gestos por cantidad de muestras
        critical = [(g, c) for g, c in counts.items() if c < 50]
        low = [(g, c) for g, c in counts.items() if 50 <= c < 80]
        good = [(g, c) for g, c in counts.items() if 80 <= c < 100]
        excellent = [(g, c) for g, c in counts.items() if c >= 100]
        
        print(f"\n🔴 CRITICOS (<50): {len(critical)} gestos")
        print(f"⚠️ BAJOS (50-79): {len(low)} gestos")
        print(f" BUENOS (80-99): {len(good)} gestos")
        print(f"🌟 EXCELENTES (100+): {len(excellent)} gestos")
        
        if critical:
            print("\n🚨 GESTOS QUE NECESITAN ATENCION URGENTE:")
            for gesture, count in sorted(critical, key=lambda x: x[1])[:10]:
                needed = 80 - count
                print(f"   • {gesture}: {count} muestras (necesita +{needed})")

    def record_gesture_session(self, gesture_name, target_new_samples=20):
        """Graba una sesion de muestras para un gesto especifico"""
        print(f"\n SESION DE GRABACION: '{gesture_name}'")
        print("=" * 40)
        print(f" Objetivo: +{target_new_samples} muestras nuevas")
        print("\n📋 PREPARACION:")
        print("   • Asegurate de tener buena iluminacion")
        print("   • Limpia el lente de la camara")
        print("   • Prepara el gesto mentalmente")
        print("   • Ten agua cerca (puedes tardar un rato)")
        
        print("\n CONSEJOS PARA ESTA SESION:")
        print("   • Varia ligeramente el angulo de las manos")
        print("   • Cambia la velocidad del gesto (lento/normal/rapido)")
        print("   • Mueve ligeramente la posicion del cuerpo")
        print("   • Si tienes personas disponibles, que ellas tambien graben")
        
        confirm = input("\n➤ Estas listo para grabar? (s/n): ").lower().strip()
        if confirm != 's':
            print("⏸️ Sesion cancelada")
            return False
        
        # Grabacion en lotes pequenos para evitar fatiga
        sessions = target_new_samples // 5  # Sesiones de 5 muestras
        remainder = target_new_samples % 5
        
        print(f"\n Dividiremos en {sessions} sesiones de 5 muestras")
        if remainder:
            print(f"   + 1 sesion final de {remainder} muestras")
        
        try:
            for i in range(sessions):
                print(f"\n Sesion {i+1}/{sessions} (5 muestras)")
                input("   Presiona ENTER cuando estes listo...")
                subprocess.run([sys.executable, "record_dataset.py", gesture_name], check=True)
                
                if i < sessions - 1:  # No preguntar en la ultima sesion
                    continue_session = input("   Continuar con la siguiente sesion? (s/n): ").lower().strip()
                    if continue_session != 's':
                        print("⏸️ Sesiones restantes canceladas")
                        break
            
            if remainder:
                print(f"\n Sesion final ({remainder} muestras)")
                input("   Presiona ENTER cuando estes listo...")
                subprocess.run([sys.executable, "record_dataset.py", gesture_name], check=True)
            
            print(f" Sesion completada para '{gesture_name}'!")
            self.log_progress(gesture_name, target_new_samples)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f" Error durante la grabacion: {e}")
            return False
        except KeyboardInterrupt:
            print(f"\n Grabacion interrumpida por el usuario")
            return False

    def suggest_priority_plan(self):
        """Sugiere un plan de prioridades basado en el analisis actual"""
        counts = self.analyze_current_dataset()
        if not counts:
            return
        
        print("\n PLAN DE PRIORIDADES RECOMENDADO")
        print("=" * 45)
        
        # Prioridad 1: Gestos criticos
        critical = [(g, c) for g, c in counts.items() if c < 50]
        if critical:
            print("\n🔴 PRIORIDAD 1 - GESTOS CRITICOS:")
            for i, (gesture, count) in enumerate(sorted(critical, key=lambda x: x[1])[:5], 1):
                needed = max(80 - count, 20)  # Al menos 20 muestras nuevas
                print(f"   {i}. {gesture}: {count} -> objetivo 80+ (+{needed} muestras)")
        
        # Prioridad 2: Gestos bajos
        low = [(g, c) for g, c in counts.items() if 50 <= c < 80]
        if low:
            print("\n⚠️ PRIORIDAD 2 - GESTOS BAJOS:")
            for i, (gesture, count) in enumerate(sorted(low, key=lambda x: x[1])[:5], 1):
                needed = 80 - count
                print(f"   {i}. {gesture}: {count} -> objetivo 80+ (+{needed} muestras)")
        
        # Prioridad 3: Nuevo vocabulario ecuatoriano
        print("\n🆕 PRIORIDAD 3 - VOCABULARIO ECUATORIANO:")
        print("   Despues de completar las prioridades 1 y 2:")
        for category, gestures in list(self.gestures_ecuatorianos.items())[:3]:
            print(f"   • {category}: {len(gestures)} gestos nuevos")

    def interactive_expansion_session(self):
        """Sesion interactiva de expansion del dataset"""
        print("\n SESION INTERACTIVA DE EXPANSION")
        print("=" * 40)
        
        while True:
            print("\n🎮 OPCIONES:")
            print("1. 📊 Ver estado actual del dataset")
            print("2.  Ver plan de prioridades recomendado")
            print("3. 🔴 Reforzar gesto critico")
            print("4. 🆕 Agregar gesto ecuatoriano nuevo")
            print("5.  Sesion de grabacion personalizada")
            print("6. 📈 Analizar progreso")
            print("7. 🚪 Salir")
            
            choice = input("\n➤ Selecciona una opcion (1-7): ").strip()
            
            if choice == "1":
                self.show_status()
            
            elif choice == "2":
                self.suggest_priority_plan()
            
            elif choice == "3":
                self.reinforce_critical_gesture()
            
            elif choice == "4":
                self.add_new_ecuadorian_gesture()
            
            elif choice == "5":
                self.custom_recording_session()
            
            elif choice == "6":
                self.analyze_progress()
            
            elif choice == "7":
                print("\n Sesion de expansion finalizada!")
                print(" Recuerda entrenar el modelo despues de agregar muchos datos:")
                print("   python train_model.py")
                break
            
            else:
                print(" Opcion invalida. Selecciona 1-7.")

    def reinforce_critical_gesture(self):
        """Refuerza un gesto critico especifico"""
        counts = self.analyze_current_dataset()
        critical = [(g, c) for g, c in counts.items() if c < 50]
        
        if not critical:
            print(" No hay gestos criticos! Todos tienen 50+ muestras.")
            return
        
        print("\n🔴 GESTOS CRITICOS DISPONIBLES:")
        critical_sorted = sorted(critical, key=lambda x: x[1])
        for i, (gesture, count) in enumerate(critical_sorted[:10], 1):
            needed = 80 - count
            print(f"   {i}. {gesture}: {count} muestras (necesita +{needed})")
        
        try:
            selection = int(input(f"\n➤ Selecciona gesto (1-{min(10, len(critical_sorted))}): "))
            if 1 <= selection <= len(critical_sorted):
                gesture, current_count = critical_sorted[selection-1]
                target_new = max(80 - current_count, 20)
                
                print(f"\n Seleccionado: '{gesture}' ({current_count} muestras actuales)")
                print(f" Objetivo: agregar {target_new} muestras nuevas")
                
                self.record_gesture_session(gesture, target_new)
            else:
                print(" Seleccion invalida")
        except ValueError:
            print(" Por favor ingresa un numero valido")

    def add_new_ecuadorian_gesture(self):
        """Agrega un nuevo gesto ecuatoriano"""
        print("\n🇪🇨 VOCABULARIO ECUATORIANO DISPONIBLE:")
        
        categories = list(self.gestures_ecuatorianos.keys())
        for i, category in enumerate(categories, 1):
            count = len(self.gestures_ecuatorianos[category])
            print(f"   {i}. {category}: {count} gestos")
        
        try:
            cat_selection = int(input(f"\n➤ Selecciona categoria (1-{len(categories)}): "))
            if 1 <= cat_selection <= len(categories):
                selected_category = categories[cat_selection-1]
                gestures = self.gestures_ecuatorianos[selected_category]
                
                print(f"\n{selected_category}:")
                for i, gesture in enumerate(gestures[:15], 1):  # Mostrar maximo 15
                    print(f"   {i:2d}. {gesture}")
                
                if len(gestures) > 15:
                    print(f"   ... y {len(gestures)-15} mas")
                
                gesture_name = input(f"\n➤ Escribe el gesto exacto a grabar: ").strip()
                if gesture_name in gestures:
                    print(f" Perfecto! Vamos a grabar '{gesture_name}'")
                    print(" Como es un gesto nuevo, recomendamos empezar con 50-80 muestras")
                    
                    target = int(input("➤ Cuantas muestras quieres grabar? (recomendado: 60): ") or "60")
                    self.record_gesture_session(gesture_name, target)
                else:
                    print(f"⚠️ '{gesture_name}' no esta en la lista, pero puedes grabarlo igual")
                    confirm = input("➤ Continuar con este gesto? (s/n): ").lower().strip()
                    if confirm == 's':
                        target = int(input("➤ Cuantas muestras? (recomendado: 60): ") or "60")
                        self.record_gesture_session(gesture_name, target)
            else:
                print(" Seleccion de categoria invalida")
        except ValueError:
            print(" Por favor ingresa numeros validos")

    def custom_recording_session(self):
        """Sesion de grabacion personalizada"""
        gesture_name = input("\n➤ Nombre del gesto a grabar: ").strip()
        if not gesture_name:
            print(" Nombre de gesto no puede estar vacio")
            return
        
        try:
            target = int(input("➤ Cuantas muestras nuevas? (recomendado: 20-40): ") or "20")
            if target <= 0:
                print(" El numero debe ser positivo")
                return
            
            self.record_gesture_session(gesture_name, target)
        except ValueError:
            print(" Por favor ingresa un numero valido")

    def log_progress(self, gesture, samples_added):
        """Registra el progreso de expansion"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "gesture": gesture,
            "samples_added": samples_added,
            "session_type": "expansion"
        }
        
        log_file = "expansion_progress.json"
        logs = []
        
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = [json.loads(line) for line in f if line.strip()]
            except:
                pass
        
        logs.append(log_entry)
        
        with open(log_file, "w", encoding="utf-8") as f:
            for log in logs:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")

    def analyze_progress(self):
        """Analiza el progreso de expansion"""
        log_file = "expansion_progress.json"
        if not os.path.exists(log_file):
            print("📊 No hay registros de progreso aun")
            return
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f if line.strip()]
            
            if not logs:
                print("📊 No hay registros de progreso")
                return
            
            print("\n📈 ANALISIS DE PROGRESO")
            print("=" * 30)
            
            total_added = sum(log.get("samples_added", 0) for log in logs)
            unique_gestures = len(set(log.get("gesture", "") for log in logs))
            
            print(f"📋 Total de muestras agregadas: {total_added}")
            print(f" Gestos trabajados: {unique_gestures}")
            print(f"📅 Sesiones de grabacion: {len(logs)}")
            
            # Ultimas 5 sesiones
            recent_logs = logs[-5:]
            print(f"\n🕒 ULTIMAS {len(recent_logs)} SESIONES:")
            for log in recent_logs:
                date = log.get("timestamp", "").split("T")[0]
                gesture = log.get("gesture", "")
                samples = log.get("samples_added", 0)
                print(f"   • {date}: {gesture} (+{samples} muestras)")
                
        except Exception as e:
            print(f" Error al analizar progreso: {e}")

def main():
    print("🇪🇨 ASISTENTE DE EXPANSION - LENGUA DE SEÑAS ECUATORIANA")
    print("=" * 60)
    print(" Este asistente te ayudara a expandir tu dataset de forma sistematica")
    print(" Recomendacion: Ten al menos 30 minutos disponibles para una buena sesion")
    
    assistant = LSEDatasetExpansion()
    assistant.interactive_expansion_session()

if __name__ == "__main__":
    main()
