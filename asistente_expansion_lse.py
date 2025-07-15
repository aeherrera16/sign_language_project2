"""
🇪🇨 ASISTENTE INTELIGENTE PARA EXPANSIÓN DE DATASET LSE
Este script te guía paso a paso para expandir tu dataset de lengua de señas ecuatoriana
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
            "🏙️ Geografía": [
                "Costa", "Sierra", "Oriente", "Galápagos",
                "Quito", "Guayaquil", "Cuenca", "Ambato", "Machala",
                "Portoviejo", "Loja", "Riobamba", "Ibarra", "Manta",
                "Mitad_del_Mundo", "Cotopaxi", "Chimborazo", "Baños", "Otavalo"
            ],
            "🍽️ Gastronomía": [
                "encebollado", "fritada", "hornado", "cuy", "locro", "fanesca",
                "bolón", "tigrillo", "corviche", "empanadas_de_viento",
                "humitas", "tamales", "llapingachos", "chicha", "colada_morada"
            ],
            "💬 Expresiones": [
                "chévere", "bacán", "jama", "chuta", "achachay", "atatay",
                "ñaño", "ñaña", "pana", "causa", "longo", "chulla",
                "yapa", "guambra", "taita"
            ],
            "👔 Profesiones": [
                "doctor", "enfermera", "profesor", "ingeniero", "abogado",
                "contador", "arquitecto", "dentista", "agricultor", "pescador",
                "artesano", "comerciante", "chofer", "cocinero", "policía"
            ],
            "🏢 Instituciones": [
                "hospital", "clínica", "banco", "registro_civil", "municipio",
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
            print("❌ No se encontró la carpeta 'data'")
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
        
        print(f"\n🔴 CRÍTICOS (<50): {len(critical)} gestos")
        print(f"⚠️ BAJOS (50-79): {len(low)} gestos")
        print(f"✅ BUENOS (80-99): {len(good)} gestos")
        print(f"🌟 EXCELENTES (100+): {len(excellent)} gestos")
        
        if critical:
            print("\n🚨 GESTOS QUE NECESITAN ATENCIÓN URGENTE:")
            for gesture, count in sorted(critical, key=lambda x: x[1])[:10]:
                needed = 80 - count
                print(f"   • {gesture}: {count} muestras (necesita +{needed})")

    def record_gesture_session(self, gesture_name, target_new_samples=20):
        """Graba una sesión de muestras para un gesto específico"""
        print(f"\n🎬 SESIÓN DE GRABACIÓN: '{gesture_name}'")
        print("=" * 40)
        print(f"🎯 Objetivo: +{target_new_samples} muestras nuevas")
        print("\n📋 PREPARACIÓN:")
        print("   • Asegúrate de tener buena iluminación")
        print("   • Limpia el lente de la cámara")
        print("   • Prepara el gesto mentalmente")
        print("   • Ten agua cerca (puedes tardar un rato)")
        
        print("\n💡 CONSEJOS PARA ESTA SESIÓN:")
        print("   • Varía ligeramente el ángulo de las manos")
        print("   • Cambia la velocidad del gesto (lento/normal/rápido)")
        print("   • Mueve ligeramente la posición del cuerpo")
        print("   • Si tienes personas disponibles, que ellas también graben")
        
        confirm = input("\n➤ ¿Estás listo para grabar? (s/n): ").lower().strip()
        if confirm != 's':
            print("⏸️ Sesión cancelada")
            return False
        
        # Grabación en lotes pequeños para evitar fatiga
        sessions = target_new_samples // 5  # Sesiones de 5 muestras
        remainder = target_new_samples % 5
        
        print(f"\n📹 Dividiremos en {sessions} sesiones de 5 muestras")
        if remainder:
            print(f"   + 1 sesión final de {remainder} muestras")
        
        try:
            for i in range(sessions):
                print(f"\n🎬 Sesión {i+1}/{sessions} (5 muestras)")
                input("   Presiona ENTER cuando estés listo...")
                subprocess.run([sys.executable, "record_dataset.py", gesture_name], check=True)
                
                if i < sessions - 1:  # No preguntar en la última sesión
                    continue_session = input("   ¿Continuar con la siguiente sesión? (s/n): ").lower().strip()
                    if continue_session != 's':
                        print("⏸️ Sesiones restantes canceladas")
                        break
            
            if remainder:
                print(f"\n🎬 Sesión final ({remainder} muestras)")
                input("   Presiona ENTER cuando estés listo...")
                subprocess.run([sys.executable, "record_dataset.py", gesture_name], check=True)
            
            print(f"✅ ¡Sesión completada para '{gesture_name}'!")
            self.log_progress(gesture_name, target_new_samples)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error durante la grabación: {e}")
            return False
        except KeyboardInterrupt:
            print(f"\n⏹️ Grabación interrumpida por el usuario")
            return False

    def suggest_priority_plan(self):
        """Sugiere un plan de prioridades basado en el análisis actual"""
        counts = self.analyze_current_dataset()
        if not counts:
            return
        
        print("\n🎯 PLAN DE PRIORIDADES RECOMENDADO")
        print("=" * 45)
        
        # Prioridad 1: Gestos críticos
        critical = [(g, c) for g, c in counts.items() if c < 50]
        if critical:
            print("\n🔴 PRIORIDAD 1 - GESTOS CRÍTICOS:")
            for i, (gesture, count) in enumerate(sorted(critical, key=lambda x: x[1])[:5], 1):
                needed = max(80 - count, 20)  # Al menos 20 muestras nuevas
                print(f"   {i}. {gesture}: {count} → objetivo 80+ (+{needed} muestras)")
        
        # Prioridad 2: Gestos bajos
        low = [(g, c) for g, c in counts.items() if 50 <= c < 80]
        if low:
            print("\n⚠️ PRIORIDAD 2 - GESTOS BAJOS:")
            for i, (gesture, count) in enumerate(sorted(low, key=lambda x: x[1])[:5], 1):
                needed = 80 - count
                print(f"   {i}. {gesture}: {count} → objetivo 80+ (+{needed} muestras)")
        
        # Prioridad 3: Nuevo vocabulario ecuatoriano
        print("\n🆕 PRIORIDAD 3 - VOCABULARIO ECUATORIANO:")
        print("   Después de completar las prioridades 1 y 2:")
        for category, gestures in list(self.gestures_ecuatorianos.items())[:3]:
            print(f"   • {category}: {len(gestures)} gestos nuevos")

    def interactive_expansion_session(self):
        """Sesión interactiva de expansión del dataset"""
        print("\n🚀 SESIÓN INTERACTIVA DE EXPANSIÓN")
        print("=" * 40)
        
        while True:
            print("\n🎮 OPCIONES:")
            print("1. 📊 Ver estado actual del dataset")
            print("2. 🎯 Ver plan de prioridades recomendado")
            print("3. 🔴 Reforzar gesto crítico")
            print("4. 🆕 Agregar gesto ecuatoriano nuevo")
            print("5. 🎬 Sesión de grabación personalizada")
            print("6. 📈 Analizar progreso")
            print("7. 🚪 Salir")
            
            choice = input("\n➤ Selecciona una opción (1-7): ").strip()
            
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
                print("\n👋 ¡Sesión de expansión finalizada!")
                print("💡 Recuerda entrenar el modelo después de agregar muchos datos:")
                print("   python train_model.py")
                break
            
            else:
                print("❌ Opción inválida. Selecciona 1-7.")

    def reinforce_critical_gesture(self):
        """Refuerza un gesto crítico específico"""
        counts = self.analyze_current_dataset()
        critical = [(g, c) for g, c in counts.items() if c < 50]
        
        if not critical:
            print("✅ ¡No hay gestos críticos! Todos tienen 50+ muestras.")
            return
        
        print("\n🔴 GESTOS CRÍTICOS DISPONIBLES:")
        critical_sorted = sorted(critical, key=lambda x: x[1])
        for i, (gesture, count) in enumerate(critical_sorted[:10], 1):
            needed = 80 - count
            print(f"   {i}. {gesture}: {count} muestras (necesita +{needed})")
        
        try:
            selection = int(input(f"\n➤ Selecciona gesto (1-{min(10, len(critical_sorted))}): "))
            if 1 <= selection <= len(critical_sorted):
                gesture, current_count = critical_sorted[selection-1]
                target_new = max(80 - current_count, 20)
                
                print(f"\n✅ Seleccionado: '{gesture}' ({current_count} muestras actuales)")
                print(f"🎯 Objetivo: agregar {target_new} muestras nuevas")
                
                self.record_gesture_session(gesture, target_new)
            else:
                print("❌ Selección inválida")
        except ValueError:
            print("❌ Por favor ingresa un número válido")

    def add_new_ecuadorian_gesture(self):
        """Agrega un nuevo gesto ecuatoriano"""
        print("\n🇪🇨 VOCABULARIO ECUATORIANO DISPONIBLE:")
        
        categories = list(self.gestures_ecuatorianos.keys())
        for i, category in enumerate(categories, 1):
            count = len(self.gestures_ecuatorianos[category])
            print(f"   {i}. {category}: {count} gestos")
        
        try:
            cat_selection = int(input(f"\n➤ Selecciona categoría (1-{len(categories)}): "))
            if 1 <= cat_selection <= len(categories):
                selected_category = categories[cat_selection-1]
                gestures = self.gestures_ecuatorianos[selected_category]
                
                print(f"\n{selected_category}:")
                for i, gesture in enumerate(gestures[:15], 1):  # Mostrar máximo 15
                    print(f"   {i:2d}. {gesture}")
                
                if len(gestures) > 15:
                    print(f"   ... y {len(gestures)-15} más")
                
                gesture_name = input(f"\n➤ Escribe el gesto exacto a grabar: ").strip()
                if gesture_name in gestures:
                    print(f"✅ Perfecto! Vamos a grabar '{gesture_name}'")
                    print("💡 Como es un gesto nuevo, recomendamos empezar con 50-80 muestras")
                    
                    target = int(input("➤ ¿Cuántas muestras quieres grabar? (recomendado: 60): ") or "60")
                    self.record_gesture_session(gesture_name, target)
                else:
                    print(f"⚠️ '{gesture_name}' no está en la lista, pero puedes grabarlo igual")
                    confirm = input("➤ ¿Continuar con este gesto? (s/n): ").lower().strip()
                    if confirm == 's':
                        target = int(input("➤ ¿Cuántas muestras? (recomendado: 60): ") or "60")
                        self.record_gesture_session(gesture_name, target)
            else:
                print("❌ Selección de categoría inválida")
        except ValueError:
            print("❌ Por favor ingresa números válidos")

    def custom_recording_session(self):
        """Sesión de grabación personalizada"""
        gesture_name = input("\n➤ Nombre del gesto a grabar: ").strip()
        if not gesture_name:
            print("❌ Nombre de gesto no puede estar vacío")
            return
        
        try:
            target = int(input("➤ ¿Cuántas muestras nuevas? (recomendado: 20-40): ") or "20")
            if target <= 0:
                print("❌ El número debe ser positivo")
                return
            
            self.record_gesture_session(gesture_name, target)
        except ValueError:
            print("❌ Por favor ingresa un número válido")

    def log_progress(self, gesture, samples_added):
        """Registra el progreso de expansión"""
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
        """Analiza el progreso de expansión"""
        log_file = "expansion_progress.json"
        if not os.path.exists(log_file):
            print("📊 No hay registros de progreso aún")
            return
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f if line.strip()]
            
            if not logs:
                print("📊 No hay registros de progreso")
                return
            
            print("\n📈 ANÁLISIS DE PROGRESO")
            print("=" * 30)
            
            total_added = sum(log.get("samples_added", 0) for log in logs)
            unique_gestures = len(set(log.get("gesture", "") for log in logs))
            
            print(f"📋 Total de muestras agregadas: {total_added}")
            print(f"🎯 Gestos trabajados: {unique_gestures}")
            print(f"📅 Sesiones de grabación: {len(logs)}")
            
            # Últimas 5 sesiones
            recent_logs = logs[-5:]
            print(f"\n🕒 ÚLTIMAS {len(recent_logs)} SESIONES:")
            for log in recent_logs:
                date = log.get("timestamp", "").split("T")[0]
                gesture = log.get("gesture", "")
                samples = log.get("samples_added", 0)
                print(f"   • {date}: {gesture} (+{samples} muestras)")
                
        except Exception as e:
            print(f"❌ Error al analizar progreso: {e}")

def main():
    print("🇪🇨 ASISTENTE DE EXPANSIÓN - LENGUA DE SEÑAS ECUATORIANA")
    print("=" * 60)
    print("🎯 Este asistente te ayudará a expandir tu dataset de forma sistemática")
    print("💡 Recomendación: Ten al menos 30 minutos disponibles para una buena sesión")
    
    assistant = LSEDatasetExpansion()
    assistant.interactive_expansion_session()

if __name__ == "__main__":
    main()
