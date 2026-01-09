#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🤟 LANZADOR PRINCIPAL - TRADUCTOR LSE
═══════════════════════════════════════════════════════════════════════════════

Menú principal para acceder a las funciones de la aplicación.

═══════════════════════════════════════════════════════════════════════════════
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
from pathlib import Path

# Configuración
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / "model"
DATA_DIR = SCRIPT_DIR / "data"

COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#0f3460',
    'accent': '#e94560',
    'text': '#ffffff',
    'text_secondary': '#a0a0a0',
    'success': '#00d26a',
    'warning': '#ffc107',
}


class LanzadorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traductor LSE")
        self.root.geometry("500x600")
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.resizable(False, False)
        
        self._create_ui()
        self._update_status()
    
    def _create_ui(self):
        # Título
        title_frame = tk.Frame(self.root, bg=COLORS['bg_dark'], pady=30)
        title_frame.pack(fill=tk.X)
        
        tk.Label(title_frame, text="🤟", font=('Helvetica', 60),
                bg=COLORS['bg_dark'], fg=COLORS['accent']).pack()
        
        tk.Label(title_frame, text="Traductor LSE", font=('Helvetica', 28, 'bold'),
                bg=COLORS['bg_dark'], fg=COLORS['text']).pack()
        
        tk.Label(title_frame, text="Lengua de Señas Ecuatoriana", font=('Helvetica', 12),
                bg=COLORS['bg_dark'], fg=COLORS['text_secondary']).pack()
        
        # Botones
        buttons_frame = tk.Frame(self.root, bg=COLORS['bg_dark'], padx=50)
        buttons_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Botón Traductor - Frame para simular botón con mejor compatibilidad
        traductor_frame = tk.Frame(buttons_frame, bg=COLORS['success'], padx=3, pady=3)
        traductor_frame.pack(fill=tk.X, pady=10)
        
        traductor_btn = tk.Label(
            traductor_frame,
            text="🎯 INICIAR TRADUCTOR",
            font=('Helvetica', 16, 'bold'),
            bg=COLORS['success'],
            fg='white',
            pady=20,
            cursor='hand2'
        )
        traductor_btn.pack(fill=tk.X)
        traductor_btn.bind('<Button-1>', lambda e: self.abrir_traductor())
        traductor_btn.bind('<Enter>', lambda e: traductor_btn.config(bg='#00b85c'))
        traductor_btn.bind('<Leave>', lambda e: traductor_btn.config(bg=COLORS['success']))
        traductor_frame.bind('<Button-1>', lambda e: self.abrir_traductor())
        
        tk.Label(buttons_frame, text="Traduce señas en tiempo real",
                font=('Helvetica', 10), bg=COLORS['bg_dark'],
                fg=COLORS['text_secondary']).pack()
        
        # Botón Entrenador - Frame para simular botón con mejor compatibilidad
        entrenar_frame = tk.Frame(buttons_frame, bg=COLORS['accent'], padx=3, pady=3)
        entrenar_frame.pack(fill=tk.X, pady=(30, 10))
        
        entrenar_btn = tk.Label(
            entrenar_frame,
            text="🎓 ENTRENAR MODELO",
            font=('Helvetica', 16, 'bold'),
            bg=COLORS['accent'],
            fg='white',
            pady=20,
            cursor='hand2'
        )
        entrenar_btn.pack(fill=tk.X)
        entrenar_btn.bind('<Button-1>', lambda e: self.abrir_entrenador())
        entrenar_btn.bind('<Enter>', lambda e: entrenar_btn.config(bg='#ff6b7a'))
        entrenar_btn.bind('<Leave>', lambda e: entrenar_btn.config(bg=COLORS['accent']))
        entrenar_frame.bind('<Button-1>', lambda e: self.abrir_entrenador())
        
        tk.Label(buttons_frame, text="Captura y entrena nuevas señas",
                font=('Helvetica', 10), bg=COLORS['bg_dark'],
                fg=COLORS['text_secondary']).pack()
        
        # Estado
        status_frame = tk.Frame(self.root, bg=COLORS['bg_medium'], padx=20, pady=15)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Label(status_frame, text="ESTADO DEL MODELO", font=('Helvetica', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text']).pack(anchor=tk.W)
        
        self.status_label = tk.Label(status_frame, text="Verificando...",
                                    font=('Helvetica', 11), bg=COLORS['bg_medium'],
                                    fg=COLORS['text_secondary'], justify=tk.LEFT)
        self.status_label.pack(anchor=tk.W, pady=5)
        
        # Versión
        tk.Label(self.root, text="v1.0.0 | Para Raspberry Pi",
                font=('Helvetica', 9), bg=COLORS['bg_dark'],
                fg=COLORS['text_secondary']).pack(pady=10)
    
    def _update_status(self):
        """Actualiza el estado del modelo"""
        model_exists = (MODEL_DIR / "best_model.h5").exists() or (MODEL_DIR / "model.tflite").exists()
        labels_exists = (MODEL_DIR / "labels.pkl").exists()
        
        if model_exists and labels_exists:
            import pickle
            with open(MODEL_DIR / "labels.pkl", 'rb') as f:
                labels = pickle.load(f)
            
            self.status_label.config(
                text=f"✅ Modelo entrenado: {len(labels)} señas\n   ({', '.join(labels)})",
                fg=COLORS['success']
            )
        else:
            # Verificar datos
            data_file = DATA_DIR / "training_data.pkl"
            if data_file.exists():
                import pickle
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                total = sum(len(v) for v in data.values())
                self.status_label.config(
                    text=f"⚠️ Datos sin entrenar: {len(data)} señas, {total} muestras\n   Abre el Entrenador para crear el modelo",
                    fg=COLORS['warning']
                )
            else:
                self.status_label.config(
                    text="❌ No hay modelo\n   Usa el Entrenador para capturar señas y entrenar",
                    fg=COLORS['accent']
                )
    
    def abrir_traductor(self):
        """Abre el traductor"""
        model_exists = (MODEL_DIR / "labels.pkl").exists()
        if not model_exists:
            messagebox.showwarning(
                "Sin Modelo",
                "No hay un modelo entrenado.\n\nPrimero usa el Entrenador para capturar algunas señas y entrenar el modelo."
            )
            return
        
        self.root.withdraw()
        traductor_path = SCRIPT_DIR / "app" / "traductor_lse.py"
        subprocess.run([sys.executable, str(traductor_path)])
        self.root.deiconify()
        self._update_status()
    
    def abrir_entrenador(self):
        """Abre el entrenador"""
        self.root.withdraw()
        entrenador_path = SCRIPT_DIR / "trainer" / "entrenar_modelo.py"
        subprocess.run([sys.executable, str(entrenador_path)])
        self.root.deiconify()
        self._update_status()
    
    def run(self):
        self.root.mainloop()


def main():
    # Crear directorios necesarios
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    app = LanzadorApp()
    app.run()


if __name__ == "__main__":
    main()
