#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSE ECUADOR - INTERFAZ ELEGANTE Y FUNCIONAL
Sistema moderno para reconocimiento de lengua de señas
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import subprocess
import threading
from datetime import datetime
import json

# Configurar variables de entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import cv2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

class ModernLSEInterface:
    """Interfaz moderna y elegante para LSE Ecuador"""
    
    def __init__(self, root):
        self.root = root
        self.running_processes = []
        self.setup_window()
        self.setup_styles()
        self.create_interface()
        self.check_system_status()
        
    def setup_window(self):
        """Configurar ventana principal"""
        self.root.title("🇪🇨 LSE Ecuador • Sistema Avanzado de Reconocimiento")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg='#1a1a2e')
        
        # Centrar ventana
        self.center_window()
        
        # Configurar icono y estilo
        try:
            self.root.iconbitmap(default='')  # Opcional: agregar icono
        except:
            pass
    
    def center_window(self):
        """Centrar ventana en pantalla"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 1200) // 2
        y = (self.root.winfo_screenheight() - 800) // 2
        self.root.geometry(f"1200x800+{x}+{y}")
    
    def setup_styles(self):
        """Configurar estilos modernos"""
        self.colors = {
            'bg_primary': '#1a1a2e',      # Azul muy oscuro
            'bg_secondary': '#16213e',     # Azul oscuro
            'bg_card': '#0f3460',         # Azul medio
            'accent': '#0f4c75',          # Azul fuerte
            'primary': '#3282b8',         # Azul brillante
            'success': '#00d4aa',         # Verde
            'warning': '#ffa726',         # Naranja
            'danger': '#ef5350',          # Rojo
            'text_primary': '#ffffff',    # Blanco
            'text_secondary': '#b0bec5',  # Gris claro
            'text_muted': '#78909c'       # Gris medio
        }
        
        # Configurar ttk styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Estilo para botones
        self.style.configure(
            'Modern.TButton',
            background=self.colors['primary'],
            foreground=self.colors['text_primary'],
            borderwidth=0,
            focuscolor='none',
            padding=(20, 15)
        )
        
        self.style.map(
            'Modern.TButton',
            background=[('active', self.colors['accent'])],
            foreground=[('active', self.colors['text_primary'])]
        )
    
    def create_interface(self):
        """Crear interfaz moderna"""
        
        # Header elegante
        self.create_modern_header()
        
        # Contenedor principal
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Panel izquierdo - Funciones principales
        left_panel = self.create_left_panel(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Panel derecho - Estado y monitoreo
        right_panel = self.create_right_panel(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Footer con información
        self.create_footer()
    
    def create_modern_header(self):
        """Crear header moderno con gradiente simulado"""
        header = Canvas(
            self.root,
            height=100,
            bg=self.colors['bg_secondary'],
            highlightthickness=0
        )
        header.pack(fill=tk.X)
        
        # Simular gradiente con rectángulos
        for i in range(100):
            alpha = i / 100
            color_val = int(15 + alpha * 35)  # Del 15 al 50 en hex
            color = f"#{color_val:02x}{color_val+20:02x}{color_val+40:02x}"
            header.create_rectangle(0, i, 1200, i+1, fill=color, outline=color)
        
        # Título principal
        header.create_text(
            600, 35,
            text="🇪🇨 LSE ECUADOR",
            font=('Segoe UI', 28, 'bold'),
            fill=self.colors['text_primary']
        )
        
        # Subtítulo
        header.create_text(
            600, 65,
            text="Sistema Avanzado de Reconocimiento de Lengua de Señas",
            font=('Segoe UI', 12),
            fill=self.colors['text_secondary']
        )
        
        # Indicador de estado
        self.status_indicator = header.create_oval(1150, 15, 1170, 35, fill=self.colors['success'], outline='')
        header.create_text(1100, 25, text="Sistema Activo", font=('Segoe UI', 10), fill=self.colors['text_secondary'])
    
    def create_left_panel(self, parent):
        """Panel izquierdo con funciones principales"""
        panel = tk.Frame(parent, bg=self.colors['bg_primary'])
        
        # Título del panel
        title_frame = tk.Frame(panel, bg=self.colors['bg_card'], height=50)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="🛠️ Funciones Principales",
            font=('Segoe UI', 16, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_card']
        ).pack(pady=15)
        
        # Grid de botones principales
        buttons_frame = tk.Frame(panel, bg=self.colors['bg_primary'])
        buttons_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid
        for i in range(3):
            buttons_frame.grid_rowconfigure(i, weight=1)
        for i in range(2):
            buttons_frame.grid_columnconfigure(i, weight=1)
        
        # Botones principales con íconos y descripciones
        self.create_function_card(
            buttons_frame, 0, 0,
            "📹", "Grabar Dataset",
            "Captura nuevas señas LSE",
            self.colors['primary'],
            self.grabar_dataset
        )
        
        self.create_function_card(
            buttons_frame, 0, 1,
            "🧠", "Entrenar Modelo", 
            "Entrenar IA con nuevos datos",
            self.colors['warning'],
            self.entrenar_modelo
        )
        
        self.create_function_card(
            buttons_frame, 1, 0,
            "🎯", "Reconocimiento",
            "Detectar señas en tiempo real",
            self.colors['success'],
            self.reconocimiento_tiempo_real
        )
        
        self.create_function_card(
            buttons_frame, 1, 1,
            "✅", "Verificar Sistema",
            "Comprobar estado completo",
            self.colors['accent'],
            self.verificar_sistema
        )
        
        self.create_function_card(
            buttons_frame, 2, 0,
            "🗑️", "Limpiar Datos",
            "Resetear para nuevas señas",
            self.colors['danger'],
            self.limpiar_datos
        )
        
        self.create_function_card(
            buttons_frame, 2, 1,
            "🔍", "Verificar Señas",
            "Validar señas LSE Ecuador",
            self.colors['bg_card'],
            self.verificar_senas
        )
        
        return panel
    
    def create_function_card(self, parent, row, col, icon, title, desc, color, command):
        """Crear tarjeta de función moderna"""
        card = tk.Frame(
            parent,
            bg=color,
            relief='flat',
            bd=0
        )
        card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        
        # Efecto hover
        def on_enter(e):
            card.configure(relief='raised', bd=2)
        
        def on_leave(e):
            card.configure(relief='flat', bd=0)
        
        def on_click(e):
            command()
        
        card.bind('<Enter>', on_enter)
        card.bind('<Leave>', on_leave)
        card.bind('<Button-1>', on_click)
        
        # Contenido de la tarjeta
        content_frame = tk.Frame(card, bg=color)
        content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Ícono
        icon_label = tk.Label(
            content_frame,
            text=icon,
            font=('Segoe UI Emoji', 32),
            fg=self.colors['text_primary'],
            bg=color
        )
        icon_label.pack(pady=(0, 10))
        icon_label.bind('<Button-1>', on_click)
        
        # Título
        title_label = tk.Label(
            content_frame,
            text=title,
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=color
        )
        title_label.pack()
        title_label.bind('<Button-1>', on_click)
        
        # Descripción
        desc_label = tk.Label(
            content_frame,
            text=desc,
            font=('Segoe UI', 10),
            fg=self.colors['text_secondary'],
            bg=color,
            wraplength=150
        )
        desc_label.pack(pady=(5, 0))
        desc_label.bind('<Button-1>', on_click)
        
        # Hacer todos los elementos clickeables
        for widget in [card, content_frame, icon_label, title_label, desc_label]:
            widget.bind('<Enter>', on_enter)
            widget.bind('<Leave>', on_leave)
            widget.bind('<Button-1>', on_click)
    
    def create_right_panel(self, parent):
        """Panel derecho con estado y monitoreo"""
        panel = tk.Frame(parent, bg=self.colors['bg_primary'])
        
        # Estado del sistema
        status_frame = tk.LabelFrame(
            panel,
            text="📊 Estado del Sistema",
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_card'],
            bd=2,
            relief='groove'
        )
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.status_text = tk.Text(
            status_frame,
            height=10,
            font=('Consolas', 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'],
            selectbackground=self.colors['accent'],
            wrap=tk.WORD
        )
        self.status_text.pack(fill=tk.X, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
        
        # Información de gestos
        gestos_frame = tk.LabelFrame(
            panel,
            text="🖐️ Gestos LSE Ecuador",
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_card'],
            bd=2,
            relief='groove'
        )
        gestos_frame.pack(fill=tk.BOTH, expand=True)
        
        gestos_info = [
            ("🖐️", "HOLA", "Mano abierta hacia adelante"),
            ("👋", "ADIOS", "Movimiento lateral (izq-der)"),
            ("🙏", "GRACIAS", "Mano hacia el pecho"),
            ("👍", "SÍ", "Puño vertical (arriba-abajo)"),
            ("👎", "NO", "Dedo horizontal (izq-der)")
        ]
        
        for i, (emoji, gesto, desc) in enumerate(gestos_info):
            gesto_frame = tk.Frame(gestos_frame, bg=self.colors['bg_card'])
            gesto_frame.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(
                gesto_frame,
                text=f"{emoji} {gesto}",
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']
            ).pack(anchor='w')
            
            tk.Label(
                gesto_frame,
                text=desc,
                font=('Segoe UI', 10),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_card']
            ).pack(anchor='w', padx=(20, 0))
        
        return panel
    
    def create_footer(self):
        """Crear footer informativo"""
        footer = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=40)
        footer.pack(fill=tk.X)
        footer.pack_propagate(False)
        
        # Información del sistema
        tk.Label(
            footer,
            text=f"🐍 Python {sys.version.split()[0]} | 🎥 Cámara: {'✅ Disponible' if CAMERA_AVAILABLE else '❌ No disponible'} | 📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            font=('Segoe UI', 9),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_secondary']
        ).pack(pady=10)
    
    def log_message(self, message, level='info'):
        """Agregar mensaje al log con colores"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Colores según nivel
        colors = {
            'info': self.colors['text_primary'],
            'success': self.colors['success'],
            'warning': self.colors['warning'], 
            'error': self.colors['danger']
        }
        
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def check_system_status(self):
        """Verificar estado inicial del sistema"""
        self.log_message("🚀 LSE Ecuador iniciado correctamente", 'success')
        
        # Verificar archivos importantes
        files_to_check = [
            ("scripts/core/record_dataset.py", "Grabador de dataset"),
            ("scripts/core/train_model.py", "Entrenador de modelo"),
            ("scripts/recognition/real_time_translate.py", "Reconocimiento en tiempo real")
        ]
        
        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                self.log_message(f"✅ {description} disponible", 'success')
            else:
                self.log_message(f"❌ {description} no encontrado", 'error')
        
        # Verificar modelo entrenado
        if os.path.exists("model/gesture_model.h5"):
            self.log_message("🧠 Modelo entrenado disponible", 'success')
        else:
            self.log_message("⚠️ Modelo no encontrado - necesita entrenamiento", 'warning')
        
        self.log_message("💡 Selecciona una función para comenzar", 'info')
    
    def run_process_safely(self, script_path, description, window_mode=False):
        """Ejecutar proceso de forma segura"""
        def run():
            try:
                self.log_message(f"🚀 Iniciando: {description}", 'info')
                
                if not os.path.exists(script_path):
                    self.log_message(f"❌ Error: No se encuentra {script_path}", 'error')
                    return
                
                if window_mode:
                    # Abrir en ventana separada
                    process = subprocess.Popen(
                        [sys.executable, script_path],
                        cwd=os.getcwd()
                    )
                    self.running_processes.append(process)
                    self.log_message(f"✅ {description} iniciado en ventana separada", 'success')
                else:
                    # Ejecutar y capturar salida
                    result = subprocess.run(
                        [sys.executable, script_path],
                        capture_output=True,
                        text=True,
                        cwd=os.getcwd()
                    )
                    
                    if result.returncode == 0:
                        self.log_message(f"✅ {description} completado exitosamente", 'success')
                        if result.stdout:
                            # Mostrar solo las últimas líneas importantes
                            lines = result.stdout.strip().split('\n')
                            for line in lines[-5:]:  # Últimas 5 líneas
                                if line.strip():
                                    self.log_message(f"📋 {line.strip()[:80]}", 'info')
                    else:
                        self.log_message(f"❌ Error en {description}", 'error')
                        if result.stderr:
                            error_lines = result.stderr.strip().split('\n')
                            for line in error_lines[:3]:  # Primeras 3 líneas del error
                                if line.strip():
                                    self.log_message(f"🚫 {line.strip()[:80]}", 'error')
                        
            except Exception as e:
                self.log_message(f"❌ Excepción en {description}: {str(e)[:80]}", 'error')
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    # Funciones de los botones
    def grabar_dataset(self):
        """Grabar dataset de señas"""
        self.run_process_safely("scripts/core/record_dataset.py", "Grabación de Dataset")
    
    def entrenar_modelo(self):
        """Entrenar modelo de IA"""
        self.run_process_safely("scripts/core/train_model.py", "Entrenamiento del Modelo")
    
    def reconocimiento_tiempo_real(self):
        """Reconocimiento en tiempo real"""
        self.run_process_safely("scripts/recognition/real_time_translate.py", "Reconocimiento en Tiempo Real", window_mode=True)
    
    def verificar_sistema(self):
        """Verificar estado del sistema"""
        self.run_process_safely("verificacion_sistema_completo.py", "Verificación del Sistema")
    
    def limpiar_datos(self):
        """Limpiar datos para nuevas grabaciones"""
        respuesta = messagebox.askyesnocancel(
            "🗑️ Confirmar Limpieza",
            "¿Estás seguro de eliminar todos los datos actuales?\n\n"
            "✅ Se creará un backup automático\n"
            "❌ Esta acción eliminará el modelo actual\n\n"
            "¿Continuar?"
        )
        
        if respuesta:
            self.run_process_safely("limpiar_data_para_nuevas_grabaciones.py", "Limpieza de Datos")
    
    def verificar_senas(self):
        """Verificar señas LSE"""
        self.run_process_safely("verificador_senas_lse.py", "Verificador de Señas LSE")
    
    def on_closing(self):
        """Manejar cierre de aplicación"""
        if self.running_processes:
            if messagebox.askokcancel("Cerrar", "¿Cerrar la aplicación? Se terminarán todos los procesos activos."):
                for process in self.running_processes:
                    try:
                        process.terminate()
                    except:
                        pass
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    """Función principal"""
    if sys.version_info < (3, 7):
        print("❌ Error: Se requiere Python 3.7 o superior")
        return
    
    if not os.path.exists("main_interface.py"):
        print("❌ Error: Ejecuta desde el directorio del proyecto")
        return
    
    try:
        root = tk.Tk()
        app = ModernLSEInterface(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error iniciando interfaz: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
