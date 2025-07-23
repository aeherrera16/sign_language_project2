#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSE ECUADOR - INTERFAZ PRINCIPAL SIMPLIFICADA
Sistema funcional para grabación, entrenamiento y reconocimiento
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
from datetime import datetime

# Configurar variables de entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class LSEEcuadorInterface:
    """Interfaz principal simplificada para LSE Ecuador"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_interface()
        
    def setup_window(self):
        """Configurar ventana principal"""
        self.root.title("🇪🇨 LSE Ecuador - Sistema de Señas")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Centrar ventana
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_reqwidth()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_reqheight()) // 2
        self.root.geometry(f"+{x}+{y}")
        
    def create_interface(self):
        """Crear interfaz principal"""
        
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🇪🇨 LSE Ecuador",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Sistema de Reconocimiento de Lengua de Señas",
            font=('Arial', 12),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        subtitle_label.pack()
        
        # Contenido principal
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Crear botones principales
        self.create_main_buttons(main_frame)
        
        # Panel de estado
        self.create_status_panel(main_frame)
        
    def create_main_buttons(self, parent):
        """Crear botones principales"""
        
        buttons_frame = tk.Frame(parent, bg='#ecf0f1')
        buttons_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        
        # Botón Grabar Dataset
        record_btn = tk.Button(
            buttons_frame,
            text="📹 Grabar Dataset",
            font=('Arial', 14, 'bold'),
            fg='white',
            bg='#3498db',
            activebackground='#2980b9',
            relief='flat',
            pady=15,
            command=self.grabar_dataset
        )
        record_btn.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        
        # Botón Entrenar Modelo
        train_btn = tk.Button(
            buttons_frame,
            text="🧠 Entrenar Modelo",
            font=('Arial', 14, 'bold'),
            fg='white',
            bg='#e74c3c',
            activebackground='#c0392b',
            relief='flat',
            pady=15,
            command=self.entrenar_modelo
        )
        train_btn.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
        
        # Botón Reconocimiento en Tiempo Real
        recognize_btn = tk.Button(
            buttons_frame,
            text="🎯 Reconocimiento en Tiempo Real",
            font=('Arial', 14, 'bold'),
            fg='white',
            bg='#2ecc71',
            activebackground='#27ae60',
            relief='flat',
            pady=15,
            command=self.reconocimiento_tiempo_real
        )
        recognize_btn.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        
        # Botones adicionales
        verify_btn = tk.Button(
            buttons_frame,
            text="✅ Verificar Sistema",
            font=('Arial', 12),
            fg='white',
            bg='#9b59b6',
            activebackground='#8e44ad',
            relief='flat',
            pady=10,
            command=self.verificar_sistema
        )
        verify_btn.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        
        clean_btn = tk.Button(
            buttons_frame,
            text="🗑️ Limpiar Datos",
            font=('Arial', 12),
            fg='white',
            bg='#f39c12',
            activebackground='#e67e22',
            relief='flat',
            pady=10,
            command=self.limpiar_datos
        )
        clean_btn.grid(row=2, column=1, padx=10, pady=10, sticky='ew')
        
        # Información de gestos
        info_frame = tk.LabelFrame(
            buttons_frame,
            text="📋 Gestos LSE Ecuador",
            font=('Arial', 11, 'bold'),
            fg='#2c3e50',
            bg='#ecf0f1'
        )
        info_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=20, sticky='ew')
        
        gestos_text = """
🖐️ HOLA: Mano abierta hacia adelante
👋 ADIOS: Mano lateral (izquierda-derecha)
🙏 GRACIAS: Mano al pecho
👍 SÍ: Puño vertical (arriba-abajo)
👎 NO: Dedo horizontal (izquierda-derecha)
        """
        
        gestos_label = tk.Label(
            info_frame,
            text=gestos_text,
            font=('Arial', 10),
            fg='#2c3e50',
            bg='#ecf0f1',
            justify=tk.LEFT
        )
        gestos_label.pack(pady=10)
        
    def create_status_panel(self, parent):
        """Crear panel de estado"""
        
        status_frame = tk.LabelFrame(
            parent,
            text="📊 Estado del Sistema",
            font=('Arial', 11, 'bold'),
            fg='#2c3e50',
            bg='#ecf0f1'
        )
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_text = tk.Text(
            status_frame,
            height=8,
            font=('Consolas', 9),
            bg='#2c3e50',
            fg='#ecf0f1',
            insertbackground='white'
        )
        self.status_text.pack(fill=tk.X, padx=10, pady=10)
        
        # Scrollbar para el texto
        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
        
        # Mensaje inicial
        self.log_message("✅ LSE Ecuador iniciado correctamente")
        self.log_message("💡 Selecciona una función para comenzar")
        
    def log_message(self, message):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def run_script_in_thread(self, script_path, description):
        """Ejecutar script en hilo separado"""
        def run():
            try:
                self.log_message(f"🚀 Iniciando: {description}")
                
                # Verificar si el script existe
                if not os.path.exists(script_path):
                    self.log_message(f"❌ Error: No se encuentra {script_path}")
                    return
                
                # Ejecutar script
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd()
                )
                
                if result.returncode == 0:
                    self.log_message(f"✅ {description} completado exitosamente")
                    if result.stdout:
                        self.log_message(f"📄 Salida: {result.stdout[:200]}")
                else:
                    self.log_message(f"❌ Error en {description}")
                    if result.stderr:
                        self.log_message(f"🚫 Error: {result.stderr[:200]}")
                        
            except Exception as e:
                self.log_message(f"❌ Excepción: {str(e)}")
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
        
    def grabar_dataset(self):
        """Función para grabar dataset"""
        script_path = "scripts/core/record_dataset.py"
        self.run_script_in_thread(script_path, "Grabación de Dataset")
        
    def entrenar_modelo(self):
        """Función para entrenar modelo"""
        script_path = "scripts/core/train_model.py"
        self.run_script_in_thread(script_path, "Entrenamiento del Modelo")
        
    def reconocimiento_tiempo_real(self):
        """Función para reconocimiento en tiempo real"""
        script_path = "scripts/recognition/real_time_translate.py"
        
        # Ejecutar en ventana separada para el reconocimiento
        try:
            self.log_message("🎯 Iniciando reconocimiento en tiempo real...")
            subprocess.Popen([sys.executable, script_path])
            self.log_message("✅ Reconocimiento iniciado en ventana separada")
        except Exception as e:
            self.log_message(f"❌ Error: {str(e)}")
            
    def verificar_sistema(self):
        """Verificar estado del sistema"""
        script_path = "verificacion_sistema_completo.py"
        self.run_script_in_thread(script_path, "Verificación del Sistema")
        
    def limpiar_datos(self):
        """Limpiar datos para nuevas grabaciones"""
        respuesta = messagebox.askyesno(
            "Confirmar Limpieza",
            "¿Estás seguro de eliminar todos los datos actuales?\n"
            "Se creará un backup automático."
        )
        
        if respuesta:
            script_path = "limpiar_data_para_nuevas_grabaciones.py"
            self.run_script_in_thread(script_path, "Limpieza de Datos")

def main():
    """Función principal"""
    # Verificar Python
    if sys.version_info < (3, 7):
        print("❌ Error: Se requiere Python 3.7 o superior")
        return
    
    # Verificar directorio de trabajo
    if not os.path.exists("main_interface.py"):
        print("❌ Error: Ejecuta desde el directorio del proyecto")
        return
        
    # Crear y ejecutar interfaz
    try:
        root = tk.Tk()
        app = LSEEcuadorInterface(root)
        
        # Manejar cierre de ventana
        def on_closing():
            root.quit()
            root.destroy()
            
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error iniciando interfaz: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
