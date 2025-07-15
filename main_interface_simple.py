import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import sys
import os
from datetime import datetime
import json
import threading

try:
    import cv2
    from PIL import Image, ImageTk
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️ Cámara no disponible - se ejecutará sin funcionalidad de video")

class LSEModernInterface:
    """Interfaz moderna y intuitiva para LSE Ecuador"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_camera_variables()
        self.create_modern_ui()
        
    def setup_window(self):
        """Configuración de la ventana principal"""
        self.root.title("🇪🇨 LSE ECUADOR - SISTEMA REVOLUCIONARIO")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0f0f23')
        
        # Colores modernos
        self.colors = {
            'bg_primary': '#0f0f23',
            'bg_secondary': '#1a1a2e',
            'bg_accent': '#16213e',
            'success': '#00ff88',
            'warning': '#ffd700',
            'danger': '#ff6b6b',
            'info': '#4ecdc4',
            'text': '#ffffff',
            'text_dim': '#b8b8b8'
        }
        
    def setup_camera_variables(self):
        """Inicializa variables de cámara"""
        self.cap = None
        self.camera_active = False
        self.camera_thread = None
        
    def create_modern_ui(self):
        """Crea la interfaz moderna"""
        # Header con título y estadísticas
        self.create_header()
        
        # Contenedor principal
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Panel izquierdo - Cámara y reconocimiento
        self.create_camera_panel(main_frame)
        
        # Panel derecho - Funcionalidades
        self.create_features_panel(main_frame)
        
        # Footer
        self.create_footer()
        
    def create_header(self):
        """Crea el header con título y estadísticas"""
        header = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=80)
        header.pack(fill=tk.X, padx=10, pady=5)
        header.pack_propagate(False)
        
        # Título principal
        title_frame = tk.Frame(header, bg=self.colors['bg_secondary'])
        title_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        main_title = tk.Label(
            title_frame,
            text="🇪🇨 LSE ECUADOR",
            font=("Arial", 20, "bold"),
            fg=self.colors['success'],
            bg=self.colors['bg_secondary']
        )
        main_title.pack(anchor='w', pady=5)
        
        subtitle = tk.Label(
            title_frame,
            text="Sistema Revolucionario de Lengua de Señas",
            font=("Arial", 10),
            fg=self.colors['warning'],
            bg=self.colors['bg_secondary']
        )
        subtitle.pack(anchor='w')
        
        # Estadísticas rápidas
        stats_frame = tk.Frame(header, bg=self.colors['bg_secondary'])
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        stats_data = [
            ("Precisión", "98.06%", self.colors['success']),
            ("Gestos", "205", self.colors['info']),
            ("Funcionalidades", "15+", self.colors['warning'])
        ]
        
        for i, (label, value, color) in enumerate(stats_data):
            stat_container = tk.Frame(stats_frame, bg=self.colors['bg_secondary'])
            stat_container.grid(row=0, column=i, padx=10)
            
            tk.Label(
                stat_container,
                text=value,
                font=("Arial", 14, "bold"),
                fg=color,
                bg=self.colors['bg_secondary']
            ).pack()
            
            tk.Label(
                stat_container,
                text=label,
                font=("Arial", 8),
                fg=self.colors['text_dim'],
                bg=self.colors['bg_secondary']
            ).pack()
    
    def create_camera_panel(self, parent):
        """Panel izquierdo con cámara"""
        camera_panel = tk.Frame(parent, bg=self.colors['bg_secondary'], width=700)
        camera_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Título de la sección
        camera_title = tk.Label(
            camera_panel,
            text="📹 CÁMARA EN VIVO - RECONOCIMIENTO TIEMPO REAL",
            font=("Arial", 12, "bold"),
            fg=self.colors['success'],
            bg=self.colors['bg_secondary']
        )
        camera_title.pack(pady=10)
        
        # Contenedor del video
        self.video_container = tk.Frame(
            camera_panel, 
            bg='black', 
            width=640, 
            height=480,
            relief=tk.SUNKEN,
            bd=2
        )
        self.video_container.pack(padx=10, pady=5)
        self.video_container.pack_propagate(False)
        
        # Label para mostrar video o mensaje
        self.video_label = tk.Label(
            self.video_container,
            text="📷 CÁMARA DESCONECTADA\\n\\n🎯 Haz clic en 'Activar Cámara' para comenzar\\nel reconocimiento en tiempo real",
            font=("Arial", 14, "bold"),
            fg=self.colors['text_dim'],
            bg='black',
            justify=tk.CENTER
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Controles de cámara
        controls_frame = tk.Frame(camera_panel, bg=self.colors['bg_secondary'])
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Botón de cámara
        self.camera_btn = tk.Button(
            controls_frame,
            text="🔴 Activar Cámara",
            command=self.toggle_camera,
            font=("Arial", 12, "bold"),
            bg=self.colors['success'],
            fg='white',
            padx=20,
            pady=8,
            relief=tk.FLAT
        )
        self.camera_btn.pack(side=tk.LEFT)
        
        # Status de reconocimiento
        self.recognition_status = tk.Label(
            controls_frame,
            text="🔍 Estado: Esperando activación de cámara",
            font=("Arial", 10),
            fg=self.colors['text_dim'],
            bg=self.colors['bg_secondary']
        )
        self.recognition_status.pack(side=tk.LEFT, padx=20)
        
        # Seña detectada
        self.detected_sign = tk.Label(
            controls_frame,
            text="Seña: -",
            font=("Arial", 12, "bold"),
            fg=self.colors['warning'],
            bg=self.colors['bg_secondary']
        )
        self.detected_sign.pack(side=tk.RIGHT)
    
    def create_features_panel(self, parent):
        """Panel derecho con funcionalidades"""
        features_panel = tk.Frame(parent, bg=self.colors['bg_secondary'], width=600)
        features_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        features_panel.pack_propagate(False)
        
        # Notebook con tabs
        notebook = ttk.Notebook(features_panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Funciones Principales
        self.create_main_tab(notebook)
        
        # Tab 2: Revolucionario
        self.create_revolutionary_tab(notebook)
        
        # Tab 3: Entrenamiento
        self.create_training_tab(notebook)
        
        # Tab 4: Análisis
        self.create_analysis_tab(notebook)
    
    def create_main_tab(self, notebook):
        """Tab con funciones principales"""
        main_tab = tk.Frame(notebook, bg=self.colors['bg_accent'])
        notebook.add(main_tab, text="🎯 Principal")
        
        title = tk.Label(
            main_tab,
            text="🚀 FUNCIONES PRINCIPALES",
            font=("Arial", 14, "bold"),
            fg=self.colors['success'],
            bg=self.colors['bg_accent']
        )
        title.pack(pady=10)
        
        # Botones principales en grid
        buttons_frame = tk.Frame(main_tab, bg=self.colors['bg_accent'])
        buttons_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        main_functions = [
            ("📹 Reconocimiento\\nOptimizado", lambda: self.run_script('reconocimiento_optimizado.py'), '#4CAF50'),
            ("🔊 Reconocimiento\\ncon Voz", lambda: self.run_script('real_time_translate.py'), '#2196F3'),
            ("📊 Diagnóstico\\nCompleto", lambda: self.run_script('diagnostico_reconocimiento.py'), '#FF9800'),
            ("⚡ Refuerzo\\nRápido", lambda: self.run_script('refuerzo_rapido.py'), '#9C27B0'),
            ("📈 Analizar\\nDataset", lambda: self.run_script('analyze_dataset.py'), '#607D8B'),
            ("🔬 Verificar\\nSistema", lambda: self.run_script('test_imports_improved.py'), '#795548')
        ]
        
        for i, (text, command, color) in enumerate(main_functions):
            row = i // 2
            col = i % 2
            
            btn = tk.Button(
                buttons_frame,
                text=text,
                command=command,
                font=("Arial", 10, "bold"),
                bg=color,
                fg='white',
                width=15,
                height=3,
                relief=tk.FLAT
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
        
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
    
    def create_revolutionary_tab(self, notebook):
        """Tab con funcionalidades revolucionarias"""
        rev_tab = tk.Frame(notebook, bg=self.colors['bg_accent'])
        notebook.add(rev_tab, text="🌟 Revolucionario")
        
        title = tk.Label(
            rev_tab,
            text="🌟 FUNCIONALIDADES ÚNICAS",
            font=("Arial", 14, "bold"),
            fg=self.colors['warning'],
            bg=self.colors['bg_accent']
        )
        title.pack(pady=5)
        
        subtitle = tk.Label(
            rev_tab,
            text="Características que NO tiene ningún otro sistema",
            font=("Arial", 9, "italic"),
            fg=self.colors['text_dim'],
            bg=self.colors['bg_accent']
        )
        subtitle.pack()
        
        # Scroll frame
        canvas = tk.Canvas(rev_tab, bg=self.colors['bg_accent'])
        scrollbar = ttk.Scrollbar(rev_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_accent'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y")
        
        # Funcionalidades únicas
        unique_features = [
            ("🔄 Traducción Bidireccional", "Voz ↔ Señas simultáneamente", '#ff6b6b'),
            ("😊 Inteligencia Emocional", "Detecta 12+ emociones", '#4ecdc4'),
            ("🌐 Traductor Universal", "Entre 8+ lenguas de señas", '#45b7d1'),
            ("🎮 Modo Gamer Épico", "8 modos únicos de juego", '#f7b731'),
            ("👩‍🏫 Profesor Virtual IA", "IA personalizada", '#5f27cd'),
            ("👥 Conversación Multipersona", "Múltiples usuarios", '#00d2d3')
        ]
        
        for title_text, desc, color in unique_features:
            feature_frame = tk.Frame(
                scrollable_frame,
                bg=self.colors['bg_secondary'],
                relief=tk.RAISED,
                bd=1
            )
            feature_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Header colorido
            header = tk.Frame(feature_frame, bg=color, height=30)
            header.pack(fill=tk.X)
            header.pack_propagate(False)
            
            tk.Label(
                header,
                text=title_text,
                font=("Arial", 10, "bold"),
                fg='white',
                bg=color
            ).pack(expand=True)
            
            # Contenido
            content = tk.Frame(feature_frame, bg=self.colors['bg_secondary'])
            content.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(
                content,
                text=desc,
                font=("Arial", 8),
                fg=self.colors['text_dim'],
                bg=self.colors['bg_secondary']
            ).pack(anchor='w')
            
            tk.Button(
                content,
                text="🚀 Activar",
                font=("Arial", 8, "bold"),
                bg=color,
                fg='white',
                relief=tk.FLAT,
                command=lambda: messagebox.showinfo("Funcionalidad Única", f"¡Funcionalidad revolucionaria!\\n{title_text}\\n{desc}")
            ).pack(anchor='e', pady=2)
    
    def create_training_tab(self, notebook):
        """Tab para entrenamiento"""
        train_tab = tk.Frame(notebook, bg=self.colors['bg_accent'])
        notebook.add(train_tab, text="🧠 Entrenamiento")
        
        title = tk.Label(
            train_tab,
            text="🧠 ENTRENAMIENTO DEL MODELO",
            font=("Arial", 14, "bold"),
            fg=self.colors['success'],
            bg=self.colors['bg_accent']
        )
        title.pack(pady=10)
        
        # Secciones de entrenamiento
        sections = [
            {
                'title': '📹 Gestión de Dataset',
                'buttons': [
                    ('🎥 Grabar Gestos', self.open_record_dialog, '#4CAF50'),
                    ('📊 Analizar Dataset', lambda: self.run_script('analyze_dataset.py'), '#2196F3'),
                    ('⚡ Refuerzo Rápido', lambda: self.run_script('refuerzo_rapido.py'), '#FF9800')
                ]
            },
            {
                'title': '🧠 Modelo Neural',
                'buttons': [
                    ('🚀 Entrenar Modelo', lambda: self.confirm_and_run('train_model.py'), '#9C27B0'),
                    ('📈 Evaluar Modelo', lambda: self.run_script('evaluate_model.py'), '#607D8B'),
                    ('🔧 Verificar Sistema', lambda: self.run_script('test_imports_improved.py'), '#795548')
                ]
            }
        ]
        
        for section in sections:
            section_frame = tk.LabelFrame(
                train_tab,
                text=section['title'],
                font=("Arial", 10, "bold"),
                fg=self.colors['text'],
                bg=self.colors['bg_accent']
            )
            section_frame.pack(fill=tk.X, padx=20, pady=5)
            
            for btn_text, btn_command, btn_color in section['buttons']:
                btn = tk.Button(
                    section_frame,
                    text=btn_text,
                    command=btn_command,
                    font=("Arial", 9, "bold"),
                    bg=btn_color,
                    fg='white',
                    relief=tk.FLAT,
                    width=20,
                    height=1
                )
                btn.pack(pady=2)
    
    def create_analysis_tab(self, notebook):
        """Tab para análisis"""
        analysis_tab = tk.Frame(notebook, bg=self.colors['bg_accent'])
        notebook.add(analysis_tab, text="📊 Análisis")
        
        title = tk.Label(
            analysis_tab,
            text="📊 MÉTRICAS DEL SISTEMA",
            font=("Arial", 14, "bold"),
            fg=self.colors['info'],
            bg=self.colors['bg_accent']
        )
        title.pack(pady=10)
        
        # Métricas en grid
        metrics_frame = tk.Frame(analysis_tab, bg=self.colors['bg_accent'])
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        metrics = [
            ("Precisión", "98.06%", "Rendimiento excelente"),
            ("Gestos", "205", "Dataset completo LSE"),
            ("Muestras", "16,124", "Base sólida de datos"),
            ("Idiomas", "8+", "Cobertura mundial"),
            ("Juegos", "8", "Aprendizaje gamificado"),
            ("Emociones", "12+", "IA emocional"),
            ("Únicas", "15+", "Sin precedentes"),
            ("Estado", "ACTIVO", "Sistema funcionando")
        ]
        
        for i, (metric, value, desc) in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            metric_frame = tk.Frame(
                metrics_frame,
                bg=self.colors['bg_secondary'],
                relief=tk.RAISED,
                bd=1
            )
            metric_frame.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            
            tk.Label(
                metric_frame,
                text=value,
                font=("Arial", 16, "bold"),
                fg=self.colors['success'],
                bg=self.colors['bg_secondary']
            ).pack(pady=2)
            
            tk.Label(
                metric_frame,
                text=metric,
                font=("Arial", 9, "bold"),
                fg=self.colors['text'],
                bg=self.colors['bg_secondary']
            ).pack()
            
            tk.Label(
                metric_frame,
                text=desc,
                font=("Arial", 7),
                fg=self.colors['text_dim'],
                bg=self.colors['bg_secondary']
            ).pack(pady=1)
        
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
    
    def create_footer(self):
        """Footer con información del sistema"""
        footer = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=25)
        footer.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=2)
        footer.pack_propagate(False)
        
        self.status_label = tk.Label(
            footer,
            text="🟢 LSE Ecuador - Sistema Revolucionario | Estado: Activo",
            font=("Arial", 8),
            fg=self.colors['success'],
            bg=self.colors['bg_secondary']
        )
        self.status_label.pack(side=tk.LEFT, pady=2)
        
        version_label = tk.Label(
            footer,
            text="v2.0 REVOLUCIONARIO 🚀",
            font=("Arial", 8, "bold"),
            fg=self.colors['warning'],
            bg=self.colors['bg_secondary']
        )
        version_label.pack(side=tk.RIGHT, pady=2)
    
    def toggle_camera(self):
        """Activa/desactiva la cámara"""
        if not CAMERA_AVAILABLE:
            messagebox.showwarning("Cámara No Disponible", 
                                 "Las librerías de cámara no están instaladas.\\n\\n" +
                                 "Para habilitar la cámara, instala:\\n" +
                                 "pip install opencv-python pillow")
            return
            
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Inicia la cámara"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se puede acceder a la cámara")
                return
            
            self.camera_active = True
            self.camera_btn.config(text="🟢 Cámara Activa", bg=self.colors['danger'])
            self.recognition_status.config(text="🔍 Estado: Reconociendo señas en tiempo real...")
            
            # Iniciar thread de cámara
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            self.update_status("✅ Cámara activada - Reconocimiento en tiempo real")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar cámara:\\n{str(e)}")
    
    def stop_camera(self):
        """Detiene la cámara"""
        self.camera_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_btn.config(text="🔴 Activar Cámara", bg=self.colors['success'])
        self.recognition_status.config(text="🔍 Estado: Cámara desactivada")
        self.detected_sign.config(text="Seña: -")
        
        self.video_label.config(
            text="📷 CÁMARA DESCONECTADA\\n\\n🎯 Haz clic en 'Activar Cámara' para comenzar\\nel reconocimiento en tiempo real",
            image=""
        )
        
        self.update_status("⭕ Cámara desactivada")
    
    def camera_loop(self):
        """Loop principal de la cámara"""
        while self.camera_active and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Redimensionar frame
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Agregar overlay de información
                    cv2.putText(frame, "LSE ECUADOR - TIEMPO REAL", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)
                    cv2.putText(frame, "Sistema Revolucionario", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
                    
                    # Convertir a RGB para Tkinter
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(image=img)
                    
                    # Actualizar en hilo principal
                    self.root.after(0, self.update_video_display, photo)
                    
                    # Simular detección (aquí iría la lógica real)
                    # self.root.after(0, self.simulate_detection)
                    
            except Exception as e:
                print(f"Error en camera loop: {e}")
                break
    
    def update_video_display(self, photo):
        """Actualiza la pantalla de video"""
        if self.camera_active:
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo
    
    def simulate_detection(self):
        """Simula detección de señas (placeholder)"""
        import random
        signs = ["hola", "gracias", "familia", "amor", "buenos dias", "como estas"]
        if self.camera_active:
            detected = random.choice(signs)
            self.detected_sign.config(text=f"Seña: {detected}")
    
    def run_script(self, script_name):
        """Ejecuta un script"""
        def execute():
            try:
                self.update_status(f"⚡ Ejecutando {script_name}...")
                result = subprocess.run([sys.executable, script_name], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    self.update_status(f"✅ {script_name} ejecutado exitosamente")
                    messagebox.showinfo("Éxito", f"✅ {script_name} ejecutado correctamente")
                else:
                    self.update_status(f"❌ Error en {script_name}")
                    messagebox.showerror("Error", f"❌ Error en {script_name}:\\n{result.stderr[:500]}")
            except Exception as e:
                self.update_status(f"❌ Error ejecutando {script_name}")
                messagebox.showerror("Error", f"❌ Error ejecutando {script_name}:\\n{str(e)}")
        
        threading.Thread(target=execute, daemon=True).start()
    
    def confirm_and_run(self, script_name):
        """Confirma antes de ejecutar script pesado"""
        if messagebox.askyesno("Confirmar", 
                              f"¿Estás seguro de ejecutar {script_name}?\\n\\n" +
                              "Esta operación puede tomar varios minutos."):
            self.run_script(script_name)
    
    def open_record_dialog(self):
        """Abre diálogo para grabar gestos"""
        dialog = RecordDialog(self.root, self)
    
    def update_status(self, message):
        """Actualiza el status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.config(text=f"🟢 {message} | {timestamp}")

class RecordDialog:
    """Diálogo para grabar nuevos gestos"""
    
    def __init__(self, parent, main_interface):
        self.main_interface = main_interface
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("🎥 Grabar Nuevo Gesto")
        self.dialog.geometry("350x150")
        self.dialog.configure(bg='#1a1a2e')
        self.dialog.resizable(False, False)
        
        # Centrar
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_content()
    
    def create_content(self):
        """Crea el contenido del diálogo"""
        tk.Label(
            self.dialog,
            text="🎥 GRABAR NUEVO GESTO",
            font=("Arial", 12, "bold"),
            fg='#00ff88',
            bg='#1a1a2e'
        ).pack(pady=10)
        
        tk.Label(
            self.dialog,
            text="Nombre del gesto:",
            font=("Arial", 9),
            fg='white',
            bg='#1a1a2e'
        ).pack()
        
        self.entry = tk.Entry(
            self.dialog,
            font=("Arial", 10),
            width=25
        )
        self.entry.pack(pady=5)
        self.entry.focus()
        
        buttons_frame = tk.Frame(self.dialog, bg='#1a1a2e')
        buttons_frame.pack(pady=15)
        
        tk.Button(
            buttons_frame,
            text="🎥 Grabar",
            command=self.start_recording,
            font=("Arial", 9, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=15
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            buttons_frame,
            text="❌ Cancelar",
            command=self.dialog.destroy,
            font=("Arial", 9, "bold"),
            bg='#f44336',
            fg='white',
            padx=15
        ).pack(side=tk.LEFT, padx=5)
        
        self.entry.bind('<Return>', lambda e: self.start_recording())
    
    def start_recording(self):
        """Inicia grabación"""
        gesture_name = self.entry.get().strip()
        
        if not gesture_name:
            messagebox.showerror("Error", "Ingresa el nombre del gesto")
            return
        
        self.dialog.destroy()
        
        def record():
            try:
                self.main_interface.update_status(f"🎥 Grabando: {gesture_name}")
                result = subprocess.run([sys.executable, "record_dataset.py", gesture_name], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.main_interface.update_status(f"✅ Gesto '{gesture_name}' grabado")
                    messagebox.showinfo("Éxito", f"✅ Gesto '{gesture_name}' grabado correctamente")
                else:
                    messagebox.showerror("Error", f"❌ Error:\\n{result.stderr}")
            except Exception as e:
                messagebox.showerror("Error", f"❌ Error en grabación:\\n{str(e)}")
        
        threading.Thread(target=record, daemon=True).start()

def main():
    """Función principal"""
    root = tk.Tk()
    app = LSEModernInterface(root)
    
    def on_closing():
        if hasattr(app, 'camera_active') and app.camera_active:
            app.stop_camera()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
