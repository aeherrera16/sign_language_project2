#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🤟 TRADUCTOR LSE - APLICACIÓN DE ESCRITORIO PARA RASPBERRY PI
═══════════════════════════════════════════════════════════════════════════════

Aplicación de escritorio completa con interfaz gráfica que:
- Muestra video de cámara en tiempo real
- Reconoce señas instantáneamente  
- Habla las traducciones
- Funciona como cualquier app nativa de escritorio

═══════════════════════════════════════════════════════════════════════════════
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import mediapipe as mp
import pickle
import subprocess
import time
import sys
import threading
import queue
from pathlib import Path
from PIL import Image, ImageTk
from collections import deque
from datetime import datetime
import os

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════════

APP_NAME = "Traductor LSE"
APP_VERSION = "1.0.0"
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 700

# Rutas del modelo
SCRIPT_DIR = Path(__file__).parent
MODEL_PATHS = [
    SCRIPT_DIR / "model",
    SCRIPT_DIR.parent / "model",
    SCRIPT_DIR.parent.parent / "backend" / "model",
    Path("/home/pi/traductor-lse/model"),
]

# Configuración de reconocimiento
CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.60,
    'STABILITY_FRAMES': 4,
    'COOLDOWN_TIME': 1.5,
    'SILENCE_TIMEOUT': 2.5,
}

# Colores de la app (tema oscuro moderno)
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#0f3460',
    'accent': '#e94560',
    'accent_hover': '#ff6b6b',
    'text': '#ffffff',
    'text_secondary': '#a0a0a0',
    'success': '#00d26a',
    'warning': '#ffc107',
}


# ════════════════════════════════════════════════════════════════════════════════
# CLASE TTS
# ════════════════════════════════════════════════════════════════════════════════

class TextToSpeech:
    def __init__(self):
        self.is_macos = sys.platform == 'darwin'
        self.lock = threading.Lock()
    
    def speak(self, text: str):
        if not text:
            return
        
        def _speak():
            with self.lock:
                try:
                    if self.is_macos:
                        subprocess.run(['say', '-v', 'Mónica', text], 
                                      capture_output=True, timeout=30)
                    else:
                        subprocess.run(['espeak', '-ves', '-s', '150', text],
                                      capture_output=True, timeout=30)
                except:
                    pass
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()


# ════════════════════════════════════════════════════════════════════════════════
# CLASE RECONOCEDOR
# ════════════════════════════════════════════════════════════════════════════════

class GestureRecognizer:
    def __init__(self):
        self.model = None
        self.labels = []
        self.loaded = False
        self.use_tflite = False
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def find_model_dir(self) -> Path:
        for path in MODEL_PATHS:
            if path.exists() and (path / "labels.pkl").exists():
                return path
        return None
    
    def load_model(self) -> tuple:
        try:
            model_dir = self.find_model_dir()
            if not model_dir:
                return False, "No se encontró el modelo"
            
            labels_path = model_dir / "labels.pkl"
            tflite_path = model_dir / "model.tflite"
            h5_path = model_dir / "best_model.h5"
            
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
            
            if tflite_path.exists():
                try:
                    try:
                        import tflite_runtime.interpreter as tflite
                    except ImportError:
                        import tensorflow as tf
                        tflite = tf.lite
                    
                    self.interpreter = tflite.Interpreter(model_path=str(tflite_path))
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    self.use_tflite = True
                except Exception as e:
                    self.use_tflite = False
            
            if not self.use_tflite and h5_path.exists():
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(h5_path))
            
            if not self.use_tflite and self.model is None:
                return False, "No se pudo cargar el modelo"
            
            self.loaded = True
            return True, f"Modelo cargado: {len(self.labels)} señas"
            
        except Exception as e:
            return False, str(e)
    
    def detect_and_predict(self, frame) -> tuple:
        if not self.loaded:
            return None, 0.0, None
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            return None, 0.0, None
        
        # Extraer landmarks
        hands_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([int(lm.x * w), int(lm.y * h), lm.z])
            hands_landmarks.append(np.array(landmarks))
        
        # Preparar entrada
        landmarks_list = [h.flatten() for h in hands_landmarks]
        if len(landmarks_list) >= 2:
            input_vector = np.concatenate(landmarks_list[:2])
        else:
            input_vector = np.concatenate([landmarks_list[0], np.zeros(63)])
        
        if input_vector.shape[0] != 126:
            input_vector = np.pad(input_vector, (0, max(0, 126 - input_vector.shape[0])))[:126]
        
        input_data = np.expand_dims(input_vector, axis=0).astype(np.float32)
        
        try:
            if self.use_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                prediction = self.model.predict(input_data, verbose=0)[0]
            
            best_idx = np.argmax(prediction)
            confidence = float(prediction[best_idx])
            gesture = self.labels[best_idx] if best_idx < len(self.labels) else None
            
            return gesture, confidence, results.multi_hand_landmarks
        except:
            return None, 0.0, results.multi_hand_landmarks


# ════════════════════════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

class TraductorLSEApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.resizable(True, True)
        
        # Estado
        self.running = False
        self.camera = None
        self.recognizer = GestureRecognizer()
        self.tts = TextToSpeech()
        
        # Tracking de gestos
        self.current_gesture = None
        self.stability_count = 0
        self.last_spoken = None
        self.last_spoken_time = 0
        self.gesture_history = deque(maxlen=10)
        self.phrase_buffer = []
        self.last_gesture_time = time.time()
        
        # Estadísticas
        self.gestures_detected = 0
        self.fps = 0
        self.fps_history = deque(maxlen=30)
        
        # Cola para comunicación entre hilos
        self.frame_queue = queue.Queue(maxsize=2)
        
        self._setup_styles()
        self._create_ui()
        self._load_model()
    
    def _setup_styles(self):
        """Configura estilos de la app"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=COLORS['bg_dark'])
        style.configure('TLabel', background=COLORS['bg_dark'], foreground=COLORS['text'])
        style.configure('Title.TLabel', font=('Helvetica', 24, 'bold'), foreground=COLORS['accent'])
        style.configure('Subtitle.TLabel', font=('Helvetica', 14), foreground=COLORS['text_secondary'])
        style.configure('Status.TLabel', font=('Helvetica', 12), foreground=COLORS['success'])
        style.configure('Gesture.TLabel', font=('Helvetica', 36, 'bold'), foreground=COLORS['accent'])
        style.configure('History.TLabel', font=('Helvetica', 11), foreground=COLORS['text_secondary'])
        
        style.configure('Accent.TButton',
                       font=('Helvetica', 12, 'bold'),
                       background=COLORS['accent'],
                       foreground=COLORS['text'])
        style.map('Accent.TButton',
                 background=[('active', COLORS['accent_hover'])])
    
    def _create_ui(self):
        """Crea la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === HEADER ===
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="🤟 Traductor LSE", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(header_frame, text="⏸️ Detenido", style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT)
        
        # === CONTENIDO PRINCIPAL ===
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Cámara
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Canvas para video
        self.video_canvas = tk.Canvas(
            left_panel,
            width=640,
            height=480,
            bg=COLORS['bg_medium'],
            highlightthickness=2,
            highlightbackground=COLORS['bg_light']
        )
        self.video_canvas.pack(pady=10)
        
        # Texto inicial en canvas
        self.video_canvas.create_text(
            320, 240,
            text="📷 Cámara desactivada\nPresiona INICIAR para comenzar",
            fill=COLORS['text_secondary'],
            font=('Helvetica', 14),
            justify=tk.CENTER,
            tags="placeholder"
        )
        
        # Info de FPS
        self.fps_label = ttk.Label(left_panel, text="FPS: --", style='Subtitle.TLabel')
        self.fps_label.pack()
        
        # Panel derecho - Traducción
        right_panel = ttk.Frame(content_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Seña detectada
        detected_frame = tk.Frame(right_panel, bg=COLORS['bg_medium'], padx=20, pady=20)
        detected_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(detected_frame, text="SEÑA DETECTADA", 
                 font=('Helvetica', 10), foreground=COLORS['text_secondary'],
                 background=COLORS['bg_medium']).pack()
        
        self.gesture_label = tk.Label(
            detected_frame,
            text="---",
            font=('Helvetica', 32, 'bold'),
            fg=COLORS['accent'],
            bg=COLORS['bg_medium']
        )
        self.gesture_label.pack(pady=10)
        
        self.confidence_label = tk.Label(
            detected_frame,
            text="Confianza: --%",
            font=('Helvetica', 12),
            fg=COLORS['text_secondary'],
            bg=COLORS['bg_medium']
        )
        self.confidence_label.pack()
        
        # Frase acumulada
        phrase_frame = tk.Frame(right_panel, bg=COLORS['bg_medium'], padx=20, pady=15)
        phrase_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(phrase_frame, text="FRASE ACTUAL",
                 font=('Helvetica', 10), foreground=COLORS['text_secondary'],
                 background=COLORS['bg_medium']).pack()
        
        self.phrase_label = tk.Label(
            phrase_frame,
            text="",
            font=('Helvetica', 14),
            fg=COLORS['text'],
            bg=COLORS['bg_medium'],
            wraplength=260,
            justify=tk.CENTER
        )
        self.phrase_label.pack(pady=10)
        
        # Historial
        history_frame = tk.Frame(right_panel, bg=COLORS['bg_medium'], padx=20, pady=15)
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(history_frame, text="HISTORIAL",
                 font=('Helvetica', 10), foreground=COLORS['text_secondary'],
                 background=COLORS['bg_medium']).pack()
        
        self.history_listbox = tk.Listbox(
            history_frame,
            font=('Helvetica', 11),
            bg=COLORS['bg_dark'],
            fg=COLORS['text'],
            selectbackground=COLORS['accent'],
            borderwidth=0,
            highlightthickness=0
        )
        self.history_listbox.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Estadísticas
        stats_frame = tk.Frame(right_panel, bg=COLORS['bg_light'], padx=15, pady=10)
        stats_frame.pack(fill=tk.X)
        
        self.stats_label = tk.Label(
            stats_frame,
            text="Detectadas: 0",
            font=('Helvetica', 11),
            fg=COLORS['text'],
            bg=COLORS['bg_light']
        )
        self.stats_label.pack()
        
        # === BOTONES ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        self.start_btn = tk.Button(
            button_frame,
            text="▶️ INICIAR",
            font=('Helvetica', 14, 'bold'),
            bg=COLORS['success'],
            fg=COLORS['text'],
            activebackground=COLORS['accent'],
            bd=0,
            padx=30,
            pady=12,
            cursor='hand2',
            command=self.toggle_translation
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Limpiar",
            font=('Helvetica', 12),
            bg=COLORS['bg_light'],
            fg=COLORS['text'],
            activebackground=COLORS['bg_medium'],
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            command=self.clear_history
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        speak_btn = tk.Button(
            button_frame,
            text="🔊 Hablar Frase",
            font=('Helvetica', 12),
            bg=COLORS['bg_light'],
            fg=COLORS['text'],
            activebackground=COLORS['bg_medium'],
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            command=self.speak_phrase
        )
        speak_btn.pack(side=tk.LEFT, padx=5)
        
        # Info del modelo
        self.model_label = ttk.Label(button_frame, text="Cargando modelo...", style='Subtitle.TLabel')
        self.model_label.pack(side=tk.RIGHT)
        
        # Cerrar ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def _load_model(self):
        """Carga el modelo en segundo plano"""
        def load():
            success, message = self.recognizer.load_model()
            self.root.after(0, lambda: self._on_model_loaded(success, message))
        
        threading.Thread(target=load, daemon=True).start()
    
    def _on_model_loaded(self, success, message):
        """Callback cuando el modelo está cargado"""
        if success:
            labels = ", ".join(self.recognizer.labels[:5])
            if len(self.recognizer.labels) > 5:
                labels += f"... (+{len(self.recognizer.labels) - 5})"
            self.model_label.config(text=f"✅ {len(self.recognizer.labels)} señas: {labels}")
            self.tts.speak("Aplicación lista")
        else:
            self.model_label.config(text=f"❌ Error: {message}")
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{message}")
    
    def toggle_translation(self):
        """Inicia o detiene la traducción"""
        if self.running:
            self.stop_translation()
        else:
            self.start_translation()
    
    def start_translation(self):
        """Inicia la captura y traducción"""
        # Buscar cámara
        for cam_id in [0, 1, 2]:
            self.camera = cv2.VideoCapture(cam_id)
            if self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    break
            self.camera.release()
        else:
            messagebox.showerror("Error", "No se encontró cámara conectada")
            return
        
        self.running = True
        self.start_btn.config(text="⏹️ DETENER", bg=COLORS['accent'])
        self.status_label.config(text="🔴 Traduciendo...", foreground=COLORS['accent'])
        self.video_canvas.delete("placeholder")
        
        self.tts.speak("Iniciando traducción")
        
        # Iniciar hilo de captura
        threading.Thread(target=self._capture_loop, daemon=True).start()
        
        # Iniciar actualización de UI
        self._update_frame()
    
    def stop_translation(self):
        """Detiene la traducción"""
        self.running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.start_btn.config(text="▶️ INICIAR", bg=COLORS['success'])
        self.status_label.config(text="⏸️ Detenido", foreground=COLORS['text_secondary'])
        
        # Limpiar canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            320, 240,
            text="📷 Cámara desactivada\nPresiona INICIAR para comenzar",
            fill=COLORS['text_secondary'],
            font=('Helvetica', 14),
            justify=tk.CENTER,
            tags="placeholder"
        )
        
        self.tts.speak("Traducción detenida")
    
    def _capture_loop(self):
        """Bucle de captura en hilo separado"""
        last_time = time.time()
        
        while self.running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Detectar y predecir
            gesture, confidence, hand_landmarks = self.recognizer.detect_and_predict(frame)
            
            # Dibujar landmarks
            if hand_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                for landmarks in hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, landmarks, 
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
            
            # Procesar gesto
            current_time = time.time()
            if gesture and confidence >= CONFIG['CONFIDENCE_THRESHOLD']:
                if gesture == self.current_gesture:
                    self.stability_count += 1
                else:
                    self.current_gesture = gesture
                    self.stability_count = 1
                
                if self.stability_count >= CONFIG['STABILITY_FRAMES']:
                    if (self.last_spoken != gesture or 
                        current_time - self.last_spoken_time > CONFIG['COOLDOWN_TIME']):
                        
                        self.last_spoken = gesture
                        self.last_spoken_time = current_time
                        self.last_gesture_time = current_time
                        self.gestures_detected += 1
                        
                        self.phrase_buffer.append(gesture)
                        self.gesture_history.append((gesture, confidence, datetime.now()))
                        
                        self.tts.speak(gesture)
            else:
                self.stability_count = max(0, self.stability_count - 1)
            
            # Auto-sintetizar frase
            if (len(self.phrase_buffer) > 0 and 
                current_time - self.last_gesture_time > CONFIG['SILENCE_TIMEOUT']):
                phrase = " ".join(self.phrase_buffer)
                self.tts.speak(phrase)
                self.phrase_buffer.clear()
            
            # Calcular FPS
            fps = 1.0 / max(current_time - last_time, 0.001)
            self.fps_history.append(fps)
            self.fps = sum(self.fps_history) / len(self.fps_history)
            last_time = current_time
            
            # Enviar frame a la cola
            try:
                # Convertir para Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                if not self.frame_queue.full():
                    self.frame_queue.put((img, gesture, confidence))
            except:
                pass
            
            time.sleep(0.01)
    
    def _update_frame(self):
        """Actualiza el frame en el canvas"""
        if not self.running:
            return
        
        try:
            if not self.frame_queue.empty():
                img, gesture, confidence = self.frame_queue.get_nowait()
                
                # Redimensionar si es necesario
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                
                # Convertir a PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Actualizar canvas
                self.video_canvas.delete("all")
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.video_canvas.image = photo  # Mantener referencia
                
                # Actualizar labels
                if gesture and confidence >= CONFIG['CONFIDENCE_THRESHOLD']:
                    self.gesture_label.config(text=gesture, fg=COLORS['success'])
                    self.confidence_label.config(text=f"Confianza: {confidence*100:.0f}%")
                else:
                    self.gesture_label.config(text="---", fg=COLORS['text_secondary'])
                    self.confidence_label.config(text="Confianza: --%")
                
                # Actualizar frase
                if self.phrase_buffer:
                    self.phrase_label.config(text=" ".join(self.phrase_buffer))
                
                # Actualizar historial
                self.history_listbox.delete(0, tk.END)
                for g, c, t in reversed(list(self.gesture_history)):
                    self.history_listbox.insert(tk.END, f"{t.strftime('%H:%M:%S')} - {g} ({c*100:.0f}%)")
                
                # Actualizar estadísticas
                self.fps_label.config(text=f"FPS: {self.fps:.1f}")
                self.stats_label.config(text=f"Detectadas: {self.gestures_detected}")
        
        except Exception as e:
            pass
        
        # Programar siguiente actualización
        self.root.after(30, self._update_frame)
    
    def clear_history(self):
        """Limpia el historial"""
        self.gesture_history.clear()
        self.phrase_buffer.clear()
        self.history_listbox.delete(0, tk.END)
        self.phrase_label.config(text="")
        self.gestures_detected = 0
        self.stats_label.config(text="Detectadas: 0")
    
    def speak_phrase(self):
        """Habla la frase acumulada"""
        if self.phrase_buffer:
            phrase = " ".join(self.phrase_buffer)
            self.tts.speak(phrase)
    
    def on_close(self):
        """Maneja el cierre de la aplicación"""
        self.running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()
    
    def run(self):
        """Ejecuta la aplicación"""
        self.root.mainloop()


# ════════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════════

def main():
    try:
        app = TraductorLSEApp()
        app.run()
    except Exception as e:
        messagebox.showerror("Error Fatal", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
