#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🤟 TRADUCTOR LSE - APLICACIÓN DE ESCRITORIO
═══════════════════════════════════════════════════════════════════════════════

Traduce lengua de señas en tiempo real usando el modelo entrenado.

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

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
MODEL_DIR = BASE_DIR / "model"

CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.55,
    'STABILITY_FRAMES': 2,
    'COOLDOWN_TIME': 0.8,
    'SILENCE_TIMEOUT': 2.0,
}

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
        # Detectar qué comando de TTS está disponible
        self.tts_cmd = None
        for cmd in ['espeak-ng', 'espeak']:
            try:
                subprocess.run([cmd, '--version'], capture_output=True, timeout=5)
                self.tts_cmd = cmd
                break
            except:
                continue
    
    def speak(self, text: str):
        if not text:
            return
        
        def _speak():
            with self.lock:
                try:
                    if self.is_macos:
                        subprocess.run(['say', '-v', 'Mónica', text], 
                                      capture_output=True, timeout=30)
                    elif self.tts_cmd:
                        subprocess.run([self.tts_cmd, '-ves', '-s', '150', text],
                                      capture_output=True, timeout=30)
                except:
                    pass
        
        threading.Thread(target=_speak, daemon=True).start()


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
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def load_model(self) -> tuple:
        try:
            if not MODEL_DIR.exists():
                return False, "No se encontró la carpeta del modelo"
            
            labels_path = MODEL_DIR / "labels.pkl"
            tflite_path = MODEL_DIR / "model.tflite"
            h5_path = MODEL_DIR / "best_model.h5"
            
            if not labels_path.exists():
                return False, "No hay modelo entrenado. Usa el Entrenador primero."
            
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
            
            # Intentar TFLite primero
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
                except:
                    self.use_tflite = False
            
            # Fallback a H5
            if not self.use_tflite and h5_path.exists():
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(h5_path))
            
            if not self.use_tflite and self.model is None:
                return False, "No se pudo cargar el modelo"
            
            self.loaded = True
            return True, f"Modelo cargado: {len(self.labels)} señas ({', '.join(self.labels)})"
            
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
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([int(lm.x * w), int(lm.y * h), lm.z])
            all_landmarks.append(np.array(landmarks))
        
        # Preparar entrada
        landmarks_list = [h.flatten() for h in all_landmarks]
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
    
    def draw_landmarks(self, frame, hand_landmarks):
        if hand_landmarks:
            for landmarks in hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )


# ════════════════════════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

class TraductorLSEApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=COLORS['bg_dark'])
        
        self.running = False
        self.camera = None
        self.recognizer = GestureRecognizer()
        self.tts = TextToSpeech()
        
        self.current_gesture = None
        self.stability_count = 0
        self.last_spoken = None
        self.last_spoken_time = 0
        self.gesture_history = deque(maxlen=10)
        self.phrase_buffer = []
        self.last_gesture_time = time.time()
        
        self.gestures_detected = 0
        self.fps = 0
        self.fps_history = deque(maxlen=30)
        
        self.frame_queue = queue.Queue(maxsize=2)
        
        self._create_ui()
        self._load_model()
    
    def _create_ui(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(header_frame, text="🤟 Traductor LSE", font=('Helvetica', 24, 'bold'),
                bg=COLORS['bg_dark'], fg=COLORS['accent']).pack(side=tk.LEFT)
        
        self.status_label = tk.Label(header_frame, text="⏸️ Detenido",
                                    font=('Helvetica', 12), bg=COLORS['bg_dark'], fg=COLORS['text_secondary'])
        self.status_label.pack(side=tk.RIGHT)
        
        # Contenido
        content_frame = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Cámara
        left_panel = tk.Frame(content_frame, bg=COLORS['bg_dark'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.video_canvas = tk.Canvas(left_panel, width=640, height=480,
                                     bg=COLORS['bg_medium'], highlightthickness=2,
                                     highlightbackground=COLORS['bg_light'])
        self.video_canvas.pack(pady=10)
        self.video_canvas.create_text(320, 240,
                                     text="📷 Presiona INICIAR para comenzar",
                                     fill=COLORS['text_secondary'], font=('Helvetica', 14),
                                     tags="placeholder")
        
        self.fps_label = tk.Label(left_panel, text="FPS: --", font=('Helvetica', 11),
                                 bg=COLORS['bg_dark'], fg=COLORS['text_secondary'])
        self.fps_label.pack()
        
        # Panel derecho
        right_panel = tk.Frame(content_frame, bg=COLORS['bg_medium'], width=300, padx=20, pady=20)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Seña detectada
        tk.Label(right_panel, text="SEÑA DETECTADA", font=('Helvetica', 10),
                bg=COLORS['bg_medium'], fg=COLORS['text_secondary']).pack()
        
        self.gesture_label = tk.Label(right_panel, text="---", font=('Helvetica', 32, 'bold'),
                                     bg=COLORS['bg_medium'], fg=COLORS['text_secondary'])
        self.gesture_label.pack(pady=10)
        
        self.confidence_label = tk.Label(right_panel, text="Confianza: --%",
                                        font=('Helvetica', 12), bg=COLORS['bg_medium'],
                                        fg=COLORS['text_secondary'])
        self.confidence_label.pack()
        
        ttk.Separator(right_panel, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Frase
        tk.Label(right_panel, text="FRASE ACTUAL", font=('Helvetica', 10),
                bg=COLORS['bg_medium'], fg=COLORS['text_secondary']).pack()
        
        self.phrase_label = tk.Label(right_panel, text="", font=('Helvetica', 14),
                                    bg=COLORS['bg_medium'], fg=COLORS['text'],
                                    wraplength=260, justify=tk.CENTER)
        self.phrase_label.pack(pady=10)
        
        # Historial
        tk.Label(right_panel, text="HISTORIAL", font=('Helvetica', 10),
                bg=COLORS['bg_medium'], fg=COLORS['text_secondary']).pack()
        
        self.history_listbox = tk.Listbox(right_panel, font=('Helvetica', 10),
                                         bg=COLORS['bg_dark'], fg=COLORS['text'],
                                         height=8, borderwidth=0)
        self.history_listbox.pack(fill=tk.X, pady=10)
        
        # Stats
        self.stats_label = tk.Label(right_panel, text="Detectadas: 0", font=('Helvetica', 11),
                                   bg=COLORS['bg_light'], fg=COLORS['text'], padx=15, pady=8)
        self.stats_label.pack(fill=tk.X)
        
        # Botones
        button_frame = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        button_frame.pack(fill=tk.X, pady=20)
        
        self.start_btn = tk.Button(button_frame, text="▶️ INICIAR",
                                  font=('Helvetica', 14, 'bold'), bg=COLORS['success'],
                                  fg=COLORS['text'], bd=0, padx=30, pady=12,
                                  command=self.toggle_translation)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🗑️ Limpiar", font=('Helvetica', 12),
                 bg=COLORS['bg_light'], fg=COLORS['text'], bd=0, padx=20, pady=12,
                 command=self.clear_history).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🔊 Hablar Frase", font=('Helvetica', 12),
                 bg=COLORS['bg_light'], fg=COLORS['text'], bd=0, padx=20, pady=12,
                 command=self.speak_phrase).pack(side=tk.LEFT, padx=5)
        
        self.model_label = tk.Label(button_frame, text="Cargando...", font=('Helvetica', 10),
                                   bg=COLORS['bg_dark'], fg=COLORS['text_secondary'])
        self.model_label.pack(side=tk.RIGHT)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def _load_model(self):
        def load():
            success, message = self.recognizer.load_model()
            self.root.after(0, lambda: self._on_model_loaded(success, message))
        
        threading.Thread(target=load, daemon=True).start()
    
    def _on_model_loaded(self, success, message):
        if success:
            self.model_label.config(text=f"✅ {message}", fg=COLORS['success'])
            self.tts.speak("Traductor listo")
        else:
            self.model_label.config(text=f"❌ {message}", fg=COLORS['accent'])
    
    def toggle_translation(self):
        if self.running:
            self.stop_translation()
        else:
            self.start_translation()
    
    def start_translation(self):
        if not self.recognizer.loaded:
            messagebox.showwarning("Aviso", "No hay modelo cargado. Usa el Entrenador primero.")
            return
        
        for cam_id in [0, 1, 2]:
            self.camera = cv2.VideoCapture(cam_id)
            if self.camera.isOpened():
                ret, _ = self.camera.read()
                if ret:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
            self.camera.release()
        else:
            messagebox.showerror("Error", "No se encontró cámara")
            return
        
        self.running = True
        self.start_btn.config(text="⏹️ DETENER", bg=COLORS['accent'])
        self.status_label.config(text="🔴 Traduciendo...")
        self.video_canvas.delete("placeholder")
        
        self.tts.speak("Iniciando traducción")
        
        threading.Thread(target=self._capture_loop, daemon=True).start()
        self._update_frame()
    
    def stop_translation(self):
        self.running = False
        if self.camera:
            self.camera.release()
        
        self.start_btn.config(text="▶️ INICIAR", bg=COLORS['success'])
        self.status_label.config(text="⏸️ Detenido")
        
        self.video_canvas.delete("all")
        self.video_canvas.create_text(320, 240, text="📷 Presiona INICIAR",
                                     fill=COLORS['text_secondary'], font=('Helvetica', 14))
    
    def _capture_loop(self):
        last_time = time.time()
        
        while self.running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            gesture, confidence, hand_landmarks = self.recognizer.detect_and_predict(frame)
            
            if hand_landmarks:
                self.recognizer.draw_landmarks(frame, hand_landmarks)
            
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
            
            if (len(self.phrase_buffer) > 0 and 
                current_time - self.last_gesture_time > CONFIG['SILENCE_TIMEOUT']):
                phrase = " ".join(self.phrase_buffer)
                self.tts.speak(phrase)
                self.phrase_buffer.clear()
            
            fps = 1.0 / max(current_time - last_time, 0.001)
            self.fps_history.append(fps)
            self.fps = sum(self.fps_history) / len(self.fps_history)
            last_time = current_time
            
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                if not self.frame_queue.full():
                    self.frame_queue.put((img, gesture, confidence))
            except:
                pass
            
            time.sleep(0.01)
    
    def _update_frame(self):
        if not self.running:
            return
        
        try:
            if not self.frame_queue.empty():
                img, gesture, confidence = self.frame_queue.get_nowait()
                
                photo = ImageTk.PhotoImage(img)
                self.video_canvas.delete("all")
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.video_canvas.image = photo
                
                if gesture and confidence >= CONFIG['CONFIDENCE_THRESHOLD']:
                    self.gesture_label.config(text=gesture, fg=COLORS['success'])
                    self.confidence_label.config(text=f"Confianza: {confidence*100:.0f}%")
                else:
                    self.gesture_label.config(text="---", fg=COLORS['text_secondary'])
                    self.confidence_label.config(text="Confianza: --%")
                
                if self.phrase_buffer:
                    self.phrase_label.config(text=" ".join(self.phrase_buffer))
                
                self.history_listbox.delete(0, tk.END)
                for g, c, t in reversed(list(self.gesture_history)):
                    self.history_listbox.insert(tk.END, f"{t.strftime('%H:%M:%S')} - {g} ({c*100:.0f}%)")
                
                self.fps_label.config(text=f"FPS: {self.fps:.1f}")
                self.stats_label.config(text=f"Detectadas: {self.gestures_detected}")
        except:
            pass
        
        self.root.after(30, self._update_frame)
    
    def clear_history(self):
        self.gesture_history.clear()
        self.phrase_buffer.clear()
        self.history_listbox.delete(0, tk.END)
        self.phrase_label.config(text="")
        self.gestures_detected = 0
        self.stats_label.config(text="Detectadas: 0")
    
    def speak_phrase(self):
        if self.phrase_buffer:
            self.tts.speak(" ".join(self.phrase_buffer))
    
    def on_close(self):
        self.running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()


def main():
    app = TraductorLSEApp()
    app.run()


if __name__ == "__main__":
    main()
