#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🎓 ENTRENADOR DE MODELO - TRADUCTOR LSE
═══════════════════════════════════════════════════════════════════════════════

Aplicación para capturar señas y entrenar el modelo de reconocimiento.
Funciona de forma completamente independiente.

Flujo:
1. Capturar muestras de cada seña
2. Entrenar el modelo con los datos capturados
3. Guardar el modelo para uso en el traductor

═══════════════════════════════════════════════════════════════════════════════
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import numpy as np
import mediapipe as mp
import pickle
import json
import os
import sys
import threading
import time
from pathlib import Path
from PIL import Image, ImageTk
from datetime import datetime
from collections import deque

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════════

APP_NAME = "Entrenador de Señas"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 750

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

# Crear directorios
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Configuración de captura
SAMPLES_PER_GESTURE = 40
MIN_SAMPLES_TO_TRAIN = 20

# Colores
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


# ════════════════════════════════════════════════════════════════════════════════
# EXTRACTOR DE LANDMARKS
# ════════════════════════════════════════════════════════════════════════════════

class HandLandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract(self, frame) -> tuple:
        """Extrae landmarks de las manos"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([int(lm.x * w), int(lm.y * h), lm.z])
            all_landmarks.append(np.array(landmarks))
        
        # Aplanar para modelo
        landmarks_list = [h.flatten() for h in all_landmarks]
        if len(landmarks_list) >= 2:
            feature_vector = np.concatenate(landmarks_list[:2])
        else:
            feature_vector = np.concatenate([landmarks_list[0], np.zeros(63)])
        
        if feature_vector.shape[0] != 126:
            feature_vector = np.pad(feature_vector, (0, max(0, 126 - feature_vector.shape[0])))[:126]
        
        return feature_vector, results.multi_hand_landmarks
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Dibuja landmarks en el frame"""
        if hand_landmarks:
            for landmarks in hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )


# ════════════════════════════════════════════════════════════════════════════════
# GESTOR DE DATOS
# ════════════════════════════════════════════════════════════════════════════════

class DataManager:
    def __init__(self):
        self.data_file = DATA_DIR / "training_data.pkl"
        self.metadata_file = DATA_DIR / "metadata.json"
        self.data = {}  # {gesture_name: [samples]}
        self.load_data()
    
    def load_data(self):
        """Carga datos existentes"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'rb') as f:
                    self.data = pickle.load(f)
            except:
                self.data = {}
    
    def save_data(self):
        """Guarda los datos"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data, f)
        
        # Guardar metadata
        metadata = {
            'gestures': list(self.data.keys()),
            'samples_per_gesture': {k: len(v) for k, v in self.data.items()},
            'total_samples': sum(len(v) for v in self.data.values()),
            'last_updated': datetime.now().isoformat()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def add_sample(self, gesture_name: str, features: np.ndarray):
        """Agrega una muestra"""
        if gesture_name not in self.data:
            self.data[gesture_name] = []
        self.data[gesture_name].append(features)
    
    def get_gesture_count(self, gesture_name: str) -> int:
        """Obtiene el conteo de muestras para un gesto"""
        return len(self.data.get(gesture_name, []))
    
    def get_all_gestures(self) -> list:
        """Obtiene lista de gestos"""
        return list(self.data.keys())
    
    def delete_gesture(self, gesture_name: str):
        """Elimina un gesto"""
        if gesture_name in self.data:
            del self.data[gesture_name]
            self.save_data()
    
    def get_training_data(self) -> tuple:
        """Prepara datos para entrenamiento"""
        X, y = [], []
        labels = sorted(self.data.keys())
        
        for label_idx, gesture in enumerate(labels):
            for sample in self.data[gesture]:
                X.append(sample)
                y.append(label_idx)
        
        return np.array(X), np.array(y), labels


# ════════════════════════════════════════════════════════════════════════════════
# ENTRENADOR DE MODELO
# ════════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    @staticmethod
    def train(X, y, labels, progress_callback=None) -> tuple:
        """Entrena el modelo"""
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping, Callback
            from tensorflow.keras.utils import to_categorical
            from sklearn.model_selection import train_test_split
            
            if progress_callback:
                progress_callback("Preparando datos...", 0.1)
            
            # Preparar datos
            num_classes = len(labels)
            y_cat = to_categorical(y, num_classes)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_cat, test_size=0.2, random_state=42, stratify=y
            )
            
            if progress_callback:
                progress_callback("Construyendo modelo...", 0.2)
            
            # Construir modelo
            model = Sequential([
                Dense(256, activation='relu', input_shape=(126,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callback de progreso
            class ProgressCallback(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if progress_callback:
                        progress = 0.2 + (epoch / 50) * 0.7
                        acc = logs.get('val_accuracy', 0) * 100
                        progress_callback(f"Entrenando... Época {epoch+1}/50 - Precisión: {acc:.1f}%", progress)
            
            if progress_callback:
                progress_callback("Entrenando modelo...", 0.3)
            
            # Entrenar
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=16,
                callbacks=[early_stop, ProgressCallback()],
                verbose=0
            )
            
            if progress_callback:
                progress_callback("Guardando modelo...", 0.95)
            
            # Guardar modelo H5
            model.save(str(MODEL_DIR / "best_model.h5"))
            
            # Guardar labels
            with open(MODEL_DIR / "labels.pkl", 'wb') as f:
                pickle.dump(labels, f)
            
            # Intentar crear TFLite
            try:
                if progress_callback:
                    progress_callback("Creando versión optimizada...", 0.97)
                
                @tf.function(input_signature=[tf.TensorSpec(shape=[1, 126], dtype=tf.float32)])
                def serving_fn(x):
                    return model(x, training=False)
                
                concrete_func = serving_fn.get_concrete_function()
                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
                with open(MODEL_DIR / "model.tflite", 'wb') as f:
                    f.write(tflite_model)
            except Exception as e:
                print(f"Nota: No se pudo crear TFLite: {e}")
            
            # Calcular precisión final
            val_acc = max(history.history['val_accuracy']) * 100
            
            if progress_callback:
                progress_callback("¡Completado!", 1.0)
            
            return True, f"Modelo entrenado con {val_acc:.1f}% de precisión"
            
        except ImportError:
            return False, "TensorFlow no está instalado. Instálalo con: pip install tensorflow"
        except Exception as e:
            return False, str(e)


# ════════════════════════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

class EntrenadorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Componentes
        self.extractor = HandLandmarkExtractor()
        self.data_manager = DataManager()
        
        # Estado
        self.camera = None
        self.running = False
        self.capturing = False
        self.current_gesture = ""
        self.samples_captured = 0
        self.capture_delay = 0.3  # segundos entre capturas
        self.last_capture_time = 0
        
        self._create_ui()
        self._update_gesture_list()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def _create_ui(self):
        """Crea la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === HEADER ===
        header = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        header.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(header, text="🎓 Entrenador de Señas", font=('Helvetica', 20, 'bold'),
                bg=COLORS['bg_dark'], fg=COLORS['accent']).pack(side=tk.LEFT)
        
        self.status_label = tk.Label(header, text="⏸️ Cámara apagada",
                                    font=('Helvetica', 12), bg=COLORS['bg_dark'], fg=COLORS['text_secondary'])
        self.status_label.pack(side=tk.RIGHT)
        
        # === CONTENIDO ===
        content = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        content.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Cámara
        left_panel = tk.Frame(content, bg=COLORS['bg_dark'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Canvas de video
        self.video_canvas = tk.Canvas(left_panel, width=640, height=480,
                                     bg=COLORS['bg_medium'], highlightthickness=2,
                                     highlightbackground=COLORS['bg_light'])
        self.video_canvas.pack(pady=10)
        self.video_canvas.create_text(320, 240, text="📷 Presiona INICIAR CÁMARA",
                                     fill=COLORS['text_secondary'], font=('Helvetica', 14))
        
        # Controles de cámara
        cam_controls = tk.Frame(left_panel, bg=COLORS['bg_dark'])
        cam_controls.pack(fill=tk.X)
        
        self.cam_btn = tk.Button(cam_controls, text="📷 INICIAR CÁMARA",
                                font=('Helvetica', 12, 'bold'), bg=COLORS['success'],
                                fg=COLORS['text'], bd=0, padx=20, pady=10,
                                command=self.toggle_camera)
        self.cam_btn.pack(side=tk.LEFT, padx=5)
        
        # Info de captura
        self.capture_info = tk.Label(cam_controls, text="", font=('Helvetica', 12),
                                    bg=COLORS['bg_dark'], fg=COLORS['warning'])
        self.capture_info.pack(side=tk.LEFT, padx=20)
        
        # Panel derecho - Gestos
        right_panel = tk.Frame(content, bg=COLORS['bg_medium'], width=350, padx=15, pady=15)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Nueva seña
        tk.Label(right_panel, text="AGREGAR NUEVA SEÑA", font=('Helvetica', 11, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text']).pack(anchor=tk.W)
        
        input_frame = tk.Frame(right_panel, bg=COLORS['bg_medium'])
        input_frame.pack(fill=tk.X, pady=10)
        
        self.gesture_entry = tk.Entry(input_frame, font=('Helvetica', 12), width=20)
        self.gesture_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.gesture_entry.bind('<Return>', lambda e: self.start_capture())
        
        tk.Button(input_frame, text="+ Agregar", font=('Helvetica', 10),
                 bg=COLORS['accent'], fg=COLORS['text'], bd=0, padx=10, pady=5,
                 command=self.start_capture).pack(side=tk.LEFT)
        
        # Separador
        ttk.Separator(right_panel, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Lista de gestos
        tk.Label(right_panel, text="SEÑAS REGISTRADAS", font=('Helvetica', 11, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text']).pack(anchor=tk.W)
        
        # Frame con scroll para lista
        list_frame = tk.Frame(right_panel, bg=COLORS['bg_dark'])
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.gesture_listbox = tk.Listbox(list_frame, font=('Helvetica', 11),
                                         bg=COLORS['bg_dark'], fg=COLORS['text'],
                                         selectbackground=COLORS['accent'],
                                         height=12, yscrollcommand=scrollbar.set)
        self.gesture_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.gesture_listbox.yview)
        
        # Botones de gestión
        manage_frame = tk.Frame(right_panel, bg=COLORS['bg_medium'])
        manage_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(manage_frame, text="➕ Más muestras", font=('Helvetica', 10),
                 bg=COLORS['bg_light'], fg=COLORS['text'], bd=0, padx=10, pady=5,
                 command=self.add_more_samples).pack(side=tk.LEFT, padx=2)
        
        tk.Button(manage_frame, text="🗑️ Eliminar", font=('Helvetica', 10),
                 bg=COLORS['accent'], fg=COLORS['text'], bd=0, padx=10, pady=5,
                 command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        
        # Separador
        ttk.Separator(right_panel, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Botón de entrenamiento
        self.train_btn = tk.Button(right_panel, text="🚀 ENTRENAR MODELO",
                                  font=('Helvetica', 14, 'bold'), bg=COLORS['success'],
                                  fg=COLORS['text'], bd=0, padx=20, pady=12,
                                  command=self.train_model)
        self.train_btn.pack(fill=tk.X)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(right_panel, variable=self.progress_var,
                                           maximum=1.0, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=10)
        
        self.progress_label = tk.Label(right_panel, text="", font=('Helvetica', 10),
                                      bg=COLORS['bg_medium'], fg=COLORS['text_secondary'])
        self.progress_label.pack()
        
        # Info del modelo
        self.model_info = tk.Label(right_panel, text="", font=('Helvetica', 10),
                                  bg=COLORS['bg_medium'], fg=COLORS['success'],
                                  wraplength=300, justify=tk.LEFT)
        self.model_info.pack(pady=10)
        
        self._check_existing_model()
    
    def _update_gesture_list(self):
        """Actualiza la lista de gestos"""
        self.gesture_listbox.delete(0, tk.END)
        for gesture in sorted(self.data_manager.get_all_gestures()):
            count = self.data_manager.get_gesture_count(gesture)
            status = "✅" if count >= MIN_SAMPLES_TO_TRAIN else "⚠️"
            self.gesture_listbox.insert(tk.END, f"{status} {gesture} ({count} muestras)")
    
    def _check_existing_model(self):
        """Verifica si existe un modelo entrenado"""
        if (MODEL_DIR / "best_model.h5").exists() and (MODEL_DIR / "labels.pkl").exists():
            with open(MODEL_DIR / "labels.pkl", 'rb') as f:
                labels = pickle.load(f)
            self.model_info.config(text=f"✅ Modelo existente: {len(labels)} señas\n({', '.join(labels)})")
    
    def toggle_camera(self):
        """Enciende/apaga la cámara"""
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Inicia la cámara"""
        for cam_id in [0, 1, 2]:
            self.camera = cv2.VideoCapture(cam_id)
            if self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
            self.camera.release()
        else:
            messagebox.showerror("Error", "No se encontró cámara")
            return
        
        self.running = True
        self.cam_btn.config(text="⏹️ DETENER CÁMARA", bg=COLORS['accent'])
        self.status_label.config(text="🔴 Cámara activa")
        
        threading.Thread(target=self._camera_loop, daemon=True).start()
    
    def stop_camera(self):
        """Detiene la cámara"""
        self.running = False
        self.capturing = False
        if self.camera:
            self.camera.release()
        
        self.cam_btn.config(text="📷 INICIAR CÁMARA", bg=COLORS['success'])
        self.status_label.config(text="⏸️ Cámara apagada")
        
        self.video_canvas.delete("all")
        self.video_canvas.create_text(320, 240, text="📷 Presiona INICIAR CÁMARA",
                                     fill=COLORS['text_secondary'], font=('Helvetica', 14))
    
    def _camera_loop(self):
        """Bucle de cámara"""
        while self.running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Extraer landmarks
            features, hand_landmarks = self.extractor.extract(frame)
            
            # Dibujar landmarks
            if hand_landmarks:
                self.extractor.draw_landmarks(frame, hand_landmarks)
                
                # Capturar si está en modo captura
                if self.capturing and features is not None:
                    current_time = time.time()
                    if current_time - self.last_capture_time >= self.capture_delay:
                        self.data_manager.add_sample(self.current_gesture, features)
                        self.samples_captured += 1
                        self.last_capture_time = current_time
                        
                        # Actualizar UI
                        self.root.after(0, self._update_capture_progress)
                        
                        if self.samples_captured >= SAMPLES_PER_GESTURE:
                            self.root.after(0, self._finish_capture)
            
            # Dibujar info de captura
            if self.capturing:
                cv2.putText(frame, f"CAPTURANDO: {self.current_gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"{self.samples_captured}/{SAMPLES_PER_GESTURE}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Mostrar frame
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(img)
                
                self.video_canvas.delete("all")
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.video_canvas.image = photo
            except:
                pass
            
            time.sleep(0.03)
    
    def start_capture(self):
        """Inicia la captura de una nueva seña"""
        gesture = self.gesture_entry.get().strip().upper()
        if not gesture:
            messagebox.showwarning("Aviso", "Ingresa el nombre de la seña")
            return
        
        if not self.running:
            messagebox.showwarning("Aviso", "Primero inicia la cámara")
            return
        
        self.current_gesture = gesture
        self.samples_captured = self.data_manager.get_gesture_count(gesture)
        self.capturing = True
        self.last_capture_time = 0
        
        self.capture_info.config(text=f"📸 Capturando '{gesture}'... Muestra la seña")
        self.gesture_entry.delete(0, tk.END)
    
    def add_more_samples(self):
        """Agrega más muestras a un gesto existente"""
        selection = self.gesture_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecciona una seña de la lista")
            return
        
        if not self.running:
            messagebox.showwarning("Aviso", "Primero inicia la cámara")
            return
        
        # Extraer nombre del gesto
        item = self.gesture_listbox.get(selection[0])
        gesture = item.split(" ")[1]
        
        self.current_gesture = gesture
        self.samples_captured = self.data_manager.get_gesture_count(gesture)
        self.capturing = True
        self.last_capture_time = 0
        
        self.capture_info.config(text=f"📸 Agregando muestras a '{gesture}'...")
    
    def _update_capture_progress(self):
        """Actualiza el progreso de captura"""
        self.capture_info.config(
            text=f"📸 {self.current_gesture}: {self.samples_captured}/{SAMPLES_PER_GESTURE}"
        )
    
    def _finish_capture(self):
        """Finaliza la captura"""
        self.capturing = False
        self.data_manager.save_data()
        self._update_gesture_list()
        self.capture_info.config(text=f"✅ '{self.current_gesture}' completado")
        messagebox.showinfo("Éxito", f"Se capturaron {self.samples_captured} muestras de '{self.current_gesture}'")
    
    def delete_selected(self):
        """Elimina el gesto seleccionado"""
        selection = self.gesture_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecciona una seña para eliminar")
            return
        
        item = self.gesture_listbox.get(selection[0])
        gesture = item.split(" ")[1]
        
        if messagebox.askyesno("Confirmar", f"¿Eliminar '{gesture}' y todas sus muestras?"):
            self.data_manager.delete_gesture(gesture)
            self._update_gesture_list()
            messagebox.showinfo("Éxito", f"'{gesture}' eliminado")
    
    def train_model(self):
        """Entrena el modelo"""
        gestures = self.data_manager.get_all_gestures()
        if len(gestures) < 2:
            messagebox.showwarning("Aviso", "Necesitas al menos 2 señas diferentes para entrenar")
            return
        
        # Verificar mínimo de muestras
        for gesture in gestures:
            count = self.data_manager.get_gesture_count(gesture)
            if count < MIN_SAMPLES_TO_TRAIN:
                messagebox.showwarning("Aviso", 
                    f"'{gesture}' tiene solo {count} muestras. Mínimo requerido: {MIN_SAMPLES_TO_TRAIN}")
                return
        
        self.train_btn.config(state=tk.DISABLED)
        
        def train_thread():
            X, y, labels = self.data_manager.get_training_data()
            
            def progress_callback(msg, progress):
                self.root.after(0, lambda: self._update_training_progress(msg, progress))
            
            success, message = ModelTrainer.train(X, y, labels, progress_callback)
            
            self.root.after(0, lambda: self._training_complete(success, message, labels))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def _update_training_progress(self, message, progress):
        """Actualiza progreso del entrenamiento"""
        self.progress_var.set(progress)
        self.progress_label.config(text=message)
    
    def _training_complete(self, success, message, labels):
        """Callback cuando termina el entrenamiento"""
        self.train_btn.config(state=tk.NORMAL)
        
        if success:
            self.model_info.config(text=f"✅ {message}\nSeñas: {', '.join(labels)}")
            messagebox.showinfo("¡Éxito!", f"Modelo entrenado correctamente.\n{message}")
        else:
            self.model_info.config(text=f"❌ Error: {message}")
            messagebox.showerror("Error", f"No se pudo entrenar el modelo:\n{message}")
    
    def on_close(self):
        """Cierra la aplicación"""
        self.running = False
        if self.camera:
            self.camera.release()
        self.data_manager.save_data()
        self.root.destroy()
    
    def run(self):
        """Ejecuta la aplicación"""
        self.root.mainloop()


# ════════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════════

def main():
    app = EntrenadorApp()
    app.run()


if __name__ == "__main__":
    main()
