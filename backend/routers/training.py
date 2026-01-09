"""
Router para entrenamiento del modelo
CON DATA AUGMENTATION y CLASS BALANCING para máxima precisión
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
import json
from sklearn.utils.class_weight import compute_class_weight

router = APIRouter()

# Estado del entrenamiento
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "",
    "history": None
}

@router.post("/start")
async def start_training(background_tasks: BackgroundTasks):
    """Inicia el entrenamiento del modelo en segundo plano"""
    
    if training_status["is_training"]:
        # En lugar de error, devolvemos éxito indicando que ya está en proceso
        return {
            "success": True,
            "message": "Entrenamiento ya en curso (petición ignorada)",
            "is_training": True
        }
    
    # Verificar que hay datos
    gestures_dir = "data/gestures"
    if not os.path.exists(gestures_dir):
        raise HTTPException(
            status_code=400,
            detail="No hay datos de entrenamiento. Captura señas primero."
        )
    
    gestures = [d for d in os.listdir(gestures_dir) if os.path.isdir(os.path.join(gestures_dir, d))]
    gestures.sort()  # IMPORTANTE: Ordenar para garantizar índices consistentes
    
    if len(gestures) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Se necesitan al menos 2 señas diferentes. Tienes: {len(gestures)}"
        )
    
    # Iniciar entrenamiento en segundo plano
    background_tasks.add_task(train_model, gestures)
    
    return {
        "success": True,
        "message": "Entrenamiento iniciado",
        "num_gestures": len(gestures),
        "gestures": gestures
    }

def train_model(gestures):
    """Función de entrenamiento (se ejecuta en segundo plano)"""
    
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["message"] = "Cargando datos..."
    
    try:
        # 1. Cargar datos - SOPORTA TANTO ESTÁTICOS COMO DINÁMICOS
        X = []
        y = []
        
        gestures_dir = "data/gestures"
        
        for label, gesture in enumerate(gestures):
            gesture_path = os.path.join(gestures_dir, gesture)
            
            for file in os.listdir(gesture_path):
                # CASO 1: Landmarks estáticos (señas sin movimiento)
                if file.startswith("landmarks_") and file.endswith(".npy"):
                    landmarks = np.load(os.path.join(gesture_path, file))
                    
                    # Asegurar forma correcta
                    # El nuevo servicio guarda (num_hands, 21, 3) -> ndim=3
                    if landmarks.ndim == 3:
                        landmarks = landmarks.flatten()
                    elif landmarks.ndim == 2:
                        landmarks = landmarks.flatten()
                    
                    # Rellenar o truncar a 126 (2 manos * 21 puntos * 3 coords)
                    if landmarks.shape[0] < 126:
                        landmarks = np.pad(landmarks, (0, 126 - landmarks.shape[0]))
                    elif landmarks.shape[0] > 126:
                        landmarks = landmarks[:126]
                    
                    X.append(landmarks)
                    y.append(label)
                
                # CASO 2: Secuencias dinámicas (señas con movimiento)
                elif (file.startswith("sequence_") or file.startswith("conadis_")) and file.endswith(".npy"):
                    sequence = np.load(os.path.join(gesture_path, file))
                    
                    if sequence.ndim == 4:
                        num_frames = sequence.shape[0]
                        
                        # ESTRATEGIA MEJORADA: Usar TODOS los frames válidos para maximizar datos
                        # Esto es crucial cuando tenemos pocos videos vs. muchas fotos estáticas
                        
                        # 1. Usar todos los frames (sampleados cada 2 para variedad si son muchos)
                        step = 1 if num_frames < 100 else 2
                        
                        for idx in range(0, num_frames, step):
                            frame_landmarks = sequence[idx].flatten()
                            
                            # Validar que no sea un frame de ceros (si el tracker perdió la mano)
                            if np.max(np.abs(frame_landmarks)) < 0.01:
                                continue
                            
                            # Normalizar a 126 dimensiones
                            if frame_landmarks.shape[0] < 126:
                                frame_landmarks = np.pad(frame_landmarks, (0, 126 - frame_landmarks.shape[0]))
                            elif frame_landmarks.shape[0] > 126:
                                frame_landmarks = frame_landmarks[:126]
                            
                            X.append(frame_landmarks)
                            y.append(label)
                            
                        # 2. Agregar frames con flip horizontal (espejo) para duplicar datos
                        # Útil para generalizar mano derecha/izquierda
                        for idx in range(0, num_frames, step * 2):
                             frame_landmarks = sequence[idx].copy()
                             # Flip X coordinate (suponiendo normalización 0-1 o centrada)
                             # En MediaPipe x está en [0,1], 0=izq, 1=der. Flip es 1-x.
                             frame_landmarks[:, :, 0] = 1.0 - frame_landmarks[:, :, 0]
                             
                             flat_flipped = frame_landmarks.flatten()
                             if flat_flipped.shape[0] < 126:
                                 flat_flipped = np.pad(flat_flipped, (0, 126 - flat_flipped.shape[0]))
                             elif flat_flipped.shape[0] > 126:
                                 flat_flipped = flat_flipped[:126]
                                 
                             X.append(flat_flipped)
                             y.append(label)
        
        if len(X) == 0:
            raise ValueError("No se encontraron datos de landmarks")
        
        X = np.array(X)
        y = np.array(y)
        
        original_count = len(X)
        training_status["message"] = f"Datos cargados: {original_count} muestras originales"
        training_status["progress"] = 10
        
        # ═══════════════════════════════════════════════════════════════════════
        # 2. DATA AUGMENTATION - Multiplicar muestras para mejor precisión
        # ═══════════════════════════════════════════════════════════════════════
        training_status["message"] = "Usando datos aumentados de disco (x20)..."
        training_status["progress"] = 15
        
        # Ya tenemos x20 en disco, no necesitamos aumentar más en memoria para esta prueba
        X_augmented, y_augmented = augment_landmarks(X, y, multiplier=1)
        
        training_status["message"] = f"Datos aumentados: {original_count} → {len(X_augmented)} muestras"
        training_status["progress"] = 20
        
        # ═══════════════════════════════════════════════════════════════════════
        # 3. CALCULAR CLASS WEIGHTS para balancear clases desiguales
        # ═══════════════════════════════════════════════════════════════════════
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_augmented),
            y=y_augmented
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"📊 Class weights: {class_weight_dict}")
        
        # 4. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_augmented, y_augmented,
            test_size=0.2,
            random_state=42,
            stratify=y_augmented
        )
        
        training_status["message"] = "Creando modelo..."
        training_status["progress"] = 30
        
        # 5. Crear modelo MEJORADO
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(126,)),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(gestures), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        training_status["message"] = "Entrenando modelo con Data Augmentation..."
        training_status["progress"] = 40
        
        # 6. Entrenar CON CLASS WEIGHTS
        os.makedirs("model/checkpoints", exist_ok=True)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'model/checkpoints/model_{epoch:02d}.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,
            batch_size=64,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            class_weight=class_weight_dict,  # ← BALANCEO DE CLASES
            verbose=0
        )
        
        training_status["progress"] = 90
        training_status["message"] = "Guardando modelo..."
        
        # 7. Guardar modelo
        model.save("model/best_model.h5")
        
        # 8. Guardar labels
        with open("model/labels.pkl", "wb") as f:
            pickle.dump(gestures, f)
        
        # 9. Evaluación final
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # 10. Guardar historial enriquecido
        rich_history = {
            "metrics": {
                "loss": [float(x) for x in history.history['loss']],
                "accuracy": [float(x) for x in history.history['accuracy']],
                "val_loss": [float(x) for x in history.history['val_loss']],
                "val_accuracy": [float(x) for x in history.history['val_accuracy']],
            },
            "final_accuracy": float(test_acc),
            "final_loss": float(test_loss),
            "timestamp": datetime.now().isoformat(),
            "gestures": gestures,
            "num_samples_original": original_count,
            "num_samples_augmented": len(X_augmented),
            "augmentation_factor": len(X_augmented) // original_count,
            "epochs_trained": len(history.history['loss'])
        }
        
        with open("model/training_history.json", "w") as f:
            json.dump(rich_history, f)
        
        training_status["progress"] = 100
        training_status["history"] = {
            "final_accuracy": float(test_acc),
            "final_loss": float(test_loss),
            "epochs_trained": len(history.history['loss']),
            "num_samples_original": original_count,
            "num_samples_augmented": len(X_augmented),
            "num_gestures": len(gestures),
            "gestures": gestures,
            "timestamp": datetime.now().isoformat()
        }
        
        training_status["message"] = f"✅ Completado: {test_acc*100:.1f}% precisión ({len(X_augmented)} muestras)"
        
        # 11. Recargar modelo en el router de reconocimiento
        from . import recognition
        recognition.load_model()
        print("✓ Modelo recargado en memoria automáticamente")
        
    except Exception as e:
        training_status["message"] = f"Error: {str(e)}"
        training_status["progress"] = 0
        print(f"Error en entrenamiento: {e}")
    
    finally:
        training_status["is_training"] = False


def augment_landmarks(X: np.ndarray, y: np.ndarray, multiplier: int = 10) -> tuple:
    """
    Data Augmentation para landmarks de manos.
    
    Técnicas aplicadas:
    1. Ruido gaussiano pequeño
    2. Escalado (simula distancia a la cámara)
    3. Rotación 2D (simula inclinación de la mano)
    4. Traslación (simula movimiento)
    5. Simetría (flip horizontal)
    
    Args:
        X: Array de landmarks [n_samples, 126]
        y: Array de etiquetas
        multiplier: Factor de multiplicación (default 10x)
    
    Returns:
        X_aug, y_aug con muestras aumentadas
    """
    X_aug = [X]  # Incluir originales
    y_aug = [y]
    
    for i in range(multiplier - 1):
        X_new = X.copy()
        
        # Diferentes aumentaciones por iteración
        aug_type = i % 5
        
        if aug_type == 0:
            # 1. Ruido gaussiano (simula variación natural)
            noise = np.random.normal(0, 0.02, X_new.shape)
            X_new = X_new + noise
            
        elif aug_type == 1:
            # 2. Escalado (0.9 a 1.1)
            scale = np.random.uniform(0.9, 1.1)
            X_new = X_new * scale
            
        elif aug_type == 2:
            # 3. Traslación pequeña
            shift = np.random.uniform(-0.05, 0.05, (1, X_new.shape[1]))
            X_new = X_new + shift
            
        elif aug_type == 3:
            # 4. Rotación 2D pequeña (-15° a +15°)
            angle = np.random.uniform(-0.26, 0.26)  # radianes
            X_new = rotate_landmarks(X_new, angle)
            
        else:
            # 5. Combinación de varios
            noise = np.random.normal(0, 0.01, X_new.shape)
            scale = np.random.uniform(0.95, 1.05)
            X_new = (X_new + noise) * scale
        
        X_aug.append(X_new)
        y_aug.append(y)
    
    return np.vstack(X_aug), np.concatenate(y_aug)


def rotate_landmarks(X: np.ndarray, angle: float) -> np.ndarray:
    """
    Rota landmarks 2D en el plano XY.
    
    Args:
        X: Landmarks [n_samples, 126]
        angle: Ángulo en radianes
    
    Returns:
        Landmarks rotados
    """
    X_rot = X.copy()
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Cada muestra tiene 126 = 2 manos * 21 puntos * 3 coords (x, y, z)
    # Rotamos solo x, y (índices 0, 1 de cada punto)
    for sample_idx in range(X_rot.shape[0]):
        for point in range(42):  # 21 puntos * 2 manos
            idx = point * 3
            if idx + 1 < X_rot.shape[1]:
                x = X_rot[sample_idx, idx]
                y = X_rot[sample_idx, idx + 1]
                X_rot[sample_idx, idx] = x * cos_a - y * sin_a
                X_rot[sample_idx, idx + 1] = x * sin_a + y * cos_a
    
    return X_rot


@router.get("/status")
async def get_training_status():
    """Obtiene el estado actual del entrenamiento"""
    return training_status

@router.get("/history")
async def get_training_history():
    """Obtiene el historial del último entrenamiento"""
    history_path = "model/training_history.json"
    
    if not os.path.exists(history_path):
        raise HTTPException(
            status_code=404,
            detail="No hay historial de entrenamiento"
        )
    
    with open(history_path, "r") as f:
        history = json.load(f)
    
    return history
