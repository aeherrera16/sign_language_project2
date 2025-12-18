"""
Router para entrenamiento del modelo
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
import json

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
        raise HTTPException(
            status_code=400,
            detail="Ya hay un entrenamiento en curso"
        )
    
    # Verificar que hay datos
    gestures_dir = "data/gestures"
    if not os.path.exists(gestures_dir):
        raise HTTPException(
            status_code=400,
            detail="No hay datos de entrenamiento. Captura señas primero."
        )
    
    gestures = [d for d in os.listdir(gestures_dir) if os.path.isdir(os.path.join(gestures_dir, d))]
    
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

async def train_model(gestures):
    """Función de entrenamiento (se ejecuta en segundo plano)"""
    
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["message"] = "Cargando datos..."
    
    try:
        # 1. Cargar datos
        X = []
        y = []
        
        gestures_dir = "data/gestures"
        
        for label, gesture in enumerate(gestures):
            gesture_path = os.path.join(gestures_dir, gesture)
            
            for file in os.listdir(gesture_path):
                if file.startswith("landmarks_") and file.endswith(".npy"):
                    landmarks = np.load(os.path.join(gesture_path, file))
                    
                    # Asegurar forma correcta
                    if landmarks.ndim == 2:  # Si hay múltiples manos
                        landmarks = landmarks.flatten()
                    
                    # Rellenar o truncar a 126 (2 manos * 21 puntos * 3 coords)
                    if landmarks.shape[0] < 126:
                        landmarks = np.pad(landmarks, (0, 126 - landmarks.shape[0]))
                    elif landmarks.shape[0] > 126:
                        landmarks = landmarks[:126]
                    
                    X.append(landmarks)
                    y.append(label)
        
        if len(X) == 0:
            raise ValueError("No se encontraron datos de landmarks")
        
        X = np.array(X)
        y = np.array(y)
        
        training_status["message"] = f"Datos cargados: {len(X)} muestras"
        training_status["progress"] = 20
        
        # 2. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        training_status["message"] = "Creando modelo..."
        training_status["progress"] = 30
        
        # 3. Crear modelo
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(126,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(gestures), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        training_status["message"] = "Entrenando modelo..."
        training_status["progress"] = 40
        
        # 4. Entrenar
        os.makedirs("model/checkpoints", exist_ok=True)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'model/checkpoints/model_{epoch:02d}.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[checkpoint, early_stopping],
            verbose=0
        )
        
        training_status["progress"] = 90
        training_status["message"] = "Guardando modelo..."
        
        # 5. Guardar modelo
        model.save("model/best_model.h5")
        
        # 6. Guardar labels
        with open("model/labels.pkl", "wb") as f:
            pickle.dump(gestures, f)
        
        # 7. Guardar historial
        history_dict = {
            "loss": [float(x) for x in history.history['loss']],
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_loss": [float(x) for x in history.history['val_loss']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']],
        }
        
        with open("model/training_history.json", "w") as f:
            json.dump(history_dict, f)
        
        # 8. Evaluación final
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        training_status["progress"] = 100
        training_status["message"] = "¡Entrenamiento completado!"
        training_status["history"] = {
            "final_accuracy": float(test_acc),
            "final_loss": float(test_loss),
            "epochs_trained": len(history.history['loss']),
            "num_samples": len(X),
            "num_gestures": len(gestures),
            "gestures": gestures,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        training_status["message"] = f"Error: {str(e)}"
        training_status["progress"] = 0
        print(f"Error en entrenamiento: {e}")
    
    finally:
        training_status["is_training"] = False

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
