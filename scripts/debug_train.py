import os
import numpy as np
import json
import pickle
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def debug_train():
    print("Iniciando debug entrenamiento...")
    DATA_DIR = "data/gestures"
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} no existe")
        return

    gestures = []
    X_data = []
    y_data = []

    # 1. Cargar datos
    print("Cargando datos...")
    for gesture_name in os.listdir(DATA_DIR):
        gesture_path = os.path.join(DATA_DIR, gesture_name)
        if not os.path.isdir(gesture_path):
            continue
            
        print(f"Procesando {gesture_name}...")
        npy_files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
        
        if not npy_files:
            print(f"  Advertencia: {gesture_name} no tiene archivos .npy")
            continue
            
        gestures.append(gesture_name)
        label_id = len(gestures) - 1
        
        count = 0
        for npy_file in npy_files:
            try:
                landmarks = np.load(os.path.join(gesture_path, npy_file))
                
                # Aplanar si es 3D (num_hands, 21, 3) -> (63,)
                if landmarks.ndim == 3:
                    # Tomar la primera mano detectada o aplanar todo?
                    # El modelo espera 126 features (2 manos * 21 * 3)? Depende.
                    # Vamos a asumir aplanado.
                    landmarks = landmarks.flatten()
                
                # Ajustar a 126 features (rellenar o cortar)
                if landmarks.shape[0] < 126:
                    landmarks = np.pad(landmarks, (0, 126 - landmarks.shape[0]))
                elif landmarks.shape[0] > 126:
                    landmarks = landmarks[:126]
                    
                X_data.append(landmarks)
                y_data.append(label_id)
                count += 1
            except Exception as e:
                print(f"  Error cargando {npy_file}: {e}")
        print(f"  Cargadas {count} muestras para {gesture_name}")

    if len(gestures) < 2:
        print("Error: Se necesitan al menos 2 gestos")
        return

    X = np.array(X_data)
    y = np.array(y_data)
    
    # Reshape para LSTM: (samples, time_steps, features)
    # Aquí tratamos cada captura estática como una secuencia de 1 paso?
    # O el modelo espera series temporales?
    # Si es captura estática, LSTM no es ideal, pero si así está definido...
    # En training.py original: X = X.reshape(X.shape[0], 1, X.shape[1])
    
    X = X.reshape(X.shape[0], 1, X.shape[1])
    y = to_categorical(y).astype(int)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Modelo
    print("Compilando modelo...")
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(1, 126)),
        LSTM(128, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(gestures), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Entrenando...")
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)
    
    print("Guardando archivos...")
    os.makedirs("model", exist_ok=True)
    model.save("model/best_model.h5")
    
    rich_history = {
        "metrics": {
            "accuracy": [float(x) for x in history.history['accuracy']],
            "loss": [float(x) for x in history.history['loss']]
        },
        "final_accuracy": float(history.history['accuracy'][-1]),
        "timestamp": datetime.now().isoformat(),
        "gestures": gestures
    }
    
    with open("model/training_history.json", "w") as f:
        json.dump(rich_history, f)
        
    print("¡Éxito! Archivos guardados en model/")

if __name__ == "__main__":
    debug_train()
