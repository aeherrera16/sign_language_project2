import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import json

def main():
    # Cargar datos
    X = []
    y = []

    if not os.path.exists("data"):
        raise FileNotFoundError("La carpeta 'data' no existe. Graba gestos primero.")

    gestures = sorted(os.listdir("data"))
    if len(gestures) == 0:
        raise ValueError("No hay gestos en la carpeta 'data'. Graba gestos primero.")

    for label, gesture in enumerate(gestures):
        gesture_path = os.path.join("data", gesture)
        for file in os.listdir(gesture_path):
            if file.endswith(".npy"):
                path = os.path.join(gesture_path, file)
                try:
                    sample = np.load(path)
                    if sample.shape != (1530,):
                        print(f"Forma inválida en {file}, se obtuvo {sample.shape}")
                        continue

                    X.append(sample)
                    y.append(label)
                except Exception as e:
                    print(f"Error al procesar {path}: {str(e)}")

    if len(X) == 0:
        raise ValueError("No se encontraron muestras válidas para entrenamiento.")

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Crear carpeta de modelo
    os.makedirs("model", exist_ok=True)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(1530,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(gestures), activation='softmax')
    ])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        'model/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Guardar modelo final
    model.save("model/gesture_model.h5")

    # Guardar etiquetas
    with open("model/labels.pkl", "wb") as f:
        pickle.dump(gestures, f)

    # Guardar historial
    with open("model/training_history.json", "w") as f:
        json.dump(history.history, f)

    # Evaluación final
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrecisión final en test: {test_acc:.4f}")
    print(f"Gestos entrenados: {gestures}")

    # Reporte de clasificación
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=gestures))

if __name__ == "__main__":
    main()
