# Prototipo Funcional LSE - Objetivo 1

Traductor de Lengua de Señas Ecuatoriana a texto y voz.

## Estructura

```
prototipo/
├── 1_grabar_senas.py     # Grabar datos
├── 2_entrenar_modelo.py  # Entrenar LSTM
├── 3_traductor.py        # Traductor con subtítulos
├── datos/                # Secuencias grabadas
└── modelo/               # Modelo entrenado
```

## Uso

### 1. Grabar señas
```bash
python 1_grabar_senas.py
```
- Escribe nombre de la seña
- Presiona **G** para grabar (1 segundo por secuencia)
- Graba 30-50 secuencias por seña
- Presiona **Q** para guardar

### 2. Entrenar
```bash
python 2_entrenar_modelo.py
```

### 3. Traducir
```bash
python 3_traductor.py
```
- Haz señas → aparecen subtítulos
- **ESPACIO** → convierte a voz
- **C** → limpia subtítulos
- **Q** → salir

## Requisitos
```bash
pip install opencv-python mediapipe tensorflow pyttsx3
```
