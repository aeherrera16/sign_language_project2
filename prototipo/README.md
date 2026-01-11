# Prototipo LSE - Objetivo 1

## Técnicas Utilizadas

Basado en investigaciones:
- **Morfín-Chávez et al. (2023)**: MediaPipe 21 puntos, F1=0.98
- **Sincan & Keles (2020)**: CNN+LSTM, 95% precisión
- **Basnin et al. (2021)**: LSTM, 88.5% precisión
- **Paper Dialnet (2025)**: MediaPipe + ML, accuracy 0.817

### Arquitectura

```
Cámara (30 fps)
    ↓
MediaPipe Hands (21 landmarks × 3 coords = 63 features/mano)
    ↓
Normalización (coordenadas relativas a muñeca)
    ↓
Buffer de 30 frames = [30 × 126] features
    ↓
LSTM (64 → 128 → 64 unidades)
    ↓
Clasificación (softmax)
    ↓
Subtítulos + Voz (TTS)
```

## Uso

```bash
cd prototipo

# 1. Grabar señas (mínimo 30 secuencias por seña, mínimo 2 señas)
python 1_grabar_senas.py

# 2. Entrenar modelo LSTM
python 2_entrenar_modelo.py

# 3. Probar traductor
python 3_traductor.py

# 4. Evaluar métricas ISO/IEC 25023
python 4_evaluar_iso25023.py
```

## Controles

### Grabador (1_grabar_senas.py)
- **G**: Grabar secuencia (1 segundo)
- **Q**: Guardar y salir

### Traductor (3_traductor.py)
- **ESPACIO**: Convertir subtítulos a voz
- **C**: Limpiar subtítulos
- **Q**: Salir

## Métricas ISO/IEC 25023

El script `4_evaluar_iso25023.py` genera:
- **Exactitud Funcional**: % de señas correctamente clasificadas
- **Tiempo de Respuesta**: ms por predicción
- **Confianza**: Certeza del modelo en cada predicción

Reporte guardado en: `modelo/evaluacion_iso25023.json`

## Requisitos

```bash
pip install opencv-python mediapipe tensorflow pyttsx3 scikit-learn
```

## Vocabulario de Noticias

Señas recomendadas para grabar:
1. PRESIDENTE
2. GOBIERNO
3. PAÍS/ECUADOR
4. DECIR/ANUNCIAR
5. AÑO
6. POBREZA
7. TRABAJO
8. SUBIR/BAJAR
9. BUENO/MALO
10. HOY
