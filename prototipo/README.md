# Prototipo Funcional LSE - Objetivo 1

## 📋 Descripción
Sistema simple para traducir Lengua de Señas Ecuatoriana (LSE) a texto (subtítulos) y voz.
Orientado a la traducción de noticias de prensa escrita.

## 🗂️ Estructura

```
prototipo/
├── 1_grabar_senas.py      # Paso 1: Grabar datos de entrenamiento
├── 2_entrenar_modelo.py   # Paso 2: Entrenar modelo LSTM
├── 3_traductor.py         # Paso 3: Prototipo con subtítulos + voz
├── datos/                 # Secuencias grabadas (se genera automáticamente)
└── modelo/                # Modelo entrenado (se genera automáticamente)
```

## 🚀 Uso Rápido

### Paso 1: Grabar señas
```bash
cd prototipo
python 1_grabar_senas.py
```
- Escribe el nombre de la seña (ej: PRESIDENTE)
- Presiona `G` para grabar una secuencia (1 segundo)
- Repite 30-50 veces por seña
- Presiona `Q` para guardar y salir
- Repite para cada seña del vocabulario

### Paso 2: Entrenar el modelo
```bash
python 2_entrenar_modelo.py
```
- Requiere al menos 2 señas grabadas
- Entrena automáticamente un modelo LSTM

### Paso 3: Usar el traductor
```bash
python 3_traductor.py
```
- Haz señas frente a la cámara
- Los subtítulos se acumulan en pantalla
- Presiona `ESPACIO` para convertir a voz
- Presiona `C` para limpiar subtítulos

## 📝 Vocabulario de Noticias (15 señas iniciales)

| # | Seña | Descripción |
|---|------|-------------|
| 1 | PRESIDENTE | Líder del país |
| 2 | GOBIERNO | Administración estatal |
| 3 | PAIS | Nación |
| 4 | ECUADOR | País específico |
| 5 | DECIR | Comunicar/anunciar |
| 6 | ANUNCIAR | Declarar públicamente |
| 7 | AÑO | Periodo de tiempo |
| 8 | DINERO | Economía/finanzas |
| 9 | POBREZA | Situación económica |
| 10 | TRABAJO | Empleo |
| 11 | SUBIR | Aumentar |
| 12 | BAJAR | Reducir |
| 13 | BUENO | Positivo |
| 14 | MALO | Negativo |
| 15 | HOY | Actualidad |

## 🔧 Requisitos

```bash
pip install opencv-python mediapipe tensorflow pyttsx3 numpy scikit-learn
```

## 📊 Métricas (ISO/IEC 25023)

El sistema mide automáticamente:
- **Precisión**: % de señas correctamente identificadas
- **Tiempo de respuesta**: Tiempo desde seña hasta detección
- **Confianza**: Umbral mínimo de 70% para aceptar una seña

## 🎯 Ejemplo de Noticia

Para traducir: "El presidente anunció reducción de pobreza"

1. Haz la seña PRESIDENTE → aparece "PRESIDENTE" en subtítulos
2. Haz la seña ANUNCIAR → aparece "PRESIDENTE ANUNCIAR"
3. Haz la seña BAJAR → aparece "PRESIDENTE ANUNCIAR BAJAR"
4. Haz la seña POBREZA → aparece "PRESIDENTE ANUNCIAR BAJAR POBREZA"
5. Presiona ESPACIO → el sistema habla la frase
