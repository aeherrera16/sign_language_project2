---
description: Plan de seguimiento del proyecto de Traductor LSE a Voz
---

# 📋 Plan de Seguimiento - Traductor LSE a Voz

## 🎯 Objetivo General
Desarrollar un prototipo funcional capaz de reconocer y traducir señas de la Lengua de Señas Ecuatoriana (LSE) a voz en español, orientado a la traducción de noticias de la prensa escrita, mediante técnicas de visión por computadora y aprendizaje automático.

---

## 📊 Estado Actual - Objetivo 1 (Prototipo Funcional PC)

### ✅ COMPLETADO
- [x] Entorno de desarrollo (Python, TensorFlow, MediaPipe)
- [x] Captura de movimientos con MediaPipe (21 landmarks/mano)
- [x] Modelo LSTM para señas dinámicas (secuencias temporales)
- [x] Subtítulos en tiempo real
- [x] Salida de voz en español (TTS pyttsx3)
- [x] Funciona offline sin Internet

### 🔄 EN PROGRESO
- [ ] Vocabulario de señas para noticias (grabar muestras)
- [ ] Métricas ISO/IEC 25023 (precisión, tiempo de respuesta)

### ⏳ PENDIENTE (Objetivos 2 y 3)
- [ ] Prototipo wearable Raspberry Pi 5
- [ ] Evaluación formal ISO/IEC 25040
- [ ] Informe técnico final

---

## 🎯 OBJETIVO ESPECÍFICO 1: Prototipo Funcional en Computadora

### Actividad 1.1: Configurar entorno ✅
- Python 3.10, TensorFlow, MediaPipe, OpenCV
- Cámara HD integrada

### Actividad 1.2: Visualización de subtítulos ✅
- Subtítulos en pantalla en tiempo real
- Texto acumulado de señas detectadas

### Actividad 1.3: Métricas ISO/IEC 25023 🔄
- **Precisión**: Calculada en entrenamiento (classification_report)
- **Tiempo de respuesta**: Medido en cada detección (ms)
- **Confianza**: Umbral mínimo 70%

---

## 📐 Técnica Implementada

### Arquitectura: MediaPipe + LSTM
```
Cámara → MediaPipe (21 landmarks) → Secuencia 30 frames → LSTM → Clasificación → Subtítulos + Voz
```

### Fundamento científico:
| Paper | Técnica | Precisión |
|-------|---------|-----------|
| Sincan & Keles (2020) | CNN+LSTM | 95% |
| Morfín-Chávez (2023) | MediaPipe | F1=0.98 |
| IJCA (2024) | MediaPipe+LSTM | 99.4% |

---

## 📝 Vocabulario de Noticias (15 señas iniciales)

Para cumplir con "orientado a noticias de prensa escrita":

| # | Seña | Contexto de noticia |
|---|------|---------------------|
| 1 | PRESIDENTE | "El presidente anunció..." |
| 2 | GOBIERNO | "El gobierno informó..." |
| 3 | PAÍS | "El país registró..." |
| 4 | ECUADOR | Nombre propio |
| 5 | DECIR | "Dijo que..." |
| 6 | ANUNCIAR | "Anunció que..." |
| 7 | AÑO | "Este año..." |
| 8 | DINERO | "La economía..." |
| 9 | POBREZA | "La pobreza bajó..." |
| 10 | TRABAJO | "El empleo..." |
| 11 | SUBIR | "Aumentó..." |
| 12 | BAJAR | "Redujo..." |
| 13 | BUENO | "Resultados positivos..." |
| 14 | MALO | "Crisis..." |
| 15 | HOY | "Hoy se informó..." |

---

## 📁 Estructura del Prototipo

```
prototipo/
├── 1_grabar_senas.py      # Captura secuencias de 30 frames
├── 2_entrenar_modelo.py   # Entrena LSTM + métricas
├── 3_traductor.py         # Reconocimiento + subtítulos + voz
├── datos/                 # Secuencias grabadas (JSON)
├── modelo/                # Modelo entrenado (.h5)
└── README.md              # Documentación
```

---

## 🚀 PRÓXIMOS PASOS

1. **Grabar señas de noticias** (mínimo 5 señas, 30 secuencias c/u)
2. **Entrenar modelo** y verificar precisión
3. **Probar traducción** de una frase de noticia
4. **Documentar métricas** ISO/IEC 25023
