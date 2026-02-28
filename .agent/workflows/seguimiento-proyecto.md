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
- [x] **Sistema de filtrado inteligente de manos**
  - [x] Zona activa de señas (ROI) - ignora manos fuera del área
  - [x] Filtro de posición natural - descarta manos en reposo
  - [x] Selección de señante principal - prioriza manos más centradas
  - [x] Feedback visual (verde=válida, rojo=ignorada)
- [x] **Consistencia grabador-traductor**: ambos usan los mismos filtros

### 🔄 EN PROGRESO
- [ ] Vocabulario de señas para noticias (grabar más muestras)
- [ ] Métricas ISO/IEC 25023 (precisión, tiempo de respuesta)
- [ ] Validar filtrado con escenarios de múltiples personas

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

### Actividad 1.3: Sistema de filtrado inteligente ✅
- **Zona activa (ROI)**: Solo procesa manos dentro del área de señas
- **Posición natural**: Descarta manos caídas/en reposo
- **Señante principal**: Si hay múltiples personas, prioriza la más centrada

### Actividad 1.4: Métricas ISO/IEC 25023 🔄
- **Precisión**: Calculada en entrenamiento (classification_report)
- **Tiempo de respuesta**: Medido en cada detección (ms)
- **Confianza**: Umbral mínimo 92%

---

## 📐 Técnica Implementada

### Arquitectura: MediaPipe + Filtrado + LSTM
```
Cámara → MediaPipe → Filtrado (ROI + reposo + centro) → Secuencia 30 frames → LSTM → Clasificación → Subtítulos + Voz
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

| # | Seña | Contexto de noticia | Estado |
|---|------|---------------------|--------|
| 1 | PRESIDENTE | "El presidente anunció..." | ✅ Grabada |
| 2 | GOBIERNO | "El gobierno informó..." | ⏳ Pendiente |
| 3 | PAÍS | "El país registró..." | ⏳ Pendiente |
| 4 | ECUADOR | Nombre propio | ✅ Grabada |
| 5 | DECIR | "Dijo que..." | ⏳ Pendiente |
| 6 | ANUNCIAR | "Anunció que..." | ⏳ Pendiente |
| 7 | AÑO/ANÑOS | "Este año..." | ✅ Grabada |
| 8 | DINERO | "La economía..." | ⏳ Pendiente |
| 9 | POBREZA | "La pobreza bajó..." | ⏳ Pendiente |
| 10 | TRABAJO | "El empleo..." | ⏳ Pendiente |
| 11 | SUBIR | "Aumentó..." | ⏳ Pendiente |
| 12 | BAJAR | "Redujo..." | ⏳ Pendiente |
| 13 | BUENO | "Resultados positivos..." | ⏳ Pendiente |
| 14 | MALO | "Crisis..." | ⏳ Pendiente |
| 15 | HOY | "Hoy se informó..." | ⏳ Pendiente |

**Señas ya grabadas adicionales**: TIENE, 2, 8

---

## 📁 Estructura del Prototipo

```
prototipo/
├── 1_grabar_senas.py      # Grabador con filtrado inteligente
├── 2_entrenar_modelo.py   # Entrena LSTM + métricas
├── 3_traductor.py         # Traductor con filtrado + subtítulos + voz
├── 4_evaluar_iso25023.py  # Evaluación ISO
├── app.py                 # Menú principal
├── datos/                 # Secuencias grabadas (JSON)
└── modelo/                # Modelo entrenado (.h5)
```

---

## 🚀 PRÓXIMOS PASOS

1. **Probar el filtrado** con múltiples personas frente a la cámara
2. **Regrabar señas** existentes usando el nuevo grabador con filtrado
3. **Grabar más señas** del vocabulario de noticias (al menos 10 señas)
4. **Reentrenar modelo** con datos limpios
5. **Probar traducción** de frases completas de noticias
6. **Documentar métricas** ISO/IEC 25023

---

## 🛡️ Problemas Resueltos

### 1. Múltiples manos detectadas (otra persona)
**Problema**: Cuando había un compañero haciendo señas, el sistema procesaba todas las manos sin distinción.
**Solución**: Sistema de filtrado que selecciona solo las 2 manos más cercanas al centro de la cámara.

### 2. Manos en posición natural
**Problema**: Las manos caídas al costado del cuerpo generaban detecciones falsas.
**Solución**: Filtro que analiza posición Y + extensión de dedos para descartar manos en reposo.

### 3. Manos fuera del área de señas
**Problema**: Movimientos casuales en cualquier parte del frame activaban el sistema.
**Solución**: Zona activa (ROI) que solo procesa manos dentro del área esperada de señas.
