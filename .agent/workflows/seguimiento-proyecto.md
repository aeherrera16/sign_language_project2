---
description: Plan de seguimiento del proyecto de Traductor LSE a Voz
---

# 📋 Plan de Seguimiento - Traductor LSE a Voz

## 🎯 Objetivo General
Desarrollar un prototipo funcional y posteriormente un prototipo wearable capaz de reconocer y traducir señas de la Lengua de Señas Ecuatoriana (LSE) a voz en español, orientado a la traducción de noticias de la prensa escrita.

---

## 📊 Estado Actual del Proyecto

### ✅ COMPLETADO
- [x] Configuración del entorno de desarrollo (Python, TensorFlow, MediaPipe)
- [x] Prototipo funcional en computadora funcionando
- [x] Módulo de captura de señas implementado
- [x] Modelo de clasificación entrenado
- [x] Interfaz web con React (TraductorIA, GrabarSenia, Dashboard)
- [x] Adaptación inicial para Raspberry Pi 4

### 🔄 EN PROGRESO
- [ ] Scraping de señas de CONADIS (herramienta creada: tools/scrape_conadis.py)
- [ ] Subtítulos en tiempo real
- [ ] Evaluación ISO/IEC 25023 (precisión y tiempo de respuesta)

### ⏳ PENDIENTE
- [ ] Integración con Raspberry Pi 5
- [ ] Prototipo wearable completo
- [ ] Conjunto de datos validado de noticias
- [ ] Evaluación formal ISO/IEC 25040
- [ ] Informe técnico final

---

## 🎯 OBJETIVO ESPECÍFICO 1: Prototipo Funcional en Computadora
**Estado: 80% Completado**

### Actividad 1.1: Configurar entorno de desarrollo ✅
- Python, TensorFlow, MediaPipe configurados
- Cámara HD integrada

### Actividad 1.2: Visualización de subtítulos 🔄
- **Estado**: Parcialmente implementado
- **Pendiente**: Mejorar visualización de texto en tiempo real antes de TTS

### Actividad 1.3: Evaluar precisión y tiempo de respuesta ⏳
- **Pendiente**: Implementar métricas ISO/IEC 25023
- **Métricas requeridas**:
  - Precisión de clasificación (%)
  - Tiempo de respuesta (ms)
  - Tasa de error

---

## 🎯 OBJETIVO ESPECÍFICO 2: Prototipo Wearable (Raspberry Pi)
**Estado: 40% Completado**

### Actividad 2.1: Integrar hardware ⏳
- [ ] Raspberry Pi 5 (actualmente probado con RPi 4)
- [ ] Cámara corporal magnética
- [ ] Parlante wearable
- [ ] Estructura física portátil
- [ ] Power Bank

### Actividad 2.2: Adaptar software para Raspberry Pi 🔄
- **Estado**: Scripts creados para RPi 4
- **Archivos**:
  - `raspberry_pi/traductor_portable.py`
  - `raspberry_pi/traductor_autostart.py`
  - `raspberry_pi/INSTALAR_EN_RASPBERRY.sh`
- **Pendiente**: Optimización para RPi 5, pruebas de TFLite

### Actividad 2.3: Pruebas técnicas básicas ⏳
- [ ] Captura funcionando
- [ ] Procesamiento en tiempo real
- [ ] Generación de voz

---

## 🎯 OBJETIVO ESPECÍFICO 3: Evaluación ISO/IEC 25023 y 25040
**Estado: 10% Completado**

### Actividad 3.1: Métricas de precisión y tiempo ⏳
- **ISO/IEC 25023** define métricas de calidad:
  - Exactitud funcional
  - Tiempo de respuesta
  - Uso de recursos

### Actividad 3.2: Documentar proceso de evaluación ⏳
- **ISO/IEC 25040** define el proceso:
  1. Establecer requisitos de evaluación
  2. Especificar la evaluación
  3. Diseñar la evaluación
  4. Ejecutar la evaluación
  5. Concluir la evaluación

---

## 🛠️ PRODUCTOS ACREDITABLES

| Producto | Estado | Entregable |
|----------|--------|------------|
| Prototipo funcional PC | 🟡 80% | Aplicación web + backend |
| Prototipo wearable RPi | 🟡 40% | Scripts de despliegue |
| Modelos ML entrenados | ✅ 100% | `best_model.h5`, `model.tflite` |
| Conjunto de datos | 🟡 60% | `backend/data/gestures/` |
| Resultados evaluación | 🔴 10% | Pendiente métricas ISO |
| Informe técnico | 🔴 0% | Pendiente redacción |

---

## 📅 PRÓXIMAS ACCIONES PRIORITARIAS

1. **Completar subtítulos en tiempo real** (Actividad 1.2)
2. **Implementar métricas ISO/IEC 25023** (Actividad 1.3)
3. **Ampliar dataset con señas de CONADIS** (scrape_conadis.py)
4. **Probar en Raspberry Pi 5** (Actividad 2.2)
5. **Documentar evaluación ISO/IEC 25040** (Actividad 3.2)

---

## 📁 Archivos Clave del Proyecto

- `backend/` - API FastAPI para reconocimiento
- `frontend/react_app/` - Interfaz web React
- `raspberry_pi/` - Scripts para despliegue wearable
- `tools/scrape_conadis.py` - Scraper de señas oficiales
- `backend/model/` - Modelos entrenados
