# 🚀 Sistema de Reconocimiento de Lenguaje de Señas

## ⚡ **INICIO RÁPIDO**
```powershell
# 1. Abrir PowerShell en la carpeta del proyecto
cd "c:\Users\Anahy\Desktop\sign_language_project2Modf"

# 2. Activar entorno virtual
.\venv310\Scripts\activate

# 3. Ejecutar la aplicación
python main_interface.py
```
**🎯 ¡Listo! La interfaz se abrirá y podrás usar el reconocimiento de señas**

---

## 📋 Descripción del Proyecto

Sistema avanzado de reconocimiento de lenguaje de señas ecuatoriano utilizando MediaPipe y TensorFlow. El proyecto permite:

- 📹 **Reconocimiento en tiempo real** de 205 gestos diferentes
- 🎯 **98.99% de precisión** en el modelo entrenado
- 🖥️ **Interfaz gráfica moderna** con organización por secciones
- 📊 **Evaluación comprehensiva** con métricas detalladas
- 🔊 **Síntesis de voz** para mejorar la accesibilidad

## � Métricas del Modelo

### Rendimiento Final:
- **Precisión (Accuracy)**: 98.99%
- **Top-5 Precisión**: 99.8%
- **Clases reconocidas**: 205 gestos
- **Muestras de entrenamiento**: 25,798
- **Muestras de prueba**: 6,450

### Características Técnicas:
- **Puntos de landmark**: 1,530 (126 manos + 1,404 cara)
- **Arquitectura**: Red neuronal profunda con BatchNormalization
- **Regularización**: Dropout y ReduceLROnPlateau
- **Aumento de datos**: Técnicas avanzadas aplicadas

## 📁 Estructura del Proyecto

```
sign_language_project2 - mod/
├── 📱 main_interface.py          # Interfaz principal con GUI moderna
├── 🧠 train_model.py             # Entrenamiento del modelo mejorado
├── 📊 evaluate_model.py          # Evaluación comprehensiva
├── 🎥 real_time_improved.py      # Reconocimiento en tiempo real
├── 🔧 utils.py                   # Funciones utilitarias
├── 📈 analyze_dataset.py         # Análisis del dataset
├── ⚙️ setup_project.py           # Configuración automática
├── 🧪 test_imports_improved.py   # Pruebas de dependencias
├── 📋 requirements.txt           # Dependencias
├── data/                         # Dataset de gestos (205 clases)
├── model/                        # Modelos entrenados y auxiliares
│   ├── gesture_model.h5          # Modelo principal (98.99% accuracy)
│   ├── labels.pkl                # Etiquetas de gestos
│   ├── scaler.pkl                # Normalizador de datos
│   └── training_history.json     # Historial de entrenamiento
└── evaluation/                   # Reportes y visualizaciones
    ├── confusion_matrix.png      # Matriz de confusión
    ├── classification_report.txt # Reporte detallado
    ├── roc_curves.png           # Curvas ROC
    ├── per_class_metrics.csv    # Métricas por clase
    └── evaluation_summary.json  # Resumen de evaluación
```

## 🛠️ Instalación y Configuración

### 📋 **REQUISITOS PREVIOS:**
- Python 3.8 o superior
- Cámara web funcional
- 4GB de RAM mínimo
- 2GB de espacio libre en disco

### 🚀 **MÉTODO 1: Configuración Automática (Recomendado)**
```powershell
# 1. Abrir PowerShell en la carpeta del proyecto
cd "ruta\del\proyecto\sign_language_project2Modf"

# 2. Ejecutar configuración automática
python setup_project.py
```

### ⚙️ **MÉTODO 2: Instalación Manual**
```powershell
# 1. Navegar al proyecto
cd "c:\Users\Anahy\Desktop\sign_language_project2Modf"

# 2. Crear entorno virtual
python -m venv venv310

# 3. Activar entorno virtual (Windows)
.\venv310\Scripts\activate

# 4. Actualizar pip
python -m pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt

# 6. Verificar instalación
python test_imports_improved.py
```

## 🎮 **CÓMO EJECUTAR EL PROYECTO**

### 🏃‍♂️ **EJECUCIÓN RÁPIDA**
```powershell
# 1. Activar entorno virtual
.\venv310\Scripts\activate

# 2. Ejecutar interfaz principal
python main_interface.py
```

### 🖥️ **Interfaz Principal Completa**
```powershell
python main_interface.py
```
**📱 Funciones disponibles:**
- 📁 **Gestión de Datos**: Grabar nuevos gestos y analizar dataset
- 🧠 **Entrenamiento**: Entrenar modelos con nuevos datos
- 📊 **Evaluación**: Analizar rendimiento del modelo actual
- 🎥 **Reconocimiento**: Probar reconocimiento en tiempo real

### 🎥 **Reconocimiento Directo (Modo Rápido)**
```powershell
# Reconocimiento mejorado con confianza
python real_time_improved.py

# Reconocimiento con síntesis de voz
python real_time_translate.py
```

### 📊 **Otras Funciones Útiles**
```powershell
# Analizar dataset existente
python analyze_dataset.py

# Evaluar modelo entrenado
python evaluate_model.py

# Grabar nuevos gestos
python record_dataset.py "nombre_del_gesto"

# Entrenar modelo con datos actuales
python train_model.py
```

### 🔧 **Resolución de Problemas**
```powershell
# Verificar que todo funciona correctamente
python test_imports_improved.py

# Si hay problemas con la cámara
# Verificar que no esté siendo usada por otra aplicación
```

### 💡 **TIPS DE USO:**
- 🎯 Asegúrate de tener buena iluminación
- 📏 Mantén las manos dentro del marco de la cámara
- ⚡ El reconocimiento funciona mejor con fondo contrastante
- 🔊 Para síntesis de voz, asegúrate de tener altavoces/audífonos

### 📊 Evaluación del Modelo
```bash
python evaluate_model.py
```

**Genera:**
- Matriz de confusión
- Curvas ROC
- Métricas por clase
- Análisis de errores

## � Arquitectura del Modelo

### Red Neuronal Profunda:
```
Input (1530 features)
    ↓
BatchNormalization
    ↓
Dense(512) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.4)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.5)
    ↓
Dense(64) + ReLU + Dropout(0.5)
    ↓
Dense(205) + Softmax
```

### Características de los Datos:
- **Puntos de mano**: 21 landmarks × 3 coordenadas × 2 manos = 126 features
- **Puntos faciales**: 468 landmarks × 3 coordenadas = 1,404 features
- **Total**: 1,530 features por muestra

## 📈 Resultados y Métricas

### Rendimiento General:
- **Accuracy**: 98.99%
- **Precision**: 99.0%
- **Recall**: 99.0%
- **F1-Score**: 99.0%

### Clases Mejor Reconocidas (100% accuracy):
- Números: 1, 2, 3, 10, 100, 1000, etc.
- Palabras básicas: "amigo", "familia", "nombre", "telefono"
- Saludos: "Buenas noches", "Gracias", "De nada"
- Colores: "Amarillo", "Verde", "Negro"

## 🎯 Gestos Reconocidos (205 clases)

### Números:
- Dígitos: 1-9
- Decenas: 10, 20, 30, ..., 90
- Centenas: 100, 200, 300, ..., 900
- Miles: 1000, 2000, ..., 9000
- Especiales: 10000, 1000000

### Alfabeto:
- Letras: a-z, ñ, ch, ll, rr

### Vocabulario Social:
- **Saludos**: "Buenos dias", "Buenas tardes", "Buenas noches", "chao"
- **Cortesía**: "Gracias", "De nada", "Disculpar", "Mucho Gusto"
- **Familia**: "papá", "mamá", "hermano", "hermana", "abuelo", "abuela"

### Información Personal:
- **Datos**: "nombre", "apellido", "cédula", "telefono", "email"
- **Estados civiles**: "soltero", "casado", "divorciado", "viudo"
- **Tiempo**: Días de la semana, meses del año

### Colores:
"Amarillo", "Azul", "Blanco", "Negro", "Rojo", "Verde", "Violeta", etc.

## 🚀 ¡Sistema Listo para Usar!

El sistema está completamente configurado y entrenado. Puedes comenzar a usar el reconocimiento de lenguaje de señas ejecutando:

```bash
python main_interface.py
```

¡Disfruta explorando las 205 clases de gestos con 98.99% de precisión! 🎯🚀

### Opcional
- **GPU compatible con CUDA** para entrenamiento más rápido
- **Micrófono** para funcionalidades adicionales

## 🚀 Instalación Rápida

### Opción 1: Configuración Automática (Recomendada)
```bash
# 1. Clonar/descargar el proyecto
cd sign_language_project2

# 2. Ejecutar configuración automática
python setup_project.py
```

### Opción 2: Instalación Manual
```bash
# 1. Crear entorno virtual
python -m venv venv310
venv310\Scripts\activate  # Windows
# source venv310/bin/activate  # Linux/Mac

# 2. Actualizar pip
python -m pip install --upgrade pip

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python test_imports_improved.py
```

## 🎯 Uso del Sistema

### 1. Iniciar la Interfaz
```bash
python main_interface.py
```

### 2. Flujo de Trabajo Completo

#### Paso 1: Análisis del Dataset
- Ejecutar **"🔍 Analizar Dataset"** para verificar datos existentes
- Revisar recomendaciones en `analysis/`

#### Paso 2: Captura de Datos
- Usar **"🎥 Grabar Nuevo Gesto"**
- Grabar al menos 50-100 muestras por gesto
- Mantener buena iluminación y fondo limpio

#### Paso 3: Entrenamiento
- Ejecutar **"🚀 Entrenar Modelo"**
- El sistema usará técnicas avanzadas de regularización
- Tiempo estimado: 5-15 minutos dependiendo del dataset

#### Paso 4: Evaluación
- Usar **"📈 Evaluar Modelo"** después del entrenamiento
- Revisar métricas en `evaluation/`

#### Paso 5: Reconocimiento
- Ejecutar **"🗣️ Traducir en Tiempo Real"**
- Controles: ESC (salir), ESPACIO (pausar), R (reiniciar)

## 📁 Estructura del Proyecto

```
sign_language_project2/
├── 📄 main_interface.py          # Interfaz principal
├── 🎥 record_dataset.py          # Captura de gestos
├── 🧠 train_model.py             # Entrenamiento mejorado
├── 📊 evaluate_model.py          # Evaluación comprensiva
├── 🔍 analyze_dataset.py         # Análisis de datos
├── 🗣️ real_time_translate.py     # Reconocimiento en tiempo real
├── 🛠️ utils.py                   # Utilidades de procesamiento
├── ⚙️ setup_project.py           # Configuración automática
├── 🧪 test_imports_improved.py   # Pruebas avanzadas
├── 📋 requirements.txt           # Dependencias
│
├── 📂 data/                      # Dataset de gestos
│   ├── gesto1/
│   ├── gesto2/
│   └── ...
│
├── 📂 model/                     # Modelos entrenados
│   ├── gesture_model.h5          # Modelo principal
│   ├── best_model.h5             # Mejor modelo durante entrenamiento
│   ├── labels.pkl                # Etiquetas de clases
│   ├── scaler.pkl                # Normalizador de datos
│   └── training_history.json     # Historial de entrenamiento
│
├── 📂 evaluation/                # Resultados de evaluación
│   ├── metrics.json              # Métricas numéricas
│   ├── confusion_matrix.png      # Matriz de confusión
│   ├── metrics_by_class.png      # Métricas por clase
│   ├── confidence_distribution.png # Distribución de confianza
│   └── report.md                 # Reporte completo
│
└── 📂 analysis/                  # Análisis del dataset
    ├── dataset_analysis.json     # Análisis completo
    ├── samples_distribution.png  # Distribución de muestras
    └── recommendations.txt       # Recomendaciones
```

## 🧠 Arquitectura del Modelo

### Red Neuronal Mejorada
- **Entrada**: 1530 características (landmarks de manos + rostro)
- **Capas densas**: 512 → 256 → 128 → 64 → salida
- **Regularización**: BatchNormalization + Dropout
- **Optimizador**: Adam con learning rate adaptativo
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### Procesamiento de Datos
- **Normalización**: Centrado y escalado de landmarks
- **Aumento de datos**: Variaciones con ruido gaussiano
- **División estratificada**: Mantiene balance de clases

## 📊 Métricas y Evaluación

El sistema genera automáticamente:

### Métricas Numéricas
- Precisión (Accuracy)
- Precision, Recall, F1-Score por clase
- Matriz de confusión
- Distribución de confianza

### Visualizaciones
- Gráficos de métricas por clase
- Matriz de confusión visual
- Historial de entrenamiento
- Análisis de errores

### Archivos Generados
- `evaluation/metrics.json`: Métricas detalladas
- `evaluation/report.md`: Reporte comprensivo
- `analysis/dataset_analysis.json`: Análisis del dataset

## 🔧 Configuración Avanzada

### Ajustar Umbrales de Confianza
En `real_time_translate.py`:
```python
predictor.confidence_threshold = 0.6  # Ajustar según necesidad
```

### Modificar Arquitectura del Modelo
En `train_model.py` función `construir_modelo_mejorado()`:
```python
# Agregar más capas o modificar neuronas
tf.keras.layers.Dense(512, activation='relu'),
```

### Configurar MediaPipe
En `real_time_translate.py`:
```python
# Ajustar sensibilidad de detección
min_detection_confidence=0.7,
min_tracking_confidence=0.5
```

## 🐛 Solución de Problemas

### Error: "No se pudo abrir la cámara"
```bash
# Verificar cámaras disponibles
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

### Error: "Modelo no encontrado"
```bash
# Verificar archivos del modelo
ls model/
# Reentrenar si es necesario
python train_model.py
```

### Error de importación de TensorFlow
```bash
# Instalar versión específica
pip install tensorflow==2.10.0
```

### Baja precisión del modelo
1. **Verificar dataset**: `python analyze_dataset.py`
2. **Aumentar datos**: Grabar más muestras (objetivo: 80+ por gesto)
3. **Mejorar calidad**: Buena iluminación, fondo uniforme
4. **Limpiar datos**: Eliminar muestras de baja calidad

## 🚀 Mejoras Futuras

- [ ] Soporte para secuencias de gestos
- [ ] Reconocimiento de expresiones faciales
- [ ] Integración con base de datos
- [ ] API REST para uso remoto
- [ ] Aplicación móvil
- [ ] Soporte multiidioma

## 📚 Dependencias Principales

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| TensorFlow | ≥2.10.0 | Machine Learning |
| MediaPipe | ≥0.10.0 | Detección de landmarks |
| OpenCV | ≥4.5.0 | Procesamiento de video |
| Scikit-learn | ≥1.0.0 | Métricas y preprocesamiento |
| NumPy | ≥1.21.0 | Computación numérica |
| Matplotlib | ≥3.5.0 | Visualizaciones |
| pyttsx3 | ≥2.90 | Síntesis de voz |

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crear rama para nueva característica
3. Commit con mensajes descriptivos
4. Push a la rama
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍💻 Autor

Desarrollado con ❤️ para la comunidad de personas sordas e hipoacúsicas.

---

## 🆘 Soporte

¿Necesitas ayuda? 

1. **Verificar documentación** arriba
2. **Ejecutar diagnósticos**: `python test_imports_improved.py`
3. **Revisar logs** de error en la consola
4. **Crear issue** con detalles del problema

¡Gracias por usar nuestro sistema de reconocimiento de señas! 🤟
