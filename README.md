# Traductor LSE - Lengua de Señas Ecuatoriana

Sistema de traducción de Lengua de Señas Ecuatoriana (LSE) a texto y voz en tiempo real.

**Objetivo**: Prototipo funcional para reconocimiento de señas orientado a noticias.

---

## 🚀 Inicio Rápido (Doble Clic)

### En macOS
1. Abre **Finder** → navega a la carpeta del proyecto
2. **Doble clic** en `Iniciar_LSE.command`
3. La primera vez instala todo automáticamente, después abre directo ✅

> **Primera vez en macOS**: Si aparece un aviso de seguridad, haz clic derecho → "Abrir" → "Abrir de todos modos".

### En Windows
1. **Doble clic** en `Iniciar_LSE.bat`
2. La primera vez descarga Python e instala todo automáticamente
3. Las siguientes veces abre directo ✅

> **No necesitas instalar nada manualmente.** El lanzador descarga e instala Python y todas las dependencias automáticamente la primera vez.

---

## 🖥️ Opciones de Ejecución

| Método | Archivo | SO | Necesita terminal? |
|--------|---------|-----|-------------------|
| **Doble clic (GUI)** | `Iniciar_LSE.command` | macOS | ❌ No |
| **Doble clic (GUI)** | `Iniciar_LSE.bat` | Windows | ❌ No |
| **Terminal (menú)** | `./ejecutar.sh` | macOS/Linux | ✅ Sí |
| **Terminal (directo)** | `./ejecutar.sh traducir` | macOS/Linux | ✅ Sí |

---

## 🔄 ¿Cómo actualizar después de hacer cambios?

### Si modificas el código (Python):
- Los cambios **se aplican automáticamente** la próxima vez que abras la app con doble clic
- No necesitas reinstalar nada

### Si agregas una nueva dependencia (pip install algo):
- **macOS**: Se activa automáticamente el `.venv` correcto
- **Windows**: Tu compañera debe ejecutar `setup_windows.bat` nuevamente

### Si subes cambios a GitHub:
1. Tu compañera hace `git pull` para descargar los cambios
2. Doble clic en `Iniciar_LSE.bat` → listo

---

## 🖱️ Menú Gráfico

Al hacer doble clic en el lanzador, se abre esta ventana:

```
  ┌─────────────────────────────────────┐
  │     🤟  TRADUCTOR LSE              │
  │     Lengua de Señas Ecuatoriana     │
  │                                     │
  │  📹 Grabación                       │
  │  [  Grabar UNA seña              ]  │
  │  [  Grabar VARIAS señas          ]  │
  │                                     │
  │  🧠 Modelo                          │
  │  [  Entrenar modelo              ]  │
  │  [  Evaluar métricas ISO         ]  │
  │                                     │
  │  🔊 Traductor                       │
  │  [  ▶ Iniciar traductor          ]  │
  │                                     │
  │  [ 🚀 Flujo completo             ]  │
  │                                     │
  │  📂 Estado: 5 señas, modelo ✅      │
  └─────────────────────────────────────┘
```

---

## ⌨️ Ejecución por Terminal (alternativa)

### Menú interactivo
```bash
cd /Users/anahy/sign_language_project2
./ejecutar.sh
```

### Comandos directos (sin menú)
```bash
./ejecutar.sh grabar HOLA     # Grabar una seña directamente
./ejecutar.sh varias          # Grabar múltiples señas de golpe
./ejecutar.sh entrenar        # Entrenar el modelo
./ejecutar.sh traducir        # Iniciar el traductor
./ejecutar.sh evaluar         # Evaluar métricas ISO
./ejecutar.sh estado          # Ver señas grabadas y modelo
./ejecutar.sh flujo           # Flujo completo: grabar → entrenar → traducir
```

> **Nota**: `ejecutar.sh` activa automáticamente el entorno virtual y silencia los warnings de TensorFlow/MediaPipe.

---

## 📋 Flujo Completo (manual)

```bash
# Activar entorno virtual (siempre primero)
source .venv/bin/activate
cd prototipo

# 1. Grabar señas con argumentos (sin preguntas)
python 1_grabar_senas.py --nombre HOLA --cantidad 30

# 2. Entrenar modelo LSTM
python 2_entrenar_modelo.py

# 3. Usar traductor
python 3_traductor.py

# 4. Evaluar métricas ISO/IEC 25023 (opcional)
python 4_evaluar_iso25023.py
```

---

## 🎮 Controles

### Grabador (`1_grabar_senas.py`)
- Grabación **automática** cuando detecta mano estable
- Acepta argumentos: `--nombre SEÑA --cantidad 30`
- Presiona **Q** en la ventana para terminar

### Traductor (`3_traductor.py`)
| Tecla | Acción |
|-------|--------|
| **C** | Limpiar subtítulos |
| **D** | Activar/desactivar modo debug |
| **Q** | Salir |

---

## 🛡️ Sistema de Detección

### Enfoque de manos con fondo difuminado
- El **fondo se difumina** automáticamente, dejando solo las manos nítidas
- Solo se detectan **máximo 2 manos** (las del señante)
- Sin recuadros ni cajas que distraigan
- Las manos caídas (posición natural de reposo) se ignoran automáticamente

### Anti-falsos positivos
- **Umbral de confianza alto**: 96% mínimo para aceptar una seña
- **5 confirmaciones consecutivas**: La misma seña debe detectarse 5 veces seguidas
- **Estabilidad requerida**: La mano debe estar quieta antes de clasificar
- **Cooldown**: 3 segundos entre detecciones para evitar repeticiones

---

## 🧠 Arquitectura Técnica

Basado en investigaciones:
- **Morfín-Chávez et al. (2023)**: MediaPipe 21 puntos, F1=0.98
- **Sincan & Keles (2020)**: CNN+LSTM, 95% precisión
- **Basnin et al. (2021)**: LSTM, 88.5% precisión
- **Paper Dialnet (2025)**: MediaPipe + ML, accuracy 0.817

```
Cámara (30 fps)
    ↓
MediaPipe Hands (máximo 2 manos)
    ↓
Filtrado:
  └── ¿Es posición natural? (descartada si mano caída)
    ↓
Difuminado de fondo (solo manos nítidas)
    ↓
Landmarks (21 × 3 coords = 63 features/mano)
    ↓
Normalización (coordenadas relativas a muñeca)
    ↓
Buffer de 30 frames = [30 × 126] features
    ↓
Verificación de estabilidad (movimiento < umbral)
    ↓
LSTM (64 → 128 → 64 unidades)
    ↓
Confirmación (5 detecciones consecutivas iguales)
    ↓
Clasificación (softmax, confianza > 96%)
    ↓
Subtítulos + Voz (TTS)
```

---

## 📊 Métricas ISO/IEC 25023

El evaluador (`4_evaluar_iso25023.py`) genera métricas **honestas**:

- **Hold-out 70/30**: Entrena un modelo nuevo y evalúa en el 30% que nunca vio
- **Cross-validation 3-Fold**: Cada fold evalúa en datos no vistos
- **Análisis de overfitting**: Compara accuracy en datos vistos vs no vistos
- **Tiempo de respuesta**: ms por predicción
- **Distribución de confianza**: Qué tan seguro está el modelo

> **Nota**: El accuracy reportado es siempre del **test set** (datos no vistos), no del train set.

Reporte guardado en: `modelo/evaluacion_iso25023.json`

---

## 📦 Requisitos

### Dependencias (ya instaladas en .venv)
```bash
pip install opencv-python mediapipe tensorflow pyttsx3 scikit-learn
```

### Versión de Python
- **Requerido**: Python 3.10 (compatible con MediaPipe)
- **El entorno virtual `.venv` ya usa Python 3.10**

---

## 📂 Estructura del Proyecto

```
sign_language_project2/
├── Iniciar_LSE.command          # 🖱️ Doble clic para abrir (macOS)
├── Iniciar_LSE.bat              # 🖱️ Doble clic para abrir (Windows)
├── ejecutar.sh                  # ⌨️ Lanzador terminal (macOS/Linux)
├── .venv/                       # Entorno virtual Python 3.10
├── prototipo/
│   ├── menu.py                  # 🖥️ Menú gráfico (tkinter, cross-platform)
│   ├── 1_grabar_senas.py        # Grabador con args (--nombre --cantidad)
│   ├── 2_entrenar_modelo.py     # Entrenador LSTM (métricas honestas)
│   ├── 3_traductor.py           # Traductor con fondo difuminado
│   ├── 4_evaluar_iso25023.py    # Evaluación ISO (cross-validation)
│   ├── app.py                   # Menú de terminal (Python)
│   ├── utils_silenciar.py       # Supresión de warnings nativos
│   ├── datos/                   # Datos de entrenamiento
│   └── modelo/                  # Modelo entrenado
└── README.md                    # Este archivo
```

---

## 📰 Vocabulario de Noticias (Recomendado)

Señas sugeridas para grabar:
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

---

## 🔧 Solución de Problemas

### macOS: "No se puede abrir porque es de un desarrollador no identificado"
Clic derecho → "Abrir" → "Abrir de todos modos". Solo necesitas hacerlo una vez.

### Windows: Python no se instaló automáticamente
1. Verifica tu conexión a internet
2. Si falla la instalación silenciosa, el lanzador intentará abrir el instalador normal
3. En ese caso, marca ✅ "Add Python to PATH" durante la instalación
4. Vuelve a hacer doble clic en `Iniciar_LSE.bat`

### Error: `command not found: python`
Usar `python3` o activar el entorno virtual primero.

### Error: `No module named 'mediapipe'`
```bash
source .venv/bin/activate    # macOS
.venv\Scripts\activate       # Windows
```

### Accuracy de 100% (sospechoso)
El modelo está memorizando los datos (overfitting). Ejecuta `./ejecutar.sh evaluar` para obtener métricas honestas con datos no vistos. Solución: grabar más secuencias (mín. 50-100 por seña).

### Falsos positivos (detecta señas que no hiciste)
- Sube `UMBRAL_CONFIANZA` en `3_traductor.py` (actual: 0.96)
- Sube `CONFIRMACIONES_REQUERIDAS` (actual: 5)
- Graba más datos variados para cada seña
