# 🤟 Traductor LSE - Lengua de Señas Ecuatoriana

Sistema de traducción de Lengua de Señas Ecuatoriana (LSE) a texto y voz en tiempo real, utilizando visión por computadora y aprendizaje profundo.

**Objetivo**: Prototipo funcional para reconocimiento de señas orientado a noticias ecuatorianas.

---

## 🚀 Descarga y Uso (Ejecutable Windows)

### Opción 1: Descargar ejecutable listo (recomendado)

1. Ve a [GitHub Actions](https://github.com/aeherrera16/sign_language_project2/actions) → último workflow exitoso de **"CD - Construir Aplicaciones"**
2. Descarga el artefacto **TraductorLSE-Windows**
3. Descomprime la carpeta
4. Ejecuta **`TraductorLSE.exe`**

> **Nota**: La primera vez Windows Defender puede mostrar un aviso. Presiona **"Más información" → "Ejecutar de todas formas"**.

**No necesitas instalar Python ni ninguna dependencia.** Todo está empaquetado en el ejecutable.

### Opción 2: Ejecutar desde código fuente (Windows)

1. **Doble clic** en `Iniciar_LSE.bat`
2. La primera vez descarga Python 3.10 e instala las dependencias automáticamente
3. Las siguientes veces abre el menú directamente

---

## 🍓 Dispositivo Dedicado (Raspberry Pi)

El sistema puede configurarse como un **sistema embebido dedicado** usando una Raspberry Pi. En este modo:
- Arranca directamente al menú de la aplicación (modo kiosk) sin escritorio.
- Se controla desde el **celular** vía Telegram o panel web — sin teclado ni mouse.
- Se actualiza automáticamente desde GitHub con un solo botón.
- Entrena el modelo en segundo plano cuando hay datos nuevos en la nube.

### Requisitos de Hardware
- Raspberry Pi 4 u 5 (mínimo 2GB RAM, recomendado 4GB+)
- MicroSD 32GB A2
- Cámara USB (wide 1080p recomendada)
- Parlante (jack 3.5mm o USB)
- Pantalla HDMI

### Instalación en Raspberry Pi

**Opción rápida** (si ya tienes `kiosk_xinit.sh` configurado):

1. Clona el repo: `git clone https://github.com/aeherrera16/sign_language_project2.git`
2. Ejecuta el lanzador:
   ```bash
   cd sign_language_project2
   bash Iniciar_LSE_RaspberryPi.sh
   ```
   La primera vez instala todas las dependencias (~20-40 min). Las siguientes veces abre el menú directamente.

**Si el Pi usa `kiosk_xinit.sh` preconfigurado**, asegúrate de que apunte al directorio correcto:
```bash
# El script debe tener estas líneas:
cd /home/pi/sign_language_project2
source .venv_pi/bin/activate
python3 prototipo/menu.py
```

### Entorno Python del Pi
El Pi usa **Python 3.11** en un entorno virtual `.venv_pi/` (no el Python del sistema). Esto es necesario para compatibilidad con TensorFlow 2.14+.

---

## 📱 Control desde el Celular

Una vez arrancado el Pi, el panel de control se inicia automáticamente junto con el menú.

### Panel Web (misma red WiFi)
Abre desde el navegador de tu celular:
```
http://192.168.1.XX:5000
```
(reemplaza con la IP de tu Pi — la imprime en consola al arrancar)

Desde el panel puedes: iniciar/detener el traductor, lanzar entrenamiento, sincronizar datos con la nube y reiniciar.

### Bot de Telegram (cualquier lugar con internet)

El bot funciona desde cualquier red — no necesitas estar en la misma WiFi.

**Configuración (una sola vez):**

1. Abre Telegram → busca **@BotFather** → escribe `/newbot`
2. Sigue los pasos y copia el token que te da (ej: `7123456789:AAHxxx...`)
3. Guarda el token en el Pi por SSH:
   ```bash
   cat > ~/sign_language_project2/prototipo/.panel_config.json << 'EOF'
   {"telegram_token": "PEGA_AQUI_TU_TOKEN", "telegram_chat_ids": []}
   EOF
   ```
4. Reinicia el menú del Pi: `pkill -f "menu.py"`
5. Busca tu bot en Telegram y escríbele **`/start`**

**Comandos disponibles:**

| Comando | Acción |
|---|---|
| `/start` | Registrar tu chat y ver ayuda |
| `/estado` | Ver si el traductor está activo |
| `/entrenar` | Entrenar el modelo en segundo plano |
| `/sincronizar` | Descargar datos nuevos de la nube |
| `/reiniciar` | Reiniciar el traductor |

> El token de Telegram está en `.panel_config.json` que está en `.gitignore` — nunca se sube a GitHub.

---

## 🖥️ Menú Principal

```
  ┌─────────────────────────────────────┐
  │     🤟  TRADUCTOR LSE              │
  │     Lengua de Señas Ecuatoriana     │
  │                                     │
  │  📹 Grabación                       │
  │  [1] Grabar UNA seña                │
  │  [2] Grabar VARIAS señas            │
  │                                     │
  │  🧠 Modelo                          │
  │  [3] Entrenar modelo                │
  │  [4] Evaluar métricas ISO           │
  │                                     │
  │  🔊 Traductor                       │
  │  [5] ▶ Iniciar traductor            │
  │                                     │
  │  [6] Flujo completo                 │
  │                                     │
  │  ⚙️  Sistema                        │
  │  [8] ⬇ Actualizar desde GitHub     │
  │                                     │
  │  📂 Estado: 19 señas, modelo ✅     │
  └─────────────────────────────────────┘
```

**`[8] Actualizar desde GitHub`**: descarga el código y modelos más recientes directamente desde el Pi, sin necesitar terminal ni SSH.

---

## 📋 Flujo de Trabajo

### 1. Grabar señas (`1_grabar_senas.py`)
- Ingresa el nombre de la seña (ej: HOLA, GRACIAS)
- La cámara se abre y graba **automáticamente** cuando detecta tu mano estable
- Se generan 30 secuencias de 30 frames cada una
- Presiona **Q** para terminar

### 2. Entrenar modelo (`2_entrenar_modelo.py`)
- Necesitas mínimo **2 señas** grabadas
- Entrena una red LSTM con los datos grabados (97.4% accuracy con 19 señas)
- Data augmentation 14x: cada secuencia genera 14 variantes (ruido, escala, velocidad, espejo)
- Guarda el modelo en 3 formatos: `.h5`, `modelo_savedmodel/` (para Raspberry Pi), `.tflite` (si disponible)
- Muestra accuracy en datos de prueba (test set)

### 3. Traducir en tiempo real (`3_traductor.py`)
- Abre la cámara y traduce las señas que hagas
- Genera subtítulos en pantalla y audio con TTS (voz femenina española)
- Carga automáticamente el modelo más ligero disponible (TFLite → SavedModel → H5)
- Se actualiza con nuevos modelos en segundo plano sin interrumpir la traducción

| Tecla | Acción |
|-------|--------|
| **C** | Limpiar subtítulos |
| **D** | Activar/desactivar modo debug |
| **Q** | Salir |

### 4. Evaluar métricas ISO (`4_evaluar_iso25023.py`)
- Evalúa el modelo con métricas honestas (cross-validation)
- Genera reporte en `modelo/evaluacion_iso25023.json`

---

## 🛡️ Sistema de Detección

### Enfoque de manos con fondo difuminado
- El **fondo se difumina** automáticamente, dejando solo las manos nítidas
- Solo se detectan **máximo 2 manos** (las del señante)
- Las manos caídas (posición natural de reposo) se ignoran automáticamente

### Anti-falsos positivos
- **Umbral de confianza**: 90% mínimo para aceptar una seña
- **3 confirmaciones consecutivas**: La misma seña debe detectarse 3 veces seguidas
- **Estabilidad requerida**: La mano debe estar quieta antes de clasificar
- **Cooldown**: 1.5 segundos entre detecciones para evitar repeticiones

---

## 🧠 Arquitectura Técnica

```
Cámara (30 fps)
    ↓
MediaPipe Hands (máximo 2 manos)
    ↓
Filtrado: descarta manos en posición de reposo
    ↓
Difuminado de fondo (solo manos nítidas)
    ↓
Landmarks (21 × 3 coords = 63 features/mano × 2 manos = 126 features)
    ↓
Normalización (coordenadas relativas a muñeca)
    ↓
Buffer de 30 frames = [30 × 126] features
    ↓
Verificación de estabilidad
    ↓
LSTM (64 → 128 → 64 unidades) + BatchNormalization
    ↓
Confirmación (3 detecciones consecutivas)
    ↓
Clasificación (softmax, confianza > 90%)
    ↓
Subtítulos + Voz (espeak TTS, voz femenina es+f3)
```

### Tecnologías utilizadas
- **MediaPipe**: Detección de manos (21 landmarks por mano)
- **TensorFlow/Keras**: Red neuronal LSTM para clasificación
- **OpenCV**: Captura de cámara y visualización
- **espeak**: Síntesis de voz (TTS nativo Linux, voz femenina española)
- **Flask**: Panel web de control remoto (puerto 5000)
- **Telegram Bot**: Control remoto desde cualquier red
- **Firebase**: Sincronización de datos y modelos en la nube (opcional)
- **scikit-learn**: Evaluación de métricas

### Referencias
- Morfín-Chávez et al. (2023): MediaPipe 21 puntos, F1=0.98
- Sincan & Keles (2020): CNN+LSTM, 95% precisión
- Basnin et al. (2021): LSTM, 88.5% precisión

---

## 📊 Métricas ISO/IEC 25023

El evaluador genera métricas honestas:

- **Hold-out 70/30**: Entrena un modelo nuevo y evalúa en el 30% que nunca vio
- **Cross-validation 3-Fold**: Cada fold evalúa en datos no vistos
- **Análisis de overfitting**: Compara accuracy en datos vistos vs no vistos
- **Tiempo de respuesta**: ms por predicción

> El accuracy reportado es siempre del **test set** (datos no vistos), no del train set.

---

## 📂 Estructura del Proyecto

```
sign_language_project2/
├── Iniciar_LSE.bat                  # Lanzador Windows (código fuente)
├── Iniciar_LSE_RaspberryPi.sh       # Lanzador Raspberry Pi
├── .github/workflows/
│   ├── ci.yml                       # Integración continua
│   └── cd.yml                       # Build ejecutable Windows
├── build/
│   ├── traductor_lse.spec           # Config PyInstaller Windows
│   └── traductor_lse_raspberry.spec # Config PyInstaller RPi
├── prototipo/
│   ├── menu.py                      # Menú gráfico principal (tkinter)
│   ├── modo_traductor.py            # Orquestador modo kiosk (--loop)
│   ├── panel_control.py             # Panel web Flask + Bot Telegram
│   ├── sync_cloud.py                # Sincronización con Firebase
│   ├── 1_grabar_senas.py            # Grabador automático de señas
│   ├── 2_entrenar_modelo.py         # Entrenador LSTM + SavedModel
│   ├── 3_traductor.py               # Traductor en tiempo real
│   ├── 4_evaluar_iso25023.py        # Evaluación métricas ISO
│   ├── utils_silenciar.py           # Supresión de warnings
│   ├── datos/                       # Señas grabadas (19 clases)
│   └── modelo/
│       ├── modelo.h5                # Modelo Keras (entrenamiento)
│       ├── modelo_savedmodel/       # Formato TF 2.x (Raspberry Pi)
│       ├── modelo.tflite            # TFLite si disponible (más ligero)
│       ├── encoder.pkl              # Codificador de etiquetas
│       └── info.json                # Métricas del último entrenamiento
└── README.md
```

---

## 📰 Vocabulario Actual (19 señas)

| Seña | Seña | Seña |
|------|------|------|
| BIENVENIDO | BUENOS_DIAS | BUENAS_TARDES |
| BUENAS_NOCHES | GRACIAS | POR_FAVOR |
| DISCULPE | SI | NO |
| HOY | QUE_PASO | REPITA |
| PRESIDENTE | GOBIERNO | ECUADOR |
| AÑO | TRABAJO | HOY |
| NINGUNA | | |

> Accuracy actual: **97.4%** (test set, 19 señas, data augmentation 14x)

---

## 📦 Requisitos (solo para desarrollo en Mac/Windows)

```bash
pip install opencv-python mediapipe tensorflow keras scikit-learn flask requests
```

**Python requerido**: 3.10–3.11 (3.13 no compatible con MediaPipe)

**Raspberry Pi**: usar `.venv_pi/` con Python 3.11 — el script `Iniciar_LSE_RaspberryPi.sh` lo configura automáticamente.

---

## 🔧 Solución de Problemas

### Windows: "Windows protegió tu equipo"
Clic en **"Más información"** → **"Ejecutar de todas formas"**. Solo necesitas hacerlo una vez.

### Error: "No se pudo abrir la cámara"
- Verifica que la cámara esté conectada
- Cierra otras aplicaciones que usen la cámara

### Raspberry Pi: pantalla negra al arrancar
El Pi usa `kiosk_xinit.sh` para lanzar la app. Si la pantalla queda negra:
```bash
ssh pi@192.168.1.XX
cat /tmp/traductor_lse_error.log   # ver el error
pkill -f "menu.py"                 # reiniciar el menú
```

### Raspberry Pi: bot Telegram no responde
1. Verifica que el token esté en `prototipo/.panel_config.json`
2. Reinicia el menú: `pkill -f "menu.py"`
3. Escribe `/start` al bot desde Telegram

### Raspberry Pi: "import keras" falla
El Pi usa el formato `modelo_savedmodel/` (compatible con cualquier TF 2.x). Si falla, el entrenamiento se hace en Mac y se sube a GitHub — el Pi solo hace inferencia.

### Accuracy de 100% (sospechoso)
El modelo puede estar memorizando (overfitting). Solución: grabar más secuencias (mínimo 30-50 por seña) o bajar `AUGMENTACIONES_POR_MUESTRA` en `2_entrenar_modelo.py`.

### Falsos positivos
- Sube `UMBRAL_CONFIANZA` en `3_traductor.py` (actual: 0.90)
- Sube `CONFIRMACIONES_REQUERIDAS` (actual: 3)
- Graba más datos variados para cada seña

---

## 🔄 CI/CD

- **CI** (`ci.yml`): Verifica compilación en cada push a `main` y `feature/prototipo-wearable`
- **CD** (`cd.yml`): Construye el ejecutable Windows (.exe) con PyInstaller

Los ejecutables se descargan desde la pestaña **Actions** del repositorio.
