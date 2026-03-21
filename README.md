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

El sistema puede configurarse como un **sistema embebido dedicado** usando una Raspberry Pi, ideal para exhibiciones o uso como traductor autónomo. En este modo:
- Arranca directamente a la aplicación (modo kiosk) sin pedir usuario ni contraseña.
- No hay escritorio ni terminal visible.
- Se controla completamente mediante el teclado (atajos numéricos del 1 al 6).

### Instalación Rápida
Si cuentas con una Raspberry Pi (recomendado modelo 5 u 4 de 8GB) con **Raspberry Pi OS Lite (64-bit)** instalado:

1. Transfiere este repositorio a la Pi: `git clone https://github.com/aeherrera16/sign_language_project2.git`
2. Entra al directorio y ejecuta el setup automatizado con sudo:
   ```bash
   cd sign_language_project2
   sudo bash raspberry/setup_imagen_embebida.sh
   ```
3. Reinicia la Raspberry Pi.

### Guía Detallada: Desde Cero hasta el Artefacto (.img)

Si necesitas instalar el sistema desde cero o crear la imagen final `.img` para distribución, sigue estos pasos:

#### 1. Requisitos de Hardware
* **Raspberry Pi**: Modelo 5 o 4 (8GB recomendado).
* **Almacenamiento**: MicroSD (mínimo 32GB A2).
* **Cámara**: Cámara USB Externa (Wide 1080p).
* **Audio**: Parlante (jack 3.5mm o USB).
* **Pantalla**: HDMI o pantalla oficial.
* Teclado USB (solo para la instalación inicial).

#### 2. Instalación Base del SO
1. Descarga **[Raspberry Pi Imager](https://www.raspberrypi.com/software/)**.
2. Selecciona **OS (Other) -> Raspberry Pi OS Lite (64-bit)** (Sin interfaz de escritorio).
3. Selecciona tu MicroSD.
4. En ajustes (⚙️):
   * Hostname: `traductor-lse`
   * Activa **SSH** (con contraseña).
   * Usuario: `pi`, Contraseña: (tu elección).
   * Configura Wi-Fi si no usarás cable ethernet.
5. Graba la imagen e inserta la SD en la Raspberry Pi.

#### 3. Ejecución del Setup Automático
Accede por SSH (`ssh pi@IP_DEL_PI`) o con teclado y pantalla física. Clona el repositorio y ejecuta el script:

```bash
sudo apt-get install -y git
git clone https://github.com/aeherrera16/sign_language_project2.git
cd sign_language_project2
sudo bash raspberry/setup_imagen_embebida.sh
```

El script verificará el hardware, instalará el entorno X11 (kiosk), empaquetará dependencias de Machine Learning (MediaPipe, TensorFlow, OpenCV) de forma aislada e inyectará los perfiles para que se despliegue automáticamente en el siguiente encendido. Además, oculta alertas del kernel de red para un "Arranque Silencioso".

Reinicia la Raspberry Pi (`sudo reboot`). En adelante, abrirá la interfaz del Traductor sin interfaz de escritorio (sistema dedicado).

#### 4. Creación del Artefacto Distribuidor (.img)
Una vez validas el funcionamiento, puedes clonar la memoria y empacar toda la distribución para repartir a usuarios que no sepan programar:
* Extrae la MicroSD de la placa apagada y pásala a tu PC.
* Encuentra el disco en la terminal (`diskutil list` en Mac, `lsblk` en Linux).
* Crea el archivo empacado bloque a bloque:
  ```bash
  sudo dd if=/dev/rdiskX of=~/Desktop/traductor_lse_release.img bs=4m status=progress
  ```
Ese archivo **.img** final es el "Software Embebido" del traductor.

#### 5. Controles sin Mouse
El modo kiosk utiliza teclas como switches directos:
* `1` - Grabar UNA seña
* `2` - Grabar VARIAS señas
* `3` - Re-entrenar modelo
* `4` - Evaluación ISO/IEC
* `5` - Iniciar Traductor en tiempo real
* `6` - Flujo End-to-End
* `0` ó `ESC` - Cerrar sesión activa

---

## 🖥️ Menú Principal

Al abrir la aplicación se presenta un menú gráfico con las siguientes opciones:

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
  │  📂 Estado: 8 señas, modelo ✅      │
  └─────────────────────────────────────┘
```

---

## 📋 Flujo de Trabajo

### 1. Grabar señas (`1_grabar_senas.py`)
- Ingresa el nombre de la seña (ej: HOLA, GRACIAS)
- La cámara se abre y graba **automáticamente** cuando detecta tu mano estable
- Se generan 30 secuencias de 30 frames cada una
- Presiona **Q** para terminar

### 2. Entrenar modelo (`2_entrenar_modelo.py`)
- Necesitas mínimo **2 señas** grabadas
- Entrena una red LSTM con los datos grabados
- Muestra accuracy en datos de prueba (test set)

### 3. Traducir en tiempo real (`3_traductor.py`)
- Abre la cámara y traduce las señas que hagas
- Genera subtítulos en pantalla y audio con TTS
- Controles:

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
- **Umbral de confianza**: 85% mínimo para aceptar una seña
- **2 confirmaciones consecutivas**: La misma seña debe detectarse 2 veces seguidas
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
Landmarks (21 × 3 coords = 63 features/mano)
    ↓
Normalización (coordenadas relativas a muñeca)
    ↓
Buffer de 30 frames = [30 × 126] features
    ↓
Verificación de estabilidad
    ↓
LSTM (64 → 128 → 64 unidades)
    ↓
Confirmación (2 detecciones consecutivas)
    ↓
Clasificación (softmax, confianza > 85%)
    ↓
Subtítulos + Voz (pyttsx3 TTS)
```

### Tecnologías utilizadas
- **MediaPipe**: Detección de manos (21 landmarks por mano)
- **TensorFlow/Keras**: Red neuronal LSTM para clasificación
- **OpenCV**: Captura de cámara y visualización
- **pyttsx3**: Síntesis de voz (text-to-speech)
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
├── Iniciar_LSE.bat              # 🖱️ Lanzador para ejecutar desde código fuente
├── .github/
│   └── workflows/
│       ├── ci.yml               # Pipeline de integración continua
│       └── cd.yml               # Pipeline de construcción del ejecutable
├── build/
│   ├── traductor_lse.spec       # Configuración de PyInstaller
│   └── hooks/                   # Hooks de PyInstaller
├── prototipo/
│   ├── menu.py                  # 🖥️ Menú gráfico principal (tkinter)
│   ├── 1_grabar_senas.py        # 📹 Grabador automático de señas
│   ├── 2_entrenar_modelo.py     # 🧠 Entrenador del modelo LSTM
│   ├── 3_traductor.py           # 🔊 Traductor en tiempo real
│   ├── 4_evaluar_iso25023.py    # 📊 Evaluación de métricas ISO
│   ├── utils_silenciar.py       # Supresión de warnings nativos
│   ├── icon.png                 # Ícono de la aplicación
│   ├── datos/                   # Datos de entrenamiento (señas grabadas)
│   └── modelo/                  # Modelo entrenado (.h5 + encoder)
└── README.md                    # Este archivo
```

---

## 📰 Vocabulario de Noticias (Señas recomendadas)

| # | Seña | Descripción |
|---|------|-------------|
| 1 | PRESIDENTE | Cargos públicos |
| 2 | GOBIERNO | Instituciones |
| 3 | ECUADOR | Lugares |
| 4 | DECIR | Verbos de comunicación |
| 5 | AÑO | Tiempo |
| 6 | POBREZA | Temas sociales |
| 7 | TRABAJO | Economía |
| 8 | SUBIR / BAJAR | Tendencias |
| 9 | BUENO / MALO | Adjetivos |
| 10 | HOY | Referencias temporales |

---

## 📦 Requisitos (solo para desarrollo)

### Dependencias Python
```bash
pip install opencv-python mediapipe tensorflow pyttsx3 scikit-learn
```

### Versión de Python
- **Requerido**: Python 3.10 (compatible con MediaPipe)

---

## 🔧 Solución de Problemas

### Windows: "Windows protegió tu equipo"
Clic en **"Más información"** → **"Ejecutar de todas formas"**. Solo necesitas hacerlo una vez.

### Error: "No se pudo abrir la cámara"
- Verifica que la cámara esté conectada
- Cierra otras aplicaciones que usen la cámara
- En VMs: asegúrate de habilitar la cámara USB del host

### Accuracy de 100% (sospechoso)
El modelo puede estar memorizando (overfitting). Ejecuta **"Evaluar métricas ISO"** para obtener métricas honestas. Solución: grabar más secuencias (mínimo 50-100 por seña).

### Falsos positivos
- Sube `UMBRAL_CONFIANZA` en `3_traductor.py` (actual: 0.85)
- Sube `CONFIRMACIONES_REQUERIDAS` (actual: 2)
- Graba más datos variados para cada seña

---

## 🔄 CI/CD

El proyecto utiliza GitHub Actions para automatizar:

- **CI** (`ci.yml`): Verifica que el código compila correctamente en cada push
- **CD** (`cd.yml`): Construye automáticamente el ejecutable de Windows (.exe) usando PyInstaller

Los ejecutables se generan como artefactos de GitHub Actions y se pueden descargar desde la pestaña **Actions** del repositorio.
