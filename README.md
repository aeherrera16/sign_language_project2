# 🤟 Traductor de Lengua de Señas Ecuatoriana (LSE)

Sistema de reconocimiento y traducción de lengua de señas en tiempo real usando inteligencia artificial.

## 📁 Estructura del Proyecto

```
sign_language_project2/
├── backend/                    # API FastAPI
│   ├── main.py                # Servidor principal (desarrollo)
│   ├── model/                 # Modelos entrenados
│   │   ├── best_model.h5      # Modelo Keras
│   │   ├── model.tflite       # Modelo TFLite (para RPi)
│   │   └── labels.pkl         # Etiquetas de señas
│   ├── routers/               # Endpoints API
│   └── services/              # Servicios (MediaPipe, LLM)
├── frontend/                   # Interfaz web React
│   └── react_app/             
├── raspberry_pi/              # 🍓 Para despliegue en Raspberry Pi
│   ├── traductor_portable.py  # App standalone
│   ├── setup_rpi_standalone.sh
│   └── requirements_rpi.txt
└── docker-compose.yml         # Para desarrollo con Docker
```

## 🚀 Inicio Rápido

### Desarrollo Local (Navegador)

```bash
# 1. Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 2. Frontend (otra terminal)
cd frontend/react_app
npm install
npm run dev
```

Abre http://localhost:5173

### Raspberry Pi (Standalone)

Ver carpeta `raspberry_pi/` para instrucciones de despliegue.

## 🎯 Señas Entrenadas

Actualmente el modelo reconoce 10 señas:
- Números: `1`, `2`, `3`, `4`
- Letras: `a`, `b`, `c`, `d`, `e`
- Palabras: `hola`

## 📖 Uso

1. **Grabar Señas**: Capturar muestras de nuevas señas
2. **Entrenar Modelo**: Crear/actualizar el modelo AI
3. **Traducir**: Reconocer señas en tiempo real

## 🔧 Requisitos

- Python 3.9+
- Node.js 18+ (para frontend)
- Cámara web
- TensorFlow 2.x

## 📝 Licencia

MIT License
