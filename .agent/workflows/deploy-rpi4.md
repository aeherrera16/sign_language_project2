---
description: Guía paso a paso para desplegar el Traductor LSE en una Raspberry Pi 4
---

# Despliegue en Raspberry Pi 4

Esta guía te ayudará a configurar tu Raspberry Pi 4 (8GB o 4GB recomendados) para ejecutar el sistema de reconocimiento de señas.

### 1. Preparación del Sistema (OS)
Se recomienda **Raspberry Pi OS (64-bit)** para asegurar compatibilidad con MediaPipe y TensorFlow.

```bash
# Actualizar el sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias de sistema para OpenCV y MediaPipe
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev libjasper-dev libqtgui4 libqt4-test libilmbase-dev libopenexr-dev libgstreamer1.0-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
```

### 2. Entorno Python
```bash
# Instalar pip y venv
sudo apt install -y python3-pip python3-venv

# Crear y activar entorno virtual
cd sign_language_project2/backend
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
# Nota: Si tensorflow no está en requirements.txt:
pip install tensorflow-cpu  # Versión optimizada para RPi
```

### 3. Configuración del Frontend
```bash
# Instalar Node.js (v18+)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Instalar y compilar frontend
cd ../frontend/react_app
npm install
npm run build
```

### 4. Servicio de IA (Ollama)
Ollama corre en RPi 4, pero es pesado. Se recomienda usar modelos pequeños como `tinyllama` o `phi` si el rendimiento es crítico.

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelo ligero
ollama pull tinyllama
```

### 5. Ejecución
Para que funcione en la red local (verlo en otros dispositivos):

**Backend:**
```bash
# Dentro de backend/
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
# Dentro de frontend/react_app/
# Usar el build de producción con un servidor estático
sudo npm install -g serve
serve -s build -l 3000
```

### 6. Optimizaciones para RPi 4
- **Cámara**: Usa `v4l2-ctl` para ajustar la resolución si el video va lento.
- **Memoria**: Aumenta el `gpu_mem` en `/boot/config.txt` a al menos 128 o 256.
- **Overclock**: (Opcional) Subir a 2.0GHz si tienes buena refrigeración.
