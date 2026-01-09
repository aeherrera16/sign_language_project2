# 🍓 Traductor LSE para Raspberry Pi

Esta carpeta contiene todo lo necesario para ejecutar el traductor de forma autónoma en una Raspberry Pi 4.

## 📦 Contenido

| Archivo | Descripción |
|---------|-------------|
| `traductor_portable.py` | Aplicación principal standalone |
| `setup_rpi_standalone.sh` | Script de instalación automática |
| `requirements_rpi.txt` | Dependencias Python optimizadas |
| `convert_to_tflite.py` | Convierte modelo H5 → TFLite |
| `install_rpi.sh` | Instalador Docker (alternativo) |

## 🚀 Instalación Rápida

```bash
# 1. Copiar proyecto a la Pi
scp -r ../sign_language_project2 pi@<IP_PI>:~/

# 2. SSH a la Pi
ssh pi@<IP_PI>

# 3. Ejecutar instalador
cd ~/sign_language_project2/raspberry_pi
chmod +x setup_rpi_standalone.sh
sudo ./setup_rpi_standalone.sh

# 4. Reiniciar
sudo reboot
```

## 🎮 Uso

```bash
# Con ventana de video
traductor-lse

# Solo audio (sin pantalla)
traductor-lse --no-display

# Servicio systemd
sudo systemctl start traductor-lse
```

## ⚙️ Requisitos de Hardware

- Raspberry Pi 4 (4GB+ recomendado)
- Cámara USB
- Bocina (USB o Jack 3.5mm)
- MicroSD 16GB+ Class 10

## 🔧 Ajustes de Audio

```bash
# Volumen
amixer set Master 90%

# Forzar salida Jack 3.5mm
amixer cset numid=3 1

# Probar
espeak -ves "Probando audio"
```
