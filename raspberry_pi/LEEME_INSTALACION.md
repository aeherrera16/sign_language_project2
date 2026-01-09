# 🤟 TRADUCTOR LSE - INSTALACIÓN AUTOMÁTICA

## ✨ Instalación TODO-EN-UNO (Recomendada)

Esta es una instalación **completamente automática** que NO requiere configuración manual.

### 📋 Requisitos Mínimos

- **Hardware**: Raspberry Pi 4 (4GB RAM mínimo, 8GB recomendado)
- **Sistema**: Raspberry Pi OS (Bullseye o Bookworm)
- **Cámara**: Cualquier cámara USB o Raspberry Pi Camera Module
- **Internet**: Conexión activa (solo durante instalación)

### 🚀 Instalación en 2 Pasos

#### Paso 1: Copiar archivos al Raspberry Pi

Transfiere la carpeta `raspberry_pi` a tu Raspberry Pi usando:

```bash
# Opción A: Desde USB
# Copia la carpeta a tu USB y luego al Raspberry Pi

# Opción B: Por red (desde tu Mac/PC)
scp -r raspberry_pi pi@raspberrypi.local:~/
```

#### Paso 2: Ejecutar instalador

En el Raspberry Pi, abre una terminal y ejecuta:

```bash
cd ~/raspberry_pi
chmod +x instalar_completo.sh
sudo ./instalar_completo.sh
```

**¡ESO ES TODO!** El script:

✅ Instala Python y todas las dependencias del sistema  
✅ Descarga e instala TensorFlow Lite automáticamente  
✅ Instala MediaPipe, OpenCV, NumPy (todo automático)  
✅ Crea la aplicación de escritorio  
✅ Agrega iconos al menú y escritorio  

**NO necesitas instalar NADA manualmente**

### ⏱️ Tiempo de instalación

- Primera vez: **10-15 minutos** (descarga dependencias)
- El script trabaja solo, no requiere intervención

### 🎯 Cómo Usar Después de Instalar

Una vez instalado, simplemente:

1. **Busca** "Traductor LSE" en el menú de aplicaciones
2. **O haz clic** en el icono del escritorio
3. **O ejecuta**: `~/TraductorLSE/lanzar.sh`

---

## 🎓 Primera Vez - Entrenar Modelo

La primera vez que abras la aplicación:

1. Selecciona **"🎓 ENTRENAR MODELO"**
2. Captura algunas señas (mínimo 3, recomendado 10+)
3. Graba 30-40 muestras por seña
4. Presiona "Entrenar Modelo"
5. Espera 2-5 minutos
6. ¡Listo! Ya puedes usar el traductor

---

## 📦 ¿Qué se instala?

El instalador automático instala:

### Dependencias del Sistema:
- Python 3.11
- Librerías de video (OpenCV dependencies)
- Herramientas de compilación
- Texto a voz (espeak)

### Dependencias Python (en entorno virtual):
- **TensorFlow Lite** 2.15 (optimizado para Raspberry Pi)
- **MediaPipe** 0.10.9 (detección de manos)
- **OpenCV** 4.8.1 (procesamiento de video)
- **NumPy** 1.24.3 (cálculos numéricos)
- **Scikit-learn** 1.3.2 (ML utilities)

**IMPORTANTE:** Todo se instala en un entorno virtual (`~/TraductorLSE/venv`)  
No afecta otras instalaciones de Python en tu sistema.

---

## 🔧 Solución de Problemas

### "Permission denied"
```bash
chmod +x instalar_completo.sh
```

### "Command not found: sudo"
Ya tienes permisos de root, ejecuta directamente:
```bash
./instalar_completo.sh
```

### La cámara no se detecta
```bash
# Verificar cámaras disponibles
v4l2-ctl --list-devices

# Si usas Raspberry Pi Camera Module
sudo raspi-config
# → Interface Options → Camera → Enable
```

### Problema con TensorFlow
El instalador usa TensorFlow Lite (versión ligera). Si hay problemas:
```bash
cd ~/TraductorLSE
source venv/bin/activate
pip install tensorflow==2.15.0
```

---

## 📁 Estructura de Archivos Instalados

```
~/TraductorLSE/
├── venv/                    # Entorno virtual Python (con todo instalado)
├── app/                     # Aplicación de traducción
├── trainer/                 # Módulo de entrenamiento
├── model/                   # Modelos entrenados
├── data/                    # Datos de entrenamiento
├── logs/                    # Registros
├── traductor_lse.py        # Lanzador principal
└── lanzar.sh               # Script de inicio
```

---

## 🗑️ Desinstalar

Para eliminar completamente la aplicación:

```bash
# Eliminar aplicación
rm -rf ~/TraductorLSE

# Eliminar accesos directos
rm ~/.local/share/applications/traductor-lse.desktop
rm ~/.local/share/icons/traductor-lse.svg
rm ~/Desktop/traductor-lse.desktop
```

Las dependencias del sistema pueden quedarse (no ocupan mucho espacio y pueden ser útiles).

---

## 🤝 Soporte

Si tienes problemas:

1. Verifica que estás usando **Raspberry Pi OS actualizado**
2. Asegúrate de tener **conexión a internet** durante instalación
3. Revisa los logs en `~/TraductorLSE/logs/`

---

## 📝 Notas Importantes

- ✅ **NO necesitas Docker**
- ✅ **NO necesitas instalar TensorFlow manualmente**
- ✅ **NO necesitas configurar nada**
- ✅ **Todo es automático**
- ✅ **Aplicación de escritorio nativa**
- ✅ **Optimizado para Raspberry Pi 4**

**¡Disfruta traduciendo señas! 🤟**
