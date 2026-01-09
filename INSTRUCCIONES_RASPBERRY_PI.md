# 🤟 TRADUCTOR LSE - INSTALACIÓN AUTOMÁTICA PARA RASPBERRY PI

## ✅ ¡TODO LISTO PARA USAR!

Ya he creado un **paquete completo TODO-EN-UNO** que instala **automáticamente** todas las dependencias sin que tengas que hacer nada manual.

---

## 📦 Archivo Generado

**Archivo:** `TraductorLSE-RaspberryPi-COMPLETO.zip` (868 KB)  
**Ubicación:** `/Users/anahy/sign_language_project2/`

---

## 🚀 INSTRUCCIONES ULTRA-SIMPLES

### Paso 1: Transferir a Raspberry Pi

**Opción A - Por Red (Más Fácil):**
```bash
# Desde tu Mac, ejecuta:
scp TraductorLSE-RaspberryPi-COMPLETO.zip pi@raspberrypi.local:~/
```

**Opción B - Por USB:**
1. Copia `TraductorLSE-RaspberryPi-COMPLETO.zip` a una USB
2. Conecta la USB al Raspberry Pi
3. Copia el archivo a tu carpeta home

---

### Paso 2: En el Raspberry Pi

Abre una terminal y ejecuta:

```bash
# 1. Descomprimir
unzip TraductorLSE-RaspberryPi-COMPLETO.zip

# 2. Entrar a la carpeta
cd raspberry_pi

# 3. Ejecutar instalador (TODO ES AUTOMÁTICO)
sudo ./instalar_completo.sh
```

**¡ESO ES TODO!** 🎉

---

## ⏱️ ¿Cuánto tarda?

- **10-15 minutos** (descarga e instala automáticamente todo)
- NO requiere intervención tuya
- NO hace preguntas
- TODO se instala solo

---

## ✅ ¿Qué se instala AUTOMÁTICAMENTE?

El script `instalar_completo.sh` descarga e instala:

✅ **Python 3.11** y todas sus dependencias  
✅ **TensorFlow Lite** (optimizado para Raspberry Pi)  
✅ **MediaPipe** (detección de manos)  
✅ **OpenCV** (procesamiento de video)  
✅ **NumPy, Pillow, Scikit-learn**  
✅ **Aplicación de escritorio** con interfaz gráfica  
✅ **Iconos y accesos directos**  
✅ **Tu modelo ya entrenado** (si existe)  

**TODO sin pedir confirmación ni configuración manual**

---

## 🎯 Después de la Instalación

### Opción 1: Desde el Menú
1. Busca "Traductor LSE" en el menú de aplicaciones
2. Haz clic para abrir

### Opción 2: Desde el Escritorio
- Haz doble clic en el icono "Traductor LSE"

### Opción 3: Desde Terminal
```bash
~/TraductorLSE/lanzar.sh
```

---

## 🎓 Primera Vez - Entrenar o Usar Modelo

### Si ya tienes modelo entrenado:
- Simplemente abre "INICIAR TRADUCTOR"
- ¡Listo para traducir!

### Si necesitas entrenar:
1. Abre la aplicación
2. Selecciona **"🎓 ENTRENAR MODELO"**
3. Captura señas (30-40 muestras por seña)
4. Presiona "Entrenar"
5. Espera 2-5 minutos
6. ¡Ya puedes usar el traductor!

---

## 📋 Requisitos del Raspberry Pi

- ✅ **Raspberry Pi 4** (4GB RAM mínimo, 8GB recomendado)
- ✅ **Raspberry Pi OS** (Bullseye o Bookworm)
- ✅ **Cámara USB** o Raspberry Pi Camera Module
- ✅ **Conexión a internet** (solo durante instalación)
- ✅ **8GB de espacio libre** en SD

---

## 🔍 Verificar que todo funcione

Después de instalar, puedes verificar que las dependencias se instalaron correctamente:

```bash
# Activar entorno virtual
source ~/TraductorLSE/venv/bin/activate

# Verificar instalaciones
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
```

Deberías ver las versiones sin errores.

---

## ❌ NO Necesitas Hacer Esto

~~Instalar TensorFlow manualmente~~  
~~Instalar MediaPipe manualmente~~  
~~Configurar entornos virtuales~~  
~~Instalar dependencias del sistema~~  
~~Crear iconos manualmente~~  

**TODO SE HACE AUTOMÁTICAMENTE** ✅

---

## 🗑️ Desinstalar (si es necesario)

```bash
# Eliminar aplicación
rm -rf ~/TraductorLSE

# Eliminar accesos directos
rm ~/.local/share/applications/traductor-lse.desktop
rm ~/.local/share/icons/traductor-lse.svg
rm ~/Desktop/traductor-lse.desktop
```

---

## 📖 Documentación Adicional

Dentro del ZIP encontrarás:

- `INICIO_RAPIDO.txt` - Guía rápida
- `README.md` - Documentación completa
- `instalar_completo.sh` - El instalador automático

---

## 🤝 ¿Problemas?

### La cámara no funciona
```bash
# Verificar cámaras
v4l2-ctl --list-devices

# Si usas Pi Camera Module
sudo raspi-config
# → Interface Options → Camera → Enable
```

### Error de permisos
```bash
chmod +x instalar_completo.sh
```

### Pantalla negra en la aplicación
- Verifica que la cámara esté conectada
- Reinicia el Raspberry Pi
- Revisa permisos: `sudo usermod -aG video $USER`

---

## 🎉 ¡Listo!

Ahora tienes:

✅ **Paquete ZIP** listo para transferir  
✅ **Instalador automático** que NO requiere configuración  
✅ **Aplicación de escritorio** completa  
✅ **Modelo preentrenado** (si lo tenías)  
✅ **TODO incluido** - Sin instalar nada manual  

**Simplemente transfiere, descomprime y ejecuta el instalador.**

---

**Archivo a transferir:**  
`/Users/anahy/sign_language_project2/TraductorLSE-RaspberryPi-COMPLETO.zip`

**Comando para transferir:**  
```bash
scp TraductorLSE-RaspberryPi-COMPLETO.zip pi@raspberrypi.local:~/
```

🤟 **¡Disfruta traduciendo señas!**
