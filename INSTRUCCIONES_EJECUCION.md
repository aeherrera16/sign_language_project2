# 🎯 INSTRUCCIONES DE EJECUCIÓN - LENGUAJE DE SEÑAS

## 🚀 **PASOS SIMPLES PARA EJECUTAR**

### ✅ **PASO 1: Preparar el entorno**
```powershell
# Abrir PowerShell y navegar al proyecto
cd "c:\Users\Anahy\Desktop\sign_language_project2Modf"

# Activar el entorno virtual
.\venv310\Scripts\activate
```

### ✅ **PASO 2: Ejecutar la aplicación**
```powershell
# Ejecutar la interfaz principal
python main_interface.py
```

## 🎮 **OPCIONES DE EJECUCIÓN**

### 🖥️ **Opción 1: Interfaz Completa (Recomendado)**
```powershell
python main_interface.py
```
- ✨ Interfaz gráfica amigable
- 📁 Gestión completa de datos
- 🧠 Entrenamiento de modelos
- 📊 Evaluación de rendimiento
- 🎥 Reconocimiento en tiempo real

### 🎥 **Opción 2: Solo Reconocimiento**
```powershell
# Reconocimiento básico
python real_time_improved.py

# Reconocimiento con voz
python real_time_translate.py
```

## 🔧 **SOLUCIÓN DE PROBLEMAS**

### ❗ **Si sale error de dependencias:**
```powershell
# Verificar instalación
python test_imports_improved.py

# Reinstalar dependencias si es necesario
pip install -r requirements.txt
```

### ❗ **Si no funciona la cámara:**
- Verificar que ninguna otra aplicación esté usando la cámara
- Cerrar Zoom, Teams, Skype, etc.
- Reiniciar y volver a intentar

### ❗ **Si el entorno virtual no se activa:**
```powershell
# Crear nuevo entorno virtual
python -m venv venv310
.\venv310\Scripts\activate
pip install -r requirements.txt
```

## 💡 **CONSEJOS DE USO**

### 🎯 **Para mejor reconocimiento:**
- 🌟 Usar buena iluminación
- 🖐️ Mantener las manos dentro del marco
- 🎨 Fondo contrastante (pared clara)
- 📏 Distancia de 50-80cm de la cámara

### 🔊 **Para síntesis de voz:**
- 🎧 Conectar altavoces o audífonos
- 🔈 Verificar que el volumen esté activo
- 🗣️ El sistema leerá los gestos reconocidos

## 📞 **CONTACTO**
Si tienes problemas, revisa que:
1. ✅ Python 3.8+ esté instalado
2. ✅ La cámara funcione correctamente
3. ✅ El entorno virtual esté activo
4. ✅ Todas las dependencias estén instaladas

---
**🎉 ¡Disfruta usando el reconocimiento de lenguaje de señas!**
