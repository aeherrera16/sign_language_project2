# 🇪🇨 LSE ECUADOR - Sistema de Reconocimiento de Señas

## ✅ SISTEMA COMPLETAMENTE FUNCIONAL

Sistema de reconocimiento de Lengua de Señas Ecuatoriana (LSE) con síntesis de voz automática en tiempo real.

### 🚀 INICIO RÁPIDO

#### **Opción 1: Super Fácil (RECOMENDADO)**
```
📁 Doble clic en: EJECUTAR_RECONOCIMIENTO.bat
🎥 Sistema iniciado automáticamente
```

#### **Opción 2: Interfaz Completa**
```  
📁 Doble clic en: EJECUTAR_LSE.bat
🖥️ Interfaz gráfica elegante
```

#### **Opción 3: Terminal**
```powershell
.\venv310\Scripts\activate
python scripts\recognition\real_time_translate.py
```

### 🎯 GESTOS DISPONIBLES

- 👋 **hola** - Saludo básico LSE
- 🙏 **gracias** - Agradecimiento  
- ✅ **si** - Afirmación
- ❌ **no** - Negación
- 👋 **adios** - Despedida

### 📁 ESTRUCTURA DEL PROYECTO

```
📂 sign_language_project2Modf/
├── 📄 EJECUTAR_RECONOCIMIENTO.bat  # Inicio directo
├── 📄 EJECUTAR_LSE.bat            # Interfaz completa
├── 📄 main_interface.py           # Interfaz principal
├── 📄 README.md                   # Esta documentación
├── 📄 requirements.txt            # Dependencias
├── 📄 utils.py                    # Utilidades
├── 📄 configuracion_rapida.py     # Setup inicial
├── 📄 verificacion_final.py       # Verificación
├── 📂 model/                      # Modelo entrenado
│   ├── gesture_model.h5           # Modelo de ML
│   └── labels.pkl                 # Etiquetas de gestos
├── 📂 scripts/                    # Scripts del sistema
│   └── recognition/
│       └── real_time_translate.py # Reconocimiento en tiempo real
├── 📂 data/                       # Datos de entrenamiento
├── 📂 venv310/                    # Entorno virtual Python
└── 📂 .git/                       # Control de versiones
```

### 📊 CARACTERÍSTICAS

- ✅ **Reconocimiento en tiempo real** a 30 FPS
- ✅ **Síntesis de voz automática** en español
- ✅ **Sin warnings de TensorFlow** - ejecución limpia
- ✅ **Umbral optimizado** (35% sensibilidad)
- ✅ **Modelo entrenado** con 5 gestos LSE
- ✅ **Interfaz elegante** y fácil de usar

### 🛠️ REQUISITOS

- Windows 10/11
- Python 3.10
- Webcam
- Altavoces/Audífonos
- 4GB RAM mínimo

### 🎮 INSTRUCCIONES DE USO

1. **🎥 Ejecuta** cualquier método de inicio
2. **👋 Realiza gestos** frente a la cámara
3. **🔊 Escucha** la pronunciación automática
4. **⏹️ Presiona 'q'** para salir

### 🔧 INSTALACIÓN (Si es necesario)

```powershell
# Clonar repositorio
git clone [url-del-repo]
cd sign_language_project2Modf

# Ejecutar setup automático
python configuracion_rapida.py

# Verificar instalación
python verificacion_final.py
```

### 🎯 PRÓXIMOS PASOS

- 🍓 Optimización para Raspberry Pi 3
- 📚 Más gestos específicos de LSE Ecuador  
- 🎯 Mejor precisión con más datos
- 💬 Reconocimiento de frases completas

### 📞 SOPORTE

Si tienes problemas:
1. Ejecuta `python verificacion_final.py` para diagnóstico
2. Verifica que la cámara esté conectada
3. Asegúrate de tener altavoces configurados

---

## 🎉 ¡LISTO PARA USAR!

**Tu traductor de señas LSE Ecuador está funcionando al 100%**

*🇪🇨 Desarrollado para la comunidad sorda de Ecuador* ❤️
