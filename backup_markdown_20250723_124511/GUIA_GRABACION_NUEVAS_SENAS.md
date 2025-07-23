# 📹 GUÍA PARA GRABAR NUEVAS SEÑAS LSE ECUADOR

## ✅ SISTEMA PREPARADO
- ✅ **Data anterior eliminada** (backup guardado)
- ✅ **Modelo anterior eliminado**
- ✅ **Carpetas vacías creadas** para nuevas señas
- ✅ **Interface abierta** y lista para usar

---

## 🎯 SEÑAS LSE ECUADOR A GRABAR

### 🖐️ **1. HOLA**
- **Descripción:** Mano abierta hacia adelante
- **Detalles:** Palma hacia la persona, movimiento suave hacia adelante
- **Objetivo:** 30+ ejemplos

### 👋 **2. ADIOS**
- **Descripción:** Mano abierta, movimiento lateral
- **Detalles:** Movimiento de lado a lado (izquierda-derecha)
- **Objetivo:** 30+ ejemplos

### 🙏 **3. GRACIAS**
- **Descripción:** Mano al pecho o hacia la persona
- **Detalles:** Gesto de agradecimiento con la mano hacia el pecho
- **Objetivo:** 30+ ejemplos

### 👍 **4. SÍ**
- **Descripción:** Puño cerrado, movimiento vertical
- **Detalles:** Movimiento arriba-abajo con el puño cerrado
- **Objetivo:** 30+ ejemplos

### 👎 **5. NO**
- **Descripción:** Dedo índice extendido, movimiento horizontal
- **Detalles:** Movimiento izquierda-derecha con el dedo índice extendido
- **Objetivo:** 30+ ejemplos

---

## 📝 PROCESO DE GRABACIÓN

### **Paso 1: Abrir Grabador**
1. En la interface principal, clic en **"Grabar Dataset"**
2. Selecciona el gesto que quieres grabar
3. Prepárate frente a la cámara

### **Paso 2: Preparación**
- ✅ **Buena iluminación** (evitar sombras)
- ✅ **Fondo despejado** (sin distracciones)
- ✅ **Manos visibles** completamente
- ✅ **Cámara estable** a altura correcta

### **Paso 3: Grabación**
- 🎯 **Haz la seña LSE correcta**
- ⏱️ **Mantén por 2-3 segundos**
- 🔄 **Varía ligeramente** (ángulos, velocidad)
- 📊 **Graba mínimo 30 ejemplos** por gesto

### **Paso 4: Repetir**
- Graba **TODOS** los 5 gestos
- Asegúrate de tener **30+ ejemplos** de cada uno
- **Calidad** es más importante que cantidad

---

## ⚡ CONSEJOS IMPORTANTES

### 🎨 **Variación en Grabación**
- **Ángulos diferentes:** Ligeramente hacia izquierda/derecha
- **Velocidades:** Rápido y lento
- **Intensidad:** Movimientos sutiles y marcados
- **Posiciones:** Diferentes alturas de la mano

### 🚫 **Errores a Evitar**
- ❌ Gestos inventados (usar solo LSE Ecuador real)
- ❌ Movimientos muy rápidos o borrosos
- ❌ Manos fuera del cuadro
- ❌ Iluminación muy oscura
- ❌ Menos de 30 ejemplos por gesto

### ✅ **Para Mejores Resultados**
- ✅ Usa **ambas manos** cuando sea natural
- ✅ **Mantén el gesto** unos segundos
- ✅ **Expresión facial** neutral
- ✅ **Movimientos fluidos** y naturales

---

## 🔄 DESPUÉS DE LA GRABACIÓN

### **1. Entrenar Modelo**
```bash
# En la interface principal
Clic en "Entrenar Modelo"
```

### **2. Verificar Sistema**
```bash
python verificacion_sistema_completo.py
```

### **3. Probar Reconocimiento**
```bash
python scripts/recognition/real_time_translate.py
```

---

## 📊 VERIFICACIÓN DE PROGRESO

### **Durante la Grabación:**
- Cuenta cuántos ejemplos has grabado
- Verifica que cada gesto se guarde correctamente
- Revisa que las carpetas tengan archivos .npy

### **Después de la Grabación:**
```bash
# Verificar archivos creados
ls data/hola/     # Debe tener 30+ archivos .npy
ls data/adios/    # Debe tener 30+ archivos .npy
ls data/gracias/  # Debe tener 30+ archivos .npy
ls data/si/       # Debe tener 30+ archivos .npy
ls data/no/       # Debe tener 30+ archivos .npy
```

---

## 🎊 ¡ESTÁS LISTO!

**La interface ya está abierta y esperando.** 

**Pasos inmediatos:**
1. 🖱️ **Clic en "Grabar Dataset"** en la interface
2. 📹 **Selecciona "hola"** para empezar
3. 🖐️ **Haz la seña LSE Ecuador correcta**
4. 🔄 **Repite 30+ veces** con variaciones
5. ➡️ **Continúa con los otros 4 gestos**

**¡Tu nuevo sistema LSE personalizado está a punto de nacer!** 🚀
