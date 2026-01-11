# Traductor LSE - Prototipo Funcional

Sistema de traducción de Lengua de Señas Ecuatoriana (LSE) a texto y voz.

**Objetivo 1**: Prototipo funcional en computadora para reconocimiento de señas orientado a noticias.

## Uso Rápido

```bash
cd prototipo

# 1. Grabar señas (mínimo 2 señas, 30 secuencias c/u)
python 1_grabar_senas.py

# 2. Entrenar modelo
python 2_entrenar_modelo.py

# 3. Usar traductor
python 3_traductor.py
```

## Requisitos

```bash
pip install opencv-python mediapipe tensorflow pyttsx3 scikit-learn
```

## Documentación

- `prototipo/README.md` - Instrucciones detalladas
- `anexo_1_formato_nota_conceptual-docente.docx` - Nota conceptual del proyecto
