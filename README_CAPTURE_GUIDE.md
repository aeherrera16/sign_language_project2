Guía de captura de alta calidad — para entrenar el modelo

Objetivo: obtener muestras consistentes y de buena calidad aun con cámaras de baja resolución.

Recomendaciones generales
- Resolución mínima: 480x360. Si la cámara entrega menos, el sistema hará un upscale y preprocesado.
- Distancia a cámara: 30-60 cm.
- Iluminación: luz frontal o lateral suave; evitar contraluz.
- Fondo: intentar fondo lo más uniforme posible.
- Movimiento: mantener la postura de la seña por 1-2 segundos antes de capturar.
- Muestras por seña: al menos 200 por seña para un modelo competitivo; al menos 50 para pruebas.

Umbrales del sistema (configurables)
- hands_percentage (porcentaje área manos): > 15% = excelente, 8-15% = buena, 5-8% = regular
- blur_score (Laplacian var / factor): >70 = excelente, 50-70 = buena, 30-50 = regular
- score final mínimo para guardar: 70 (configurable en frontend `CapturaIA.jsx`)

Flujo recomendado para etiquetado
1) Crear o seleccionar la seña en el backend (POST /api/gestures)
2) En Captura IA: visualizar overlay de segmentación y asegurarse de que manos ocupen suficiente área
3) Presionar "Guardar Seña" — el frontend validará score >= 70 antes de subir
4) Revisar capturas en `uploads/gestures/` y en la DB `data/gestures/db.sqlite`

Mejoras futuras
- Integrar un modelo de super-resolution si las cámaras son consistentemente muy bajas (ESRGAN o Real-ESRGAN).
- Añadir UI de revisión para aceptar/rechazar manualmente capturas.
- Automatizar augmentations y creación de dataset listo para entrenamiento (TFRecord / folders por label).
