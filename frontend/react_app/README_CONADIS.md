CONADIS Diccionario — integración

Este proyecto usa el diccionario oficial de CONADIS (Ecuador) como referencia primaria para mapear y estandarizar las señas.

Fuente: http://www.plataformaconadis.gob.ec/~platafor/diccionario/

Cómo usarlo en el flujo del proyecto
- Revisa las entradas del diccionario para identificar las señas principales que quieres soportar.
- Crea una entrada (gesture) en el backend con el nombre y descripción.
- Usa la interfaz "Captura IA" para grabar múltiples muestras de cada seña, etiquetándolas con la `gesture_id` devuelta por el endpoint.

Endpoints útiles (backend)
- POST /api/gestures  -> crear una nueva seña { name, description }
- POST /api/capture/save -> subir una captura (multipart/form-data) fields: image (file), gesture_id (string), metadata (string)

Consejo: crea una lista priorizada (top 50) de señas del diccionario y empieza la recolección con ellas para obtener una base sólida de entrenamiento.
