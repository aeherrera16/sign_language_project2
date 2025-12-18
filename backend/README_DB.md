Esquema de base de datos (SQLite) — gestures / captures

Tablas principales

1) gestures
- id: TEXT PRIMARY KEY (uuid)
- name: TEXT NOT NULL
- description: TEXT
- created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP

2) captures
- id: TEXT PRIMARY KEY (uuid)
- gesture_id: TEXT NULL (foreign key to gestures.id)
- filename: TEXT (ruta relativa en uploads/gestures)
- metadata: TEXT (JSON string con landmarks, score, etc)
- created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP

Cómo usar
- Crear una seña: POST /api/gestures { name, description }
- Guardar captura: POST /api/capture/save (multipart) fields:
  - image: file
  - gesture_id: optional string
  - metadata: optional JSON string

Notas
- Prototipo usa SQLite en `data/gestures/db.sqlite` y archivos en `uploads/gestures`.
- Para producción usar PostgreSQL y un almacenamiento de objetos (S3) para los archivos.
