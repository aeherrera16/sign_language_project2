from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import sqlite3
import os
import uuid
from typing import Optional

router = APIRouter()
DB_PATH = 'data/gestures/db.sqlite'
UPLOADS_DIR = 'uploads/gestures'

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS gestures (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS captures (
        id TEXT PRIMARY KEY,
        gesture_id TEXT,
        filename TEXT,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(gesture_id) REFERENCES gestures(id)
    )
    ''')
    conn.commit()
    conn.close()

init_db()

class GestureCreate(BaseModel):
    name: str
    description: Optional[str]

@router.post('/api/gestures', tags=['gestures'])
async def create_gesture(payload: GestureCreate):
    gid = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('INSERT INTO gestures(id,name,description) VALUES (?,?,?)', (gid, payload.name, payload.description))
    conn.commit()
    conn.close()
    return {'id': gid, 'name': payload.name}


@router.get('/api/gestures', tags=['gestures'])
async def list_gestures():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, name, description, created_at FROM gestures ORDER BY name ASC')
    rows = cur.fetchall()
    conn.close()
    gestures = []
    for r in rows:
        gestures.append({'id': r[0], 'name': r[1], 'description': r[2], 'created_at': r[3]})
    return gestures


@router.get('/api/gestures/{gesture_id}', tags=['gestures'])
async def get_gesture(gesture_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, name, description, created_at FROM gestures WHERE id = ?', (gesture_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail='Gesture not found')
    return {'id': row[0], 'name': row[1], 'description': row[2], 'created_at': row[3]}

@router.post('/api/capture/save', tags=['gestures'])
async def save_capture(image: UploadFile = File(...), gesture_id: Optional[str] = Form(None), metadata: Optional[str] = Form(None)):
    # save file
    ext = os.path.splitext(image.filename)[1] or '.jpg'
    cid = str(uuid.uuid4())
    filename = f"{cid}{ext}"
    path = os.path.join(UPLOADS_DIR, filename)
    with open(path, 'wb') as f:
        f.write(await image.read())

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('INSERT INTO captures(id,gesture_id,filename,metadata) VALUES (?,?,?,?)', (cid, gesture_id, filename, metadata))
    conn.commit()
    conn.close()

    return { 'id': cid, 'filename': filename, 'gesture_id': gesture_id }
