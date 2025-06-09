import os
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np

load_dotenv()

try:
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("COLLECTION_NAME")]
    
    # Test de conexión
    db.command("ping")
    print("✅ ¡Todo funciona correctamente!")
    print(f"Base de datos: {db.name}")
    print(f"Colección: {collection.name}")
    
    # Insertar dato de prueba
    test_landmarks = np.random.rand(21, 3).astype(np.float16)
    collection.insert_one({
        "name": "test",
        "landmarks": test_landmarks.tolist()
    })
    print("📝 Documento de prueba insertado")
    
except Exception as e:
    print("❌ Error:", e)