#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🔄 SINCRONIZADOR DE MODELOS - WEB ↔ RASPBERRY PI
═══════════════════════════════════════════════════════════════════════════════

Este script sincroniza los modelos entre:
- Versión Web (proyecto principal)
- Versión Standalone (Raspberry Pi)

USO:
  python sincronizar.py --desde-web     # Copiar modelo de web a standalone
  python sincronizar.py --hacia-web     # Copiar modelo de standalone a web

═══════════════════════════════════════════════════════════════════════════════
"""

import shutil
import argparse
from pathlib import Path
import pickle

# Rutas
SCRIPT_DIR = Path(__file__).parent
STANDALONE_MODEL = SCRIPT_DIR / "model"
STANDALONE_DATA = SCRIPT_DIR / "data"

# Buscar proyecto web
WEB_PROJECT_PATHS = [
    SCRIPT_DIR.parent / "backend" / "model",  # Si está dentro del proyecto
    Path.home() / "sign_language_project2" / "backend" / "model",
    Path("/home/pi/sign_language_project2/backend/model"),
]

def find_web_model():
    """Encuentra el modelo del proyecto web"""
    for path in WEB_PROJECT_PATHS:
        if path.exists() and (path / "labels.pkl").exists():
            return path
    return None

def copy_model(source: Path, dest: Path, name: str):
    """Copia archivos del modelo"""
    dest.mkdir(exist_ok=True, parents=True)
    
    files = ["labels.pkl", "best_model.h5", "model.tflite"]
    copied = 0
    
    for f in files:
        src_file = source / f
        if src_file.exists():
            shutil.copy2(src_file, dest / f)
            print(f"  ✅ {f}")
            copied += 1
    
    if copied > 0:
        # Mostrar señas
        labels_file = dest / "labels.pkl"
        if labels_file.exists():
            with open(labels_file, 'rb') as f:
                labels = pickle.load(f)
            print(f"\n📋 Señas en {name}: {', '.join(labels)}")
    
    return copied > 0

def desde_web():
    """Copia modelo desde web a standalone"""
    web_model = find_web_model()
    if not web_model:
        print("❌ No se encontró el modelo del proyecto web")
        print("   Ubicaciones buscadas:")
        for p in WEB_PROJECT_PATHS:
            print(f"     - {p}")
        return False
    
    print(f"📥 Copiando desde: {web_model}")
    print(f"📤 Hacia: {STANDALONE_MODEL}")
    print()
    
    return copy_model(web_model, STANDALONE_MODEL, "Standalone")

def hacia_web():
    """Copia modelo desde standalone a web"""
    if not (STANDALONE_MODEL / "labels.pkl").exists():
        print("❌ No hay modelo en standalone para copiar")
        return False
    
    web_model = find_web_model()
    if not web_model:
        # Intentar crear en ubicación por defecto
        web_model = SCRIPT_DIR.parent / "backend" / "model"
    
    print(f"📥 Copiando desde: {STANDALONE_MODEL}")
    print(f"📤 Hacia: {web_model}")
    print()
    
    return copy_model(STANDALONE_MODEL, web_model, "Web")

def mostrar_estado():
    """Muestra el estado de ambos modelos"""
    print("═" * 60)
    print("📊 ESTADO DE LOS MODELOS")
    print("═" * 60)
    
    # Standalone
    print("\n🔹 STANDALONE (Raspberry Pi):")
    if (STANDALONE_MODEL / "labels.pkl").exists():
        with open(STANDALONE_MODEL / "labels.pkl", 'rb') as f:
            labels = pickle.load(f)
        print(f"   ✅ {len(labels)} señas: {', '.join(labels)}")
    else:
        print("   ❌ Sin modelo")
    
    # Web
    print("\n🔹 WEB (Proyecto Principal):")
    web_model = find_web_model()
    if web_model and (web_model / "labels.pkl").exists():
        with open(web_model / "labels.pkl", 'rb') as f:
            labels = pickle.load(f)
        print(f"   ✅ {len(labels)} señas: {', '.join(labels)}")
        print(f"   📂 {web_model}")
    else:
        print("   ❌ Sin modelo o no encontrado")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='Sincronizador de modelos')
    parser.add_argument('--desde-web', action='store_true', help='Copiar modelo de web a standalone')
    parser.add_argument('--hacia-web', action='store_true', help='Copiar modelo de standalone a web')
    parser.add_argument('--estado', action='store_true', help='Mostrar estado de los modelos')
    
    args = parser.parse_args()
    
    print("🔄 SINCRONIZADOR DE MODELOS")
    print("═" * 60)
    
    if args.desde_web:
        success = desde_web()
        if success:
            print("\n✅ ¡Sincronización completada!")
    elif args.hacia_web:
        success = hacia_web()
        if success:
            print("\n✅ ¡Sincronización completada!")
    elif args.estado:
        mostrar_estado()
    else:
        mostrar_estado()
        print("Uso:")
        print("  python sincronizar.py --desde-web   # Web → Standalone")
        print("  python sincronizar.py --hacia-web   # Standalone → Web")
        print("  python sincronizar.py --estado      # Ver estado")

if __name__ == "__main__":
    main()
