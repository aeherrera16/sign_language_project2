#!/usr/bin/env python3
"""
SINCRONIZACIÓN EN LA NUBE - Firebase Firestore + Storage
=========================================================
Módulo de sincronización para el Traductor LSE.
Funciona en Windows, macOS y Raspberry Pi.

PRINCIPIO: El prototipo NUNCA depende de internet para traducir.
- La traducción funciona 100% offline con datos locales.
- Este módulo solo sincroniza cuando hay conexión disponible.
- Si no hay internet, simplemente no sincroniza (sin errores).

Funcionalidades:
- Subir modelo entrenado a la nube
- Descargar modelo actualizado desde la nube
- Sincronizar datos de señas (secuencias)
- Sincronizar métricas y metadatos
- Verificación de conexión
"""

import os
import sys
import json
import hashlib
import shutil
import base64
import time
from datetime import datetime, timezone

# Directorio del proyecto
DIR = os.path.dirname(os.path.abspath(__file__))
DIR_MODELO = os.path.join(DIR, "modelo")
DIR_DATOS = os.path.join(DIR, "datos")
CREDENTIALS_PATH = os.path.join(DIR, "firebase_credentials.json")

# Archivo local para rastrear estado de sincronización
SYNC_STATE_FILE = os.path.join(DIR, "modelo", ".sync_state.json")


# =============================================================================
# UTILIDADES
# =============================================================================

def hay_internet():
    """Verifica si hay conexión a internet (rápido, timeout 3s)."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except Exception:
        return False


def calcular_hash_archivo(filepath):
    """Calcula SHA256 de un archivo para detectar cambios."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for bloque in iter(lambda: f.read(8192), b''):
            sha256.update(bloque)
    return sha256.hexdigest()


def cargar_estado_sync():
    """Carga el estado de la última sincronización."""
    if os.path.exists(SYNC_STATE_FILE):
        try:
            with open(SYNC_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def guardar_estado_sync(estado):
    """Guarda el estado de sincronización."""
    os.makedirs(os.path.dirname(SYNC_STATE_FILE), exist_ok=True)
    with open(SYNC_STATE_FILE, 'w') as f:
        json.dump(estado, f, indent=2)


# =============================================================================
# CLASE PRINCIPAL DE SINCRONIZACIÓN
# =============================================================================

class SyncCloud:
    """
    Maneja la sincronización bidireccional con Firebase.
    
    Uso:
        sync = SyncCloud()
        if sync.conectar():
            sync.subir_modelo()
            sync.descargar_modelo_si_hay_nuevo()
            sync.sincronizar_datos()
    """

    # Firestore tiene límite de 1MB por documento.
    # Modelo .h5 (~2.5MB) se divide en chunks de 750KB.
    CHUNK_SIZE = 750_000  # bytes

    def __init__(self, credentials_path=None):
        self.credentials_path = credentials_path or CREDENTIALS_PATH
        self.db = None          # Firestore client
        self.conectado = False
        self._app = None

    def conectar(self):
        """
        Conecta a Firebase. Retorna True si la conexión fue exitosa.
        No lanza excepciones si falla - simplemente retorna False.
        """
        if self.conectado:
            return True

        # Verificar que existe el archivo de credenciales
        if not os.path.exists(self.credentials_path):
            print("⚠️  Archivo de credenciales no encontrado.")
            print(f"   Esperado en: {self.credentials_path}")
            print("   Consulta la guía de configuración de Firebase.")
            return False

        # Verificar internet
        if not hay_internet():
            print("📡 Sin conexión a internet. Trabajando en modo offline.")
            return False

        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            # Evitar inicializar múltiples veces
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.credentials_path)
                self._app = firebase_admin.initialize_app(cred)
            else:
                self._app = firebase_admin.get_app()

            self.db = firestore.client()
            self.conectado = True
            print("✅ Conectado a Firebase")
            return True

        except ImportError:
            print("⚠️  firebase-admin no instalado.")
            print("   Ejecuta: pip install firebase-admin")
            return False
        except Exception as e:
            print(f"⚠️  Error al conectar a Firebase: {e}")
            return False

    def desconectar(self):
        """Limpia la conexión."""
        self.conectado = False
        self.db = None

    # =========================================================================
    # UTILIDADES: Archivos en Firestore (sin Storage)
    # =========================================================================

    def _subir_archivo_firestore(self, filepath, coleccion, doc_id):
        """
        Sube un archivo binario a Firestore dividiéndolo en chunks base64.
        Firestore tiene límite de 1MB por documento, así que archivos
        grandes se dividen en múltiples documentos.
        """
        with open(filepath, 'rb') as f:
            datos = f.read()

        # Codificar en base64
        datos_b64 = base64.b64encode(datos).decode('utf-8')
        total_size = len(datos_b64)

        # Dividir en chunks
        chunks = []
        for i in range(0, total_size, self.CHUNK_SIZE):
            chunks.append(datos_b64[i:i + self.CHUNK_SIZE])

        # Guardar metadatos
        self.db.collection(coleccion).document(f"{doc_id}_meta").set({
            "nombre": os.path.basename(filepath),
            "size_original": len(datos),
            "num_chunks": len(chunks),
            "fecha": datetime.now(timezone.utc).isoformat(),
        })

        # Guardar cada chunk
        for idx, chunk in enumerate(chunks):
            self.db.collection(coleccion).document(f"{doc_id}_chunk_{idx}").set({
                "data": chunk,
                "index": idx,
            })

        return len(chunks)

    def _descargar_archivo_firestore(self, coleccion, doc_id, destino):
        """
        Descarga un archivo que fue subido en chunks base64 desde Firestore.
        """
        # Leer metadatos
        meta_doc = self.db.collection(coleccion).document(f"{doc_id}_meta").get()
        if not meta_doc.exists:
            return False

        meta = meta_doc.to_dict()
        num_chunks = meta.get("num_chunks", 0)

        # Leer y reconstruir chunks
        datos_b64 = ""
        for idx in range(num_chunks):
            chunk_doc = self.db.collection(coleccion).document(f"{doc_id}_chunk_{idx}").get()
            if not chunk_doc.exists:
                print(f"   ⚠️  Chunk {idx} no encontrado")
                return False
            datos_b64 += chunk_doc.to_dict().get("data", "")

        # Decodificar y guardar
        datos = base64.b64decode(datos_b64)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        with open(destino, 'wb') as f:
            f.write(datos)

        return True

    # =========================================================================
    # MODELO: Subir / Descargar
    # =========================================================================

    def subir_modelo(self):
        """
        Sube el modelo entrenado (modelo.h5 + encoder.pkl + info.json) a Firestore.
        Los archivos binarios se codifican en base64 y se dividen en chunks.
        Solo sube si el modelo local cambió desde la última sincronización.
        """
        if not self.conectado:
            print("⚠️  No conectado a Firebase")
            return False

        modelo_path = os.path.join(DIR_MODELO, "modelo.h5")
        encoder_path = os.path.join(DIR_MODELO, "encoder.pkl")
        info_path = os.path.join(DIR_MODELO, "info.json")

        if not os.path.exists(modelo_path):
            print("❌ No hay modelo local para subir")
            return False

        # Calcular hash para verificar si cambió
        hash_actual = calcular_hash_archivo(modelo_path)
        estado = cargar_estado_sync()

        if estado.get("ultimo_hash_modelo_subido") == hash_actual:
            print("ℹ️  Modelo ya está sincronizado (sin cambios)")
            return True

        print("☁️  Subiendo modelo a Firestore...")

        try:
            # Subir modelo.h5 en chunks
            n = self._subir_archivo_firestore(modelo_path, "archivos_modelo", "modelo_h5")
            print(f"   ✓ modelo.h5 subido ({n} chunks)")

            # Subir encoder.pkl en chunks
            if os.path.exists(encoder_path):
                n = self._subir_archivo_firestore(encoder_path, "archivos_modelo", "encoder_pkl")
                print(f"   ✓ encoder.pkl subido ({n} chunks)")

            # Guardar info.json como documento directo (es pequeño)
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info_data = json.load(f)
                self.db.collection("archivos_modelo").document("info_json").set(info_data)
                print("   ✓ info.json subido")
            else:
                info_data = {}

            # Guardar metadatos del modelo
            self.db.collection("modelos").document("actual").set({
                "version": hash_actual[:8],
                "hash_completo": hash_actual,
                "fecha_subida": datetime.now(timezone.utc).isoformat(),
                "accuracy_test": info_data.get("accuracy_test", info_data.get("accuracy", 0)),
                "clases": info_data.get("clases", []),
                "frames": info_data.get("frames", 30),
                "features": info_data.get("features", 126),
                "subido_desde": sys.platform,
            })
            print("   ✓ Metadatos guardados")

            # Actualizar estado local
            estado["ultimo_hash_modelo_subido"] = hash_actual
            estado["ultima_subida"] = datetime.now().isoformat()
            guardar_estado_sync(estado)

            print(f"✅ Modelo subido exitosamente (versión: {hash_actual[:8]})")
            return True

        except Exception as e:
            print(f"❌ Error al subir modelo: {e}")
            return False

    def descargar_modelo_si_hay_nuevo(self):
        """
        Verifica si hay un modelo más nuevo en la nube y lo descarga.
        Compara hash local vs hash en la nube.
        Retorna True si se descargó un nuevo modelo, False si no hay cambios.
        """
        if not self.conectado:
            return False

        try:
            # Obtener info del modelo en la nube
            doc = self.db.collection("modelos").document("actual").get()
            if not doc.exists:
                print("ℹ️  No hay modelo en la nube todavía")
                return False

            cloud_data = doc.to_dict()
            cloud_hash = cloud_data.get("hash_completo", "")

            # Comparar con modelo local
            modelo_path = os.path.join(DIR_MODELO, "modelo.h5")
            if os.path.exists(modelo_path):
                local_hash = calcular_hash_archivo(modelo_path)
                if local_hash == cloud_hash:
                    print("ℹ️  Modelo local está actualizado")
                    return False

            print("📥 Modelo nuevo detectado en la nube. Descargando...")
            print(f"   Versión: {cloud_hash[:8]}")
            print(f"   Clases: {cloud_data.get('clases', [])}")
            print(f"   Accuracy: {cloud_data.get('accuracy_test', 0):.1%}")

            # Crear directorio si no existe
            os.makedirs(DIR_MODELO, exist_ok=True)

            # Backup del modelo actual
            if os.path.exists(modelo_path):
                backup_path = modelo_path + ".backup"
                shutil.copy2(modelo_path, backup_path)
                print("   ✓ Backup del modelo anterior creado")

            # Descargar modelo.h5 desde Firestore chunks
            if self._descargar_archivo_firestore("archivos_modelo", "modelo_h5", modelo_path):
                print("   ✓ modelo.h5 descargado")
            else:
                print("   ⚠️  No se pudo descargar modelo.h5")
                return False

            # Descargar encoder.pkl
            encoder_path = os.path.join(DIR_MODELO, "encoder.pkl")
            if self._descargar_archivo_firestore("archivos_modelo", "encoder_pkl", encoder_path):
                print("   ✓ encoder.pkl descargado")

            # Descargar info.json
            info_doc = self.db.collection("archivos_modelo").document("info_json").get()
            if info_doc.exists:
                info_path = os.path.join(DIR_MODELO, "info.json")
                with open(info_path, 'w') as f:
                    json.dump(info_doc.to_dict(), f, indent=2)
                print("   ✓ info.json descargado")

            # Actualizar estado
            estado = cargar_estado_sync()
            estado["ultimo_hash_modelo_descargado"] = cloud_hash
            estado["ultima_descarga"] = datetime.now().isoformat()
            guardar_estado_sync(estado)

            print(f"✅ Modelo actualizado desde la nube (versión: {cloud_hash[:8]})")
            return True

        except Exception as e:
            print(f"⚠️  Error al verificar/descargar modelo: {e}")
            return False

    # =========================================================================
    # DATOS DE SEÑAS: Sincronización
    # =========================================================================

    def subir_datos_senas(self):
        """
        Sube los datos de secuencias de señas a Firestore usando chunks.
        Los archivos JSON son grandes (~1.5MB) así que se usa el sistema de chunks.
        Solo sube archivos que no se hayan subido antes.
        """
        if not self.conectado:
            return False

        if not os.path.isdir(DIR_DATOS):
            print("❌ No hay datos de señas para subir")
            return False

        estado = cargar_estado_sync()
        archivos_subidos = estado.get("archivos_datos_subidos", [])
        nuevos = 0

        print("☁️  Sincronizando datos de señas...")

        for sena_dir in sorted(os.listdir(DIR_DATOS)):
            sena_path = os.path.join(DIR_DATOS, sena_dir)
            if not os.path.isdir(sena_path):
                continue

            for archivo in sorted(os.listdir(sena_path)):
                if not archivo.endswith('.json'):
                    continue

                archivo_rel = f"{sena_dir}/{archivo}"
                if archivo_rel in archivos_subidos:
                    continue

                filepath = os.path.join(sena_path, archivo)

                try:
                    # Usar sistema de chunks (los JSON son >1MB)
                    doc_id = archivo_rel.replace('/', '__').replace('.json', '')
                    self._subir_archivo_firestore(filepath, "datos_senas", doc_id)

                    # Guardar referencia para descarga
                    self.db.collection("datos_senas").document(f"{doc_id}_ref").set({
                        "sena": sena_dir,
                        "archivo": archivo,
                        "archivo_rel": archivo_rel,
                        "fecha_sync": datetime.now(timezone.utc).isoformat(),
                    })

                    archivos_subidos.append(archivo_rel)
                    nuevos += 1
                except Exception as e:
                    print(f"   ⚠️  Error subiendo {archivo_rel}: {e}")

        # Actualizar catálogo de señas
        try:
            senas_info = {}
            for sena_dir in sorted(os.listdir(DIR_DATOS)):
                sena_path = os.path.join(DIR_DATOS, sena_dir)
                if os.path.isdir(sena_path):
                    archivos = [f for f in os.listdir(sena_path) if f.endswith('.json')]
                    senas_info[sena_dir] = len(archivos)

            self.db.collection("catalogo").document("senas").set({
                "senas": senas_info,
                "total_archivos": sum(senas_info.values()),
                "ultima_actualizacion": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass

        # Guardar estado
        estado["archivos_datos_subidos"] = archivos_subidos
        estado["ultima_sync_datos"] = datetime.now().isoformat()
        guardar_estado_sync(estado)

        if nuevos > 0:
            print(f"✅ {nuevos} archivos de datos nuevos subidos")
        else:
            print("ℹ️  Datos de señas ya sincronizados")

        return True

    def descargar_datos_senas(self):
        """
        Descarga datos de señas desde Firestore que no existen localmente.
        Útil para sincronizar datos entre dispositivos.
        """
        if not self.conectado:
            return False

        print("📥 Verificando datos de señas en la nube...")

        try:
            # Buscar documentos de referencia (_ref)
            docs = self.db.collection("datos_senas").where("archivo_rel", "!=", "").stream()
            descargados = 0

            for doc in docs:
                if not doc.id.endswith("_ref"):
                    continue

                data = doc.to_dict()
                sena = data.get("sena", "")
                archivo = data.get("archivo", "")
                archivo_rel = data.get("archivo_rel", "")

                if not sena or not archivo:
                    continue

                local_path = os.path.join(DIR_DATOS, sena, archivo)

                if os.path.exists(local_path):
                    continue

                # Descargar usando chunks
                doc_id = archivo_rel.replace('/', '__').replace('.json', '')
                os.makedirs(os.path.join(DIR_DATOS, sena), exist_ok=True)

                if self._descargar_archivo_firestore("datos_senas", doc_id, local_path):
                    descargados += 1

            if descargados > 0:
                print(f"✅ {descargados} archivos de datos descargados")
            else:
                print("ℹ️  Datos locales están completos")

            return True

        except Exception as e:
            print(f"⚠️  Error descargando datos: {e}")
            return False

    # =========================================================================
    # MÉTRICAS: Sincronizar evaluación ISO
    # =========================================================================

    def subir_metricas(self):
        """Sube las métricas de evaluación ISO a Firestore."""
        if not self.conectado:
            return False

        eval_path = os.path.join(DIR_MODELO, "evaluacion_iso25023.json")
        if not os.path.exists(eval_path):
            print("ℹ️  Sin métricas de evaluación para subir")
            return False

        try:
            with open(eval_path, 'r') as f:
                metricas = json.load(f)

            self.db.collection("metricas").document("evaluacion_iso25023").set({
                **metricas,
                "fecha_sync": datetime.now(timezone.utc).isoformat(),
                "dispositivo": sys.platform,
            })

            print("✅ Métricas de evaluación subidas a la nube")
            return True

        except Exception as e:
            print(f"⚠️  Error subiendo métricas: {e}")
            return False

    # =========================================================================
    # SINCRONIZACIÓN COMPLETA
    # =========================================================================

    def sincronizar_todo(self):
        """
        Sincronización completa bidireccional:
        1. Descarga modelo nuevo si existe
        2. Sube modelo local si es más nuevo
        3. Sincroniza datos de señas en ambas direcciones
        4. Sube métricas
        """
        if not self.conectar():
            return False

        print("\n" + "=" * 60)
        print("  SINCRONIZACIÓN CON LA NUBE")
        print("=" * 60)

        # 1. Verificar si hay modelo nuevo en la nube
        modelo_descargado = self.descargar_modelo_si_hay_nuevo()

        # 2. Si no se descargó nada nuevo, subir el local
        if not modelo_descargado:
            self.subir_modelo()

        # 3. Sincronizar datos de señas (bidireccional)
        print()
        self.descargar_datos_senas()
        self.subir_datos_senas()

        # 4. Subir métricas
        print()
        self.subir_metricas()

        print("\n" + "=" * 60)
        print("  ✅ SINCRONIZACIÓN COMPLETADA")
        print("=" * 60 + "\n")

        return True

    # =========================================================================
    # VERIFICACIÓN
    # =========================================================================

    def verificar(self):
        """Verifica que la conexión a Firebase funcione correctamente."""
        print("\n🔍 Verificando conexión a Firebase...\n")

        if not self.conectar():
            print("❌ No se pudo conectar")
            return False

        # Test Firestore
        try:
            self.db.collection("_test").document("ping").set({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "from": sys.platform,
            })
            # Limpiar documento de test
            self.db.collection("_test").document("ping").delete()
            print("✅ Firestore: OK")
        except Exception as e:
            print(f"❌ Firestore: Error - {e}")
            return False

        # Info del proyecto
        try:
            with open(self.credentials_path, 'r') as f:
                cred_data = json.load(f)
            print(f"\n📋 Proyecto: {cred_data.get('project_id', 'N/A')}")
        except Exception:
            pass

        print("\n✅ Conexión a Firebase establecida correctamente\n")
        return True

    def obtener_estado_sync(self):
        """Retorna un resumen del estado de sincronización para la UI."""
        estado = cargar_estado_sync()
        
        resultado = {
            "conectado": self.conectado,
            "internet": hay_internet(),
            "credenciales": os.path.exists(self.credentials_path),
            "ultima_subida": estado.get("ultima_subida", "Nunca"),
            "ultima_descarga": estado.get("ultima_descarga", "Nunca"),
            "ultima_sync_datos": estado.get("ultima_sync_datos", "Nunca"),
        }
        
        return resultado


# =============================================================================
# CLI - Uso desde terminal
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sincronización Firebase - Traductor LSE')
    parser.add_argument('--verificar', action='store_true', help='Verificar conexión a Firebase')
    parser.add_argument('--subir-modelo', action='store_true', help='Subir modelo a la nube')
    parser.add_argument('--descargar-modelo', action='store_true', help='Descargar modelo de la nube')
    parser.add_argument('--sync-datos', action='store_true', help='Sincronizar datos de señas')
    parser.add_argument('--sync-todo', action='store_true', help='Sincronización completa')
    parser.add_argument('--estado', action='store_true', help='Ver estado de sincronización')
    parser.add_argument('--credenciales', type=str, help='Ruta al archivo de credenciales')
    args = parser.parse_args()

    cred_path = args.credenciales if args.credenciales else None
    sync = SyncCloud(credentials_path=cred_path)

    if args.verificar:
        sync.verificar()
    elif args.subir_modelo:
        if sync.conectar():
            sync.subir_modelo()
    elif args.descargar_modelo:
        if sync.conectar():
            sync.descargar_modelo_si_hay_nuevo()
    elif args.sync_datos:
        if sync.conectar():
            sync.descargar_datos_senas()
            sync.subir_datos_senas()
    elif args.sync_todo:
        sync.sincronizar_todo()
    elif args.estado:
        estado = sync.obtener_estado_sync()
        print("\n📊 Estado de sincronización:")
        print(f"   Internet:      {'✅ Sí' if estado['internet'] else '❌ No'}")
        print(f"   Credenciales:  {'✅ Sí' if estado['credenciales'] else '❌ No'}")
        print(f"   Última subida: {estado['ultima_subida']}")
        print(f"   Última descarga: {estado['ultima_descarga']}")
        print(f"   Última sync datos: {estado['ultima_sync_datos']}")
    else:
        # Sin argumentos: hacer sync completo
        sync.sincronizar_todo()


if __name__ == "__main__":
    main()
