#!/usr/bin/env python3
"""
MODO KIOSKO - Traductor LSE directo (sin menú)
===============================================
Para Raspberry Pi: arranca directo al traductor.
1. Auto-sincroniza modelo y datos desde la nube (si hay internet)
2. Lanza el traductor inmediatamente
3. Si el traductor se cierra, vuelve a abrirlo (loop infinito)

Uso:
    python3 modo_traductor.py          # Ejecuta una vez
    python3 modo_traductor.py --loop   # Se reinicia solo al cerrarse
"""

import os
import sys
import time

DIR = os.path.dirname(os.path.abspath(__file__))
DIR_MODELO = os.path.join(DIR, "modelo")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'


def auto_sync():
    """Sincroniza modelo y datos desde la nube (silencioso)."""
    try:
        from sync_cloud import SyncCloud
        sync = SyncCloud()
        if sync.conectar():
            print("☁️  Sincronizando con la nube...")

            # Descargar modelo nuevo si existe
            descargado = sync.descargar_modelo_si_hay_nuevo()
            if descargado:
                print("☁️  ✅ Modelo actualizado desde la nube")

            # Descargar datos nuevos
            sync.descargar_datos_senas()

            print("☁️  Sincronización completada")
        else:
            print("☁️  Sin conexión. Usando datos locales.")
    except ImportError:
        print("☁️  firebase-admin no instalado. Modo offline.")
    except Exception as e:
        print(f"☁️  Error de sync (no crítico): {e}")


def verificar_modelo():
    """Verifica que existe un modelo entrenado."""
    modelo_path = os.path.join(DIR_MODELO, "modelo.h5")
    encoder_path = os.path.join(DIR_MODELO, "encoder.pkl")

    if not os.path.exists(modelo_path) or not os.path.exists(encoder_path):
        print("\n" + "=" * 50)
        print("  ❌ NO HAY MODELO ENTRENADO")
        print("=" * 50)
        print("\n  Para usar el traductor necesitas:")
        print("  1. Grabar señas:  python3 1_grabar_senas.py")
        print("  2. Entrenar:      python3 2_entrenar_modelo.py")
        print("  O sincronizar un modelo desde la nube.")
        print()
        return False
    return True


def ejecutar_traductor():
    """Lanza el traductor directamente."""
    print("\n" + "=" * 50)
    print("  🤟 TRADUCTOR LSE - MODO DIRECTO")
    print("=" * 50)
    print()

    from importlib import import_module

    # Importar y ejecutar el traductor
    sys.path.insert(0, DIR)
    os.chdir(DIR)

    traductor_mod = import_module('3_traductor')
    traductor = traductor_mod.TraductorLSE()
    traductor.ejecutar()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Traductor LSE - Modo Directo')
    parser.add_argument('--loop', action='store_true',
                        help='Reiniciar automáticamente al cerrarse')
    parser.add_argument('--no-sync', action='store_true',
                        help='No sincronizar con la nube al iniciar')
    args = parser.parse_args()

    while True:
        # 1. Sincronizar
        if not args.no_sync:
            auto_sync()

        # 2. Verificar modelo
        if not verificar_modelo():
            if args.loop:
                print("  Reintentando en 30 segundos...")
                time.sleep(30)
                continue
            else:
                sys.exit(1)

        # 3. Ejecutar traductor
        try:
            ejecutar_traductor()
        except KeyboardInterrupt:
            print("\n👋 Traductor cerrado por el usuario")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        # 4. Si no es loop, salir
        if not args.loop:
            break

        print("\n🔄 Reiniciando traductor en 3 segundos...")
        time.sleep(3)


if __name__ == "__main__":
    main()
