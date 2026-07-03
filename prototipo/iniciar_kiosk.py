#!/usr/bin/env python3
"""
MODO KIOSKO - Traductor LSE directo (sin menú), para el prototipo funcional (PC)
==================================================================================
Punto de entrada del .exe de Windows: arranca directo al traductor en pantalla
completa, sin mostrar ningún menú ni código — pensado para que el profesor o
un evaluador solo tenga que abrir el programa y ver la traducción funcionando.

A diferencia de modo_traductor.py (Raspberry Pi), esta versión NO sincroniza
con Firebase ni reentrena en segundo plano: es una demo de escritorio de un
solo uso, no un dispositivo desatendido.

Uso:
    python3 iniciar_kiosk.py          # Ejecuta una vez, cierra al presionar Q
    python3 iniciar_kiosk.py --loop   # Se reabre solo al cerrarse
"""

import os
import sys

# Protección para PyInstaller windowed (console=False), donde stdout/stderr
# pueden ser None y cualquier print()/flush() sin esto provocaría un crash.
class _NullWriter:
    def write(self, text): pass
    def flush(self): pass
    def fileno(self): raise OSError("No fd")
    def isatty(self): return False
    def __bool__(self): return True

if sys.stdout is None:
    sys.stdout = _NullWriter()
if sys.stderr is None:
    sys.stderr = _NullWriter()

import time

FROZEN = getattr(sys, 'frozen', False)
if FROZEN:
    DIR = os.path.join(sys._MEIPASS, 'prototipo')
else:
    DIR = os.path.dirname(os.path.abspath(__file__))

DIR_MODELO = os.path.join(DIR, "modelo")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'


def verificar_modelo():
    """Verifica que existe un modelo entrenado (h5, savedmodel o tflite)."""
    tiene_h5 = os.path.exists(os.path.join(DIR_MODELO, "modelo.h5"))
    tiene_sm = os.path.isdir(os.path.join(DIR_MODELO, "modelo_savedmodel"))
    tiene_tflite = os.path.exists(os.path.join(DIR_MODELO, "modelo.tflite"))
    tiene_encoder = os.path.exists(os.path.join(DIR_MODELO, "encoder.pkl"))

    if not tiene_encoder or not (tiene_h5 or tiene_sm or tiene_tflite):
        print("\n" + "=" * 50)
        print("  ❌ NO HAY MODELO ENTRENADO")
        print("=" * 50)
        return False
    return True


def ejecutar_traductor():
    """Lanza el traductor directamente, sin menú."""
    print("\n" + "=" * 50)
    print("  🤟 TRADUCTOR LSE - MODO KIOSKO")
    print("=" * 50)
    print()

    from importlib import import_module

    sys.path.insert(0, DIR)
    os.chdir(DIR)

    traductor_mod = import_module('3_traductor')
    traductor = traductor_mod.TraductorLSE()
    traductor.ejecutar()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Traductor LSE - Modo Kiosko')
    parser.add_argument('--loop', action='store_true',
                         help='Reiniciar automáticamente al cerrarse')
    args = parser.parse_args()

    while True:
        if not verificar_modelo():
            sys.exit(1)

        try:
            ejecutar_traductor()
        except KeyboardInterrupt:
            print("\n👋 Traductor cerrado por el usuario")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        if not args.loop:
            break

        print("\n🔄 Reiniciando traductor en 3 segundos...")
        time.sleep(3)


if __name__ == "__main__":
    main()
