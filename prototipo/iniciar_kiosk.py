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


def _log_dir():
    """Carpeta junto al .exe (o al script en modo desarrollo) para error_log.txt."""
    if FROZEN:
        return os.path.dirname(sys.executable)
    return DIR


def _registrar_error(titulo, detalle):
    """Guarda el error en error_log.txt junto al ejecutable, para poder diagnosticar
    después — en modo ventana (console=False) no hay ninguna otra forma de verlo."""
    try:
        log_path = os.path.join(_log_dir(), "error_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n{titulo}\n{detalle}\n")
    except Exception:
        pass


def _mostrar_error(mensaje):
    """Muestra un cuadro de diálogo visible — imprescindible en modo ventana,
    donde el usuario no tiene consola ni forma de saber qué pasó."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Traductor LSE", mensaje)
        root.destroy()
    except Exception:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Traductor LSE - Modo Kiosko')
    parser.add_argument('--loop', action='store_true',
                         help='Reiniciar automáticamente al cerrarse')
    args = parser.parse_args()

    while True:
        if not verificar_modelo():
            _mostrar_error(
                "No se encontró un modelo entrenado en la carpeta 'modelo'.\n\n"
                "Reinstala el programa o contacta a quien te lo compartió."
            )
            sys.exit(1)

        try:
            ejecutar_traductor()
        except KeyboardInterrupt:
            print("\n👋 Traductor cerrado por el usuario")
        except Exception as e:
            import traceback
            detalle = traceback.format_exc()
            _registrar_error(f"Error: {e}", detalle)

            texto = str(e)
            if "cámara" in texto.lower() or "camera" in texto.lower():
                mensaje = (
                    "No se pudo acceder a la cámara.\n\n"
                    "Verifica que haya una cámara conectada, que no esté siendo "
                    "usada por otro programa (Zoom, Teams, etc.), y que Windows "
                    "tenga permiso de cámara activado para aplicaciones de "
                    "escritorio (Configuración > Privacidad > Cámara)."
                )
            else:
                mensaje = (
                    f"Ocurrió un error inesperado y el programa debe cerrarse:\n\n{e}\n\n"
                    f"Se guardó el detalle en error_log.txt, junto al programa."
                )
            _mostrar_error(mensaje)

        if not args.loop:
            break

        print("\n🔄 Reiniciando traductor en 3 segundos...")
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        _registrar_error(f"Error fatal antes de iniciar: {e}", traceback.format_exc())
        _mostrar_error(f"El programa no pudo iniciar:\n\n{e}\n\nSe guardó el detalle en error_log.txt.")
        sys.exit(1)
