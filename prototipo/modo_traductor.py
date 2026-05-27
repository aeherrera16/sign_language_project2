#!/usr/bin/env python3
"""
MODO KIOSKO - Traductor LSE directo (sin menú)
===============================================
Para Raspberry Pi: arranca directo al traductor.
1. Lanza el traductor inmediatamente (no espera internet)
2. Si el traductor se cierra, vuelve a abrirlo (--loop)
3. Hilo daemon: cada 30 min verifica internet → descarga datos
   nuevos → reentrena en segundo plano → recarga el modelo
   sin interrumpir la traducción.

Uso:
    python3 modo_traductor.py          # Ejecuta una vez
    python3 modo_traductor.py --loop   # Se reinicia solo al cerrarse
"""

import os
import sys
import time
import threading
import subprocess

DIR = os.path.dirname(os.path.abspath(__file__))
DIR_MODELO = os.path.join(DIR, "modelo")
DIR_DATOS  = os.path.join(DIR, "datos")

# Archivo flag: cuando existe, el traductor recarga el modelo sin reiniciarse
RELOAD_FLAG = os.path.join(DIR_MODELO, ".reload_model")

INTERVALO_BG_MIN = 30   # Minutos entre chequeos de actualización

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'


# =============================================================================
# SINCRONIZACIÓN INICIAL (al arrancar, si hay internet)
# =============================================================================

def auto_sync():
    """Descarga modelo/datos de la nube al arrancar (silencioso, no bloquea)."""
    try:
        from sync_cloud import SyncCloud
        sync = SyncCloud()
        if sync.conectar():
            print("☁️  Sincronizando con la nube...")
            descargado = sync.descargar_modelo_si_hay_nuevo()
            if descargado:
                print("☁️  ✅ Modelo actualizado desde la nube")
            sync.descargar_datos_senas()
            print("☁️  Sincronización completada")
        else:
            print("☁️  Sin conexión. Usando datos locales.")
    except ImportError:
        print("☁️  firebase-admin no instalado. Modo offline.")
    except Exception as e:
        print(f"☁️  Error de sync (no crítico): {e}")


# =============================================================================
# WORKER DE ACTUALIZACIÓN EN SEGUNDO PLANO
# =============================================================================

def _contar_secuencias():
    total = 0
    if os.path.isdir(DIR_DATOS):
        for sena in os.listdir(DIR_DATOS):
            sena_path = os.path.join(DIR_DATOS, sena)
            if os.path.isdir(sena_path):
                total += len([f for f in os.listdir(sena_path) if f.endswith('.json')])
    return total


def _ciclo_actualizacion():
    """Un ciclo de sync+reentrenamiento. Se llama desde el hilo daemon."""
    try:
        from sync_cloud import SyncCloud, hay_internet
    except ImportError:
        return  # firebase-admin no instalado

    if not hay_internet():
        return

    sync = SyncCloud()
    if not sync.conectar():
        return

    # 1. ¿Hay un modelo más nuevo en la nube?
    if sync.descargar_modelo_si_hay_nuevo():
        print("[BG] Modelo nuevo descargado. Señalizando recarga...")
        open(RELOAD_FLAG, 'w').close()
        return  # Ya tenemos el modelo nuevo, no reentrenar

    # 2. ¿Hay datos nuevos?
    antes = _contar_secuencias()
    sync.descargar_datos_senas()
    despues = _contar_secuencias()

    if despues <= antes:
        return  # Sin datos nuevos, nada que hacer

    # 3. Reentrenas en subprocess separado — no bloquea el traductor
    print(f"[BG] {despues - antes} secuencias nuevas. Iniciando entrenamiento en segundo plano...")
    script = os.path.join(DIR, '2_entrenar_modelo.py')
    env = {**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'}
    try:
        proc = subprocess.run(
            [sys.executable, script],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1800  # Máximo 30 min
        )
        if proc.returncode == 0:
            # Subir modelo nuevo a la nube
            try:
                sync2 = SyncCloud()
                if sync2.conectar():
                    sync2.subir_modelo()
            except Exception:
                pass
            print("[BG] Entrenamiento completado. Señalizando recarga al traductor...")
            open(RELOAD_FLAG, 'w').close()
        else:
            print("[BG] El entrenamiento terminó con error. Se mantiene el modelo anterior.")
    except subprocess.TimeoutExpired:
        print("[BG] Entrenamiento cancelado (superó 30 min). Se mantiene el modelo anterior.")
    except Exception as e:
        print(f"[BG] Error durante entrenamiento: {e}")


def _hilo_bg(intervalo_seg):
    """Hilo daemon: espera 1 min al arrancar y luego cicla cada intervalo_seg."""
    time.sleep(60)  # Dar tiempo al traductor de cargarse antes del primer chequeo
    while True:
        try:
            _ciclo_actualizacion()
        except Exception as e:
            print(f"[BG] Error inesperado: {e}")
        time.sleep(intervalo_seg)


def iniciar_worker_bg():
    """Lanza el hilo daemon de actualización. No hace nada si ya está corriendo."""
    t = threading.Thread(
        target=_hilo_bg,
        args=(INTERVALO_BG_MIN * 60,),
        daemon=True,
        name="bg-actualizacion"
    )
    t.start()
    print(f"🔄 Actualizaciones automáticas activas (cada {INTERVALO_BG_MIN} min si hay internet)")


# =============================================================================
# VERIFICACIÓN DE MODELO
# =============================================================================

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

    # Sincronización inicial (no bloquea si no hay internet)
    if not args.no_sync:
        auto_sync()

    # Iniciar panel web + bot Telegram (daemons, no bloquean)
    try:
        from panel_control import iniciar_todo
        iniciar_todo(port=5000)
    except Exception as e:
        print(f"ℹ️  Panel de control no disponible: {e}")

    # Iniciar worker de actualización en segundo plano (siempre, daemon thread)
    if not args.no_sync:
        iniciar_worker_bg()

    while True:
        # Verificar modelo
        if not verificar_modelo():
            if args.loop:
                print("  Reintentando en 30 segundos...")
                time.sleep(30)
                continue
            else:
                sys.exit(1)

        # Ejecutar traductor
        try:
            ejecutar_traductor()
        except KeyboardInterrupt:
            print("\n👋 Traductor cerrado por el usuario")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        # Si no es loop, salir
        if not args.loop:
            break

        print("\n🔄 Reiniciando traductor en 3 segundos...")
        time.sleep(3)


if __name__ == "__main__":
    main()
