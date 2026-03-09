"""
Módulo auxiliar para suprimir warnings de C++ de MediaPipe/TensorFlow/absl.
Estos warnings vienen del código nativo y no se pueden controlar con
os.environ o logging de Python. Se redirige el file descriptor real de stderr.

Compatible con PyInstaller (console=False) donde sys.stderr puede ser None
o un objeto Redirector sin fileno()/flush().
"""

import os
import sys


def _safe_flush(stream):
    """Intenta hacer flush de un stream de forma segura."""
    if stream is None:
        return
    try:
        stream.flush()
    except (AttributeError, OSError, ValueError):
        pass


def _stderr_disponible():
    """Verifica si stderr tiene file descriptor disponible (no PyInstaller windowed)."""
    try:
        if sys.stderr is None:
            return False
        sys.stderr.fileno()
        return True
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation if 'io' in dir() else Exception):
        return False


def suprimir_stderr():
    """Redirige stderr a /dev/null a nivel de file descriptor del OS."""
    if not _stderr_disponible():
        return None  # No hay stderr real (PyInstaller windowed mode)

    try:
        _safe_flush(sys.stderr)
        stderr_fd = sys.stderr.fileno()
        stderr_copy = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        return stderr_copy
    except (OSError, ValueError, AttributeError):
        return None


def restaurar_stderr(stderr_copy):
    """Restaura stderr desde la copia guardada."""
    if stderr_copy is None:
        return  # No se suprimió (PyInstaller windowed mode)

    try:
        _safe_flush(sys.stderr)
        stderr_fd = 2  # default
        try:
            if sys.stderr is not None:
                stderr_fd = sys.stderr.fileno()
        except (AttributeError, OSError, ValueError):
            pass
        os.dup2(stderr_copy, stderr_fd)
        os.close(stderr_copy)
    except (OSError, ValueError, AttributeError):
        pass


def init_mediapipe(max_hands=4, detection_conf=0.7, tracking_conf=0.5, static_mode=False):
    """
    Importa e inicializa MediaPipe Hands sin mostrar warnings.
    Hace un warm-up call para que los warnings tardíos también se supriman.
    Retorna: (mp, mp_hands, mp_draw, hands)
    """
    import numpy as np

    stderr_copy = suprimir_stderr()

    import mediapipe as mp
    # Imports explícitos para que funcionen en PyInstaller (frozen)
    # mp.solutions usa lazy loading que falla en ejecutables congelados
    import mediapipe.solutions
    import mediapipe.solutions.hands
    import mediapipe.solutions.drawing_utils
    mp_hands = mediapipe.solutions.hands
    mp_draw = mediapipe.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=static_mode,
        max_num_hands=max_hands,
        model_complexity=1,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf
    )

    # Warm-up: procesar una imagen dummy para disparar los warnings tardíos
    # (XNNPACK delegate, inference_feedback_manager)
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    hands.process(dummy)

    restaurar_stderr(stderr_copy)

    return mp, mp_hands, mp_draw, hands


def init_tensorflow():
    """Importa TensorFlow sin mostrar warnings."""
    stderr_copy = suprimir_stderr()

    import tensorflow as tf

    restaurar_stderr(stderr_copy)

    return tf
