# Runtime hook for mediapipe in PyInstaller
# Forces explicit import of mediapipe.solutions submodules at startup
# This is needed because mediapipe uses lazy loading that breaks in frozen executables
import importlib

try:
    import mediapipe
    # Force load the solutions subpackage and its modules
    importlib.import_module('mediapipe.solutions')
    importlib.import_module('mediapipe.solutions.hands')
    importlib.import_module('mediapipe.solutions.drawing_utils')
    importlib.import_module('mediapipe.solutions.drawing_styles')
except Exception:
    pass
