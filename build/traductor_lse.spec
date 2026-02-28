# traductor_lse.spec - PyInstaller configuration (optimized size)
import sys
import os
from PyInstaller.utils.hooks import collect_all

# Base directory: project root (one level up from build/)
BASEDIR = os.path.abspath(os.path.join(SPECPATH, '..'))

def p(*args):
    """Resolve path relative to project root."""
    return os.path.join(BASEDIR, *args)

# Collect MediaPipe (has hand tracking models and native libraries)
mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')

# Icon
if sys.platform == 'win32':
    icon_file = p('prototipo', 'icon.ico')
elif sys.platform == 'darwin':
    icon_file = p('prototipo', 'icon.png')
else:
    icon_file = None

a = Analysis(
    [p('prototipo', 'menu.py')],
    pathex=[BASEDIR],
    binaries=mp_binaries,
    datas=mp_datas + [
        (p('prototipo', '1_grabar_senas.py'), 'prototipo'),
        (p('prototipo', '2_entrenar_modelo.py'), 'prototipo'),
        (p('prototipo', '3_traductor.py'), 'prototipo'),
        (p('prototipo', '4_evaluar_iso25023.py'), 'prototipo'),
        (p('prototipo', 'utils_silenciar.py'), 'prototipo'),
        (p('prototipo', 'icon.png'), 'prototipo'),
        (p('prototipo', 'datos'), 'prototipo/datos'),
        (p('prototipo', 'modelo'), 'prototipo/modelo'),
    ],
    hiddenimports=mp_hiddenimports + [
        'pyttsx3.drivers',
        'pyttsx3.drivers.sapi5',
        'pyttsx3.drivers.nsss',
        'pyttsx3.drivers.espeak',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # === AHORRO ~400 MB ===
        'tensorboard', 'tensorboard_data_server', 'tensorboard_plugin_wit',
        'google.cloud', 'google.auth',
        'keras.src.testing',
        # === IDEs / Notebooks ===
        'matplotlib', 'notebook', 'jupyterlab', 'IPython',
        'sphinx', 'docutils', 'pygments',
        # === Testing ===
        'pytest', 'unittest', '_pytest',
        # === No necesarios ===
        'PIL', 'Pillow',
        'setuptools', 'pip', 'wheel',
        'email', 'html', 'http.server', 'xmlrpc',
        'pydoc', 'pdb', 'profile', 'cProfile',
        'lib2to3', 'ensurepip', 'venv',
        'idlelib', 'turtledemo', 'turtle',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TraductorLSE',
    debug=False,
    strip=True,  # Strip debug symbols
    upx=True,    # UPX compression
    console=False,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='TraductorLSE',
)

# macOS: create .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='TraductorLSE.app',
        icon=p('prototipo', 'icon.png'),
        bundle_identifier='com.lse.traductor',
        info_plist={
            'CFBundleName': 'Traductor LSE',
            'CFBundleDisplayName': 'Traductor LSE',
            'CFBundleVersion': '1.0.0',
            'NSCameraUsageDescription': 'Necesita acceso a la cámara para detectar señas.',
            'NSMicrophoneUsageDescription': 'Necesita micrófono para funciones de audio.',
            'NSHighResolutionCapable': True,
        },
    )
