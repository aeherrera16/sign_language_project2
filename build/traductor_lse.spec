# traductor_lse.spec - PyInstaller configuration
import sys
import os
from PyInstaller.utils.hooks import collect_all

# Collect MediaPipe (has hand tracking models and native libraries)
mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')

# Icon
if sys.platform == 'win32':
    icon_file = 'prototipo/icon.ico'
elif sys.platform == 'darwin':
    icon_file = 'prototipo/icon.png'
else:
    icon_file = None

a = Analysis(
    ['prototipo/menu.py'],
    pathex=[],
    binaries=mp_binaries,
    datas=mp_datas + [
        ('prototipo/1_grabar_senas.py', 'prototipo'),
        ('prototipo/2_entrenar_modelo.py', 'prototipo'),
        ('prototipo/3_traductor.py', 'prototipo'),
        ('prototipo/4_evaluar_iso25023.py', 'prototipo'),
        ('prototipo/utils_silenciar.py', 'prototipo'),
        ('prototipo/icon.png', 'prototipo'),
        ('prototipo/datos', 'prototipo/datos'),
        ('prototipo/modelo', 'prototipo/modelo'),
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
    excludes=['matplotlib', 'notebook', 'jupyterlab'],
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
    strip=False,
    upx=True,
    console=False,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='TraductorLSE',
)

# macOS: create .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='TraductorLSE.app',
        icon='prototipo/icon.png',
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
