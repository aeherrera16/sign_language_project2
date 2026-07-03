# traductor_lse.spec - PyInstaller configuration (optimized size)
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Base directory: project root (one level up from build/)
BASEDIR = os.path.abspath(os.path.join(SPECPATH, '..'))

def p(*args):
    """Resolve path relative to project root."""
    return os.path.join(BASEDIR, *args)

# =====================================================================
# MediaPipe: recopilar TODO el paquete de forma agresiva
# =====================================================================
# collect_submodules encuentra TODOS los submódulos Python importables
# collect_all encuentra datos, binarios y más hidden imports
mp_hiddenimports = collect_submodules('mediapipe')
print(f"[SPEC] MediaPipe submodules found: {len(mp_hiddenimports)}")

try:
    mp_datas, mp_binaries, _extra_hi = collect_all('mediapipe')
    mp_hiddenimports += _extra_hi
    print(f"[SPEC] MediaPipe datas: {len(mp_datas)}, binaries: {len(mp_binaries)}")
except Exception as e:
    print(f"[SPEC] collect_all failed: {e}, falling back to manual")
    import importlib.util
    mp_spec = importlib.util.find_spec('mediapipe')
    if mp_spec and mp_spec.submodule_search_locations:
        mp_pkg_dir = mp_spec.submodule_search_locations[0]
    else:
        import subprocess as _sp
        result = _sp.run(
            [sys.executable, '-c', 'import mediapipe; import os; print(os.path.dirname(mediapipe.__file__))'],
            capture_output=True, text=True
        )
        mp_pkg_dir = result.stdout.strip()
    mp_datas = [(mp_pkg_dir, 'mediapipe')]
    mp_binaries = []

# De-duplicate
mp_hiddenimports = list(set(mp_hiddenimports))

# Icon (solo Windows)
if sys.platform == 'win32':
    icon_file = p('prototipo', 'icon.ico')
else:
    icon_file = None

# VERSION es opcional: el workflow de release la escribe antes de compilar
# (a partir del tag de git) para que el traductor la muestre en el título
# de la ventana. En builds locales/manuales sin tag, simplemente no existe.
_extra_datas = []
if os.path.exists(p('prototipo', 'VERSION')):
    _extra_datas.append((p('prototipo', 'VERSION'), 'prototipo'))

# version_info.txt (recurso de versión del .exe) es opcional por la misma razón.
_version_resource = p('build', 'version_info.txt')
if not os.path.exists(_version_resource):
    _version_resource = None

a = Analysis(
    [p('prototipo', 'iniciar_kiosk.py')],
    pathex=[BASEDIR],
    binaries=mp_binaries,
    datas=mp_datas + _extra_datas + [
        (p('prototipo', '3_traductor.py'), 'prototipo'),
        (p('prototipo', 'utils_silenciar.py'), 'prototipo'),
        (p('prototipo', 'icon.png'), 'prototipo'),
        (p('prototipo', 'modelo'), 'prototipo/modelo'),
    ],
    hiddenimports=mp_hiddenimports + [
        'utils_silenciar',
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
        # === AHORRO ~400 MB (solo paquetes pesados que NO se usan) ===
        'tensorboard', 'tensorboard_data_server', 'tensorboard_plugin_wit',
        'google.cloud', 'google.auth',
        'keras.src.testing',
        # === IDEs / Notebooks ===
        'notebook', 'jupyterlab', 'IPython',
        'sphinx', 'docutils',
        # === Herramientas de desarrollo ===
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
    version=_version_resource,
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

