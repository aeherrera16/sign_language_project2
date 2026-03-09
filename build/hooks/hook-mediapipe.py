# hook-mediapipe.py - PyInstaller hook for mediapipe
# Ensures all mediapipe submodules are collected, especially 'solutions'
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all submodules to ensure mediapipe.solutions is included
hiddenimports = collect_submodules('mediapipe')

# Also collect all data files (models, configs, etc.)
datas, binaries, more_hiddenimports = collect_all('mediapipe')
hiddenimports += more_hiddenimports
