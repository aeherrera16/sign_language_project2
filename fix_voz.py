import os
path = "prototipo/3_traductor.py"
with open(path, "r") as f:
    code = f.read()

if "def _get_piper_bin():" not in code:
    code = code.replace(
        'def _piper_disponible(voz):',
        'def _get_piper_bin():\n    import shutil\n    if shutil.which("piper"): return "piper"\n    local_bin = os.path.expanduser("~/.local/bin/piper")\n    if os.path.exists(local_bin): return local_bin\n    return None\n\ndef _piper_disponible(voz):'
    )
    code = code.replace('shutil.which("piper")', '_get_piper_bin()')
    code = code.replace('cfg = PIPER_VOICES["neural"]', 'cfg = PIPER_VOICES["neural"]\n            piper_bin = _get_piper_bin()')
    code = code.replace('["piper", "--model"', '[piper_bin, "--model"')
    
    with open(path, "w") as f:
        f.write(code)
    print("¡Archivo corregido exitosamente!")
else:
    print("El archivo ya estaba corregido.")
