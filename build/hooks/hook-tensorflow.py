# Custom TensorFlow hook for PyInstaller - Rosetta-safe
# -------------------------------------------------------
# The default hook from pyinstaller-hooks-contrib calls
# collect_submodules('tensorflow'), which spawns a subprocess that
# tries to import every tensorflow submodule.  Under Rosetta 2
# (x86_64 emulation on ARM macOS runners) that subprocess crashes
# with exit-code -4 (SIGILL).
#
# This hook sidesteps the problem entirely by:
# 1. Declaring hidden-imports statically (no subprocess)
# 2. Walking the filesystem to collect data files (no subprocess)
# -------------------------------------------------------
import os
import sys
import importlib.util
import glob

# ---------- hidden imports (static, no subprocess) --------------------------
hiddenimports = [
    # Core TF
    'tensorflow',
    'tensorflow.python',
    'tensorflow.python.client',
    'tensorflow.python.client.session',
    'tensorflow.python.eager',
    'tensorflow.python.eager.context',
    'tensorflow.python.eager.def_function',
    'tensorflow.python.framework',
    'tensorflow.python.framework.ops',
    'tensorflow.python.framework.dtypes',
    'tensorflow.python.framework.tensor_shape',
    'tensorflow.python.framework.versions',
    'tensorflow.python.ops',
    'tensorflow.python.ops.gen_math_ops',
    'tensorflow.python.ops.gen_array_ops',
    'tensorflow.python.ops.math_ops',
    'tensorflow.python.ops.array_ops',
    'tensorflow.python.ops.nn_ops',
    'tensorflow.python.ops.variables',
    'tensorflow.python.platform',
    'tensorflow.python.platform.self_check',
    'tensorflow.python.saved_model',
    'tensorflow.python.saved_model.loader',
    'tensorflow.python.training',
    'tensorflow.python.training.tracking',
    'tensorflow.python.util',
    'tensorflow.python.util.tf_export',
    'tensorflow.python.data',
    'tensorflow.python.distribute',
    'tensorflow.python.keras',
    'tensorflow.core',
    'tensorflow.core.framework',
    'tensorflow.core.protobuf',
    'tensorflow.lite',
    'tensorflow.lite.python',
    'tensorflow.lite.python.lite',
    'tensorflow.tools',
    # Keras (TF 2.14 ships internal keras)
    'keras',
    'keras.api',
    'keras.api._v2',
    'keras.api._v2.keras',
    'keras.src',
    'keras.src.engine',
    'keras.src.engine.base_layer',
    'keras.src.engine.functional',
    'keras.src.engine.sequential',
    'keras.src.engine.training',
    'keras.src.layers',
    'keras.src.layers.rnn',
    'keras.src.layers.core',
    'keras.src.layers.normalization',
    'keras.src.layers.regularization',
    'keras.src.models',
    'keras.src.optimizers',
    'keras.src.utils',
    'keras.src.saving',
    # Protobuf & deps
    'google',
    'google.protobuf',
    'google.protobuf.descriptor',
    'google.protobuf.descriptor_pool',
    'google.protobuf.message',
    'google.protobuf.reflection',
    'google.protobuf.symbol_database',
    'google.protobuf.text_format',
    'google.protobuf.json_format',
    # Other TF dependencies
    'h5py',
    'h5py._hl',
    'h5py._hl.files',
    'opt_einsum',
    'gast',
    'astunparse',
    'termcolor',
    'wrapt',
    'flatbuffers',
    'absl',
    'absl.logging',
    'absl.flags',
    'absl.flags._flag',
    'pasta',
    'tensorflow_estimator',
    'packaging',
    'packaging.version',
    'typing_extensions',
]

# ---------- excluded imports ------------------------------------------------
excludedimports = [
    'tensorboard',
    'tensorboard_data_server',
    'tensorboard_plugin_wit',
    'google.cloud',
    'google.auth',
    'keras.src.testing',
]

# ---------- data / binaries -------------------------------------------------
datas = []
binaries = []

def _find_package_dir(name):
    """Find a package directory without importing it (safe under Rosetta)."""
    spec = importlib.util.find_spec(name)
    if spec and spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    return None

# Collect TensorFlow package as data (filesystem walk, no subprocess)
tf_dir = _find_package_dir('tensorflow')
if tf_dir:
    datas.append((tf_dir, 'tensorflow'))

# Collect keras package as data
keras_dir = _find_package_dir('keras')
if keras_dir:
    datas.append((keras_dir, 'keras'))

# Collect tensorflow_estimator if present
te_dir = _find_package_dir('tensorflow_estimator')
if te_dir:
    datas.append((te_dir, 'tensorflow_estimator'))
