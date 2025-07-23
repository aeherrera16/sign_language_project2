# -*- coding: utf-8 -*-
"""
Configuración global para TensorFlow
Importar este módulo antes que TensorFlow para silenciar warnings
"""

import os
import warnings

# Configurar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo errores
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactivar oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Usar solo CPU por defecto

# Silenciar otros warnings comunes
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def configure_tensorflow():
    """Configuración adicional de TensorFlow después de importar"""
    try:
        import tensorflow as tf
        
        # Configurar para usar solo CPU si no hay GPU
        tf.config.set_visible_devices([], 'GPU')
        
        # Configurar threading
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        print("✅ TensorFlow configurado correctamente")
        
    except ImportError:
        print("⚠️  TensorFlow no está instalado")
    except Exception as e:
        print(f"⚠️  Error configurando TensorFlow: {e}")

# Aplicar configuración automáticamente al importar
configure_tensorflow()
