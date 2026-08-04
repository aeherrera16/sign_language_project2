#!/usr/bin/env python3
"""
CONVERSOR A TFLITE — un solo método por invocación, aislado en proceso propio.

La conversión de Keras a TFLite puede abortar el proceso con un error fatal
de LLVM/MLIR en ciertas combinaciones de TensorFlow/Keras (visto con
TF 2.16 + Keras 3 + BatchNormalization: "LLVM ERROR: Failed to infer result
type(s)"). Ese tipo de error NO es una excepción de Python — es un abort()
a nivel nativo que se lleva el proceso completo, incluyendo cualquier salida
en buffer no flusheada todavía, y también se lleva cualquier otro intento
de conversión que estuviera programado a continuación EN EL MISMO proceso.

Por eso cada método se invoca como un subproceso independiente desde
2_entrenar_modelo.py — si uno crashea, el siguiente todavía corre, porque
vive en otro proceso del sistema operativo.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import argparse
import sys


def convertir_via_concrete_function(args):
    """No pasa por SavedModel — evita el bug de MLIR visto con BatchNormalization."""
    import tensorflow as tf
    import keras
    modelo = keras.models.load_model(args.h5_path, compile=False)

    @tf.function(input_signature=[tf.TensorSpec([1, args.frames, args.features], tf.float32)])
    def _predict(x):
        return modelo(x, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [_predict.get_concrete_function()], modelo
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convertir_via_saved_model(args):
    """Vía SavedModel — más compatible en general, pero es el que puede disparar el abort de MLIR."""
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(args.savedmodel_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


METODOS = {
    "concrete": convertir_via_concrete_function,
    "savedmodel": convertir_via_saved_model,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metodo', required=True, choices=list(METODOS))
    parser.add_argument('--savedmodel-path', required=True)
    parser.add_argument('--h5-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--features', type=int, required=True)
    args = parser.parse_args()

    try:
        tflite_model = METODOS[args.metodo](args)
        with open(args.output, 'wb') as f:
            f.write(tflite_model)
        print(f"OK:{len(tflite_model)}")
        return 0
    except Exception as e:
        print(f"FALLO:{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
