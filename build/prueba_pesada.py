#!/usr/bin/env python3
"""Prueba con las mismas librerías pesadas del traductor (mediapipe + tensorflow)
pero sin la lógica real — solo para ver si el bloqueo en Windows es por el
contenido específico de estas librerías, ya que los .exe mínimos (sin ellas)
sí funcionaron."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import mediapipe as mp
import tensorflow as tf

import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.withdraw()
messagebox.showinfo(
    "Prueba LSE",
    f"¡Funciona! mediapipe {mp.__version__} y tensorflow {tf.__version__} sí pudieron abrirse."
)
