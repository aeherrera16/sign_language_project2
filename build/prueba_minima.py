#!/usr/bin/env python3
"""Script mínimo para diagnóstico: si esto tampoco corre en la PC de prueba,
el bloqueo es contra cualquier .exe sin firmar generado por este pipeline,
no un problema específico del traductor (mediapipe/tensorflow/tamaño)."""

import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.withdraw()
messagebox.showinfo("Prueba LSE", "¡Funciona! Este .exe mínimo sí pudo abrirse.")
