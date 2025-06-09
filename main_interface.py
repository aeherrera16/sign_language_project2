import tkinter as tk
from tkinter import messagebox, simpledialog
import subprocess
import sys
import os
import threading

def run_script(script_name, args=None):
    def task():
        try:
            if not os.path.exists(script_name):
                messagebox.showerror("Error", f"El archivo {script_name} no existe")
                return

            if script_name == "real_time_translate.py":
                try:
                    import pyttsx3
                except ImportError:
                    messagebox.showerror("Error", "Necesitas instalar pyttsx3 (pip install pyttsx3)")
                    return

            cmd = [sys.executable, script_name]
            if args:
                cmd.extend(args)

            subprocess.run(cmd)

            messagebox.showinfo("Completado", f"El script {script_name} ha finalizado.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar {script_name}:\n{str(e)}")

    threading.Thread(target=task).start()

def ask_gesture_name_and_record():
    gesture_name = simpledialog.askstring("Nombre del gesto", "Ingrese el nombre de la seña:")
    if gesture_name:
        run_script("record_dataset.py", args=[gesture_name])
    else:
        messagebox.showwarning("Aviso", "Debe ingresar un nombre para grabar el gesto.")

def check_data_folder():
    if not os.path.exists("data"):
        os.makedirs("data")
        messagebox.showinfo("Info", "Se creó la carpeta 'data' para almacenar gestos")

def test_voice():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say("Prueba de voz correcta.")
        engine.runAndWait()
    except Exception as e:
        messagebox.showerror("Error de voz", str(e))

# Crear ventana principal
root = tk.Tk()
root.title("Reconocimiento de Señas")
root.geometry("350x370")
root.configure(bg="#f0f0f0")

check_data_folder()

label = tk.Label(root, text="Sign Language App", font=("Arial", 16, "bold"), bg="#f0f0f0")
label.pack(pady=20)

button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack()

btn_record = tk.Button(button_frame, text="Grabar Gesto", font=("Arial", 12), width=20,
                       command=ask_gesture_name_and_record,
                       bg="#4CAF50", fg="white")
btn_record.pack(pady=5)

btn_train = tk.Button(button_frame, text="Entrenar Modelo", font=("Arial", 12), width=20,
                      command=lambda: run_script("train_model.py"),
                      bg="#2196F3", fg="white")
btn_train.pack(pady=5)

btn_translate = tk.Button(button_frame, text="Traducir Seña", font=("Arial", 12), width=20,
                          command=lambda: run_script("real_time_translate.py"),
                          bg="#FF9800", fg="white")
btn_translate.pack(pady=5)

btn_test_voice = tk.Button(button_frame, text="Probar Voz", font=("Arial", 12), width=20,
                           command=test_voice,
                           bg="#9C27B0", fg="white")
btn_test_voice.pack(pady=5)

btn_exit = tk.Button(root, text="Salir", font=("Arial", 12), width=20,
                     command=root.destroy,
                     bg="#f44336", fg="white")
btn_exit.pack(pady=20)

root.mainloop()
