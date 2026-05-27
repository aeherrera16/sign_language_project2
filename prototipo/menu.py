#!/usr/bin/env python3
"""
MENÚ GRÁFICO - Traductor LSE
Compatible con macOS y Windows (usa tkinter, incluido con Python).
Aplicación autosuficiente: el usuario no necesita ver terminal ni código.
Soporta ejecución normal (python) y congelada (PyInstaller).
"""

import os
import sys
import subprocess
import threading
import json
import runpy
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext

# Sincronización con la nube (no falla si no está configurado)
try:
    from sync_cloud import SyncCloud
    SYNC_DISPONIBLE = True
except ImportError:
    SYNC_DISPONIBLE = False

# Detectar si estamos en un ejecutable congelado (PyInstaller)
FROZEN = getattr(sys, 'frozen', False)

# === PROTECCIÓN para PyInstaller windowed (console=False) ===
# En Windows con console=False, sys.stdout y sys.stderr pueden ser None
# lo que causa crashes en cualquier print() o .flush()
class NullWriter:
    """Stream nulo seguro para PyInstaller windowed mode."""
    def write(self, text):
        pass
    def flush(self):
        pass
    def fileno(self):
        raise OSError("NullWriter does not have a file descriptor")
    def isatty(self):
        return False
    def __bool__(self):
        return True

if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()

if FROZEN:
    # PyInstaller: scripts están en _MEIPASS/prototipo/
    DIR = os.path.join(sys._MEIPASS, 'prototipo')
else:
    DIR = os.path.dirname(os.path.abspath(__file__))

# Asegurar que el directorio de prototipo esté en sys.path
# para que scripts como 3_traductor.py puedan hacer 'from utils_silenciar import ...'
if DIR not in sys.path:
    sys.path.insert(0, DIR)

DIR_DATOS   = os.path.join(DIR, "datos")
DIR_MODELO  = os.path.join(DIR, "modelo")
PROYECTO_DIR = os.path.dirname(DIR)  # Raíz del repo (un nivel arriba de prototipo/)

# Silenciar warnings desde el inicio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('ABSL_MIN_LOG_LEVEL', '3')


def obtener_env():
    """Configura el entorno para silenciar warnings."""
    env = os.environ.copy()
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['TF_ENABLE_ONEDNN_OPTS'] = '0'
    env['MEDIAPIPE_DISABLE_GPU'] = '1'
    env['GLOG_minloglevel'] = '3'
    env['ABSL_MIN_LOG_LEVEL'] = '3'
    env['PYTHONWARNINGS'] = 'ignore'
    return env


def obtener_estado():
    """Obtiene el estado actual del proyecto."""
    senas = {}
    if os.path.isdir(DIR_DATOS):
        for d in sorted(os.listdir(DIR_DATOS)):
            ruta = os.path.join(DIR_DATOS, d)
            if os.path.isdir(ruta):
                archivos = [f for f in os.listdir(ruta) if f.endswith('.json')]
                senas[d] = len(archivos)

    modelo_existe = os.path.exists(os.path.join(DIR_MODELO, "modelo.h5"))

    info_modelo = {}
    info_path = os.path.join(DIR_MODELO, "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path) as f:
                info_modelo = json.load(f)
        except Exception:
            pass

    return senas, modelo_existe, info_modelo


class VentanaProgreso(tk.Toplevel):
    """Ventana que muestra el progreso de una tarea en tiempo real."""

    def __init__(self, parent, titulo, descripcion=""):
        super().__init__(parent)
        self.title(titulo)
        self.configure(bg='#1a1a2e')
        self.resizable(True, True)

        ancho, alto = 600, 400
        x = (self.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.winfo_screenheight() // 2) - (alto // 2)
        self.geometry(f'{ancho}x{alto}+{x}+{y}')
        self.transient(parent)

        # Título
        tk.Label(self, text=titulo, font=('Helvetica', 14, 'bold'),
                 fg='white', bg='#0f3460', pady=10).pack(fill='x')

        if descripcion:
            tk.Label(self, text=descripcion, font=('Helvetica', 10),
                     fg='#a0c4ff', bg='#1a1a2e', pady=5).pack(fill='x')

        # Área de texto con scroll para mostrar la salida
        self.texto = scrolledtext.ScrolledText(
            self, bg='#0d1117', fg='#c9d1d9',
            font=('Courier', 10), wrap='word',
            insertbackground='white', relief='flat'
        )
        self.texto.pack(fill='both', expand=True, padx=10, pady=10)
        self.texto.configure(state='disabled')

        # Barra de estado
        self.estado = tk.Label(self, text="⏳ Ejecutando...",
                               font=('Helvetica', 10, 'bold'),
                               fg='#ffc107', bg='#1a1a2e', pady=5)
        self.estado.pack(fill='x')

        self.proceso = None
        self.protocol("WM_DELETE_WINDOW", self.cerrar)

    def agregar_texto(self, texto):
        """Agrega texto al área de salida."""
        self.texto.configure(state='normal')
        self.texto.insert('end', texto)
        self.texto.see('end')
        self.texto.configure(state='disabled')

    def marcar_completado(self, exito=True):
        """Marca la tarea como completada."""
        if exito:
            self.estado.configure(text="✅ Completado", fg='#2ecc71')
        else:
            self.estado.configure(text="❌ Error", fg='#e74c3c')

    def cerrar(self):
        """Cierra la ventana y termina el proceso si sigue corriendo."""
        if self.proceso and self.proceso.poll() is None:
            self.proceso.terminate()
        self.destroy()


class MenuLSE:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤟 Traductor LSE - Lengua de Señas Ecuatoriana")
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(False, False)

        # Tamaño y centrar (Estirado verticalmente para Logs)
        ancho, alto = 550, 880
        x = (self.root.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.root.winfo_screenheight() // 2) - (alto // 2)
        self.root.geometry(f'{ancho}x{alto}+{x}+{y}')

        self.crear_ui()

    def crear_ui(self):
        bg = '#1a1a2e'

        # === TÍTULO ===
        frame_titulo = tk.Frame(self.root, bg='#0f3460', pady=15)
        frame_titulo.pack(fill='x')

        tk.Label(frame_titulo, text="🤟  TRADUCTOR LSE",
                 font=('Helvetica', 20, 'bold'), fg='white', bg='#0f3460').pack()
        tk.Label(frame_titulo, text="Lengua de Señas Ecuatoriana",
                 font=('Helvetica', 11), fg='#a0c4ff', bg='#0f3460').pack()
        tk.Label(frame_titulo, text="Prototipo funcional",
                 font=('Helvetica', 9, 'italic'), fg='#7090bf', bg='#0f3460').pack()

        # === ESTADO DEL HARDWARE ===
        self.frame_hw = tk.Frame(frame_titulo, bg='#0f3460')
        self.frame_hw.place(relx=0.96, rely=0.08, anchor='ne')

        self.lbl_cam = tk.Label(self.frame_hw, text="📷 Cam: ?", font=('Helvetica', 9, 'bold'), bg='#0f3460', fg='gray')
        self.lbl_cam.pack(anchor='e')
        
        self.lbl_mic = tk.Label(self.frame_hw, text="🔊 Aud: ?", font=('Helvetica', 9, 'bold'), bg='#0f3460', fg='gray')
        self.lbl_mic.pack(anchor='e')

        self.revisar_hardware()

        # === SECCIÓN: GRABACIÓN ===
        frame_grab = tk.LabelFrame(self.root, text=" 📹 Grabación ",
                                    font=('Helvetica', 10, 'bold'),
                                    fg='#a0c4ff', bg=bg, bd=1, relief='groove',
                                    padx=10, pady=5)
        frame_grab.pack(fill='x', padx=15, pady=(10, 3))

        self._crear_boton(frame_grab, "[1] Grabar UNA seña", self.grabar_una, '#2d6a4f')
        self._crear_boton(frame_grab, "[2] Grabar VARIAS señas", self.grabar_varias, '#2d6a4f')

        # === SECCIÓN: MODELO ===
        frame_modelo = tk.LabelFrame(self.root, text=" 🧠 Modelo ",
                                      font=('Helvetica', 10, 'bold'),
                                      fg='#a0c4ff', bg=bg, bd=1, relief='groove',
                                      padx=10, pady=5)
        frame_modelo.pack(fill='x', padx=15, pady=3)

        self._crear_boton(frame_modelo, "[3] Entrenar modelo", self.entrenar, '#e76f51')
        self._crear_boton(frame_modelo, "[4] Evaluar métricas ISO", self.evaluar, '#7209b7')

        # === SECCIÓN: TRADUCTOR ===
        frame_trad = tk.LabelFrame(self.root, text=" 🔊 Traductor ",
                                    font=('Helvetica', 10, 'bold'),
                                    fg='#a0c4ff', bg=bg, bd=1, relief='groove',
                                    padx=10, pady=5)
        frame_trad.pack(fill='x', padx=15, pady=3)

        self._crear_boton(frame_trad, "[5] ▶  Iniciar traductor en tiempo real", self.traducir, '#0077b6')

        # === SECCIÓN: NUBE ===
        frame_nube = tk.LabelFrame(self.root, text=" ☁️ Nube (Firebase) ",
                                    font=('Helvetica', 10, 'bold'),
                                    fg='#a0c4ff', bg=bg, bd=1, relief='groove',
                                    padx=10, pady=5)
        frame_nube.pack(fill='x', padx=15, pady=3)

        self._crear_boton(frame_nube, "[7] ☁️  Sincronizar con la nube", self.sincronizar_nube, '#0d47a1')

        # Estado de la nube
        self.lbl_nube = tk.Label(frame_nube, text="☁️ Nube: Verificando...",
                                  font=('Helvetica', 9), fg='gray', bg=bg, anchor='w')
        self.lbl_nube.pack(fill='x', padx=5)

        # === FLUJO COMPLETO ===
        frame_flujo = tk.Frame(self.root, bg=bg)
        frame_flujo.pack(fill='x', padx=15, pady=(8, 3))

        btn_flujo = tk.Button(frame_flujo,
                              text="[6] 🚀  Flujo completo (Grabar → Entrenar → Traducir)",
                              command=self.flujo_completo,
                              font=('Helvetica', 11, 'bold'), fg='white', bg='#c9184a',
                              activebackground='#ff4d6d', activeforeground='white',
                              relief='flat', cursor='hand2', pady=10)
        btn_flujo.pack(fill='x')
        btn_flujo.bind('<Enter>', lambda e: btn_flujo.configure(bg='#ff4d6d'))
        btn_flujo.bind('<Leave>', lambda e: btn_flujo.configure(bg='#c9184a'))

        # === ESTADO ===
        frame_estado = tk.LabelFrame(self.root, text=" 📂 Estado del proyecto ",
                                      font=('Helvetica', 10, 'bold'),
                                      fg='#a0c4ff', bg='#16213e', bd=1, relief='groove',
                                      padx=10, pady=5)
        frame_estado.pack(fill='x', padx=15, pady=(8, 3))

        self.label_estado = tk.Label(frame_estado, text="Cargando...",
                                     font=('Helvetica', 10), fg='#c0c0c0', bg='#16213e',
                                     justify='left', anchor='w')
        self.label_estado.pack(fill='x', padx=5, pady=3)

        btn_refrescar = tk.Button(frame_estado, text="🔄 Actualizar estado",
                                  command=self.actualizar_estado,
                                  font=('Helvetica', 9), fg='#a0c4ff', bg='#1a2744',
                                  relief='flat', cursor='hand2', bd=0)
        btn_refrescar.pack(anchor='e', pady=(0, 3))

        self.actualizar_estado()

        # === SECCIÓN: SISTEMA ===
        frame_sistema = tk.LabelFrame(self.root, text=" 🔧 Sistema ",
                                      font=('Helvetica', 10, 'bold'),
                                      fg='#a0c4ff', bg=bg, bd=1, relief='groove',
                                      padx=10, pady=5)
        frame_sistema.pack(fill='x', padx=15, pady=3)

        self._crear_boton(frame_sistema, "[8] ⬇  Actualizar desde GitHub", self.actualizar_sistema, '#37474f')

        # === SALIR ===
        tk.Button(self.root, text="[0] Salir", command=self.root.quit,
                  font=('Helvetica', 10), fg='#888', bg=bg,
                  relief='flat', cursor='hand2', bd=0).pack(pady=(2, 5))

        # === CONSOLA DE LOGGING EN VIVO ===
        frame_logs = tk.LabelFrame(self.root, text=" 📝 Terminal del Sistema ", font=('Helvetica', 9, 'bold'), fg='#a0c4ff', bg='#0a0f1c', bd=1)
        frame_logs.pack(fill='both', expand=True, padx=15, pady=(0, 10))
        
        self.text_log = tk.Text(frame_logs, height=6, font=('Consolas', 8), bg='#02050b', fg='#00ff00', state='disabled', wrap='word')
        self.text_log.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.imprimir_log("Terminal inicializada. Listo para recibir comandos.")

        # === ATAJOS DE TECLADO ===
        self.root.bind('1', lambda e: self.grabar_una())
        self.root.bind('2', lambda e: self.grabar_varias())
        self.root.bind('3', lambda e: self.entrenar())
        self.root.bind('4', lambda e: self.evaluar())
        self.root.bind('5', lambda e: self.traducir())
        self.root.bind('6', lambda e: self.flujo_completo())
        self.root.bind('7', lambda e: self.sincronizar_nube())
        self.root.bind('8', lambda e: self.actualizar_sistema())
        self.root.bind('0', lambda e: self.root.quit())
        self.root.bind('<Escape>', lambda e: self.root.quit())

        # Auto-sync silencioso al iniciar (en background)
        if SYNC_DISPONIBLE:
            self.root.after(2000, self._auto_sync_inicio)

    def imprimir_log(self, mensaje):
        """Escribe texto en la consola del GUI de manera Thread-Safe"""
        if not hasattr(self, 'text_log') or not self.root.winfo_exists(): return
        
        def escribir():
            self.text_log.config(state='normal')
            self.text_log.insert('end', str(mensaje) + "\n")
            self.text_log.see('end')
            self.text_log.config(state='disabled')
        
        self.root.after(0, escribir)

    def revisar_hardware(self):
        """Verifica en background el estado de cámara y parlante (Bluetooth o Cable)."""
        def test_hw():
            cam_ok = False
            # Check Cam (buscando cámara física principal /dev/video0)
            if sys.platform == 'linux':
                import os
                if os.path.exists('/dev/video0'):
                    cam_ok = True
            else:
                import cv2
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    cam_ok = True
                    cap.release()

            audio_ok = False
            audio_modo = "Cable"
            
            if sys.platform == 'linux':
                # Intentar detectar dispositivos Bluetooth conectados a bluez
                bt_str = ""
                try:
                    bt_str = subprocess.check_output(['bluetoothctl', 'devices', 'Connected'], stderr=subprocess.STDOUT, text=True)
                except Exception:
                    pass
                    
                if bt_str.strip():
                    # Existe algo por Bluetooth conectado, se asume es dispositivo Audio/Mixer
                    audio_ok = True
                    audio_modo = "BT"
                else:
                    # Alternativa por tarjetas Alsa (Jack / USB)
                    try:
                        al_str = subprocess.check_output(['aplay', '-l'], stderr=subprocess.STDOUT, text=True)
                        if 'card ' in al_str:
                            audio_ok = True
                    except Exception:
                        pass
            else:
                # Windows/Mac asume positivo por simpleza
                audio_ok = True

            def update_ui():
                if cam_ok:
                    self.lbl_cam.config(text="📷 Cam: OK", fg='#00ff00')
                else:
                    self.lbl_cam.config(text="📷 Cam: OFF", fg='#ff4d4d')
                    
                if audio_ok:
                    self.lbl_mic.config(text=f"🔊 Aud ({audio_modo}): OK", fg='#00ff00')
                else:
                    self.lbl_mic.config(text="🔊 Aud: OFF", fg='#ff4d4d')

            # Update interface en hilo seguro
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, update_ui)

        # Disparar chequeo en fondo para no trabar app
        hilo = threading.Thread(target=test_hw, daemon=True)
        hilo.start()
        
        # Repetir chequeo cada 4 segundos
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(4000, self.revisar_hardware)

    def _crear_boton(self, parent, texto, comando, color):
        """Crea un botón estilizado con hover."""
        btn = tk.Button(parent, text=texto, command=comando,
                        font=('Helvetica', 12), fg='white', bg=color,
                        activebackground=color, activeforeground='white',
                        relief='flat', cursor='hand2', pady=7)
        btn.pack(fill='x', pady=3)

        lighter = self._lighten(color)
        btn.bind('<Enter>', lambda e, b=btn, c=lighter: b.configure(bg=c))
        btn.bind('<Leave>', lambda e, b=btn, c=color: b.configure(bg=c))
        return btn

    def _lighten(self, hex_color):
        """Aclara un color hex para efecto hover."""
        r = min(255, int(hex_color[1:3], 16) + 30)
        g = min(255, int(hex_color[3:5], 16) + 30)
        b = min(255, int(hex_color[5:7], 16) + 30)
        return f'#{r:02x}{g:02x}{b:02x}'

    def actualizar_estado(self):
        """Actualiza la barra de estado."""
        senas, modelo, info = obtener_estado()

        if senas:
            total = sum(senas.values())
            nombres = ", ".join(senas.keys())
            txt = f"Señas: {len(senas)} ({total} archivos)\n  → {nombres}"
        else:
            txt = "⚠️ Sin señas grabadas"

        if modelo:
            acc = info.get('accuracy_test', info.get('accuracy', 0))
            clases = info.get('clases', [])
            txt += f"\nModelo: ✅ Entrenado — Accuracy: {acc:.0%}"
            txt += f"\n  → Clases: {', '.join(clases) if clases else 'N/A'}"
        else:
            txt += "\nModelo: ❌ No entrenado"

        self.label_estado.configure(text=txt)

    def ejecutar_con_progreso(self, script, args=None, titulo="Ejecutando...", descripcion=""):
        """Ejecuta un script mostrando la salida en una ventana de progreso."""
        ventana = VentanaProgreso(self.root, titulo, descripcion)

        if FROZEN:
            # Modo congelado: usar runpy en un hilo con captura de salida
            script_path = os.path.join(DIR, script)

            def run():
                old_argv = sys.argv
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.argv = [script_path] + (args or [])

                class Redirector:
                    def __init__(self, callback):
                        self.callback = callback
                    def write(self, text):
                        if text:
                            try:
                                self.callback(text)
                            except Exception:
                                pass
                    def flush(self):
                        pass
                    def fileno(self):
                        raise OSError("Redirector does not have a file descriptor")
                    def isatty(self):
                        return False
                    def __bool__(self):
                        return True

                sys.stdout = Redirector(ventana.agregar_texto)
                sys.stderr = Redirector(ventana.agregar_texto)
                try:
                    runpy.run_path(script_path, run_name='__main__')
                    ventana.marcar_completado(True)
                except SystemExit:
                    ventana.marcar_completado(True)
                except Exception as e:
                    ventana.agregar_texto(f"\n❌ Error: {e}\n")
                    ventana.marcar_completado(False)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    sys.argv = old_argv
                    self.root.after(500, self.actualizar_estado)

            hilo = threading.Thread(target=run, daemon=True)
            hilo.start()
        else:
            # Modo normal: subprocess
            cmd = [sys.executable, os.path.join(DIR, script)]
            if args:
                cmd.extend(args)

            def run():
                try:
                    proceso = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env=obtener_env(), cwd=DIR
                    )
                    ventana.proceso = proceso

                    for linea in proceso.stdout:
                        ventana.agregar_texto(linea)

                    proceso.wait()
                    ventana.marcar_completado(proceso.returncode == 0)
                except Exception as e:
                    ventana.agregar_texto(f"\n❌ Error: {e}\n")
                    ventana.marcar_completado(False)
                self.root.after(500, self.actualizar_estado)

            hilo = threading.Thread(target=run, daemon=True)
            hilo.start()

    def ejecutar_ventana(self, script, args=None):
        """Ejecuta un script que abre su propia ventana (grabador, traductor)."""
        if FROZEN:
            # Modo congelado: ocultar menú, ejecutar con runpy, restaurar menú
            script_path = os.path.join(DIR, script)
            self.root.withdraw()
            old_argv = sys.argv
            sys.argv = [script_path] + (args or [])
            # Asegurar que prototipo dir esté en sys.path para imports
            if DIR not in sys.path:
                sys.path.insert(0, DIR)
            try:
                runpy.run_path(script_path, run_name='__main__')
            except SystemExit:
                pass
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                # Guardar log de error junto al ejecutable
                try:
                    exe_dir = os.path.dirname(sys.executable) if hasattr(sys, 'executable') else '.'
                    log_path = os.path.join(exe_dir, 'error_log.txt')
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"Script: {script}\n")
                        f.write(f"Args: {args}\n")
                        f.write(f"Error:\n{error_detail}\n")
                except Exception:
                    pass
                # Restaurar ventana ANTES de mostrar error para que sea visible
                self.root.deiconify()
                self.root.lift()
                messagebox.showerror("Error", f"Error al ejecutar {script}:\n\n{e}\n\nVer error_log.txt para detalles")
                # Ya restauramos, salir del finally sin duplicar
                sys.argv = old_argv
                self.actualizar_estado()
                return
            finally:
                sys.argv = old_argv
                self.root.deiconify()
                self.root.lift()
                self.actualizar_estado()
        else:
            # Modo normal: subprocess con captura de errores enviada a Consola
            script_path = os.path.join(DIR, script)
            cmd = [sys.executable, script_path]
            if args:
                cmd.extend(args)
            
            env = obtener_env()
            env['PYTHONPATH'] = DIR + os.pathsep + env.get('PYTHONPATH', '')
            
            try:
                self.imprimir_log(f"Lanzando proceso: {' '.join(cmd)}")
                proceso = subprocess.Popen(
                    cmd, env=env, cwd=DIR,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                
                # Hilo para leer standard output/error y postearlo en la UI
                def monitor_logs():
                    for linea in iter(proceso.stdout.readline, ''):
                        if linea.strip():
                            # Filtrar basurilla de warnings comunes de TensorFlow si son repetitivos
                            if "TF-TRT" in linea or "I tensorflow/" in linea: continue
                            self.imprimir_log(linea.strip())
                    
                    proceso.stdout.close()
                    retcode = proceso.wait()
                    if retcode != 0:
                        self.imprimir_log(f"⚠️ Proceso finalizó con código de error {retcode}")
                    else:
                        self.imprimir_log(f"Proceso completado exitosamente.")
                
                threading.Thread(target=monitor_logs, daemon=True).start()
            except Exception as e:
                self.imprimir_log(f"❌ No se pudo ejecutar el script: {e}")

    def grabar_una(self):
        nombre = simpledialog.askstring("Grabar seña",
                                         "Nombre de la seña:",
                                         parent=self.root)
        if nombre and nombre.strip():
            nombre = nombre.strip().upper()
            # Bypassed messagebox para evitar lock en kiosk
            self.ejecutar_ventana("1_grabar_senas.py",
                                  ["--nombre", nombre, "--cantidad", "30"])

    def grabar_varias(self):
        texto = simpledialog.askstring("Grabar varias señas",
                                        "Nombres separados por coma:\n"
                                        "(ej: HOLA, GRACIAS, BUENO)",
                                        parent=self.root)
        if texto and texto.strip():
            nombres = [n.strip().upper() for n in texto.split(",") if n.strip()]
            # Bypassed messagebox principal
            for nombre in nombres:
                self.ejecutar_ventana("1_grabar_senas.py",
                                      ["--nombre", nombre, "--cantidad", "30"])

    def entrenar(self):
        senas, _, _ = obtener_estado()
        if len(senas) < 2:
            self.imprimir_log("⚠️ Advertencia: Necesitas mínimo 2 señas grabadas para entrenar.")
            return

        self.imprimir_log("🚀 Iniciando entrenamiento (2_entrenar_modelo.py) ...")
        self.ejecutar_ventana("2_entrenar_modelo.py")

    def traducir(self):
        if not os.path.exists(os.path.join(DIR_MODELO, "modelo.h5")):
            # Advertencia si no hay modelo, intentamos evitar bloqueo usando print o asumiendo que el User lo sabe
            print("Sin modelo")
            return

        # Lanzar directo sin modal
        self.ejecutar_ventana("3_traductor.py")

    def evaluar(self):
        if not os.path.exists(os.path.join(DIR_MODELO, "modelo.h5")):
            self.imprimir_log("❌ Error al Evaluar: Sin modelo entrenado evaluable.")
            return

        self.imprimir_log("🚀 Iniciando Evaluación de Métricas ISO ...")
        self.ejecutar_ventana("4_evaluar_iso25023.py")

    def sincronizar_nube(self):
        """Sincronización manual con la nube."""
        if not SYNC_DISPONIBLE:
            self.imprimir_log("⚠️ firebase-admin no instalado. Ejecuta: pip install firebase-admin")
            return

        self.imprimir_log("☁️ Iniciando sincronización con la nube...")
        self.ejecutar_con_progreso(
            "sync_cloud.py",
            args=["--sync-todo"],
            titulo="☁️ Sincronización con la Nube",
            descripcion="Sincronizando modelo y datos con Firebase..."
        )
        # Actualizar estado de la nube después
        self.root.after(5000, self._actualizar_estado_nube)

    def _auto_sync_inicio(self):
        """Auto-sync silencioso al iniciar la app (descarga modelo y datos nuevos)."""
        def sync_bg():
            try:
                sync = SyncCloud()
                if sync.conectar():
                    # 1. Descargar modelo nuevo si existe
                    descargado = sync.descargar_modelo_si_hay_nuevo()
                    if descargado:
                        self.imprimir_log("☁️ Modelo actualizado desde la nube")
                        self.root.after(0, self.actualizar_estado)

                    # 2. Descargar datos de señas nuevos
                    sync.descargar_datos_senas()

                    self._actualizar_estado_nube(sync)
                else:
                    self._actualizar_estado_nube()
            except Exception as e:
                self.imprimir_log(f"☁️ Sync: {e}")
                self._actualizar_estado_nube()

        threading.Thread(target=sync_bg, daemon=True).start()

    def _actualizar_estado_nube(self, sync=None):
        """Actualiza el indicador visual del estado de la nube."""
        if not hasattr(self, 'lbl_nube'):
            return

        def actualizar():
            if not SYNC_DISPONIBLE:
                self.lbl_nube.config(text="☁️ Nube: No configurada (pip install firebase-admin)", fg='#ff9800')
                return

            try:
                if sync and sync.conectado:
                    estado = sync.obtener_estado_sync()
                else:
                    s = SyncCloud()
                    estado = s.obtener_estado_sync()

                if not estado['credenciales']:
                    self.lbl_nube.config(text="☁️ Nube: Sin credenciales (ver guía)", fg='#ff9800')
                elif not estado['internet']:
                    self.lbl_nube.config(text="☁️ Nube: Sin internet (modo offline)", fg='#ffc107')
                else:
                    ultima = estado.get('ultima_subida', 'Nunca')
                    if ultima != 'Nunca':
                        # Formatear fecha
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(ultima)
                            ultima = dt.strftime('%d/%m %H:%M')
                        except Exception:
                            pass
                    self.lbl_nube.config(text=f"☁️ Nube: Conectada | Última sync: {ultima}", fg='#4caf50')
            except Exception:
                self.lbl_nube.config(text="☁️ Nube: Estado desconocido", fg='gray')

        self.root.after(0, actualizar)

    def actualizar_sistema(self):
        """Descarga los últimos cambios de GitHub e instala dependencias nuevas."""
        git_dir = os.path.join(PROYECTO_DIR, '.git')
        if not os.path.exists(git_dir):
            self.imprimir_log("⚠️ No es un repositorio git. Actualiza manualmente.")
            return

        ventana = VentanaProgreso(
            self.root,
            "⬇  Actualizando sistema",
            "Descargando cambios de GitHub e instalando dependencias..."
        )

        def run():
            exito = True

            # Paso 1: git pull
            ventana.agregar_texto("▶ git pull origin feature/prototipo-wearable\n")
            try:
                proc = subprocess.Popen(
                    ['git', 'pull', 'origin', 'feature/prototipo-wearable'],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=PROYECTO_DIR
                )
                ventana.proceso = proc
                for linea in proc.stdout:
                    ventana.agregar_texto(linea)
                proc.wait()
                if proc.returncode != 0:
                    ventana.agregar_texto("⚠️ git pull terminó con errores\n")
                    exito = False
            except FileNotFoundError:
                ventana.agregar_texto("❌ git no encontrado. Instala git en el sistema.\n")
                exito = False
            except Exception as e:
                ventana.agregar_texto(f"❌ Error: {e}\n")
                exito = False

            # Paso 2: pip install dependencias nuevas
            ventana.agregar_texto("\n▶ Instalando dependencias nuevas...\n")
            try:
                proc2 = subprocess.Popen(
                    [sys.executable, '-m', 'pip', 'install',
                     'flask', 'requests', '--quiet', '--exists-action', 'i'],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=PROYECTO_DIR
                )
                ventana.proceso = proc2
                for linea in proc2.stdout:
                    ventana.agregar_texto(linea)
                proc2.wait()
            except Exception as e:
                ventana.agregar_texto(f"⚠️ pip: {e}\n")

            if exito:
                ventana.agregar_texto("\n✅ Sistema actualizado correctamente.\n")
                ventana.agregar_texto("   Cierra y vuelve a abrir la app para aplicar todos los cambios.\n")
            ventana.marcar_completado(exito)
            self.root.after(500, self.actualizar_estado)

        threading.Thread(target=run, daemon=True).start()

    def flujo_completo(self):
        nombre = simpledialog.askstring("Flujo completo",
                                         "Nombre de la seña a grabar:",
                                         parent=self.root)
        if not nombre or not nombre.strip():
            return

        nombre = nombre.strip().upper()
        # Directo sin modals
        self.ejecutar_ventana("1_grabar_senas.py",
                              ["--nombre", nombre, "--cantidad", "30"])

    def iniciar(self):
        self.root.mainloop()


if __name__ == "__main__":
    MenuLSE().iniciar()
