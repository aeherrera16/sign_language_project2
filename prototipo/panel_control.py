#!/usr/bin/env python3
"""
Panel de Control - Web + Bot Telegram
======================================
Controla el Traductor LSE desde el teléfono SIN teclado ni ratón.

  Panel web  → abre  http://<ip-del-pi>:5000  en el navegador del móvil
  Telegram   → configura token en prototipo/.panel_config.json y manda /start

Configurar Telegram (una sola vez):
  1. Abre Telegram → busca @BotFather → escribe /newbot
  2. Pon el token en prototipo/.panel_config.json:
         {"telegram_token": "123456789:ABC-DEF..."}
  3. Manda /start al bot desde tu teléfono → queda registrado
"""

import os, sys, json, time, threading, subprocess, socket, logging

logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Silenciar logs de Flask

DIR         = os.path.dirname(os.path.abspath(__file__))
DIR_MODELO  = os.path.join(DIR, "modelo")
DIR_DATOS   = os.path.join(DIR, "datos")
RELOAD_FLAG = os.path.join(DIR_MODELO, ".reload_model")
CONFIG_PATH = os.path.join(DIR, ".panel_config.json")

# =============================================================================
# ESTADO COMPARTIDO
# =============================================================================

_lock   = threading.Lock()
_estado = {
    "traductor":     "activo",
    "entrenando":    False,
    "sincronizando": False,
    "ultima_accion": "—",
    "clases":        [],
    "accuracy":      0.0,
    "n_secuencias":  0,
}

def _refrescar_estado():
    info_path = os.path.join(DIR_MODELO, "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path) as f:
                info = json.load(f)
            with _lock:
                _estado["clases"]   = info.get("clases", [])
                _estado["accuracy"] = info.get("accuracy_test", 0.0)
        except Exception:
            pass
    total = 0
    if os.path.isdir(DIR_DATOS):
        for sena in os.listdir(DIR_DATOS):
            sp = os.path.join(DIR_DATOS, sena)
            if os.path.isdir(sp):
                total += len([f for f in os.listdir(sp) if f.endswith('.json')])
    with _lock:
        _estado["n_secuencias"] = total

def _set_accion(msg):
    with _lock:
        _estado["ultima_accion"] = msg

def get_estado_json():
    _refrescar_estado()
    with _lock:
        return dict(_estado)

# =============================================================================
# ACCIONES
# =============================================================================

def accion_entrenar():
    with _lock:
        if _estado["entrenando"]:
            return "⏳ Entrenamiento ya en curso, espera..."
        _estado["entrenando"]    = True
        _estado["ultima_accion"] = "Entrenamiento iniciado"

    def _run():
        try:
            env  = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
            proc = subprocess.run(
                [sys.executable, os.path.join(DIR, "2_entrenar_modelo.py")],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=1800
            )
            msg = "✅ Entrenamiento completado" if proc.returncode == 0 \
                  else "⚠️ Entrenamiento terminó con error"
            if proc.returncode == 0:
                open(RELOAD_FLAG, "w").close()
        except subprocess.TimeoutExpired:
            msg = "⚠️ Entrenamiento cancelado (>30 min)"
        except Exception as e:
            msg = f"⚠️ Error: {e}"
        finally:
            _refrescar_estado()
            with _lock:
                _estado["entrenando"]    = False
                _estado["ultima_accion"] = msg
        _notificar_telegram(msg)

    threading.Thread(target=_run, daemon=True).start()
    return "🧠 Entrenamiento iniciado en segundo plano (puede tardar minutos)"


def accion_sincronizar():
    with _lock:
        if _estado["sincronizando"]:
            return "⏳ Sincronización ya en curso..."
        _estado["sincronizando"] = True
        _estado["ultima_accion"] = "Sincronización iniciada"

    def _run():
        try:
            from sync_cloud import SyncCloud
            sync = SyncCloud()
            if sync.conectar():
                sync.descargar_datos_senas()
                if sync.descargar_modelo_si_hay_nuevo():
                    open(RELOAD_FLAG, "w").close()
                    msg = "✅ Modelo nuevo descargado"
                else:
                    msg = "✅ Datos sincronizados (modelo ya al día)"
            else:
                msg = "📡 Sin internet o sin credenciales Firebase"
        except Exception as e:
            msg = f"⚠️ {e}"
        finally:
            _refrescar_estado()
            with _lock:
                _estado["sincronizando"] = False
                _estado["ultima_accion"] = msg
        _notificar_telegram(msg)

    threading.Thread(target=_run, daemon=True).start()
    return "☁️ Sincronizando con la nube..."


def accion_reiniciar():
    """Escribe una flag para que modo_traductor.py reinicie el traductor."""
    open(os.path.join(DIR_MODELO, ".reiniciar"), "w").close()
    _set_accion("Reinicio solicitado")
    return "🔄 El traductor se reiniciará en breve"


# Referencia al bot para poder notificar desde las acciones
_bot_ref = None

def _notificar_telegram(msg):
    global _bot_ref
    if _bot_ref:
        try:
            _bot_ref.broadcast(msg)
        except Exception:
            pass

# =============================================================================
# HTML DEL PANEL WEB (mobile-first, sin dependencias externas)
# =============================================================================

PANEL_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>LSE Control</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0d0d1a;color:#e8e8f0;min-height:100vh}
.hdr{background:linear-gradient(135deg,#1a1a3e,#0d1b4b);
     padding:18px 20px;display:flex;align-items:center;gap:12px;
     box-shadow:0 2px 12px #0006}
.hdr h1{font-size:1.25rem;color:#60c8ff;letter-spacing:.5px}
.dot{width:14px;height:14px;border-radius:50%;background:#00ff88;
     box-shadow:0 0 8px #00ff8888;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.dot.busy{background:#ffaa00;box-shadow:0 0 8px #ffaa0088}

.card{margin:14px 16px;background:#1a1a2e;border-radius:14px;
      padding:16px;border:1px solid #2a2a4a}
.card h3{font-size:.8rem;text-transform:uppercase;letter-spacing:1px;
         color:#6070a0;margin-bottom:10px}
.row{display:flex;justify-content:space-between;align-items:center;
     padding:7px 0;border-bottom:1px solid #1e2040;font-size:.92rem}
.row:last-child{border:none}
.val{color:#60c8ff;font-weight:600}
.ok{color:#00ff88}.warn{color:#ffaa00}.bad{color:#ff5555}

.grid{margin:0 16px;display:grid;grid-template-columns:1fr 1fr;gap:12px}
.btn{padding:20px 12px;border:none;border-radius:14px;font-size:.95rem;
     font-weight:700;cursor:pointer;transition:all .15s;text-align:center;
     letter-spacing:.3px;-webkit-tap-highlight-color:transparent}
.btn:active{transform:scale(.93);filter:brightness(.8)}
.btn-train {background:linear-gradient(135deg,#7c3aed,#5b21b6);color:#fff}
.btn-sync  {background:linear-gradient(135deg,#0ea5e9,#0369a1);color:#fff}
.btn-reset {background:linear-gradient(135deg,#f59e0b,#d97706);color:#111}
.btn-reload{background:linear-gradient(135deg,#22c55e,#15803d);color:#111}
.btn-full  {grid-column:1/-1}

.tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.tag{background:#1e2050;color:#60c8ff;border:1px solid #303070;
     border-radius:8px;padding:4px 12px;font-size:.8rem}

.log-bar{margin:14px 16px 20px;background:#1a1a2e;border-radius:10px;
         padding:12px 16px;font-size:.85rem;color:#808090;
         border-left:3px solid #60c8ff}
.log-bar b{color:#60c8ff}
</style>
</head>
<body>
<div class="hdr">
  <div class="dot" id="dot"></div>
  <h1>🤟 Traductor LSE</h1>
</div>

<div class="card">
  <h3>Estado del sistema</h3>
  <div class="row"><span>Traductor</span><span class="val" id="s-trad">—</span></div>
  <div class="row"><span>Precisión (test)</span><span id="s-acc">—</span></div>
  <div class="row"><span>Secuencias grabadas</span><span class="val" id="s-seq">—</span></div>
</div>

<div class="grid">
  <button class="btn btn-train"  onclick="accion('entrenar')">🧠 Entrenar</button>
  <button class="btn btn-sync"   onclick="accion('sincronizar')">☁️ Sincronizar</button>
  <button class="btn btn-reset"  onclick="accion('reiniciar')">🔄 Reiniciar</button>
  <button class="btn btn-reload" onclick="refrescar()">📊 Actualizar</button>
</div>

<div class="card" style="margin-top:14px">
  <h3>Señas en el modelo</h3>
  <div class="tags" id="tags"><span style="color:#6070a0">—</span></div>
</div>

<div class="log-bar"><b>Última acción:</b> <span id="log-msg">—</span></div>

<script>
function refrescar(){
  fetch('/estado').then(r=>r.json()).then(d=>{
    document.getElementById('s-trad').textContent =
      d.traductor==='activo' ? '✅ Activo' : '⏸ Detenido';
    const pct = (d.accuracy*100).toFixed(1)+'%';
    const el  = document.getElementById('s-acc');
    el.textContent = pct;
    el.className   = d.accuracy>=.85 ? 'ok val' : d.accuracy>=.70 ? 'warn val' : 'bad val';
    document.getElementById('s-seq').textContent = d.n_secuencias;
    document.getElementById('log-msg').textContent = d.ultima_accion;
    const busy = d.entrenando || d.sincronizando;
    document.getElementById('dot').className = 'dot' + (busy ? ' busy' : '');
    const tags = d.clases.map(c=>`<span class="tag">${c}</span>`).join('');
    document.getElementById('tags').innerHTML = tags || '<span style="color:#6070a0">—</span>';
  }).catch(()=>{});
}
function accion(nombre){
  fetch('/accion/'+nombre,{method:'POST'}).then(r=>r.json()).then(d=>{
    document.getElementById('log-msg').textContent = d.mensaje;
    setTimeout(refrescar, 2000);
  });
}
refrescar();
setInterval(refrescar, 5000);
</script>
</body>
</html>"""

# =============================================================================
# FLASK WEB SERVER
# =============================================================================

def _ip_local():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "localhost"


def iniciar_panel_web(port=5000):
    try:
        from flask import Flask, jsonify
    except ImportError:
        print("⚠️  Flask no instalado — panel web desactivado")
        print("     Instala con: pip install flask")
        return

    app = Flask(__name__)

    @app.route("/")
    def index():
        return PANEL_HTML

    @app.route("/estado")
    def estado():
        return jsonify(get_estado_json())

    @app.route("/accion/<nombre>", methods=["POST"])
    def accion(nombre):
        dispatch = {
            "entrenar":    accion_entrenar,
            "sincronizar": accion_sincronizar,
            "reiniciar":   accion_reiniciar,
        }
        fn  = dispatch.get(nombre)
        msg = fn() if fn else f"Acción desconocida: {nombre}"
        return jsonify({"ok": bool(fn), "mensaje": msg})

    ip = _ip_local()
    print(f"\n📱 Panel web activo → abre en el navegador del teléfono:")
    print(f"   http://{ip}:{port}\n")

    # Intentar mostrar QR en terminal si está disponible
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(f"http://{ip}:{port}")
        qr.make()
        qr.print_ascii(invert=True)
    except ImportError:
        pass

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port,
                               use_reloader=False, threaded=True),
        daemon=True
    ).start()


# =============================================================================
# TELEGRAM BOT (long-polling, sin librerías extra — solo requests)
# =============================================================================

def _cargar_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _guardar_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)


class _TelegramBot:
    API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token):
        self.token    = token
        self.offset   = 0
        cfg = _cargar_config()
        self.chat_ids = set(cfg.get("telegram_chat_ids", []))

    def _url(self, method):
        return self.API.format(token=self.token, method=method)

    def _get(self, method, **params):
        import requests
        try:
            r = requests.get(self._url(method), params=params, timeout=35)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def _post(self, method, **data):
        import requests
        try:
            requests.post(self._url(method), json=data, timeout=10)
        except Exception:
            pass

    def send(self, chat_id, text):
        self._post("sendMessage", chat_id=chat_id, text=text, parse_mode="HTML")

    def broadcast(self, text):
        for cid in self.chat_ids:
            self.send(cid, text)

    def _registrar(self, chat_id):
        if chat_id not in self.chat_ids:
            self.chat_ids.add(chat_id)
            cfg = _cargar_config()
            cfg["telegram_chat_ids"] = list(self.chat_ids)
            _guardar_config(cfg)

    def _handle(self, msg):
        chat_id = msg.get("chat", {}).get("id")
        text    = (msg.get("text") or "").strip()
        if not chat_id or not text.startswith("/"):
            return

        self._registrar(chat_id)
        cmd = text.split()[0].lstrip("/").split("@")[0].lower()

        if cmd == "start":
            self.send(chat_id,
                "🤟 <b>Traductor LSE — Control remoto</b>\n\n"
                "Comandos disponibles:\n"
                "/estado — Estado del sistema\n"
                "/entrenar — Entrenar el modelo con datos nuevos\n"
                "/sincronizar — Descargar datos/modelo de la nube\n"
                "/reiniciar — Reiniciar el traductor\n"
            )
        elif cmd == "estado":
            d   = get_estado_json()
            acc = f"{d['accuracy']*100:.1f}%"
            cl  = ", ".join(d["clases"]) or "—"
            self.send(chat_id,
                f"📊 <b>Estado del sistema</b>\n\n"
                f"Traductor: {'✅ Activo' if d['traductor']=='activo' else '⏸ Detenido'}\n"
                f"Precisión: {acc}\n"
                f"Secuencias grabadas: {d['n_secuencias']}\n"
                f"Entrenando: {'⏳ Sí' if d['entrenando'] else 'No'}\n"
                f"Sincronizando: {'⏳ Sí' if d['sincronizando'] else 'No'}\n"
                f"Señas: {cl}\n"
                f"Última acción: {d['ultima_accion']}"
            )
        elif cmd == "entrenar":
            self.send(chat_id, accion_entrenar())
        elif cmd == "sincronizar":
            self.send(chat_id, accion_sincronizar())
        elif cmd == "reiniciar":
            self.send(chat_id, accion_reiniciar())
        else:
            self.send(chat_id, f"❓ Comando desconocido: <code>/{cmd}</code>\nUsa /start para ver opciones")

    def poll(self):
        print("🤖 Bot de Telegram activo (esperando comandos...)")
        while True:
            try:
                resp = self._get("getUpdates", offset=self.offset, timeout=25)
                for upd in resp.get("result", []):
                    self.offset = upd["update_id"] + 1
                    msg = upd.get("message") or upd.get("edited_message")
                    if msg:
                        self._handle(msg)
            except Exception:
                pass
            time.sleep(1)


def iniciar_bot_telegram():
    global _bot_ref
    cfg   = _cargar_config()
    token = cfg.get("telegram_token", "").strip()
    if not token:
        print("ℹ️  Bot de Telegram no configurado.")
        print(f"   Crea un bot con @BotFather y pon el token en:")
        print(f"   {CONFIG_PATH}")
        print('   Formato: {"telegram_token": "123456:ABC..."}')
        return None

    try:
        import requests  # noqa: F401
    except ImportError:
        print("⚠️  requests no instalado — bot de Telegram desactivado")
        print("     pip install requests")
        return None

    bot = _TelegramBot(token)
    _bot_ref = bot
    threading.Thread(target=bot.poll, daemon=True).start()
    return bot


# =============================================================================
# PUNTO DE ENTRADA: inicia ambos servicios como daemons
# =============================================================================

def iniciar_todo(port=5000):
    """Llama esto desde modo_traductor.py para levantar web + telegram."""
    _refrescar_estado()
    iniciar_panel_web(port=port)
    iniciar_bot_telegram()
