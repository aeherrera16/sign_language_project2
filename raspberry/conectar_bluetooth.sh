#!/bin/bash
# ============================================================
#  VINCULADOR BLUETOOTH LSE - Para Raspberry Pi
#  Busca y conecta automáticamente tu parlante, ej: "XTS-600"
# ============================================================

NOMBRE_PARLANTE="${1:-XTS-600}"
SCAN_SECONDS="${SCAN_SECONDS:-15}"

echo "========================================"
echo " 🎧 BUSCADOR DE PARLANTE: $NOMBRE_PARLANTE"
echo "========================================"
echo "Asegúrate de que tu parlante esté encendido y en MODO EMPAREJAMIENTO (titilando)."
echo "Buscando... (esto puede tardar hasta ${SCAN_SECONDS} segundos)"

if ! command -v bluetoothctl >/dev/null 2>&1; then
    echo "❌ No se encontró 'bluetoothctl'."
    echo "Instala Bluetooth con: sudo apt install -y bluez"
    exit 1
fi

# Asegurar que el Bluetooth esté encendido
sudo rfkill unblock bluetooth
sudo systemctl start bluetooth >/dev/null 2>&1 || true
bluetoothctl power on >/dev/null
bluetoothctl agent on >/dev/null
bluetoothctl default-agent >/dev/null

echo "Reiniciando escaneo..."
bluetoothctl scan off >/dev/null 2>&1 || true
sleep 1
bluetoothctl scan on >/dev/null 2>&1 || true
sleep "$SCAN_SECONDS"
bluetoothctl scan off >/dev/null 2>&1 || true

# Buscar por nombre exacto/parcial en dispositivos detectados y/o ya vinculados
MAC=$( (bluetoothctl devices; bluetoothctl paired-devices) | grep -iF "$NOMBRE_PARLANTE" | awk '{print $2}' | head -n 1 )



if [ -z "$MAC" ]; then
    echo "❌ No se encontró ningún dispositivo llamado '$NOMBRE_PARLANTE'."
    echo "Intenta de nuevo asegurándote de que esté en modo emparejamiento."
    echo "Dispositivos detectados recientemente:"
    bluetoothctl devices | sed 's/^/  - /'
    exit 1
fi

echo "✅ Parlante encontrado ($NOMBRE_PARLANTE). Dirección MAC: $MAC"
echo "Vinculando y conectando..."

# Ejecutar comandos de conexión
bluetoothctl pair "$MAC" >/dev/null 2>&1 || true
sleep 2
bluetoothctl trust "$MAC" >/dev/null 2>&1 || true
sleep 1
bluetoothctl connect "$MAC" >/dev/null

if ! bluetoothctl info "$MAC" | grep -q "Connected: yes"; then
    echo "❌ No se pudo conectar al dispositivo '$NOMBRE_PARLANTE' ($MAC)."
    echo "Prueba ponerlo otra vez en modo emparejamiento y ejecutar de nuevo."
    exit 1
fi

echo "========================================"
echo "🎉 ¡Listo! El audio ahora saldrá por el $NOMBRE_PARLANTE."
echo "La próxima vez que lo enciendas, se conectará automáticamente."
