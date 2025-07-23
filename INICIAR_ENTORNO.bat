@echo off
echo.
echo 🇪🇨 LSE ECUADOR - ACTIVANDO ENTORNO
echo ========================================
echo.

REM Verificar si estamos en el directorio correcto
if not exist "main_interface.py" (
    echo ❌ Error: No se encuentra main_interface.py
    echo 💡 Asegurate de ejecutar este archivo desde la carpeta del proyecto
    echo.
    pause
    exit /b 1
)

REM Verificar si existe el entorno virtual
if not exist "venv310\Scripts\activate.bat" (
    echo ⚠️ Entorno virtual no encontrado
    echo 💡 Creando entorno virtual...
    python -m venv venv310
    if errorlevel 1 (
        echo ❌ Error creando entorno virtual
        pause
        exit /b 1
    )
)

REM Activar entorno virtual
echo 🔧 Activando entorno virtual...
call venv310\Scripts\activate.bat

REM Verificar que Python funciona
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no disponible
    pause
    exit /b 1
)

echo ✅ Entorno activado correctamente
echo.
echo 🚀 COMANDOS DISPONIBLES:
echo    python main_interface.py                    - Interfaz principal FUNCIONAL
echo    python scripts\recognition\real_time_translate.py - Reconocimiento directo
echo    python verificacion_sistema_completo.py     - Verificar sistema completo
echo    python limpiar_data_para_nuevas_grabaciones.py - Limpiar datos
echo    python verificador_senas_lse.py             - Verificar señas LSE
echo.
echo 💡 El entorno está listo. La interfaz principal ya NO tiene errores!
echo.

REM Mantener la ventana abierta con el entorno activado
cmd /k
