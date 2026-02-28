@echo off
REM ============================================================
REM  TRADUCTOR LSE - Lengua de Señas Ecuatoriana
REM  Doble clic para abrir. Instala todo automáticamente.
REM ============================================================

title Traductor LSE - Lengua de Señas Ecuatoriana
cd /d "%~dp0"

REM Verificar que Python esté instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ===============================================
    echo   ERROR: Python no esta instalado
    echo ===============================================
    echo.
    echo Descarga Python 3.10 desde:
    echo   https://www.python.org/downloads/release/python-31011/
    echo.
    echo IMPORTANTE: Al instalar, marca la opcion:
    echo   [X] Add Python to PATH
    echo.
    echo Despues de instalar Python, vuelve a ejecutar este archivo.
    echo.
    pause
    exit /b 1
)

REM Si no existe el entorno virtual, crearlo e instalar dependencias
if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo ===============================================
    echo   Primera ejecucion - Configurando...
    echo   Esto solo ocurre una vez.
    echo ===============================================
    echo.
    echo [1/3] Creando entorno virtual...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo   Entorno virtual creado.

    call .venv\Scripts\activate.bat

    echo.
    echo [2/3] Actualizando pip...
    python -m pip install --upgrade pip >nul 2>&1

    echo.
    echo [3/3] Instalando dependencias (puede tardar unos minutos)...
    pip install opencv-python mediapipe tensorflow pyttsx3 scikit-learn
    if errorlevel 1 (
        echo ERROR: No se pudieron instalar las dependencias.
        pause
        exit /b 1
    )

    echo.
    echo ===============================================
    echo   Configuracion completada!
    echo   Abriendo la aplicacion...
    echo ===============================================
    echo.
) else (
    call .venv\Scripts\activate.bat
)

REM Silenciar warnings
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
set MEDIAPIPE_DISABLE_GPU=1
set GLOG_minloglevel=3
set ABSL_MIN_LOG_LEVEL=3
set PYTHONWARNINGS=ignore

REM Abrir menú gráfico
python prototipo\menu.py

if errorlevel 1 (
    echo.
    echo Ha ocurrido un error. Revisa los mensajes anteriores.
    pause
)
