@echo off
REM ============================================================
REM  TRADUCTOR LSE - Lengua de Señas Ecuatoriana
REM  Doble clic para abrir. Instala TODO automaticamente.
REM ============================================================

title Traductor LSE - Lengua de Señas Ecuatoriana
cd /d "%~dp0"

REM ============================================================
REM  PASO 1: Verificar si Python esta disponible
REM ============================================================
python --version >nul 2>&1
if not errorlevel 1 goto :python_ok

REM Intentar con python3
python3 --version >nul 2>&1
if not errorlevel 1 goto :python_ok

REM Intentar con py launcher
py -3 --version >nul 2>&1
if not errorlevel 1 goto :python_ok

REM ============================================================
REM  PASO 1b: Python NO encontrado - Instalar automaticamente
REM ============================================================
echo.
echo ===============================================
echo   Python no detectado - Instalando...
echo   Esto solo ocurre una vez.
echo ===============================================
echo.

REM Descargar Python 3.10.11 (compatible con MediaPipe)
set PYTHON_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
set INSTALLER=%TEMP%\python_installer.exe

echo [1/2] Descargando Python 3.10...
echo       Esto puede tardar unos minutos...
powershell -Command "(New-Object Net.WebClient).DownloadFile('%PYTHON_URL%', '%INSTALLER%')"

if not exist "%INSTALLER%" (
    echo.
    echo ERROR: No se pudo descargar Python.
    echo Verifica tu conexion a internet e intenta de nuevo.
    echo.
    echo Alternativa manual: descarga Python 3.10 desde
    echo   https://www.python.org/downloads/release/python-31011/
    echo   Marca: [X] Add Python to PATH
    echo.
    pause
    exit /b 1
)

echo [2/2] Instalando Python 3.10 (instalacion silenciosa)...
echo       Espera, esto puede tardar unos minutos...
"%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_tcltk=1 Include_launcher=1

if errorlevel 1 (
    echo.
    echo La instalacion silenciosa fallo.
    echo Intentando instalacion normal (sigue las instrucciones)...
    echo IMPORTANTE: Marca la casilla "Add Python to PATH"
    echo.
    "%INSTALLER%"
)

REM Limpiar instalador
del "%INSTALLER%" >nul 2>&1

REM Actualizar PATH para esta sesion
set "PATH=%LOCALAPPDATA%\Programs\Python\Python310\;%LOCALAPPDATA%\Programs\Python\Python310\Scripts\;%PATH%"

REM Verificar de nuevo
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ===============================================
    echo   Python se instalo pero necesitas reiniciar.
    echo   Cierra esta ventana y vuelve a hacer
    echo   doble clic en Iniciar_LSE.bat
    echo ===============================================
    echo.
    pause
    exit /b 0
)

echo.
echo   Python instalado correctamente!
echo.

:python_ok
REM ============================================================
REM  PASO 2: Configurar entorno virtual si no existe
REM ============================================================
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
        py -3 -m venv .venv
        if errorlevel 1 (
            echo ERROR: No se pudo crear el entorno virtual.
            pause
            exit /b 1
        )
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

REM ============================================================
REM  PASO 3: Abrir la aplicacion
REM ============================================================
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
set MEDIAPIPE_DISABLE_GPU=1
set GLOG_minloglevel=3
set ABSL_MIN_LOG_LEVEL=3
set PYTHONWARNINGS=ignore

python prototipo\menu.py

if errorlevel 1 (
    echo.
    echo Ha ocurrido un error. Revisa los mensajes anteriores.
    pause
)
