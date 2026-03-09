@echo off
REM ============================================================
REM  TRADUCTOR LSE - Configuración y lanzamiento
REM  Este archivo es llamado por Traductor_LSE.vbs
REM  También puede ejecutarse directamente como fallback.
REM ============================================================

title Traductor LSE - Configurando...
cd /d "%~dp0"

REM ============================================================
REM  PASO 1: Buscar Python disponible
REM ============================================================
set PYTHON_CMD=

REM Intentar python
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :python_found
)

REM Intentar python3
python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    goto :python_found
)

REM Intentar py launcher
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3
    goto :python_found
)

REM Intentar en la ruta por defecto de instalacion per-user
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python310\;%LOCALAPPDATA%\Programs\Python\Python310\Scripts\;%PATH%"
    goto :python_found
)

REM ============================================================
REM  PASO 1b: Python NO encontrado - Descargar e instalar
REM ============================================================
echo.
echo ===============================================
echo   Descargando Python 3.10...
echo   Esto solo ocurre una vez.
echo ===============================================
echo.

set PYTHON_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
set INSTALLER=%TEMP%\python_installer_lse.exe

echo   Descargando desde python.org...
echo   (Puede tardar unos minutos segun tu conexion)
echo.

REM Intentar con PowerShell
powershell -Command "& {$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('%PYTHON_URL%','%INSTALLER%')}" 2>nul

if not exist "%INSTALLER%" (
    REM Intentar con curl
    curl -L -o "%INSTALLER%" "%PYTHON_URL%" 2>nul
)

if not exist "%INSTALLER%" (
    echo.
    echo   ERROR: No se pudo descargar Python.
    echo   Verifica tu conexion a internet.
    echo.
    pause
    exit /b 1
)

echo   Descarga completada. Instalando...
echo.

REM Instalar Python silenciosamente (per-user, no necesita admin)
"%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_tcltk=1 Include_launcher=1 Include_test=0

REM Esperar a que termine
timeout /t 3 /nobreak >nul

REM Limpiar instalador
del "%INSTALLER%" >nul 2>&1

REM Actualizar PATH para esta sesion
set "PATH=%LOCALAPPDATA%\Programs\Python\Python310\;%LOCALAPPDATA%\Programs\Python\Python310\Scripts\;%PATH%"

REM Verificar
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    echo   Python instalado correctamente.
    echo.
    goto :python_found
)

REM Si la instalacion silenciosa fallo, intentar la normal
echo   La instalacion silenciosa no funciono.
echo   Se abrira el instalador de Python.
echo   IMPORTANTE: Marca [X] Add Python to PATH
echo.

REM Descargar de nuevo si se borro
if not exist "%INSTALLER%" (
    powershell -Command "& {$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('%PYTHON_URL%','%INSTALLER%')}" 2>nul
)

if exist "%INSTALLER%" (
    "%INSTALLER%"
    del "%INSTALLER%" >nul 2>&1
)

REM Verificar de nuevo
set "PATH=%LOCALAPPDATA%\Programs\Python\Python310\;%LOCALAPPDATA%\Programs\Python\Python310\Scripts\;%PATH%"
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :python_found
)

echo.
echo   No se pudo instalar Python automaticamente.
echo   Por favor descarga Python 3.10 manualmente desde:
echo   https://www.python.org/downloads/release/python-31011/
echo   Marca: [X] Add Python to PATH
echo.
pause
exit /b 1

:python_found
echo   Python encontrado: %PYTHON_CMD%

REM ============================================================
REM  PASO 2: Configurar entorno virtual
REM ============================================================
if exist ".venv\Scripts\pythonw.exe" goto :launch

echo.
echo ===============================================
echo   Configurando la aplicacion...
echo ===============================================
echo.

echo   [1/3] Creando entorno virtual...
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
    echo   ERROR: No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo   [2/3] Actualizando pip...
python -m pip install --upgrade pip >nul 2>&1

echo   [3/3] Instalando dependencias...
echo         (opencv, mediapipe, tensorflow, pyttsx3, scikit-learn)
echo         Esto puede tardar varios minutos...
echo.
pip install opencv-python mediapipe tensorflow pyttsx3 scikit-learn

if errorlevel 1 (
    echo.
    echo   ERROR: No se pudieron instalar las dependencias.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo   Configuracion completada!
echo ===============================================
echo.

REM ============================================================
REM  PASO 3: Lanzar la aplicacion
REM ============================================================
:launch
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
set MEDIAPIPE_DISABLE_GPU=1
set GLOG_minloglevel=3
set ABSL_MIN_LOG_LEVEL=3
set PYTHONWARNINGS=ignore

REM Lanzar con pythonw (SIN terminal visible)
start "" ".venv\Scripts\pythonw.exe" prototipo\menu.py
exit /b 0
