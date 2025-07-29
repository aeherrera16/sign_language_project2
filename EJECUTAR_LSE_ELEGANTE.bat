@echo off
echo 🇪🇨 LSE ECUADOR - Sistema de Reconocimiento de Señas
echo ================================================
echo.
echo 🚀 Iniciando interfaz elegante...
echo.

cd /d "%~dp0"
call venv310\Scripts\activate.bat
python main_interface_elegante.py

pause
