@echo off
title LSE Ecuador - Sistema de Reconocimiento

echo.
echo 🇪🇨 LSE ECUADOR - INICIO RÁPIDO
echo ===============================
echo.

REM Activar entorno
call venv310\Scripts\activate.bat

REM Lanzar interfaz directamente
echo 🚀 Iniciando interfaz principal...
echo.
python main_interface.py

echo.
echo 👋 ¡Gracias por usar LSE Ecuador!
pause
