@echo off
title LSE Ecuador - Sistema de Reconocimiento de Señas
color 0A

echo.
echo 🇪🇨 LSE ECUADOR - SISTEMA DE RECONOCIMIENTO DE SEÑAS
echo ===================================================
echo.
echo Activando entorno virtual...
call venv310\Scripts\activate.bat

echo.
echo ✅ Entorno activado
echo 🚀 Iniciando sistema de reconocimiento...
echo.
echo INSTRUCCIONES:
echo  👋 Realiza gestos frente a la cámara
echo  🎯 Los gestos reconocidos se pronunciarán automáticamente  
echo  ⏹️ Presiona 'q' en la ventana de la cámara para salir
echo  📊 Gestos disponibles: hola, gracias, si, no, adios
echo.

python scripts\recognition\real_time_translate.py

echo.
echo 👋 Sistema cerrado. ¡Gracias por usar LSE Ecuador!
pause
