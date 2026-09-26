@echo off
setlocal

cd /d "%~dp0"

rem Refresca el PATH desde el registro (Machine + User), por si ffmpeg o
rem espeak-ng se instalaron despues de abrir esta terminal.
for /f "usebackq tokens=2,*" %%A in (`reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path`) do set "SYS_PATH=%%B"
for /f "usebackq tokens=2,*" %%A in (`reg query "HKCU\Environment" /v Path 2^>nul`) do set "USER_PATH=%%B"
set "PATH=%SYS_PATH%;%USER_PATH%;%PATH%"

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No se encontro ffmpeg en el PATH. Instalalo antes de continuar.
    pause
    exit /b 1
)

where espeak-ng >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No se encontro espeak-ng en el PATH. Instalalo antes de continuar.
    pause
    exit /b 1
)

set "VENV_PY=%~dp0.venv311\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] No se encontro el entorno virtual .venv311. Crealo con:
    echo   py -3.11 -m venv .venv311
    echo   .venv311\Scripts\python.exe -m pip install -r requirements.txt flask kokoro
    pause
    exit /b 1
)

echo Iniciando dubbing_engine en http://127.0.0.1:5000 ...
"%VENV_PY%" -m dubbing_engine.webapp.app

pause
