@echo off
rem localm launcher - double-click to open the graphical launcher.
rem Lets you pick GUI / chat / server / coder mode, model, debug, and options.
rem (Use this instead of launcher.pyw - .pyw files have no association when
rem Python comes from uv; this always uses the clone's own .venv.)

rem ---- an update that is not finished --------------------------------------------
rem  Undoing it may replace this file: the command ends in the new start or in exit.
if not exist "%~dp0.localm-update\job.json" goto :update_gated
if defined LOCALM_GATE goto :update_gated
set "LOCALM_GATE=1"
set "GATE=%~dp0scripts\update_gate.bat"
call "%%GATE%%" && "%~dp0localm-launcher.bat" %* || (
    set "LOCALM_GATE="
    set "GATE="
    pause
    exit /b 1
)
:update_gated
set "LOCALM_GATE="
set "GATE="
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "launcher.pyw"
    exit /b 0
)

echo No .venv found in %~dp0
echo Run setup.bat in that folder first - it creates a private .venv there.
pause
