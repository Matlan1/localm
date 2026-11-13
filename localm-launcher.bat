@echo off
rem localm launcher - double-click to open the graphical launcher.
rem Lets you pick GUI / chat / server / coder mode, model, debug, and options.
rem (Use this instead of launcher.pyw - .pyw files have no association when
rem Python comes from uv; this always uses the clone's own .venv.)
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "launcher.pyw"
    exit /b 0
)

echo No .venv found in %~dp0
echo Run setup.bat in that folder first - it creates a private .venv there.
pause
