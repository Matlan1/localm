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

where pythonw >nul 2>nul
if not errorlevel 1 (
    start "" pythonw "launcher.pyw"
    exit /b 0
)

echo No Python environment found.
echo Run setup.bat first - it creates a private .venv in this folder.
pause
