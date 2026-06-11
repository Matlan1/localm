@echo off
rem localm launcher — double-click to open the graphical launcher.
rem Lets you pick GUI / chat / server / coder mode, model, debug, and options.
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "launcher.pyw"
) else (
    start "" pythonw "launcher.pyw"
)
