@echo off
rem localm launcher - double-click to open the graphical launcher.
rem Lets you pick GUI / chat / server / coder mode, model, debug, and options.
rem (Use this instead of launcher.pyw - .pyw files have no association when
rem Python comes from uv; this always uses the clone's own .venv.)
cd /d "%~dp0"
setlocal

rem Runs only this folder's own Python, with inherited uv, pip and Python
rem settings and LOCALM_HOME cleared, so this install's own data folder
rem choice (localm-home.cfg or .\home) decides where data lives.
set "ComSpec=%SystemRoot%\System32\cmd.exe"
for /f "delims==" %%v in ('set UV_ 2^>nul') do set "%%v="
for /f "delims==" %%v in ('set PIP_ 2^>nul') do set "%%v="
for %%v in (PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX LOCALM_HOME) do set "%%v="
set "UV_NO_CONFIG=1"
set "PIP_CONFIG_FILE=NUL"
set "PYTHONNOUSERSITE=1"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "launcher.pyw"
    exit /b 0
)

echo LocaLM is not set up in this folder yet: there is no .venv here.
echo Run setup.bat first - it creates a private .venv in this folder.
pause
exit /b 1
