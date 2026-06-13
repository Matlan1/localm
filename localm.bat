@echo off
rem localm launcher - double-click to open an interactive chat.
rem Picks the first registered model unless you set LOCALM_MODEL.
cd /d "%~dp0"
title localm

rem Prefer this clone's own .venv (created by setup.bat) - self-contained
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not "%LOCALM_MODEL%"=="" (
    set "MODEL=%LOCALM_MODEL%"
    goto run
)

rem Default: first model name from the registry
for /f "usebackq delims=" %%m in (`"%PY%" -c "from localm.config import load_registry; r=load_registry(); print(sorted(r)[0] if r else '')"`) do set "MODEL=%%m"

if "%MODEL%"=="" (
    echo No models registered yet.
    echo   localm pull ^<name^>   to download one
    echo   localm add ^<path^>    to register a local file
    pause
    exit /b 1
)

:run
echo Starting localm with model: %MODEL%
"%PY%" -m localm run %MODEL%
if errorlevel 1 pause
