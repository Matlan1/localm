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

rem Default: first model name from the registry.
rem NB: do not quote %PY% inside this for /f - "usebackq" strips the leading
rem quote of the backtick command, which mangled the path into
rem '.venv\Scripts\python.exe" -c "from' ... not recognized. %PY% is always a
rem space-free value here ("python" or the relative .venv path), so it is safe
rem unquoted; the -c argument keeps its own quotes.
for /f "usebackq delims=" %%m in (`%PY% -c "from localm.config import load_registry; r=load_registry(); print(sorted(r)[0] if r else '')"`) do set "MODEL=%%m"

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
