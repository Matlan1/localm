@echo off
rem localm launcher. No arguments: opens an interactive chat with the first
rem registered model (or LOCALM_MODEL if set). With arguments: forwards them
rem to the CLI - needed because cmd.exe checks the current directory before
rem PATH, so a bare "localm ..." typed from this folder would otherwise hit
rem this file instead of the global shim and silently open chat.
cd /d "%~dp0"
title LocaLM

rem Prefer this clone's own .venv (created by setup.bat) - self-contained
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

rem Forwarding must stay outside any ( ... ) block: cmd.exe expands
rem %errorlevel% once at parse time, so a block would always report the
rem errorlevel from before it started, not from the command that just ran.
rem The goto below keeps "exit /b %errorlevel%" its own line so it reads the
rem real exit code after "%PY%" -m localm %* runs.
if not "%~1"=="" goto :forward

if not "%LOCALM_MODEL%"=="" (
    set "MODEL=%LOCALM_MODEL%"
    goto run
)

rem Default: first model name from the registry.
rem %PY% must stay unquoted here - usebackq strips the backtick command's
rem leading quote, which mangles a quoted path. %PY% is always space-free
rem ("python" or the relative .venv path), so this is safe.
for /f "usebackq delims=" %%m in (`%PY% -c "from localm.config import load_registry; r=load_registry(); print(sorted(r)[0] if r else '')"`) do set "MODEL=%%m"

if "%MODEL%"=="" (
    echo No models registered yet.
    echo   localm pull ^<name^>   to download one
    echo   localm add ^<path^>    to register a local file
    pause
    exit /b 1
)

:run
echo Starting LocaLM with model: %MODEL%
"%PY%" -m localm run %MODEL%
set "RC=%errorlevel%"
if not "%RC%"=="0" goto :held
exit /b 0

:forward
"%PY%" -m localm %*
set "RC=%errorlevel%"
if not "%RC%"=="0" goto :held
rem Debug asks for the log to be readable, so the window is held even on a
rem clean exit.
echo %* | findstr /I /C:"--debug" >nul && goto :held
exit /b 0

rem "if errorlevel 1" is a >= test against a SIGNED value, so a native fault's
rem negative exit code (an access violation exits -1073741819) never matched
rem it and the window closed on exactly the crashes worth reading. RC is
rem compared as a string, which has no sign to get wrong.
:held
pause
exit /b %RC%
