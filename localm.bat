@echo off
rem localm launcher. No arguments: opens an interactive chat with the first
rem registered chat model (or LOCALM_MODEL if set). With arguments: forwards them
rem to the CLI - needed because cmd.exe checks the current directory before
rem PATH, so a bare "localm ..." typed from this folder would otherwise hit
rem this file instead of the global shim and silently open chat.
rem The current directory is left as it is, so a relative path argument means
rem the folder it was typed in.
title LocaLM

set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto :have_venv
echo No .venv found in %~dp0
echo Run setup.bat in that folder first - it creates a private .venv there.
set "RC=1"
goto :held

:have_venv
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

rem Default: the first chat model in the registry, by the rule the launcher
rem and the Models page use (is_llm).
rem The query runs from this folder with the relative .venv path: usebackq
rem strips the backtick command's leading quote, so the interpreter path must
rem be unquoted and space-free there.
set "MODEL="
pushd "%~dp0"
for /f "usebackq delims=" %%m in (`.venv\Scripts\python.exe -c "from localm.config import load_registry; from localm.model_manager import is_llm; r=load_registry(); n=sorted(k for k, v in r.items() if is_llm(v)); print(n[0] if n else '')"`) do set "MODEL=%%m"
popd

if "%MODEL%"=="" (
    echo No chat model registered yet. From this window you can run:
    echo   "%~dp0localm.bat" pull ^<name^>   to download one
    echo   "%~dp0localm.bat" add ^<path^>    to register a local file
    echo   "%~dp0localm.bat" gui            to open the web GUI, which offers both
    set "RC=1"
    goto :held
)

:run
echo Starting LocaLM with model: %MODEL%
"%PY%" -m localm run "%MODEL%"
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
