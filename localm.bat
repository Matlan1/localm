@echo off
rem localm launcher. No arguments: opens an interactive chat with the first
rem registered chat model (or LOCALM_MODEL if set). With arguments: forwards them
rem to the CLI, so a bare "localm ..." typed in this folder (where cmd.exe finds
rem this file before the global shim on PATH) still reaches the CLI.
rem The current directory is left as it is, so a relative path argument means
rem the folder it was typed in.
rem An update or rollback replaces this file while localm runs. Each command that
rem starts localm is one block that ends in "exit", so nothing more is read from
rem the file after localm returns. A failure (any nonzero exit code, including a
rem native fault's negative one) or --debug holds the window.
title LocaLM
goto :main

rem Each "set" line below starts at a byte offset where an older localm.bat
rem resumes when an update replaced it while localm ran. Keep each at its offset.
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem-------
set "RC=%errorlevel%" & goto :resumed
rem------------------------------------------------------
set "RC=%errorlevel%" & goto :resumed
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem---------------------------------------------------------------------------
rem--------------------------
set "RC=%errorlevel%" & goto :resumed
rem------------------------------------------------------
set "RC=%errorlevel%" & goto :resumed
:main
rem ---- an update that is not finished --------------------------------------------
rem  Undoing it may replace this file: the command ends in the new start or in exit.
if not exist "%~dp0.localm-update\job.json" goto :update_gated
if defined LOCALM_GATE goto :update_gated
set "LOCALM_GATE=1"
call "%~dp0scripts\update_gate.bat" && "%~dp0localm.bat" %* || (
    pause
    exit /b 1
)
:update_gated
set "LOCALM_GATE="
set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto :have_venv
echo No .venv found in %~dp0
echo Run setup.bat in that folder first - it creates a private .venv there.
set "RC=1"
goto :held

:have_venv
set "HOLD="
if "%~1"=="" goto :chat
echo %* | findstr /I /C:"--debug" >nul && set "HOLD=1"
"%PY%" -m localm %* && (
    if defined HOLD pause
    exit /b 0
) || (
    call set "RC=%%errorlevel%%"
    pause
    call exit /b %%RC%%
)

:chat
if not "%LOCALM_MODEL%"=="" (
    set "MODEL=%LOCALM_MODEL%"
    goto :run
)
rem Default: the first chat model in the registry, by the rule the launcher and
rem the Models page use (is_llm). usebackq strips the leading quote of the
rem command, so the interpreter path is relative, unquoted and space-free.
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
"%PY%" -m localm run "%MODEL%" && exit /b 0 || (
    call set "RC=%%errorlevel%%"
    pause
    call exit /b %%RC%%
)

:resumed
if not "%RC%"=="0" goto :held
echo %* | findstr /I /C:"--debug" >nul && goto :held
exit /b 0

:held
pause
exit /b %RC%
