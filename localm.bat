@echo off
rem localm launcher - double-click (no arguments) to open an interactive chat,
rem picking the first registered model unless you set LOCALM_MODEL.
rem
rem Called WITH arguments instead (localm doctor, localm serve, localm pull...):
rem forward them verbatim to the real CLI. cmd.exe searches the CURRENT
rem DIRECTORY before PATH, so typing a bare "localm <anything>" from inside
rem this clone finds THIS file first, even if the proper global `localm` shim
rem (installed by setup.bat) is also on PATH. Without this forwarding, every
rem subcommand typed from this folder silently no-ops into "open chat" instead
rem (CODER-DOCTOR-NOOP) - "localm doctor" silently ran chat instead of the
rem diagnostics the user actually asked for.
cd /d "%~dp0"
title LocaLM

rem Prefer this clone's own .venv (created by setup.bat) - self-contained
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

rem NB: this must NOT be "if (...) ( cmd & exit /b %errorlevel% )" - cmd.exe
rem expands %errorlevel% ONCE, at PARSE time for the whole parenthesized block,
rem before any line inside it runs. Inside a block that would always re-report
rem whatever errorlevel existed BEFORE the block started (here, always 0, from
rem the "if exist ...set PY=..." line above) - so a forwarded command that
rem genuinely fails would still exit 0, silently hiding the failure from any
rem script/CI that checks localm.bat's exit code. A "goto" out of the block
rem sidesteps this: "exit /b %errorlevel%" below is its own standalone line,
rem so %errorlevel% is read fresh, after "%PY%" -m localm %* actually runs.
rem Known narrow gap: %~1 strips quotes, so a literal empty-quoted first
rem argument ("" ...) reads as "no arguments" and falls through to the
rem double-click path instead of forwarding. No real caller in this repo
rem (a user typing a command, setup.bat, or the globalcmd.py shim) ever
rem passes that shape, so it is left as-is rather than adding fragile
rem quote-parsing for an unreachable case.
if not "%~1"=="" goto :forward

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
echo Starting LocaLM with model: %MODEL%
"%PY%" -m localm run %MODEL%
if errorlevel 1 pause
exit /b %errorlevel%

:forward
"%PY%" -m localm %*
exit /b %errorlevel%
