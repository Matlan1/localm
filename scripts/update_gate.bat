@echo off
rem  Waits for an update of LocaLM that is running, or puts the previous version back
rem  after one that was interrupted. The launchers and setup call this, while there is
rem  an unfinished update, before anything of LocaLM runs. Exit code 0: LocaLM may
rem  start; anything else: it may not, and the reason was shown.
rem  The Python is the one in .venv, then the one in .python, then the one .venv was
rem  made from; never one from PATH. With the argument "window" it says everything in a
rem  window (update_gate.pyw).
rem  An update replaces this file while it runs: the run is one block that ends in exit.
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0.."
set "PY="
set "CAND=.venv\Scripts\python.exe"
call :try
for /d %%D in (".python\*") do (
    set "CAND=%%~D\python.exe"
    call :try
)
if not defined PY if exist ".venv\pyvenv.cfg" for /f "usebackq tokens=1,* delims== " %%A in (".venv\pyvenv.cfg") do if /i "%%A"=="home" (
    set "CAND=%%~B\python.exe"
    call :try
)
set "RUN=localm\__init__.py"
if "%~1"=="window" set "RUN=scripts\update_gate.pyw"
if defined PY (
    "%PY%" -I -S "%CD%\%RUN%"
    call set "RC=%%errorlevel%%"
    call exit /b %%RC%%
)
echo An update of LocaLM was interrupted and no Python is left in "%CD%" 1>&2
echo that can undo it. Download LocaLM, unzip it over that folder, delete the folder 1>&2
echo .localm-update in it and run setup. Your chats, settings and models stay. 1>&2
if "%~1"=="window" "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [void][Windows.Forms.MessageBox]::Show('An update of LocaLM was interrupted and no Python is left that can undo it in ' + (Get-Location) + '. Download LocaLM, unzip it over that folder, delete the folder .localm-update in it and run setup. Your chats, settings and models stay.', 'LocaLM')"
exit /b 4

:try
if defined PY exit /b 0
if not exist "%CAND%" exit /b 0
"%CAND%" -I -S -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>nul && set "PY=%CAND%"
exit /b 0
