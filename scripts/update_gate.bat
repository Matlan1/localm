@echo off
rem  Waits for an update of LocaLM that is running, or puts the previous version back
rem  after one that was interrupted. The launchers and setup call this, while there is
rem  an unfinished update, before anything of LocaLM runs. Exit code 0: LocaLM may
rem  start; anything else: it may not, and the reason was shown.
rem  The Python is the one in .venv, then the one in .python, then the one .venv was
rem  made from; never one from PATH.
rem  An update replaces this file while it runs: the run is one block that ends in exit.
setlocal EnableExtensions DisableDelayedExpansion
set "ROOT=%~dp0.."
set "PY="
call :try "%ROOT%\.venv\Scripts\python.exe"
for /d %%D in ("%ROOT%\.python\*") do call :try "%%~D\python.exe"
if not defined PY if exist "%ROOT%\.venv\pyvenv.cfg" for /f "usebackq tokens=1,* delims== " %%A in ("%ROOT%\.venv\pyvenv.cfg") do if /i "%%A"=="home" call :try "%%~B\python.exe"
if defined PY (
    "%PY%" -I -S "%ROOT%\localm\__init__.py"
    call set "RC=%%errorlevel%%"
    call exit /b %%RC%%
)
echo An update of LocaLM was interrupted and no Python is left in "%ROOT%" 1>&2
echo that can undo it. Download LocaLM, unzip it over that folder, delete the folder 1>&2
echo .localm-update in it and run setup. Your chats, settings and models stay. 1>&2
exit /b 4

:try
if defined PY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -I -S -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>nul && set "PY=%~1"
exit /b 0
