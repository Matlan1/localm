@echo off
rem ===========================================================================
rem  LocaLM - roll back a bad update, even when localm will not start.
rem
rem  Double-click this to restore the PREVIOUS localm version from the last
rem  update's backup. Use it when an update left localm broken or misbehaving and
rem  `localm update --rollback` will not run because the new build will not start.
rem
rem  Optional args pass straight through:  --yes  (skip the confirmation prompt)
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title LocaLM - roll back an update

rem Runs only a Python localm brought: the clone's venv, else the clone's own
rem .python, else the Shared folder's Python, with inherited uv, pip and Python
rem settings and LOCALM_HOME cleared. The rollback script is stdlib-only and
rem loads the restore helper by file path, so a venv that cannot import localm
rem still rolls back fine.
set "ComSpec=%SystemRoot%\System32\cmd.exe"
for /f "delims==" %%v in ('set UV_ 2^>nul') do set "%%v="
for /f "delims==" %%v in ('set PIP_ 2^>nul') do set "%%v="
for %%v in (PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX LOCALM_HOME) do set "%%v="
set "UV_NO_CONFIG=1"
set "PIP_CONFIG_FILE=NUL"
set "PYTHONNOUSERSITE=1"
set "RBPY="
if exist "%~dp0.venv\Scripts\python.exe" set "RBPY=%~dp0.venv\Scripts\python.exe"
if not defined RBPY for /d %%P in ("%~dp0.python\cpython-3*") do if not defined RBPY if exist "%%~P\python.exe" set "RBPY=%%~P\python.exe"
if not defined RBPY if defined LOCALAPPDATA for /d %%P in ("%LOCALAPPDATA%\localm\shared\python\cpython-3*") do if not defined RBPY if exist "%%~P\python.exe" set "RBPY=%%~P\python.exe"
if not defined RBPY goto :no_python
"%RBPY%" "%~dp0scripts\rollback_update.py" %*
set "RC=%errorlevel%"
echo.
pause
exit /b %RC%

:no_python
echo LocaLM's own Python is not in this folder, so the rollback cannot run.
echo Run setup.bat to set LocaLM up again, or restore the previous version by
echo hand: copy the contents of home\updates\backup back into this folder.
echo.
pause
exit /b 1
