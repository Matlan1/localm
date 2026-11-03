@echo off
rem ===========================================================================
rem  LocaLM - report a problem, even when localm will not start.
rem
rem  Double-click this to file a bug report as an account-less GitHub issue via
rem  the localm proxy (no GitHub login). It shows you exactly what will be sent
rem  and asks you to confirm first. It works even when the install is broken or
rem  missing: it runs the self-contained reporter on a Python localm brought, and
rem  falls back to a pure-PowerShell reporter when there is none.
rem
rem  Optional args (used by setup.bat's failure paths) pass straight through:
rem    --summary "..."  --detail "..."  --log <path>  --yes
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title LocaLM - report a problem

rem Runs only a Python localm brought: the clone's venv, else the clone's own
rem .python, else the Shared folder's Python, with inherited uv, pip and Python
rem settings and LOCALM_HOME cleared. All run the SAME stdlib-only reporter,
rem which needs no working localm install (so a broken venv that cannot import
rem localm still reports fine).
set "ComSpec=%SystemRoot%\System32\cmd.exe"
for /f "delims==" %%v in ('set UV_ 2^>nul') do set "%%v="
for /f "delims==" %%v in ('set PIP_ 2^>nul') do set "%%v="
for %%v in (PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX LOCALM_HOME) do set "%%v="
set "UV_NO_CONFIG=1"
set "PIP_CONFIG_FILE=NUL"
set "PYTHONNOUSERSITE=1"
set "RIPY="
if exist "%~dp0.venv\Scripts\python.exe" set "RIPY=%~dp0.venv\Scripts\python.exe"
if not defined RIPY for /d %%P in ("%~dp0.python\cpython-3*") do if not defined RIPY if exist "%%~P\python.exe" set "RIPY=%%~P\python.exe"
if not defined RIPY if defined LOCALAPPDATA for /d %%P in ("%LOCALAPPDATA%\localm\shared\python\cpython-3*") do if not defined RIPY if exist "%%~P\python.exe" set "RIPY=%%~P\python.exe"

if defined RIPY (
    "%RIPY%" "%~dp0scripts\report_issue.py" %*
    goto :done
)

rem No Python of localm's own (e.g. setup failed before uv provisioned one): pure PowerShell.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\report_issue.ps1" %*

:done
echo.
pause
