@echo off
rem ===========================================================================
rem  LocaLM - roll back a bad update, even when localm will not start.
rem
rem  Double-click this to restore the PREVIOUS localm version from the last
rem  update's backup. Use it when an update left localm broken or misbehaving and
rem  `localm update --rollback` will not run because the new build will not start.
rem
rem  It also undoes an update that was interrupted.
rem
rem  Optional args pass straight through:  --yes  (skip the confirmation prompt and
rem  the key press at the end)
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title LocaLM - roll back an update

set "HOLD=1"
for %%A in (%*) do (
    if /i "%%~A"=="--yes" set "HOLD="
    if /i "%%~A"=="-y" set "HOLD="
)

rem An update that is not finished is undone; an older backup is then left alone.
if exist ".localm-update\job.json" (
    call scripts\update_gate.bat
    call set "RC=%%errorlevel%%"
    echo.
    if defined HOLD pause
    call exit /b %%RC%%
)

rem Prefer the clone's venv Python; else any Python on PATH. The rollback script is
rem stdlib-only and loads the restore helper by file path, so a venv that cannot
rem import localm still rolls back fine.
set "RBPY="
if exist "%~dp0.venv\Scripts\python.exe" set "RBPY=%~dp0.venv\Scripts\python.exe"
if not defined RBPY for %%P in (py.exe python.exe python3.exe) do if not defined RBPY (
    where %%P >nul 2>nul && set "RBPY=%%P"
)

rem The rollback replaces this file while it runs, so the run, the hold and the
rem exit are one block: nothing more is read from the file after it returns.
if defined RBPY (
    "%RBPY%" "%~dp0scripts\rollback_update.py" %*
    call set "RC=%%errorlevel%%"
    echo.
    if defined HOLD pause
    call exit /b %%RC%%
)
echo No Python found to run the rollback. Restore the previous version by hand:
echo copy the contents of home\updates\backup back into this folder.
echo.
if defined HOLD pause
exit /b 1
