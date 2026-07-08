@echo off
rem ===========================================================================
rem  LocaLM - report a problem, even when localm will not start.
rem
rem  Double-click this to file a bug report as an account-less GitHub issue via
rem  the localm proxy (no GitHub login). It shows you exactly what will be sent
rem  and asks you to confirm first. It works even when the install is broken or
rem  missing: it runs the self-contained reporter on any Python it can find, and
rem  falls back to a pure-PowerShell reporter when there is no Python at all.
rem
rem  Optional args (used by setup.bat's failure paths) pass straight through:
rem    --summary "..."  --detail "..."  --log <path>  --yes
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title LocaLM - report a problem

rem Prefer the clone's venv Python; else any Python on PATH. All run the SAME
rem stdlib-only reporter, which needs no working localm install (so a broken venv
rem that cannot import localm still reports fine).
set "RIPY="
if exist "%~dp0.venv\Scripts\python.exe" set "RIPY=%~dp0.venv\Scripts\python.exe"
if not defined RIPY for %%P in (py.exe python.exe python3.exe) do if not defined RIPY (
    where %%P >nul 2>nul && set "RIPY=%%P"
)

if defined RIPY (
    "%RIPY%" "%~dp0scripts\report_issue.py" %*
    goto :done
)

rem No Python at all (e.g. setup failed before uv provisioned one): pure PowerShell.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\report_issue.ps1" %*

:done
echo.
pause
