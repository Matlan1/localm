@echo off
rem  LocaLM graphical setup - the windowed alternative to setup.bat.
rem
rem  Double-click this file. It bootstraps uv (the only part that cannot be
rem  graphical, because something has to provide a Python before a window can
rem  exist), then hands the whole install over to installer\gui.py, which runs
rem  on uv's managed CPython and needs no other dependency: tkinter ships with
rem  it.
rem
rem  setup.bat remains the console installer and is unchanged. This is the same
rem  install, asked for in a window.
rem
rem  Delayed expansion stays off in this whole file, so a folder path containing
rem  "!" works. See test_setup_gui_runs_from_a_folder_with_a_bang_in_its_path.
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

echo.
echo   LocaLM graphical setup
echo.

rem ---- locate uv -------------------------------------------------------------
rem  Prefer a portable uv already inside this folder (setup.bat's Portable
rem  option puts one there), then whatever is on PATH.
set "UVEXE="
if exist ".uv\uv.exe" set "UVEXE=%CD%\.uv\uv.exe"
if not defined UVEXE where uv >nul 2>nul && set "UVEXE=uv"
if defined UVEXE goto open_window

echo   uv ^(the Python package manager LocaLM builds on^) is not installed yet.
echo   It is a small download and is needed before any window can open.
echo.
set "GETUV="
set /p "GETUV=  Install it now? [Y/n]: "
if not defined GETUV set "GETUV=Y"
if /i "%GETUV:~0,1%"=="N" goto uv_declined
echo   Installing uv ...
rem  UV_UNMANAGED_INSTALL keeps it in .\.uv without adding that folder to
rem  the user PATH or writing an install receipt under %LOCALAPPDATA%\uv.
set "UV_INSTALL_DIR=%CD%\.uv"
set "UV_UNMANAGED_INSTALL=%CD%\.uv"
set "UV_INSTALLER_VERSION=0.13.0"
set "UV_INSTALLER_SHA256=6d7ef89ba04838a03d1ebf7f84be4ba2b98b848a0a9ab8c2855658f13106c2c2"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; $f=Join-Path ([IO.Path]::GetTempPath()) ('uv-installer-'+[guid]::NewGuid()+'.ps1'); try { try { Invoke-WebRequest -UseBasicParsing -Uri ('https://github.com/astral-sh/uv/releases/download/'+$env:UV_INSTALLER_VERSION+'/uv-installer.ps1') -OutFile $f -ErrorAction Stop } catch { exit 61 }; $h=$null; try { $h=[BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([IO.File]::ReadAllBytes($f))).Replace('-','') } catch { }; if ($h -ne $env:UV_INSTALLER_SHA256) { exit 62 }; $env:PSModulePath=$null; & powershell -NoProfile -ExecutionPolicy Bypass -File $f; exit $LASTEXITCODE } finally { Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue }"
set "UVRC=%errorlevel%"
if "%UVRC%"=="61" echo   [!] Could not download the uv %UV_INSTALLER_VERSION% installer.
if "%UVRC%"=="62" echo   [!] The downloaded uv installer did not match its expected checksum and was not run.
if "%UVRC%"=="61" goto uv_refused
if "%UVRC%"=="62" goto uv_refused
rem  Prepend every directory uv may have been installed to, in setup.bat's own
rem  order, so the uv just installed is callable in this shell right now.
set "PATH=%CD%\.uv;%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%HOMEDRIVE%%HOMEPATH%\.local\bin;%PATH%"
if exist ".uv\uv.exe" set "UVEXE=%CD%\.uv\uv.exe"
if not defined UVEXE where uv >nul 2>nul && set "UVEXE=uv"
if defined UVEXE goto open_window

echo.
echo   [!] uv still is not callable, so the graphical setup cannot start.
echo       Open a NEW terminal and run setup.bat instead.
pause
exit /b 1

:uv_refused
echo.
echo   uv was not installed, so the graphical setup cannot start.
echo       Install uv yourself ^(winget install astral-sh.uv^), then run this again.
pause
exit /b 1

:uv_declined
echo.
echo   Nothing was installed. Run setup.bat for the console installer.
pause
exit /b 1

:open_window
rem ---- open the installer window ---------------------------------------------
rem  --no-project so uv never tries to resolve this repo as its own project, and
rem  an explicit --python so the interpreter is the managed 3.12 the install
rem  targets rather than whatever else is on the machine.
echo   Opening the setup window ...
rem  Keep the interpreter this window runs on inside the folder, so a portable
rem  install reuses it rather than downloading a second copy.
set "UV_PYTHON_INSTALL_DIR=%CD%\.python"
set "UV_CACHE_DIR=%CD%\.cache"
set "UV_SYSTEM_CERTS=1"
"%UVEXE%" run --no-project --python 3.12 python "installer\gui.py"
set "RC=%errorlevel%"

rem  42 and 43: an uninstall finished in the window; the Python runtime it ran
rem  on is removed now that it has closed. 43 and 44: something asked for was
rem  kept. 45: the uninstall did not finish.
set "PARTIAL="
if "%RC%"=="42" goto finish_uninstall
if "%RC%"=="43" set "PARTIAL=1"
if "%RC%"=="43" goto finish_uninstall
if "%RC%"=="44" goto uninstall_partial
if "%RC%"=="45" goto uninstall_failed
if "%RC%"=="0" exit /b 0
echo.
echo   [!] The setup window could not run (exit %RC%).
echo       Use the console installer instead:  setup.bat
pause
exit /b %RC%

:finish_uninstall
echo   Removing the last LocaLM folders ...
call ".\setup.bat" finish-uninstall
if errorlevel 1 goto finish_uninstall_left
if defined PARTIAL goto uninstall_partial
echo.
echo   LocaLM was uninstalled.
pause
exit /b 0
:uninstall_partial
echo.
echo   [!] LocaLM was removed, but some things you asked to delete were not
echo       deleted - the setup window listed them under REFUSED.
pause
exit /b 2
:uninstall_failed
echo.
echo   [!] The uninstall did not finish - the setup window listed why. Close
echo       any LocaLM window and run the setup again.
pause
exit /b 1
:finish_uninstall_left
echo.
echo   [!] Some LocaLM folders could not be removed - close any LocaLM window
echo       and delete them by hand, or run  setup.bat finish-uninstall  again.
pause
exit /b 1
