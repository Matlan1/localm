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

rem ---- portable vs shared: where localm's Python tooling lives ---------------
rem  The question setup.bat asks, asked once, here. The window reads the answer
rem  from LOCALM_GUI_STORE (portable or shared), the uv in use from LOCALM_GUI_UV,
rem  and LOCALM_GUI_UV_INSTALLED=1 when uv was installed at its per-user location
rem  by this run. Portable uses only .\.uv\uv.exe, installing it there when it is
rem  missing. Shared uses the uv on PATH, never .\.uv, and installs uv at its
rem  normal per-user location when there is none.
echo  Keep localm's Python tooling ^(uv itself, its runtime, and downloads^) inside this folder?
echo    [1] Portable - everything in this folder (self-contained; re-downloads per clone)
echo    [2] Shared   - reuse/install uv at its normal per-user location (faster; lives in your user profile)
set "STOREPICK="
rem SETUP-Q: tooling-store
set /p "STOREPICK=  Pick 1 or 2 [1]: "
if not defined STOREPICK set "STOREPICK=1"
set "UVEXE="
set "LOCALM_GUI_UV_INSTALLED="
if "%STOREPICK%"=="1" goto portable_uv

set "LOCALM_GUI_STORE=shared"
echo  Shared: reusing/installing uv and its Python + cache ^(outside this folder^).
call :find_shared_uv
if defined UVEXE goto have_uv
call :ask_install_uv
if errorlevel 1 goto uv_declined
echo.
echo  Installing uv ...
set "UV_INSTALL_DIR="
set "UV_UNMANAGED_INSTALL="
goto install_uv

:portable_uv
set "LOCALM_GUI_STORE=portable"
echo  Portable: uv, Python, and downloads all under this folder
if exist ".uv\uv.exe" goto portable_have_uv
call :ask_install_uv
if errorlevel 1 goto uv_declined
echo.
echo  Installing uv ...
rem  UV_UNMANAGED_INSTALL keeps it in .\.uv without adding that folder to
rem  the user PATH or writing an install receipt under %LOCALAPPDATA%\uv.
set "UV_INSTALL_DIR=%CD%\.uv"
set "UV_UNMANAGED_INSTALL=%CD%\.uv"
echo  Portable: installing uv itself under .\.uv

:install_uv
rem  The installer script and everything Astral's installer downloads stay in
rem  .\.localm-setup-tmp, which is removed again when it is empty.
set "STMP=.localm-setup-tmp"
if not exist "%STMP%" mkdir "%STMP%" >nul 2>nul
set "UV_INSTALLER_VERSION=0.13.6"
set "UV_INSTALLER_SHA256=d00580051699bee58a2e2d9b925f45788b35f7e0fff7966fd5bedfc3c547dc7d"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; $f=$null; $t=$null; try { $d=$env:STMP; if ($d) { $d=(Resolve-Path -LiteralPath $d -ErrorAction Stop).ProviderPath } else { $d=[IO.Path]::GetTempPath() }; $f=Join-Path $d ('uv-installer-'+[guid]::NewGuid()+'.ps1'); try { Invoke-WebRequest -UseBasicParsing -Uri ('https://github.com/astral-sh/uv/releases/download/'+$env:UV_INSTALLER_VERSION+'/uv-installer.ps1') -OutFile $f -ErrorAction Stop } catch { exit 61 }; $h=$null; $fs=$null; try { $fs=[IO.File]::Open($f,'Open','Read','Read'); $h=[BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($fs)).Replace('-','') } catch { }; if ($h -ne $env:UV_INSTALLER_SHA256) { exit 62 }; $t=Join-Path $d ('uv-tmp-'+[guid]::NewGuid()); try { $null=New-Item -ItemType Directory -Path $t -ErrorAction Stop } catch { $t=$null; exit 63 }; $env:TEMP=$t; $env:TMP=$t; $env:PSModulePath=$null; & powershell -NoProfile -ExecutionPolicy Bypass -File $f; exit $LASTEXITCODE } finally { if ($fs) { $fs.Dispose() }; if ($f) { Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue }; if ($t) { Remove-Item -LiteralPath $t -Recurse -Force -ErrorAction SilentlyContinue } }"
set "UVRC=%errorlevel%"
rd "%STMP%" >nul 2>nul
if "%UVRC%"=="61" echo   [!] Could not download the uv %UV_INSTALLER_VERSION% installer.
if "%UVRC%"=="62" echo   [!] The downloaded uv installer did not match its expected checksum, or could not be read to check it, and was not run.
if "%UVRC%"=="63" echo   [!] Could not create a temporary folder in %STMP% for the uv installer.
if "%UVRC%"=="61" goto uv_refused
if "%UVRC%"=="62" goto uv_refused
if "%UVRC%"=="63" goto uv_refused
if "%LOCALM_GUI_STORE%"=="portable" goto portable_installed
set "LOCALM_GUI_UV_INSTALLED=1"
rem  Prepend every directory the per-user install may have used, in setup.bat's
rem  own order.
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%HOMEDRIVE%%HOMEPATH%\.local\bin;%PATH%"
call :find_shared_uv
if defined UVEXE goto have_uv
goto uv_not_callable

:portable_installed
if not exist ".uv\uv.exe" goto uv_not_callable
:portable_have_uv
set "UVEXE=%CD%\.uv\uv.exe"

:have_uv
set "LOCALM_GUI_UV=%UVEXE%"
echo  Using %UVEXE%
goto open_window

:uv_not_callable
echo.
echo   [!] uv still is not callable, so the graphical setup cannot start.
if "%LOCALM_GUI_STORE%"=="portable" echo       Looked for %CD%\.uv\uv.exe ^(the installer did not leave one there^).
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
rem  Portable keeps the interpreter this window runs on inside the folder, so the
rem  install reuses it rather than downloading a second copy.
if "%LOCALM_GUI_STORE%"=="portable" set "UV_PYTHON_INSTALL_DIR=%CD%\.python"
if "%LOCALM_GUI_STORE%"=="portable" set "UV_CACHE_DIR=%CD%\.cache"
set "UV_SYSTEM_CERTS=1"
"%UVEXE%" run --no-project --isolated --python 3.12 python "installer\gui.py"
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

rem ===========================================================================
rem  :find_shared_uv - sets UVEXE to the first uv on PATH that is not this
rem  folder's .\.uv copy, or leaves it undefined.
rem ===========================================================================
:find_shared_uv
set "UVEXE="
for /f "delims=" %%i in ('where uv 2^>nul') do if not defined UVEXE if /i not "%%~dpi"=="%CD%\.uv\" set "UVEXE=%%i"
goto :eof

rem ===========================================================================
rem  :ask_install_uv - setup.bat's uv-install question. Exit code 1 when the
rem  user declines.
rem ===========================================================================
:ask_install_uv
echo  [!] uv (the Python package manager localm builds on) is not installed.
set "GETUV="
rem SETUP-Q: uv-install
set /p "GETUV=  Install it now with Astral's official installer? [Y/n]: "
if not defined GETUV set "GETUV=Y"
if /i "%GETUV:~0,1%"=="N" exit /b 1
exit /b 0
