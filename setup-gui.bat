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
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo   LocaLM graphical setup
echo.

rem ---- locate uv -------------------------------------------------------------
rem  Prefer a portable uv already inside this folder (setup.bat's Portable
rem  option puts one there), then whatever is on PATH.
set "UVEXE="
if exist ".uv\uv.exe" set "UVEXE=%CD%\.uv\uv.exe"
if not defined UVEXE (
    where uv >nul 2>nul
    if not errorlevel 1 set "UVEXE=uv"
)

if not defined UVEXE (
    echo   uv ^(the Python package manager LocaLM builds on^) is not installed yet.
    echo   It is a small download and is needed before any window can open.
    echo.
    set "GETUV="
    set /p "GETUV=  Install it now? [Y/n]: "
    if not defined GETUV set "GETUV=Y"
    if /i "!GETUV:~0,1!"=="N" (
        echo.
        echo   Nothing was installed. Run setup.bat for the console installer.
        pause
        exit /b 1
    )
    echo   Installing uv ...
    set "UV_INSTALL_DIR=%CD%\.uv"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    rem  Astral's installer updates the persistent user PATH, which this already
    rem  running shell does not see. Prepend every directory it may have used, in
    rem  setup.bat's own order, so the uv just installed is callable right now.
    set "PATH=%CD%\.uv;%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%HOMEDRIVE%%HOMEPATH%\.local\bin;%PATH%"
    if exist ".uv\uv.exe" (
        set "UVEXE=%CD%\.uv\uv.exe"
    ) else (
        where uv >nul 2>nul
        if not errorlevel 1 set "UVEXE=uv"
    )
)

if not defined UVEXE (
    echo.
    echo   [!] uv still is not callable, so the graphical setup cannot start.
    echo       Open a NEW terminal and run setup.bat instead.
    pause
    exit /b 1
)

rem ---- open the installer window ---------------------------------------------
rem  --no-project so uv never tries to resolve this repo as its own project, and
rem  an explicit --python so the interpreter is the managed 3.12 the install
rem  targets rather than whatever else is on the machine.
echo   Opening the setup window ...
rem  Keep the interpreter this window runs on inside the folder, so a portable
rem  install reuses it rather than downloading a second copy.
set "UV_PYTHON_INSTALL_DIR=%CD%\.python"
set "UV_CACHE_DIR=%CD%\.cache"
"%UVEXE%" run --no-project --python 3.12 python "installer\gui.py"
set "RC=!errorlevel!"

if not "!RC!"=="0" (
    echo.
    echo   [!] The setup window could not run ^(exit !RC!^).
    echo       Use the console installer instead:  setup.bat
    pause
    exit /b !RC!
)
endlocal
