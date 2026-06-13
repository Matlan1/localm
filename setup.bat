@echo off
rem ===========================================================================
rem  localm self-contained setup - double-click after cloning, anywhere.
rem
rem  Creates a private .venv in THIS folder and (optionally) keeps all data
rem  (models, config, logs, generated images) in THIS folder too.  Multiple
rem  clones on one machine never see or affect each other.  Nothing is
rem  installed globally and PATH is not modified.
rem ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title localm setup

echo.
echo  localm setup - self-contained install in: %CD%
echo.

rem ---- uv is required (fast, reliable resolver; handles the ROCm wheels) ----
where uv >nul 2>nul
if errorlevel 1 (
    echo  [!] uv is not installed. Install it first:
    echo      winget install astral-sh.uv
    echo      or: powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo.
    pause
    exit /b 1
)

rem ---- choose the install flavour -------------------------------------------
echo  Install flavour:
echo    [1] Full - AMD GPU stack (ROCm torch, transformers; needs Python 3.12)
echo    [2] Base - CPU / GGUF-via-llama.dll only (any Python 3.10+)
choice /c 12 /n /m "  Pick 1 or 2: "
if errorlevel 2 (set "FLAVOUR=base") else (set "FLAVOUR=gpu")

rem ---- create the venv in the repo root -------------------------------------
if "%FLAVOUR%"=="gpu" (set "PYVER=3.12") else (set "PYVER=3.12")
echo.
echo  Creating .venv (Python %PYVER%) ...
uv venv --python %PYVER% .venv
if errorlevel 1 (
    echo  [!] Could not create the environment. Is Python %PYVER% available?
    echo      uv can fetch it:  uv python install %PYVER%
    pause
    exit /b 1
)

rem ---- install localm (editable) into the venv ------------------------------
rem  [voice] ships the speech-to-text package preinstalled; the Whisper model
rem  itself is only downloaded after the user consents in the GUI (privacy:
rem  that one fetch is the only network access, transcription is local).
echo.
echo  Installing localm into .venv ...
if "%FLAVOUR%"=="gpu" (
    uv pip install -p .venv -e ".[gpu,coder,audio,voice]"
) else (
    uv pip install -p .venv -e ".[coder,voice]"
)
if errorlevel 1 (
    echo  [!] Install failed - see the error above.
    pause
    exit /b 1
)

rem ---- install the native-runtime wheel (self-contained inference) ----------
rem  localm-llama-runtime carries llama.dll + ggml inside this venv so the
rem  project never depends on a folder elsewhere on disk. Installed empty here;
rem  `localm setup-llama` downloads/copies the actual binaries into it.
uv pip install -p .venv -e ".\runtime"

rem ---- provision the native llama.cpp binaries ------------------------------
rem  The DLLs (llama.dll + ggml + ROCm runtime) are large and license- and
rem  provenance-sensitive, so they are never committed to git - a clone does
rem  not carry them. Place them in this venv now so the install is runnable.
if "%FLAVOUR%"=="gpu" goto provision_gpu
echo.
echo  CPU GGUF inference needs a CPU-built llama.dll (the default prebuilt is a
echo  GPU build). Provision it yourself with:
echo     .venv\Scripts\localm setup-llama --from ^<your cpu llama.cpp build dir^>
echo.
goto provision_done
:provision_gpu
echo.
echo  Native llama.cpp binaries are not in the repo and must be placed in this
echo  venv once. How do you want to provide them?
echo    [1] Download the default gfx1030 prebuilt   one network fetch, lemonade-sdk
echo    [2] Copy from your OWN llama.cpp build dir   no download
echo    [3] Skip for now
choice /c 123 /n /m "  Pick 1, 2 or 3: "
set "LLAMAPICK=%errorlevel%"
if "%LLAMAPICK%"=="1" .venv\Scripts\localm setup-llama
if "%LLAMAPICK%"=="2" (
    set /p "LLAMABUILD=  Path to your build dir with llama.dll: "
    .venv\Scripts\localm setup-llama --from "!LLAMABUILD!"
)
if "%LLAMAPICK%"=="3" echo  Skipped - provision later: .venv\Scripts\localm setup-llama
:provision_done

rem ---- choose where data lives ----------------------------------------------
echo.
echo  Where should localm keep its data (models, config, logs, images)?
echo    [1] Inside this folder (.\home) - fully portable, isolated per clone
echo    [2] Shared per-user folder (%USERPROFILE%\.localm) - clones share
echo        models and settings
echo    [3] Custom path
choice /c 123 /n /m "  Pick 1, 2 or 3: "
set "DATAPICK=%errorlevel%"
if "%DATAPICK%"=="1" (
    if not exist "home" mkdir "home"
    if exist "localm-home.cfg" del "localm-home.cfg"
    echo  Data directory: %CD%\home
)
if "%DATAPICK%"=="2" (
    if exist "localm-home.cfg" del "localm-home.cfg"
    rem an empty/no marker + no home\ dir = shared default; remove a stale
    rem portable dir only if it is empty
    if exist "home" rd "home" 2>nul
    echo  Data directory: %USERPROFILE%\.localm  (shared)
)
if "%DATAPICK%"=="3" (
    set /p CUSTOMHOME="  Enter data directory path: "
    if "!CUSTOMHOME!"=="" (
        echo  [!] Empty path - falling back to shared %USERPROFILE%\.localm
        if exist "localm-home.cfg" del "localm-home.cfg"
    ) else (
        > "localm-home.cfg" echo !CUSTOMHOME!
        if not exist "!CUSTOMHOME!" mkdir "!CUSTOMHOME!"
        echo  Data directory: !CUSTOMHOME!  (recorded in localm-home.cfg)
    )
)

rem ---- optional desktop shortcut ----------------------------------------------
echo.
choice /c YN /n /m "  Create a desktop shortcut that opens the localm GUI? [Y/N] "
if not errorlevel 2 (
    powershell -NoProfile -Command ^
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\localm.lnk');" ^
        "$s.TargetPath = '%CD%\.venv\Scripts\localm.exe';" ^
        "$s.Arguments = 'gui';" ^
        "$s.WorkingDirectory = '%CD%';" ^
        "$s.IconLocation = '%CD%\assets\localm.ico';" ^
        "$s.Description = 'localm - open the web GUI';" ^
        "$s.Save()"
    if not errorlevel 1 echo  Shortcut created: Desktop\localm.lnk
)

rem ---- done ------------------------------------------------------------------
echo.
echo  Done. This clone is self-contained:
echo    localm-launcher.bat   graphical launcher (GUI / chat / server / coder)
echo                          (use this, not launcher.pyw - .pyw has no file
echo                          association when Python comes from uv)
echo    localm.bat            terminal chat with the default model
echo    .venv\Scripts\localm  CLI directly, e.g.:
echo        .venv\Scripts\localm pull ^<model^>
echo        .venv\Scripts\localm gui
echo.
if "%FLAVOUR%"=="gpu" (
    echo  Re-provision or update the native binaries any time with:
    echo        .venv\Scripts\localm setup-llama --force
    echo.
)
echo  Tip: avoid "uv tool install" for this project - tool installs are
echo  global per package name and clones would overwrite each other.
echo.
pause
