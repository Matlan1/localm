@echo off
rem ===========================================================================
rem  localm self-contained setup - double-click after cloning, anywhere.
rem
rem  Creates a private .venv in THIS folder and (optionally) keeps all data
rem  (models, config, logs, generated images) in THIS folder too.  Multiple
rem  clones on one machine never see or affect each other.  Nothing is
rem  installed globally and PATH is not modified.
rem
rem  Detects your GPU and provisions the matching llama.cpp backend, so an
rem  NVIDIA, Intel, AMD, or CPU-only machine all get a working install. Vulkan
rem  is the universal default (any GPU, no vendor toolkit); CUDA/ROCm are
rem  offered for peak performance; CPU for machines with no GPU.
rem ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title LocaLM setup

echo.
echo  LocaLM setup - self-contained install in: %CD%
echo.

rem ---- uninstall / rollback (report first, then remove) ---------------------
set "PURGE=0"
if /i "%~2"=="--purge-data" set "PURGE=1"
if /i "%~1"=="uninstall"   goto uninstall
if /i "%~1"=="--uninstall" goto uninstall
if /i "%~1"=="--rollback"  goto uninstall

rem ---- uv is required (fast, reliable resolver; handles the GPU wheels) ------
where uv >nul 2>nul
if errorlevel 1 (
    echo  [!] uv is not installed. Install it first:
    echo      winget install astral-sh.uv
    echo      or: powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo.
    pause
    exit /b 1
)

rem ---- create the venv in the repo root -------------------------------------
rem  An existing .venv is reused unless the user opts to replace it, so a
rem  re-run never ejects with a misleading "could not create" error (uv refuses
rem  to clobber an existing environment and returns non-zero). Python 3.12 is
rem  required by the AMD ROCm torch wheels; we standardise on it for all flavours.
set "PYVER=3.12"

if not exist ".venv" goto venv_create

rem .venv already exists - is it one we created, or a foreign one?
set "OURS=0"
if exist ".venv\.localm-venv" set "OURS=1"
if exist ".venv\Scripts\localm.exe" set "OURS=1"
echo.
if "%OURS%"=="1" (
    echo  An existing localm .venv was found in this folder.
    choice /c YN /n /m "  Replace it and reinstall from scratch? [y/N]: "
) else (
    echo  [!] A .venv exists here but does not look like a localm environment.
    echo      Replacing it deletes its current contents.
    choice /c YN /n /m "  Replace this foreign .venv? [y/N]: "
)
rem choice sets errorlevel: 1=Y, 2=N. Test the higher index first.
if errorlevel 2 (
    echo  Keeping the existing .venv and continuing setup.
    goto venv_done
)

:venv_create
echo.
echo  Creating .venv (Python %PYVER%) ...
uv venv --python %PYVER% --clear .venv
if errorlevel 1 (
    echo  [!] Could not create the environment. Is Python %PYVER% available?
    echo      uv can fetch it:  uv python install %PYVER%
    pause
    exit /b 1
)
type nul > ".venv\.localm-venv"
:venv_done

rem ---- install localm (editable) into the venv ------------------------------
rem  Base install first: GGUF chat needs no PyTorch, so this alone is a working
rem  install. The GPU/torch stack for HuggingFace models is added below to match
rem  the detected vendor. [voice] ships speech-to-text; its Whisper model is only
rem  downloaded after the user consents in the GUI.
echo.
echo  Installing localm into .venv ...
uv pip install -p .venv -e ".[coder,voice,monitor]"
if errorlevel 1 (
    echo  [!] Install failed - see the error above.
    pause
    exit /b 1
)

rem ---- install the native-runtime wheel (self-contained inference) ----------
rem  localm-llama-runtime carries llama.dll + ggml inside this venv so the
rem  project never depends on a folder elsewhere on disk. Installed empty here;
rem  `localm setup-llama` downloads/copies the actual binaries into it below.
uv pip install -p .venv -e ".\runtime"

rem ---- detect the GPU + recommended backend (the ONE tested policy) ----------
rem  `python -m localm.hwdetect` prints "<vendor> <install-backend>" - the same
rem  arch-aware policy setup.sh uses, so the two installers can never drift. It
rem  knows the self-contained gfx103X ROCm build only fits AMD RX 6000 / unknown
rem  on Windows; a clearly newer/older AMD card is steered to vulkan instead.
echo.
echo  Detecting graphics hardware ...
set "VENDOR=none"
set "REC=cpu"
.venv\Scripts\python -m localm.hwdetect > "%TEMP%\localm_hw.txt" 2>nul
if exist "%TEMP%\localm_hw.txt" (
    for /f "usebackq tokens=1,2" %%a in ("%TEMP%\localm_hw.txt") do (
        set "VENDOR=%%a"
        set "REC=%%b"
    )
)
del "%TEMP%\localm_hw.txt" 2>nul
if "%VENDOR%"=="" set "VENDOR=none"
if "%REC%"=="" set "REC=cpu"
echo  Detected graphics vendor: %VENDOR%
echo  Recommended inference backend: %REC%

rem ---- choose the llama.cpp backend (recommended pre-selected) ---------------
echo.
echo  Native inference runtime (llama.cpp) - press Enter to accept the recommendation:
echo    [1] %REC%   (recommended for your hardware)
echo    [2] vulkan     - any GPU (AMD/NVIDIA/Intel), no vendor toolkit
echo    [3] cuda       - NVIDIA, peak performance (fetches the CUDA runtime for you)
echo    [4] amd-rocm   - AMD RX 6000 (gfx103X), self-contained
echo    [5] cpu        - no GPU
echo    [6] I will build / provide my own (skip the download)
echo    (the chosen backend is verified after install; if it cannot load here,
echo     setup falls back to vulkan, then cpu, so you are never left broken)
set "BSEL="
set /p "BSEL=  Pick 1-6 [1]: "
if not defined BSEL set "BSEL=1"
set "BACKEND=%REC%"
if "%BSEL%"=="2" set "BACKEND=vulkan"
if "%BSEL%"=="3" set "BACKEND=cuda"
if "%BSEL%"=="4" set "BACKEND=amd-rocm"
if "%BSEL%"=="5" set "BACKEND=cpu"
if "%BSEL%"=="6" set "BACKEND=own"

rem ---- PyTorch + transformers for HuggingFace models (FOLLOWS your backend) --
rem  PyTorch powers the HuggingFace/transformers backend; GGUF chat needs none of
rem  it. The variant FOLLOWS the llama.cpp backend you picked above (not just the
rem  detected vendor), so choosing the vendor-neutral 'vulkan' runtime does NOT
rem  drag in the AMD ROCm stack (the reported SETUP-1 surprise). One shared policy
rem  decides it - `hwdetect torch-args <backend>` resolves the exact wheel SOURCE
rem  for THIS hardware+OS (including AMD-on-Windows per gfx family: gfx103X uses the
rem  bundled self-contained build, RX 7000/9000 use AMD's Windows ROCm wheels), so
rem  setup.bat and setup.sh can never disagree and every card gets correct packages.
set "TORCHSPEC="
.venv\Scripts\python -m localm.hwdetect torch-args %BACKEND% > "%TEMP%\localm_torch.txt" 2>nul
if exist "%TEMP%\localm_torch.txt" for /f "usebackq delims=" %%a in ("%TEMP%\localm_torch.txt") do set "TORCHSPEC=%%a"
del "%TEMP%\localm_torch.txt" 2>nul
echo.
if not defined TORCHSPEC (
    echo  Skipping the PyTorch/transformers stack ^(not needed for GGUF chat^).
    echo  You picked the '%BACKEND%' runtime, so no vendor GPU torch was auto-installed.
    echo  To use HuggingFace transformers models, add PyTorch for your setup later:
    echo      CPU ^(any machine^):    uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo      NVIDIA CUDA:          uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126
    echo      AMD ROCm ^(gfx103X^):   uv pip install -p .venv -e ".[gpu]"
) else if "%TORCHSPEC%"=="-e .[gpu]" (
    rem  gfx103X (RX 6000): the bundled self-contained build carries torch + the HF
    rem  stack + the ROCm runtime; add audio (soundfile) for unified-audio models.
    echo  Installing PyTorch ^(AMD ROCm, gfx103X^) + transformers for HuggingFace models ...
    uv pip install -p .venv -e ".[gpu,audio]" || echo  [!] ROCm torch stack failed - GGUF chat still works without it.
) else (
    echo  Installing PyTorch + transformers for HuggingFace models ...
    echo %TORCHSPEC% | find "rocm6.4" >nul && echo    ^(note: AMD ROCm on Windows is in public preview - expect rough edges^)
    uv pip install -p .venv %TORCHSPEC% || echo  [!] torch install failed - GGUF chat still works without it.
    uv pip install -p .venv "transformers[kernels]~=5.12" "tokenizers==0.22.2" "accelerate>=1.0" "pillow>=10.0" "soundfile>=0.12" || echo  [!] transformers stack failed - GGUF chat still works.
)

rem ---- provision the native llama.cpp binaries ------------------------------
rem  The binaries are large and license/provenance-sensitive, so they are never
rem  committed to git. setup-llama fetches the prebuilt matching the chosen
rem  backend from upstream llama.cpp releases (AMD uses a self-contained ROCm
rem  build), and places them in this venv so the install is runnable.
echo.
if /i "%BACKEND%"=="own" (
    set "LLAMABUILD="
    set /p "LLAMABUILD=  Path to your llama.cpp build dir with llama.dll (blank = skip): "
    if not "!LLAMABUILD!"=="" (
        .venv\Scripts\localm setup-llama --from "!LLAMABUILD!"
        if errorlevel 1 echo  [!] Provisioning failed - run later: .venv\Scripts\localm setup-llama --from "!LLAMABUILD!"
    ) else (
        echo  Skipped - provision later: .venv\Scripts\localm setup-llama --backend ^<vulkan^|cuda^|amd-rocm^|cpu^>
    )
) else (
    .venv\Scripts\localm setup-llama --backend %BACKEND%
    if errorlevel 1 echo  [!] Provisioning failed - run later: .venv\Scripts\localm setup-llama --backend %BACKEND%
)

rem ---- choose where data lives ----------------------------------------------
echo.
echo  Where should localm keep its data (models, config, logs, images)?
echo    [1] Inside this folder (.\home) - fully portable, isolated per clone
echo    [2] Shared per-user folder (%USERPROFILE%\.localm) - clones share
echo        models and settings
echo    [3] Custom path
rem  set /p (type a number, then Enter), NOT `choice`: `choice` returns on a single
rem  keypress, so the user's habitual confirming Enter used to leak into the custom
rem  path's set /p below and be read as an empty path (SETUP-2). With set /p the
rem  Enter belongs to THIS prompt, so the path prompt starts clean.
set "DATAPICK="
set /p "DATAPICK=  Pick 1, 2 or 3 [1]: "
if not defined DATAPICK set "DATAPICK=1"
rem DATADIR + DATACREATED feed the install manifest (DATACREATED=1 only when WE
rem made the dir, so uninstall --purge-data never removes a pre-existing folder).
set "DATADIR=%USERPROFILE%\.localm"
set "DATACREATED=0"
if "%DATAPICK%"=="1" (
    if not exist "home" mkdir "home"
    if exist "localm-home.cfg" del "localm-home.cfg"
    set "DATADIR=%CD%\home"
    set "DATACREATED=1"
    echo  Data directory: %CD%\home
)
if "%DATAPICK%"=="2" (
    if exist "localm-home.cfg" del "localm-home.cfg"
    rem an empty/no marker + no home\ dir = shared default; remove a stale
    rem portable dir only if it is empty
    if exist "home" rd "home" 2>nul
    if exist "home" (
        rem A non-empty ./home survived: localm prefers a portable ./home, so
        rem shared mode would be silently ignored. Warn loudly and record the
        rem dir localm will actually use, rather than printing a false "shared".
        echo  [!] ./home is not empty - shared mode will NOT take effect while it
        echo      exists ^(localm keeps using the portable ./home^). It may hold
        echo      your models/config, so it was left in place. Remove it manually
        echo      and re-run setup to switch to the shared dir.
        set "DATADIR=%CD%\home"
    ) else (
        echo  Data directory: %USERPROFILE%\.localm  ^(shared^)
    )
)
rem  Single-line `if ... call` into a goto/label subroutine (defined at the end):
rem  a `call` plus nested if/else INSIDE this `if (...)` block trips cmd.exe's
rem  parenthesis parser ("The syntax of the command is incorrect."), so keep the
rem  custom-path flow out of the block entirely.
if "%DATAPICK%"=="3" call :do_custom_home

rem ---- optional desktop shortcut ----------------------------------------------
echo.
echo  Desktop shortcut - what should it open?
echo    [1] Launcher (pick mode/model, set an API key)   recommended
echo    [2] Web GUI directly
echo    [3] No shortcut
rem  set /p for a consistent "type a number then Enter" across every menu.
set "SCPICK="
set /p "SCPICK=  Pick 1, 2 or 3 [1]: "
if not defined SCPICK set "SCPICK=1"
set "SCPATH="
if "%SCPICK%"=="1" set "SCPATH=%USERPROFILE%\Desktop\LocaLM.lnk"
if "%SCPICK%"=="2" set "SCPATH=%USERPROFILE%\Desktop\LocaLM.lnk"
if "%SCPICK%"=="1" (
    powershell -NoProfile -Command ^
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\LocaLM.lnk');" ^
        "$s.TargetPath = '%CD%\localm-launcher.bat';" ^
        "$s.WorkingDirectory = '%CD%';" ^
        "$s.IconLocation = '%CD%\assets\localm.ico';" ^
        "$s.Description = 'LocaLM - open the launcher';" ^
        "$s.Save()"
    if not errorlevel 1 echo  Shortcut created: Desktop\LocaLM.lnk  ^(opens the launcher^)
)
if "%SCPICK%"=="2" (
    powershell -NoProfile -Command ^
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\LocaLM.lnk');" ^
        "$s.TargetPath = '%CD%\.venv\Scripts\localm.exe';" ^
        "$s.Arguments = 'gui';" ^
        "$s.WorkingDirectory = '%CD%';" ^
        "$s.IconLocation = '%CD%\assets\localm.ico';" ^
        "$s.Description = 'LocaLM - open the web GUI';" ^
        "$s.Save()"
    if not errorlevel 1 echo  Shortcut created: Desktop\LocaLM.lnk  ^(opens the GUI^)
)
if "%SCPICK%"=="3" echo  No shortcut created.

rem ---- choose which plugins to enable (chat is always on) -------------------
echo.
echo  Which optional features (plugins) do you want? chat is always on.
.venv\Scripts\localm plugin setup

rem ---- record what we installed (uninstall removes ONLY what we created) -----
set "CRD="
if "%DATACREATED%"=="1" set "CRD=--data-created"
.venv\Scripts\python -m localm.install_manifest record --root . --venv "%CD%\.venv" --lib-dir "%CD%\runtime\localm_llama_runtime\lib" --data-dir "%DATADIR%" %CRD% --shortcut "%SCPATH%" >nul 2>nul
if errorlevel 1 echo  [!] Could not record the install manifest (uninstall will be conservative).

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
echo  Re-provision or change the inference backend any time with:
echo        .venv\Scripts\localm setup-llama --backend ^<auto^|vulkan^|cuda^|amd-rocm^|cpu^> --force
echo.
echo  Tip: avoid "uv tool install" for this project - tool installs are
echo  global per package name and clones would overwrite each other.
echo.
pause
exit /b 0

rem ===========================================================================
rem  Uninstall / rollback. The actual removal is delegated to the tested
rem  localm.install_manifest module, which removes ONLY what install recorded
rem  (never a derived/globbed/empty path) and hard-guards the one rm. The shell
rem  only removes the marker-checked .venv afterwards (a running interpreter
rem  cannot delete its own venv).
rem ===========================================================================
:uninstall
echo.
echo  LocaLM uninstall / rollback for this clone:
echo    %CD%
echo.
set "PYBIN=.venv\Scripts\python.exe"
set "PFLAG="
if "%PURGE%"=="1" set "PFLAG=--purge-data"
if exist "%PYBIN%" (
    echo  Planned removals ^(from the install manifest .localm-install.json^):
    "%PYBIN%" -m localm.install_manifest uninstall --root . %PFLAG% --dry-run
) else (
    echo  [!] No venv Python found - only the marked .venv will be removed.
)
echo.
choice /c YN /n /m "  Proceed? [y/N]: "
if errorlevel 2 (
    echo  Aborted - nothing changed.
    pause
    exit /b 0
)
rem --force: the dry-run above showed any unrecorded items + the at-your-own-risk
rem warning and the user chose to continue; the module still refuses catastrophic
rem paths (root/%USERPROFILE%/repo) regardless.
if exist "%PYBIN%" "%PYBIN%" -m localm.install_manifest uninstall --root . %PFLAG% --force
rem The manifest never deletes the running venv; remove it here, marker-checked.
if exist ".venv\.localm-venv" (
    rmdir /s /q .venv
    echo  Removed .\.venv
)
echo.
echo  Done. To reinstall: setup.bat
pause
exit /b 0

rem ===========================================================================
rem  :do_custom_home - prompt for a custom data directory and confirm it.
rem  goto/label flow (NO parenthesised blocks) so it is robust under cmd.exe, and
rem  every prompt is set /p: a habitual confirming Enter is consumed by the prompt
rem  it belongs to instead of leaking into the next one (the SETUP-2 bug, where
rem  `choice` auto-advanced and the stray Enter became an empty path). Records the
rem  path (DATADIR/DATACREATED + localm-home.cfg), or falls back to the shared
rem  default when left blank.
rem ===========================================================================
:do_custom_home
set "CUSTOMHOME="
set /p "CUSTOMHOME=  Enter the data directory path, or leave blank for the shared default: "
if not defined CUSTOMHOME goto custom_home_blank
set "OKHOME="
set /p "OKHOME=  Use '!CUSTOMHOME!'? [Y/n]: "
if not defined OKHOME set "OKHOME=Y"
if /i "!OKHOME:~0,1!"=="N" goto do_custom_home
> "localm-home.cfg" echo !CUSTOMHOME!
if not exist "!CUSTOMHOME!" mkdir "!CUSTOMHOME!"
set "DATADIR=!CUSTOMHOME!"
set "DATACREATED=1"
echo  Data directory: !CUSTOMHOME!  ^(recorded in localm-home.cfg^)
exit /b 0
:custom_home_blank
echo  [!] No path given - falling back to shared %USERPROFILE%\.localm
if exist "localm-home.cfg" del "localm-home.cfg"
exit /b 0
