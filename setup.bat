@echo off
rem ===========================================================================
rem  localm self-contained setup - double-click after cloning, anywhere.
rem
rem  Creates a private .venv in THIS folder and (optionally) keeps all data
rem  (models, config, logs, generated images) - and the Python runtime itself -
rem  in THIS folder too.  Multiple clones on one machine never see or affect each
rem  other.  Nothing is installed globally, and your PATH is left unchanged unless
rem  you opt into the global `localm` command near the end.
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
    echo  [!] uv is not installed. Install it first.
    echo      winget install astral-sh.uv
    echo.
    pause
    exit /b 1
)

rem ---- portable vs shared: where the Python runtime + downloads live --------
rem  Portable pulls uv's managed Python AND its wheel cache INTO this folder, so
rem  the clone is truly self-contained (delete it and nothing is left behind) at
rem  the cost of a per-clone re-download. Shared reuses uv's per-user Python +
rem  cache (one download, less disk) but lives in your user profile. The UV_* vars
rem  are set for THIS setup process only (not setx / not global), so they never
rem  touch any other uv project. --python-preference only-managed forces the
rem  contained download instead of reusing a system Python.
echo.
echo  Keep localm's Python runtime + downloads inside this folder?
echo    [1] Portable - everything in this folder (self-contained; re-downloads per clone)
echo    [2] Shared   - reuse uv's per-user Python + cache (faster; lives in your user profile)
set "STOREPICK="
set /p "STOREPICK=  Pick 1 or 2 [1]: "
if not defined STOREPICK set "STOREPICK=1"
set "CONTAINED=0"
set "PYPREF="
if "%STOREPICK%"=="1" (
    set "CONTAINED=1"
    set "UV_PYTHON_INSTALL_DIR=%CD%\.python"
    set "UV_CACHE_DIR=%CD%\.cache"
    set "PYPREF=--python-preference only-managed"
    echo  Portable: Python under .\.python and downloads under .\.cache
) else (
    echo  Shared: reusing uv's per-user Python + cache ^(outside this folder^).
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
    choice /c YN /n /m "  LocalM .venv found. Replace it? [y/N]: "
) else (
    choice /c YN /n /m "  Foreign .venv found. Replace it? [y/N]: "
)
rem choice sets errorlevel: 1=Y, 2=N. Test the higher index first.
if errorlevel 2 (
    echo  Keeping existing .venv.
    goto venv_done
)

:venv_create
echo.
echo  Creating .venv (Python %PYVER%) ...
uv venv --python %PYVER% %PYPREF% --clear .venv
if errorlevel 1 (
    echo  [!] Could not create the environment. Install Python %PYVER%.
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
if /i "%VENDOR%"=="nvidia" echo  NVIDIA note: [1] vulkan works out of the box; pick [3] cuda for peak performance.

rem ---- choose the llama.cpp backend (recommended pre-selected) ---------------
echo.
echo  Native inference runtime (llama.cpp) - press Enter to accept the recommendation:
echo    [1] %REC%   (recommended for your hardware)
echo    [2] vulkan     - any GPU (AMD/NVIDIA/Intel), no vendor toolkit
echo    [3] cuda       - NVIDIA, peak performance (fetches the CUDA runtime for you)
echo    [4] amd-rocm   - AMD RX 6000 (gfx103X), self-contained
echo    [5] cpu        - no GPU
echo    [6] I will build / provide my own (skip the download)
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
) else if "%TORCHSPEC%"=="-e .[gpu]" (
    rem  gfx103X (RX 6000): the bundled self-contained build carries torch + the HF
    rem  stack + the ROCm runtime; add audio (soundfile) for unified-audio models.
    echo  Installing PyTorch ^(AMD ROCm, gfx103X^) + transformers ...
    uv pip install -p .venv -e ".[gpu,audio]" || echo  [!] ROCm torch install failed. GGUF chat still works.
) else (
    echo  Installing PyTorch + transformers ...

    uv pip install -p .venv %TORCHSPEC% || echo  [!] torch install failed. GGUF chat still works.
    uv pip install -p .venv "transformers[kernels]~=5.12" "tokenizers==0.22.2" "accelerate>=1.0" "pillow>=10.0" "soundfile>=0.12" || echo  [!] transformers install failed. GGUF chat still works.
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
        if errorlevel 1 (
            echo  [!] Provisioning failed - run later: .venv\Scripts\localm setup-llama --from "!LLAMABUILD!"
            pause
            exit /b 1
        )
    ) else (
        echo  Skipped - provision later: .venv\Scripts\localm setup-llama --backend ^<vulkan^|cuda^|amd-rocm^|cpu^>
    )
) else (
    .venv\Scripts\localm setup-llama --backend %BACKEND%
    if errorlevel 1 (
        echo  [!] Provisioning failed - run later: .venv\Scripts\localm setup-llama --backend %BACKEND%
        pause
        exit /b 1
    )
)

rem ---- choose where data lives ----------------------------------------------
rem  Default is CONTAINED (.\home): there is NO silent ~/.localm fallback. Anyone
rem  who wants a shared / other location picks Custom, and it is recorded
rem  explicitly in localm-home.cfg (asked + recorded, never guessed).
echo.
echo  Where should localm keep its data (models, config, logs, images)?
echo    [1] Portable (.\home) - self-contained; delete this folder and it is all gone
echo    [2] Custom path       - a folder you choose (e.g. a shared models drive)
rem  set /p (type a number, then Enter), NOT `choice`: `choice` returns on a single
rem  keypress, so the user's habitual confirming Enter used to leak into the custom
rem  path's set /p below and be read as an empty path (SETUP-2). With set /p the
rem  Enter belongs to THIS prompt, so the path prompt starts clean.
set "DATAPICK="
set /p "DATAPICK=  Pick 1 or 2 [1]: "
if not defined DATAPICK set "DATAPICK=1"
rem DATADIR + DATACREATED feed the install manifest (DATACREATED=1 only when WE
rem made the dir, so uninstall --purge-data never removes a pre-existing folder).
set "DATADIR=%CD%\home"
set "DATACREATED=0"
if "%DATAPICK%"=="1" (
    if not exist "home" mkdir "home"
    if exist "localm-home.cfg" del "localm-home.cfg"
    set "DATADIR=%CD%\home"
    set "DATACREATED=1"
    echo  Data directory: %CD%\home
)
rem  Single-line `if ... call` into a goto/label subroutine (defined at the end):
rem  a `call` plus nested if/else INSIDE this `if (...)` block trips cmd.exe's
rem  parenthesis parser ("The syntax of the command is incorrect."), so keep the
rem  custom-path flow out of the block entirely.
if "%DATAPICK%"=="2" call :do_custom_home

rem ---- optional desktop shortcut ----------------------------------------------
echo.
echo  Create desktop shortcut?
echo    [1] Launcher
echo    [2] Web GUI directly
echo    [3] None
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

rem ---- optional: make `localm` runnable from any terminal --------------------
rem  Adds a small `localm` shim in .\bin and appends ONLY .\bin to your USER PATH
rem  via the registry (never setx, which truncates + corrupts PATH; never the venv
rem  Scripts dir, which would shadow your own python/pip). Fully reversible by the
rem  uninstaller. Default No - the CLI already works via .venv\Scripts\localm.
echo.
echo  Make 'localm' runnable from any terminal? (adds .\bin to your PATH)
set "GLOBALPICK="
set /p "GLOBALPICK=  [y/N]: "
if not defined GLOBALPICK set "GLOBALPICK=N"
set "PATHDIR="
set "CMDSHIM="
set "PATHMOD="
set "GCRC=99"
if /i "%GLOBALPICK%"=="y" .venv\Scripts\python -m localm.globalcmd install --root .
if /i "%GLOBALPICK%"=="y" set "GCRC=!errorlevel!"
rem  globalcmd exit code: 0 = installed + PATH modified; 20 = installed but PATH was
rem  already set (record the command but NOT --path-modified, so the manifest never
rem  claims a change it did not make); anything else = failed (record nothing). Flat
rem  single-line ifs, not a paren block, to dodge cmd.exe's nested-paren parser.
if "!GCRC!"=="0" set "PATHMOD=--path-modified"
if "!GCRC!"=="0" set "PATHDIR=%CD%\bin"
if "!GCRC!"=="0" set "CMDSHIM=%CD%\bin\localm.cmd"
if "!GCRC!"=="20" set "PATHDIR=%CD%\bin"
if "!GCRC!"=="20" set "CMDSHIM=%CD%\bin\localm.cmd"

rem ---- choose which plugins to enable ---------------------------------------
rem  `localm plugin setup` prints its own header (it states chat is always on),
rem  so this is just a section divider - do not repeat that line here.
echo.
echo  Optional features (plugins):
.venv\Scripts\localm plugin setup

rem ---- record what we installed (uninstall removes ONLY what we created) -----
set "CRD="
if "%DATACREATED%"=="1" set "CRD=--data-created"
set "RCFLAG="
set "PYDIR="
set "CACHEDIR="
if "%CONTAINED%"=="1" (
    set "RCFLAG=--runtime-contained"
    set "PYDIR=%CD%\.python"
    set "CACHEDIR=%CD%\.cache"
)
.venv\Scripts\python -m localm.install_manifest record --root . --venv "%CD%\.venv" --lib-dir "%CD%\runtime\localm_llama_runtime\lib" --data-dir "%DATADIR%" %CRD% --shortcut "%SCPATH%" %RCFLAG% --python-dir "%PYDIR%" --cache-dir "%CACHEDIR%" --path-dir "%PATHDIR%" --command-shim "%CMDSHIM%" %PATHMOD% >nul 2>nul
if errorlevel 1 echo  [!] Could not record the install manifest (uninstall will be conservative).

rem ---- done ------------------------------------------------------------------
echo.
echo  Done. Setup complete.
echo  Run localm-launcher.bat to start.
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
set /p "CUSTOMHOME=  Enter the data directory path, or leave blank for the portable .\home: "
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
echo  [!] No path given - using the portable .\home instead.
if exist "localm-home.cfg" del "localm-home.cfg"
if not exist "home" mkdir "home"
set "DATADIR=%CD%\home"
set "DATACREATED=1"
exit /b 0
