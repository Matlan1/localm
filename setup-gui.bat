@echo off
rem  LocaLM graphical setup - the windowed alternative to setup.bat.
rem
rem  Double-click this file. It puts localm's own pinned uv in place (the only
rem  part that cannot be graphical, because something has to provide a Python
rem  before a window can exist), then hands the whole install over to
rem  installer\gui.py, which runs on uv's managed CPython and needs no other
rem  dependency: tkinter ships with it.
rem
rem  setup.bat remains the console installer. This is the same install, asked
rem  for in a window.
rem
rem  Delayed expansion stays off in this whole file, so a folder path containing
rem  "!" works. See test_setup_gui_runs_from_a_folder_with_a_bang_in_its_path.
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

rem ---- only what localm brings ------------------------------------------------
rem  Clears every inherited uv and pip setting and the Python variables that
rem  change which code or config a Python loads, and LOCALM_HOME (the data
rem  folder is the one this setup records), then sets the ones setup uses.
rem  System programs run by their full path under %SystemRoot%. TEMP and TMP
rem  point at .\.localm-setup-tmp for the whole run, the window included.
set "ComSpec=%SystemRoot%\System32\cmd.exe"
set "PSEXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
for /f "delims==" %%v in ('set UV_ 2^>nul') do set "%%v="
for /f "delims==" %%v in ('set PIP_ 2^>nul') do set "%%v="
for %%v in (PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX LOCALM_HOME) do set "%%v="
set "UV_NO_CONFIG=1"
set "PIP_CONFIG_FILE=NUL"
set "PYTHONNOUSERSITE=1"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_INSTALLER_VERSION=0.13.6"
set "UV_INSTALLER_SHA256=d00580051699bee58a2e2d9b925f45788b35f7e0fff7966fd5bedfc3c547dc7d"
set "STMP=.localm-setup-tmp"
if not exist "%STMP%" mkdir "%STMP%" >nul 2>nul
set "TEMP=%CD%\.localm-setup-tmp"
set "TMP=%CD%\.localm-setup-tmp"

echo.
echo   LocaLM graphical setup
echo.

rem ---- portable vs shared: where localm's own Python tooling lives ----------
rem  The question setup.bat asks, asked once, here. The window reads the answer
rem  from LOCALM_GUI_STORE (portable or shared) and LOCALM_GUI_UV_INSTALLED=1
rem  when this run installed uv into the Shared folder; it finds the store's uv
rem  itself. Portable keeps uv, Python and downloads in .\.uv .\.python .\.cache;
rem  Shared in %LOCALAPPDATA%\localm\shared\uv, \python and \cache.
echo  Keep localm's Python tooling ^(uv itself, its runtime, and downloads^) inside this folder?
echo    [1] Portable - everything in this folder (self-contained; re-downloads per clone)
echo    [2] Shared   - one localm folder in your user profile, reused by every Shared install (faster)
set "STOREPICK="
rem SETUP-Q: tooling-store
set /p "STOREPICK=  Pick 1 or 2 [1]: "
if not defined STOREPICK set "STOREPICK=1"
set "LOCALM_GUI_UV_INSTALLED="
if "%STOREPICK%"=="1" goto portable_store
if not defined LOCALAPPDATA goto store_unavailable
set "LOCALM_GUI_STORE=shared"
set "UVDIR=%LOCALAPPDATA%\localm\shared\uv"
set "PYDIR=%LOCALAPPDATA%\localm\shared\python"
set "CACHEDIR=%LOCALAPPDATA%\localm\shared\cache"
echo  Shared: uv, Python, and downloads in "%LOCALAPPDATA%\localm\shared"
goto store_chosen
:portable_store
set "LOCALM_GUI_STORE=portable"
set "UVDIR=%CD%\.uv"
set "PYDIR=%CD%\.python"
set "CACHEDIR=%CD%\.cache"
echo  Portable: uv, Python, and downloads all under this folder
:store_chosen
set "UVEXE=%UVDIR%\uv.exe"
rem  uv keeps its credentials store (and its lock file) in UVDIR\credentials,
rem  and puts no Python shim in a user folder and no Python registry entry.
set "UV_CREDENTIALS_DIR=%UVDIR%\credentials"
set "UV_PYTHON_INSTALL_BIN=0"
set "UV_PYTHON_INSTALL_REGISTRY=0"

rem ---- uv: localm's own pinned copy --------------------------------------------
rem  A uv already in UVDIR is used only when it reports the pinned version;
rem  otherwise Astral's installer (pinned by version and checksum) puts that
rem  version there. UV_UNMANAGED_INSTALL keeps the installer from changing the
rem  user PATH or writing a receipt. No other uv is ever looked for.
call :uv_version
if "%UVOK%"=="1" goto have_uv
echo.
echo  Installing uv %UV_INSTALLER_VERSION% ...
if "%LOCALM_GUI_STORE%"=="portable" echo  Portable: installing uv itself under .\.uv
set "UV_INSTALL_DIR=%UVDIR%"
set "UV_UNMANAGED_INSTALL=%UVDIR%"
rem  The installer's own Expand-Archive/Copy-Item/Remove-Item calls run as .NET
rem  literal-path calls, so a [ or ] in this folder's path works. See
rem  test_setup_gui_bat_installs_uv_from_an_awkward_folder.
"%PSEXE%" -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; $f=$null; $t=$null; $fs=$null; try { $d=$env:STMP; if (-not $d) { exit 63 }; $d=[IO.Path]::GetFullPath($d); if (-not [IO.Directory]::Exists($d)) { exit 63 }; $f=[IO.Path]::Combine($d, ('uv-installer-'+[guid]::NewGuid()+'.ps1')); try { (New-Object Net.WebClient).DownloadFile('https://github.com/astral-sh/uv/releases/download/'+$env:UV_INSTALLER_VERSION+'/uv-installer.ps1', $f) } catch { exit 61 }; $h=$null; try { $fs=[IO.File]::Open($f,'Open','Read','Read'); $h=[BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($fs)).Replace('-','') } catch { }; if ($h -ne $env:UV_INSTALLER_SHA256) { exit 62 }; $t=[IO.Path]::Combine($d, ('uv-tmp-'+[guid]::NewGuid())); try { $null=[IO.Directory]::CreateDirectory($t) } catch { $t=$null; exit 63 }; $env:TEMP=$t; $env:TMP=$t; $env:PSModulePath=$null; $env:LOCALM_UV_INSTALLER=$f; $c='function Expand-Archive { param([string]$Path, [string]$DestinationPath) Add-Type -AssemblyName System.IO.Compression.FileSystem; [IO.Compression.ZipFile]::ExtractToDirectory($Path, $DestinationPath) }; function Copy-Item { [CmdletBinding()] param([Parameter(Position=0)][string]$Path, [string]$Destination) [IO.File]::Copy($Path, [IO.Path]::Combine($Destination, [IO.Path]::GetFileName($Path)), $true) }; function Remove-Item { [CmdletBinding()] param([Parameter(Position=0)][string]$Path, [switch]$Recurse, [switch]$Force) [IO.File]::Delete($Path) }; & ([ScriptBlock]::Create([IO.File]::ReadAllText($env:LOCALM_UV_INSTALLER))); exit $LASTEXITCODE'; & $env:PSEXE -NoProfile -ExecutionPolicy Bypass -Command $c; exit $LASTEXITCODE } finally { if ($fs) { $fs.Dispose() }; if ($f) { Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue }; if ($t) { Remove-Item -LiteralPath $t -Recurse -Force -ErrorAction SilentlyContinue } }"
set "UVRC=%errorlevel%"
set "UV_INSTALL_DIR="
set "UV_UNMANAGED_INSTALL="
if "%UVRC%"=="61" echo   [!] Could not download the uv %UV_INSTALLER_VERSION% installer.
if "%UVRC%"=="62" echo   [!] The downloaded uv installer did not match its expected checksum, or could not be read to check it, and was not run.
if "%UVRC%"=="63" echo   [!] Could not create a temporary folder in %STMP% for the uv installer.
if "%UVRC%"=="61" goto uv_refused
if "%UVRC%"=="62" goto uv_refused
if "%UVRC%"=="63" goto uv_refused
if "%LOCALM_GUI_STORE%"=="shared" set "LOCALM_GUI_UV_INSTALLED=1"
call :uv_version
if not "%UVOK%"=="1" goto uv_not_callable

:have_uv
echo  Using "%UVEXE%"
goto open_window

:uv_not_callable
echo.
echo   [!] uv still is not callable, so the graphical setup cannot start.
echo       Looked for "%UVEXE%" reporting uv %UV_INSTALLER_VERSION%.
echo       Check the internet connection, then run this again.
rd "%STMP%" >nul 2>nul
pause
exit /b 1

:uv_refused
echo.
echo   uv was not installed, so the graphical setup cannot start.
echo       Check the internet connection, then run this again.
rd "%STMP%" >nul 2>nul
pause
exit /b 1

:store_unavailable
echo.
echo   [!] LOCALAPPDATA is not set, so the Shared folder in your user profile cannot be found.
echo       Run this again and pick [1] Portable.
rd "%STMP%" >nul 2>nul
pause
exit /b 1

:open_window
rem ---- open the installer window ---------------------------------------------
rem  --no-project so uv never tries to resolve this repo as its own project, and
rem  --python 3.12 with --python-preference only-managed so the window runs on
rem  the managed 3.12 in the chosen store.
echo   Opening the setup window ...
set "UV_PYTHON_INSTALL_DIR=%PYDIR%"
set "UV_CACHE_DIR=%CACHEDIR%"
set "UV_SYSTEM_CERTS=1"
"%UVEXE%" run --no-project --isolated --python 3.12 --python-preference only-managed "installer\gui.py"
set "RC=%errorlevel%"

rem  42 and 43: an uninstall finished in the window; the Python runtime it ran
rem  on is removed now that it has closed. 43 and 44: something asked for was
rem  kept. 45: the uninstall did not finish.
set "PARTIAL="
if "%RC%"=="0" rd /s /q "%STMP%" >nul 2>nul
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
if exist "%STMP%" rd /s /q "%STMP%" >nul 2>nul
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
rem  :uv_version - sets UVOK=1 when UVEXE exists and its --version reports
rem  UV_INSTALLER_VERSION, else UVOK=0. UVVER holds the version it reported.
rem ===========================================================================
:uv_version
set "UVOK=0"
set "UVVER="
if not exist "%UVEXE%" goto :eof
"%UVEXE%" --version > "%STMP%\localm_uvver.txt" 2>nul
for /f "usebackq tokens=1,2" %%a in ("%STMP%\localm_uvver.txt") do if /i "%%a"=="uv" if not defined UVVER set "UVVER=%%b"
del "%STMP%\localm_uvver.txt" >nul 2>nul
if "%UVVER%"=="%UV_INSTALLER_VERSION%" set "UVOK=1"
goto :eof
