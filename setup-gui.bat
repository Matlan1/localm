@echo off
rem  LocaLM graphical setup - the windowed alternative to setup.bat.
rem
rem  Double-click this file. It asks nothing in a console: it starts itself again
rem  under a console host with no window (conhost --headless), shows a small
rem  "getting ready" window while it puts uv into .\.uv (and uv puts Python 3.12
rem  into .\.python and its downloads into .\.cache) when they are not there
rem  yet, then hands the whole install to installer\gui.py, which runs on that
rem  Python and needs no other dependency: tkinter ships with it. The window's
rem  first page asks Portable or Shared. Every message is shown in a message box.
rem
rem  setup.bat remains the console installer. This is the same install, asked
rem  for in a window.
rem
rem  LOCALM_SETUP_GUI_INPLACE=1 runs it in the calling console instead, with
rem  messages printed there. Exit codes: 0 done, or started again without a
rem  console; 1 the window could not be opened, or an uninstall did not finish;
rem  2 an uninstall kept something that was asked for; any other the window's
rem  own exit code when it could not run.
rem
rem  Delayed expansion stays off in this whole file, so a folder path containing
rem  "!" works. See test_setup_gui_runs_from_a_folder_with_a_bang_in_its_path.
setlocal EnableExtensions DisableDelayedExpansion
set "LOCALM_SETUP_GUI_SELF=%~f0"
cd /d "%~dp0"

rem ---- start again with no console window ------------------------------------
rem  USEBOX=1: messages go to message boxes and the "getting ready" window shows.
set "USEBOX=1"
if "%LOCALM_SETUP_GUI_INPLACE%"=="1" set "USEBOX=0"
if "%LOCALM_SETUP_GUI_HIDDEN%"=="1" goto start_setup
if "%USEBOX%"=="0" goto start_setup
if not exist "%SystemRoot%\System32\conhost.exe" goto start_setup
set "LOCALM_SETUP_GUI_HIDDEN=1"
start "" "%SystemRoot%\System32\conhost.exe" --headless "%ComSpec%" /d /c ""%%LOCALM_SETUP_GUI_SELF%%""
if errorlevel 1 goto start_setup
exit /b 0

:start_setup
set "LOCALM_SETUP_GUI_HIDDEN="
set "LOCALM_SETUP_GUI_INPLACE="
set "LOCALM_SETUP_GUI_SELF="
for %%n in (1 2 3 4 5) do set "LOCALM_MSG%%n="
echo.
echo   LocaLM graphical setup
echo.

rem ---- the uv that opens the window --------------------------------------------
rem  The window reads LOCALM_GUI_STORE (the choice its first page starts on:
rem  portable or shared), LOCALM_GUI_UV (the uv in use), LOCALM_GUI_BOOTSTRAP
rem  (what this run put into the folder to open the window: uv, python, cache),
rem  LOCALM_GUI_SPLASH (the file whose removal closes the "getting ready"
rem  window) and, once Shared was chosen, LOCALM_GUI_REPOINTED=1 and
rem  LOCALM_GUI_UV_INSTALLED=1 when the window installed that per-user uv.
rem  The folder's own .\.uv\uv.exe is used when it is there; an earlier Shared
rem  install (no .\.uv, an install record) reuses the per-user uv; anything else
rem  installs uv into .\.uv without asking.
set "UVEXE="
set "SPLASHN=0"
set "LOCALM_GUI_BOOTSTRAP="
set "LOCALM_GUI_REPOINTED="
set "LOCALM_GUI_UV_INSTALLED="
set "LOCALM_GUI_SPLASH="
set "LOCALM_SPLASH_TEXT=Getting the LocaLM setup window ready. The first time, this downloads uv and Python 3.12 (about 50 MB) into this folder; it opens by itself."
call :splash_start
set "LOCALM_GUI_STORE=portable"
if exist ".uv\uv.exe" goto portable_have_uv
set "HAVEINSTALL="
if exist ".localm-install.json" set "HAVEINSTALL=1"
if exist ".venv\.localm-venv" set "HAVEINSTALL=1"
if not defined HAVEINSTALL goto install_uv
call :find_shared_uv
if not defined UVEXE goto install_uv
set "LOCALM_GUI_STORE=shared"
goto have_uv

:install_uv
echo  Installing uv into .\.uv ...
rem  UV_UNMANAGED_INSTALL keeps it in .\.uv without adding that folder to the
rem  user PATH or writing an install receipt under %LOCALAPPDATA%\uv. The
rem  installer script and everything Astral's installer downloads stay in
rem  .\.localm-setup-tmp, which is removed again when it is empty.
set "UV_INSTALL_DIR=%CD%\.uv"
set "UV_UNMANAGED_INSTALL=%CD%\.uv"
set "STMP=.localm-setup-tmp"
if not exist "%STMP%" mkdir "%STMP%" >nul 2>nul
set "UV_INSTALLER_VERSION=0.13.8"
set "UV_INSTALLER_SHA256=200a377ed0f8e719b1c9cbe849c736e7712ee684212f61dc9565d747ac163db0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; $f=$null; $t=$null; $w=$null; try { $d=$env:STMP; if ($d) { $d=[IO.Path]::GetFullPath($d); if (-not [IO.Directory]::Exists($d)) { exit 63 }; if ($d.IndexOfAny([char[]]'[]') -ge 0) { $w=Join-Path ([IO.Path]::GetTempPath()) ('localm-uv-'+[guid]::NewGuid()); try { $null=[IO.Directory]::CreateDirectory($w) } catch { $w=$null; exit 63 }; $d=$w } } else { $d=[IO.Path]::GetTempPath() }; $f=Join-Path $d ('uv-installer-'+[guid]::NewGuid()+'.ps1'); try { Invoke-WebRequest -UseBasicParsing -Uri ('https://github.com/astral-sh/uv/releases/download/'+$env:UV_INSTALLER_VERSION+'/uv-installer.ps1') -OutFile $f -ErrorAction Stop } catch { exit 61 }; $h=$null; $fs=$null; try { $fs=[IO.File]::Open($f,'Open','Read','Read'); $h=[BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($fs)).Replace('-','') } catch { }; if ($h -ne $env:UV_INSTALLER_SHA256) { exit 62 }; $t=Join-Path $d ('uv-tmp-'+[guid]::NewGuid()); try { $null=[IO.Directory]::CreateDirectory($t) } catch { $t=$null; exit 63 }; $env:TEMP=$t; $env:TMP=$t; $env:PSModulePath=$null; & powershell -NoProfile -ExecutionPolicy Bypass -File $f; exit $LASTEXITCODE } finally { if ($fs) { $fs.Dispose() }; if ($f) { Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue }; if ($t) { Remove-Item -LiteralPath $t -Recurse -Force -ErrorAction SilentlyContinue }; if ($w) { Remove-Item -LiteralPath $w -Recurse -Force -ErrorAction SilentlyContinue } }"
set "UVRC=%errorlevel%"
set "UV_INSTALL_DIR="
set "UV_UNMANAGED_INSTALL="
rd "%STMP%" >nul 2>nul
set "LOCALM_MSG1="
if "%UVRC%"=="61" set "LOCALM_MSG1=[!] Could not download the uv %UV_INSTALLER_VERSION% installer from github.com."
if "%UVRC%"=="61" set "LOCALM_MSG3=Check that this computer can reach github.com (a proxy or firewall may block it), or install uv yourself (winget install astral-sh.uv), then run this again. setup.bat is the console installer."
if "%UVRC%"=="62" set "LOCALM_MSG1=[!] The downloaded uv installer did not match its expected checksum, or could not be read to check it, and was not run."
if "%UVRC%"=="63" set "LOCALM_MSG1=[!] Could not create a temporary folder in %CD%\%STMP% for the uv installer."
if defined LOCALM_MSG1 goto uv_refused
if not exist ".uv\uv.exe" goto uv_not_callable
set "LOCALM_GUI_BOOTSTRAP=uv"

:portable_have_uv
set "UVEXE=%CD%\.uv\uv.exe"

:have_uv
set "LOCALM_GUI_UV=%UVEXE%"
echo  Using "%UVEXE%"
goto open_window

:uv_not_callable
set "LOCALM_MSG1=[!] uv still is not callable, so the graphical setup cannot start."
if "%LOCALM_GUI_STORE%"=="portable" set "LOCALM_MSG2=Looked for %CD%\.uv\uv.exe (the installer did not leave one there)."
if not "%LOCALM_GUI_STORE%"=="portable" set "LOCALM_MSG2=Looked on PATH and in %USERPROFILE%\.local\bin for the per-user uv."
set "LOCALM_MSG3=Open a NEW terminal and run setup.bat instead."
call :notify Error
exit /b 1

:uv_refused
set "LOCALM_MSG2=uv was not installed, so the graphical setup cannot start."
if not defined LOCALM_MSG3 set "LOCALM_MSG3=Install uv yourself (winget install astral-sh.uv), then run this again. setup.bat is the console installer."
call :notify Error
exit /b 1

:open_window
rem ---- open the installer window ---------------------------------------------
rem  --no-project so uv never tries to resolve this repo as its own project, and
rem  an explicit --python so the interpreter is the managed 3.12 the install
rem  targets rather than whatever else is on the machine. Portable keeps the
rem  interpreter this window runs on inside the folder, so the install reuses it
rem  rather than downloading a second copy.
echo   Opening the setup window ...
if not "%LOCALM_GUI_STORE%"=="portable" goto window_env_done
if not exist ".python\" set "LOCALM_GUI_BOOTSTRAP=%LOCALM_GUI_BOOTSTRAP% python"
if not exist ".cache\" set "LOCALM_GUI_BOOTSTRAP=%LOCALM_GUI_BOOTSTRAP% cache"
set "UV_PYTHON_INSTALL_DIR=%CD%\.python"
set "UV_CACHE_DIR=%CD%\.cache"
:window_env_done
set "UV_SYSTEM_CERTS=1"
"%UVEXE%" run --no-project --isolated --python 3.12 python "installer\gui.py"
set "RC=%errorlevel%"
call :splash_stop

rem  46 and 47: Shared was chosen on the first page and the per-user uv is in
rem  place (47: the window installed it); the window opens again on it. 42 and
rem  43: an uninstall finished in the window; the Python runtime it ran on is
rem  removed now that it has closed. 43 and 44: something asked for was kept.
rem  45: the uninstall did not finish.
if "%RC%"=="47" set "LOCALM_GUI_UV_INSTALLED=1"
if "%RC%"=="46" goto use_shared
if "%RC%"=="47" goto use_shared
set "PARTIAL="
if "%RC%"=="42" goto finish_uninstall
if "%RC%"=="43" set "PARTIAL=1"
if "%RC%"=="43" goto finish_uninstall
if "%RC%"=="44" goto uninstall_partial
if "%RC%"=="45" goto uninstall_failed
if "%RC%"=="0" exit /b 0
:window_failed
set "LOCALM_MSG1=[!] The setup window could not run (exit %RC%)."
set "LOCALM_MSG2=It runs on Python 3.12, which uv downloads the first time: check the internet connection, or use the console installer instead:  setup.bat"
call :notify Error
exit /b %RC%

:use_shared
rem ---- Shared: drop what this run put in the folder, reopen on the per-user uv -
if defined LOCALM_GUI_REPOINTED goto window_failed
set "LOCALM_GUI_REPOINTED=1"
set "UV_PYTHON_INSTALL_DIR="
set "UV_CACHE_DIR="
call :remove_bootstrap
set "LOCALM_GUI_BOOTSTRAP="
call :find_shared_uv
if defined UVEXE goto shared_found
rem  Every directory the per-user install may have used, in setup.bat's own order.
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%HOMEDRIVE%%HOMEPATH%\.local\bin;%PATH%"
if defined XDG_BIN_HOME set "PATH=%XDG_BIN_HOME%;%PATH%"
call :find_shared_uv
if not defined UVEXE goto uv_not_callable
:shared_found
set "LOCALM_GUI_STORE=shared"
set "LOCALM_GUI_UV=%UVEXE%"
echo  Shared: using "%UVEXE%"
if not defined LEFTOVER goto shared_reopen
set "LOCALM_MSG1=[!] Could not remove%LEFTOVER% from this folder."
set "LOCALM_MSG2=Setup put them there to open its window; the Shared choice does not use them. Close any LocaLM window and delete them by hand."
call :notify Warning
:shared_reopen
set "LOCALM_SPLASH_TEXT=Opening the LocaLM setup window on the shared uv. The first time, uv downloads Python 3.12 (about 30 MB) to its per-user location."
call :splash_start
goto open_window

:finish_uninstall
echo   Removing the last LocaLM folders ...
call ".\setup.bat" finish-uninstall
if errorlevel 1 goto finish_uninstall_left
if defined PARTIAL goto uninstall_partial
set "LOCALM_MSG1=LocaLM was uninstalled."
call :notify Information
exit /b 0
:uninstall_partial
set "LOCALM_MSG1=[!] LocaLM was removed, but some things you asked to delete were not"
set "LOCALM_MSG2=deleted - the setup window listed them under REFUSED."
call :notify Warning
exit /b 2
:uninstall_failed
set "LOCALM_MSG1=[!] The uninstall did not finish - the setup window listed why. Close"
set "LOCALM_MSG2=any LocaLM window and run the setup again."
call :notify Error
exit /b 1
:finish_uninstall_left
set "LOCALM_MSG1=[!] Some LocaLM folders could not be removed - close any LocaLM window"
set "LOCALM_MSG2=and delete them by hand, or run  setup.bat finish-uninstall  again."
call :notify Error
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
rem  :remove_bootstrap - removes each folder named in LOCALM_GUI_BOOTSTRAP (uv,
rem  python, cache: .\.uv .\.python .\.cache), retrying while a closing program
rem  still holds files. LEFTOVER lists the ones that could not be removed.
rem ===========================================================================
:remove_bootstrap
set "LEFTOVER="
for %%n in (%LOCALM_GUI_BOOTSTRAP%) do call :remove_tooling "%%n"
goto :eof

:remove_tooling
set "DNAME=.%~1"
if /i not "%DNAME%"==".uv" if /i not "%DNAME%"==".python" if /i not "%DNAME%"==".cache" goto :eof
if not exist "%DNAME%\" goto :eof
set "DTRIES=0"
:remove_tooling_retry
attrib -R "%DNAME%\*" /S /D >nul 2>nul
rmdir /s /q "%DNAME%" >nul 2>nul
if not exist "%DNAME%\" goto :eof
set /a DTRIES+=1
if %DTRIES% lss 5 ping -n 2 127.0.0.1 >nul
if %DTRIES% lss 5 goto remove_tooling_retry
set "LEFTOVER=%LEFTOVER% .\%DNAME%"
goto :eof

rem ===========================================================================
rem  :notify Error^|Warning^|Information - shows LOCALM_MSG1..LOCALM_MSG5 as one
rem  message: a message box when started by a double-click (USEBOX=1), the
rem  console otherwise. Closes the "getting ready" window first and clears the
rem  message variables after.
rem ===========================================================================
:notify
call :splash_stop
if not "%USEBOX%"=="1" goto notify_console
set "LOCALM_MSG_ICON=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; $m=@(1..5 | ForEach-Object { [Environment]::GetEnvironmentVariable('LOCALM_MSG'+$_) } | Where-Object { $_ }) -join [Environment]::NewLine; $o=New-Object System.Windows.Forms.Form; $o.TopMost=$true; [void][System.Windows.Forms.MessageBox]::Show($o, $m, 'LocaLM Setup', 'OK', $env:LOCALM_MSG_ICON); $o.Dispose()"
goto notify_clear
:notify_console
setlocal EnableDelayedExpansion
echo.
for %%n in (1 2 3 4 5) do if defined LOCALM_MSG%%n echo   !LOCALM_MSG%%n!
endlocal
:notify_clear
for %%n in (1 2 3 4 5) do set "LOCALM_MSG%%n="
set "LOCALM_MSG_ICON="
goto :eof

rem ===========================================================================
rem  :splash_start - with USEBOX=1, shows LOCALM_SPLASH_TEXT in a small window
rem  that closes itself once the file named by LOCALM_GUI_SPLASH is gone (the
rem  setup window removes it when it opens; :splash_stop when it has closed),
rem  and after 30 minutes at the latest.
rem ===========================================================================
:splash_start
if not "%USEBOX%"=="1" goto :eof
set /a SPLASHN+=1
set "LOCALM_GUI_SPLASH=%CD%\.localm-setup-starting-%SPLASHN%"
type nul > "%LOCALM_GUI_SPLASH%" 2>nul
if not exist "%LOCALM_GUI_SPLASH%" set "LOCALM_GUI_SPLASH="
if not defined LOCALM_GUI_SPLASH goto :eof
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; $f=New-Object System.Windows.Forms.Form; $f.Text='LocaLM Setup'; $f.FormBorderStyle='FixedDialog'; $f.MaximizeBox=$false; $f.MinimizeBox=$false; $f.StartPosition='CenterScreen'; $f.Width=480; $f.Height=170; $f.TopMost=$true; $l=New-Object System.Windows.Forms.Label; $l.Text=$env:LOCALM_SPLASH_TEXT; $l.SetBounds(16,14,436,62); $f.Controls.Add($l); $p=New-Object System.Windows.Forms.ProgressBar; $p.Style='Marquee'; $p.SetBounds(16,84,436,18); $f.Controls.Add($p); $s=[DateTime]::UtcNow; $k=$env:LOCALM_GUI_SPLASH; $t=New-Object System.Windows.Forms.Timer; $t.Interval=400; $t.Add_Tick({ if (-not [IO.File]::Exists($k) -or ([DateTime]::UtcNow - $s).TotalMinutes -gt 30) { $t.Stop(); $f.Close() } }); $t.Start(); [void]$f.ShowDialog()"
goto :eof

:splash_stop
if defined LOCALM_GUI_SPLASH del "%LOCALM_GUI_SPLASH%" >nul 2>nul
set "LOCALM_GUI_SPLASH="
goto :eof
