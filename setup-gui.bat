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
rem  The quoted path travels in LOCALM_SETUP_GUI_CMD, which the started cmd
rem  expands with delayed expansion (/v:on) after it has parsed its command
rem  line; the console host re-quotes and expands what is on that line. This
rem  file turns delayed expansion off again on its first line. See
rem  test_the_real_start_without_a_console_runs_setup_from_an_awkward_folder.
setlocal EnableDelayedExpansion
set "LOCALM_SETUP_GUI_CMD="!LOCALM_SETUP_GUI_SELF!""
start "" "%SystemRoot%\System32\conhost.exe" --headless "%ComSpec%" /d /v:on /c ^^!LOCALM_SETUP_GUI_CMD^^!
set "STARTRC=!errorlevel!"
endlocal & set "STARTRC=%STARTRC%"
if not "%STARTRC%"=="0" goto start_setup
exit /b 0

:start_setup
set "LOCALM_SETUP_GUI_HIDDEN="
set "LOCALM_SETUP_GUI_INPLACE="
set "LOCALM_SETUP_GUI_SELF="
set "LOCALM_SETUP_GUI_CMD="
for %%n in (1 2 3 4 5) do set "LOCALM_MSG%%n="
echo.
echo   LocaLM graphical setup
echo.

rem ---- the uv that opens the window --------------------------------------------
rem  The window reads LOCALM_GUI_STORE (the choice its first page starts on:
rem  portable or shared), LOCALM_GUI_UV (the uv in use), LOCALM_GUI_BOOTSTRAP
rem  (what this run put into the folder to open the window: uv, python, cache),
rem  LOCALM_GUI_SPLASH (the file whose removal closes the "getting ready"
rem  window), LOCALM_GUI_LOG (the file the window's output goes to when there is
rem  no console) and, once Shared was chosen, LOCALM_GUI_REPOINTED=1 and
rem  LOCALM_GUI_UV_INSTALLED=1 when the window installed that per-user uv.
rem  The folder's own .\.uv\uv.exe is used when it is there; an earlier Shared
rem  install (no .\.uv, an install record) reuses the per-user uv; anything else
rem  installs uv into .\.uv without asking.
rem  .localm-gui-bootstrap names what a run of this file put into the folder and
rem  no install has taken over yet (the window removes it when a Portable
rem  install starts), so a later Shared choice removes those folders too.
set "UVEXE="
set "UVRC="
set "SPLASHN=0"
set "LOCALM_GUI_BOOTSTRAP="
set "B_UV="
set "B_PYTHON="
set "B_CACHE="
set "LOCALM_GUI_LOG="
if "%USEBOX%"=="1" set "LOCALM_GUI_LOG=%CD%\.localm-setup-gui.log"
set "PREV="
if exist ".localm-gui-bootstrap" for /f "usebackq delims=" %%l in (".localm-gui-bootstrap") do set "PREV=%%l"
for %%n in (%PREV%) do if /i "%%n"=="uv" if exist ".uv\" set "B_UV=1"
for %%n in (%PREV%) do if /i "%%n"=="python" if exist ".python\" set "B_PYTHON=1"
for %%n in (%PREV%) do if /i "%%n"=="cache" if exist ".cache\" set "B_CACHE=1"
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
set "UV_INSTALLER_VERSION=0.13.9"
set "UV_INSTALLER_SHA256=43b64ed13b8abfcdb3bcfc6ad6a1600dec0849d9e656c18f8ad81305b5d6d4a2"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; $f=$null; $t=$null; $w=$null; try { $d=$env:STMP; if ($d) { $d=[IO.Path]::GetFullPath($d); if (-not [IO.Directory]::Exists($d)) { exit 63 }; if ($d.IndexOfAny([char[]]'[]') -ge 0) { $w=Join-Path ([IO.Path]::GetTempPath()) ('localm-uv-'+[guid]::NewGuid()); try { $null=[IO.Directory]::CreateDirectory($w) } catch { $w=$null; exit 63 }; $d=$w } } else { $d=[IO.Path]::GetTempPath() }; $f=Join-Path $d ('uv-installer-'+[guid]::NewGuid()+'.ps1'); try { Invoke-WebRequest -UseBasicParsing -Uri ('https://github.com/astral-sh/uv/releases/download/'+$env:UV_INSTALLER_VERSION+'/uv-installer.ps1') -OutFile $f -ErrorAction Stop } catch { exit 61 }; $h=$null; $fs=$null; try { $fs=[IO.File]::Open($f,'Open','Read','Read'); $h=[BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($fs)).Replace('-','') } catch { }; if ($h -ne $env:UV_INSTALLER_SHA256) { exit 62 }; $t=Join-Path $d ('uv-tmp-'+[guid]::NewGuid()); try { $null=[IO.Directory]::CreateDirectory($t) } catch { $t=$null; exit 63 }; $env:TEMP=$t; $env:TMP=$t; $env:PSModulePath=$null; & powershell -NoProfile -ExecutionPolicy Bypass -File $f; exit $LASTEXITCODE } finally { if ($fs) { $fs.Dispose() }; if ($f) { Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue }; if ($t) { Remove-Item -LiteralPath $t -Recurse -Force -ErrorAction SilentlyContinue }; if ($w) { Remove-Item -LiteralPath $w -Recurse -Force -ErrorAction SilentlyContinue } }"
set "UVRC=%errorlevel%"
set "UV_INSTALL_DIR="
set "UV_UNMANAGED_INSTALL="
rd "%STMP%" >nul 2>nul
if "%UVRC%"=="61" echo   [!] Could not download the uv %UV_INSTALLER_VERSION% installer.
if "%UVRC%"=="62" echo   [!] The downloaded uv installer did not match its expected checksum, or could not be read to check it, and was not run.
if "%UVRC%"=="63" echo   [!] Could not create a temporary folder in %STMP% for the uv installer.
if "%UVRC%"=="61" goto uv_refused
if "%UVRC%"=="62" goto uv_refused
if "%UVRC%"=="63" goto uv_refused
if not exist ".uv\uv.exe" goto uv_not_callable
set "B_UV=1"

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
if defined UVRC if not "%UVRC%"=="0" set "LOCALM_MSG4=The uv installer exited %UVRC%."
call :notify Error
exit /b 1

:uv_refused
rem  With USEBOX=1 the reason echoed above also goes into the message box.
if not "%USEBOX%"=="1" goto uv_refused_remedy
if "%UVRC%"=="61" set "LOCALM_MSG1=[!] Could not download the uv %UV_INSTALLER_VERSION% installer from github.com."
if "%UVRC%"=="62" set "LOCALM_MSG1=[!] The downloaded uv installer did not match its expected checksum, or could not be read to check it, and was not run."
if "%UVRC%"=="63" set "LOCALM_MSG1=[!] Could not create a temporary folder in %CD%\%STMP% for the uv installer."
:uv_refused_remedy
set "LOCALM_MSG2=uv was not installed, so the graphical setup cannot start."
set "LOCALM_MSG3=Install uv yourself (winget install astral-sh.uv), then run this again. setup.bat is the console installer."
if "%UVRC%"=="61" set "LOCALM_MSG3=Check that this computer can reach github.com (a proxy or firewall may block it), or install uv yourself (winget install astral-sh.uv), then run this again. setup.bat is the console installer."
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
if not exist ".python\" set "B_PYTHON=1"
if not exist ".cache\" set "B_CACHE=1"
set "UV_PYTHON_INSTALL_DIR=%CD%\.python"
set "UV_CACHE_DIR=%CD%\.cache"
:window_env_done
call :write_bootstrap
set "UV_SYSTEM_CERTS=1"
if defined LOCALM_GUI_LOG call :run_window > "%LOCALM_GUI_LOG%" 2>&1
if not defined LOCALM_GUI_LOG call :run_window
set "RC=%errorlevel%"
call :splash_stop
if "%RC%"=="0" call :drop_log
if "%RC%"=="46" call :drop_log
if "%RC%"=="47" call :drop_log

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
if defined LOCALM_GUI_LOG if exist "%LOCALM_GUI_LOG%" set "LOCALM_MSG3=What it printed is in %LOCALM_GUI_LOG%; its last lines:"
if defined LOCALM_MSG3 set "LOCALM_MSG_TAIL=%LOCALM_GUI_LOG%"
call :notify Error
exit /b %RC%

:use_shared
rem ---- Shared: drop what this run put in the folder, reopen on the per-user uv -
if defined LOCALM_GUI_REPOINTED goto window_failed
set "LOCALM_GUI_REPOINTED=1"
set "LOCALM_GUI_STORE=shared"
set "LOCALM_SPLASH_TEXT=Opening the LocaLM setup window on the shared uv. The first time, uv downloads Python 3.12 (about 30 MB) to its per-user location."
call :splash_start
set "UV_PYTHON_INSTALL_DIR="
set "UV_CACHE_DIR="
call :find_shared_uv
if defined UVEXE goto shared_found
rem  Every directory the per-user install may have used, in setup.bat's own order.
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%HOMEDRIVE%%HOMEPATH%\.local\bin;%PATH%"
if defined XDG_BIN_HOME set "PATH=%XDG_BIN_HOME%;%PATH%"
call :find_shared_uv
if not defined UVEXE goto uv_not_callable
:shared_found
set "LOCALM_GUI_UV=%UVEXE%"
echo  Shared: using "%UVEXE%"
call :remove_bootstrap
call :write_bootstrap
if not defined LEFTOVER goto shared_reopen
set "LOCALM_MSG1=[!] Could not remove%LEFTOVER% from this folder."
set "LOCALM_MSG2=Setup put them there to open its window; the Shared choice does not use them. Close any LocaLM window and delete them by hand."
call :notify Warning
:shared_reopen
if not defined LOCALM_GUI_SPLASH call :splash_start
goto open_window

:finish_uninstall
echo   Removing the last LocaLM folders ...
del ".localm-gui-bootstrap" >nul 2>nul
if defined LOCALM_GUI_LOG call ".\setup.bat" finish-uninstall > "%LOCALM_GUI_LOG%" 2>&1
if not defined LOCALM_GUI_LOG call ".\setup.bat" finish-uninstall
if errorlevel 1 goto finish_uninstall_left
call :drop_log
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
if defined LOCALM_GUI_LOG if exist "%LOCALM_GUI_LOG%" set "LOCALM_MSG_TAIL=%LOCALM_GUI_LOG%"
call :notify Error
call :drop_log
exit /b 1

rem ===========================================================================
rem  :run_window - runs the setup window; the errorlevel is the window's.
rem ===========================================================================
:run_window
"%UVEXE%" run --no-project --isolated --python 3.12 python "installer\gui.py"
goto :eof

:drop_log
if defined LOCALM_GUI_LOG del "%LOCALM_GUI_LOG%" >nul 2>nul
goto :eof

rem ===========================================================================
rem  :write_bootstrap - sets LOCALM_GUI_BOOTSTRAP from B_UV, B_PYTHON and
rem  B_CACHE and writes it to .localm-gui-bootstrap, or removes that file when
rem  none is set.
rem ===========================================================================
:write_bootstrap
set "LOCALM_GUI_BOOTSTRAP="
if defined B_UV set "LOCALM_GUI_BOOTSTRAP=uv"
if defined B_PYTHON set "LOCALM_GUI_BOOTSTRAP=%LOCALM_GUI_BOOTSTRAP% python"
if defined B_CACHE set "LOCALM_GUI_BOOTSTRAP=%LOCALM_GUI_BOOTSTRAP% cache"
if not defined LOCALM_GUI_BOOTSTRAP del ".localm-gui-bootstrap" >nul 2>nul
if defined LOCALM_GUI_BOOTSTRAP echo %LOCALM_GUI_BOOTSTRAP%> ".localm-gui-bootstrap"
goto :eof

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
if not exist ".uv\" set "B_UV="
if not exist ".python\" set "B_PYTHON="
if not exist ".cache\" set "B_CACHE="
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
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; $m=@(1..5 | ForEach-Object { [Environment]::GetEnvironmentVariable('LOCALM_MSG'+$_) } | Where-Object { $_ }) -join [Environment]::NewLine; if ($env:LOCALM_MSG_TAIL -and [IO.File]::Exists($env:LOCALM_MSG_TAIL)) { $m += [Environment]::NewLine + [Environment]::NewLine + ((Get-Content -LiteralPath $env:LOCALM_MSG_TAIL -Tail 12) -join [Environment]::NewLine) }; $o=New-Object System.Windows.Forms.Form; $o.TopMost=$true; $o.ShowInTaskbar=$false; $o.Opacity=0; $o.FormBorderStyle='None'; $o.StartPosition='CenterScreen'; $o.Width=1; $o.Height=1; $o.Show(); $o.Activate(); [void][System.Windows.Forms.MessageBox]::Show($o, $m, 'LocaLM Setup', 'OK', $env:LOCALM_MSG_ICON); $o.Close()"
goto notify_clear
:notify_console
setlocal EnableDelayedExpansion
echo.
for %%n in (1 2 3 4 5) do if defined LOCALM_MSG%%n echo   !LOCALM_MSG%%n!
endlocal
:notify_clear
for %%n in (1 2 3 4 5) do set "LOCALM_MSG%%n="
set "LOCALM_MSG_ICON="
set "LOCALM_MSG_TAIL="
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
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; [System.Windows.Forms.Application]::EnableVisualStyles(); $f=New-Object System.Windows.Forms.Form; $f.Text='LocaLM Setup'; $f.FormBorderStyle='FixedDialog'; $f.MaximizeBox=$false; $f.MinimizeBox=$false; $f.StartPosition='CenterScreen'; $f.Width=480; $f.Height=170; $f.TopMost=$true; $l=New-Object System.Windows.Forms.Label; $l.Text=$env:LOCALM_SPLASH_TEXT; $l.SetBounds(16,14,436,62); $f.Controls.Add($l); $p=New-Object System.Windows.Forms.ProgressBar; $p.Style='Marquee'; $p.SetBounds(16,84,436,18); $f.Controls.Add($p); $s=[DateTime]::UtcNow; $k=$env:LOCALM_GUI_SPLASH; $t=New-Object System.Windows.Forms.Timer; $t.Interval=400; $t.Add_Tick({ if (-not [IO.File]::Exists($k) -or ([DateTime]::UtcNow - $s).TotalMinutes -gt 30) { $t.Stop(); $f.Close() } }); $t.Start(); [void]$f.ShowDialog()"
goto :eof

:splash_stop
if defined LOCALM_GUI_SPLASH del "%LOCALM_GUI_SPLASH%" >nul 2>nul
set "LOCALM_GUI_SPLASH="
goto :eof
