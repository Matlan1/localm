# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native app identity: give the running localm server a real, branded process identity - 'LocaLM.exe' in Task Manager and the LocaLM icon on the taskbar - instead of a bare python.exe."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# The window title / display name.
APP_NAME = "LocaLM"
# The taskbar grouping identity (Windows AppUserModelID). A distinct dotted id so
# the running server is grouped as LocaLM rather than under the generic Python host.
APP_USER_MODEL_ID = "LocaLM.Server"


# --------------------------------------------------------------------------- #
#  Asset + interpreter-location helpers                                        #
# --------------------------------------------------------------------------- #

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ico_path() -> Optional[str]:
    """The bundled LocaLM .ico (Windows), or None if missing."""
    from localm.appface import icon_path
    return icon_path()


def svg_icon_path() -> Optional[str]:
    """The bundled LocaLM .svg, for the Linux .desktop Icon= (freedesktop renders SVG icons from an absolute path)."""
    p = _repo_root() / "assets" / "localm.svg"
    return str(p) if p.is_file() else None


def _venv_root() -> Path:
    """This venv's root (``sys.prefix``)."""
    return Path(sys.prefix)


def _base_interpreter() -> Optional[Path]:
    """The REAL interpreter behind this process."""
    be = getattr(sys, "_base_executable", None) or sys.executable
    try:
        p = Path(be).resolve()
    except OSError:
        p = Path(be)
    if p.is_file():
        return p
    from localm._mp_spawn import real_base_python
    return real_base_python()


@dataclass
class LauncherResult:
    """Outcome of a launcher build. ``notes`` carries the human-readable surface of what happened (and why a best-effort step did not), so a caller can print it and a failure is never silent."""
    ok: bool
    path: Optional[Path] = None
    icon_stamped: bool = False
    desktop_file: Optional[Path] = None
    notes: List[str] = field(default_factory=list)


def _copy_runtime_dlls(src_dir: Path, dst_dir: Path) -> List[str]:
    """Copy the loader-critical DLLs (python3*.dll, vcruntime*.dll) next to the launcher so a base-interpreter copy can start outside its original directory."""
    copied: List[str] = []
    for pattern in ("python3*.dll", "vcruntime*.dll"):
        for dll in src_dir.glob(pattern):
            try:
                shutil.copy2(dll, dst_dir / dll.name)
                copied.append(dll.name)
            except OSError:
                pass
    return copied


def _owns_console() -> bool:
    """True when this process is the ONLY one attached to its console - i.e. it was given its OWN console (a double-click, or the launcher's CREATE_NEW_CONSOLE), not launched into an existing terminal that it shares."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        buf = (ctypes.c_uint * 8)()
        n = ctypes.windll.kernel32.GetConsoleProcessList(buf, 8)
        return n == 1
    except Exception:
        return False


def _self_check(exe: Path) -> bool:
    """Run the just-built launcher and confirm it starts venv-aware."""
    try:
        r = subprocess.run(
            [str(exe), "-c", "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 3)"],
            capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  Windows: LocaLM.exe (a branded copy of the base interpreter)               #
# --------------------------------------------------------------------------- #

def windows_launcher_dir() -> Path:
    """Where the Windows launcher + its DLLs live: ``<venv>/localm-app``."""
    return _venv_root() / "localm-app"


def windows_launcher_path() -> Path:
    """The Windows launcher: ``<venv>/localm-app/LocaLM.exe``."""
    return windows_launcher_dir() / f"{APP_NAME}.exe"


def _copy_replacing_possibly_running_exe(src: Path, dst: Path) -> bool:
    """Copy *src* onto *dst*, handling *dst* being the image THIS process is currently executing from (``make-launcher --force`` invoked from the branded LocaLM.exe launcher itself, to refresh it after a Python upgrade - ``_base_interpreter`` then resolves *src* to the real base interpreter, a different fi..."""
    try:
        shutil.copy2(src, dst)
        return False
    except OSError as copy_err:
        old = dst.with_name(dst.name + ".old")
        try:
            if old.exists():
                old.unlink()
            dst.rename(old)
            shutil.copy2(src, dst)
        except OSError:
            # Surface the ORIGINAL failure (why the fallback was needed at
            # all) as the primary error; Python's implicit chaining still
            # attaches the fallback's own error as __context__, so neither is
            # hidden if this ever reaches a full traceback.
            raise copy_err
        try:
            old.unlink()
        except OSError:
            pass  # still running; freed on the next rebuild instead
        return True


def make_windows_launcher(*, force: bool = False) -> LauncherResult:
    """Create ``<venv>/localm-app/LocaLM.exe`` as a copy of the base interpreter (+ its loader DLLs) and stamp the LocaLM icon into it."""
    base = _base_interpreter()
    if base is None:
        return LauncherResult(ok=False,
                              notes=["could not locate the base interpreter to copy; "
                                     "cannot build LocaLM.exe"])
    dst = windows_launcher_path()
    notes: List[str] = []
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not force and dst.is_file():
            notes.append(f"{dst.name} already present (use --force to refresh)")
            return LauncherResult(ok=True, path=dst, notes=notes)
        # base is always a different file than dst (the base interpreter never
        # lives under dst's own <venv>/localm-app/ dir) - but when rebuilding
        # via --force from the already-running LocaLM.exe, dst IS this
        # process's own executing image, so the copy needs the running-exe
        # fallback below rather than a plain shutil.copy2.
        replaced_running = _copy_replacing_possibly_running_exe(base, dst)
        dlls = _copy_runtime_dlls(base.parent, dst.parent)
        notes.append(f"built {dst.name} from {base.name} + {len(dlls)} runtime DLL(s)"
                     + (" (it was running; renamed the old copy aside to replace it)"
                        if replaced_running else ""))
    except OSError as e:
        return LauncherResult(ok=False, path=dst,
                              notes=[f"could not build {dst.name}: {e}"])

    stamped = False
    ico = ico_path()
    if ico:
        stamped = _stamp_exe_icon(dst, ico)
        notes.append("stamped the LocaLM icon into the exe" if stamped else
                     "could not stamp the exe icon (the LocaLM icon still shows on "
                     "the taskbar, shortcut and tray at runtime)")
    else:
        notes.append("no localm.ico found; skipped icon stamping")

    # Verify the final artifact actually runs venv-aware before claiming success -
    # never ship a broken launcher (do-not-hide-problems).
    if not _self_check(dst):
        notes.append("the built launcher did not start correctly; removed it. "
                     "`localm gui` still works (Task Manager may then show "
                     "python.exe). Please report this.")
        try:
            dst.unlink()
        except OSError:
            pass
        return LauncherResult(ok=False, path=dst, icon_stamped=stamped, notes=notes)

    return LauncherResult(ok=True, path=dst, icon_stamped=stamped, notes=notes)


# ---- PE icon stamping (ctypes UpdateResource, no dependency) --------------- #

def _parse_ico(data: bytes) -> List[tuple]:
    """Parse a .ico into ``[(header_fields, image_bytes), ...]``."""
    if len(data) < 6:
        return []
    _reserved, itype, count = struct.unpack("<HHH", data[:6])
    if itype != 1 or count == 0:
        return []
    entries: List[tuple] = []
    off = 6
    for _ in range(count):
        if off + 16 > len(data):
            return []
        fields = struct.unpack("<BBBBHHII", data[off:off + 16])
        (_w, _h, _cc, _res, _planes, _bits, size, offset) = fields
        img = data[offset:offset + size]
        if len(img) != size:
            return []
        entries.append((fields[:7], img))  # drop the file offset; keep size at [6]
        off += 16
    return entries


def _build_group_icon(entries: List[tuple]) -> bytes:
    """Build the RT_GROUP_ICON directory referencing RT_ICON ids 1..N."""
    out = struct.pack("<HHH", 0, 1, len(entries))
    for idx, (fields, _img) in enumerate(entries, start=1):
        (w, h, cc, res, planes, bits, size) = fields
        # GRPICONDIRENTRY: like ICONDIRENTRY but the trailing 4-byte file offset is
        # replaced by a 2-byte resource id.
        out += struct.pack("<BBBBHHIH", w, h, cc, res, planes, bits, size, idx)
    return out


def _stamp_exe_icon(exe: Path, ico: str) -> bool:
    """Stamp *ico* into *exe*'s PE resources (RT_ICON + RT_GROUP_ICON) so Explorer / Task Manager show the LocaLM icon on the file itself."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        entries = _parse_ico(Path(ico).read_bytes())
        if not entries:
            return False

        RT_ICON = 3
        RT_GROUP_ICON = 14
        LANG_EN_US = 0x0409

        k32 = ctypes.windll.kernel32
        k32.BeginUpdateResourceW.restype = wintypes.HANDLE
        k32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
        k32.UpdateResourceW.restype = wintypes.BOOL
        k32.UpdateResourceW.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID,
            wintypes.WORD, wintypes.LPVOID, wintypes.DWORD]
        k32.EndUpdateResourceW.restype = wintypes.BOOL
        k32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]

        h = k32.BeginUpdateResourceW(str(exe), False)
        if not h:
            return False

        def _update(rtype: int, rid: int, blob: bytes) -> bool:
            buf = ctypes.create_string_buffer(blob, len(blob))
            return bool(k32.UpdateResourceW(
                h, ctypes.c_void_p(rtype), ctypes.c_void_p(rid), LANG_EN_US,
                ctypes.cast(buf, wintypes.LPVOID), len(blob)))

        ok = False
        try:
            ok = True
            for idx, (_fields, img) in enumerate(entries, start=1):
                ok = _update(RT_ICON, idx, img) and ok
            ok = _update(RT_GROUP_ICON, 1, _build_group_icon(entries)) and ok
        finally:
            # Always close the BeginUpdateResource handle - commit on full success,
            # else discard - so an exception mid-update never leaks it or leaves a
            # half-written resource directory behind.
            committed = bool(k32.EndUpdateResourceW(h, not ok))
        return ok and committed
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  Windows: live-window identity (taskbar grouping + console icon)             #
# --------------------------------------------------------------------------- #

def apply_window_identity(*, app_id: str = APP_USER_MODEL_ID) -> bool:
    """Give this running process a real app identity on Windows: an explicit AppUserModelID (taskbar groups it as LocaLM) and the LocaLM icon on the console window (taskbar button, alt-tab, window corner)."""
    if sys.platform != "win32":
        return False
    applied = False
    try:
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
            applied = True
        except Exception:
            pass  # AppUserModelID is a nicety; the console icon below still helps
        ico = ico_path()
        if ico:
            applied = _set_console_icon(ico) or applied
        # If we ARE the dedicated LocaLM.exe launcher and we own our console (a
        # double-click / the launcher's CREATE_NEW_CONSOLE, not a shared terminal),
        # opt into the console-less background-app behavior: winconsole.hide_console
        # fires on LOCALM_OWN_CONSOLE once the server is up, so the LocaLM.exe
        # shortcut matches the launcher's console-less experience. A dev running
        # LocaLM.exe inside a terminal shares that console, so it is left visible.
        if (os.path.basename(sys.executable).lower() == "localm.exe"
                and _owns_console()
                and not os.environ.get("LOCALM_DEBUG")
                and "--debug" not in sys.argv):
            os.environ.setdefault("LOCALM_OWN_CONSOLE", "1")
    except Exception:
        return applied
    return applied


def _set_console_icon(ico: str) -> bool:
    """Set the LocaLM icon on this process's console window via WM_SETICON."""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            return False  # no console attached - nothing to icon

        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040
        WM_SETICON = 0x0080
        ICON_SMALL, ICON_BIG = 0, 1

        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR,
                                      wintypes.UINT, ctypes.c_int, ctypes.c_int,
                                      wintypes.UINT]
        user32.SendMessageW.restype = ctypes.c_ssize_t
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                        wintypes.WPARAM, wintypes.LPARAM]

        # These HICONs are owned by the console window for its (process) lifetime:
        # WM_SETICON does not copy them, so we intentionally do NOT DestroyIcon here
        # (that would blank the icon we just set). apply_window_identity runs once per
        # process, so this is at most two handles held until exit and reclaimed by the
        # OS then - not an accumulating leak.
        big = user32.LoadImageW(None, ico, IMAGE_ICON, 0, 0,
                                LR_LOADFROMFILE | LR_DEFAULTSIZE)
        small = user32.LoadImageW(None, ico, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
        ok = False
        if big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
            ok = True
        if small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)
            ok = True
        return ok
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  Linux: bin/LocaLM + LocaLM.desktop                                          #
# --------------------------------------------------------------------------- #

def linux_launcher_path() -> Path:
    """Where the Linux launcher is built: ``<venv>/bin/LocaLM``."""
    return _venv_root() / "bin" / APP_NAME


def _desktop_entry_text(*, exec_path: Path, workdir: Path,
                        icon: Optional[str]) -> str:
    """The freedesktop .desktop launcher text for LocaLM."""
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        f"Name={APP_NAME}",
        "Comment=Local AI, offline",
        f"Exec={exec_path} -m localm gui",
        f"Path={workdir}",
        # Terminal=false to match setup.sh's menu entry: the GUI opens in the
        # browser, so no terminal window is needed (and a copied LocaLM.desktop then
        # behaves the same as the setup-installed one).
        "Terminal=false",
        "Categories=Utility;Development;",
    ]
    if icon:
        lines.insert(5, f"Icon={icon}")
    return "\n".join(lines) + "\n"


def make_linux_launcher(*, force: bool = False) -> LauncherResult:
    """Create ``<venv>/bin/LocaLM`` (a copy of the interpreter, so a process monitor shows 'LocaLM') and write a ``LocaLM.desktop`` launcher next to the clone."""
    base = _base_interpreter()
    if base is None:
        return LauncherResult(ok=False,
                              notes=["no interpreter found to copy for bin/LocaLM"])
    dst = linux_launcher_path()
    notes: List[str] = []
    built_ok = False
    try:
        entrypoint = _venv_root() / "bin" / "localm"
        is_case_insensitive_clash = False
        try:
            if entrypoint.exists() and dst.exists() and os.path.samefile(entrypoint, dst):
                is_case_insensitive_clash = True
        except OSError:
            pass

        if is_case_insensitive_clash:
            notes.append("case-insensitive filesystem detected; skipping LocaLM binary to preserve the 'localm' command")
        elif force or not dst.exists():
            shutil.copy2(base, dst)
            try:
                dst.chmod(0o755)
            except OSError:
                pass
            notes.append(f"built {dst.name} (copy of {base.name})")
        else:
            notes.append(f"{dst.name} already present (use --force to refresh)")
        
        if not is_case_insensitive_clash:
            built_ok = _self_check(dst)
        if not built_ok:
            # Do not leave a launcher that does not run; the .desktop falls back to
            # the venv python (still works, just shows python in a process monitor).
            try:
                dst.unlink()
            except OSError:
                pass
            notes.append("the copied launcher did not start standalone (its runtime "
                         "libs are not resolvable next to it); the .desktop will use "
                         "the venv python instead, so it still works but a process "
                         "monitor shows python. An AppImage is the robust path - see "
                         "docs/native-app.md.")
    except OSError as e:
        notes.append(f"could not build {dst.name}: {e}; the .desktop will use the "
                     "venv python instead")

    exec_path = dst if built_ok else Path(sys.executable)
    repo_root = _repo_root()
    desktop = repo_root / f"{APP_NAME}.desktop"
    written: Optional[Path] = None
    try:
        desktop.write_text(
            _desktop_entry_text(exec_path=exec_path, workdir=repo_root,
                                icon=svg_icon_path()),
            encoding="utf-8")
        written = desktop
        notes.append(f"wrote {desktop.name} (copy it to "
                     "~/.local/share/applications/ to add it to your menu)")
    except OSError as e:
        notes.append(f"could not write {desktop.name}: {e}")
    return LauncherResult(ok=True, path=(dst if built_ok else None),
                          desktop_file=written, notes=notes)


# --------------------------------------------------------------------------- #
#  Cross-platform entry point                                                  #
# --------------------------------------------------------------------------- #

def make_launcher(*, force: bool = False) -> LauncherResult:
    """Build the native launcher for the current OS (LocaLM.exe on Windows, bin/ LocaLM + LocaLM.desktop on Linux)."""
    if sys.platform == "win32":
        return make_windows_launcher(force=force)
    if sys.platform.startswith("linux"):
        return make_linux_launcher(force=force)
    return LauncherResult(
        ok=False,
        notes=[f"no native launcher for platform {sys.platform!r} yet; "
               "run 'localm gui' directly"])
