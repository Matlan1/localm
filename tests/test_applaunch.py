# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 3 native app identity (localm/applaunch.py).

Covers the pure / guarded logic that is safe to exercise on any platform: the .ico
-> RT_GROUP_ICON transform, the .desktop text, launcher path resolution, the
restart-argv identity contract, and the best-effort guards (every entry point must
be a no-op, never raise, off its platform). The live LocaLM.exe end-to-end
(process name, tray, restart) is verified by hand on Windows - see
dev-notes/native-app-identity/WORKLOG.md."""

import struct
import sys
from pathlib import Path, PurePosixPath

from localm import applaunch


# ------------------------------------------------------------------ #
#  .ico parsing + RT_GROUP_ICON build (the PE icon-stamp inputs)      #
# ------------------------------------------------------------------ #

def _ico_bytes() -> bytes:
    ico = applaunch.ico_path()
    assert ico, "assets/localm.ico must ship for the launcher icon"
    return Path(ico).read_bytes()


def test_parse_ico_reads_every_image():
    data = _ico_bytes()
    count = struct.unpack("<HHH", data[:6])[2]
    entries = applaunch._parse_ico(data)
    assert len(entries) == count and count >= 1
    # Every parsed image is the exact byte-length its directory entry declares.
    for fields, img in entries:
        assert len(img) == fields[6]  # dwBytesInRes


def test_parse_ico_rejects_garbage():
    assert applaunch._parse_ico(b"") == []
    assert applaunch._parse_ico(b"not an icon at all") == []
    # Right magic, but the directory points past the end of the buffer.
    bad = struct.pack("<HHH", 0, 1, 1) + struct.pack("<BBBBHHII", 16, 16, 0, 0, 1, 32, 999999, 999999)
    assert applaunch._parse_ico(bad) == []


def test_build_group_icon_layout():
    entries = applaunch._parse_ico(_ico_bytes())
    grp = applaunch._build_group_icon(entries)
    reserved, itype, count = struct.unpack("<HHH", grp[:6])
    assert (reserved, itype, count) == (0, 1, len(entries))
    # 6-byte header + one 14-byte GRPICONDIRENTRY per image.
    assert len(grp) == 6 + 14 * len(entries)
    # The last 2 bytes of each entry are the RT_ICON resource id, 1..N in order.
    for idx in range(len(entries)):
        entry = grp[6 + idx * 14: 6 + (idx + 1) * 14]
        assert struct.unpack("<H", entry[12:14])[0] == idx + 1


# ------------------------------------------------------------------ #
#  .desktop generation (Linux)                                        #
# ------------------------------------------------------------------ #

def test_desktop_entry_has_required_keys():
    # PurePosixPath so the assertion holds regardless of the host OS the test runs
    # on: the .desktop file is a Linux artifact where paths use forward slashes.
    text = applaunch._desktop_entry_text(
        exec_path=PurePosixPath("/opt/venv/bin/LocaLM"),
        workdir=PurePosixPath("/opt/clone"), icon="/opt/clone/assets/localm.svg")
    assert text.startswith("[Desktop Entry]")
    assert "Name=LocaLM" in text
    assert "Type=Application" in text
    assert "Exec=/opt/venv/bin/LocaLM -m localm gui" in text
    assert "Path=/opt/clone" in text
    assert "Icon=/opt/clone/assets/localm.svg" in text
    assert "Terminal=false" in text


def test_desktop_entry_omits_icon_when_absent():
    text = applaunch._desktop_entry_text(
        exec_path=PurePosixPath("/x/LocaLM"), workdir=PurePosixPath("/x"), icon=None)
    assert "Icon=" not in text


# ------------------------------------------------------------------ #
#  Launcher path resolution                                           #
# ------------------------------------------------------------------ #

def test_windows_launcher_path_is_contained_in_venv():
    # Always under the venv root's localm-app dir, whatever the interpreter name.
    d = applaunch.windows_launcher_dir()
    assert d == Path(sys.prefix) / "localm-app"
    assert applaunch.windows_launcher_path() == d / "LocaLM.exe"


def test_linux_launcher_path_is_in_venv_bin():
    assert applaunch.linux_launcher_path() == Path(sys.prefix) / "bin" / "LocaLM"


def test_svg_icon_ships():
    # The Linux .desktop points Icon= at this; it must exist in the repo.
    assert applaunch.svg_icon_path() is not None


# ------------------------------------------------------------------ #
#  Restart identity contract (the reason a restart keeps LocaLM.exe)  #
# ------------------------------------------------------------------ #

def test_restart_argv_preserves_the_running_executable():
    from localm.inference import http_server
    argv = http_server._restart_argv()
    # The re-exec target is THIS interpreter - so if it was launched as
    # LocaLM.exe, the restarted process is LocaLM.exe too. This is the whole
    # mechanism by which a restart keeps the native identity.
    assert argv[0] == sys.executable
    assert argv[1:3] == ["-m", "localm"]


# ------------------------------------------------------------------ #
#  Best-effort guards: never raise, no-op off platform               #
# ------------------------------------------------------------------ #

def test_stamp_exe_icon_is_guarded():
    # A non-existent target / non-Windows: must return False, never raise.
    assert applaunch._stamp_exe_icon(Path("does-not-exist.exe"), "nope.ico") is False


def test_self_check_of_bogus_exe_is_false():
    assert applaunch._self_check(Path("definitely-not-a-real-exe-xyz")) is False


def test_owns_console_is_guarded_bool():
    # Returns a bool on any platform; off Windows it is a no-op False. Never raises.
    result = applaunch._owns_console()
    assert isinstance(result, bool)
    if sys.platform != "win32":
        assert result is False


def test_apply_window_identity_never_raises():
    # Returns a bool on any platform; off Windows it is a no-op False.
    result = applaunch.apply_window_identity()
    assert isinstance(result, bool)
    if sys.platform != "win32":
        assert result is False


def test_apply_window_identity_skips_console_own_in_debug(monkeypatch):
    import os
    monkeypatch.delenv("LOCALM_OWN_CONSOLE", raising=False)
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    # Mock sys.executable basename to be localm.exe
    monkeypatch.setattr(os.path, "basename", lambda path: "localm.exe" if "python" not in path else os.path.basename(path))
    # We can mock sys.executable specifically
    monkeypatch.setattr(sys, "executable", "C:\\some\\path\\localm.exe")
    monkeypatch.setattr(applaunch, "_owns_console", lambda: True)
    
    applaunch.apply_window_identity()
    assert os.environ.get("LOCALM_OWN_CONSOLE") is None


def test_make_launcher_returns_result_without_raising():
    # Do not mutate the real venv here: only assert the call is safe and typed.
    # (The building itself is exercised by hand in a throwaway venv.)
    if sys.platform not in ("win32",) and not sys.platform.startswith("linux"):
        res = applaunch.make_launcher()
        assert res.ok is False and res.notes
