# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native app identity (localm/applaunch.py).

Covers the pure / guarded logic that is safe to exercise on any platform: the .ico
-> RT_GROUP_ICON transform, the .desktop text, launcher path resolution, the
restart-argv identity contract, and the best-effort guards (every entry point must
be a no-op, never raise, off its platform). The live LocaLM.exe end-to-end
(process name, tray, restart) is not covered here."""

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

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
    # PurePosixPath: the .desktop file is a Linux artifact whose paths use
    # forward slashes.
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
#  Restart identity contract                                         #
# ------------------------------------------------------------------ #

def test_restart_argv_preserves_the_running_executable():
    from localm.inference import http_server
    argv = http_server._restart_argv()
    # The re-exec target is THIS interpreter, so a process launched as
    # LocaLM.exe restarts as LocaLM.exe.
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
    monkeypatch.delenv("LOCALM_OWN_CONSOLE", raising=False)
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    # Mock sys.executable basename to be localm.exe
    monkeypatch.setattr(os.path, "basename", lambda path: "localm.exe" if "python" not in path else os.path.basename(path))
    # Mock sys.executable itself.
    monkeypatch.setattr(sys, "executable", "Z:\\some\\path\\localm.exe")
    monkeypatch.setattr(applaunch, "_owns_console", lambda: True)
    
    applaunch.apply_window_identity()
    assert os.environ.get("LOCALM_OWN_CONSOLE") is None


def test_make_launcher_returns_result_without_raising():
    # Assert only that the call is safe and typed; do not mutate the real venv.
    if sys.platform not in ("win32",) and not sys.platform.startswith("linux"):
        res = applaunch.make_launcher()
        assert res.ok is False and res.notes


# ------------------------------------------------------------------ #
#  --force rebuild FROM the already-running branded launcher          #
# ------------------------------------------------------------------ #

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only bug")
class TestForceRebuildFromRunningLauncher:
    """``localm make-launcher --force`` run FROM the already-built LocaLM.exe
    itself (e.g. to refresh the copy after a Python upgrade).

    ``sys._base_executable`` is computed by CPython as ``<base_prefix>/
    <basename of the CURRENTLY RUNNING exe>``, so once running AS the renamed
    LocaLM.exe copy it resolves to ``<base_prefix>/LocaLM.exe``, a file that
    never exists (LocaLM.exe only ever lives under ``<venv>/localm-app/``).
    ``_base_interpreter()`` must fall back to ``_mp_spawn.real_base_python()``,
    and once ``base`` resolves to a DIFFERENT file than ``dst``, copying onto
    ``dst`` needs the running-exe rename fallback, because ``dst`` IS this
    process's own executing image and a direct ``shutil.copy2`` onto it raises
    WinError 32.

    The fake launcher must be named exactly ``LocaLM.exe`` (so
    ``windows_launcher_path()`` resolves to ITS OWN path) and it invokes the
    real ``localm make-launcher --force`` CLI command on itself, not a bare
    interpreter round trip.
    """

    @staticmethod
    def _build_fake_launcher(tmp_path: Path) -> Path:
        """A fake venv (pyvenv.cfg pointing at the REAL base interpreter's own
        directory) with a renamed copy of that base interpreter one level
        under it, named LocaLM.exe - exactly applaunch.py's
        ``<venv>/localm-app/LocaLM.exe`` layout, so ``windows_launcher_path()``
        resolves to this same file when run from it."""
        real_base = Path(getattr(sys, "_base_executable", None) or sys.executable).resolve()
        fake_venv = tmp_path / "fakevenv"
        fake_venv.mkdir()
        (fake_venv / "pyvenv.cfg").write_text(
            f"home = {real_base.parent}\n", encoding="utf-8")
        launcher_dir = fake_venv / "localm-app"
        launcher_dir.mkdir()
        fake_launcher = launcher_dir / "LocaLM.exe"
        shutil.copy2(real_base, fake_launcher)
        for pattern in ("python3*.dll", "vcruntime*.dll"):
            for dll in real_base.parent.glob(pattern):
                shutil.copy2(dll, launcher_dir / dll.name)
        return fake_launcher

    @staticmethod
    def _child_pythonpath(repo_root: Path) -> str:
        """The fake launcher is a bare copied interpreter with no site-packages
        of its own. Forward the worktree source (first, so it wins over any
        other localm on sys.path) plus every entry already on THIS process's
        sys.path (site-packages, editable-install source, ...) so the child
        can import click/rich/pydantic/... exactly as the real LocaLM.exe
        would - without hardcoding a venv location that only exists on one
        machine (rule 1: no machine-specific paths)."""
        entries = [str(repo_root)] + [p for p in sys.path if p]
        return os.pathsep.join(dict.fromkeys(entries))

    def test_force_rebuild_from_running_launcher_succeeds(self, tmp_path):
        fake_launcher = self._build_fake_launcher(tmp_path)
        repo_root = Path(__file__).resolve().parents[1]
        script = tmp_path / "run_make_launcher.py"
        script.write_text(
            "import sys\n"
            "from localm.cli import main as cli_main\n"
            "print('_base_executable:', getattr(sys, '_base_executable', None))\n"
            "try:\n"
            "    cli_main(['make-launcher', '--force'], standalone_mode=False)\n"
            "except SystemExit as e:\n"
            "    if e.code not in (0, None):\n"
            "        print(f'REGRESSION_TEST: FAILED (exit {e.code})')\n"
            "        sys.exit(e.code)\n"
            "print('REGRESSION_TEST: SUCCESS')\n",
            encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = self._child_pythonpath(repo_root)
        # LOCALM_HOME is inherited from the autouse _isolate_localm_home
        # fixture (conftest.py), so this never touches the developer's real
        # data dir or the repo's own portable-mode home/.

        result = subprocess.run(
            [str(fake_launcher), str(script)],
            capture_output=True, text=True, timeout=60, env=env)

        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        assert "REGRESSION_TEST: SUCCESS" in result.stdout
        # The failure note must not appear.
        assert "could not locate the base interpreter to copy" not in result.stdout
        # The fallback must have been exercised: dst really was this process's
        # own running image.
        assert "renamed the old copy aside to replace it" in result.stdout
        # The launcher file itself must have survived the rebuild.
        assert fake_launcher.is_file()
