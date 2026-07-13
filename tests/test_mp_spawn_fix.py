# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression pin for #617 and its follow-up: a GGUF model load (or the
voice/STT worker) must not fail when running under the branded LocaLM.exe
launcher - neither at spawn time nor on the worker's first real use.

Bug 1 (the original #617): CPython's multiprocessing spawns a Windows child
via ``sys._base_executable`` whenever it differs from ``sys.executable`` (its
definition of "running in a venv") - computed as ``<base_prefix>/<basename of
the running exe>``, which does not exist once the running exe has been
renamed (LocaLM.exe, see applaunch.py's ``make_windows_launcher``). Spawning a
child then fails immediately with ``FileNotFoundError: [WinError 2] The
system cannot find the file specified``.

Bug 2 (found fixing bug 1, NOT caught by a bare "does spawn succeed" check):
redirecting to the venv's OWN ``Scripts/python.exe`` fixes the spawn, but that
file is itself a TRAMPOLINE that re-spawns the real base interpreter as a
NESTED child. Windows multiprocessing hands a spawned child its Queue/Lock
semaphore handles via a single ``DuplicateHandle`` call targeting that
child's OWN process handle (``Popen.duplicate_for_child`` in
``popen_spawn_win32.py``); the handle lands in the trampoline's process, not
in the nested worker that actually uses it, so the worker's first
``Queue.get()``/``Lock`` use fails with ``OSError: [WinError 6] The handle is
invalid``. See localm/_mp_spawn.py for the full account and why the fix
redirects straight to the base interpreter (``sys.base_prefix``) instead.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.spawn
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from localm import _mp_spawn


@pytest.fixture(autouse=True)
def _restore_mp_executable():
    # multiprocessing.set_executable mutates module-global state; never leak a
    # test-set value into a later test or the real interpreter's own default.
    original = multiprocessing.spawn.get_executable()
    yield
    multiprocessing.set_executable(original)


def test_noop_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    before = multiprocessing.spawn.get_executable()
    _mp_spawn.ensure_spawn_uses_venv_python()
    assert multiprocessing.spawn.get_executable() == before


def test_repoints_to_base_interpreter_on_windows(monkeypatch, tmp_path):
    # A fake venv layout: sys.base_prefix is the BASE install directory, and a
    # never-renamed python.exe sits directly under it - NOT under Scripts/,
    # since Scripts/python.exe is itself a trampoline (bug 2 above) and must
    # not be the redirect target.
    base_python = tmp_path / "python.exe"
    base_python.write_bytes(b"")   # existence is all this function checks

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path))

    _mp_spawn.ensure_spawn_uses_venv_python()

    assert multiprocessing.spawn.get_executable() == str(base_python)


def test_missing_base_python_leaves_default_untouched(monkeypatch, tmp_path):
    # No python.exe under this base_prefix (an unexpected layout) - must not
    # point multiprocessing at a file that does not exist.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path))
    before = multiprocessing.spawn.get_executable()

    _mp_spawn.ensure_spawn_uses_venv_python()

    assert multiprocessing.spawn.get_executable() == before


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only bug")
class TestRealRenamedLauncherEndToEnd:
    """Builds an ACTUAL renamed-copy launcher (mirroring applaunch.py's
    make_windows_launcher construction byte-for-byte) and drives a REAL
    multiprocessing.Queue round trip through it - the exact shape
    ModelRunner/voice._spawn_worker use (send a command, get a response back).
    A bare "does .start() succeed" check is not enough: bug 2 above only
    surfaces on the child's first actual Queue/Lock use, which is why the
    original fix verification (spawn success + a print statement) missed it.
    """

    @staticmethod
    def _build_fake_launcher(tmp_path: Path) -> Path:
        """A fake venv (pyvenv.cfg pointing at the REAL base interpreter's own
        directory) with a renamed copy of that base interpreter one level
        under it - exactly applaunch.py's ``<venv>/localm-app/LocaLM.exe``
        layout, so it reproduces CPython's OWN sys._base_executable
        derivation bug (```<base_prefix>/<basename of THIS renamed exe>```)."""
        real_base = Path(getattr(sys, "_base_executable", None) or sys.executable).resolve()
        fake_venv = tmp_path / "fakevenv"
        (fake_venv).mkdir()
        (fake_venv / "pyvenv.cfg").write_text(
            f"home = {real_base.parent}\n", encoding="utf-8")
        launcher_dir = fake_venv / "localm-app"
        launcher_dir.mkdir()
        fake_launcher = launcher_dir / "FakeLauncher.exe"
        shutil.copy2(real_base, fake_launcher)
        for pattern in ("python3*.dll", "vcruntime*.dll"):
            for dll in real_base.parent.glob(pattern):
                shutil.copy2(dll, launcher_dir / dll.name)
        return fake_launcher

    def test_queue_roundtrip_survives_the_renamed_launcher(self, tmp_path):
        fake_launcher = self._build_fake_launcher(tmp_path)
        repo_root = Path(__file__).resolve().parents[1]
        script = tmp_path / "roundtrip.py"
        script.write_text(textwrap.dedent(f"""\
            import multiprocessing as mp
            import sys
            sys.path.insert(0, {str(repo_root)!r})
            from localm._mp_spawn import ensure_spawn_uses_venv_python

            def child_main(req_q, resp_q):
                resp_q.put(("ok", req_q.get()))

            if __name__ == "__main__":
                print("child sys.executable:", sys.executable)
                ensure_spawn_uses_venv_python()
                ctx = mp.get_context("spawn")
                req_q, resp_q = ctx.Queue(), ctx.Queue()
                proc = ctx.Process(target=child_main, args=(req_q, resp_q), daemon=True)
                proc.start()
                req_q.put({{"cmd": "load"}})
                result = resp_q.get(timeout=20)
                print("RESULT:", result)
                proc.join(5)
                assert proc.exitcode == 0, f"worker exitcode {{proc.exitcode}}"
        """), encoding="utf-8")

        result = subprocess.run(
            [str(fake_launcher), str(script)],
            capture_output=True, text=True, timeout=45)

        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        assert "RESULT: ('ok', {'cmd': 'load'})" in result.stdout
        # The two known failure modes must not appear even in stderr noise.
        assert "WinError 2" not in result.stderr
        assert "WinError 6" not in result.stderr
