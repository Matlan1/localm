# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression pin for #621 (the "ComfyUI setup did not finish" report): creating
the managed ComfyUI's own fresh venv must not fail when running under the branded
LocaLM.exe launcher.

Root cause: ``managed_comfy_fresh.py`` used to pass ``sys.executable`` straight to
``-m venv``. When the running process IS the branded LocaLM.exe copy (a raw byte
copy of the base interpreter renamed by applaunch.py's ``make_windows_launcher``),
CPython's own stdlib ``venv.EnvBuilder`` matches file names on the RUNNING
executable's basename (``LocaLM.exe``) to decide what to copy into the new venv's
``Scripts/`` dir - a file that was never copied there in the first place, so it is
silently skipped. The new venv then has no launcher of its own, and venv's
mandatory ensurepip bootstrap (``_setup_pip``) tries to invoke that nonexistent
file, failing with ``FileNotFoundError: [WinError 2] The system cannot find the
file specified`` - reproduced live via GitHub issue #621, byte-for-byte matching
the user's own report. See localm/_mp_spawn.py's ``real_base_python()`` (the fix:
always hand a subprocess the real, never-renamed base interpreter, not whatever
``sys.executable`` happens to currently be named).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from localm import _mp_spawn
from localm.media import managed_comfy as mc
from localm.media import managed_comfy_fresh as fresh


def test_provision_fresh_uses_real_base_python_not_sys_executable(monkeypatch, tmp_path):
    """Drives the REAL provision_fresh() (not a strawman) - mocks only the clone (no
    network/git needed) and stops it right after the venv-creation step (the fake
    venv_python path is never created, matching a real broken run) so the ONE _run
    call this test cares about is captured with no other side effects. Must never
    blindly pass sys.executable - it must ask real_base_python() first."""
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return True, ""

    monkeypatch.setattr(fresh, "_run", _fake_run)
    monkeypatch.setattr(fresh, "_clone_at_commit", lambda *a, **k: (True, ""))

    fake_root = tmp_path / "comfyui"
    fake_paths = mc.ManagedComfyPaths(
        root=fake_root,
        models_dir=tmp_path / "comfyui-models",
        main_py=fake_root / "main.py",
        venv_python=fake_root / "venv" / "Scripts" / "python.exe",  # never created here
        extra_model_paths=fake_root / "extra_model_paths.yaml",
    )
    monkeypatch.setattr(fresh.mc, "managed_comfy_paths", lambda: fake_paths)

    # Force sys.executable to a value that would be WRONG to use directly (mimics
    # running as the branded, renamed copy) and confirm the resolved real base
    # python is used instead.
    fake_renamed = tmp_path / "LocaLM.exe"
    fake_renamed.write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(fake_renamed))
    real_base = tmp_path / "realbase.exe"
    real_base.write_bytes(b"")
    monkeypatch.setattr(fresh, "real_base_python", lambda: real_base)

    result = fresh.provision_fresh(
        cfg={}, comfyui_repo="https://example/repo.git", comfyui_commit="deadbeef",
        custom_nodes=[], install_torch=False)

    # The venv was never actually created (fake_run is a no-op), so provision_fresh
    # correctly reports failure at the post-venv verification check - that's expected
    # and irrelevant here; what matters is HOW it tried to create the venv.
    assert result.ok is False
    venv_calls = [c for c in calls if "venv" in c]
    assert len(venv_calls) == 1, f"expected exactly one venv-creation call, got: {calls}"
    assert venv_calls[0][0] == str(real_base)
    assert venv_calls[0][0] != str(fake_renamed)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only bug (#621)")
class TestRealRenamedLauncherEndToEnd:
    """Builds an ACTUAL renamed-copy launcher (mirroring applaunch.py's
    make_windows_launcher construction byte-for-byte, same helper shape as
    test_mp_spawn_fix.py's TestRealRenamedLauncherEndToEnd) and drives a REAL
    `-m venv` invocation through it - no model needed, no mocking of the actual
    bug mechanism."""

    @staticmethod
    def _build_fake_launcher(tmp_path: Path) -> Path:
        real_base = Path(getattr(sys, "_base_executable", None) or sys.executable).resolve()
        fake_venv = tmp_path / "fakevenv"
        fake_venv.mkdir()
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

    def test_renamed_launcher_reproduces_winerror2_directly(self, tmp_path):
        """Pin the BROKEN behavior: invoking `-m venv` via the renamed copy itself
        (what the old code effectively did by passing sys.executable) must still
        fail exactly like the live #621 report - proves this test's repro is real,
        not a strawman."""
        fake_launcher = self._build_fake_launcher(tmp_path)
        result = subprocess.run(
            [str(fake_launcher), "-m", "venv", str(tmp_path / "broken_venv")],
            capture_output=True, text=True, timeout=60)
        assert result.returncode != 0
        assert "WinError 2" in (result.stdout + result.stderr)

    def test_real_base_python_resolves_and_creates_a_working_venv(self, tmp_path):
        """The fix: running FROM INSIDE the renamed launcher, real_base_python()
        must resolve to the true (never-renamed) interpreter, and using THAT to
        create the venv must fully succeed - a real, runnable nested venv, not
        just a spawn-succeeded check."""
        fake_launcher = self._build_fake_launcher(tmp_path)
        repo_root = Path(__file__).resolve().parents[1]
        dest = tmp_path / "fixed_venv"
        script = tmp_path / "make_venv.py"
        script.write_text(textwrap.dedent(f"""\
            import subprocess
            import sys
            sys.path.insert(0, {str(repo_root)!r})
            from localm._mp_spawn import real_base_python

            found = real_base_python()
            print("resolved:", found)
            assert found is not None, "real_base_python() returned None"
            assert found.name.lower() != {str(fake_launcher.name).lower()!r}, (
                "resolved the renamed launcher itself, not the real base interpreter")

            r = subprocess.run([str(found), "-m", "venv", {str(dest)!r}],
                               capture_output=True, text=True, timeout=120)
            print("STDOUT:", r.stdout)
            print("STDERR:", r.stderr)
            assert r.returncode == 0, f"venv creation failed: {{r.stdout}} {{r.stderr}}"
        """), encoding="utf-8")

        result = subprocess.run(
            [str(fake_launcher), str(script)],
            capture_output=True, text=True, timeout=90)

        assert result.returncode == 0, (
            f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}")
        assert "WinError 2" not in result.stderr

        # And the produced venv must actually be usable - not just files on disk.
        new_python = dest / "Scripts" / "python.exe"
        assert new_python.is_file()
        r = subprocess.run([str(new_python), "-c", "print('ok')"],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        assert "ok" in r.stdout
