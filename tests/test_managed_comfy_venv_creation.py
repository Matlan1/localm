# SPDX-License-Identifier: AGPL-3.0-or-later
"""Creating the managed ComfyUI's own fresh venv must not fail when running
under the branded LocaLM.exe launcher.

When the running process IS the branded LocaLM.exe copy (a raw byte copy of the
base interpreter renamed by applaunch.py's ``make_windows_launcher``), CPython's
stdlib ``venv.EnvBuilder`` matches file names on the RUNNING executable's
basename to decide what to copy into the new venv's ``Scripts/`` dir - a file
that was never copied there, so it is skipped. The new venv then has no launcher
of its own, and venv's mandatory ensurepip bootstrap (``_setup_pip``) invokes
that nonexistent file, failing with ``FileNotFoundError: [WinError 2]``. So
``managed_comfy_fresh.py`` hands ``-m venv`` the result of
``localm/_mp_spawn.py``'s ``real_base_python()``, the real never-renamed base
interpreter, rather than ``sys.executable``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from localm import hwdetect
from localm.media import managed_comfy as mc
from localm.media import managed_comfy_fresh as fresh


def test_provision_fresh_uses_real_base_python_not_sys_executable(monkeypatch, tmp_path):
    """Drives the REAL provision_fresh(), mocking only the clone (no network or
    git needed) and stopping right after the venv-creation step - the fake
    venv_python path is never created, matching a real broken run - so the one
    _run call under test is captured with no other side effects. The venv call
    must use real_base_python(), never sys.executable."""
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return True, ""

    monkeypatch.setattr(fresh, "_run", _fake_run)
    monkeypatch.setattr(fresh, "_clone_at_commit", lambda *a, **k: (True, ""))
    # provision_fresh computes comfy_torch_spec() regardless of install_torch,
    # which on an NVIDIA box would shell out to nvidia-smi via the Blackwell
    # detection; stubbed so the result does not depend on the host hardware.
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])

    fake_root = tmp_path / "comfyui"
    fake_paths = mc.ManagedComfyPaths(
        root=fake_root,
        models_dir=tmp_path / "comfyui-models",
        main_py=fake_root / "main.py",
        venv_python=fake_root / "venv" / "Scripts" / "python.exe",  # never created here
        extra_model_paths=fake_root / "extra_model_paths.yaml",
    )
    monkeypatch.setattr(fresh.mc, "managed_comfy_paths", lambda: fake_paths)

    # Force sys.executable to a value that would be WRONG to use directly (as
    # when running the branded, renamed copy) and confirm the resolved real base
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

    # The venv is never actually created (fake_run is a no-op), so
    # provision_fresh reports failure at the post-venv verification check. What
    # is asserted here is HOW it tried to create the venv.
    assert result.ok is False
    venv_calls = [c for c in calls if "venv" in c]
    assert len(venv_calls) == 1, f"expected exactly one venv-creation call, got: {calls}"
    assert venv_calls[0][0] == str(real_base)
    assert venv_calls[0][0] != str(fake_renamed)


def test_provision_fresh_fails_loudly_when_venv_has_no_pip(monkeypatch, tmp_path):
    """`-m venv` can report success (return code 0, the interpreter file
    present) while its own mandatory ensurepip bootstrap silently failed.
    provision_fresh() probes right after creation and fails there, naming the
    cause, before attempting the torch install.

    install_torch is left at its default (True): without the probe the fake _run
    below would receive the subsequent torch `pip install` call and raise."""
    calls = []
    fake_root = tmp_path / "comfyui"
    venv_python = fake_root / "venv" / "Scripts" / "python.exe"

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "venv" in cmd:
            # Must not pre-exist BEFORE this call: provision_fresh() refuses to
            # run when root already exists, so it is created here instead, at
            # the moment the fake venv-creation step succeeds - interpreter file
            # present, pip missing.
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_bytes(b"")
            return True, ""
        if "pip" in cmd and "--version" in cmd:
            return False, "No module named pip"
        raise AssertionError(f"unexpected call reached past the pip probe: {cmd}")

    monkeypatch.setattr(fresh, "_run", _fake_run)
    monkeypatch.setattr(fresh, "_clone_at_commit", lambda *a, **k: (True, ""))
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])

    fake_paths = mc.ManagedComfyPaths(
        root=fake_root,
        models_dir=tmp_path / "comfyui-models",
        main_py=fake_root / "main.py",
        venv_python=venv_python,
        extra_model_paths=fake_root / "extra_model_paths.yaml",
    )
    monkeypatch.setattr(fresh.mc, "managed_comfy_paths", lambda: fake_paths)

    result = fresh.provision_fresh(
        cfg={}, comfyui_repo="https://example/repo.git", comfyui_commit="deadbeef",
        custom_nodes=[])

    assert result.ok is False
    assert "no working pip" in result.message
    assert "localm doctor" in result.message
    install_calls = [c for c in calls if "install" in c]
    assert install_calls == [], f"reached an install call past the probe: {install_calls}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only bug (#621)")
class TestRealRenamedLauncherEndToEnd:
    """Builds an ACTUAL renamed-copy launcher, mirroring applaunch.py's
    make_windows_launcher construction byte-for-byte, and drives a REAL `-m venv`
    invocation through it: no model needed, and the mechanism itself is not
    mocked."""

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
        """Invoking `-m venv` via the renamed copy itself - what passing
        sys.executable amounts to - fails with WinError 2."""
        fake_launcher = self._build_fake_launcher(tmp_path)
        result = subprocess.run(
            [str(fake_launcher), "-m", "venv", str(tmp_path / "broken_venv")],
            capture_output=True, text=True, timeout=60)
        assert result.returncode != 0
        assert "WinError 2" in (result.stdout + result.stderr)

    def test_real_base_python_resolves_and_creates_a_working_venv(self, tmp_path):
        """Running FROM INSIDE the renamed launcher, real_base_python() resolves
        to the true (never-renamed) interpreter, and using THAT to create the
        venv produces a real, runnable nested venv."""
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
