# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression pin for GitHub issue #1989: on Windows with an Intel XPU torch,
every HuggingFace model failed to load with

    [WinError 126] ... Error loading "<venv>/Lib/site-packages/torch/lib/c10_xpu.dll"
    or one of its dependencies.

The model loads in a multiprocessing-spawn worker, and
``_mp_spawn.ensure_spawn_uses_venv_python`` starts every worker as the BASE
interpreter: the venv reaches it only as the ``sys.path`` multiprocessing hands
it, and its ``sys.exec_prefix`` names the base install. Before loading its own
DLLs, torch registers ``<sys.exec_prefix>/Library/bin`` as a DLL directory, and
that is where the Intel oneAPI runtime a ``+xpu`` torch depends on is installed
(``intel-sycl-rt``, ``umf``, ``tcmlib``, ...). So the worker looked for it under
the base install, while the same import worked in the server process.
``_mp_spawn.add_venv_dll_directories`` registers the venv's directories in the
worker before anything there imports torch.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from localm import _mp_spawn


@pytest.fixture
def registered(monkeypatch):
    """What os.add_dll_directory was asked to register, instead of changing
    this process's real DLL search path (the function does not exist off
    Windows at all). Each test starts with nothing registered yet."""
    calls = []

    def _add(path):
        calls.append(path)
        return object()   # stands in for the handle os.add_dll_directory returns

    monkeypatch.setattr(os, "add_dll_directory", _add, raising=False)
    monkeypatch.setattr(_mp_spawn, "_venv_dll_directories", {})
    return calls


def _sys_path_without_venvs() -> list:
    """This interpreter's sys.path minus its site-packages entries, so the test
    alone decides which venv is visible while the stdlib stays importable for
    anything that imports lazily during the test."""
    return [p for p in sys.path if Path(p).name.lower() != "site-packages"]


def _worker_state(monkeypatch, tmp_path, *dll_dirs) -> Path:
    """The HF worker's interpreter state on Windows: the BASE interpreter
    (``prefix == base_prefix``) with the venv's site-packages handed over in
    sys.path. Creates each of *dll_dirs* under the venv root."""
    venv = tmp_path / "venv"
    sp = venv / "Lib" / "site-packages"
    sp.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = elsewhere\n", encoding="utf-8")
    for rel in dll_dirs:
        (venv / rel).mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "base"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr(sys, "path", [str(sp)] + _sys_path_without_venvs())
    return venv


class TestAddVenvDllDirectories:
    def test_worker_registers_the_venvs_library_bin(
            self, monkeypatch, tmp_path, registered):
        venv = _worker_state(monkeypatch, tmp_path, "Library/bin")
        expected = [str(venv / "Library" / "bin")]

        assert _mp_spawn.add_venv_dll_directories() == expected
        assert registered == expected
        # Held: closing the handle would take the directory off the path again.
        assert len(_mp_spawn._venv_dll_directories) == 1

    def test_bin_is_registered_too_as_torch_does(
            self, monkeypatch, tmp_path, registered):
        """torch registers <exec_prefix>/bin beside <exec_prefix>/Library/bin."""
        venv = _worker_state(monkeypatch, tmp_path, "Library/bin", "bin")
        expected = [str(venv / "Library" / "bin"), str(venv / "bin")]

        assert _mp_spawn.add_venv_dll_directories() == expected
        assert registered == expected

    def test_missing_directories_register_nothing(
            self, monkeypatch, tmp_path, registered):
        _worker_state(monkeypatch, tmp_path)
        assert _mp_spawn.add_venv_dll_directories() == []
        assert registered == []

    def test_a_repeat_call_registers_nothing_twice(
            self, monkeypatch, tmp_path, registered):
        venv = _worker_state(monkeypatch, tmp_path, "Library/bin")
        _mp_spawn.add_venv_dll_directories()

        assert _mp_spawn.add_venv_dll_directories() == []
        assert registered == [str(venv / "Library" / "bin")]

    def test_process_running_from_the_venv_is_left_to_torch(
            self, monkeypatch, tmp_path, registered):
        """The server process itself: torch registers these on its own there."""
        venv = _worker_state(monkeypatch, tmp_path, "Library/bin")
        monkeypatch.setattr(sys, "prefix", str(venv))
        assert _mp_spawn.add_venv_dll_directories() == []
        assert registered == []

    def test_noop_off_windows(self, monkeypatch, tmp_path, registered):
        _worker_state(monkeypatch, tmp_path, "Library/bin")
        monkeypatch.setattr(sys, "platform", "linux")
        assert _mp_spawn.add_venv_dll_directories() == []
        assert registered == []

    def test_no_venv_on_sys_path_registers_nothing(
            self, monkeypatch, tmp_path, registered):
        """A system-python setup with localm on PYTHONPATH: there is no venv
        whose directories torch would have missed."""
        _worker_state(monkeypatch, tmp_path, "Library/bin")
        monkeypatch.setattr(sys, "path", _sys_path_without_venvs())
        assert _mp_spawn.add_venv_dll_directories() == []
        assert registered == []

    def test_a_refused_directory_is_skipped_not_raised(
            self, monkeypatch, tmp_path):
        venv = _worker_state(monkeypatch, tmp_path, "Library/bin", "bin")
        monkeypatch.setattr(_mp_spawn, "_venv_dll_directories", {})
        refused = str(venv / "Library" / "bin")

        def _add(path):
            if path == refused:
                raise OSError(87, "The parameter is incorrect")
            return object()

        monkeypatch.setattr(os, "add_dll_directory", _add, raising=False)
        assert _mp_spawn.add_venv_dll_directories() == [str(venv / "bin")]


class TestHfWorkerRegistersThemBeforeTorch:
    def test_registered_before_the_worker_is_built_or_loaded(self, monkeypatch):
        """The real worker imports torch in HFWorker.load(), so the directories
        must already be registered by the time the worker exists."""
        from localm.inference.backends import _hf_runner, _hf_worker

        monkeypatch.setattr(_mp_spawn, "install_parent_death_watchdog", lambda *a: None)
        monkeypatch.setattr(_mp_spawn, "suppress_native_error_dialogs", lambda *a: None)
        events = []
        monkeypatch.setattr(_mp_spawn, "add_venv_dll_directories",
                            lambda: events.append("dll directories") or [])

        class _FakeWorker:
            supports_images = False
            can_embed = False
            resolved_device = "xpu"
            context_capacity = None

            def __init__(self, **kw):
                events.append("worker built")

            def load(self):
                events.append("model loaded")

            def unload(self):
                pass

        monkeypatch.setattr(_hf_worker, "HFWorker", _FakeWorker)

        req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
        t = threading.Thread(target=_hf_runner._runner_main,
                             args=(req_q, resp_q, ctrl_q),
                             name="hf-dispatch-under-test", daemon=True)
        t.start()
        try:
            req_q.put(("load", {}))
            assert resp_q.get(timeout=5)[0] == "ok"
        finally:
            req_q.put(None)
            ctrl_q.put(None)
            t.join(timeout=5)

        assert events == ["dll directories", "worker built", "model loaded"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL search path")
class TestRealSpawnWorkerFindsTheVenvsDlls:
    """A REAL spawn worker, started the way every localm worker is
    (``ensure_spawn_uses_venv_python``) from a real venv interpreter, loading a
    DLL that exists only in that venv's Library/bin by bare name - the way
    torch's c10_xpu.dll resolves the oneAPI runtime's sycl*.dll."""

    @staticmethod
    def _probe_dll(base_dir: Path) -> "Path | None":
        """A DLL CPython itself ships, needing nothing beyond the C runtime, to
        copy under a name nothing else on the search path has."""
        for name in ("libffi-8.dll", "sqlite3.dll"):
            candidate = base_dir / "DLLs" / name
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _build_fake_venv(tmp_path: Path, real_base: Path) -> "tuple[Path, Path]":
        """A venv (pyvenv.cfg pointing at the REAL base interpreter's directory)
        whose interpreter is a copy of that base interpreter one level under it,
        the layout of applaunch.py's ``<venv>/localm-app/LocaLM.exe``. Returns
        (venv root, its interpreter)."""
        venv = tmp_path / "fakevenv"
        (venv / "Lib" / "site-packages").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(
            f"home = {real_base.parent}\n", encoding="utf-8")
        exe_dir = venv / "localm-app"
        exe_dir.mkdir()
        exe = exe_dir / "FakeLauncher.exe"
        shutil.copy2(real_base, exe)
        for pattern in ("python3*.dll", "vcruntime*.dll"):
            for dll in real_base.parent.glob(pattern):
                shutil.copy2(dll, exe_dir / dll.name)
        return venv, exe

    def test_worker_loads_a_dll_from_the_venvs_library_bin(self, tmp_path):
        real_base = Path(getattr(sys, "_base_executable", None)
                         or sys.executable).resolve()
        probe = self._probe_dll(real_base.parent)
        if probe is None:
            pytest.skip("this Python install ships no DLL to copy")
        venv, exe = self._build_fake_venv(tmp_path, real_base)
        dll_name = "localm_issue1989_probe.dll"
        (venv / "Library" / "bin").mkdir(parents=True)
        shutil.copy2(probe, venv / "Library" / "bin" / dll_name)

        repo_root = Path(__file__).resolve().parents[1]
        script = tmp_path / "spawn_probe.py"
        script.write_text(textwrap.dedent(f"""\
            import ctypes
            import json
            import multiprocessing as mp
            import sys
            sys.path.insert(0, {str(repo_root)!r})
            from localm._mp_spawn import (add_venv_dll_directories,
                                          ensure_spawn_uses_venv_python)

            def try_load():
                try:
                    ctypes.CDLL({dll_name!r})
                except OSError as e:
                    return "failed: " + str(e)
                return "loaded"

            def worker(resp_q):
                before = try_load()
                added = add_venv_dll_directories()
                resp_q.put({{"exec_prefix": sys.exec_prefix, "before": before,
                            "added": added, "after": try_load()}})

            if __name__ == "__main__":
                ensure_spawn_uses_venv_python()
                ctx = mp.get_context("spawn")
                resp_q = ctx.Queue()
                proc = ctx.Process(target=worker, args=(resp_q,), daemon=True)
                proc.start()
                print("RESULT:" + json.dumps(resp_q.get(timeout=30)))
                proc.join(10)
        """), encoding="utf-8")

        # Nothing inherited may make the copied interpreter think it is some
        # other venv, or put another tree on its path.
        env = {k: v for k, v in os.environ.items()
               if k.upper() not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH")}
        result = subprocess.run([str(exe), str(script)], capture_output=True,
                                text=True, timeout=60, env=env)
        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        lines = [ln for ln in result.stdout.splitlines() if ln.startswith("RESULT:")]
        assert lines, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        report = json.loads(lines[0][len("RESULT:"):])

        def norm(p) -> str:
            return os.path.normcase(os.path.realpath(p))

        # The premise: the worker's exec_prefix, which torch derives Library/bin
        # from, is not the venv the worker serves.
        assert norm(report["exec_prefix"]) != norm(venv), report
        assert report["before"].startswith("failed"), report
        assert [norm(p) for p in report["added"]] == [norm(venv / "Library" / "bin")], report
        assert report["after"] == "loaded", report
