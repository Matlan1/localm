# SPDX-License-Identifier: AGPL-3.0-or-later
"""Privacy mode must not arm a native-fault crash trace for the isolated
model-worker processes (GGUF, HF, embedder).

Each runner's ``_spawn()`` used to call ``debuglog.child_crash_trace_path()``
unconditionally on every worker start, in every session mode - the crash
guard's own trace (bugreport.py) and the server's hang alarm are already
gated on ``audit.diagnostics_allowed()`` (log/full mode, or privacy with
``keep_diagnostics`` on); these three call sites were the ones still missed.

Mocks the ``multiprocessing.get_context`` seam (same technique as
``test_doctor_worker_spawn.py``) so ``_spawn()`` runs for real without an
actual child process ever starting - the gate under test is entirely
parent-side, decided before ``ctx.Process(...)`` is even constructed.
"""

from __future__ import annotations

import queue as _queue


class _FakeProcess:
    def __init__(self, target=None, args=(), name=None, daemon=None):
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.exitcode = None

    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return False

    def join(self, timeout=None) -> None:
        pass

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class _FakeContext:
    """Stand-in for ``multiprocessing.get_context("spawn")``: real queues (so
    parent-side code that touches them does not blow up), but a Process that
    never actually starts a child - nothing this test does should spawn a
    real OS process or run any child-side code."""

    def __init__(self) -> None:
        self.processes: list[_FakeProcess] = []

    def Queue(self):
        return _queue.Queue()

    def Process(self, **kwargs) -> _FakeProcess:
        p = _FakeProcess(**kwargs)
        self.processes.append(p)
        return p


def _set_mode(monkeypatch, mode: str, keep_diagnostics: bool = False) -> None:
    from localm import config as config_mod
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.delenv("LOCALM_KEEP_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(config_mod, "load_config",
                        lambda: {"mode": mode, "keep_diagnostics": keep_diagnostics})


def _no_crash_trace_files_on_disk() -> bool:
    from localm.debuglog import logs_dir
    d = logs_dir()
    return not d.exists() or not list(d.glob("crash_*.txt"))


# --------------------------------------------------------------------------- #
# GGUF backend (llamacpp/_runner.py's ModelRunner)
# --------------------------------------------------------------------------- #

class TestGgufRunnerCrashTracePrivacy:
    def _spawn(self, monkeypatch):
        from localm.inference.backends.llamacpp import _runner as runner_mod
        ctx = _FakeContext()
        monkeypatch.setattr(runner_mod.mp, "get_context", lambda name: ctx)
        r = runner_mod.ModelRunner()
        r._spawn()
        return r, ctx

    def test_privacy_mode_mints_no_trace_path(self, monkeypatch):
        _set_mode(monkeypatch, "privacy")
        r, ctx = self._spawn(monkeypatch)
        assert r._crash_trace_path is None
        assert ctx.processes[0].args[-1] is None
        assert _no_crash_trace_files_on_disk()

    def test_privacy_mode_with_keep_diagnostics_still_mints(self, monkeypatch):
        _set_mode(monkeypatch, "privacy", keep_diagnostics=True)
        r, ctx = self._spawn(monkeypatch)
        assert r._crash_trace_path is not None
        assert ctx.processes[0].args[-1] == r._crash_trace_path

    def test_log_mode_mints_a_trace_path(self, monkeypatch):
        _set_mode(monkeypatch, "log")
        r, ctx = self._spawn(monkeypatch)
        assert r._crash_trace_path is not None
        assert r._crash_trace_path.name.startswith("crash_gguf-worker_")
        assert ctx.processes[0].args[-1] == r._crash_trace_path


# --------------------------------------------------------------------------- #
# HF backend (backends/_hf_runner.py's HFRunner)
# --------------------------------------------------------------------------- #

class TestHfRunnerCrashTracePrivacy:
    def _spawn(self, monkeypatch):
        from localm.inference.backends import _hf_runner as runner_mod
        ctx = _FakeContext()
        monkeypatch.setattr(runner_mod.mp, "get_context", lambda name: ctx)
        r = runner_mod.HFRunner()
        r._spawn()
        return r, ctx

    def test_privacy_mode_mints_no_trace_path(self, monkeypatch):
        _set_mode(monkeypatch, "privacy")
        r, ctx = self._spawn(monkeypatch)
        assert r._crash_trace_path is None
        assert ctx.processes[0].args[-1] is None
        assert _no_crash_trace_files_on_disk()

    def test_privacy_mode_with_keep_diagnostics_still_mints(self, monkeypatch):
        _set_mode(monkeypatch, "privacy", keep_diagnostics=True)
        r, ctx = self._spawn(monkeypatch)
        assert r._crash_trace_path is not None
        assert ctx.processes[0].args[-1] == r._crash_trace_path

    def test_log_mode_mints_a_trace_path(self, monkeypatch):
        _set_mode(monkeypatch, "log")
        r, ctx = self._spawn(monkeypatch)
        assert r._crash_trace_path is not None
        assert r._crash_trace_path.name.startswith("crash_hf-worker_")
        assert ctx.processes[0].args[-1] == r._crash_trace_path


# --------------------------------------------------------------------------- #
# Embedder (_embedder_runner.py's EmbedderRunner)
# --------------------------------------------------------------------------- #

class TestEmbedderRunnerCrashTracePrivacy:
    def _spawn(self, monkeypatch):
        from localm.inference import _embedder_runner as runner_mod
        ctx = _FakeContext()
        monkeypatch.setattr(runner_mod.mp, "get_context", lambda name: ctx)
        r = runner_mod.EmbedderRunner()
        r._spawn()
        return r, ctx

    def test_privacy_mode_mints_no_trace_path(self, monkeypatch):
        _set_mode(monkeypatch, "privacy")
        r, ctx = self._spawn(monkeypatch)
        assert r._crash_trace_path is None
        assert ctx.processes[0].args[-1] is None
        assert _no_crash_trace_files_on_disk()

    def test_privacy_mode_with_keep_diagnostics_still_mints(self, monkeypatch):
        _set_mode(monkeypatch, "privacy", keep_diagnostics=True)
        r, ctx = self._spawn(monkeypatch)
        assert r._crash_trace_path is not None
        assert ctx.processes[0].args[-1] == r._crash_trace_path

    def test_log_mode_mints_a_trace_path(self, monkeypatch):
        _set_mode(monkeypatch, "log")
        r, ctx = self._spawn(monkeypatch)
        assert r._crash_trace_path is not None
        assert r._crash_trace_path.name.startswith("crash_embedder-worker_")
        assert ctx.processes[0].args[-1] == r._crash_trace_path
