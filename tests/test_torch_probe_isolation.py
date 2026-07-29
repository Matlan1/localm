"""The cold torch GPU probe must run OUT of process (issue #833).

A cold ``import torch`` on Windows runs a loop of ``LoadLibraryExW`` calls, which
holds the OS loader lock; creating a thread needs that same lock, so no thread
anywhere in the process can start while it runs, and the asyncio event loop
stalls. A report's watchdog dump caught the request thread and a
``subprocess.run`` both parked at the last line of ``Thread.start()`` while the
GPU probe was inside torch's DLL loading, for 10.9s.

The regression guard that matters is structural rather than timed: after a probe
on a process where torch was NOT already resident, torch must still not be
resident, because the enumeration happened in a child. A wall-clock assertion
would be flaky on a loaded runner; this one is deterministic.
"""
from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from localm import discover


class TestColdProbeStaysOutOfProcess:

    def test_cold_probe_uses_the_child_and_never_imports_torch_here(self, monkeypatch):
        """The whole point of #833: a cold probe must not import torch here."""
        calls = []
        monkeypatch.setattr(discover, "_torch_is_resident", lambda: False)
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        monkeypatch.setattr(discover, "_apply_device_global_free", lambda gpus: None)
        monkeypatch.setattr(
            discover, "_torch_gpus_resident",
            lambda: pytest.fail("cold probe imported torch IN THIS PROCESS - "
                                "that is the #833 event-loop freeze"))
        monkeypatch.setattr(
            discover, "_torch_gpus_isolated",
            lambda: calls.append("isolated") or [
                {"index": 0, "name": "GPU 0", "total": 8, "free": 4}])

        out = discover._list_gpus_probe()

        assert calls == ["isolated"]
        assert out == [{"index": 0, "name": "GPU 0", "total": 8, "free": 4}]

    def test_resident_torch_stays_in_process(self, monkeypatch):
        """A resident torch is a free sys.modules cache hit that takes no loader
        lock, so it must NOT pay a process spawn."""
        monkeypatch.setattr(discover, "_torch_is_resident", lambda: True)
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        monkeypatch.setattr(discover, "_apply_device_global_free", lambda gpus: None)
        monkeypatch.setattr(
            discover, "_torch_gpus_isolated",
            lambda: pytest.fail("spawned a child for an already-resident torch"))
        monkeypatch.setattr(
            discover, "_torch_gpus_resident",
            lambda: [{"index": 0, "name": "GPU 0", "total": 8, "free": 4}])

        assert discover._list_gpus_probe()[0]["name"] == "GPU 0"

    def test_residency_is_read_from_sys_modules(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        assert discover._torch_is_resident() is False
        monkeypatch.setitem(sys.modules, "torch", MagicMock())
        assert discover._torch_is_resident() is True


class TestIsolatedProbeDegradesHonestly:
    """Every failure mode must return [] so the caller falls through to
    nvidia-smi exactly as an in-process failure used to, and must say why at
    debug rather than leaving "no GPU" indistinguishable from "could not ask"
    (AGENTS.md rule 5)."""

    def _run(self, monkeypatch, **kw):
        monkeypatch.setattr(subprocess, "run", MagicMock(**kw))
        return discover._torch_gpus_isolated()

    def test_timeout_falls_through_and_is_logged(self, monkeypatch, caplog):
        caplog.set_level("DEBUG")
        out = self._run(monkeypatch,
                        side_effect=subprocess.TimeoutExpired("py", 20.0))
        assert out == []
        assert "did not answer" in caplog.text

    def test_spawn_failure_falls_through_and_is_logged(self, monkeypatch, caplog):
        caplog.set_level("DEBUG")
        out = self._run(monkeypatch, side_effect=OSError("no interpreter"))
        assert out == []
        assert "could not spawn" in caplog.text

    def test_malformed_reply_is_rejected(self, monkeypatch, caplog):
        caplog.set_level("DEBUG")
        out = self._run(monkeypatch, return_value=MagicMock(
            stdout='[{"index": "zero", "total": 1, "free": 1}]', stderr=""))
        assert out == []
        assert "unusable" in caplog.text

    def test_non_json_reply_is_rejected(self, monkeypatch, caplog):
        caplog.set_level("DEBUG")
        out = self._run(monkeypatch, return_value=MagicMock(
            stdout="Traceback (most recent call last):", stderr="boom"))
        assert out == []
        assert "unusable" in caplog.text

    def test_empty_reply_is_an_empty_list_not_a_crash(self, monkeypatch):
        assert self._run(monkeypatch,
                         return_value=MagicMock(stdout="", stderr="")) == []

    def test_child_failure_cause_reaches_the_log(self, monkeypatch, caplog):
        """The child prints its cause to stderr before answering []; that reason
        must not die with the discarded stream."""
        caplog.set_level("DEBUG")
        out = self._run(monkeypatch, return_value=MagicMock(
            stdout="[]", stderr="torch GPU probe failed: OSError: WinError 126"))
        assert out == []
        assert "WinError 126" in caplog.text

    def test_good_reply_is_passed_through(self, monkeypatch):
        payload = [{"index": 0, "name": "RTX", "total": 8, "free": 4}]
        assert self._run(monkeypatch, return_value=MagicMock(
            stdout=json.dumps(payload), stderr="")) == payload

    def test_child_is_spawned_via_the_localm_interpreter_resolver(self, monkeypatch):
        """Bare sys.executable is the BASE interpreter inside a Windows
        multiprocessing-spawn worker, whose children cannot import localm or
        torch at all."""
        fake = MagicMock(return_value=MagicMock(stdout="[]", stderr=""))
        monkeypatch.setattr(subprocess, "run", fake)
        monkeypatch.setattr("localm._mp_spawn.interpreter_for_localm_children",
                            lambda: "SENTINEL-PY")
        discover._torch_gpus_isolated()
        argv = fake.call_args[0][0]
        assert argv[0] == "SENTINEL-PY"
        assert argv[-2:] == ["-m", "localm._torch_gpu_probe"]
        assert fake.call_args.kwargs["timeout"] == \
            discover._ISOLATED_TORCH_PROBE_TIMEOUT


class TestChildProbeContract:
    """The child must mirror the in-process branch field for field, so a caller
    cannot tell which path produced a reading."""

    def test_child_emits_one_json_line_and_nothing_else(self):
        torch = pytest.importorskip("torch")
        from localm._mp_spawn import interpreter_for_localm_children
        proc = subprocess.run(
            [interpreter_for_localm_children(), "-u", "-m",
             "localm._torch_gpu_probe"],
            capture_output=True, text=True, timeout=120)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        assert len(lines) == 1, f"child printed {len(lines)} stdout lines"
        devices = json.loads(lines[0])
        assert isinstance(devices, list)
        for d in devices:
            assert set(d) == {"index", "name", "total", "free"}
            assert isinstance(d["index"], int) and isinstance(d["name"], str)
            assert d["total"] > 0 and d["free"] >= 0
        if torch.cuda.is_available():
            assert len(devices) == torch.cuda.device_count()

    def test_child_reports_a_torch_failure_rather_than_dying_silently(self):
        """An unimportable torch must still produce a parseable [] plus a cause
        on stderr, never an empty stdout the parent cannot distinguish from a
        hang."""
        from localm._mp_spawn import interpreter_for_localm_children
        proc = subprocess.run(
            [interpreter_for_localm_children(), "-u", "-c",
             "import sys; sys.modules['torch'] = None; "
             "import runpy; runpy.run_module('localm._torch_gpu_probe', "
             "run_name='__main__')"],
            capture_output=True, text=True, timeout=120)
        assert proc.stdout.strip() == "[]"
        assert "torch GPU probe failed" in proc.stderr
