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

    # caplog must name the "localm" logger, not just set a level. caplog's own
    # level lands on the ROOT logger, and discover logs through the "localm"
    # one, whose level a sibling test can leave above DEBUG (and whose
    # isEnabledFor answer is memoised in Logger._cache until setLevel clears
    # it - see test_deferred_import_log's fixture docstring). Without the
    # logger= argument these assertions pass standalone and fail in a full-suite
    # run, which is exactly how they first failed.
    _LOGGER = "localm"

    def _run(self, monkeypatch, **kw):
        monkeypatch.setattr(subprocess, "run", MagicMock(**kw))
        return discover._torch_gpus_isolated()

    def test_timeout_raises_wedged_not_a_plain_cannot_ask(self, monkeypatch, caplog):
        """A timeout means TORCH is wedging, which must never be retried
        in-process. It is signalled distinctly for that reason."""
        caplog.set_level("DEBUG", logger=self._LOGGER)
        monkeypatch.setattr(subprocess, "run", MagicMock(
            side_effect=subprocess.TimeoutExpired("py", 10.0)))
        with pytest.raises(discover._IsolatedTorchWedged):
            discover._torch_gpus_isolated()
        assert "did not answer" in caplog.text

    def test_spawn_failure_falls_through_and_is_logged(self, monkeypatch, caplog):
        caplog.set_level("DEBUG", logger=self._LOGGER)
        out = self._run(monkeypatch, side_effect=OSError("no interpreter"))
        assert out is None
        assert "could not spawn" in caplog.text

    def test_malformed_reply_is_rejected(self, monkeypatch, caplog):
        caplog.set_level("DEBUG", logger=self._LOGGER)
        out = self._run(monkeypatch, return_value=MagicMock(
            stdout='[{"index": "zero", "total": 1, "free": 1}]', stderr=""))
        assert out is None
        assert "unusable" in caplog.text

    def test_non_json_reply_is_rejected(self, monkeypatch, caplog):
        caplog.set_level("DEBUG", logger=self._LOGGER)
        out = self._run(monkeypatch, return_value=MagicMock(
            stdout="Traceback (most recent call last):", stderr="boom"))
        assert out is None
        assert "unusable" in caplog.text

    def test_a_child_that_printed_nothing_is_unavailable_not_empty(
            self, monkeypatch, caplog):
        """The child always prints one line, "[]" included on its own failure
        path. Empty stdout therefore means it DIED before printing, which is
        could-not-ask. Calling that "no device" would report "no GPU" on a box
        whose GPU torch can see perfectly well."""
        caplog.set_level("DEBUG", logger=self._LOGGER)
        out = self._run(monkeypatch,
                        return_value=MagicMock(stdout="", stderr="", returncode=-9))
        assert out is None, "a child that printed nothing was read as 'no device'"
        assert "printed nothing" in caplog.text

    def test_a_child_that_answered_empty_IS_empty(self, monkeypatch):
        """The other side of the same line: an explicit "[]" is a real answer
        (torch imported, no CUDA/HIP device) and must not be read as a failure."""
        assert self._run(monkeypatch, return_value=MagicMock(
            stdout="[]", stderr="", returncode=0)) == []


class TestWedgedTorchIsNotRetriedForever:
    """`list_gpus` re-probes on every call by design (no TTL). So a box whose
    torch cannot answer must not pay the full timeout every single probe, or it
    never reaches the nvidia-smi fallback inside the caller's 15s deadline - the
    sm_120 case in the report this came from."""

    def setup_method(self):
        discover._reset_gpu_probe_cache()

    def teardown_method(self):
        discover._reset_gpu_probe_cache()

    def test_a_cannot_answer_latches_and_the_child_is_not_respawned(self, monkeypatch):
        spawns = []

        def _fail():
            spawns.append(1)
            raise discover._IsolatedTorchWedged()

        monkeypatch.setattr(discover, "_torch_gpus_isolated", _fail)
        assert discover._torch_gpus_isolated_once() == []
        assert discover._torch_gpus_isolated_once() == []
        assert discover._torch_gpus_isolated_once() == []
        assert len(spawns) == 1, (
            f"respawned a known-unanswerable child {len(spawns)} times - that is "
            "the full timeout on every probe, forever")

    def test_wedged_torch_is_never_retried_in_process(self, monkeypatch):
        """The in-process import IS the multi-minute hang. A wedged torch must
        fall through to nvidia-smi, never back to importing here."""
        monkeypatch.setattr(discover, "_torch_gpus_isolated",
                            MagicMock(side_effect=discover._IsolatedTorchWedged))
        monkeypatch.setattr(
            discover, "_torch_gpus_resident",
            lambda: pytest.fail("retried a WEDGED torch in-process - that is the "
                                "73s startup hang coming straight back"))
        assert discover._torch_gpus_isolated_once() == []

    def test_broken_isolation_degrades_in_process_rather_than_losing_the_gpu(
            self, monkeypatch, caplog):
        """Cannot-spawn tells us nothing about torch. Falling through to
        nvidia-smi would report "no GPU" on every AMD and Intel box."""
        caplog.set_level("WARNING", logger="localm")
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: None)
        monkeypatch.setattr(discover, "_torch_gpus_resident",
                            lambda: [{"index": 0, "name": "Radeon",
                                      "total": 8, "free": 4}])
        out = discover._torch_gpus_isolated_once()
        assert out and out[0]["name"] == "Radeon", (
            "broken isolation silently lost a real GPU that torch could see")
        assert "falling back to importing torch" in caplog.text, (
            "degraded silently - rule 5 requires saying so")

    def test_the_degrade_warning_is_emitted_once_not_per_probe(
            self, monkeypatch, caplog):
        """The live VRAM meter drives a probe about every 2.5s. An unconditional
        warning here would emit ~24 lines a minute for the life of the server."""
        caplog.set_level("WARNING", logger="localm")
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: None)
        monkeypatch.setattr(discover, "_torch_gpus_resident", lambda: [])
        for _ in range(5):
            discover._torch_gpus_isolated_once()
        warnings = [r for r in caplog.records
                    if "could not run the isolated GPU probe" in r.message]
        assert len(warnings) == 1, (
            f"emitted the degrade warning {len(warnings)} times across 5 probes")

    def test_broken_isolation_does_NOT_latch(self, monkeypatch):
        """Latching on a cannot-spawn would disable the torch path permanently
        on a box where torch works fine."""
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: None)
        monkeypatch.setattr(discover, "_torch_gpus_resident", lambda: [])
        discover._torch_gpus_isolated_once()
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unavailable is False

    def test_a_real_empty_answer_does_NOT_latch(self, monkeypatch):
        """[] means torch answered and sees no device. That is not a failure, and
        latching on it would disable the torch path on a box where it works."""
        spawns = []
        monkeypatch.setattr(discover, "_torch_gpus_isolated",
                            lambda: spawns.append(1) or [])
        discover._torch_gpus_isolated_once()
        discover._torch_gpus_isolated_once()
        assert len(spawns) == 2, "an honest [] answer must not disable the probe"

    def test_the_latch_is_cleared_by_the_probe_cache_reset(self, monkeypatch):
        monkeypatch.setattr(discover, "_torch_gpus_isolated",
                            MagicMock(side_effect=discover._IsolatedTorchWedged))
        discover._torch_gpus_isolated_once()
        discover._reset_gpu_probe_cache()
        spawns = []
        monkeypatch.setattr(discover, "_torch_gpus_isolated",
                            lambda: spawns.append(1) or [{"index": 0, "name": "G",
                                                          "total": 8, "free": 4}])
        assert discover._torch_gpus_isolated_once()[0]["name"] == "G"
        assert spawns == [1], "reset must re-enable the torch path"

    def test_the_timeout_fits_inside_the_caller_deadline(self):
        """A ceiling ABOVE the caller's deadline means a wedged box never reaches
        the fallback within the caller's window at all."""
        assert (discover._ISOLATED_TORCH_PROBE_TIMEOUT + 5.0
                <= discover._GPU_PROBE_DEADLINE), (
            "the child timeout plus nvidia-smi's own timeout=5 must fit inside "
            "_GPU_PROBE_DEADLINE")

    def test_child_failure_cause_reaches_the_log(self, monkeypatch, caplog):
        """The child prints its cause to stderr before answering []; that reason
        must not die with the discarded stream."""
        caplog.set_level("DEBUG", logger=self._LOGGER)
        out = self._run(monkeypatch, return_value=MagicMock(
            stdout="[]", stderr="torch GPU probe failed: OSError: WinError 126"))
        assert out == [], "the child ANSWERED (with []); that is not a cannot-ask"
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
