"""The cold torch GPU probe must run OUT of process.

A cold ``import torch`` on Windows runs a loop of ``LoadLibraryExW`` calls, which
holds the OS loader lock; creating a thread needs that same lock, so no thread
anywhere in the process can start while it runs, and the asyncio event loop
stalls.

The regression guard is structural rather than timed: after a probe on a process
where torch was NOT already resident, torch must still not be resident, because
the enumeration happened in a child.
"""
from __future__ import annotations

import json
import logging
import pathlib
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from localm import discover
from tests._sysprog_fakes import _nvidia_smi_found  # noqa: F401


class TestColdProbeStaysOutOfProcess:

    def test_cold_probe_uses_the_child_and_never_imports_torch_here(self, monkeypatch):
        """A cold probe must not import torch here."""
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
    nvidia-smi, and must say why at debug, so "no GPU" is never
    indistinguishable from "could not ask"."""

    # caplog must name the "localm" logger, not just set a level. caplog's own
    # level lands on the ROOT logger, and discover logs through the "localm"
    # one, whose level a sibling test can leave above DEBUG and whose
    # isEnabledFor answer is memoised in Logger._cache until setLevel clears it.
    _LOGGER = "localm"

    def _run(self, monkeypatch, **kw):
        """kw mirrors the old subprocess.run(**kw) contract these tests were
        written against (return_value=<obj with .stdout/.stderr/.returncode>
        or side_effect=<exception>), adapted to the Popen()+communicate()
        split _torch_gpus_isolated now uses: an OSError-family side_effect
        fires from Popen() itself (a spawn failure); anything else (e.g.
        TimeoutExpired) fires from the first communicate() call, with the
        post-kill drain communicate() then returning empty output."""
        fake_popen = MagicMock()
        side_effect = kw.get("side_effect")
        result = kw.get("return_value")
        if isinstance(side_effect, OSError):
            fake_popen.side_effect = side_effect
        elif side_effect is not None:
            fake_popen.return_value.communicate.side_effect = [side_effect, ("", "")]
        else:
            fake_popen.return_value.communicate.return_value = (result.stdout, result.stderr)
            fake_popen.return_value.returncode = result.returncode
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        return discover._torch_gpus_isolated()

    def test_timeout_raises_wedged_not_a_plain_cannot_ask(self, monkeypatch, caplog):
        """A timeout means TORCH is wedging, which must never be retried
        in-process. It is signalled distinctly for that reason."""
        caplog.set_level("DEBUG", logger=self._LOGGER)
        fake_popen = MagicMock()
        fake_popen.return_value.communicate.side_effect = [
            subprocess.TimeoutExpired("py", 10.0), ("", "")]
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        with pytest.raises(discover._IsolatedTorchWedged):
            discover._torch_gpus_isolated()
        assert "did not answer" in caplog.text

    def test_a_wedged_post_kill_drain_is_bounded_not_left_open_ended(
            self, monkeypatch, caplog):
        """subprocess.run's own Windows kill-path drains the pipes with a
        SECOND communicate() call that carries no timeout of its own (verified
        against CPython's subprocess.run source: its comment reads
        "communicate() _after_ kill() is required to collect that"). If a
        grandchild the probe spawned inherited the pipe handle and stays
        alive, that drain never sees EOF and blocks forever - which would
        wedge the shared single-flight GPU-probe lock (_gpu_probe_inflight)
        for the rest of the process's life, well past this call returning.
        The drain here must be bounded by its own timeout and give up rather
        than risk an unbounded wait."""
        caplog.set_level("DEBUG", logger=self._LOGGER)
        fake_popen = MagicMock()
        fake_popen.return_value.communicate.side_effect = [
            subprocess.TimeoutExpired("py", 10.0),   # the probe itself wedges
            subprocess.TimeoutExpired("py", 3.0),    # the post-kill drain ALSO wedges
        ]
        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        with pytest.raises(discover._IsolatedTorchWedged):
            discover._torch_gpus_isolated()

        fake_popen.return_value.kill.assert_called_once()
        calls = fake_popen.return_value.communicate.call_args_list
        assert len(calls) == 2, (
            "must attempt exactly one bounded drain after kill(), not retry "
            "unboundedly")
        assert calls[1].kwargs.get("timeout") is not None, (
            "the post-kill drain communicate() call has no timeout of its "
            "own - a wedged grandchild holding the pipe open can block this "
            "forever, exactly the CPython subprocess.run Windows bug this "
            "test guards against")

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
    def test_child_failure_cause_reaches_the_log(self, monkeypatch, caplog):
        """The child prints its cause to stderr before answering []; that reason
        must not die with the discarded stream."""
        caplog.set_level("DEBUG", logger=self._LOGGER)
        out = self._run(monkeypatch, return_value=MagicMock(
            stdout="[]", stderr="torch GPU probe failed: OSError: WinError 126"))
        assert out == [], "the child ANSWERED (with []); that is not a cannot-ask"
        assert "WinError 126" in caplog.text

    # A short synthetic stderr cannot exercise truncation. This one is 456 chars:
    # a 180-char virtualenv install-path warnings.warn() prefix ahead of the
    # actionable GPU-architecture-list message, so a cap applied to the head
    # drops everything a user would need to act on.
    _LONG_PATH_PREFIX = (
        "/opt/pyenv/versions/3.12.4/lib/python3.12/site-packages/torch/cuda/"
        "__init__.py:422: UserWarning: Found GPU0 NVIDIA RTX PRO 4000 Blackwell "
        "which is of compute capability (CC) 12.0.\n"
    )
    _ACTIONABLE_TAIL = (
        "The following list of GPU architectures compatible with this version "
        "of PyTorch is: sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.\n"
        "If you want to use the RTX PRO 4000 Blackwell GPU with PyTorch, please "
        "check the instructions at https://pytorch.org/get-started/locally/"
    )

    def test_long_child_stderr_keeps_the_actionable_tail(self, monkeypatch, caplog):
        """The whole point of the message - which GPU architectures are
        supported - must survive a realistically long path prefix, not just
        the boilerplate ahead of it."""
        caplog.set_level("DEBUG", logger=self._LOGGER)
        stderr = self._LONG_PATH_PREFIX + self._ACTIONABLE_TAIL
        assert len(stderr) > 200, "fixture must exceed the old cap to prove anything"
        out = self._run(monkeypatch, return_value=MagicMock(stdout="[]", stderr=stderr))
        assert out == []
        assert "sm_90" in caplog.text and "get-started/locally" in caplog.text, (
            "truncated the child's stderr before its actionable content survived: "
            + caplog.text)

    def test_stderr_beyond_the_cap_says_so_rather_than_cutting_silently(
            self, monkeypatch, caplog):
        """A cap must still exist (an adversarial/huge child stream cannot be
        logged unbounded), but a truncated diagnostic must SAY it was
        truncated, never stop mid-sentence with no indication."""
        caplog.set_level("DEBUG", logger=self._LOGGER)
        huge = self._LONG_PATH_PREFIX + self._ACTIONABLE_TAIL * 20
        assert len(huge) > discover._CHILD_STDERR_LOG_CAP
        out = self._run(monkeypatch, return_value=MagicMock(stdout="[]", stderr=huge))
        assert out == []
        assert "truncated" in caplog.text, (
            "cut a diagnostic without saying so: " + caplog.text)

    def test_good_reply_is_passed_through(self, monkeypatch):
        payload = [{"index": 0, "name": "RTX", "total": 8, "free": 4}]
        assert self._run(monkeypatch, return_value=MagicMock(
            stdout=json.dumps(payload), stderr="")) == payload

    def test_child_is_spawned_via_the_localm_interpreter_resolver(self, monkeypatch):
        """Bare sys.executable is the BASE interpreter inside a Windows
        multiprocessing-spawn worker, whose children cannot import localm or
        torch at all."""
        fake_popen = MagicMock()
        fake_popen.return_value.communicate.return_value = ("[]", "")
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr("localm._mp_spawn.interpreter_for_localm_children",
                            lambda: "SENTINEL-PY")
        discover._torch_gpus_isolated()
        argv = fake_popen.call_args[0][0]
        assert argv[0] == "SENTINEL-PY"
        assert argv[-2:] == ["-m", "localm._torch_gpu_probe"]
        assert fake_popen.return_value.communicate.call_args.kwargs["timeout"] == \
            discover._ISOLATED_TORCH_PROBE_TIMEOUT




class TestWedgedTorchIsNotRetriedForever:
    """`list_gpus` re-probes on every call (no TTL), so a box whose torch cannot
    answer must not pay the full timeout every single probe, or it never reaches
    the nvidia-smi fallback inside the caller's 15s deadline."""

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
        monkeypatch.setattr(discover, "_in_process_torch_import_can_freeze",
                            lambda: False)
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
        monkeypatch.setattr(discover, "_in_process_torch_import_can_freeze",
                            lambda: False)
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
        monkeypatch.setattr(discover, "_in_process_torch_import_can_freeze",
                            lambda: False)
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


class _DiscoverClock:
    """discover's ``time`` module with monotonic() reading ``now[0]``."""

    def __init__(self, now):
        self._now = now

    def monotonic(self):
        return self._now[0]

    def __getattr__(self, name):
        import time as real_time
        return getattr(real_time, name)


def _set_discover_clock(monkeypatch, start):
    now = [start]
    monkeypatch.setattr(discover, "time", _DiscoverClock(now))
    return now


def _after_the_retry_wait(monkeypatch):
    """Move the probe's clock past the isolated probe's retry wait."""
    with discover._gpu_probe_lock:
        at = discover._isolated_torch_retry_at
    _set_discover_clock(monkeypatch, at + 1.0)


class TestBrokenIsolationOnWindowsNeverImportsInProcess:
    """On Windows a cold ``import torch`` holds the OS loader lock and freezes the
    server, and a child that died printing nothing usually means torch's import
    itself faulted. A broken isolated probe there must not be retried by importing
    here; the non-torch sources answer instead."""

    _RADEON = {"index": 0, "name": "AMD Radeon RX 6900 XT", "total": 16 * 1024 ** 3}

    def setup_method(self):
        discover._reset_gpu_probe_cache()

    def teardown_method(self):
        discover._reset_gpu_probe_cache()

    @staticmethod
    def _windows_with_broken_isolation(monkeypatch):
        monkeypatch.setattr(discover, "_in_process_torch_import_can_freeze",
                            lambda: True)
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        monkeypatch.setattr(discover, "_torch_is_resident", lambda: False)
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: None)
        monkeypatch.setattr(
            discover, "_torch_gpus_resident",
            lambda: pytest.fail("imported torch IN THIS PROCESS on Windows after "
                                "the isolated probe failed"))

    @staticmethod
    def _no_nvidia_smi(monkeypatch):
        monkeypatch.setattr(subprocess, "Popen",
                            MagicMock(side_effect=FileNotFoundError("no nvidia-smi")))

    def test_returns_empty_without_importing_torch_here(self, monkeypatch):
        self._windows_with_broken_isolation(monkeypatch)
        assert discover._torch_gpus_isolated_once() == []
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unanswered is True

    def test_says_so_at_warning_exactly_once(self, monkeypatch, caplog):
        caplog.set_level("WARNING", logger="localm")
        self._windows_with_broken_isolation(monkeypatch)
        for _ in range(4):
            discover._torch_gpus_isolated_once()
        hits = [r for r in caplog.records
                if "could not run the isolated GPU probe" in r.message]
        assert len(hits) == 1
        assert "falling back to importing torch" not in hits[0].message, (
            "the warning claims a fallback import that no longer happens")

    def test_does_not_latch_a_round_after_the_wait_still_asks_the_child(
            self, monkeypatch):
        self._windows_with_broken_isolation(monkeypatch)
        discover._torch_gpus_isolated_once()
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unavailable is False
        _after_the_retry_wait(monkeypatch)
        monkeypatch.setattr(discover, "_torch_gpus_isolated",
                            lambda: [{"index": 0, "name": "G", "total": 8, "free": 4}])
        assert discover._torch_gpus_isolated_once()[0]["name"] == "G"

    def test_probe_falls_through_to_the_amd_registry_source(self, monkeypatch):
        self._windows_with_broken_isolation(monkeypatch)
        self._no_nvidia_smi(monkeypatch)
        monkeypatch.setattr(discover, "_windows_largest_adapter_registry_entry",
                            lambda: dict(self._RADEON))
        monkeypatch.setattr(
            discover, "_apply_device_global_free",
            lambda gpus: [g.update(free=g["total"] // 2) for g in gpus])
        out = discover._list_gpus_probe()
        assert [g["name"] for g in out] == ["AMD Radeon RX 6900 XT"]
        assert out[0]["free"] == self._RADEON["total"] // 2

    def test_probe_falls_through_to_nvidia_smi(self, monkeypatch):
        self._windows_with_broken_isolation(monkeypatch)
        fake_popen = MagicMock()
        fake_popen.return_value.communicate.return_value = (
            "0, RTX 4090, 24576, 20000\n", "")
        fake_popen.return_value.returncode = 0
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        out = discover._list_gpus_probe()
        assert out and out[0]["name"] == "RTX 4090"

    def test_empty_reading_is_inconclusive_not_no_gpu(self, monkeypatch):
        self._windows_with_broken_isolation(monkeypatch)
        self._no_nvidia_smi(monkeypatch)
        monkeypatch.setattr(discover, "_windows_largest_adapter_registry_entry",
                            lambda: None)
        gpus, status = discover.list_gpus(return_status=True)
        assert gpus == []
        assert status == discover.GPU_PROBE_INCONCLUSIVE, (
            "could not ask torch and nothing else answered: that is not 'no GPU'")

    def test_the_flag_is_per_round_a_good_round_clears_it(self, monkeypatch):
        self._windows_with_broken_isolation(monkeypatch)
        self._no_nvidia_smi(monkeypatch)
        monkeypatch.setattr(discover, "_windows_largest_adapter_registry_entry",
                            lambda: None)
        discover._list_gpus_probe()
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unanswered is True
        _after_the_retry_wait(monkeypatch)
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: [])
        discover._list_gpus_probe()
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unanswered is False

    def test_off_windows_keeps_the_in_process_degrade(self, monkeypatch):
        monkeypatch.setattr(discover, "_in_process_torch_import_can_freeze",
                            lambda: False)
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: None)
        monkeypatch.setattr(discover, "_torch_gpus_resident",
                            lambda: [{"index": 0, "name": "G", "total": 8, "free": 4}])
        assert discover._torch_gpus_isolated_once()[0]["name"] == "G"
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unanswered is False

    def test_the_freeze_guard_is_true_on_windows_only(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert discover._in_process_torch_import_can_freeze() is True
        for plat in ("linux", "darwin"):
            monkeypatch.setattr(sys, "platform", plat)
            assert discover._in_process_torch_import_can_freeze() is False


class TestBrokenIsolationIsNotRespawnedEveryRound:
    """While isolation stays broken the child is spawned again only after a wait
    that doubles per failure, and any answer ends the wait."""

    def setup_method(self):
        discover._reset_gpu_probe_cache()

    def teardown_method(self):
        discover._reset_gpu_probe_cache()

    @staticmethod
    def _clock(monkeypatch, start=1000.0):
        return _set_discover_clock(monkeypatch, start)

    @staticmethod
    def _broken(monkeypatch, *, windows=True):
        spawns = []

        def isolated():
            spawns.append(1)
            return None

        monkeypatch.setattr(discover, "_in_process_torch_import_can_freeze",
                            lambda: windows)
        monkeypatch.setattr(discover, "_torch_gpus_isolated", isolated)
        return spawns

    def test_rounds_inside_the_wait_spawn_no_child(self, monkeypatch):
        now = self._clock(monkeypatch)
        spawns = self._broken(monkeypatch)
        for _ in range(20):
            assert discover._torch_gpus_isolated_once() == []
            now[0] += 1.0
        assert spawns == [1], (
            f"the broken isolated probe was spawned {len(spawns)} times in 20 rounds")

    def test_a_round_inside_the_wait_is_reported_like_a_failed_one(self, monkeypatch):
        now = self._clock(monkeypatch)
        self._broken(monkeypatch)
        discover._torch_gpus_isolated_once()
        with discover._gpu_probe_lock:
            discover._isolated_torch_unanswered = False
        now[0] += 1.0
        assert discover._torch_gpus_isolated_once() == []
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unanswered is True
            assert discover._isolated_torch_unavailable is False

    def test_the_wait_doubles_up_to_its_cap(self, monkeypatch):
        now = self._clock(monkeypatch)
        spawns = self._broken(monkeypatch)
        waits = []
        for _ in range(8):
            discover._torch_gpus_isolated_once()
            with discover._gpu_probe_lock:
                waits.append(discover._isolated_torch_retry_wait)
                now[0] = discover._isolated_torch_retry_at
        first, cap = (discover._ISOLATED_TORCH_RETRY_FIRST,
                      discover._ISOLATED_TORCH_RETRY_MAX)
        assert waits[:3] == [first, 2 * first, 4 * first]
        assert max(waits) == cap and waits[-1] == cap
        assert len(spawns) == 8

    def test_an_answer_ends_the_wait(self, monkeypatch):
        now = self._clock(monkeypatch)
        spawns = self._broken(monkeypatch)
        discover._torch_gpus_isolated_once()
        with discover._gpu_probe_lock:
            now[0] = discover._isolated_torch_retry_at
        monkeypatch.setattr(discover, "_torch_gpus_isolated",
                            lambda: spawns.append(1) or [])
        assert discover._torch_gpus_isolated_once() == []
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_retry_wait == 0.0
            assert discover._isolated_torch_retry_at == 0.0
        discover._torch_gpus_isolated_once()
        assert len(spawns) == 3, "a round after an answer did not ask the child"

    def test_the_reset_clears_the_wait(self, monkeypatch):
        self._clock(monkeypatch)
        spawns = self._broken(monkeypatch)
        discover._torch_gpus_isolated_once()
        discover._reset_gpu_probe_cache()
        self._broken(monkeypatch)
        discover._torch_gpus_isolated_once()
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_retry_wait == (
                discover._ISOLATED_TORCH_RETRY_FIRST)
        assert spawns == [1]

    def test_off_windows_a_round_inside_the_wait_still_degrades_in_process(
            self, monkeypatch):
        now = self._clock(monkeypatch)
        spawns = self._broken(monkeypatch, windows=False)
        monkeypatch.setattr(discover, "_torch_gpus_resident",
                            lambda: [{"index": 0, "name": "G", "total": 8, "free": 4}])
        first = discover._torch_gpus_isolated_once()
        now[0] += 1.0
        second = discover._torch_gpus_isolated_once()
        assert first[0]["name"] == second[0]["name"] == "G"
        assert spawns == [1]

    def test_a_wedge_latches_and_waits_before_asking_again(self, monkeypatch):
        now = self._clock(monkeypatch)
        spawns = []

        def wedged():
            spawns.append(1)
            raise discover._IsolatedTorchWedged()

        monkeypatch.setattr(discover, "_torch_gpus_isolated", wedged)
        waits = []
        for _ in range(8):
            assert discover._torch_gpus_isolated_once() == []
            with discover._gpu_probe_lock:
                assert discover._isolated_torch_unavailable is True
                waits.append(discover._isolated_torch_retry_wait)
            now[0] += 1.0
            assert discover._torch_gpus_isolated_once() == []
            with discover._gpu_probe_lock:
                now[0] = discover._isolated_torch_retry_at
        first, cap = (discover._ISOLATED_TORCH_RETRY_FIRST,
                      discover._ISOLATED_TORCH_RETRY_MAX)
        assert waits[:3] == [first, 2 * first, 4 * first]
        assert waits[-1] == cap
        assert len(spawns) == 8, "a round inside the wait spawned the child"

    def test_an_answer_after_a_wedge_clears_the_latch(self, monkeypatch):
        now = self._clock(monkeypatch)
        monkeypatch.setattr(discover, "_torch_gpus_isolated",
                            MagicMock(side_effect=discover._IsolatedTorchWedged))
        discover._torch_gpus_isolated_once()
        assert discover.isolated_torch_unavailable() is True
        with discover._gpu_probe_lock:
            now[0] = discover._isolated_torch_retry_at
        answer = [{"index": 0, "name": "G", "total": 8, "free": 4}]
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: answer)
        assert discover._torch_gpus_isolated_once() == answer
        assert discover.isolated_torch_unavailable() is False
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_retry_wait == 0.0
            assert discover._isolated_torch_retry_at == 0.0

    def test_a_child_failure_after_a_wedge_never_imports_in_process(
            self, monkeypatch, caplog):
        caplog.set_level("WARNING", logger="localm")
        now = self._clock(monkeypatch)
        monkeypatch.setattr(discover, "_in_process_torch_import_can_freeze",
                            lambda: False)
        monkeypatch.setattr(discover, "_torch_gpus_isolated",
                            MagicMock(side_effect=discover._IsolatedTorchWedged))
        discover._torch_gpus_isolated_once()
        with discover._gpu_probe_lock:
            now[0] = discover._isolated_torch_retry_at
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: None)
        monkeypatch.setattr(
            discover, "_torch_gpus_resident",
            lambda: pytest.fail("imported a torch that timed out in-process"))
        assert discover._torch_gpus_isolated_once() == []
        assert discover.isolated_torch_unavailable() is True
        assert "could not run the isolated GPU probe" not in caplog.text


_PROBE_STUB_ANSWER = """
class _Cuda:
    @staticmethod
    def is_available():
        return True
    @staticmethod
    def device_count():
        return 1
    @staticmethod
    def mem_get_info(i):
        return (6 * 2**30, 8 * 2**30)
    @staticmethod
    def get_device_name(i):
        return "STUB GPU"
cuda = _Cuda()
"""

# First import: publishes the real pid and blocks. Every later import answers.
_PROBE_STUB_BLOCKS_ONCE = """
import os, pathlib, time
_marker = pathlib.Path(os.environ["PROBE_STUB_MARKER"])
if not _marker.exists():
    _tmp = _marker.with_suffix(".tmp")
    _tmp.write_text(str(os.getpid()), encoding="utf-8")
    os.replace(_tmp, _marker)
    time.sleep(300)
""" + _PROBE_STUB_ANSWER


def _stub_torch_env(monkeypatch, tmp_path, stub):
    """Make the real probe child import *stub* as torch. Returns the marker path
    :data:`_PROBE_STUB_BLOCKS_ONCE` publishes its pid at."""
    import os
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "torch.py").write_text(stub, encoding="utf-8")
    repo = str(pathlib.Path(discover.__file__).resolve().parent.parent)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub_dir), repo]))
    marker = tmp_path / "blocked-pid"
    monkeypatch.setenv("PROBE_STUB_MARKER", str(marker))
    return marker


def _wait_for_pid(marker, timeout=60.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            return int(marker.read_text(encoding="utf-8"))
        time.sleep(0.02)
    raise AssertionError("the stub torch never started importing in the child")


def _kill_from_outside(pid):
    """Terminate *pid* the way a process-tree kill does."""
    import os
    import signal
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True,
                       check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        os.kill(pid, signal.SIGKILL)


class TestProbeChildThatDiedIsAskedAgain:
    """A child that exited printing nothing says nothing about torch: a process-tree
    kill takes it down too. It is asked once more in the same round."""

    _WARNING = "could not run the isolated GPU probe"

    def setup_method(self):
        discover._reset_gpu_probe_cache()

    def teardown_method(self):
        discover._reset_gpu_probe_cache()

    def test_a_child_killed_from_outside_is_asked_again(
            self, monkeypatch, tmp_path, caplog):
        import threading
        caplog.set_level("DEBUG", logger="localm")
        marker = _stub_torch_env(monkeypatch, tmp_path, _PROBE_STUB_BLOCKS_ONCE)
        monkeypatch.setattr(discover, "_ISOLATED_TORCH_PROBE_TIMEOUT", 120.0)
        killer = threading.Thread(target=lambda: _kill_from_outside(_wait_for_pid(marker)))
        killer.start()
        try:
            devices = discover._torch_gpus_isolated_once()
        finally:
            killer.join(90)
        assert "printed nothing" in caplog.text, "the child was never killed"
        assert devices and devices[0]["name"] == "STUB GPU", devices
        assert self._WARNING not in caplog.text
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unanswered is False
            assert discover._isolated_torch_retry_at == 0.0

    def test_a_child_that_keeps_dying_is_still_reported(
            self, monkeypatch, tmp_path, caplog):
        caplog.set_level("DEBUG", logger="localm")
        monkeypatch.setattr(discover, "_in_process_torch_import_can_freeze",
                            lambda: True)
        _stub_torch_env(monkeypatch, tmp_path, "import os\nos._exit(1)\n")
        monkeypatch.setattr(discover, "_ISOLATED_TORCH_PROBE_TIMEOUT", 120.0)
        assert discover._torch_gpus_isolated_once() == []
        assert caplog.text.count("printed nothing") == 2
        warnings = [r for r in caplog.records if self._WARNING in r.getMessage()]
        assert len(warnings) == 1 and warnings[0].levelname == "WARNING"
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unanswered is True
            assert discover._isolated_torch_retry_at > 0.0

    @staticmethod
    def _dying_popen(monkeypatch, now, seconds_per_child):
        fake_popen = MagicMock()

        def communicate(timeout=None):
            now[0] += seconds_per_child
            return ("", "")

        fake_popen.return_value.communicate.side_effect = communicate
        fake_popen.return_value.returncode = 1
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        return fake_popen

    def test_the_second_ask_gets_only_the_time_that_is_left(self, monkeypatch):
        now = _set_discover_clock(monkeypatch, 1000.0)
        fake_popen = self._dying_popen(monkeypatch, now, 3.0)
        assert discover._torch_gpus_isolated() is None
        timeouts = [c.kwargs["timeout"]
                    for c in fake_popen.return_value.communicate.call_args_list]
        assert timeouts == [discover._ISOLATED_TORCH_PROBE_TIMEOUT,
                            discover._ISOLATED_TORCH_PROBE_TIMEOUT - 3.0]

    def test_no_second_ask_when_too_little_time_is_left(self, monkeypatch):
        now = _set_discover_clock(monkeypatch, 1000.0)
        fake_popen = self._dying_popen(
            monkeypatch, now, discover._ISOLATED_TORCH_PROBE_TIMEOUT
            - discover._ISOLATED_TORCH_REASK_MIN_LEFT + 0.1)
        assert discover._torch_gpus_isolated() is None
        assert fake_popen.call_count == 1

    def test_the_pause_and_the_second_ask_fit_inside_the_timeout(self):
        assert (discover._ISOLATED_TORCH_REASK_PAUSE
                + discover._ISOLATED_TORCH_REASK_MIN_LEFT
                < discover._ISOLATED_TORCH_PROBE_TIMEOUT)

    def test_a_spawn_failure_is_not_asked_again(self, monkeypatch):
        fake_popen = MagicMock(side_effect=OSError("no interpreter"))
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        assert discover._torch_gpus_isolated() is None
        assert fake_popen.call_count == 1


class TestSlowFirstImportIsNotFinal:
    """The first import of torch after an install compiles its bytecode and can
    overrun the timeout. The next child resumes from what the first one wrote."""

    def setup_method(self):
        discover._reset_gpu_probe_cache()

    def teardown_method(self):
        discover._reset_gpu_probe_cache()

    def test_a_slow_first_import_answers_on_a_later_round(self, monkeypatch, tmp_path):
        import time
        psutil = pytest.importorskip("psutil")
        marker = _stub_torch_env(monkeypatch, tmp_path, _PROBE_STUB_BLOCKS_ONCE)
        monkeypatch.setattr(discover, "_ISOLATED_TORCH_PROBE_TIMEOUT", 5.0)
        assert discover._torch_gpus_isolated_once() == []
        assert discover.isolated_torch_unavailable() is True
        assert discover._torch_gpus_isolated_once() == [], "asked inside the wait"

        if marker.exists():
            pid = int(marker.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 30
            while psutil.pid_exists(pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not psutil.pid_exists(pid), "the timed-out child outlived its kill"
        else:
            marker.write_text("0", encoding="utf-8")

        monkeypatch.setattr(discover, "_ISOLATED_TORCH_PROBE_TIMEOUT", 120.0)
        _after_the_retry_wait(monkeypatch)
        devices = discover._torch_gpus_isolated_once()
        assert devices and devices[0]["name"] == "STUB GPU", devices
        assert discover.isolated_torch_unavailable() is False


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32", reason="taskkill is the kill under test")
def test_a_debug_server_whose_probe_is_killed_passes_the_log_audit(tmp_path):
    """A real ``localm gui --debug``: the probe child is killed from outside while
    the server stays up, the server's next ask answers through the real list_gpus
    path, then the tree is killed, and the real log audit reads the log."""
    import os
    import socket
    import time
    import urllib.request
    psutil = pytest.importorskip("psutil")
    repo = pathlib.Path(discover.__file__).resolve().parent.parent
    no_window = subprocess.CREATE_NO_WINDOW
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(
        json.dumps({"update_url": "", "bugreport_upload_url": ""}), encoding="utf-8")
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "torch.py").write_text(_PROBE_STUB_BLOCKS_ONCE, encoding="utf-8")
    marker = tmp_path / "blocked-pid"
    env = dict(os.environ, LOCALM_HOME=str(home), LOCALM_CRASH_WATCHDOG="off",
               PROBE_STUB_MARKER=str(marker), PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([str(stub_dir), str(repo)]))
    subprocess.run([sys.executable, "-m", "localm", "key", "generate"], env=env,
                   cwd=str(home), capture_output=True, check=True, timeout=120,
                   stdin=subprocess.DEVNULL, creationflags=no_window)
    key = (home / "auth.key").read_text(encoding="utf-8").strip()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    def stats():
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/stats")
        req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except OSError:
            return None

    with open(tmp_path / "server.txt", "wb") as out:
        server = subprocess.Popen(
            [sys.executable, "-m", "localm", "gui", "--no-model", "--no-browser",
             "--isolated", "--new", "--debug", "--host", "127.0.0.1", "--port", str(port)],
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
            cwd=str(home),
            creationflags=no_window | subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and stats() is None:
                time.sleep(0.5)
            _kill_from_outside(_wait_for_pid(marker))
            expected_total = 8 * 2**30
            total = None
            while time.monotonic() < deadline and total != expected_total:
                time.sleep(1.0)
                total = ((stats() or {}).get("vram") or {}).get("total")
            assert total == expected_total, "the server never read the probe's answer"
            logs = list((home / "logs").glob("localm_*.log"))
            assert len(logs) == 1, logs
            assert "printed nothing" in logs[0].read_text(errors="replace"), (
                "the child was never killed")
        finally:
            kids = psutil.Process(server.pid).children(recursive=True)
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(server.pid)],
                           capture_output=True, creationflags=no_window)
            server.wait(60)
            for kid in kids:
                try:
                    kid.kill()
                except psutil.Error:
                    pass
    manifest = tmp_path / "run.manifest.json"
    manifest.write_text(json.dumps({"run": "probe killed from outside"}), encoding="utf-8")
    audit = subprocess.run(
        [sys.executable, str(repo / "scripts" / "log_audit.py"), "--manifest",
         str(manifest), "--strict", "--home", str(home), str(logs[0])],
        capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
        creationflags=no_window)
    assert audit.returncode == 0, audit.stdout + audit.stderr


class TestHalfImportedTorchOffWindowsIsInconclusive:
    """Off Windows, a failed isolated probe while another thread is still
    importing torch gives an inconclusive empty reading, which leaves the last
    good reading in place."""

    _GOOD = [{"index": 0, "name": "AMD Radeon RX 6900 XT", "total": 16 * 1024 ** 3,
              "free": 8 * 1024 ** 3}]

    def setup_method(self):
        discover._reset_gpu_probe_cache()

    def teardown_method(self):
        discover._reset_gpu_probe_cache()

    @staticmethod
    def _half_imported_torch(monkeypatch):
        import types

        from localm import gpu_usage
        monkeypatch.setattr(discover, "_in_process_torch_import_can_freeze",
                            lambda: False)
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        monkeypatch.setattr(discover, "_torch_is_resident", lambda: False)
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: None)
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        monkeypatch.setattr(gpu_usage, "torch_fully_imported", lambda: False)
        monkeypatch.setattr(
            discover, "_torch_gpus_resident",
            lambda: pytest.fail("read a half-imported torch in this process"))
        monkeypatch.setattr(subprocess, "Popen",
                            MagicMock(side_effect=FileNotFoundError("no nvidia-smi")))
        monkeypatch.setattr(discover, "_windows_largest_adapter_registry_entry",
                            lambda: None)

    def test_the_round_is_marked_unanswered(self, monkeypatch):
        self._half_imported_torch(monkeypatch)
        assert discover._torch_gpus_isolated_once() == []
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unanswered is True

    def test_an_empty_reading_is_inconclusive_not_no_gpu(self, monkeypatch):
        self._half_imported_torch(monkeypatch)
        gpus, status = discover.list_gpus(return_status=True)
        assert gpus == []
        assert status == discover.GPU_PROBE_INCONCLUSIVE

    def test_the_last_good_reading_survives_the_round(self, monkeypatch):
        with discover._gpu_probe_lock:
            discover._gpu_last_good = [dict(g) for g in self._GOOD]
        self._half_imported_torch(monkeypatch)
        discover.list_gpus(return_status=True)
        assert discover.last_known_gpus() == self._GOOD

    def test_a_conclusive_empty_reading_still_replaces_it(self, monkeypatch):
        with discover._gpu_probe_lock:
            discover._gpu_last_good = [dict(g) for g in self._GOOD]
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        monkeypatch.setattr(discover, "_torch_is_resident", lambda: False)
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: [])
        monkeypatch.setattr(subprocess, "Popen",
                            MagicMock(side_effect=FileNotFoundError("no nvidia-smi")))
        gpus, status = discover.list_gpus(return_status=True)
        assert (gpus, status) == ([], discover.GPU_PROBE_OK)
        assert discover.last_known_gpus() == []


class TestChildProbeContract:
    """The child must mirror the in-process branch field for field, so a caller
    cannot tell which path produced a reading."""

    # A STUB torch, so the enumeration contract is covered on EVERY platform
    # rather than only where a real torch happens to be installed. CI installs
    # `.[dev,rag]`, which has no torch, so the importorskip test below skips
    # there. This one runs the child's real _enumerate() against a fake torch, so
    # the JSON contract, the field set and the int coercion are all exercised.
    _STUB_TORCH = """
class _Cuda:
    @staticmethod
    def is_available():
        return True
    @staticmethod
    def device_count():
        return 2
    @staticmethod
    def mem_get_info(i):
        return (1000 + i, 4000 + i)      # (free, total), deliberately not round
    @staticmethod
    def get_device_name(i):
        return "STUB GPU %d" % i
cuda = _Cuda()
"""

    def _run_child_with_stub_torch(self, tmp_path, stub=None):
        """Spawn the real child module with *stub* shadowing torch on sys.path."""
        import os
        from localm._mp_spawn import interpreter_for_localm_children
        stub_dir = tmp_path / "stub"
        stub_dir.mkdir()
        (stub_dir / "torch.py").write_text(
            self._STUB_TORCH if stub is None else stub, encoding="utf-8")
        env = dict(os.environ)
        repo = str(pathlib.Path(discover.__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join([str(stub_dir), repo])
        return subprocess.run(
            [interpreter_for_localm_children(), "-u", "-m",
             "localm._torch_gpu_probe"],
            capture_output=True, text=True, timeout=120, env=env)

    def test_child_enumerates_against_a_stub_torch_on_any_platform(self, tmp_path):
        proc = self._run_child_with_stub_torch(tmp_path)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        assert len(lines) == 1, f"child printed {len(lines)} lines: {proc.stdout!r}"
        devices = json.loads(lines[0])
        assert devices == [
            {"index": 0, "name": "STUB GPU 0", "total": 4000, "free": 1000},
            {"index": 1, "name": "STUB GPU 1", "total": 4001, "free": 1001},
        ], f"enumeration contract drifted: {devices!r}"

    def test_child_skips_a_device_that_cannot_report_memory(self, tmp_path):
        """One device failing must never hide the rest: the in-process branch
        has that property and the child matches it."""
        stub = """
class _Cuda:
    @staticmethod
    def is_available():
        return True
    @staticmethod
    def device_count():
        return 2
    @staticmethod
    def mem_get_info(i):
        if i == 0:
            raise RuntimeError("device 0 is sulking")
        return (1000 + i, 4000 + i)
    @staticmethod
    def get_device_name(i):
        return "STUB GPU %d" % i
cuda = _Cuda()
"""
        proc = self._run_child_with_stub_torch(tmp_path, stub=stub)
        devices = json.loads(proc.stdout.strip())
        assert [d["index"] for d in devices] == [1], (
            f"a failing device hid the healthy one: {devices!r}")

    def test_child_without_torch_answers_no_device_and_exits_0(self, tmp_path):
        stub = 'raise ModuleNotFoundError("No module named \'torch\'", name="torch")\n'
        proc = self._run_child_with_stub_torch(tmp_path, stub=stub)
        assert proc.stdout.strip() == "[]"
        assert proc.returncode == 0

    def test_child_whose_torch_fails_exits_1(self, tmp_path):
        stub = 'raise RuntimeError("CUDA driver is broken")\n'
        proc = self._run_child_with_stub_torch(tmp_path, stub=stub)
        assert proc.stdout.strip() == "[]"
        assert proc.returncode == 1
        assert "CUDA driver is broken" in proc.stderr

    def test_child_reports_no_device_when_stub_torch_has_none(self, tmp_path):
        stub = self._STUB_TORCH.replace("return True", "return False")
        proc = self._run_child_with_stub_torch(tmp_path, stub=stub)
        assert proc.stdout.strip() == "[]"
        assert proc.returncode == 0

    def test_child_emits_one_json_line_and_nothing_else(self):
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            pytest.skip("llama.cpp's native runtime is already loaded in this "
                         "process (a real compute-device probe ran earlier in "
                         "this same pytest worker) - a fresh torch import here "
                         "is the known-doomed DLL-identity conflict, not this "
                         "test's own subject")
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
            required = {"index", "name", "total", "free"}
            assert required <= set(d) <= required | {"integrated", "pci_bus_id"}
            assert isinstance(d.get("integrated", False), bool)
            assert isinstance(d.get("pci_bus_id", 0), int)
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


# --------------------------------------------------------------------------- #
#  The child's stderr is relayed ONCE per distinct cause, not once per probe   #
# --------------------------------------------------------------------------- #

from localm import discover as _discover


@pytest.fixture
def probe_log(monkeypatch, caplog):
    """Reset the process-wide latch and hand back a driver for the real
    _torch_gpus_isolated with a faked child."""
    monkeypatch.setattr(_discover, "_child_stderr_seen", set())
    monkeypatch.setattr(_discover, "_child_stderr_cap_reported", False)
    monkeypatch.setattr("localm._mp_spawn.interpreter_for_localm_children",
                        lambda: "python")

    def _drive(stderrs, stdout='[]'):
        replies = iter(stderrs)

        def _fake_popen(*a, **kw):
            fake = MagicMock()
            fake.communicate.return_value = (stdout, next(replies))
            fake.returncode = 0
            return fake

        monkeypatch.setattr(subprocess, "Popen", _fake_popen)
        # caplog.records ACCUMULATES across calls, so a test that drives this
        # helper twice would otherwise read the FIRST drive's lines back out of
        # the second drive's result and pass no matter what the second one did.
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="localm"):
            for _ in stderrs:
                _discover._torch_gpus_isolated()
        return [r.getMessage() for r in caplog.records]

    return _drive


def test_the_same_probe_failure_is_relayed_once_not_every_probe(probe_log):
    boom = "RuntimeError: HIP error: no ROCm-capable device is detected"
    messages = probe_log([boom] * 5)

    carrying = [m for m in messages if boom in m]
    assert len(carrying) == 1, (
        f"the child's stderr was relayed {len(carrying)} times across 5 probes "
        f"- at the VRAM meter's poll rate that is a log line every 2.5s: {carrying}")


def test_a_DIFFERENT_probe_failure_is_still_relayed(probe_log):
    """The latch is keyed on the relayed TEXT, not on a once-only bool: a
    second, different cause is still relayed rather than swallowed.
    """
    first = "RuntimeError: HIP error: no ROCm-capable device is detected"
    second = "ImportError: libtorch_hip.so: cannot open shared object file"
    messages = probe_log([first, first, second, second])

    assert any(first in m for m in messages), "the first cause was never relayed"
    assert any(second in m for m in messages), (
        "a DIFFERENT probe failure was swallowed - the latch is hiding new "
        "information, not just suppressing repeats")


def test_a_probe_that_says_nothing_relays_nothing(probe_log):
    # Empty stderr must not produce an empty "child said: " clause.
    messages = probe_log(["", ""])
    assert not any("child said" in m for m in messages)


def test_the_latch_announces_itself_rather_than_going_silently_blind(probe_log):
    # Pathological input (stderr that differs every probe) is bounded, and the
    # log says the cap was reached rather than dropping causes silently.
    messages = probe_log([f"distinct failure {i}" for i in range(12)])
    assert any("further distinct causes suppressed" in m for m in messages), (
        "the cap was reached with no line saying so")


def test_resetting_the_probe_cache_also_clears_the_stderr_latch(probe_log):
    """_reset_gpu_probe_cache clears the stderr latch as well as
    _isolated_torch_broken_warned, so the same cause is relayed again after a
    reset.
    """
    boom = "RuntimeError: HIP error: no ROCm-capable device is detected"
    assert any(boom in m for m in probe_log([boom])), "test premise: first relay"

    _discover._reset_gpu_probe_cache()

    relayed_again = [m for m in probe_log([boom]) if boom in m]
    assert relayed_again, (
        "after a probe-cache reset the same failure was still suppressed - the "
        "reset does not clear the stderr latch, so it only partially resets")
