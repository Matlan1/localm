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


# --------------------------------------------------------------------------- #
#  The child's stderr is relayed ONCE per distinct cause, not once per probe   #
# --------------------------------------------------------------------------- #

from localm import discover as _discover


class _FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


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

        def _fake_run(*a, **kw):
            return _FakeCompleted(stdout=stdout, stderr=next(replies))

        monkeypatch.setattr(subprocess, "run", _fake_run)
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
