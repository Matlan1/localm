# SPDX-License-Identifier: AGPL-3.0-or-later
"""The isolated speech worker: native faults and hangs cost the worker, never the
caller; a cancel stops the synthesis at a frame boundary and leaves the model
loaded for the next request; progress reaches the caller."""

import os
import queue
import threading
import time

import pytest

from localm.inference import _speech_runner as sr
from localm.inference.backends.llamacpp import mtmd_gen as g


@pytest.fixture
def fault_env(monkeypatch):
    def set_fault(mode):
        monkeypatch.setenv(sr._FAULT_ENV, mode)
    yield set_fault
    os.environ.pop(sr._FAULT_ENV, None)


class TestRealProcessContainment:
    def test_a_hard_exit_during_load_is_contained(self, fault_env):
        fault_env("exit")
        runner = sr.SpeechRunner()
        with pytest.raises(RuntimeError, match="crashed"):
            runner.spawn_and_load({"model_path": "x"}, timeout=60)
        assert not runner.is_alive()

    def test_a_native_abort_during_load_is_contained(self, fault_env):
        fault_env("abort")
        runner = sr.SpeechRunner()
        with pytest.raises(RuntimeError, match="crashed"):
            runner.spawn_and_load({"model_path": "x"}, timeout=60)
        assert not runner.is_alive()

    def test_a_hung_load_times_out_and_is_killed(self, fault_env):
        fault_env("hang")
        runner = sr.SpeechRunner()
        with pytest.raises(RuntimeError, match="timed out"):
            runner.spawn_and_load({"model_path": "x"}, timeout=3)
        assert not runner.is_alive()

    def test_a_crash_while_speaking_is_contained(self, fault_env):
        fault_env("speak-exit")
        runner = sr.SpeechRunner()
        runner._spawn()
        try:
            with pytest.raises(RuntimeError, match="crashed"):
                runner.speak({"text": "hi"})
        finally:
            runner.shutdown(grace=0)
        assert not runner.is_alive()

    def test_a_hung_synthesis_is_stopped_as_hung(self, fault_env):
        fault_env("speak-hang")
        runner = sr.SpeechRunner()
        runner._spawn()
        try:
            with pytest.raises(sr.SpeechWorkerHung, match="no progress"):
                runner.speak({"text": "hi"}, stall_timeout=2)
        finally:
            runner.shutdown(grace=0)
        assert not runner.is_alive()

    def test_speaking_without_a_worker_is_a_clean_error(self):
        with pytest.raises(RuntimeError, match="not running"):
            sr.SpeechRunner().speak({"text": "hi"})

    def test_double_shutdown_is_safe(self):
        runner = sr.SpeechRunner()
        runner.shutdown()
        runner.shutdown()


class _AliveProc:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.alive = False


def _wire(runner):
    runner._proc = _AliveProc()
    runner._req_q = queue.Queue()
    runner._resp_q = queue.Queue()
    runner._cancel = threading.Event()
    return runner


def _child(runner, *, honour_cancel=True, frames=1000, stop=None):
    """A stand-in child: one progress message per 20 ms until done or cancelled."""
    def run():
        cmd = runner._req_q.get()
        assert cmd[0] == "speak"
        for n in range(1, frames + 1):
            if stop is not None and stop.is_set():
                return
            if honour_cancel and runner._cancel.is_set():
                runner._resp_q.put(("error", "cancelled", "SpeechCancelled"))
                return
            runner._resp_q.put(("progress", n))
            time.sleep(0.02)
        runner._resp_q.put(("ok", {"wav": b"RIFF", "frames": frames}))
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


class TestParentProtocol:
    def test_progress_reaches_the_caller_and_the_result_returns(self):
        runner = _wire(sr.SpeechRunner())
        _child(runner, frames=5)
        seen = []
        assert runner.speak({"text": "hi"}, on_progress=seen.append)["frames"] == 5
        assert seen == [1, 2, 3, 4, 5]

    def test_cancel_stops_the_worker_and_keeps_it(self, monkeypatch):
        runner = _wire(sr.SpeechRunner())
        _child(runner)
        seen = []
        with pytest.raises(g.SpeechCancelled):
            runner.speak({"text": "hi"}, on_progress=seen.append,
                         should_cancel=lambda: len(seen) >= 3)
        assert runner._cancel.is_set() and runner._proc.alive is True
        assert 3 <= len(seen) < 1000

    def test_the_cancel_event_is_cleared_for_the_next_request(self):
        runner = _wire(sr.SpeechRunner())
        runner._cancel.set()
        _child(runner, frames=2)
        assert runner.speak({"text": "hi"})["frames"] == 2

    def test_a_worker_that_ignores_the_cancel_is_killed_after_the_grace(self, monkeypatch):
        monkeypatch.setattr(sr, "CANCEL_GRACE", 0.3)
        runner = _wire(sr.SpeechRunner())
        proc = runner._proc
        stop = threading.Event()
        _child(runner, honour_cancel=False, stop=stop)
        try:
            with pytest.raises(g.SpeechCancelled, match="did not stop in time"):
                runner.speak({"text": "hi"}, should_cancel=lambda: True)
        finally:
            stop.set()
        assert proc.alive is False and runner._proc is None

    def test_tagged_errors_are_raised_by_type(self):
        runner = _wire(sr.SpeechRunner())
        runner._resp_q.put(("error", "bad text", "SpeechInputError"))
        with pytest.raises(g.SpeechInputError, match="bad text"):
            runner.speak({"text": "hi"})
        runner._resp_q.put(("error", "too long", "SpeechBudgetExceeded"))
        with pytest.raises(g.SpeechBudgetExceeded):
            runner.speak({"text": "hi"})
        runner._resp_q.put(("error", "plain", None))
        with pytest.raises(RuntimeError, match="plain"):
            runner.speak({"text": "hi"})


class _FakeSynth:
    instances = []

    def __init__(self, **params):
        self.params = params
        self.pipeline = 1
        self.sample_rate = 24000
        self.encoder_sample_rate = 24000
        self.projector_on_gpu = False
        self.closed = False
        _FakeSynth.instances.append(self)

    def synthesize(self, text, *, language=None, reference=None, seed=None,
                   on_progress=None, should_stop=None):
        for n in range(1, 6):
            if should_stop():
                raise g.SpeechCancelled()
            on_progress(n)
            if text == "cancel me" and n == 2:
                self.cancel_event.set()
        return g.SpeechResult(wav=b"RIFF" + text.encode(), sample_rate=24000,
                              n_samples=9600, frames=5, seed=seed or 1)

    def close(self):
        self.closed = True


@pytest.fixture
def in_process(monkeypatch):
    import localm._mp_spawn as mps
    import localm.debuglog as dl
    for name in ("install_parent_death_watchdog", "ignore_interrupt_signals",
                 "suppress_native_error_dialogs"):
        monkeypatch.setattr(mps, name, lambda: None)
    monkeypatch.setattr(dl, "attach_child_logging", lambda: None)
    monkeypatch.setattr(g, "SpeechSynthesizer", _FakeSynth)
    monkeypatch.setattr(sr, "PROGRESS_INTERVAL", 0.0)
    _FakeSynth.instances.clear()


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


class TestChildLoop:
    def test_load_speak_cancel_speak_shutdown(self, in_process):
        req, resp, cancel = queue.Queue(), queue.Queue(), threading.Event()
        req.put(("load", {"model_path": "m", "mmproj_path": "p", "cpu_only": False}))
        req.put(("speak", {"text": "cancel me", "seed": 3}))
        req.put(("speak", {"text": "again", "seed": 4}))
        req.put(("shutdown", None))

        _FakeSynth.cancel_event = cancel
        orig_get = req.get

        def get_and_clear(*a, **k):
            cmd = orig_get(*a, **k)
            if cmd and cmd[0] == "speak" and cmd[1]["text"] == "again":
                cancel.clear()
            return cmd
        req.get = get_and_clear
        sr._runner_main(req, resp, cancel)
        msgs = _drain(resp)
        kinds = [m[0] for m in msgs]
        assert msgs[0] == ("ok", {"pipeline": 1, "sample_rate": 24000,
                                  "encoder_sample_rate": 24000, "projector_on_gpu": False})
        assert ("error", "", "SpeechCancelled") in [(m[0], m[1], m[2]) for m in msgs if m[0] == "error"]
        assert kinds[-1] == "ok" and msgs[-1][1]["wav"] == b"RIFFagain"
        assert _FakeSynth.instances[0].closed is True
        assert len(_FakeSynth.instances) == 1
        assert _FakeSynth.instances[0].params == {"model_path": "m", "mmproj_path": "p"}

    def test_speak_before_load_and_unknown_commands_are_errors(self, in_process):
        req, resp, cancel = queue.Queue(), queue.Queue(), threading.Event()
        req.put(("speak", {"text": "x"}))
        req.put(("bogus", None))
        req.put(None)
        sr._runner_main(req, resp, cancel)
        msgs = _drain(resp)
        assert msgs[0][0] == "error" and "before a model" in msgs[0][1]
        assert msgs[1][0] == "error" and "unknown speech-runner command" in msgs[1][1]

    def test_a_load_failure_is_reported_with_its_tag(self, in_process, monkeypatch):
        class Broken:
            def __init__(self, **params):
                raise g.SpeechUnavailable("no speech stages")
        monkeypatch.setattr(g, "SpeechSynthesizer", Broken)
        req, resp, cancel = queue.Queue(), queue.Queue(), threading.Event()
        req.put(("load", {}))
        req.put(None)
        sr._runner_main(req, resp, cancel)
        assert _drain(resp) == [("error", "no speech stages", "SpeechUnavailable")]

    def test_cpu_only_hides_gpu_devices_before_the_load(self, in_process, monkeypatch):
        for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(var, raising=False)
        seen = {}

        class Spy(_FakeSynth):
            def __init__(self, **params):
                seen["hip"] = os.environ.get("HIP_VISIBLE_DEVICES")
                super().__init__(**params)
        monkeypatch.setattr(g, "SpeechSynthesizer", Spy)
        req, resp, cancel = queue.Queue(), queue.Queue(), threading.Event()
        req.put(("load", {"model_path": "m", "cpu_only": True}))
        req.put(None)
        sr._runner_main(req, resp, cancel)
        assert seen["hip"] == "-1" and "cpu_only" not in Spy.instances[-1].params
