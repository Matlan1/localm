# SPDX-License-Identifier: AGPL-3.0-or-later
"""Failures inside the isolated model workers reach the parent with enough to
act on: a load error keeps its exception type, a defect in localm is not
reported as a broken runtime, a grammar the worker dropped mid-generation is
reported and latched, and setup problems on the load path are logged."""

import logging
import queue
import threading
from unittest.mock import MagicMock

import pytest



@pytest.mark.parametrize("exc,text", [
    (RuntimeError("Failed to load model: m.gguf"), "Failed to load model: m.gguf"),
    (OSError("cannot open"), "cannot open"),
    (TypeError("bad arg"), "TypeError: bad arg"),
    (KeyError("qwen9"), "KeyError: 'qwen9'"),
    (MemoryError(), "MemoryError"),
    (RuntimeError(""), "RuntimeError"),
])
def test_load_error_text(exc, text):
    from localm.inference.backends.llamacpp._runner import load_error_text
    assert load_error_text(exc) == text


def test_load_error_text_shortens_a_huge_number_from_a_model_config():
    from localm.inference.backends.llamacpp._runner import load_error_text
    huge = "7" * 4300
    text = load_error_text(ValueError(
        f"The hidden size ({huge}) is not a multiple of the number of attention "
        f"heads ({huge})."))
    assert len(text) < 200
    assert text.count("(4300 digits)") == 2
    assert "ValueError: The hidden size (777777777777...(4300 digits))" in text


def test_load_error_text_keeps_a_number_that_is_not_huge():
    from localm.inference.backends.llamacpp._runner import load_error_text
    assert load_error_text(ValueError("id 12345678901234567890123456789012345678 is odd")) == (
        "ValueError: id 12345678901234567890123456789012345678 is odd")


def _drive(main, worker_module, worker_attr, worker_cls, monkeypatch, *, hf=False):
    import localm._mp_spawn as mp_spawn
    monkeypatch.setattr(mp_spawn, "install_parent_death_watchdog", lambda *a: None)
    monkeypatch.setattr(mp_spawn, "suppress_native_error_dialogs", lambda *a: None)
    if hf:
        from localm.inference.backends import _hf_hub_gate
        monkeypatch.setattr(_hf_hub_gate, "close_hub_gate", lambda: None)
    monkeypatch.setattr(worker_module, worker_attr, worker_cls)
    req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    t = threading.Thread(target=main, args=(req_q, resp_q, ctrl_q), daemon=True)
    t.start()
    return req_q, resp_q, t


def _failing_worker(exc):
    class _W:
        def __init__(self, **kw):
            pass

        def load(self):
            raise exc

        def close(self):
            pass
    return _W


@pytest.mark.parametrize("exc,envelope", [
    (TypeError("bad arg"), ("error", "TypeError: bad arg", "UnexpectedLoadError")),
    (MemoryError(), ("error", "MemoryError", "UnexpectedLoadError")),
    (RuntimeError("Failed to load model"), ("error", "Failed to load model")),
])
def test_gguf_load_error_envelope_keeps_the_type(monkeypatch, exc, envelope):
    from localm.inference.backends.llamacpp import _runner, _worker
    req_q, resp_q, t = _drive(_runner._runner_main, _worker, "GgufWorker",
                              _failing_worker(exc), monkeypatch)
    try:
        req_q.put(("load", {}))
        assert resp_q.get(timeout=5) == envelope
    finally:
        req_q.put(None)
        t.join(timeout=5)


def test_hf_load_error_envelope_keeps_the_type(monkeypatch):
    from localm.inference.backends import _hf_runner, _hf_worker
    req_q, resp_q, t = _drive(_hf_runner._runner_main, _hf_worker, "HFWorker",
                              _failing_worker(KeyError("qwen9")), monkeypatch, hf=True)
    try:
        req_q.put(("load", {}))
        assert resp_q.get(timeout=5) == ("error", "KeyError: 'qwen9'")
    finally:
        req_q.put(None)
        t.join(timeout=5)


def test_an_abi_mismatch_travels_as_a_runtime_failure(monkeypatch):
    from localm.inference.backends.gguf import _load_failure_message
    from localm.inference.backends.llamacpp import _runner, _worker
    from localm.inference.backends.llamacpp._abi import AbiMismatch
    req_q, resp_q, t = _drive(_runner._runner_main, _worker, "GgufWorker",
                              _failing_worker(AbiMismatch("struct layout drifted")),
                              monkeypatch)
    try:
        req_q.put(("load", {}))
        envelope = resp_q.get(timeout=5)
    finally:
        req_q.put(None)
        t.join(timeout=5)
    assert envelope == ("error", "struct layout drifted")
    msg = _load_failure_message(RuntimeError(envelope[1]))
    assert "unexpected error in localm" not in msg
    assert "struct layout drifted" in msg and "setup-llama" in msg


def test_an_unexpected_load_error_is_not_blamed_on_the_runtime():
    from localm.inference.backends.gguf import _load_failure_message
    from localm.inference.backends.llamacpp._runner import UnexpectedLoadError
    msg = _load_failure_message(UnexpectedLoadError("TypeError: bad arg"))
    assert "unexpected error in localm" in msg and "TypeError: bad arg" in msg
    assert "setup-llama" not in msg


def test_an_empty_load_error_still_says_something():
    from localm.inference.backends.gguf import _load_failure_message
    assert "RuntimeError" in _load_failure_message(RuntimeError(""))


def test_safe_grammar_processor_reports_that_it_dropped_the_constraint():
    from localm.inference.backends._hf_worker import _SafeGrammarProcessor

    def _inner(ids, scores):
        raise RuntimeError("triton missing")
    proc = _SafeGrammarProcessor(_inner)
    assert proc.failed is False
    assert proc(None, "scores") == "scores"
    assert proc.failed is True


def test_hf_done_envelope_carries_a_dropped_grammar(monkeypatch):
    from localm.inference.backends import _hf_runner, _hf_worker

    class _W:
        last_finish_reason = "stop"
        supports_images = False
        processor_error = None
        can_embed = False
        resolved_device = "cpu"
        grammar_dropped = False

        def __init__(self, **kw):
            pass

        def load(self):
            pass

        def chat_stream(self, **kw):
            yield "x"
            self.grammar_dropped = True

        def close(self):
            pass

    req_q, resp_q, t = _drive(_hf_runner._runner_main, _hf_worker, "HFWorker", _W,
                              monkeypatch, hf=True)
    try:
        req_q.put(("load", {}))
        assert resp_q.get(timeout=5)[0] == "ok"
        req_q.put(("chat_stream", {"messages": [{"role": "user", "content": "hi"}]}, 1))
        got = [resp_q.get(timeout=5) for _ in range(2)]
        assert got[0] == ("chunk", "x")
        assert got[1][0] == "done" and got[1][1]["grammar_unsupported"] is True
    finally:
        req_q.put(None)
        t.join(timeout=5)


def test_hf_backend_latches_a_dropped_grammar_and_refuses_the_next(monkeypatch, caplog):
    from localm.inference.backends.base import GrammarUnsupportedError
    from localm.inference.backends.hf import HFBackend

    runner = MagicMock()
    runner.is_alive.return_value = True
    runner.last_done = {"finish_reason": "stop", "grammar_unsupported": True}
    runner.chat_stream.side_effect = lambda **kw: iter(["ok"])
    b = object.__new__(HFBackend)
    b._runner = runner
    b._loaded = True
    b._supports_images = False
    b._processor_error = None
    monkeypatch.setattr(HFBackend, "_first_token_timeout_seconds",
                        staticmethod(lambda: 5.0))
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    msgs = [{"role": "user", "content": "hi"}]

    with caplog.at_level(logging.WARNING, logger="localm"):
        assert list(b.chat_stream(msgs, grammar='root ::= "a"')) == ["ok"]

    assert b.supports_grammar is False
    assert any("later grammar requests to this model are refused" in r.getMessage() for r in caplog.records)
    with pytest.raises(GrammarUnsupportedError):
        list(b.chat_stream(msgs, grammar='root ::= "a"'))


def test_a_split_device_whose_vram_cannot_be_read_is_named(monkeypatch, caplog):
    from localm.inference.backends._hf_worker import _cuda_device_map
    monkeypatch.setattr("localm.discover.list_gpus",
                        lambda: [{"index": 0}, {"index": 1}, {"index": 2}])
    fake = MagicMock()
    mem = {0: (8_000_000_000, 16_000_000_000), 1: (8_000_000_000, 16_000_000_000)}

    def _info(idx):
        if idx == 2:
            raise RuntimeError("device 2 hidden")
        return mem[idx]
    fake.cuda.mem_get_info.side_effect = _info

    with caplog.at_level(logging.WARNING, logger="localm"):
        result = _cuda_device_map(fake, config={"gpu_split_indices": [0, 1, 2]})

    assert {k for k in result["max_memory"] if isinstance(k, int)} == {0, 1}
    assert any("device 2" in r.getMessage() and "left out" in r.getMessage()
               for r in caplog.records)


def test_a_non_path_binary_dir_is_reported(monkeypatch, caplog):
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr("localm.config.load_config", lambda: {"binary_dir": 5})
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)

    with caplog.at_level(logging.WARNING, logger="localm"):
        _loader._candidate_dirs()

    assert any("binary_dir" in r.getMessage() for r in caplog.records)


def test_a_backend_registration_fault_is_logged_with_its_cause(monkeypatch, caplog):
    from localm.inference.backends.llamacpp import _loader

    def _boom(*a, **kw):
        raise TypeError("registration code broke")
    monkeypatch.setattr(_loader, "_register_ggml_backends", _boom)

    with caplog.at_level(logging.WARNING, logger="localm"):
        assert _loader._register_compute_backends(None, None) is False

    assert any(r.exc_info and "registration code broke" in str(r.exc_info[1])
               for r in caplog.records)


class _Survivor:
    """A process stand-in that ignores shutdown and terminate, and exits on kill."""

    def __init__(self, dies_on_kill=True):
        self.pid = 4242
        self._alive = True
        self._dies_on_kill = dies_on_kill
        self.killed = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        pass

    def kill(self):
        self.killed = True
        if self._dies_on_kill:
            self._alive = False


@pytest.mark.parametrize("runner_path", [
    "localm.inference.backends.llamacpp._runner:ModelRunner",
    "localm.inference.backends._hf_runner:HFRunner",
])
def test_shutdown_kills_a_worker_that_survives_terminate(runner_path, caplog):
    import importlib
    mod_name, cls_name = runner_path.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    for dies in (True, False):
        r = object.__new__(cls)
        proc = _Survivor(dies_on_kill=dies)
        r._proc = proc
        r._req_q = MagicMock()
        r._resp_q = None
        r._ctrl_q = None
        r._crash_trace_path = None
        r._discard_native_crash_trace = lambda: None
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="localm"):
            r.shutdown(grace=0)
        assert proc.killed is True
        warned = any("did not exit after kill" in rec.getMessage() for rec in caplog.records)
        assert warned is (not dies)
