"""A chat request to an HF model without a chat template reports a clear typed
error and the worker stays alive; an untyped exception the worker dies on is
relayed with its own text instead of an opaque exit code."""
import queue
import threading

import pytest

import localm._mp_spawn as mp_spawn
from localm.inference.backends import _hf_runner, _hf_worker
from localm.inference.backends.base import ChatTemplateMissingError
from localm.inference.backends._hf_runner import HFRunner


class _Templated:
    def __init__(self, chat_template):
        self.chat_template = chat_template


class TestRequireChatTemplate:
    def test_a_tokenizer_without_a_template_is_refused(self):
        with pytest.raises(ChatTemplateMissingError) as ei:
            _hf_worker._require_chat_template(_Templated(None))
        assert str(ei.value) == _hf_worker.CHAT_TEMPLATE_MISSING_MESSAGE

    def test_an_empty_template_counts_as_missing(self):
        with pytest.raises(ChatTemplateMissingError):
            _hf_worker._require_chat_template(_Templated(""))

    def test_an_object_without_the_attribute_is_refused(self):
        with pytest.raises(ChatTemplateMissingError):
            _hf_worker._require_chat_template(object())

    def test_a_template_on_the_tokenizer_passes(self):
        _hf_worker._require_chat_template(_Templated("{{ messages }}"))

    def test_a_template_on_the_processor_alone_passes(self):
        _hf_worker._require_chat_template(
            _Templated("{{ messages }}"), _Templated(None))


def _drive_dispatch(monkeypatch, chat_stream):
    """Run ``_runner_main`` on a thread with a fake worker whose chat_stream is
    *chat_stream*. Returns ``(req_q, resp_q, died, thread)``."""
    monkeypatch.setattr(mp_spawn, "install_parent_death_watchdog", lambda *a: None)
    monkeypatch.setattr(mp_spawn, "suppress_native_error_dialogs", lambda *a: None)

    class _FakeWorker:
        last_finish_reason = "stop"
        supports_images = False
        processor_error = None
        can_embed = False
        resolved_device = "cpu"

        def __init__(self, **kw):
            pass

        def load(self):
            pass

        def chat_stream(self, **kw):
            return chat_stream()

        def count_tokens(self, payload):
            return 42

    monkeypatch.setattr(_hf_worker, "HFWorker", _FakeWorker)
    req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    died: list = []

    def _run():
        try:
            _hf_runner._runner_main(req_q, resp_q, ctrl_q)
        except BaseException as e:      # noqa: BLE001 - escaping = the worker dies
            died.append(e)

    t = threading.Thread(target=_run, name="hf-dispatch-under-test", daemon=True)
    t.start()
    req_q.put(("load", {}))
    assert resp_q.get(timeout=5)[0] == "ok"
    return req_q, resp_q, died, t


class TestChildDispatch:
    def test_a_missing_template_is_tagged_and_the_worker_keeps_serving(
            self, monkeypatch):
        def chat_stream():
            raise ChatTemplateMissingError("no template here")
            yield   # pragma: no cover

        req_q, resp_q, died, t = _drive_dispatch(monkeypatch, chat_stream)
        try:
            req_q.put(("chat_stream",
                       {"messages": [{"role": "user", "content": "hi"}]}, 1))
            envelope = resp_q.get(timeout=5)
            assert envelope == ("error", "no template here",
                                "ChatTemplateMissingError")
            req_q.put(("count_tokens", "still alive?"))
            assert resp_q.get(timeout=5) == ("ok", 42)
            assert not died, f"the dispatch loop died instead of reporting: {died!r}"
        finally:
            req_q.put(None)
            t.join(timeout=5)

    def test_an_untyped_exception_is_relayed_and_the_worker_still_dies(
            self, monkeypatch):
        def chat_stream():
            raise ValueError("tokenizer exploded mid-stream")
            yield   # pragma: no cover

        req_q, resp_q, died, t = _drive_dispatch(monkeypatch, chat_stream)
        req_q.put(("chat_stream",
                   {"messages": [{"role": "user", "content": "hi"}]}, 1))
        envelope = resp_q.get(timeout=5)
        t.join(timeout=5)
        assert envelope == ("error", "ValueError: tokenizer exploded mid-stream",
                            "WorkerFault")
        assert len(died) == 1 and isinstance(died[0], ValueError), died


class _FakeProc:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.alive = False


def _runner_with_response(envelope) -> HFRunner:
    r = HFRunner()
    r._proc = _FakeProc()
    r._req_q, r._resp_q, r._ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    r._resp_q.put(envelope)
    return r


class TestParentRelay:
    def test_a_missing_template_is_re_raised_as_its_own_type(self):
        r = _runner_with_response(
            ("error", "no template here", "ChatTemplateMissingError"))
        with pytest.raises(ChatTemplateMissingError) as ei:
            list(r.chat_stream(messages=[]))
        assert str(ei.value) == "no template here"
        assert r._proc is not None and r._proc.alive

    def test_a_worker_fault_names_the_real_cause_and_unloads(self):
        r = _runner_with_response(
            ("error", "ValueError: tokenizer exploded mid-stream", "WorkerFault"))
        proc = r._proc
        with pytest.raises(RuntimeError) as ei:
            list(r.chat_stream(messages=[]))
        msg = str(ei.value)
        assert "ValueError: tokenizer exploded mid-stream" in msg
        assert "No native fault trace" not in msg
        assert not proc.alive and r._proc is None
