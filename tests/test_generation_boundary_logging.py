# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression coverage for the prefill/decode boundary logging added to
LlamaCpp._generate (llama.py, runs inside the isolated GGUF worker) and
ModelRunner.chat_stream (_runner.py, runs in the parent). See dev-notes/
generation-path-logging-instrumentation-2026-08-12.md for the full design:
between "model loaded" and either a token or a corpse, this code path used
to emit nothing at any level, so a crash mid-generation could not be placed
in prefill vs decode.

Two independent test classes, because the two files are instrumented for two
different audiences: llama.py's markers only reach a bug report once --debug
is on (they run inside the isolated child process), while _runner.py's
markers reach the always-on ring buffer unconditionally (they run in the
parent). Each class drives the REAL method under test - a mocked native
`api` layer for llama.py (there is no real GGUF model here), real
multiprocessing.Queue objects plus a fake process-liveness stand-in and a
thread playing the child's protocol by hand for _runner.py, mirroring
test_kv_cache.py's TestInferenceLock and test_runner_stream_timeouts.py's
_make_runner/_fake_child patterns respectively (both already the established
patterns for driving these two methods without a real native model).
"""

import logging
import multiprocessing as mp
import threading
from unittest.mock import MagicMock, patch

import pytest

import localm.inference.backends.llamacpp.llama as llama_mod
import localm.inference.backends.llamacpp._runner as runner_mod
from localm.inference.backends.llamacpp.llama import LlamaCpp
from localm.inference.backends.llamacpp._runner import ModelRunner
from tests._bare_llama import make_bare_llama
from tests._fake_batch import fake_batch_init


# ---------------------------------------------------------------------------
#  llama.py: LlamaCpp._generate (child-side)
# ---------------------------------------------------------------------------

def _bare_llama() -> LlamaCpp:
    """Construct a LlamaCpp without running __init__ (no DLL access) - same
    shared builder as test_kv_cache.py's helper of the same name."""
    return make_bare_llama(_model_ptr=111, _ctx_ptr=222)


# Fake-pointer teardown is handled globally by tests/conftest.py's autouse
# _neutralise_bare_llama_pointers fixture.


def _mock_native_api() -> MagicMock:
    """A fully-wired mock api module: prefill (KV-reuse path) plus decode
    loop, all succeeding by default. llama_model_chat_template=None (used
    only by the vision path's own _apply_model_template call) takes the
    plain ChatML fallback instead of exercising the Jinja-template C-array
    machinery, which is not what these tests are about."""
    mock_api = MagicMock()
    mock_api.has_memory_api.return_value = True
    mock_api.llama_get_memory.return_value = 333
    mock_api.llama_memory_seq_rm.return_value = True
    mock_api.llama_decode.return_value = 0
    mock_api.llama_batch_init.side_effect = fake_batch_init
    mock_api.llama_sampler_sample.return_value = 42
    mock_api.llama_sampler_free = MagicMock()
    mock_api.llama_model_chat_template.return_value = None
    # Every predicate has to be answered explicitly: an unset MagicMock attribute
    # returns a TRUTHY mock, so leaving this one out claims the model uses M-RoPE
    # and silently sends _can_reuse_kv down the no-reuse path.
    mock_api.llama_model_has_mrope.return_value = False
    return mock_api


def _bare_llama_vision() -> LlamaCpp:
    """_bare_llama() plus a mocked mtmd handle, for _generate_image."""
    llm = _bare_llama()
    llm._mtmd = MagicMock()
    llm._mtmd.marker = "<image>"
    llm._mtmd.eval_into.return_value = 3   # pos after prefill
    return llm


_VISION_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]


def _messages(records) -> str:
    return "\n".join(r.getMessage() for r in records)


class TestLlamaCppGenerateBoundaryLogging:
    def test_normal_generation_logs_all_boundaries_in_order(self, monkeypatch, caplog):
        # Interval patched down so 6 mocked tokens produce visible checkpoints
        # without needing 50 real loop iterations - the interval's own value
        # is not the property under test, the CHECKPOINTING behaviour is.
        monkeypatch.setattr(llama_mod, "_DECODE_PROGRESS_INTERVAL", 2)
        llm = _bare_llama()
        llm._tokenizer.is_eog.return_value = False
        mock_api = _mock_native_api()

        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
             patch("localm.inference.backends.llamacpp.llama._build_sampler",
                   return_value=999), \
             caplog.at_level(logging.DEBUG, logger="localm"):
            tokens = list(llm._generate(
                prompt_tokens=[1, 2, 3], max_new_tokens=6,
                temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1))

        assert tokens == [42] * 6
        assert llm.last_finish_reason == "length"

        msgs = [r.getMessage() for r in caplog.records]
        joined = "\n".join(msgs)
        assert "gguf generate: prefill starting, 3 prompt token(s)" in joined
        assert "gguf generate: prefill complete" in joined
        assert "kv_reuse=True" in joined
        assert "gguf generate: entering decode loop" in joined
        assert "gguf generate: complete, 6 token(s)" in joined
        assert "finish_reason=length" in joined
        assert "aborted" not in joined

        # Decode progress is coarse (interval=2 over 6 tokens -> checkpoints
        # at 2, 4, 6) AND specifically DEBUG, not INFO - the ring buffer's
        # fixed 400-record budget is shared with everything else the server
        # logs, so only the boundary markers are affordable at INFO.
        progress = [r for r in caplog.records
                    if "gguf generate: decode progress" in r.getMessage()]
        assert len(progress) == 3
        assert all(r.levelname == "DEBUG" for r in progress)
        boundaries = [r for r in caplog.records
                      if "gguf generate: decode progress" not in r.getMessage()]
        assert boundaries and all(r.levelname == "INFO" for r in boundaries)

        # Order matters: a reader must be able to tell WHERE a silent death
        # would have landed from the sequence, not just presence.
        order = [i for i, m in enumerate(msgs) if any(
            key in m for key in ("prefill starting", "prefill complete",
                                  "entering decode loop", "generate: complete"))]
        assert order == sorted(order)

    def test_prefill_failure_logs_start_but_not_complete_then_aborts(self, caplog):
        llm = _bare_llama()
        mock_api = _mock_native_api()
        mock_api.llama_decode.return_value = -1   # prefill's own decode fails

        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
             caplog.at_level(logging.INFO, logger="localm"):
            with pytest.raises(RuntimeError, match="prefill"):
                list(llm._generate(
                    prompt_tokens=[1, 2, 3], max_new_tokens=6,
                    temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1))

        joined = _messages(caplog.records)
        assert "gguf generate: prefill starting, 3 prompt token(s)" in joined
        # The exact defect this instrumentation fixes: a failure must not
        # look identical to a silent hang. "starting" with no "complete" and
        # an explicit "aborted" line is the whole point.
        assert "gguf generate: prefill complete" not in joined
        assert "gguf generate: entering decode loop" not in joined
        assert ("gguf generate: aborted (exception) during prefill, "
                "0 token(s) generated") in joined

    def test_cancellation_mid_decode_logs_aborted_with_partial_count(self, caplog):
        llm = _bare_llama()
        llm._tokenizer.is_eog.return_value = False
        mock_api = _mock_native_api()

        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
             patch("localm.inference.backends.llamacpp.llama._build_sampler",
                   return_value=999), \
             caplog.at_level(logging.INFO, logger="localm"):
            gen = llm._generate(
                prompt_tokens=[1, 2, 3], max_new_tokens=50,
                temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1)
            # 1st next(): prefill + decode-loop entry + sample token1 + yield.
            assert next(gen) == 42
            # 2nd next(): feed token1 back (tokens_generated -> 1), sample
            # token2, yield again - paused right before token2's feedback.
            assert next(gen) == 42
            gen.close()   # GeneratorExit fires at that suspend point

        joined = _messages(caplog.records)
        assert ("gguf generate: aborted (cancelled) during decode, "
                "1 token(s) generated") in joined
        assert "gguf generate: complete" not in joined


class TestLlamaCppGenerateImageBoundaryLogging:
    """Same scheme, same three cases, for the OTHER live generation path in
    this file (_generate_image) - see the module docstring on _generate_image
    itself for why this path needed covering too: the real crash log this
    whole change was validated against was a vision-model load, so leaving
    _generate_image dark would have reproduced the exact gap being fixed."""

    def test_normal_generation_logs_all_boundaries_in_order(self, monkeypatch, caplog):
        monkeypatch.setattr(llama_mod, "_DECODE_PROGRESS_INTERVAL", 2)
        llm = _bare_llama_vision()
        llm._tokenizer.is_eog.return_value = False
        mock_api = _mock_native_api()

        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
             patch("localm.inference.backends.llamacpp.llama._build_sampler",
                   return_value=999), \
             caplog.at_level(logging.DEBUG, logger="localm"):
            tokens = list(llm._generate_image(
                _VISION_MESSAGES, max_new_tokens=6,
                temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1))

        assert tokens == [42] * 6
        assert llm.last_finish_reason == "length"

        msgs = [r.getMessage() for r in caplog.records]
        joined = "\n".join(msgs)
        assert "gguf generate (vision): prefill starting" in joined
        assert "gguf generate (vision): prefill complete" in joined
        assert "0 image(s)" in joined   # _VISION_MESSAGES carries no image_url part
        assert "gguf generate (vision): entering decode loop" in joined
        assert "gguf generate (vision): complete, 6 token(s)" in joined
        assert "finish_reason=length" in joined
        assert "aborted" not in joined

        progress = [r for r in caplog.records
                    if "gguf generate (vision): decode progress" in r.getMessage()]
        assert len(progress) == 3
        assert all(r.levelname == "DEBUG" for r in progress)
        boundaries = [r for r in caplog.records
                      if "gguf generate (vision): decode progress" not in r.getMessage()
                      and "gguf generate (vision)" in r.getMessage()]
        assert boundaries and all(r.levelname == "INFO" for r in boundaries)

        order = [i for i, m in enumerate(msgs) if any(
            key in m for key in ("prefill starting", "prefill complete",
                                  "entering decode loop", "generate (vision): complete"))]
        assert order == sorted(order)

    def test_prefill_failure_logs_start_but_not_complete_then_aborts(self, caplog):
        llm = _bare_llama_vision()
        llm._mtmd.eval_into.side_effect = RuntimeError("mtmd eval failed")
        mock_api = _mock_native_api()

        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
             caplog.at_level(logging.INFO, logger="localm"):
            with pytest.raises(RuntimeError, match="mtmd eval failed"):
                list(llm._generate_image(
                    _VISION_MESSAGES, max_new_tokens=6,
                    temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1))

        joined = _messages(caplog.records)
        assert "gguf generate (vision): prefill starting" in joined
        assert "gguf generate (vision): prefill complete" not in joined
        assert "gguf generate (vision): entering decode loop" not in joined
        assert ("gguf generate (vision): aborted (exception) during prefill, "
                "0 token(s) generated") in joined

    def test_cancellation_mid_decode_logs_aborted_with_partial_count(self, caplog):
        llm = _bare_llama_vision()
        llm._tokenizer.is_eog.return_value = False
        mock_api = _mock_native_api()

        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
             patch("localm.inference.backends.llamacpp.llama._build_sampler",
                   return_value=999), \
             caplog.at_level(logging.INFO, logger="localm"):
            gen = llm._generate_image(
                _VISION_MESSAGES, max_new_tokens=50,
                temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1)
            assert next(gen) == 42
            assert next(gen) == 42
            gen.close()

        joined = _messages(caplog.records)
        assert ("gguf generate (vision): aborted (cancelled) during decode, "
                "1 token(s) generated") in joined
        assert "gguf generate (vision): complete" not in joined


# ---------------------------------------------------------------------------
#  _runner.py: ModelRunner.chat_stream (parent-side)
# ---------------------------------------------------------------------------

class _FakeProc:
    """Stands in for the worker process's liveness check only - same shape
    as test_runner_stream_timeouts.py's _AliveProc, plus a crash-like
    exitcode so _death_report()/_exit_reason() have something to decode."""

    def __init__(self):
        self.terminated = False
        self.exitcode = -11   # signal-like, for a plausible crash message

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        return None


def _make_runner() -> ModelRunner:
    ctx = mp.get_context("spawn")
    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    r._proc = _FakeProc()
    return r


def _fake_child(r, stop, *, tokens, finish_reason="stop"):
    """Mimics the worker's observable protocol: waits for the request, then
    streams *tokens* and confirms "done"."""
    while not stop.is_set():
        try:
            cmd = r._req_q.get(timeout=0.05)
        except Exception:
            continue
        if cmd[0] != "chat_stream":
            continue
        for t in tokens:
            r._resp_q.put(("chunk", t))
        r._resp_q.put(("done", {"finish_reason": finish_reason}))
        return


class TestModelRunnerChatStreamBoundaryLogging:
    def test_normal_generation_logs_first_response_and_completion(
            self, monkeypatch, caplog):
        monkeypatch.setattr(runner_mod, "_STREAM_PROGRESS_INTERVAL", 2)
        r = _make_runner()
        stop = threading.Event()
        child = threading.Thread(
            target=_fake_child, args=(r, stop),
            kwargs=dict(tokens=["a", "b", "c", "d"]), daemon=True)
        child.start()
        try:
            with caplog.at_level(logging.DEBUG, logger="localm"):
                out = list(r.chat_stream(messages=[]))
        finally:
            stop.set()
            child.join(2)

        assert out == ["a", "b", "c", "d"]
        msgs = [rec.getMessage() for rec in caplog.records]
        joined = "\n".join(msgs)
        assert "gguf worker: prefill complete, first response after" in joined
        assert "kind=chunk" in joined
        assert ("gguf worker: generation complete, 4 token(s), "
                "finish_reason=stop") in joined

        # Decode progress is coarse (interval=2 over 4 chunks -> checkpoints
        # at 2 and 4) AND specifically DEBUG, not INFO - see
        # _STREAM_PROGRESS_INTERVAL for the ring-buffer-budget reasoning.
        progress = [r for r in caplog.records
                    if "gguf worker: decode progress" in r.getMessage()]
        assert len(progress) == 2
        assert all(r.levelname == "DEBUG" for r in progress)
        boundaries = [r for r in caplog.records
                      if "gguf worker: decode progress" not in r.getMessage()]
        assert boundaries and all(r.levelname == "INFO" for r in boundaries)

    def test_worker_death_before_any_response_reports_prefill_phase(self, caplog):
        r = _make_runner()

        def _child():
            r._req_q.get(timeout=5)
            r._proc.terminated = True   # crash: no envelope ever sent

        child = threading.Thread(target=_child, daemon=True)
        child.start()
        try:
            with caplog.at_level(logging.ERROR, logger="localm"):
                with pytest.raises(RuntimeError):
                    list(r.chat_stream(messages=[]))
        finally:
            child.join(2)

        joined = _messages(caplog.records)
        assert ("gguf worker: died mid-stream during prefill/dispatch "
                "(no response received yet)") in joined

    def test_worker_death_after_n_chunks_reports_decode_phase_with_count(
            self, caplog):
        r = _make_runner()

        def _child():
            r._req_q.get(timeout=5)
            r._resp_q.put(("chunk", "a"))
            r._resp_q.put(("chunk", "b"))
            r._proc.terminated = True   # crash mid-decode, no "done" ever sent

        child = threading.Thread(target=_child, daemon=True)
        child.start()
        try:
            with caplog.at_level(logging.ERROR, logger="localm"):
                with pytest.raises(RuntimeError):
                    list(r.chat_stream(messages=[]))
        finally:
            child.join(2)

        joined = _messages(caplog.records)
        assert ("gguf worker: died mid-stream during decode "
                "(2 token(s) already streamed)") in joined

    def test_cancellation_logs_aborted_with_partial_count(self, caplog):
        r = _make_runner()

        def _child():
            r._req_q.get(timeout=5)
            r._resp_q.put(("chunk", "a"))
            r._resp_q.put(("chunk", "b"))
            # Confirm the cancel promptly so the drain never has to time out.
            r._ctrl_q.get(timeout=5)
            r._resp_q.put(("done", {"finish_reason": "stop"}))

        child = threading.Thread(target=_child, daemon=True)
        child.start()
        try:
            with caplog.at_level(logging.INFO, logger="localm"):
                gen = r.chat_stream(messages=[])
                assert next(gen) == "a"
                assert next(gen) == "b"
                gen.close()
        finally:
            child.join(2)

        joined = _messages(caplog.records)
        assert ("gguf worker: generation cancelled by caller after "
                "2 token(s)") in joined
