# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for GgufWorker - the child-process-resident class that owns the
real native LlamaCpp instance (see llamacpp/_worker.py). No subprocess is
involved here: these exercise the class directly, in-process, the same way
GgufBackend's own unit tests always have - only the isolation BOUNDARY around
this class is new (see test_gguf_runner_isolation.py for that), not its logic.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.base import InvalidGrammarError
from localm.inference.backends.llamacpp._worker import GgufWorker


def _worker(model_path="model.gguf", **overrides):
    kwargs = dict(
        model_path=model_path,
        mmproj_path=None,
        n_ctx=4096,
        n_gpu_layers=99,
        n_ctx_max=None,
        n_ctx_grow=4096,
        cancel_event=None,
    )
    kwargs.update(overrides)
    return GgufWorker(**kwargs)


class _StubLlm:
    def __init__(self, tokens=None, model_ptr=1234, n_layers=32,
                 kv_bytes_per_token=90_000, supports_images=False):
        self._model_ptr = model_ptr
        self.n_layers = n_layers
        self.kv_bytes_per_token = kv_bytes_per_token
        self.supports_images = supports_images
        self._tokens = tokens if tokens is not None else [1, 2, 3]

    def tokenize(self, text, add_bos=True):
        return list(self._tokens)

    def check_grammar(self, grammar):
        pass

    def close(self):
        pass


class TestLoad:
    def test_load_wires_check_context_fit_and_returns_metadata(self, tmp_path):
        """End to end: GgufWorker.load() must pass ITS OWN bound
        _check_context_fit as LlamaCpp's vram_check (context-growth guard),
        and report back the metadata GgufBackend needs to cache
        (n_layers/kv_bytes_per_token/supports_images) - this is the exact
        wiring test retargeted from test_vram_preflight.py's
        test_load_native_wires_check_context_fit_into_llamacpp, now that the
        real LlamaCpp construction lives here instead of on GgufBackend."""
        w = _worker(str(tmp_path / "m.gguf"))
        fake_llamacpp_module = MagicMock()
        fake_llamacpp_module.LlamaCpp.return_value = _StubLlm(
            n_layers=42, kv_bytes_per_token=12_345, supports_images=True)
        with patch("localm.inference.backends.llamacpp._loader.load_lib"), \
             patch.dict(sys.modules,
                        {"localm.inference.backends.llamacpp": fake_llamacpp_module}):
            meta = w.load()
        _, kwargs = fake_llamacpp_module.LlamaCpp.call_args
        assert kwargs.get("vram_check") == w._check_context_fit
        assert kwargs.get("cancel_event") is None
        assert kwargs.get("n_gpu_layers") == 99
        assert meta == {"n_layers": 42, "kv_bytes_per_token": 12_345,
                         "supports_images": True}
        assert w._loaded is True

    def test_load_passes_resolved_gpu_layers_and_ctx_max_verbatim(self, tmp_path):
        """The worker never re-derives auto-sizing - it trusts whatever the
        parent already resolved, so preflight and the actual load can never
        disagree."""
        w = _worker(str(tmp_path / "m.gguf"), n_gpu_layers=24, n_ctx_max=32768)
        assert w.effective_gpu_layers == 24   # mirrors n_gpu_layers, set in __init__
        fake_llamacpp_module = MagicMock()
        fake_llamacpp_module.LlamaCpp.return_value = _StubLlm()
        with patch("localm.inference.backends.llamacpp._loader.load_lib"), \
             patch.dict(sys.modules,
                        {"localm.inference.backends.llamacpp": fake_llamacpp_module}):
            w.load()
        _, kwargs = fake_llamacpp_module.LlamaCpp.call_args
        assert kwargs.get("n_gpu_layers") == 24
        assert kwargs.get("n_ctx_max") == 32768


class TestTokenisation:
    def test_count_tokens(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = _StubLlm(tokens=[1, 2, 3, 4, 5])
        assert w.count_tokens("hello world") == 5

    def test_count_messages_tokens_applies_template_and_tokenizes(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = _StubLlm(tokens=[1, 2, 3])
        with patch("localm.inference.backends.llamacpp.llama._apply_model_template",
                   return_value="<|im_start|>user\nhi<|im_end|>\n") as fake_template:
            count = w.count_messages_tokens([{"role": "user", "content": "hi"}])
        assert count == 3
        args, _ = fake_template.call_args
        assert args[0] == w._llm._model_ptr
        assert args[1] == [{"role": "user", "content": "hi"}]

    def test_count_messages_tokens_flattens_multipart_content(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = _StubLlm(tokens=[1])
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "part one"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": "part two"},
        ]}]
        with patch("localm.inference.backends.llamacpp.llama._apply_model_template",
                   return_value="prompt") as fake_template:
            w.count_messages_tokens(messages)
        args, _ = fake_template.call_args
        assert args[1] == [{"role": "user", "content": "part one part two"}]

    def test_check_grammar_delegates_to_llm(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = MagicMock()
        w.check_grammar("root ::= \"a\"")
        w._llm.check_grammar.assert_called_once_with("root ::= \"a\"")

    def test_check_grammar_noop_for_empty(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = MagicMock()
        w.check_grammar("")
        w._llm.check_grammar.assert_not_called()

    def test_check_grammar_propagates_invalid_grammar_error(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = MagicMock()
        w._llm.check_grammar.side_effect = InvalidGrammarError("bad grammar")
        with pytest.raises(InvalidGrammarError):
            w.check_grammar("not valid gbnf {{{")


class _ChatStreamLlm:
    """Stub whose create_chat_completion can fail once (grammar fault) then
    succeed on retry, or fail unconditionally, mirroring the real native
    OSError-on-grammar-fault shape."""

    def __init__(self, fail_with_grammar=False, fail_always=False, tokens=("a", "b", "c")):
        self.fail_with_grammar = fail_with_grammar
        self.fail_always = fail_always
        self.tokens = tokens
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_always:
            raise OSError("native access violation")
        if self.fail_with_grammar and kwargs.get("grammar") is not None:
            raise OSError("grammar sampler fault")
        for t in self.tokens:
            yield {"choices": [{"delta": {"content": t}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


class TestChatStream:
    def test_grammar_fault_retries_without_grammar_when_nothing_yielded_yet(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = _ChatStreamLlm(fail_with_grammar=True)
        out = list(w.chat_stream([{"role": "user", "content": "hi"}], grammar="root ::= \"x\""))
        assert out == ["a", "b", "c"]
        assert w.grammar_unsupported_this_call is True
        assert w.last_finish_reason == "stop"
        # First call attempted with grammar, retry omitted it.
        assert w._llm.calls[0]["grammar"] == "root ::= \"x\""
        assert w._llm.calls[1]["grammar"] is None

    def test_non_grammar_fault_propagates_uncaught(self, tmp_path):
        """A fault unrelated to grammar (or one after tokens were already
        yielded) must NOT be softened here - it propagates out of the
        generator uncaught, so the runner's dispatch loop (and ultimately the
        whole child process) is allowed to die, matching voice.py's crash
        containment philosophy: an unknown-state native fault should not
        limp along in a possibly-corrupted context."""
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = _ChatStreamLlm(fail_always=True)
        gen = w.chat_stream([{"role": "user", "content": "hi"}])
        with pytest.raises(OSError):
            list(gen)

    def test_fault_after_partial_yield_propagates_even_with_grammar(self, tmp_path):
        """If tokens were already emitted, a later fault must not silently
        retry (that would duplicate/corrupt output) - propagate as-is."""
        class _PartialThenFail:
            def create_chat_completion(self, **kwargs):
                yield {"choices": [{"delta": {"content": "a"}, "finish_reason": None}]}
                raise OSError("fault mid-stream")

        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = _PartialThenFail()
        gen = w.chat_stream([{"role": "user", "content": "hi"}], grammar="root ::= \"x\"")
        collected = []
        with pytest.raises(OSError):
            for tok in gen:
                collected.append(tok)
        assert collected == ["a"]

    def test_clean_generation_reports_no_grammar_fault(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = _ChatStreamLlm()
        out = list(w.chat_stream([{"role": "user", "content": "hi"}]))
        assert out == ["a", "b", "c"]
        assert w.grammar_unsupported_this_call is False
        assert w.last_finish_reason == "stop"


class TestClose:
    def test_close_clears_llm_on_success(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w._llm = _StubLlm()
        w._loaded = True
        w.close()
        assert w._llm is None
        assert w._loaded is False

    def test_close_logs_but_does_not_raise_on_native_failure(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        stub = _StubLlm()
        stub.close = MagicMock(side_effect=RuntimeError("native teardown fault"))
        w._llm = stub
        w._loaded = True
        w.close()   # must not raise
        assert w._llm is None
        assert w._loaded is False

    def test_close_is_noop_when_never_loaded(self, tmp_path):
        w = _worker(str(tmp_path / "m.gguf"))
        w.close()   # must not raise
        assert w._llm is None
