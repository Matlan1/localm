# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lazy (text-or-tool) grammar plumbing, end to end.

The lazy grammar leaves generation unconstrained until the output matches a
trigger pattern (e.g. <tool_call>), then the GBNF grammar enforces from that
point, so thinking models think freely and a started tool call must be valid.
These tests pin the sampler-chain selection, the never-silently-strict
fallbacks, and the server API contract; the @integration test proves activation
on a real model."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localm.inference.backends.base import (
    GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE,
    GRAMMAR_LAZY_UNSUPPORTED_MESSAGE,
    GrammarUnsupportedError,
)
from localm.inference.backends.llamacpp.llama import _build_sampler
from localm.inference.gbnf import TOOL_CALL_TRIGGER, TOOL_CALLS_ONLY

_API = "localm.inference.backends.llamacpp.llama.api"


class TestBuildSamplerLazySelection:
    def _mock_api(self, lazy_supported=True):
        mock_api = MagicMock()
        mock_api.has_lazy_grammar.return_value = lazy_supported
        return mock_api

    def test_lazy_with_triggers_uses_lazy_init(self):
        mock_api = self._mock_api()
        with patch(_API, mock_api):
            _build_sampler(vocab=1, grammar="root ::= \"x\"", grammar_lazy=True,
                           grammar_triggers=["[\\s\\S]*?(<t>[\\s\\S]*)"])
        mock_api.llama_sampler_init_grammar_lazy_patterns.assert_called_once_with(
            1, b'root ::= "x"', b"root", [b"[\\s\\S]*?(<t>[\\s\\S]*)"])
        mock_api.llama_sampler_init_grammar.assert_not_called()

    def test_lazy_without_triggers_refuses_and_never_falls_back(self):
        # A lazy request without triggers raises instead of building either a
        # strict grammar stage or no grammar stage at all.
        mock_api = self._mock_api()
        with patch(_API, mock_api):
            with pytest.raises(GrammarUnsupportedError) as ei:
                _build_sampler(vocab=1, grammar="root ::= \"x\"", grammar_lazy=True)
        assert GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE in str(ei.value)
        mock_api.llama_sampler_init_grammar_lazy_patterns.assert_not_called()
        mock_api.llama_sampler_init_grammar.assert_not_called()

    def test_lazy_on_grammarless_build_refuses_and_never_falls_back(self):
        mock_api = self._mock_api(lazy_supported=False)
        with patch(_API, mock_api):
            with pytest.raises(GrammarUnsupportedError) as ei:
                _build_sampler(vocab=1, grammar="root ::= \"x\"", grammar_lazy=True,
                               grammar_triggers=["p"])
        assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE in str(ei.value)
        mock_api.llama_sampler_init_grammar_lazy_patterns.assert_not_called()
        mock_api.llama_sampler_init_grammar.assert_not_called()

    def test_refusal_frees_the_chain_on_both_arms(self):
        """A refusal must not leak the native sampler chain it had already
        allocated. Both InvalidGrammarError arms in _build_sampler free it before
        raising; these two must match, or a client retrying a lazy request
        against an old build leaks one chain per attempt."""
        for kwargs in ({}, {"grammar_triggers": ["p"]}):
            mock_api = self._mock_api(lazy_supported=not kwargs)
            mock_api.llama_sampler_chain_init.return_value = 4242
            with patch(_API, mock_api):
                with pytest.raises(GrammarUnsupportedError):
                    _build_sampler(vocab=1, grammar="root ::= \"x\"",
                                   grammar_lazy=True, **kwargs)
            mock_api.llama_sampler_free.assert_called_once_with(4242)

    def test_the_three_grammar_messages_are_mutually_non_containing(self):
        """``coder/agent/context.py`` routes a refusal by SUBSTRING match against
        these strings, testing the lazy one first. If any were a substring of
        another, a no-triggers refusal (a caller bug) would latch
        _lazy_grammar_confirmed_unsupported and disable trigger-gated tool calls
        for the whole session, blaming the backend."""
        from localm.inference.backends.base import GRAMMAR_UNSUPPORTED_MESSAGE
        msgs = [GRAMMAR_UNSUPPORTED_MESSAGE, GRAMMAR_LAZY_UNSUPPORTED_MESSAGE,
                GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE]
        for a in msgs:
            for b in msgs:
                if a is not b:
                    assert a not in b, f"{a!r} is a substring of {b!r}"

    def test_strict_path_unchanged(self):
        mock_api = self._mock_api()
        with patch(_API, mock_api):
            _build_sampler(vocab=1, grammar="root ::= \"x\"")
        mock_api.llama_sampler_init_grammar.assert_called_once_with(
            1, b'root ::= "x"', b"root")
        mock_api.llama_sampler_init_grammar_lazy_patterns.assert_not_called()


class TestServerApiContract:
    def _client_and_engine(self):
        from localm.inference.http_server import create_app
        engine = MagicMock()
        _state = {"loaded": True}
        engine.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])
        engine.count_tokens.return_value = 1
        engine.display_name = "test-model"
        engine.supports_images = False
        engine.can_be_multimodal = False
        engine.last_finish_reason = "stop"
        type(engine).loaded = property(lambda self: _state["loaded"])
        return TestClient(create_app(engine)), engine

    def test_lazy_without_triggers_is_400(self):
        client, _ = self._client_and_engine()
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": TOOL_CALLS_ONLY, "grammar_lazy": True,
        })
        assert r.status_code == 400
        assert "grammar_triggers" in r.text

    def test_lazy_without_grammar_is_400(self):
        client, _ = self._client_and_engine()
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar_lazy": True, "grammar_triggers": [TOOL_CALL_TRIGGER],
        })
        assert r.status_code == 400

    def test_lazy_fields_reach_the_engine(self):
        client, engine = self._client_and_engine()
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": TOOL_CALLS_ONLY, "grammar_lazy": True,
            "grammar_triggers": [TOOL_CALL_TRIGGER],
        })
        assert r.status_code == 200
        kw = engine.chat_stream.call_args.kwargs
        assert kw.get("grammar") == TOOL_CALLS_ONLY
        assert kw.get("grammar_lazy") is True
        assert kw.get("grammar_triggers") == [TOOL_CALL_TRIGGER]

    def test_plain_grammar_does_not_send_lazy_fields(self):
        client, engine = self._client_and_engine()
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": TOOL_CALLS_ONLY,
        })
        assert r.status_code == 200
        kw = engine.chat_stream.call_args.kwargs
        assert "grammar_lazy" not in kw


class TestCoderBodyForwardsLazyFields:
    def test_body_includes_lazy_fields(self):
        from localm.plugins.coder.backends.http import HTTPBackend
        b = HTTPBackend("http://127.0.0.1:8080/v1", "test-model")
        body = b._body([{"role": "user", "content": "hi"}], stream=False,
                       grammar=TOOL_CALLS_ONLY, grammar_lazy=True,
                       grammar_triggers=[TOOL_CALL_TRIGGER])
        assert body["grammar"] == TOOL_CALLS_ONLY
        assert body["grammar_lazy"] is True
        assert body["grammar_triggers"] == [TOOL_CALL_TRIGGER]


class TestSupportsGrammarIsLocalmOnly:
    """Grammar kwargs are sent BY DEFAULT now, so supports_grammar must mean
    what its comment always claimed: OUR server only. The old URL blacklist
    mislabelled every third-party OpenAI-compatible server as grammar-capable
    (unknown body fields can 400 there)."""

    def test_localm_backend_supports_grammar(self):
        from localm.plugins.coder.backends.http import make_localm_backend
        assert make_localm_backend("m", port=8642).supports_grammar is True

    def test_arbitrary_url_does_not(self):
        from localm.plugins.coder.backends.http import HTTPBackend
        # A hand-typed --backend URL (LM Studio, vLLM, a remote server) must
        # not receive grammar kwargs it may reject.
        assert HTTPBackend("http://127.0.0.1:1234/v1", "m").supports_grammar is False
        assert HTTPBackend("https://my-remote.example/v1", "m").supports_grammar is False

    def test_explicit_flag_enables(self):
        from localm.plugins.coder.backends.http import HTTPBackend
        b = HTTPBackend("https://my-remote.example/v1", "m", localm_server=True)
        assert b.supports_grammar is True


# --------------------------------------------------------------------------- #
# Real-model proof: force the trigger into the generated stream, then the
# continuation MUST be a valid tool call, while the pre-trigger text flowed
# unconstrained.
# --------------------------------------------------------------------------- #

_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"


@pytest.mark.integration
@pytest.mark.real_gguf
def test_lazy_grammar_activates_at_trigger_on_real_model():
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned: {e}")
    from localm.inference.backends.llamacpp import _api as api
    if not api.has_lazy_grammar():
        pytest.skip("this llama build lacks the lazy grammar export")
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_REPO, filename=_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_REPO}/{_FILE}: {e}")

    from localm.inference.backends.llamacpp._structs import llama_token
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    llm = LlamaCpp(path, n_ctx=1024, n_gpu_layers=99, verbose=False)
    try:
        vocab = api.llama_model_get_vocab(llm._model_ptr)
        smp = api.llama_sampler_init_grammar_lazy_patterns(
            vocab, TOOL_CALLS_ONLY.encode(), b"root", [TOOL_CALL_TRIGGER.encode()])
        assert smp, "lazy grammar sampler failed to init"
        cp = api.llama_sampler_chain_default_params()
        cp.no_perf = True
        chain = api.llama_sampler_chain_init(cp)
        api.llama_sampler_chain_add(chain, smp)
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_greedy())
        try:
            prompt_toks = llm._tokenizer.encode(
                "<|im_start|>user\nRead main.py with the read_file tool."
                "<|im_end|>\n<|im_start|>assistant\n")
            arr = (llama_token * len(prompt_toks))(*prompt_toks)
            assert api.llama_decode(llm._ctx_ptr,
                                    api.llama_batch_get_one(arr, len(prompt_toks))) == 0

            # Force the trigger into the GENERATED stream (accept + decode each
            # token, exactly as if the model emitted it), then let it continue.
            out = []
            for t in llm._tokenizer.encode("Reading the file now. <tool_call>",
                                           add_bos=False):
                api.llama_sampler_accept(chain, t)
                one = (llama_token * 1)(t)
                assert api.llama_decode(llm._ctx_ptr,
                                        api.llama_batch_get_one(one, 1)) == 0
                out.append(llm._tokenizer.token_to_piece(t))
            for _ in range(100):
                tok = api.llama_sampler_sample(chain, llm._ctx_ptr, -1)
                if llm._tokenizer.is_eog(tok):
                    break
                out.append(llm._tokenizer.token_to_piece(tok))
                one = (llama_token * 1)(tok)
                if api.llama_decode(llm._ctx_ptr,
                                    api.llama_batch_get_one(one, 1)) != 0:
                    break
            text = "".join(out)
        finally:
            api.llama_sampler_free(chain)

        from localm.plugins.coder.parser import parse_tool_calls
        calls = parse_tool_calls(text)
        assert calls, f"the post-trigger continuation must be a valid tool call: {text!r}"
        assert calls[0].name, "the forced-valid call carries a name"
    finally:
        llm.close() if hasattr(llm, "close") else None
