# SPDX-License-Identifier: AGPL-3.0-or-later
"""``POST /v1/completions`` and Ollama ``/api/generate`` with ``raw: true`` are
raw text completions: the prompt reaches the backend as ``raw_prompt``, with no
chat template and no inlet hook, and every backend layer passes it down to the
tokenizer as given.

API callers get their own prompt unchanged: the chat plugin's style and
thinking notes apply only to localm's own chat (``X-Client-Id: chat``), and the
``chat_style_note`` setting turns the style note off there too.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.plugins.builtin.chat.plug import _ROLE_WORD_INSTRUCTION, _THINK_INSTRUCTION
from tests.test_completions_hooks_audit import HOOK_PLUGIN, _install, _make_plugin, _sse_text


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


PROMPT = "def fibonacci(n):\n    if n < 2:\n        return n\n    return"


def _recording_engine(reply=" fibonacci(n - 1) + fibonacci(n - 2)",
                      name="test-model", raw=True):
    """Engine stub recording every chat_stream call; it yields *reply*."""
    engine = MagicMock()
    engine.display_name = name
    engine.count_tokens.return_value = 5
    engine.supports_raw_prompt = raw
    engine.supports_images = True
    type(engine).loaded = property(lambda self: True)
    calls: list = []

    def _stream(messages, **kw):
        calls.append({"messages": json.loads(json.dumps(messages)), **kw})
        return iter([reply])

    engine.chat_stream.side_effect = _stream
    engine.calls = calls
    return engine


def _app(engine):
    from localm.inference.http_server import create_app
    return create_app(engine)


# --------------------------------------------------------------------------- #
#  /v1/completions                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stream", [False, True])
def test_the_prompt_reaches_the_backend_as_a_raw_prompt(env, stream):
    engine = _recording_engine()
    with TestClient(_app(engine)) as c:
        r = c.post("/v1/completions", json={
            "model": "m", "prompt": PROMPT, "max_tokens": 40, "temperature": 0,
            "stream": stream})
    assert r.status_code == 200, r.text
    assert len(engine.calls) == 1
    assert engine.calls[0]["raw_prompt"] == PROMPT
    text = _sse_text(r) if stream else r.json()["choices"][0]["text"]
    assert text == " fibonacci(n - 1) + fibonacci(n - 2)"


@pytest.mark.parametrize("stream", [False, True])
def test_echo_prepends_the_prompt_exactly_once(env, stream):
    engine = _recording_engine(reply=", there was a")
    with TestClient(_app(engine)) as c:
        r = c.post("/v1/completions", json={
            "model": "m", "prompt": "Once upon a time", "echo": True, "stream": stream})
    assert r.status_code == 200, r.text
    assert engine.calls[0]["raw_prompt"] == "Once upon a time"
    text = _sse_text(r) if stream else r.json()["choices"][0]["text"]
    assert text == "Once upon a time, there was a"


@pytest.mark.parametrize("stream", [False, True])
def test_no_inlet_hook_touches_the_prompt(env, stream):
    engine = _recording_engine(reply="ok")
    app = _app(engine)
    _make_plugin(env / "plugins", "hookplug", HOOK_PLUGIN)
    _install(app, env / "plugins", "hookplug")
    with TestClient(app) as c:
        r = c.post("/v1/completions", json={
            "model": "m", "prompt": "hi", "stream": stream},
            headers={"X-Client-Id": "chat"})
    assert r.status_code == 200, r.text
    call = engine.calls[0]
    assert call["raw_prompt"] == "hi"
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_sampling_stop_and_budget_reach_the_backend(env):
    engine = _recording_engine(reply="abc STOP def")
    with TestClient(_app(engine)) as c:
        r = c.post("/v1/completions", json={
            "model": "m", "prompt": "p", "max_tokens": 7, "temperature": 0.3,
            "top_p": 0.8, "top_k": 9, "seed": 3, "stop": ["STOP"]})
    assert r.status_code == 200, r.text
    call = engine.calls[0]
    assert (call["max_tokens"], call["temperature"], call["top_p"], call["top_k"],
            call["seed"]) == (7, 0.3, 0.8, 9, 3)
    assert r.json()["choices"][0]["text"] == "abc "
    assert r.json()["usage"]["prompt_tokens"] == 5


@pytest.mark.parametrize("stream", [False, True])
def test_a_spent_token_budget_finishes_with_length(env, stream):
    engine = _recording_engine(reply="abc")
    engine.last_finish_reason = "length"
    with TestClient(_app(engine)) as c:
        r = c.post("/v1/completions", json={
            "model": "m", "prompt": "p", "max_tokens": 1, "stream": stream})
        cut = c.post("/v1/completions", json={
            "model": "m", "prompt": "p", "stop": ["b"], "stream": stream})
    assert r.status_code == 200 and cut.status_code == 200
    if stream:
        reasons = [json.loads(line[6:])["choices"][0]["finish_reason"]
                   for resp in (r, cut) for line in resp.text.splitlines()
                   if line.startswith("data: {") and json.loads(line[6:])["choices"]]
        assert "length" in reasons and reasons.count("stop") == 1
    else:
        assert r.json()["choices"][0]["finish_reason"] == "length"
        assert cut.json()["choices"][0]["finish_reason"] == "stop"


def test_a_backend_without_a_raw_path_is_refused_not_answered_as_chat(env):
    engine = _recording_engine(raw=False)
    with TestClient(_app(engine)) as c:
        r = c.post("/v1/completions", json={"model": "m", "prompt": "hi"})
    assert r.status_code == 400
    assert "raw prompt" in r.json()["detail"]
    assert engine.calls == []


def test_the_debug_log_shows_the_raw_prompt(env, monkeypatch):
    import localm.debuglog as dl
    import localm.inference.http_server as hs
    seen: list = []
    monkeypatch.setattr(dl, "debug_content_enabled", lambda: True)
    monkeypatch.setattr(dl.logger, "debug", lambda msg, *a, **k: seen.append(msg % a))
    hs._log_assembled_prompt([{"role": "user", "content": PROMPT}], PROMPT)
    assert seen and "raw completion prompt" in seen[0] and seen[0].endswith(PROMPT)


# --------------------------------------------------------------------------- #
#  Ollama /api/generate raw: true                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stream", [False, True])
def test_ollama_raw_generate_is_a_raw_completion(env, stream):
    engine = _recording_engine(reply=" world")
    with TestClient(_app(engine)) as c:
        r = c.post("/api/generate", json={
            "model": "test-model", "prompt": "hello", "raw": True, "stream": stream,
            "system": "ignored in raw mode", "options": {"num_predict": 12,
                                                        "temperature": 0}})
    assert r.status_code == 200, r.text
    call = engine.calls[0]
    assert call["raw_prompt"] == "hello"
    assert call["max_tokens"] == 12
    if stream:
        lines = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        assert "".join(line.get("response", "") for line in lines) == " world"
        assert lines[-1]["done"] is True
    else:
        body = r.json()
        assert body["response"] == " world" and body["done"] is True


def test_ollama_raw_generate_refuses_images(env):
    engine = _recording_engine()
    with TestClient(_app(engine)) as c:
        r = c.post("/api/generate", json={
            "model": "test-model", "prompt": "hello", "raw": True,
            "images": ["iVBORw0KGgo="], "stream": False})
    assert r.status_code == 400
    assert "raw" in r.json()["error"]
    assert engine.calls == []


# --------------------------------------------------------------------------- #
#  The backend layers pass raw_prompt down and skip the chat template          #
# --------------------------------------------------------------------------- #

def test_engine_refuses_a_raw_prompt_its_backend_cannot_take():
    from localm.inference.backends.base import UnsupportedInputError
    from localm.inference.engine import Engine
    engine = Engine.__new__(Engine)
    engine._backend = MagicMock(spec=["loaded", "chat_stream"])
    with pytest.raises(UnsupportedInputError):
        engine.chat_stream([{"role": "user", "content": "x"}], raw_prompt="x")
    engine._backend.chat_stream.assert_not_called()


def test_the_gguf_text_path_tokenises_the_raw_prompt_without_the_template(monkeypatch):
    from localm.inference.backends.llamacpp import llama as L
    from tests._bare_llama import make_bare_llama
    llm = make_bare_llama(_model_ptr=1, _ctx_ptr=1)
    template = MagicMock(side_effect=AssertionError("chat template applied"))
    monkeypatch.setattr(L, "_apply_model_template", template)
    encoded: list = []
    llm._tokenizer.encode = (lambda text, add_bos=True, untrusted_ranges=():
                             encoded.append((text, add_bos, untrusted_ranges)) or [1, 2])
    got: dict = {}
    monkeypatch.setattr(llm, "_generate", lambda tokens, **kw: got.update(
        tokens=tokens, **kw) or iter(()))
    llm.create_chat_completion([{"role": "user", "content": "ignored"}],
                               raw_prompt=PROMPT, thinking=False)
    template.assert_not_called()
    assert encoded == [(PROMPT, True, ())]
    assert got["tokens"] == [1, 2]


def test_a_raw_prompt_that_spells_its_start_token_gets_no_second_one(monkeypatch):
    from tests._bare_llama import make_bare_llama
    llm = make_bare_llama(_model_ptr=1, _ctx_ptr=1)
    encoded: list = []
    llm._tokenizer.encode = (lambda text, add_bos=True, untrusted_ranges=():
                             encoded.append(add_bos) or [1])
    monkeypatch.setattr(llm, "_generate", lambda tokens, **kw: iter(()))
    llm.create_chat_completion([], raw_prompt="<s>[INST] hi [/INST]")
    llm.create_chat_completion([], raw_prompt="[INST] hi [/INST]")
    assert encoded == [False, True]


def test_the_gguf_raw_path_refuses_a_prompt_that_tokenises_to_nothing(monkeypatch):
    from localm.inference.backends.base import UnsupportedInputError
    from tests._bare_llama import make_bare_llama
    llm = make_bare_llama(_model_ptr=1, _ctx_ptr=1)
    llm._tokenizer.encode = lambda text, add_bos=True, untrusted_ranges=(): []
    with pytest.raises(UnsupportedInputError):
        llm.create_chat_completion([], raw_prompt="")


def test_the_gguf_worker_and_backend_pass_raw_prompt_down(tmp_path):
    from localm.inference.backends.gguf import GgufBackend
    from localm.inference.backends.llamacpp._worker import GgufWorker
    w = GgufWorker("m.gguf", None, 512, 0, None, 512)
    captured: dict = {}

    class _Llm:
        def create_chat_completion(self, **kw):
            captured.update(kw)
            return iter(())
    w._llm = _Llm()
    list(w.chat_stream([{"role": "user", "content": "hi"}], raw_prompt="hi"))
    assert captured["raw_prompt"] == "hi"
    captured.clear()
    list(w.chat_stream([{"role": "user", "content": "hi"}]))
    assert "raw_prompt" not in captured

    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 64)
    b = GgufBackend(str(path))
    assert b.supports_raw_prompt is True
    runner = MagicMock()
    runner.chat_stream.side_effect = lambda **kw: iter(["x"])
    b._runner = runner
    list(b.chat_stream([{"role": "user", "content": "hi"}], raw_prompt="hi"))
    assert runner.chat_stream.call_args.kwargs["raw_prompt"] == "hi"


def test_the_hf_text_path_tokenises_the_raw_prompt_without_the_template(monkeypatch):
    import sys
    import types

    from localm.inference.backends import _hf_worker as hfmod
    be = hfmod.HFWorker.__new__(hfmod.HFWorker)
    be._processor = None
    be._is_multimodal = False
    be._model = MagicMock()
    be.context_capacity = 1024
    tok = MagicMock()
    tok.apply_chat_template.side_effect = AssertionError("chat template applied")
    ids = MagicMock()
    ids.shape = (1, 6)
    tok.return_value.to.return_value = {"input_ids": ids, "attention_mask": MagicMock()}
    be._tokenizer = tok

    class _FakeStreamer:
        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            return iter(())

        def end(self):
            pass

    fake = types.ModuleType("transformers")
    fake.TextIteratorStreamer = _FakeStreamer
    fake.StoppingCriteriaList = list
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(hfmod, "_grammar_processor", lambda *a, **k: None)
    list(be.chat_stream([{"role": "user", "content": "ignored"}], raw_prompt=PROMPT,
                        thinking=False, max_tokens=8))
    tok.apply_chat_template.assert_not_called()
    assert tok.call_args.args == (PROMPT,)
    assert tok.call_args.kwargs.get("add_special_tokens", True) is True
    assert be._model.generate.call_args.kwargs["input_ids"] is ids


def test_the_hf_raw_path_refuses_a_prompt_that_tokenises_to_nothing(monkeypatch):
    import sys
    import types

    from localm.inference.backends import _hf_worker as hfmod
    from localm.inference.backends.base import EMPTY_RAW_PROMPT_MESSAGE, UnsupportedInputError
    be = hfmod.HFWorker.__new__(hfmod.HFWorker)
    be._processor = None
    be._is_multimodal = False
    be._model = MagicMock()
    be.context_capacity = 1024
    tok = MagicMock()
    ids = MagicMock()
    ids.shape = (1, 0)
    tok.return_value.to.return_value = {"input_ids": ids, "attention_mask": MagicMock()}
    be._tokenizer = tok

    class _FakeStreamer:
        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            return iter(())

        def end(self):
            pass

    fake = types.ModuleType("transformers")
    fake.TextIteratorStreamer = _FakeStreamer
    fake.StoppingCriteriaList = list
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(hfmod, "_grammar_processor", lambda *a, **k: None)
    caught = None
    try:
        list(be.chat_stream([{"role": "user", "content": "ignored"}], raw_prompt="",
                            thinking=False, max_tokens=8))
    except UnsupportedInputError as e:
        caught = e
    be._model.generate.assert_not_called()
    assert caught is not None
    assert str(caught) == EMPTY_RAW_PROMPT_MESSAGE


def test_the_hf_backend_passes_raw_prompt_to_its_worker():
    from localm.inference.backends.hf import HFBackend
    b = HFBackend("does-not-need-to-exist")
    assert b.supports_raw_prompt is True
    runner = MagicMock(spec=["chat_stream", "last_done"])
    runner.chat_stream.side_effect = lambda **kw: iter(["x"])
    runner.last_done = {}
    b._runner = runner
    b._loaded = True
    assert list(b.chat_stream([{"role": "user", "content": "hi"}], raw_prompt="hi")) == ["x"]
    assert runner.chat_stream.call_args.kwargs["raw_prompt"] == "hi"


# --------------------------------------------------------------------------- #
#  The chat plugin's notes: own chat surface only                              #
# --------------------------------------------------------------------------- #

SYSTEM = "You always answer with exactly the single word BANANA and nothing else."


def _routes():
    return [
        ("/v1/chat/completions", {"model": "qwen3-8b", "stream": False, "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]}),
        ("/v1/messages", {"model": "qwen3-8b", "max_tokens": 16, "system": SYSTEM,
                          "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "qwen3-8b", "instructions": SYSTEM, "input": "hi"}),
        ("/api/chat", {"model": "qwen3-8b", "stream": False, "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]}),
    ]


def _system_text(call) -> str:
    return "\n".join(m["content"] for m in call["messages"]
                     if m.get("role") == "system" and isinstance(m.get("content"), str))


@pytest.mark.parametrize("path,body", _routes(), ids=[p for p, _ in _routes()])
def test_an_api_callers_system_prompt_arrives_unchanged(env, path, body):
    engine = _recording_engine(reply="BANANA", name="qwen3-8b")
    with TestClient(_app(engine)) as c:
        r = c.post(path, json=body)
    assert r.status_code == 200, r.text
    assert len(engine.calls) == 1
    assert _system_text(engine.calls[0]) == SYSTEM


def test_an_api_caller_with_no_system_prompt_gets_none_inserted(env):
    engine = _recording_engine(reply="Paris", name="qwen3-8b")
    with TestClient(_app(engine)) as c:
        r = c.post("/v1/chat/completions", json={"model": "qwen3-8b", "messages": [
            {"role": "user", "content": "What is the capital of France? One word."}]})
    assert r.status_code == 200, r.text
    assert engine.calls[0]["messages"] == [
        {"role": "user", "content": "What is the capital of France? One word."}]


def test_localms_own_chat_still_gets_both_notes(env):
    engine = _recording_engine(reply="BANANA", name="qwen3-8b")
    with TestClient(_app(engine)) as c:
        r = c.post("/v1/chat/completions", headers={"X-Client-Id": "chat"}, json={
            "model": "qwen3-8b", "messages": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    system = _system_text(engine.calls[0])
    assert system.startswith(SYSTEM)
    assert _ROLE_WORD_INSTRUCTION in system and _THINK_INSTRUCTION in system


def test_the_setting_turns_the_style_note_off_for_the_own_chat(env):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg["chat_style_note"] = False
    save_config(cfg)
    engine = _recording_engine(reply="BANANA", name="qwen3-8b")
    with TestClient(_app(engine)) as c:
        r = c.post("/v1/chat/completions", headers={"X-Client-Id": "chat"}, json={
            "model": "qwen3-8b", "messages": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    system = _system_text(engine.calls[0])
    assert _ROLE_WORD_INSTRUCTION not in system
    assert _THINK_INSTRUCTION in system


def test_the_coder_surface_gets_neither_note(env):
    engine = _recording_engine(reply="ok", name="qwen3-8b")
    with TestClient(_app(engine)) as c:
        r = c.post("/v1/chat/completions", headers={"X-Client-Id": "coder"}, json={
            "model": "qwen3-8b", "messages": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert _system_text(engine.calls[0]) == SYSTEM
