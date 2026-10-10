# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token log probabilities on /v1/chat/completions and /v1/completions.

The engine stand-in turns scripted token records into text through the GGUF
worker's real decode chain (UTF-8 assembly, end-of-turn filter, marker scrub)
and the engine's scrub, sending the records ahead of the text the way the
worker's runner does, so every route under test sees exactly what a GGUF model
would produce."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.backends.llamacpp.llama import (
    _filtered_stream, _scrub_stream, _utf8_pieces)
from localm.inference.http_server import create_app
from localm.inference.tool_calling import CLOSE_TAG, OPEN_TAG
from localm.textnorm import scrub_stream

MODEL = "lp-model"
WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".localm"
    root.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(root))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", root)
    monkeypatch.setattr(_cfg, "MODELS_DIR", root / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", root / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", root / "registry.json")
    return root


def records_for(pieces) -> list:
    """One record per token: *pieces* are str or bytes token texts; token i
    gets logprob ``-(i + 1) / 10`` and three alternatives."""
    out = []
    for i, piece in enumerate(pieces):
        data = piece.encode("utf-8") if isinstance(piece, str) else piece
        lp = -(i + 1) / 10
        top = ((data, lp), (b"alt" + bytes([97 + i % 26]), lp - 1.0), (b"zz", lp - 2.0))
        out.append((data, lp, top))
    return out


class Served:
    """An app over an engine whose replies are the token *pieces*."""

    def __init__(self, pieces=("Hi",), supports=True):
        self.calls: list = []
        self.records = records_for(pieces)
        engine = MagicMock()

        def chat_stream(messages, logprobs=None, on_logprobs=None, **kwargs):
            self.calls.append((messages, dict(kwargs, logprobs=logprobs,
                                              on_logprobs=on_logprobs)))
            sink: list = []

            def scored():
                for record in self.records:
                    sink.append(record)
                    yield record[0]

            def worker():
                for piece in _scrub_stream(_filtered_stream(_utf8_pieces(scored()))):
                    if on_logprobs is not None and sink:
                        on_logprobs(list(sink))
                        sink.clear()
                    yield piece
                if on_logprobs is not None and sink:
                    on_logprobs(list(sink))

            return scrub_stream(worker())

        engine.chat_stream.side_effect = chat_stream
        engine.unsupported_sampling.side_effect = lambda names: []
        engine.supports_logprobs = supports
        engine.supports_grammar = True
        engine.display_name = MODEL
        engine.model_path = ""
        engine.count_tokens.return_value = 2
        engine.count_messages_tokens.return_value = 3
        engine.gpu_placement = None
        engine.last_finish_reason = "stop"
        engine.context_capacity.return_value = 4096
        engine.loaded = True
        self.engine = engine
        self.app = create_app(engine)
        self.client = TestClient(self.app)

    def chat(self, **extra):
        body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}
        return self.client.post("/v1/chat/completions", json=body)

    def complete(self, **extra):
        body = {"model": MODEL, "prompt": "Once", **extra}
        return self.client.post("/v1/completions", json=body)

    @property
    def kwargs(self):
        return self.calls[-1][1]


def sse(response):
    assert response.status_code == 200, response.text
    return [json.loads(line[5:]) for line in response.text.split("\n")
            if line.startswith("data:") and "[DONE]" not in line]


def streamed_chat(served, **extra):
    chunks = sse(served.chat(stream=True, **extra))
    text, entries = [], []
    for c in chunks:
        if not c.get("choices"):
            continue
        choice = c["choices"][0]
        if choice["delta"].get("content"):
            text.append(choice["delta"]["content"])
        if choice.get("logprobs"):
            entries.extend(choice["logprobs"]["content"])
        elif choice["delta"].get("content"):
            raise AssertionError(f"a content delta without logprobs: {c}")
    return "".join(text), entries


def spelled(entries) -> str:
    return "".join(e["token"] for e in entries)


# ------------------------------------------------------------------ request checks


@pytest.mark.parametrize("extra, needle", [
    ({"logprobs": True, "top_logprobs": 21}, "top_logprobs must be between 0 and 20"),
    ({"logprobs": True, "top_logprobs": -1}, "top_logprobs must be between 0 and 20"),
    ({"top_logprobs": 3}, "top_logprobs requires logprobs: true"),
    ({"logprobs": False, "top_logprobs": 2}, "top_logprobs requires logprobs: true"),
])
def test_an_invalid_chat_logprobs_request_is_a_400(home, extra, needle):
    served = Served()
    r = served.chat(**extra)
    assert r.status_code == 400, r.text
    assert needle in r.json()["detail"]
    assert served.calls == []


@pytest.mark.parametrize("extra, needle", [
    ({"logprobs": 21}, "logprobs must be between 0 and 20"),
    ({"logprobs": -1}, "logprobs must be between 0 and 20"),
    ({"logprobs": 2, "echo": True}, "echo with logprobs"),
])
def test_an_invalid_completion_logprobs_request_is_a_400(home, extra, needle):
    served = Served()
    r = served.complete(**extra)
    assert r.status_code == 400, r.text
    assert needle in r.json()["detail"]
    assert served.calls == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_model_that_cannot_report_logprobs_refuses_them_by_name(home, stream):
    served = Served(supports=False)
    r = served.chat(logprobs=True, stream=stream)
    assert r.status_code == 400, r.text
    assert "logprobs cannot be returned by model 'lp-model'" in r.json()["detail"]
    r = served.complete(logprobs=1, stream=stream)
    assert r.status_code == 400, r.text
    assert "logprobs cannot be returned by model 'lp-model'" in r.json()["detail"]
    assert served.calls == []


def test_a_streamed_request_is_refused_while_a_plugin_rewrites_the_stream(home):
    served = Served(pieces=["Hi", " there"])
    served.app.state.chat_pipeline.add_hook("stream", lambda token, ctx: token.upper(),
                                            plugin="shout")
    r = served.chat(logprobs=True, stream=True)
    assert r.status_code == 400, r.text
    assert "rewrites the stream" in r.json()["detail"]
    r = served.complete(logprobs=1, stream=True)
    assert r.status_code == 400, r.text
    r = served.chat(logprobs=True)
    assert r.status_code == 200, r.text
    assert spelled(r.json()["choices"][0]["logprobs"]["content"]) == "Hi there"


def test_without_logprobs_nothing_is_asked_of_the_backend(home):
    served = Served(pieces=["Hi"])
    r = served.chat()
    assert r.status_code == 200
    assert served.kwargs["logprobs"] is None and served.kwargs["on_logprobs"] is None
    assert r.json()["choices"][0]["logprobs"] is None
    r = served.complete()
    assert "logprobs" not in r.json()["choices"][0]
    assert served.kwargs["logprobs"] is None and served.kwargs["on_logprobs"] is None


def test_top_logprobs_zero_alone_changes_nothing(home):
    served = Served(pieces=["Hi"])
    r = served.chat(top_logprobs=0)
    assert r.status_code == 200
    assert served.kwargs["logprobs"] is None
    assert r.json()["choices"][0]["logprobs"] is None


# ------------------------------------------------------------------ chat shapes


def test_chat_reports_each_content_token_with_its_alternatives(home):
    served = Served(pieces=["The", " sky", " is", " blue", "."])
    r = served.chat(logprobs=True, top_logprobs=2)
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    content = choice["logprobs"]["content"]
    assert choice["message"]["content"] == "The sky is blue."
    assert [e["token"] for e in content] == ["The", " sky", " is", " blue", "."]
    assert [e["logprob"] for e in content] == pytest.approx([-0.1, -0.2, -0.3, -0.4, -0.5])
    assert content[1]["bytes"] == list(b" sky")
    assert content[1]["top_logprobs"] == [
        {"token": " sky", "logprob": pytest.approx(-0.2), "bytes": list(b" sky")},
        {"token": "altb", "logprob": pytest.approx(-1.2), "bytes": list(b"altb")}]
    assert served.kwargs["logprobs"] == 2 and callable(served.kwargs["on_logprobs"])


def test_logprobs_true_without_top_logprobs_reports_no_alternatives(home):
    served = Served(pieces=["Hi", "!"])
    content = served.chat(logprobs=True).json()["choices"][0]["logprobs"]["content"]
    assert [e["top_logprobs"] for e in content] == [[], []]
    assert served.kwargs["logprobs"] == 0


def test_streamed_and_whole_replies_report_the_same_tokens(home):
    pieces = ["<think>", "Let", " me", " think", "</think>", "\n\n", "Four", " is", " 2", "+2", "."]
    whole = Served(pieces=pieces).chat(logprobs=True, top_logprobs=1).json()["choices"][0]
    text, entries = streamed_chat(Served(pieces=pieces), logprobs=True, top_logprobs=1)
    assert text == whole["message"]["content"] == "\n\nFour is 2+2."
    assert entries == whole["logprobs"]["content"]


def test_reasoning_tokens_are_left_out(home):
    pieces = ["<think>", "Count", " the", " sides", "</think>", "\n\n", "Three", "."]
    choice = Served(pieces=pieces).chat(logprobs=True).json()["choices"][0]
    assert choice["message"]["reasoning_content"] == "Count the sides"
    assert spelled(choice["logprobs"]["content"]) == "\n\nThree."
    text, entries = streamed_chat(Served(pieces=pieces), logprobs=True)
    assert spelled(entries) == text == "\n\nThree."


def test_harmony_markers_and_reasoning_are_left_out(home):
    pieces = ["<|channel|>", "analysis", "<|message|>", "plan", " it",
              "<|channel|>", "final", "<|message|>", "Yes", "."]
    choice = Served(pieces=pieces).chat(logprobs=True).json()["choices"][0]
    assert choice["message"]["reasoning_content"] == "\nplan it\n"
    assert choice["message"]["content"] == "\nYes."
    assert [e["token"] for e in choice["logprobs"]["content"]] == ["Yes", "."]
    text, entries = streamed_chat(Served(pieces=pieces), logprobs=True)
    assert text == "\nYes." and [e["token"] for e in entries] == ["Yes", "."]


def test_an_end_of_turn_marker_ends_the_reply_inside_its_reasoning(home):
    pieces = ["<|channel|>", "analysis", "<|message|>", "plan", "<|end|>",
              "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "Yes"]
    choice = Served(pieces=pieces).chat(logprobs=True).json()["choices"][0]
    assert choice["message"]["content"] == ""
    assert choice["logprobs"] == {"content": []}


def test_a_stop_sequence_drops_the_tokens_after_it(home):
    pieces = ["one", ",", " two", ",", " ", "three", ",", " four"]
    choice = Served(pieces=pieces).chat(logprobs=True, stop=["three"]).json()["choices"][0]
    assert choice["message"]["content"] == "one, two, "
    want = ["one", ",", " two", ",", " "]
    assert [e["token"] for e in choice["logprobs"]["content"]] == want
    text, entries = streamed_chat(Served(pieces=pieces), logprobs=True, stop=["three"])
    assert text == "one, two, " and [e["token"] for e in entries] == want


def test_a_token_split_by_the_stop_sequence_is_kept(home):
    pieces = ["Red", ", green", " and blue"]
    choice = Served(pieces=pieces).chat(logprobs=True, stop=["green"]).json()["choices"][0]
    assert choice["message"]["content"] == "Red, "
    assert [e["token"] for e in choice["logprobs"]["content"]] == ["Red", ", green"]


def test_tool_call_tokens_are_left_out(home):
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
    pieces = ["Checking", ".", OPEN_TAG, call[:10], call[10:], CLOSE_TAG]
    choice = Served(pieces=pieces).chat(
        logprobs=True, tools=[WEATHER], tool_choice="auto").json()["choices"][0]
    assert choice["message"]["tool_calls"]
    assert choice["message"]["content"] == "Checking."
    assert spelled(choice["logprobs"]["content"]) == "Checking."
    text, entries = streamed_chat(Served(pieces=pieces), logprobs=True, tools=[WEATHER],
                                  tool_choice="auto")
    assert spelled(entries) == text == "Checking."


def test_a_reply_that_is_only_a_call_has_no_content_tokens(home):
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Oslo"}})
    pieces = [OPEN_TAG, call, CLOSE_TAG]
    choice = Served(pieces=pieces).chat(
        logprobs=True, tools=[WEATHER], tool_choice="required").json()["choices"][0]
    assert choice["message"]["tool_calls"]
    assert choice["logprobs"] == {"content": []}


def test_a_character_split_over_tokens_is_reported_on_every_token_that_spells_it(home):
    euro = "€".encode()
    pieces = [b"Price ", euro[:1], euro[1:], b" 5"]
    choice = Served(pieces=pieces).chat(logprobs=True).json()["choices"][0]
    assert choice["message"]["content"] == "Price € 5"
    content = choice["logprobs"]["content"]
    assert [e["bytes"] for e in content] == [list(b"Price "), list(euro[:1]),
                                             list(euro[1:]), list(b" 5")]
    assert b"".join(bytes(e["bytes"]) for e in content).decode() == "Price € 5"
    text, entries = streamed_chat(Served(pieces=pieces), logprobs=True)
    assert entries == content and text == "Price € 5"


def test_a_reply_cut_inside_a_character_still_aligns(home):
    euro = "€".encode()
    pieces = [b"Cost ", euro[:2]]
    choice = Served(pieces=pieces).chat(logprobs=True).json()["choices"][0]
    assert choice["message"]["content"] == "Cost �"
    assert [e["bytes"] for e in choice["logprobs"]["content"]] == [list(b"Cost "), list(euro[:2])]
    text, entries = streamed_chat(Served(pieces=pieces), logprobs=True)
    assert text == "Cost �" and len(entries) == 2


def test_an_end_of_turn_marker_written_as_text_is_left_out(home):
    pieces = ["Done", ".", "<|im_", "end|>", "ignored"]
    choice = Served(pieces=pieces).chat(logprobs=True).json()["choices"][0]
    assert choice["message"]["content"] == "Done."
    assert [e["token"] for e in choice["logprobs"]["content"]] == ["Done", "."]


def test_an_outlet_that_rewrites_the_reply_withholds_its_logprobs(home, caplog):
    served = Served(pieces=["secret", " text"])
    served.app.state.chat_pipeline.add_hook("outlet", lambda text, m, ctx: "[redacted]",
                                            plugin="redact")
    with caplog.at_level("WARNING", logger="localm"):
        choice = served.chat(logprobs=True).json()["choices"][0]
    assert choice["message"]["content"] == "[redacted]"
    assert choice["logprobs"] is None
    assert "outlet hook changed the reply" in caplog.text
    served.app.state.chat_pipeline.remove_plugin("redact")
    served.app.state.chat_pipeline.add_hook("outlet", lambda text, m, ctx: None,
                                            plugin="observer")
    choice = served.chat(logprobs=True).json()["choices"][0]
    assert spelled(choice["logprobs"]["content"]) == "secret text"


def test_text_the_tokens_do_not_spell_is_reported_not_guessed(home, caplog):
    served = Served(pieces=["Hello"])

    def mismatched(messages, logprobs=None, on_logprobs=None, **kwargs):
        if on_logprobs is not None:
            on_logprobs([(b"Bye", -0.5, ())])
        yield "Hello"

    served.engine.chat_stream.side_effect = mismatched
    with caplog.at_level("WARNING", logger="localm"):
        choice = served.chat(logprobs=True).json()["choices"][0]
        text, entries = streamed_chat_lenient(served)
    assert choice["message"]["content"] == "Hello"
    assert choice["logprobs"] is None
    assert text == "Hello" and entries == []
    assert caplog.text.count("could not be matched") >= 2


def streamed_chat_lenient(served):
    chunks = sse(served.chat(stream=True, logprobs=True))
    text, entries = [], []
    for c in chunks:
        if c.get("choices"):
            text.append(c["choices"][0]["delta"].get("content") or "")
            entries.extend((c["choices"][0].get("logprobs") or {}).get("content") or [])
    return "".join(text), entries


# ------------------------------------------------------------------ legacy completions


def test_a_completion_reports_the_legacy_shape(home):
    served = Served(pieces=["Paris", " is", " big", "."])
    r = served.complete(logprobs=1)
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    lp = choice["logprobs"]
    assert choice["text"] == "Paris is big."
    assert lp["tokens"] == ["Paris", " is", " big", "."]
    assert lp["token_logprobs"] == pytest.approx([-0.1, -0.2, -0.3, -0.4])
    assert lp["text_offset"] == [0, 5, 8, 12]
    assert lp["top_logprobs"][1] == {" is": pytest.approx(-0.2)}
    r = served.complete(logprobs=2)
    assert r.json()["choices"][0]["logprobs"]["top_logprobs"][1] == {
        " is": pytest.approx(-0.2), "altb": pytest.approx(-1.2)}


def test_a_completion_with_zero_logprobs_still_reports_the_sampled_token(home):
    lp = Served(pieces=["A", "B"]).complete(logprobs=0).json()["choices"][0]["logprobs"]
    assert lp["top_logprobs"] == [{"A": pytest.approx(-0.1)}, {"B": pytest.approx(-0.2)}]


def test_a_streamed_completion_carries_logprobs_per_chunk(home):
    pieces = ["Paris", " is", " big", ".", " Also", " old"]
    whole = Served(pieces=pieces).complete(logprobs=1, stop=["Also"]).json()["choices"][0]
    chunks = sse(Served(pieces=pieces).complete(logprobs=1, stop=["Also"], stream=True))
    texts, tokens, offsets = [], [], []
    for c in chunks:
        choice = c["choices"][0] if c.get("choices") else None
        if choice and choice.get("text"):
            texts.append(choice["text"])
            tokens += choice["logprobs"]["tokens"]
            offsets += choice["logprobs"]["text_offset"]
    assert "".join(texts) == whole["text"] == "Paris is big. "
    assert tokens == whole["logprobs"]["tokens"] == ["Paris", " is", " big", ".", " Also"]
    assert offsets == whole["logprobs"]["text_offset"]


# ------------------------------------------------------------------ the scorer's arithmetic


def test_logprobs_of_a_full_distribution_sum_to_one():
    """Pure-Python reference for what the native scorer reports: the
    alternatives of a row whose vocabulary is all listed sum to one."""
    from localm.inference.backends.llamacpp._logprobs import clamp_logprob
    logits = [2.0, 1.0, 0.5, -3.0]
    m = max(logits)
    lse = m + math.log(sum(math.exp(x - m) for x in logits))
    probs = [math.exp(clamp_logprob(x - lse)) for x in logits]
    assert sum(probs) == pytest.approx(1.0)
    assert clamp_logprob(float("-inf")) == -9999.0
    assert clamp_logprob(float("nan")) == -9999.0
    assert clamp_logprob(1e-7) == 0.0
