# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token log probabilities on /v1/responses.

The engine stand-in turns scripted token records into text through the GGUF
worker's real decode chain (UTF-8 assembly, end-of-turn filter, marker scrub)
and the engine's scrub, sending the records ahead of the text the way the
worker's runner does, so the Responses route sees exactly what a GGUF model
would produce through the chat route."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference import responses_protocol as R
from localm.inference.backends.llamacpp.llama import (
    _filtered_stream, _scrub_stream, _utf8_pieces)
from localm.inference.http_server import create_app
from localm.textnorm import scrub_stream

MODEL = "resp-lp-model"
INCLUDE = ["message.output_text.logprobs"]


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
    from localm.inference.routes import responses as routes
    routes.STORE = R.ResponseStore()
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

    def post(self, **extra):
        body = {"model": MODEL, "input": "hi", **extra}
        return self.client.post("/v1/responses", json=body)

    @property
    def kwargs(self):
        return self.calls[-1][1]


def events(response):
    assert response.status_code == 200, response.text
    out = []
    kind = None
    for line in response.text.split("\n"):
        if line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:"):
            data = json.loads(line[5:])
            assert data["type"] == kind
            out.append(data)
    return out


def spelled(entries) -> str:
    return "".join(e["token"] for e in entries)


def message_part(body):
    [item] = [i for i in body["output"] if i["type"] == "message"]
    return item["content"][0]


# ------------------------------------------------------------------ request


def test_include_asks_the_chat_route_for_logprobs_with_top_logprobs(home):
    served = Served(pieces=["The", " sky"])
    r = served.post(include=INCLUDE, top_logprobs=3)
    assert r.status_code == 200, r.text
    assert served.kwargs["logprobs"] == 3 and callable(served.kwargs["on_logprobs"])


def test_include_without_top_logprobs_asks_for_no_alternatives(home):
    served = Served(pieces=["Hi", "!"])
    part = message_part(served.post(include=INCLUDE).json())
    assert served.kwargs["logprobs"] == 0
    assert [e["top_logprobs"] for e in part["logprobs"]] == [[], []]


def test_top_logprobs_without_include_returns_none(home):
    served = Served(pieces=["Hi"])
    r = served.post(top_logprobs=3)
    assert r.status_code == 200, r.text
    assert served.kwargs["logprobs"] is None and served.kwargs["on_logprobs"] is None
    assert message_part(r.json())["logprobs"] == []


def test_without_include_nothing_is_asked_of_the_backend(home):
    served = Served(pieces=["Hi"])
    r = served.post()
    assert served.kwargs["logprobs"] is None and served.kwargs["on_logprobs"] is None
    assert message_part(r.json())["logprobs"] == []


@pytest.mark.parametrize("value", [-1, 21])
def test_top_logprobs_outside_zero_to_twenty_is_a_400(home, value):
    served = Served()
    r = served.post(include=INCLUDE, top_logprobs=value)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and err["param"] == "top_logprobs"
    assert "between 0 and 20" in err["message"]
    assert served.calls == []


def test_the_limits_of_top_logprobs_are_accepted(home):
    served = Served(pieces=["Hi"])
    for value in (0, 20):
        r = served.post(include=INCLUDE, top_logprobs=value)
        assert r.status_code == 200, r.text
        assert served.kwargs["logprobs"] == value


# ------------------------------------------------------------------ non-stream


def test_the_output_text_part_reports_each_token_with_its_alternatives(home):
    served = Served(pieces=["The", " sky", " is", " blue", "."])
    body = served.post(include=INCLUDE, top_logprobs=2).json()
    part = message_part(body)
    entries = part["logprobs"]
    assert part["text"] == "The sky is blue." == spelled(entries)
    assert all(e["logprob"] <= 0 for e in entries)
    assert [e["logprob"] for e in entries] == pytest.approx([-0.1, -0.2, -0.3, -0.4, -0.5])
    assert entries[1] == {
        "token": " sky", "logprob": pytest.approx(-0.2), "bytes": list(b" sky"),
        "top_logprobs": [
            {"token": " sky", "logprob": pytest.approx(-0.2), "bytes": list(b" sky")},
            {"token": "altb", "logprob": pytest.approx(-1.2), "bytes": list(b"altb")}]}


def test_reasoning_tokens_are_left_out(home):
    pieces = ["<think>", "Count", " the", " sides", "</think>", "\n\n", "Three", "."]
    body = Served(pieces=pieces).post(include=INCLUDE).json()
    [reasoning] = [i for i in body["output"] if i["type"] == "reasoning"]
    assert reasoning["content"][0]["text"] == "Count the sides"
    part = message_part(body)
    assert spelled(part["logprobs"]) == part["text"] == "\n\nThree."


def test_a_reply_that_is_only_a_call_has_no_message_logprobs(home):
    from localm.inference.tool_calling import CLOSE_TAG, OPEN_TAG
    reply = f'{OPEN_TAG}{json.dumps({"name": "f", "arguments": {}})}{CLOSE_TAG}'
    tool = {"type": "function", "name": "f", "parameters": {"type": "object"}}
    body = Served(pieces=[reply]).post(include=INCLUDE, tools=[tool]).json()
    assert [i["type"] for i in body["output"]] == ["function_call"]


def test_include_may_name_the_logprobs_value_among_accepted_ones(home):
    served = Served(pieces=["Hi"])
    r = served.post(include=["reasoning.encrypted_content", INCLUDE[0]], top_logprobs=1)
    assert r.status_code == 200, r.text
    assert served.kwargs["logprobs"] == 1 and message_part(r.json())["logprobs"]


def test_a_stored_reply_with_logprobs_continues_without_them(home):
    served = Served(pieces=["Hi", " there"])
    first = served.post(include=INCLUDE).json()
    assert message_part(first)["logprobs"]
    r = served.post(previous_response_id=first["id"], input="and then?")
    assert r.status_code == 200, r.text
    sent = json.dumps(served.calls[-1][0])
    assert "logprob" not in sent and "Hi there" in sent
    assert message_part(r.json())["logprobs"] == []


def test_an_outlet_that_rewrites_the_reply_withholds_its_logprobs(home):
    served = Served(pieces=["secret", " text"])
    served.app.state.chat_pipeline.add_hook("outlet", lambda text, m, ctx: "[redacted]",
                                            plugin="redact")
    r = served.post(include=INCLUDE, top_logprobs=1)
    assert r.status_code == 200, r.text
    part = message_part(r.json())
    assert part["text"] == "[redacted]" and part["logprobs"] == []


# ------------------------------------------------------------------ stream


def streamed(served, **extra):
    return events(served.post(stream=True, include=INCLUDE, **extra))


def test_streamed_deltas_carry_their_tokens_and_the_close_carries_all(home):
    pieces = ["The", " sky", " is", " blue", "."]
    whole = message_part(Served(pieces=pieces).post(include=INCLUDE, top_logprobs=2).json())
    evs = streamed(Served(pieces=pieces), top_logprobs=2)
    deltas = [e for e in evs if e["type"] == "response.output_text.delta"]
    assert deltas
    assert all(isinstance(d["logprobs"], list) for d in deltas)
    assert "".join(d["delta"] for d in deltas) == whole["text"]
    assert [e for d in deltas for e in d["logprobs"]] == whole["logprobs"]
    assert spelled(whole["logprobs"]) == whole["text"]
    [done] = [e for e in evs if e["type"] == "response.output_text.done"]
    assert done["logprobs"] == whole["logprobs"] and done["text"] == whole["text"]
    [part_done] = [e for e in evs if e["type"] == "response.content_part.done"]
    assert part_done["part"]["logprobs"] == whole["logprobs"]
    [item_done] = [e for e in evs if e["type"] == "response.output_item.done"]
    assert item_done["item"]["content"][0]["logprobs"] == whole["logprobs"]
    final = [e for e in evs if e["type"] == "response.completed"][0]["response"]
    assert message_part(final)["logprobs"] == whole["logprobs"]
    assert all(e["logprob"] <= 0 for e in whole["logprobs"])


def test_streamed_reasoning_deltas_have_no_logprobs_and_the_reply_has_its_own(home):
    pieces = ["<think>", "Let", " me", "</think>", "\n\n", "Four", "."]
    evs = streamed(Served(pieces=pieces))
    assert not [e for e in evs if e["type"] == "response.reasoning_text.delta"
                and "logprobs" in e]
    [done] = [e for e in evs if e["type"] == "response.output_text.done"]
    assert spelled(done["logprobs"]) == done["text"] == "\n\nFour."


def test_a_character_split_over_tokens_streams_every_entry(home):
    pieces = ["caf", b"\xc3", b"\xa9", "!"]
    whole = message_part(Served(pieces=pieces).post(include=INCLUDE).json())
    evs = streamed(Served(pieces=pieces))
    deltas = [e for e in evs if e["type"] == "response.output_text.delta"]
    assert "".join(d["delta"] for d in deltas) == whole["text"] == "café!"
    assert [e for d in deltas for e in d["logprobs"]] == whole["logprobs"]
    [done] = [e for e in evs if e["type"] == "response.output_text.done"]
    assert done["logprobs"] == whole["logprobs"]
    assert len(whole["logprobs"]) == 4


def run_stream(chunks):
    """The Responses events for chat *chunks* (dicts) with the logprobs include."""
    import asyncio

    async def source():
        for chunk in chunks:
            yield chunk

    async def collect():
        shell = R.Shell(R.ResponsesRequest(model=MODEL, include=INCLUDE))
        return [line async for line in R.response_stream(shell, source())]

    return [json.loads(raw.decode().split("data: ", 1)[1]) for raw in asyncio.run(collect())]


def chunk(text, *tokens, finish=None):
    lp = {"content": [{"token": t, "logprob": -0.5, "bytes": list(t.encode()),
                       "top_logprobs": []} for t in tokens]} if tokens else None
    delta = {"content": text} if text else {}
    return {"model": MODEL, "choices": [{"index": 0, "delta": delta, "logprobs": lp,
                                         "finish_reason": finish}]}


def test_text_that_looks_like_an_error_keeps_its_logprobs_when_more_text_follows():
    evs = run_stream([chunk("[inference error is a phrase", "[inference error", " is a phrase"),
                      chunk(".", "."), chunk(None, finish="stop")])
    deltas = [e for e in evs if e["type"] == "response.output_text.delta"]
    assert [d["delta"] for d in deltas] == ["[inference error is a phrase", "."]
    assert [spelled(d["logprobs"]) for d in deltas] == ["[inference error is a phrase", "."]
    [done] = [e for e in evs if e["type"] == "response.output_text.done"]
    assert spelled(done["logprobs"]) == done["text"] == "[inference error is a phrase."


def test_text_that_looks_like_an_error_at_the_end_keeps_its_logprobs():
    evs = run_stream([chunk("[inference error: x", "[inference", " error: x"),
                      chunk(None, finish="stop")])
    [delta] = [e for e in evs if e["type"] == "response.output_text.delta"]
    assert spelled(delta["logprobs"]) == delta["delta"] == "[inference error: x"
    assert evs[-1]["type"] == "response.completed"


def test_entries_sent_without_text_join_the_next_delta_or_the_close():
    evs = run_stream([chunk(None, "a"), chunk("xy", "b"), chunk(None, "c"),
                      chunk(None, finish="stop")])
    [delta] = [e for e in evs if e["type"] == "response.output_text.delta"]
    assert delta["delta"] == "xy" and spelled(delta["logprobs"]) == "ab"
    [done] = [e for e in evs if e["type"] == "response.output_text.done"]
    assert spelled(done["logprobs"]) == "abc"
    final = evs[-1]["response"]
    assert spelled(message_part(final)["logprobs"]) == "abc"


def test_a_stream_without_include_keeps_empty_logprobs(home):
    evs = events(Served(pieces=["Hi", "!"]).post(stream=True, top_logprobs=2))
    for e in evs:
        if e["type"] in ("response.output_text.delta", "response.output_text.done"):
            assert e["logprobs"] == []


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("stream", [False, True])
def test_a_model_that_cannot_report_logprobs_is_refused_by_name(home, stream):
    served = Served(supports=False)
    r = served.post(include=INCLUDE, stream=stream)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "'resp-lp-model'" in err["message"] and "logprobs" in err["message"]
    assert served.calls == []


def test_a_streamed_request_is_refused_while_a_plugin_rewrites_the_stream(home):
    served = Served(pieces=["Hi", " there"])
    served.app.state.chat_pipeline.add_hook("stream", lambda token, ctx: token.upper(),
                                            plugin="shout")
    r = served.post(include=INCLUDE, stream=True)
    assert r.status_code == 400, r.text
    assert "rewrites the stream" in r.json()["error"]["message"]
    assert message_part(served.post(include=INCLUDE).json())["logprobs"]


# ------------------------------------------------------------------ units


@pytest.mark.parametrize("block", [None, {}, {"content": None}, {"content": "x"}, 5,
                                   {"content": [None, 3]}])
def test_logprob_entries_of_nothing_or_junk_is_empty(block):
    assert R.logprob_entries(block) == []


def test_logprob_entries_normalises_missing_fields():
    [entry] = R.logprob_entries({"content": [{"token": "a", "logprob": -1.0,
                                              "top_logprobs": [{"token": "b", "logprob": -2.0}, 7]}]})
    assert entry == {"token": "a", "logprob": -1.0, "bytes": [],
                     "top_logprobs": [{"token": "b", "logprob": -2.0, "bytes": []}]}
