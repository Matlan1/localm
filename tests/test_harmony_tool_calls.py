# SPDX-License-Identifier: AGPL-3.0-or-later
"""gpt-oss (harmony) function calls written as GGUF text.

A harmony call is a commentary message addressed to ``functions.NAME``:
``<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}``,
ended by the ``<|call|>`` end-of-generation token (which the GGUF backend never
yields). The scrubber rewrites it into a canonical ``<tool_call>`` block, so the
server's tool-call parser and the coder's parser both read it as a call and the
analysis message before it stays reasoning."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.backends.llamacpp.llama import (
    _STOP_STRINGS, _filtered_stream, _scrub_stream)
from localm.inference.engine import Engine
from localm.inference.http_server import _ReplyRouter, create_app
from localm.textnorm import scrub_stream, scrub_text, split_think

ANALYSIS = "<|channel|>analysis<|message|>Need to use function get_weather.<|end|>"
HEADER = "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"
ROLE_HEADER = "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>"
ARGS = '{"location":"San Francisco"}'
CALL_BLOCK = '<tool_call>\n{"name": "get_weather", "arguments": ' + ARGS + '}\n</tool_call>'
HARMONY_STOPS = _STOP_STRINGS - {"<|end|>"}

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                   "required": ["location"]}}}


def _chunks(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def _gguf_chain(raw, size):
    """The GGUF backend's text pipeline followed by the engine's scrub."""
    return scrub_stream(_scrub_stream(_filtered_stream(iter(_chunks(raw, size)), HARMONY_STOPS)))


class TestScrub:
    def test_a_call_after_analysis_becomes_a_tool_call_block(self):
        out = scrub_text(ANALYSIS + HEADER + ARGS)
        assert out == "<think>\nNeed to use function get_weather.\n</think>" + CALL_BLOCK

    def test_the_recipient_in_the_role_section_is_read_too(self):
        assert scrub_text(ANALYSIS + ROLE_HEADER + ARGS).endswith("</think>" + CALL_BLOCK)

    def test_a_call_without_analysis_closes_no_think_block(self):
        assert scrub_text(HEADER + ARGS) == CALL_BLOCK

    def test_call_token_in_the_text_closes_the_call_once(self):
        assert scrub_text(HEADER + ARGS + "<|call|>") == CALL_BLOCK

    def test_a_message_after_the_call_ends_it(self):
        out = scrub_text(HEADER + ARGS + "<|end|><|start|>assistant<|channel|>final<|message|>Done.")
        assert out == CALL_BLOCK + "\n</think>\nDone."

    def test_a_final_message_after_a_call_does_not_close_think_again(self):
        text = ANALYSIS + HEADER + ARGS + "<|end|><|start|>assistant<|channel|>final<|message|>Done."
        out = scrub_text(text)
        assert out.count("</think>") == 1
        router = _ReplyRouter(None, {"get_weather"}, {})
        events = router.feed(out) + router.flush()
        assert "".join(v for k, v in events if k == "text").strip() == "Done."
        assert [v.name for k, v in events if k == "call"] == ["get_weather"]

    def test_two_calls_are_two_blocks(self):
        other = HEADER.replace("get_weather", "get_time")
        out = scrub_text(HEADER + ARGS + "<|call|>" + other + "{}")
        assert out == CALL_BLOCK + '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'

    def test_the_block_closes_where_the_arguments_end(self):
        args = '{"a": {"b": [1, {"c": "}]\\" {"}]}, "d": "x"}'
        assert json.loads(args)["a"]["b"][1]["c"] == '}]" {'
        out = scrub_text(HEADER + args + " trailing")
        assert out == ('<tool_call>\n{"name": "get_weather", "arguments": ' + args
                       + '}\n</tool_call> trailing')
        assert json.loads(out[len("<tool_call>\n"):out.index("\n</tool_call>")])["arguments"] \
            == json.loads(args)

    def test_arguments_cut_short_leave_the_block_open(self):
        out = scrub_text(HEADER + '{"location": "San Fr')
        assert out == '<tool_call>\n{"name": "get_weather", "arguments": {"location": "San Fr'

    def test_arguments_that_are_not_json_close_only_at_a_boundary(self):
        assert scrub_text(HEADER + "Paris") == \
            '<tool_call>\n{"name": "get_weather", "arguments": Paris'
        assert scrub_text(HEADER + "Paris<|call|>") == \
            '<tool_call>\n{"name": "get_weather", "arguments": Paris}\n</tool_call>'

    def test_a_recipient_without_the_functions_prefix_is_a_call(self):
        bare = HEADER.replace("functions.get_weather", "get_weather")
        assert scrub_text(ANALYSIS + bare + ARGS).endswith("</think>" + CALL_BLOCK)
        role = ROLE_HEADER.replace("functions.get_weather", "get_weather")
        assert scrub_text(role + ARGS) == CALL_BLOCK

    @pytest.mark.parametrize("recipient", ["python", "browser.search", "container.exec"])
    @pytest.mark.parametrize("constraint", [" code", " json", " <|constrain|>json", ""])
    def test_builtin_recipients_stay_reasoning(self, recipient, constraint):
        for header in (f"<|channel|>analysis to={recipient}{constraint}<|message|>",
                       f"<|start|>assistant to={recipient}<|channel|>commentary{constraint}<|message|>"):
            out = scrub_text(ANALYSIS + header + '{"q": 1}')
            assert "<tool_call>" not in out, out
            content, reasoning = split_think(out)
            assert not content.strip() and '{"q": 1}' in reasoning, out

    def test_a_function_named_like_a_builtin_is_a_call(self):
        out = scrub_text(HEADER.replace("get_weather", "python") + ARGS)
        assert out.startswith('<tool_call>\n{"name": "python"')

    def test_a_name_over_64_characters_is_not_a_call(self):
        out = scrub_text(HEADER.replace("get_weather", "x" * 65) + ARGS)
        assert "<tool_call>" not in out

    def test_a_reply_cut_inside_the_header_leaves_no_call(self):
        for end in range(len(HEADER)):
            out = scrub_text(ANALYSIS + HEADER[:end])
            assert "<tool_call>" not in out and "}" not in out, end

    def test_streaming_matches_one_shot_at_every_split_and_chunk_size(self):
        for text in (ANALYSIS + HEADER + ARGS, ANALYSIS + ROLE_HEADER + ARGS + "<|call|>",
                     "prose " + HEADER + ARGS + "<|end|><|channel|>final<|message|>ok"):
            want = scrub_text(text)
            for i in range(len(text) + 1):
                assert "".join(scrub_stream(iter([text[:i], text[i:]]))) == want, (text, i)
            for size in (1, 2, 3, 5, 8, 13):
                assert "".join(scrub_stream(iter(_chunks(text, size)))) == want, (text, size)

    def test_scrubbing_is_idempotent(self):
        once = scrub_text(ANALYSIS + HEADER + ARGS)
        assert scrub_text(once) == once


class TestReplyRouter:
    @pytest.mark.parametrize("size", [1, 3, 7, 1000])
    def test_the_call_and_the_reasoning_are_routed_apart(self, size):
        router = _ReplyRouter(None, {"get_weather"}, {})
        events = []
        for piece in _gguf_chain(ANALYSIS + HEADER + ARGS, size):
            events += router.feed(piece)
        events += router.flush()
        assert "".join(v for k, v in events if k == "reasoning").strip() == \
            "Need to use function get_weather."
        assert not "".join(v for k, v in events if k == "text").strip()
        [call] = [v for k, v in events if k == "call"]
        assert (call.name, call.arguments) == ("get_weather", {"location": "San Francisco"})

    def test_a_call_to_a_function_not_offered_stays_visible_text(self):
        router = _ReplyRouter(None, {"get_time"}, {})
        text = scrub_text(HEADER + ARGS)
        events = router.feed(text) + router.flush()
        assert not [v for k, v in events if k == "call"]
        assert "get_weather" in "".join(v for k, v in events if k == "text")

    def test_a_call_cut_short_is_visible_text_not_a_call(self):
        router = _ReplyRouter(None, {"get_weather"}, {})
        events = router.feed(scrub_text(HEADER + '{"location": "San')) + router.flush()
        assert not [v for k, v in events if k == "call"]
        assert '"San' in "".join(v for k, v in events if k == "text")


def test_scrub_map_keeps_the_close_when_it_settles_exactly_at_the_arguments_end():
    from localm.textnorm import _MARKER_HOLD, ScrubMap

    head = "p" * 10 + HEADER
    text = head + ARGS + "q" * (_MARKER_HOLD + 40)
    args_end = len(head) + len(ARGS)
    want = scrub_text(text)
    for first in range(args_end - 3, args_end + _MARKER_HOLD + 4):
        m = ScrubMap()
        m.spans(text[:first])
        got = "".join(text[s0:s1] if rep is None else rep for s0, s1, _o0, _o1, rep in m.spans(text))
        assert got == want, first


def test_logprobs_of_a_call_followed_by_long_text_stream_like_the_whole_reply():
    from localm.inference.backends.llamacpp.llama import _utf8_pieces
    from localm.inference.logprobs import LogprobAligner

    raw = ANALYSIS + HEADER + ARGS + "q" * 400
    tokens = [raw[:len(ANALYSIS) + len(HEADER)].encode()] + [c.encode() for c in raw[len(ANALYSIS) + len(HEADER):]]
    recs = [(t, -float(i), ((t, -float(i)),)) for i, t in enumerate(tokens)]
    for tools in (None, {"get_weather"}, {"f"}):
        router = _ReplyRouter(None, tools, {})
        aligner = LogprobAligner()
        sink: list = []

        def scored(sink=sink):
            for r in recs:
                sink.append(r)
                yield r[0]

        taken: list = []
        seen = 0
        for piece in scrub_stream(_scrub_stream(_filtered_stream(_utf8_pieces(scored()), HARMONY_STOPS))):
            aligner.add(sink)
            sink.clear()
            aligner.received(piece)
            router.feed(piece)
            taken += aligner.take(router.text_spans[seen:])
            seen = len(router.text_spans)
        aligner.add(sink)
        aligner.finish()
        router.flush()
        taken += aligner.take(router.text_spans[seen:])
        assert b"".join(e["bytes"] for e in taken).decode().endswith("q" * 400), tools


@pytest.mark.parametrize("after", ["<|im_end|>tail", ""])
def test_logprobs_of_a_rejected_call_cover_its_tokens(after):
    from localm.inference.backends.llamacpp.llama import _utf8_pieces
    from localm.inference.logprobs import LogprobAligner

    raw = ANALYSIS + HEADER + ARGS + after
    tokens = [t.encode() for t in _chunks(raw, 7)]
    got = "".join(scrub_stream(_scrub_stream(
        _filtered_stream(_utf8_pieces(iter(tokens)), HARMONY_STOPS))))
    assert got.endswith(CALL_BLOCK)
    router = _ReplyRouter(None, {"get_time"}, {})
    router.feed(got)
    router.flush()
    aligner = LogprobAligner()
    aligner.add([(t, -float(i), ((t, -float(i)),)) for i, t in enumerate(tokens)])
    aligner.finish()
    aligner.received(got)
    spelled = b"".join(e["bytes"] for e in aligner.take(router.text_spans)).decode()
    assert "get_weather" in spelled and ARGS in spelled and "tail" not in spelled


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


def _client(raw, size=4):
    engine = MagicMock()
    engine.chat_stream.side_effect = lambda messages, **kwargs: _gguf_chain(raw, size)
    engine.display_name = "gpt-oss"
    engine.model_path = ""
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.gpu_placement = None
    engine.last_finish_reason = "stop"
    engine.context_capacity.return_value = 4096
    engine.loaded = True
    return TestClient(create_app(engine))


_BODY = {"model": "gpt-oss", "messages": [{"role": "user", "content": "Weather in SF?"}],
         "tools": [WEATHER]}


class TestChatCompletions:
    def test_non_streamed_reply_carries_tool_calls(self, home):
        r = _client(ANALYSIS + HEADER + ARGS).post("/v1/chat/completions", json=_BODY)
        assert r.status_code == 200, r.text
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        message = choice["message"]
        assert message["reasoning_content"].strip() == "Need to use function get_weather."
        assert not (message["content"] or "").strip()
        [tc] = message["tool_calls"]
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "San Francisco"}

    def test_streamed_reply_carries_one_tool_call_delta(self, home):
        r = _client(ANALYSIS + ROLE_HEADER + ARGS, size=3).post(
            "/v1/chat/completions", json={**_BODY, "stream": True})
        assert r.status_code == 200, r.text
        deltas = [json.loads(line[5:])["choices"][0]["delta"]
                  for line in r.text.split("\n")
                  if line.startswith("data:") and "[DONE]" not in line
                  and json.loads(line[5:]).get("choices")]
        assert not "".join(d.get("content") or "" for d in deltas).strip()
        assert "get_weather" in "".join(d.get("reasoning_content") or "" for d in deltas)
        calls = [tc for d in deltas for tc in (d.get("tool_calls") or [])]
        assert [c["function"]["name"] for c in calls] == ["get_weather"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"location": "San Francisco"}


_SCHEMA = WEATHER["function"]["parameters"]
_BRIDGED = [
    ("/v1/responses",
     {"input": "Weather in SF?",
      "tools": [{"type": "function", "name": "get_weather", "parameters": _SCHEMA}]},
     lambda b: [(i["name"], json.loads(i["arguments"])) for i in b["output"]
                if i["type"] == "function_call"]),
    ("/v1/messages",
     {"max_tokens": 64, "messages": [{"role": "user", "content": "Weather in SF?"}],
      "tools": [{"name": "get_weather", "input_schema": _SCHEMA}]},
     lambda b: [(c["name"], c["input"]) for c in b["content"] if c["type"] == "tool_use"]),
    ("/api/chat",
     {"stream": False, "messages": [{"role": "user", "content": "Weather in SF?"}],
      "tools": [WEATHER]},
     lambda b: [(c["function"]["name"], c["function"]["arguments"])
                for c in b["message"]["tool_calls"]]),
]


@pytest.mark.parametrize("path,body,calls_of", _BRIDGED, ids=["responses", "anthropic", "ollama"])
def test_the_bridged_routes_return_the_call(home, path, body, calls_of):
    client = _client(ANALYSIS + HEADER + ARGS)
    headers = {"Authorization": f"Bearer {client.app.state.shell_token}"}
    r = client.post(path, json={"model": "gpt-oss", **body}, headers=headers)
    assert r.status_code == 200, r.text
    assert calls_of(r.json()) == [("get_weather", {"location": "San Francisco"})]


class _RawBackend:
    loaded = True

    def __init__(self, raw):
        self._raw = raw

    def chat_stream(self, messages, **kwargs):
        yield from _scrub_stream(_filtered_stream(iter(_chunks(self._raw, 5)), HARMONY_STOPS))


@pytest.mark.parametrize("recipient", ["functions.get_weather", "get_weather"])
def test_the_coder_reads_the_call_from_a_local_engine(recipient):
    from localm.plugins.coder.backends.local_engine import LocalEngineBackend
    from localm.plugins.coder.parser import parse_tool_calls

    engine = Engine.__new__(Engine)
    header = HEADER.replace("functions.get_weather", recipient)
    engine._backend = _RawBackend(ANALYSIS + header + '{"path": "utils.py"}')
    engine.display_name = "gpt-oss"
    backend = LocalEngineBackend.__new__(LocalEngineBackend)
    backend._engine = engine
    backend._loaded = True
    backend._last_reasoning = ""
    for answer in (backend.chat([{"role": "user", "content": "read it"}]),
                   "".join(backend.chat_stream([{"role": "user", "content": "read it"}]))):
        [call] = parse_tool_calls(answer)
        assert (call.name, call.args) == ("get_weather", {"path": "utils.py"})
        assert "Need to use function" in backend.last_reasoning
