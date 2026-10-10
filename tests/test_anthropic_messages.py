# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anthropic ``/v1/messages`` and ``/v1/messages/count_tokens`` on top of the
chat route: the request reaches the model as the equivalent chat request
(system, images, tools, tool results, stop sequences, thinking), the reply and
the event stream come back in the Messages shape, errors in Anthropic's error
shape, and the routes take ``x-api-key`` as well as a bearer token."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference import anthropic_protocol as A
from localm.inference.http_server import create_app
from localm.inference.tool_calling import CLOSE_TAG, OPEN_TAG

MODEL = "msg-model"
KEY = "lm-anthropic-test-key"
WEATHER = {"name": "get_weather", "description": "Weather for a city",
           "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                            "required": ["city"]}}
PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def call(name="get_weather", **arguments):
    return f'{OPEN_TAG}{json.dumps({"name": name, "arguments": arguments or {"city": "Oslo"}})}{CLOSE_TAG}'


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


class Served:
    def __init__(self, tokens=("Hello",), finish="stop", vision=False):
        self.calls: list = []
        engine = MagicMock()

        def chat_stream(messages, **kwargs):
            self.calls.append((messages, kwargs))
            yield from tokens

        engine.chat_stream.side_effect = chat_stream
        engine.unsupported_sampling.return_value = []
        engine.display_name = MODEL
        engine.model_path = ""
        engine.count_tokens.return_value = 2
        engine.count_messages_tokens.return_value = 11
        engine.gpu_placement = None
        engine.last_finish_reason = finish
        engine.context_capacity.return_value = 4096
        engine.supports_images = vision
        engine.loaded = True
        self.engine = engine
        self.client = TestClient(create_app(engine))

    def post(self, path="/v1/messages", headers=None, **body):
        payload = {"model": MODEL, "max_tokens": 64,
                   "messages": [{"role": "user", "content": "hi"}], **body}
        return self.client.post(path, json=payload, headers=headers or {})

    @property
    def messages(self):
        return self.calls[-1][0]

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


# ------------------------------------------------------------------ requests


def test_a_plain_reply(home):
    r = Served(["Hel", "lo"]).post()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["id"].startswith("msg_")
    assert body["content"] == [{"type": "text", "text": "Hello"}]
    assert body["stop_reason"] == "end_turn" and body["stop_sequence"] is None
    assert body["usage"]["input_tokens"] == 11 and body["usage"]["output_tokens"] == 2


def test_system_blocks_and_sampling_reach_the_model(home):
    served = Served()
    served.post(system=[{"type": "text", "text": "Be terse."},
                        {"type": "text", "text": "Use metric."}],
                temperature=0.2, top_p=0.9, top_k=20, max_tokens=33)
    first = served.messages[0]
    assert first["role"] == "system" and first["content"].startswith("Be terse.\n\nUse metric.")
    kw = served.kwargs
    assert (kw["temperature"], kw["top_p"], kw["top_k"], kw["max_tokens"]) == (0.2, 0.9, 20, 33)


def test_thinking_is_off_unless_asked_and_returned_only_when_asked(home):
    served = Served(["<think>hmm</think>", "Yes"])
    r = served.post()
    assert served.kwargs["thinking"] is False
    assert r.json()["content"] == [{"type": "text", "text": "Yes"}]
    r = served.post(thinking={"type": "enabled", "budget_tokens": 2048})
    assert served.kwargs["thinking"] is True
    assert r.json()["content"][0] == {"type": "thinking", "thinking": "hmm", "signature": ""}
    assert r.json()["content"][1] == {"type": "text", "text": "Yes"}


def test_an_image_block_reaches_the_model_as_an_image_part(home):
    served = Served(vision=True)
    r = served.post(messages=[{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]}])
    assert r.status_code == 200, r.text
    parts = served.messages[-1]["content"]
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["image_url"]["url"] == f"data:image/png;base64,{PNG}"


def test_a_tool_call_comes_back_as_tool_use(home):
    served = Served(["Checking. ", call()])
    r = served.post(tools=[WEATHER])
    body = r.json()
    assert body["stop_reason"] == "tool_use"
    text, use = body["content"]
    assert text["type"] == "text" and text["text"].strip() == "Checking."
    assert use["type"] == "tool_use" and use["name"] == "get_weather"
    assert use["input"] == {"city": "Oslo"} and use["id"]


def test_a_tool_round_trip_reaches_the_model_in_order(home):
    served = Served(["It is sunny."])
    r = served.post(tools=[WEATHER], messages=[
        {"role": "user", "content": "weather in Oslo?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "need a tool", "signature": "x"},
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
             "input": {"city": "Oslo"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": [{"type": "text", "text": "sunny, 21C"}]},
            {"type": "text", "text": "thanks"}]}])
    assert r.status_code == 200, r.text
    rendered = json.dumps(served.messages)
    assert "need a tool" not in rendered
    assert "sunny, 21C" in rendered and "get_weather" in rendered
    assert rendered.index("Let me check.") < rendered.index("sunny, 21C") < rendered.index("thanks")


def test_an_error_tool_result_is_marked(home):
    served = Served()
    served.post(tools=[WEATHER], messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                           "name": "get_weather", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": "city not found", "is_error": True}]}])
    assert "Error: city not found" in json.dumps(served.messages)


@pytest.mark.parametrize("choice, grammar_needle", [
    ({"type": "any"}, '"<tool_call>"'),
    ({"type": "tool", "name": "get_weather"}, '"<tool_call>"'),
])
def test_a_forced_tool_choice_constrains_the_reply(home, choice, grammar_needle):
    served = Served([call()])
    served.post(tools=[WEATHER], tool_choice=choice)
    assert grammar_needle in served.kwargs["grammar"]
    assert "grammar_lazy" not in served.kwargs


def test_tool_choice_none_hides_the_tools(home):
    served = Served(["fine"])
    served.post(tools=[WEATHER], tool_choice={"type": "none"})
    assert "grammar" not in served.kwargs
    assert "get_weather" not in json.dumps(served.messages)


def test_disable_parallel_tool_use_allows_one_call(home):
    served = Served([call(), call(city="Rome")])
    body = served.post(tools=[WEATHER], tool_choice={
        "type": "auto", "disable_parallel_tool_use": True}).json()
    assert [b["type"] for b in body["content"]] == ["tool_use"]


def test_a_stop_sequence_is_reported(home):
    served = Served(["one two ", "STOP three"])
    body = served.post(stop_sequences=["STOP"]).json()
    assert body["content"] == [{"type": "text", "text": "one two "}]
    assert body["stop_reason"] == "stop_sequence" and body["stop_sequence"] == "STOP"


def test_a_length_finish_is_max_tokens(home):
    body = Served(["abc"], finish="length").post().json()
    assert body["stop_reason"] == "max_tokens"


# ------------------------------------------------------------------ streaming


def test_the_event_sequence_for_text(home):
    evs = events(Served(["Hel", "lo"]).post(stream=True))
    kinds = [e["type"] for e in evs]
    assert kinds[:2] == ["message_start", "ping"]
    assert kinds[-2:] == ["message_delta", "message_stop"]
    assert kinds.count("content_block_start") == 1 and kinds.count("content_block_stop") == 1
    text = "".join(e["delta"]["text"] for e in evs if e["type"] == "content_block_delta")
    assert text == "Hello"
    delta = evs[-2]
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"]["output_tokens"] == 2 and delta["usage"]["input_tokens"] == 11


def test_streamed_thinking_text_and_tool_use_blocks(home):
    served = Served(["<think>plan</think>", "Sure. ", call()])
    evs = events(served.post(stream=True, tools=[WEATHER], thinking={"type": "enabled",
                                                                       "budget_tokens": 1024}))
    starts = [e["content_block"] for e in evs if e["type"] == "content_block_start"]
    assert [b["type"] for b in starts] == ["thinking", "text", "tool_use"]
    assert [e["index"] for e in evs if e["type"] == "content_block_start"] == [0, 1, 2]
    deltas = [e["delta"] for e in evs if e["type"] == "content_block_delta"]
    assert {"type": "thinking_delta", "thinking": "plan"} in deltas
    assert {"type": "signature_delta", "signature": ""} in deltas
    [args] = [d["partial_json"] for d in deltas if d["type"] == "input_json_delta"]
    assert json.loads(args) == {"city": "Oslo"}
    assert evs[-2]["delta"]["stop_reason"] == "tool_use"


def test_a_failed_generation_ends_with_an_error_event(home):
    served = Served(["partial"])

    def broken(messages, **kwargs):
        yield "partial"
        raise RuntimeError("out of memory")
    served.engine.chat_stream.side_effect = broken
    evs = events(served.post(stream=True))
    assert evs[-1]["type"] == "error"
    assert "out of memory" in evs[-1]["error"]["message"]
    assert "message_stop" not in [e["type"] for e in evs]


def test_anthropic_sdk_reads_the_stream_and_the_reply(home):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    import asyncio
    served = Served(["Checking. ", call()])

    async def drive():
        client = anthropic.AsyncAnthropic(
            base_url="http://testserver", api_key=KEY,
            http_client=httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=served.client.app),
                base_url="http://testserver"))
        tool = {k: v for k, v in WEATHER.items()}
        async with client.messages.stream(
                model=MODEL, max_tokens=64, tools=[tool],
                messages=[{"role": "user", "content": "weather?"}]) as stream:
            final = await stream.get_final_message()
        reply = await client.messages.create(
            model=MODEL, max_tokens=64, tools=[tool],
            messages=[{"role": "user", "content": "weather?"}])
        count = await client.messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": "weather?"}], tools=[tool])
        return final, reply, count

    final, reply, count = asyncio.run(drive())
    for message in (final, reply):
        assert message.stop_reason == "tool_use"
        assert message.content[0].text.strip() == "Checking."
        assert message.content[1].name == "get_weather"
        assert message.content[1].input == {"city": "Oslo"}
    assert final.usage.output_tokens == 2 and final.usage.input_tokens == 11
    assert count.input_tokens == 11


# ------------------------------------------------------------------ errors and auth


@pytest.mark.parametrize("extra, needle", [
    ({"messages": [{"role": "system", "content": "x"}]}, "role"),
    ({"messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]},
     "document"),
    ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}, "web_search"),
    ({"tool_choice": {"type": "sometimes"}}, "tool_choice"),
    ({"thinking": {"type": "maybe"}}, "thinking"),
    ({"mcp_servers": [{"url": "http://x"}]}, "mcp_servers"),
    ({"messages": [{"role": "user", "content": [{"type": "image", "source": {
        "type": "file", "file_id": "f"}}]}]}, "file"),
])
def test_an_unsupported_request_is_an_anthropic_400(home, extra, needle):
    served = Served()
    r = served.post(**extra)
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["type"] == "error" and body["error"]["type"] == "invalid_request_error"
    assert needle in body["error"]["message"]
    assert served.calls == []


def test_a_missing_max_tokens_is_an_anthropic_400(home):
    r = Served().client.post("/v1/messages", json={
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in r.json()["error"]["message"]


def test_an_unknown_model_is_not_found(home):
    (home / "registry.json").write_text(json.dumps({MODEL: {"path": "m.gguf"}}),
                                        encoding="utf-8")
    r = Served().post(model="claude-sonnet-4-5")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found_error"


def test_count_tokens_measures_the_prompt_with_the_tools(home):
    served = Served()
    r = served.post("/v1/messages/count_tokens", tools=[WEATHER])
    assert r.status_code == 200, r.text
    assert r.json() == {"input_tokens": 11}
    counted = served.engine.count_messages_tokens.call_args.args[0]
    assert "get_weather" in json.dumps(counted)


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
def test_auth_takes_x_api_key_or_bearer_and_refuses_without(home, monkeypatch, path):
    monkeypatch.setenv("LOCALM_API_KEY", KEY)
    served = Served()
    r = served.post(path)
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"
    assert served.post(path, headers={"x-api-key": "wrong"}).status_code == 401
    assert served.post(path, headers={"x-api-key": KEY}).status_code == 200
    assert served.post(path, headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_the_routes_are_cross_origin_callable_like_chat(home):
    served = Served()
    r = served.post(headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------ translation


def test_consecutive_tool_results_become_tool_messages_before_the_text():
    out = A.messages_to_openai(None, [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a", "name": "f", "input": {"x": 1}},
            {"type": "tool_use", "id": "b", "name": "f", "input": {"x": 2}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "one"},
            {"type": "tool_result", "tool_use_id": "b", "content": "two"},
            {"type": "text", "text": "go on"}]}])
    assert [m["role"] for m in out] == ["assistant", "tool", "tool", "user"]
    assert [c["id"] for c in out[0]["tool_calls"]] == ["a", "b"]
    assert out[1] == {"role": "tool", "tool_call_id": "a", "content": "one"}
    assert out[3] == {"role": "user", "content": "go on"}


def test_stop_reason_mapping():
    assert A.stop_reason("stop", None) == "end_turn"
    assert A.stop_reason("stop", "X") == "stop_sequence"
    assert A.stop_reason("length", None) == "max_tokens"
    assert A.stop_reason("tool_calls", None) == "tool_use"


# ------------------------------------------------------------------ stop_sequence on the chat reply


def test_the_chat_reply_names_the_stop_sequence_that_ended_it(home):
    served = Served(["one END two"])
    body = {"model": MODEL, "messages": [{"role": "user", "content": "x"}], "stop": ["END"]}
    r = served.client.post("/v1/chat/completions", json=body)
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["stop_sequence"] == "END"
    r = served.client.post("/v1/chat/completions", json={**body, "stream": True})
    finish = [json.loads(line[5:]) for line in r.text.split("\n")
              if line.startswith("data:") and "[DONE]" not in line][-1]
    assert finish["choices"][0]["finish_reason"] == "stop"
    assert finish["choices"][0]["stop_sequence"] == "END"
    r = Served(["no stop here"]).client.post("/v1/chat/completions", json=body)
    assert r.json()["choices"][0]["stop_sequence"] is None


def test_stop_filter_names_the_earliest_match():
    from localm.inference.stop_sequences import StopFilter, apply_stop_matched
    flt = StopFilter(["B", "A"])
    assert flt.feed("xxAyyB") == "xx" and flt.matched == "A"
    assert apply_stop_matched("abcd", ["cd", "c"]) == ("ab", "cd")
    assert apply_stop_matched("abcd", ["z"]) == ("abcd", None)


def test_x_api_key_works_on_the_openai_routes_too(home, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", KEY)
    served = Served()
    body = {"model": MODEL, "messages": [{"role": "user", "content": "x"}]}
    assert served.client.post("/v1/chat/completions", json=body).status_code == 401
    assert served.client.post("/v1/chat/completions", json=body,
                              headers={"x-api-key": KEY}).status_code == 200


def test_output_config_format_constrains_the_reply_to_the_schema(home):
    schema = {"type": "object", "properties": {"city": {"type": "string"}},
              "required": ["city"], "additionalProperties": False}
    served = Served(['{"city": "Oslo"}'])
    r = served.post(output_config={"format": {"type": "json_schema", "schema": schema},
                                   "effort": "low"})
    assert r.status_code == 200, r.text
    assert json.loads(r.json()["content"][0]["text"]) == {"city": "Oslo"}
    assert '"\\"city\\""' in served.kwargs["grammar"]
    assert served.kwargs["thinking"] is False


@pytest.mark.parametrize("config", [
    {"format": {"type": "json_object"}}, {"format": {"type": "json_schema"}}, "json",
    {"format": {"type": "text", "schema": {"type": "object"}}}])
def test_a_malformed_output_config_is_a_400(home, config):
    r = Served().post(output_config=config)
    assert r.status_code == 400 and "output_config" in r.json()["error"]["message"]


def test_between_tools_thinking_turns_thinking_on(home):
    served = Served()
    served.post(thinking={"type": "between_tools"})
    assert served.kwargs["thinking"] is True


# ------------------------------------------------------------------ review fixes


def test_a_tool_call_ended_by_a_stop_sequence_reports_tool_use_only(home):
    served = Served([call(), " then STOP more"])
    body = served.post(tools=[WEATHER], stop_sequences=["STOP"]).json()
    assert body["stop_reason"] == "tool_use" and body["stop_sequence"] is None
    evs = events(served.post(tools=[WEATHER], stop_sequences=["STOP"], stream=True))
    assert evs[-2]["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}
    chat = served.client.post("/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "parameters": WEATHER["input_schema"]}}],
        "stop": ["STOP"]}).json()
    assert chat["choices"][0]["finish_reason"] == "tool_calls"
    assert chat["choices"][0]["stop_sequence"] is None


def test_only_the_finish_chunk_carries_stop_sequence(home):
    served = Served(["one ", "END"])
    r = served.client.post("/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": "x"}],
        "stop": ["END"], "stream": True})
    chunks = [json.loads(line[5:]) for line in r.text.split("\n")
              if line.startswith("data:") and "[DONE]" not in line]
    assert all("stop_sequence" not in c["choices"][0] for c in chunks[:-1])
    assert chunks[-1]["choices"][0]["stop_sequence"] == "END"


def test_text_blocks_stay_separate_parts(home):
    served = Served()
    served.post(messages=[{"role": "user", "content": [
        {"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]}])
    assert served.messages[-1]["content"] == [{"type": "text", "text": "Hello"},
                                              {"type": "text", "text": "World"}]


def test_consecutive_same_role_turns_are_joined():
    out = A.messages_to_openai(None, [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "t",
                                           "signature": ""}]},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "f",
                                           "input": {}}]}])
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[0]["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert out[1]["content"] == "c" and out[1]["tool_calls"][0]["id"] == "x"


def test_a_streamed_overflow_refusal_keeps_its_reason(home, monkeypatch):
    import localm.inference.compact as compact
    served = Served()
    served.engine.count_messages_tokens.side_effect = lambda ms: 9000 if len(ms) > 3 else 5000
    monkeypatch.setattr(compact, "compact_messages", lambda ms, gen: (ms[-2:], True))
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
             for i in range(7)]
    evs = events(served.post(stream=True, messages=turns))
    assert evs[-1]["type"] == "error"
    assert "exceeds" in evs[-1]["error"]["message"]
    assert not [e for e in evs if e["type"] == "content_block_delta"]
    r = served.client.post("/v1/chat/completions", json={
        "model": MODEL, "messages": turns, "stream": True})
    refusal = [json.loads(line[5:]) for line in r.text.split("\n")
               if line.startswith("data:") and "localm_error" in line]
    assert refusal and refusal[0]["localm_error"]["status"] == 413


def test_count_tokens_is_answered_by_the_peer_that_serves_the_model(home, monkeypatch):
    from fastapi.responses import JSONResponse

    from localm import peer_routing
    seen = {}

    async def fake_forward(route, request, path, *, body=None, headers=None):
        seen.update(path=path, body=json.loads(body))
        return JSONResponse({"input_tokens": 99})
    monkeypatch.setattr(peer_routing, "get_route",
                        lambda name: object() if name == MODEL else None)
    monkeypatch.setattr(peer_routing, "forward", fake_forward)
    monkeypatch.setattr(peer_routing, "forward_body", lambda route, raw: raw)
    served = Served()
    r = served.post("/v1/messages/count_tokens")
    assert r.json() == {"input_tokens": 99}
    assert seen["path"] == "/v1/messages/count_tokens"
    served.engine.count_messages_tokens.assert_not_called()


def test_an_unexpected_error_keeps_the_anthropic_shape(home, monkeypatch):
    def broken(req, model):
        raise RuntimeError("boom")
    monkeypatch.setattr(A, "plan_messages", broken)
    r = TestClient(create_app(Served().engine), raise_server_exceptions=False).post(
        "/v1/messages", json={"model": MODEL, "max_tokens": 5,
                              "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 500
    assert r.json() == {"type": "error", "error": {"type": "api_error",
                                                   "message": "Internal server error"}}


def test_the_translator_drops_a_stop_sequence_on_a_tool_use_reply():
    import asyncio
    data = {"choices": [{"finish_reason": "tool_calls", "stop_sequence": "STOP",
                         "message": {"content": "", "tool_calls": [
                             {"id": "c", "function": {"name": "f", "arguments": "{}"}}]}}]}
    assert A.message_from_completion(data, MODEL, False)["stop_sequence"] is None

    async def chunks():
        yield {"choices": [{"delta": {"tool_calls": data["choices"][0]["message"]["tool_calls"]}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "tool_calls", "stop_sequence": "STOP"}]}

    async def collect():
        return [line async for line in A.message_stream(chunks(), model=MODEL, want_thinking=False)]
    lines = asyncio.run(collect())
    delta = [json.loads(x.decode().split("data: ", 1)[1]) for x in lines
             if b"message_delta" in x][0]["delta"]
    assert delta == {"stop_reason": "tool_use", "stop_sequence": None}


_HOSTILE_DOCS = pytest.mark.parametrize(
    "doc", ["[" * 100_000, "9" * 5_000], ids=["deep", "bigint"])


@_HOSTILE_DOCS
def test_a_hostile_tool_call_argument_string_becomes_an_empty_input(doc):
    block = A._tool_use_block({"id": "c", "function": {"name": "f", "arguments": doc}})
    assert block["input"] == {} and block["name"] == "f"


@_HOSTILE_DOCS
def test_a_hostile_chat_reply_is_a_502_not_a_crash(home, monkeypatch, doc):
    from fastapi.responses import Response

    from localm.inference.routes import anthropic as routes

    async def inner(*args, **kwargs):
        return Response(content=doc.encode(), media_type="application/json")

    monkeypatch.setattr(routes, "chat_endpoint", lambda app, path: inner)
    r = Served().post()
    assert r.status_code == 502
    assert r.json()["error"]["message"] == "the chat route returned a reply that is not JSON"


@_HOSTILE_DOCS
def test_a_hostile_chat_error_body_is_passed_on_as_text(home, monkeypatch, doc):
    from fastapi.responses import Response

    from localm.inference.routes import anthropic as routes

    async def inner(*args, **kwargs):
        return Response(content=doc.encode(), status_code=500, media_type="application/json")

    monkeypatch.setattr(routes, "chat_endpoint", lambda app, path: inner)
    r = Served().post()
    assert r.status_code == 500
    assert r.json()["error"]["message"] == doc[:500]
