# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI request fields on /v1/chat/completions and /v1/completions:
``response_format`` reaches the backend as a grammar, the sampling options
reach ``chat_stream`` or are refused, ``stream_options.include_usage`` moves
the usage to a last chunk, and every field localm does not serve is a 400
naming it."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.backends.base import GrammarUnsupportedError
from localm.inference.http_server import create_app
from localm.inference.response_format import (
    ResponseFormat, ResponseFormatError, after_think, combine, format_grammar,
    parse_response_format, prefix_rules,
)
from localm.inference.tool_calling import CLOSE_TAG, OPEN_TAG

MODEL = "conf-model"
WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
PERSON = {"type": "object", "properties": {"name": {"type": "string"},
                                           "age": {"type": "integer"}},
          "required": ["name", "age"]}


def json_schema(schema=PERSON, strict=True, name="person"):
    return {"type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": strict}}


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
    def __init__(self, tokens=("Hi",), refused=()):
        self.calls: list = []
        engine = MagicMock()

        def chat_stream(messages, **kwargs):
            self.calls.append((messages, kwargs))
            yield from tokens

        engine.chat_stream.side_effect = chat_stream
        engine.unsupported_sampling.side_effect = (
            lambda names: [n for n in names if n in refused])
        engine.display_name = MODEL
        engine.model_path = ""
        engine.count_tokens.return_value = 2
        engine.count_messages_tokens.return_value = 3
        engine.gpu_placement = None
        engine.last_finish_reason = "stop"
        engine.context_capacity.return_value = 4096
        engine.loaded = True
        self.engine = engine
        self.client = TestClient(create_app(engine))

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


# ------------------------------------------------------------------ response_format


def test_json_object_constrains_the_reply_to_an_object(home):
    served = Served(['{"a": 1}'])
    r = served.chat(response_format={"type": "json_object"})
    assert r.status_code == 200, r.text
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"a": 1}
    grammar = served.kwargs["grammar"]
    assert grammar.startswith("root ::= ") and '"{"' in grammar
    assert "grammar_lazy" not in served.kwargs


def test_text_format_is_no_grammar(home):
    served = Served()
    assert served.chat(response_format={"type": "text"}).status_code == 200
    assert "grammar" not in served.kwargs
    assert "thinking" not in served.kwargs


def test_json_schema_reaches_the_backend_as_the_compiled_schema(home):
    served = Served(['{"name": "Ada", "age": 36}'])
    assert served.chat(response_format=json_schema()).status_code == 200
    grammar = served.kwargs["grammar"]
    assert '"\\"name\\""' in grammar and '"\\"age\\""' in grammar


def test_a_format_turns_thinking_off_unless_the_client_asked_for_it(home):
    served = Served(['{}'])
    served.chat(response_format={"type": "json_object"})
    assert served.kwargs["thinking"] is False
    served.chat(response_format={"type": "json_object"},
                chat_template_kwargs={"enable_thinking": True})
    assert served.kwargs["thinking"] is True
    assert served.kwargs["grammar"].startswith('root ::= ("<think>" think-body "</think>"')


def test_reasoning_effort_none_turns_thinking_off(home):
    served = Served()
    served.chat(reasoning_effort="none")
    assert served.kwargs["thinking"] is False
    served.chat(reasoning_effort="high")
    assert "thinking" not in served.kwargs


def test_strict_schema_with_a_keyword_the_grammar_cannot_enforce_is_a_400(home):
    schema = {"type": "object", "properties": {"code": {"type": "string", "pattern": "^A"}}}
    served = Served()
    r = served.chat(response_format=json_schema(schema, strict=True))
    assert r.status_code == 400
    assert "pattern" in r.json()["detail"]
    assert served.calls == []


def test_non_strict_schema_drops_only_the_keyword_it_cannot_enforce(home):
    schema = {"type": "object", "properties": {"code": {"type": "string", "pattern": "^A"}},
              "required": ["code"]}
    served = Served(['{"code": "x"}'])
    assert served.chat(response_format=json_schema(schema, strict=False)).status_code == 200
    assert '"\\"code\\""' in served.kwargs["grammar"]


@pytest.mark.parametrize("value, needle", [
    ("json", "must be an object"),
    ({"type": "xml"}, "'xml'"),
    ({"type": "json_schema"}, "json_schema must be an object"),
    ({"type": "json_schema", "json_schema": {"schema": PERSON}}, "name"),
    ({"type": "json_schema", "json_schema": {"name": "a b", "schema": PERSON}}, "name"),
    ({"type": "json_schema", "json_schema": {"name": "x", "schema": []}}, "schema must be"),
    ({"type": "json_schema", "json_schema": {"name": "x", "schema": PERSON,
                                             "strict": "yes"}}, "strict"),
])
def test_a_malformed_response_format_is_a_400(home, value, needle):
    served = Served()
    r = served.chat(response_format=value)
    assert r.status_code == 400, r.text
    assert needle in r.json()["detail"]
    assert served.calls == []


def test_response_format_with_a_grammar_is_a_400(home):
    r = Served().chat(response_format={"type": "json_object"}, grammar='root ::= "x"')
    assert r.status_code == 400
    assert "response_format" in r.json()["detail"]


def test_response_format_with_auto_tools_allows_a_call_or_the_json(home):
    call = f'{OPEN_TAG}{json.dumps({"name": "get_weather", "arguments": {"city": "Oslo"}})}{CLOSE_TAG}'
    served = Served([call])
    r = served.chat(tools=[WEATHER], response_format=json_schema())
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"
    grammar = served.kwargs["grammar"]
    assert grammar.startswith("root ::= alt0-root | alt1-root")
    assert '"<tool_call>"' in grammar and '"\\"age\\""' in grammar
    assert "grammar_lazy" not in served.kwargs

    served = Served(['{"name": "Ada", "age": 36}'])
    r = served.chat(tools=[WEATHER], response_format=json_schema())
    message = r.json()["choices"][0]["message"]
    assert json.loads(message["content"]) == {"name": "Ada", "age": 36}
    assert message["tool_calls"] is None


def test_a_required_tool_call_ignores_the_format(home):
    served = Served()
    served.chat(tools=[WEATHER], tool_choice="required", response_format=json_schema())
    grammar = served.kwargs["grammar"]
    assert '"<tool_call>"' in grammar and "age" not in grammar
    assert "thinking" not in served.kwargs


@pytest.mark.parametrize("extra", [{}, {"tools": [WEATHER]}])
def test_a_backend_without_grammar_refuses_response_format(home, extra):
    served = Served()
    served.engine.validate_grammar.side_effect = GrammarUnsupportedError("no grammar here")
    r = served.chat(response_format={"type": "json_object"}, **extra)
    assert r.status_code == 400
    assert "response_format" in r.json()["detail"]
    assert served.calls == []


def test_streamed_response_format_reaches_the_backend(home):
    served = Served(['{"a"', ': 1}'])
    events = sse(served.chat(stream=True, response_format={"type": "json_object"}))
    text = "".join(e["choices"][0]["delta"].get("content") or "" for e in events
                   if e.get("choices"))
    assert json.loads(text) == {"a": 1}
    assert served.kwargs["grammar"].startswith("root ::= ")


# ------------------------------------------------------------------ sampling


def test_sampling_options_reach_the_backend(home):
    served = Served()
    r = served.chat(presence_penalty=0.5, frequency_penalty=-0.25, min_p=0.1)
    assert r.status_code == 200, r.text
    kw = served.kwargs
    assert (kw["presence_penalty"], kw["frequency_penalty"], kw["min_p"]) == (0.5, -0.25, 0.1)


def test_unset_sampling_options_are_not_sent(home):
    served = Served()
    served.chat()
    assert not {"presence_penalty", "frequency_penalty", "min_p"} & set(served.kwargs)
    served.engine.unsupported_sampling.assert_not_called()


def test_a_backend_that_cannot_apply_an_option_refuses_it_by_name(home):
    served = Served(refused=("presence_penalty",))
    r = served.chat(presence_penalty=1.0, min_p=0.2)
    assert r.status_code == 400
    assert "presence_penalty" in r.json()["detail"] and "min_p" not in r.json()["detail"]
    assert served.calls == []
    r = served.complete(presence_penalty=1.0)
    assert r.status_code == 400 and "presence_penalty" in r.json()["detail"]


@pytest.mark.parametrize("field, value", [
    ("presence_penalty", 2.5), ("frequency_penalty", -3), ("min_p", 1.5), ("min_p", -0.1)])
def test_an_out_of_range_option_is_rejected(home, field, value):
    assert Served().chat(**{field: value}).status_code == 422


def test_max_completion_tokens_caps_the_reply(home):
    served = Served()
    served.chat(max_completion_tokens=7)
    assert served.kwargs["max_tokens"] == 7
    served.chat(max_completion_tokens=7, max_tokens=7)
    assert served.kwargs["max_tokens"] == 7
    r = served.chat(max_completion_tokens=7, max_tokens=9)
    assert r.status_code == 400 and "max_completion_tokens" in r.json()["detail"]


# ------------------------------------------------------------------ refused fields


@pytest.mark.parametrize("extra, field", [
    ({"n": 2}, "n=2"),
    ({"n": 0}, "n=0"),
    ({"logprobs": True}, "logprobs"),
    ({"top_logprobs": 3}, "top_logprobs"),
    ({"functions": [{"name": "f"}]}, "functions"),
    ({"function_call": "auto"}, "function_call"),
    ({"audio": {"voice": "alloy", "format": "wav"}}, "audio"),
    ({"modalities": ["text", "audio"]}, "modalities"),
    ({"web_search_options": {}}, "web_search_options"),
    ({"logit_bias": {"50256": -100}}, "logit_bias"),
    ({"stream_options": "yes"}, "stream_options"),
    ({"stream_options": {"include_usage": "yes"}}, "include_usage"),
])
def test_a_field_localm_does_not_serve_is_a_400_naming_it(home, extra, field):
    served = Served()
    r = served.chat(**extra)
    assert r.status_code == 400, r.text
    assert field in r.json()["detail"]
    assert served.calls == []


@pytest.mark.parametrize("extra", [
    {"n": 1}, {"logprobs": False}, {"top_logprobs": 0}, {"modalities": ["text"]},
    {"logit_bias": {}}, {"user": "u"}, {"metadata": {"k": "v"}}, {"store": False},
    {"service_tier": "auto"}, {"verbosity": "low"}, {"prediction": {"type": "content",
                                                                   "content": "x"}},
    {"stream_options": {"include_usage": False}},
])
def test_fields_that_change_nothing_are_accepted(home, extra):
    assert Served().chat(**extra).status_code == 200


def test_the_developer_role_is_read_as_system(home):
    served = Served()
    r = served.client.post("/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "developer", "content": "be terse"},
                                     {"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    first = served.calls[-1][0][0]
    assert first["role"] == "system" and first["content"].startswith("be terse")


# ------------------------------------------------------------------ include_usage


def _usage_chunks(events):
    return [e for e in events if e.get("usage")]


def test_include_usage_moves_the_usage_to_a_last_chunk_with_no_choices(home):
    events = sse(Served(["Hel", "lo"]).chat(
        stream=True, stream_options={"include_usage": True}))
    finish = [e for e in events if e.get("choices") and e["choices"][0]["finish_reason"]]
    assert finish[-1]["usage"] is None
    [usage] = _usage_chunks(events)
    assert usage is events[-1]
    assert usage["choices"] == []
    assert usage["usage"]["prompt_tokens"] == 3 and usage["usage"]["completion_tokens"] == 2


def test_without_include_usage_the_finish_chunk_carries_it(home):
    events = sse(Served(["Hel", "lo"]).chat(stream=True))
    [usage] = _usage_chunks(events)
    assert usage["choices"][0]["finish_reason"] == "stop"
    assert all(e.get("choices") for e in events)


def test_openai_sdk_reads_the_usage_chunk_and_the_schema_reply(home):
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    import asyncio
    served = Served(['{"name": "Ada", "age": 36}'])

    async def drive():
        client = openai.AsyncOpenAI(
            base_url="http://testserver/v1", api_key="x",
            http_client=httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=served.client.app),
                base_url="http://testserver"))
        stream = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}], stream=True,
            stream_options={"include_usage": True})
        chunks = [c async for c in stream]
        reply = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "who?"}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "person", "schema": PERSON, "strict": True}},
            presence_penalty=0.5)
        return chunks, reply

    chunks, reply = asyncio.run(drive())
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 2
    assert all(c.usage is None for c in chunks[:-1])
    assert json.loads(reply.choices[0].message.content) == {"name": "Ada", "age": 36}
    assert served.kwargs["presence_penalty"] == 0.5


# ------------------------------------------------------------------ /v1/completions


def test_completion_sampling_echo_and_usage(home):
    served = Served([" upon", " a time"])
    r = served.complete(echo=True, frequency_penalty=0.3, min_p=0.05)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["text"] == "Once upon a time"
    assert served.kwargs["frequency_penalty"] == 0.3 and served.kwargs["min_p"] == 0.05

    events = sse(served.complete(stream=True, echo=True,
                                 stream_options={"include_usage": True}))
    texts = [e["choices"][0]["text"] for e in events if e.get("choices")]
    assert "".join(texts) == "Once upon a time" and texts[0] == "Once"
    assert events[-1]["choices"] == [] and events[-1]["usage"]["completion_tokens"] == 2
    assert all("usage" not in e for e in events[:-1])


def test_a_one_prompt_list_is_a_prompt(home):
    served = Served()
    assert served.complete(prompt=["Hello"]).status_code == 200
    assert served.calls[-1][0][-1] == {"role": "user", "content": "Hello"}


@pytest.mark.parametrize("extra, field", [
    ({"prompt": ["a", "b"]}, "prompt"),
    ({"prompt": [[1, 2, 3]]}, "prompt"),
    ({"prompt": [1, 2]}, "prompt"),
    ({"best_of": 3}, "best_of"),
    ({"suffix": "end"}, "suffix"),
    ({"logprobs": 0}, "logprobs"),
    ({"n": 2}, "n=2"),
    ({"logit_bias": {"1": 1}}, "logit_bias"),
])
def test_a_completion_field_localm_does_not_serve_is_a_400(home, extra, field):
    served = Served()
    r = served.complete(**extra)
    assert r.status_code == 400, r.text
    assert field in r.json()["detail"]
    assert served.calls == []


# ------------------------------------------------------------------ grammar text


def test_prefix_rules_renames_rules_but_not_literals_classes_or_counts():
    grammar = ('root ::= item ("," item){0,3}\n'
               'item ::= "item" [a-z]+ "\\"root\\"" # root\n')
    out = prefix_rules(grammar, "p-")
    assert out == ('p-root ::= p-item ("," p-item){0,3}\n'
                   'p-item ::= "item" [a-z]+ "\\"root\\"" # root\n')


def test_combine_and_after_think_produce_grammars_that_pass_the_structure_check():
    from localm.inference.gbnf import check_grammar_structure
    one, _ = format_grammar(ResponseFormat("json_schema", "p", PERSON, True))
    two, _ = format_grammar(ResponseFormat("json_object"))
    both = combine(one, two)
    assert both.startswith("root ::= alt0-root | alt1-root\n")
    assert len([ln for ln in both.splitlines() if ln.startswith("root ::=")]) == 1
    check_grammar_structure(both)
    think = after_think(both)
    check_grammar_structure(think)
    assert len([ln for ln in think.splitlines() if ln.startswith("root ::=")]) == 1


def test_json_schema_without_a_schema_is_any_object():
    fmt = parse_response_format({"type": "json_schema", "json_schema": {"name": "x"}})
    assert fmt == ResponseFormat("json_object", name="x", strict=False)


def test_an_unknown_keyword_in_a_strict_schema_names_it():
    with pytest.raises(ResponseFormatError, match="patternProperties"):
        format_grammar(ResponseFormat("json_schema", "x", {
            "type": "object", "patternProperties": {"^a": {"type": "string"}}}, True))
