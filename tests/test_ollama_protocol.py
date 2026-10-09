# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Ollama wire translation (localm.inference.ollama_protocol), no server."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from localm.inference import ollama_protocol as P
from localm.inference.gbnf import check_grammar_structure


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ paths

def test_origin_exempt_sets_are_exact_paths_never_prefixes():
    assert "/api/embed" in P.CROSS_ORIGIN_OK_PATHS
    assert "/api/embedding/warmup" not in P.CROSS_ORIGIN_OK_PATHS
    for path in P.CROSS_ORIGIN_OK_PATHS | P.OPEN_MODE_GET_PATHS:
        assert not path.endswith("/")
    for mutating in ("/api/pull", "/api/push", "/api/create", "/api/delete",
                     P.COPY_PATH):
        assert mutating not in P.CROSS_ORIGIN_OK_PATHS
        assert mutating not in P.OPEN_MODE_GET_PATHS


# ------------------------------------------------------------------ requests

def test_stream_defaults_to_true_and_false_is_honoured():
    on = P.plan_chat(P.OllamaChatRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]), "m")
    off = P.plan_chat(P.OllamaChatRequest(
        model="m", stream=False, messages=[{"role": "user", "content": "hi"}]), "m")
    assert on.stream is True and on.body["stream"] is True
    assert off.stream is False and off.body["stream"] is False


def test_options_map_to_chat_fields():
    fields, stop, ignored = P.options_to_fields({
        "temperature": 0.2, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1,
        "seed": 7, "num_predict": 64, "stop": ["</s>", ""],
        "num_ctx": 8192, "mirostat": 2, "presence_penalty": None})
    assert fields == {"temperature": 0.2, "top_p": 0.9, "top_k": 40,
                      "repeat_penalty": 1.1, "seed": 7, "max_tokens": 64}
    assert stop == ["</s>"]
    assert ignored == ["mirostat", "num_ctx"]


@pytest.mark.parametrize("num_predict", [-1, -2, 0, None])
def test_num_predict_without_a_cap_sets_no_max_tokens(num_predict):
    fields, _stop, _ignored = P.options_to_fields({"num_predict": num_predict})
    assert "max_tokens" not in fields


def test_stop_may_be_one_string_and_must_be_strings():
    assert P.options_to_fields({"stop": "END"})[1] == ["END"]
    with pytest.raises(P.OllamaError) as exc:
        P.options_to_fields({"stop": [1, 2]})
    assert exc.value.status == 400


def test_format_json_yields_a_grammar_localm_accepts():
    grammar = P.format_to_grammar("json")
    assert grammar and grammar.startswith("root")
    check_grammar_structure(grammar)


def test_format_none_and_empty_mean_no_grammar():
    assert P.format_to_grammar(None) is None
    assert P.format_to_grammar("") is None


def test_format_schema_and_unknown_strings_are_refused():
    with pytest.raises(P.OllamaError) as schema:
        P.format_to_grammar({"type": "object"})
    assert schema.value.status == 400 and "schema" in schema.value.message
    with pytest.raises(P.OllamaError) as other:
        P.format_to_grammar("xml")
    assert other.value.status == 400


@pytest.mark.parametrize("think,expected", [
    (None, None), (True, True), (False, False), ("high", True), ("low", True),
    ("false", False), ("", False)])
def test_thinking_requested(think, expected):
    assert P.thinking_requested(think) is expected


def test_think_flows_into_chat_template_kwargs():
    plan = P.plan_chat(P.OllamaChatRequest(
        model="m", think=False,
        messages=[{"role": "user", "content": "hi"}]), "m")
    assert plan.body["chat_template_kwargs"] == {"enable_thinking": False}
    assert plan.want_thinking is False
    plan = P.plan_chat(P.OllamaChatRequest(
        model="m", think=True,
        messages=[{"role": "user", "content": "hi"}]), "m")
    assert plan.body["chat_template_kwargs"] == {"enable_thinking": True}
    assert plan.want_thinking is True


def test_tools_and_tool_calls_are_refused_not_dropped():
    with pytest.raises(P.OllamaError) as tools:
        P.plan_chat(P.OllamaChatRequest(
            model="m", tools=[{"type": "function"}],
            messages=[{"role": "user", "content": "hi"}]), "m")
    assert tools.value.status == 400
    with pytest.raises(P.OllamaError) as calls:
        P.plan_chat(P.OllamaChatRequest(model="m", messages=[
            {"role": "assistant", "content": "", "tool_calls": [{"function": {}}]}]), "m")
    assert calls.value.status == 400


def test_unknown_role_is_refused():
    with pytest.raises(P.OllamaError):
        P.message_to_openai(P.OllamaMessage(role="narrator", content="x"))


def test_images_become_data_url_parts_with_a_sniffed_mime():
    import base64
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 20).decode()
    jpeg = base64.b64encode(b"\xff\xd8\xff\xe0" + b"0" * 20).decode()
    msg = P.message_to_openai(P.OllamaMessage(
        role="user", content="what is this", images=[png, jpeg]))
    assert msg["content"][0] == {"type": "text", "text": "what is this"}
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert msg["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_image_data_url_passes_through_and_garbage_is_refused():
    url = "data:image/webp;base64,AAAA"
    assert P.image_data_url(url) == url
    with pytest.raises(P.OllamaError):
        P.image_data_url("!!!not base64!!!")


def test_generate_builds_system_then_user_message():
    plan = P.plan_generate(P.OllamaGenerateRequest(
        model="m", prompt="Why?", system="Be brief."), "m")
    assert plan.body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Why?"}]


@pytest.mark.parametrize("field,value", [
    ("raw", True), ("suffix", "tail"), ("template", "{{ .Prompt }}"),
    ("context", [1, 2, 3])])
def test_generate_refuses_what_it_cannot_honour(field, value):
    req = P.OllamaGenerateRequest(model="m", prompt="p", **{field: value})
    with pytest.raises(P.OllamaError) as exc:
        P.plan_generate(req, "m")
    assert exc.value.status == 400


def test_generate_accepts_the_empty_forms_of_those_fields():
    P.plan_generate(P.OllamaGenerateRequest(
        model="m", prompt="p", raw=False, suffix="", template="", context=[]), "m")


def test_unknown_request_keys_are_reported():
    req = P.OllamaChatRequest(model="m", messages=[], mystery=1, other=2)
    assert P.unknown_fields(req) == ["mystery", "other"]


@pytest.mark.parametrize("value,expected", [
    (0, True), (0.0, True), ("0", True), ("0s", True), ("0m", True), (" 0 ", True),
    (-1, False), (300, False), ("5m", False), ("10s", False), (None, False),
    (False, False), ("", False)])
def test_keep_alive_is_zero(value, expected):
    assert P.keep_alive_is_zero(value) is expected


def test_request_model_name():
    assert P.request_model_name(P.OllamaShowRequest(model=" m ")) == "m"
    assert P.request_model_name(P.OllamaShowRequest(name="legacy")) == "legacy"
    with pytest.raises(P.OllamaError) as exc:
        P.request_model_name(P.OllamaShowRequest())
    assert exc.value.status == 400


def test_resolve_model_name():
    assert P.resolve_model_name("qwen", ["qwen"]) == "qwen"
    assert P.resolve_model_name("qwen:latest", ["qwen"]) == "qwen"
    assert P.resolve_model_name("qwen:latest", ["qwen:latest"]) == "qwen:latest"
    assert P.resolve_model_name("qwen:7b", ["qwen"]) == "qwen:7b"
    assert P.resolve_model_name("nope", ["qwen"]) == "nope"


# ------------------------------------------------------------------ stop filter

def test_stop_filter_cuts_at_the_first_stop_and_emits_nothing_after():
    flt = P.StopFilter(["END"])
    assert flt.feed("hello ") == "hello "
    assert flt.feed("worldEND and more") == "world"
    assert flt.hit is True
    assert flt.feed("still more") == ""
    assert flt.flush() == ""


def test_stop_filter_finds_a_stop_split_across_chunks():
    flt = P.StopFilter(["</s>"])
    pieces = [flt.feed(p) for p in ("ab<", "/", "s", ">cd")]
    assert "".join(pieces) == "ab"
    assert flt.hit is True


def test_stop_filter_releases_a_held_tail_that_never_completed():
    flt = P.StopFilter(["</s>"])
    assert flt.feed("ab<") == "ab"
    assert flt.hit is False
    assert flt.flush() == "<"


def test_stop_filter_picks_the_earliest_of_several_stops():
    flt = P.StopFilter(["two", "one"])
    assert flt.feed("zero one two") == "zero "
    assert flt.hit is True


def test_stop_filter_without_stops_is_a_passthrough():
    flt = P.StopFilter([])
    assert flt.feed("anything") == "anything"
    assert flt.flush() == ""
    assert flt.hit is False


def test_apply_stop():
    assert P.apply_stop("Hello world", ["wor"]) == ("Hello ", True)
    assert P.apply_stop("Hello world", ["xyz"]) == ("Hello world", False)
    assert P.apply_stop("ends with <", ["</s>"]) == ("ends with <", False)


# ------------------------------------------------------------------ responses

def test_usage_stats_reports_only_what_was_measured():
    stats = P.usage_stats({"prompt_tokens": 12, "completion_tokens": 30,
                           "ttft_ms": 250.0, "tokens_per_sec": 60.0}, 2_000_000_000)
    assert stats == {"total_duration": 2_000_000_000, "prompt_eval_count": 12,
                     "eval_count": 30, "prompt_eval_duration": 250_000_000,
                     "eval_duration": 500_000_000}
    assert P.usage_stats(None, 5) == {"total_duration": 5}
    assert "eval_duration" not in P.usage_stats(
        {"prompt_tokens": 1, "completion_tokens": 2, "tokens_per_sec": 0}, 1)


def test_reply_object_shapes():
    chat = P.reply_object("chat", "m", content="hi")
    assert chat["message"] == {"role": "assistant", "content": "hi"}
    assert chat["done"] is False and "done_reason" not in chat
    gen = P.reply_object("generate", "m", content="hi", thinking="hmm")
    assert gen["response"] == "hi" and gen["thinking"] == "hmm"
    final = P.reply_object("chat", "m", done=True, reason="length",
                           stats={"eval_count": 3})
    assert final["done"] is True and final["done_reason"] == "length"
    assert final["eval_count"] == 3 and final["message"]["content"] == ""
    assert final["created_at"].endswith("Z")


def test_encode_line_is_one_compact_newline_terminated_json_line():
    line = P.encode_line({"a": "é", "b": [1, 2]})
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert json.loads(line) == {"a": "é", "b": [1, 2]}
    assert b" " not in line


def test_encode_line_survives_splitlines_on_unicode_line_breaks():
    text = "a\u2028b\u2029c\x85d\x0be\x1cf"
    line = P.encode_line({"m": text})
    assert len(line.decode("utf-8").splitlines()) == 1
    assert json.loads(line) == {"m": text}


def _completion(content="Hello world", reasoning=None, finish="stop", usage=None):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 3, "completion_tokens": 2}}


def test_completion_to_reply_plain():
    out = P.completion_to_reply("chat", _completion(), "m", want_thinking=False,
                                stop=[], total_ns=10)
    assert out["message"]["content"] == "Hello world"
    assert out["done"] is True and out["done_reason"] == "stop"
    assert out["prompt_eval_count"] == 3 and out["eval_count"] == 2


def test_completion_to_reply_length_and_stop():
    out = P.completion_to_reply("generate", _completion(finish="length"), "m",
                                want_thinking=False, stop=[], total_ns=1)
    assert out["done_reason"] == "length" and out["response"] == "Hello world"
    out = P.completion_to_reply("generate", _completion(), "m",
                                want_thinking=False, stop=["wor"], total_ns=1)
    assert out["response"] == "Hello " and out["done_reason"] == "stop"


def test_completion_to_reply_thinking_only_when_asked():
    data = _completion(reasoning="because")
    off = P.completion_to_reply("chat", data, "m", want_thinking=False, stop=[], total_ns=1)
    on = P.completion_to_reply("chat", data, "m", want_thinking=True, stop=[], total_ns=1)
    assert "thinking" not in off["message"]
    assert on["message"]["thinking"] == "because"


def test_completion_to_reply_error_finish_is_a_500():
    data = _completion(content="\n[inference error: out of memory]", finish="error")
    with pytest.raises(P.OllamaError) as exc:
        P.completion_to_reply("chat", data, "m", want_thinking=False, stop=[], total_ns=1)
    assert exc.value.status == 500
    assert exc.value.message == "[inference error: out of memory]"


# ------------------------------------------------------------------ streaming

class _Upstream:
    """An async iterator of SSE strings that records whether it was closed and
    how many items it handed out."""

    def __init__(self, items):
        self._items = list(items)
        self.closed = False
        self.handed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        self.handed += 1
        return self._items.pop(0)

    async def aclose(self):
        self.closed = True


def _sse(delta=None, finish=None, usage=None, extra=None):
    chunk = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
             "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        chunk["usage"] = usage
    if extra:
        chunk.update(extra)
    return "data: " + json.dumps(chunk) + "\n\n"


def _translate(items, *, kind="chat", want_thinking=False, stop=()):
    upstream = _Upstream(items)

    async def go():
        out = []
        async for line in P.ndjson_stream(
                P.iter_sse_json(upstream), kind=kind, model="m",
                want_thinking=want_thinking, stop=list(stop),
                started=time.perf_counter()):
            out.append(line)
        return out

    lines = _run(go())
    return [json.loads(line) for line in lines], lines, upstream


def test_stream_translation_chat():
    objs, raw, upstream = _translate([
        _sse({"role": "assistant"}),
        _sse({"status": "Processing prompt...", "status_code": "processing"}),
        ": keepalive\n\n",
        _sse({"content": "Hello"}),
        _sse({"content": " world"}),
        _sse(finish="stop", usage={"prompt_tokens": 3, "completion_tokens": 2,
                                   "ttft_ms": 10.0, "tokens_per_sec": 5.0}),
        "data: [DONE]\n\n"])
    assert [o["message"]["content"] for o in objs] == ["Hello", " world", ""]
    assert [o["done"] for o in objs] == [False, False, True]
    final = objs[-1]
    assert final["done_reason"] == "stop"
    assert final["prompt_eval_count"] == 3 and final["eval_count"] == 2
    assert all(line.endswith(b"\n") and b"\n" not in line[:-1] for line in raw)
    assert all(line.strip() for line in raw)
    assert upstream.closed is True


def test_stream_translation_generate_uses_the_response_field():
    objs, _raw, _up = _translate([_sse({"content": "hi"}), _sse(finish="length")],
                                 kind="generate")
    assert objs[0]["response"] == "hi" and "message" not in objs[0]
    assert objs[-1]["done_reason"] == "length"


def test_stream_thinking_lines_only_when_requested():
    items = [_sse({"reasoning_content": "step"}), _sse({"content": "ans"}),
             _sse(finish="stop")]
    off, _r, _u = _translate(items, want_thinking=False)
    assert all("thinking" not in o["message"] for o in off)
    on, _r, _u = _translate(items, want_thinking=True)
    assert on[0]["message"]["thinking"] == "step" and on[0]["message"]["content"] == ""
    assert on[1]["message"]["content"] == "ans"


def test_stream_stop_sequence_ends_the_reply_and_closes_upstream():
    objs, _raw, upstream = _translate([
        _sse({"content": "Hello"}), _sse({"content": " wo"}), _sse({"content": "rld"}),
        _sse({"content": " never read"}), _sse(finish="stop")],
        stop=["world"])
    assert "".join(o["message"]["content"] for o in objs) == "Hello "
    assert objs[-1]["done"] is True and objs[-1]["done_reason"] == "stop"
    assert upstream.closed is True
    assert upstream.handed == 3
    assert "eval_count" not in objs[-1]


def test_stream_releases_a_held_partial_stop_at_the_end():
    objs, _raw, _up = _translate([
        _sse({"content": "ab<"}), _sse(finish="stop")], stop=["</s>"])
    assert "".join(o["message"]["content"] for o in objs) == "ab<"
    assert objs[-1]["done_reason"] == "stop"


def test_stream_error_finish_is_one_error_line_without_the_error_text_as_content():
    objs, _raw, upstream = _translate([
        _sse({"content": "partial"}),
        _sse({"content": "\n[inference error: out of memory]"}),
        _sse(finish="error"),
        "data: [DONE]\n\n"])
    assert objs[0]["message"]["content"] == "partial"
    assert objs[-1] == {"error": "[inference error: out of memory]"}
    assert not any("out of memory" in o.get("message", {}).get("content", "")
                   for o in objs)
    assert upstream.closed is True


def test_stream_a_text_that_only_looks_like_an_error_is_still_content():
    objs, _raw, _up = _translate([
        _sse({"content": "[inference error handling is covered below]"}),
        _sse({"content": " ok"}), _sse(finish="stop")])
    text = "".join(o["message"]["content"] for o in objs)
    assert text == "[inference error handling is covered below] ok"


def test_stream_localm_error_chunk_becomes_the_error_line():
    objs, _raw, _up = _translate([
        _sse({"content": "Model failed to load"},
             extra={"localm_error": {"status": 503, "detail": "Model failed to load"}}),
        _sse(finish="error", extra={"localm_error": {"status": 503,
                                                     "detail": "Model failed to load"}})])
    assert objs == [{"error": "Model failed to load"}]


def test_iter_sse_json_handles_arbitrary_byte_chunking_and_utf8():
    payload = ("data: " + json.dumps({"x": "héllo"}, ensure_ascii=False) + "\n\n"
               + "data: " + json.dumps({"y": 2}) + "\n\n").encode("utf-8")

    async def chunks():
        for i in range(0, len(payload), 3):
            yield payload[i:i + 3]

    async def go():
        return [o async for o in P.iter_sse_json(chunks())]

    assert _run(go()) == [{"x": "héllo"}, {"y": 2}]


def test_iter_sse_json_skips_comments_done_and_garbage():
    upstream = _Upstream([": keepalive\n\n", "data: [DONE]\n\n", "data: {not json}\n\n",
                          "event: ping\n\n", "data: [1,2]\n\n", 'data: {"ok":1}\n\n'])

    async def go():
        return [o async for o in P.iter_sse_json(upstream)]

    assert _run(go()) == [{"ok": 1}]
    assert upstream.closed is True


# ------------------------------------------------------------------ listings

def test_digest_of_strips_the_prefix_and_tolerates_absence():
    assert P.digest_of({"sha256": "sha256:abc"}) == "abc"
    assert P.digest_of({"sha256": "abc"}) == "abc"
    assert P.digest_of({}) == ""
    assert P.digest_of({"sha256": None}) == ""


def test_model_details_shape():
    d = P.model_details(fmt="gguf", family="llama")
    assert d["format"] == "gguf" and d["family"] == "llama"
    assert d["families"] == ["llama"] and d["parent_model"] == ""
    assert P.model_details(fmt="gguf")["families"] is None


# ------------------------------------------------------------------ collect

def _collect(objs, kind="chat"):
    async def lines():
        for obj in objs:
            yield P.encode_line(obj)

    return _run(P.collect_reply(lines(), kind))


def test_collect_reply_merges_a_chat_stream_into_one_object():
    out = _collect([
        P.reply_object("chat", "m", thinking="because "),
        P.reply_object("chat", "m", thinking="reasons"),
        P.reply_object("chat", "m", content="Hel"),
        P.reply_object("chat", "m", content="lo"),
        P.reply_object("chat", "m", done=True, reason="stop", stats={"eval_count": 2}),
    ])
    assert out["message"] == {"role": "assistant", "content": "Hello",
                              "thinking": "because reasons"}
    assert out["done"] is True and out["done_reason"] == "stop" and out["eval_count"] == 2


def test_collect_reply_merges_a_generate_stream():
    out = _collect([P.reply_object("generate", "m", content="a"),
                    P.reply_object("generate", "m", content="b"),
                    P.reply_object("generate", "m", done=True)], kind="generate")
    assert out["response"] == "ab" and out["done"] is True and "thinking" not in out


def test_collect_reply_turns_an_error_line_into_a_500():
    with pytest.raises(P.OllamaError) as exc:
        _collect([P.reply_object("chat", "m", content="par"), {"error": "boom"}])
    assert exc.value.status == 500 and exc.value.message == "boom"


def test_collect_reply_without_a_final_object_is_a_502():
    with pytest.raises(P.OllamaError) as exc:
        _collect([P.reply_object("chat", "m", content="par")])
    assert exc.value.status == 502
