# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI ``/v1/responses`` on top of the chat route: input items, instructions,
function tools and their outputs, ``text.format`` and the sampling fields reach
the model as the equivalent chat request; the Response object and the event
stream come back in the Responses shape; a stored response continues through
``previous_response_id`` for the principal that created it only."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference import responses_protocol as R
from localm.inference.http_server import create_app
from localm.inference.tool_calling import CLOSE_TAG, OPEN_TAG

MODEL = "resp-model"
KEY = "lm-responses-test-key"
WEATHER = {"type": "function", "name": "get_weather", "description": "Weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                          "required": ["city"]}}
PERSON = {"type": "object", "properties": {"name": {"type": "string"},
                                           "age": {"type": "integer"}},
          "required": ["name", "age"]}


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
    from localm.inference.routes import responses as routes
    routes.STORE = R.ResponseStore()
    return root


class Served:
    def __init__(self, *replies, finish="stop"):
        self.calls: list = []
        self.replies = list(replies) or [["Hello"]]
        engine = MagicMock()

        def chat_stream(messages, **kwargs):
            self.calls.append((messages, kwargs))
            tokens = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
            yield from tokens

        engine.chat_stream.side_effect = chat_stream
        engine.unsupported_sampling.return_value = []
        engine.display_name = MODEL
        engine.model_path = ""
        engine.count_tokens.return_value = 2
        engine.count_messages_tokens.return_value = 9
        engine.gpu_placement = None
        engine.last_finish_reason = finish
        engine.context_capacity.return_value = 4096
        engine.supports_images = True
        engine.loaded = True
        self.engine = engine
        self.client = TestClient(create_app(engine))

    def post(self, headers=None, **body):
        payload = {"model": MODEL, "input": "hi", **body}
        return self.client.post("/v1/responses", json=payload, headers=headers or {})

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


def test_a_text_reply(home):
    r = Served(["Hel", "lo"]).post()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "response" and body["status"] == "completed"
    assert body["id"].startswith("resp_")
    [item] = body["output"]
    assert item["type"] == "message" and item["role"] == "assistant"
    assert item["content"][0] == {"type": "output_text", "text": "Hello", "annotations": [],
                                  "logprobs": []}
    assert body["usage"]["input_tokens"] == 9 and body["usage"]["output_tokens"] == 2


def test_instructions_and_input_items_reach_the_model(home):
    served = Served()
    served.post(instructions="Be terse.", input=[
        {"role": "developer", "content": "Use metric."},
        {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]}],
        max_output_tokens=50, temperature=0.3, top_p=0.8)
    roles = [m["role"] for m in served.messages]
    assert roles[0] == "system" and served.messages[0]["content"].startswith("Be terse.")
    assert served.messages[-1] == {"role": "user", "content": "weather?"}
    assert any(m["role"] == "system" and m["content"].startswith("Use metric.")
               for m in served.messages)
    kw = served.kwargs
    assert (kw["max_tokens"], kw["temperature"], kw["top_p"]) == (50, 0.3, 0.8)


def test_an_input_image_reaches_the_model(home):
    served = Served()
    url = "data:image/png;base64,iVBORw0KGgo="
    served.post(input=[{"role": "user", "content": [
        {"type": "input_text", "text": "what is this"},
        {"type": "input_image", "image_url": url}]}])
    parts = served.messages[-1]["content"]
    assert parts[1] == {"type": "image_url", "image_url": {"url": url}}


def test_a_function_call_comes_back_as_an_item(home):
    body = Served(["Checking. ", call()]).post(tools=[WEATHER]).json()
    kinds = [i["type"] for i in body["output"]]
    assert kinds == ["message", "function_call"]
    fc = body["output"][1]
    assert fc["name"] == "get_weather" and json.loads(fc["arguments"]) == {"city": "Oslo"}
    assert fc["call_id"] and fc["id"].startswith("fc_") and fc["status"] == "completed"


def test_a_function_call_output_round_trip(home):
    served = Served(["It is sunny."])
    r = served.post(tools=[WEATHER], input=[
        {"role": "user", "content": "weather in Oslo?"},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather",
         "arguments": '{"city": "Oslo"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny, 21C"}])
    assert r.status_code == 200, r.text
    rendered = json.dumps(served.messages)
    assert "sunny, 21C" in rendered and rendered.index("Oslo") < rendered.index("sunny, 21C")


def test_text_format_json_schema_constrains_the_reply(home):
    served = Served(['{"name": "Ada", "age": 36}'])
    body = served.post(text={"format": {"type": "json_schema", "name": "person",
                                        "schema": PERSON, "strict": True}}).json()
    assert json.loads(body["output"][0]["content"][0]["text"]) == {"name": "Ada", "age": 36}
    assert '"\\"age\\""' in served.kwargs["grammar"]


def test_reasoning_comes_back_as_a_reasoning_item(home):
    body = Served(["<think>plan</think>", "Done"]).post().json()
    reasoning, message = body["output"]
    assert reasoning["type"] == "reasoning"
    assert reasoning["content"] == [{"type": "reasoning_text", "text": "plan"}]
    assert message["content"][0]["text"] == "Done"


def test_reasoning_effort_none_turns_thinking_off(home):
    served = Served()
    served.post(reasoning={"effort": "none"})
    assert served.kwargs["thinking"] is False


def test_a_length_finish_is_incomplete(home):
    body = Served(["abc"], finish="length").post().json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_the_event_stream(home):
    evs = events(Served(["<think>p</think>", "Hel", "lo", call()]).post(
        stream=True, tools=[WEATHER]))
    kinds = [e["type"] for e in evs]
    assert kinds[:2] == ["response.created", "response.in_progress"]
    assert kinds[-1] == "response.completed"
    assert [e["sequence_number"] for e in evs] == list(range(len(evs)))
    added = [e["item"]["type"] for e in evs if e["type"] == "response.output_item.added"]
    assert added == ["reasoning", "message", "function_call"]
    text = "".join(e["delta"] for e in evs if e["type"] == "response.output_text.delta")
    assert text == "Hello"
    [done] = [e for e in evs if e["type"] == "response.function_call_arguments.done"]
    assert json.loads(done["arguments"]) == {"city": "Oslo"}
    final = evs[-1]["response"]
    assert [i["type"] for i in final["output"]] == ["reasoning", "message", "function_call"]
    assert final["usage"]["input_tokens"] == 9 and final["usage"]["output_tokens"] > 0
    assert final["output"][1]["content"][0]["text"] == "Hello"


def test_a_failed_stream_ends_with_response_failed(home):
    served = Served()

    def broken(messages, **kwargs):
        yield "partial"
        raise RuntimeError("out of memory")
    served.engine.chat_stream.side_effect = broken
    evs = events(served.post(stream=True))
    assert evs[-1]["type"] == "response.failed"
    assert "out of memory" in evs[-1]["response"]["error"]["message"]


def test_previous_response_id_continues_the_conversation(home):
    served = Served(["Ada is 36."], ["She is a mathematician."])
    first = served.post(input="Who is Ada?").json()
    second = served.post(input="What does she do?", previous_response_id=first["id"])
    assert second.status_code == 200, second.text
    rendered = json.dumps(served.messages)
    assert rendered.index("Who is Ada?") < rendered.index("Ada is 36.") \
        < rendered.index("What does she do?")
    assert second.json()["previous_response_id"] == first["id"]


def test_an_unknown_previous_response_id_is_a_404(home):
    r = Served().post(previous_response_id="resp_nope")
    assert r.status_code == 404
    assert r.json()["error"]["param"] == "previous_response_id"


def test_store_false_keeps_nothing(home):
    served = Served(["Hi."])
    rid = served.post(store=False).json()["id"]
    assert served.post(previous_response_id=rid).status_code == 404


def test_instructions_do_not_carry_over(home):
    served = Served(["one"], ["two"])
    rid = served.post(instructions="SECRET-RULES", input="hi").json()["id"]
    served.post(previous_response_id=rid, input="again")
    assert "SECRET-RULES" not in json.dumps(served.messages)


def test_another_key_cannot_continue_a_stored_response(home, monkeypatch):
    from localm.auth import create_key
    monkeypatch.setenv("LOCALM_API_KEY", KEY)
    other = create_key("other", ["chat"])["key"]
    served = Served(["Secret."], ["leak?"])
    owner = {"Authorization": f"Bearer {KEY}"}
    rid = served.post(headers=owner).json()["id"]
    r = served.post(headers={"Authorization": f"Bearer {other}"}, previous_response_id=rid)
    assert r.status_code == 404
    assert served.post(headers=owner, previous_response_id=rid).status_code == 200


@pytest.mark.parametrize("extra, needle", [
    ({"tools": [{"type": "web_search"}]}, "web_search"),
    ({"background": True}, "background"),
    ({"input": [{"type": "item_reference", "id": "x"}]}, "item_reference"),
    ({"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]},
     "input_file"),
    ({"text": {"format": {"type": "yaml"}}}, "yaml"),
    ({"top_logprobs": 2}, "top_logprobs"),
    ({"tool_choice": {"type": "allowed_tools"}}, "tool_choice"),
    ({"include": ["message.output_text.logprobs"]}, "logprobs"),
    ({"include": ["something.new"]}, "something.new"),
    ({"text": {"verbosity": "low"}}, "verbosity"),
    ({"truncation": "middle"}, "truncation"),
    ({"input": [{"role": "tool", "content": "x"}]}, "role"),
])
def test_an_unsupported_request_is_an_openai_400(home, extra, needle):
    served = Served()
    r = served.post(**extra, **({"tools": [WEATHER]} if "tool_choice" in extra else {}))
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and needle in err["message"]
    assert served.calls == []


def test_tool_choice_required_without_tools_is_a_400(home):
    served = Served()
    r = served.post(tool_choice="required")
    assert r.status_code == 400, r.text
    assert "required" in r.json()["error"]["message"]
    assert served.calls == []


def test_accepted_no_op_fields_do_not_refuse(home):
    served = Served()
    r = served.post(include=["reasoning.encrypted_content"], truncation="auto",
                    text={"verbosity": "medium"}, metadata={"k": "v"}, user="u1",
                    prompt_cache_key="c", max_tool_calls=3)
    assert r.status_code == 200, r.text
    assert r.json()["metadata"] == {"k": "v"} and r.json()["truncation"] == "auto"


def test_a_streamed_response_can_be_continued(home):
    served = Served(["Ada is 36."], ["noted"])
    evs = events(served.post(input="Who is Ada?", stream=True))
    rid = evs[-1]["response"]["id"]
    assert evs[0]["response"]["id"] == rid
    r = served.post(input="And then?", previous_response_id=rid)
    assert r.status_code == 200, r.text
    rendered = json.dumps(served.messages)
    assert rendered.index("Who is Ada?") < rendered.index("Ada is 36.") < rendered.index("And then?")


@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_response_is_not_stored(home, stream):
    served = Served()

    def broken(messages, **kwargs):
        yield "partial"
        raise RuntimeError("boom")
    served.engine.chat_stream.side_effect = broken
    r = served.post(stream=stream)
    final = events(r)[-1]["response"] if stream else r.json()
    assert final["status"] == "failed", final
    assert "boom" in final["error"]["message"]
    assert served.post(previous_response_id=final["id"]).status_code == 404


def test_a_cross_origin_page_reaches_the_route(home):
    served = Served(["Hi."])
    r = served.client.post("/v1/responses", json={"model": MODEL, "input": "hi"},
                           headers={"Origin": "http://localhost:9999"})
    assert r.status_code == 200, r.text


class TestStore:
    CONV = [{"role": "user", "content": "x"}]

    def test_a_principal_sees_only_its_own_responses(self):
        store = R.ResponseStore()
        assert store.put("p1", "resp_a", self.CONV) is True
        assert store.get("p1", "resp_a") == self.CONV
        assert store.get("p2", "resp_a") is None
        assert store.get(None, "resp_a") is None

    def test_the_oldest_is_dropped_past_the_count_bound(self):
        store = R.ResponseStore(max_items=2)
        for i in range(3):
            store.put(None, f"resp_{i}", [])
        assert store.get(None, "resp_0") is None
        assert store.get(None, "resp_1") == [] and store.get(None, "resp_2") == []

    def test_the_oldest_is_dropped_past_the_byte_bound(self):
        big = [{"role": "user", "content": "x" * 400}]
        store = R.ResponseStore(max_bytes=1000)
        for i in range(3):
            store.put(None, f"resp_{i}", big)
        assert store.get(None, "resp_0") is None and store.get(None, "resp_2") == big

    def test_the_byte_bound_counts_utf8_bytes(self):
        store = R.ResponseStore(max_bytes=1000)
        store.put(None, "resp_small", [])
        assert store.put(None, "resp_cjk", [{"role": "user", "content": "\u4e2d" * 340}]) is False
        assert store.get(None, "resp_small") == []

    def test_a_response_larger_than_the_bound_is_not_kept_and_evicts_nothing(self):
        store = R.ResponseStore(max_bytes=200)
        store.put(None, "resp_small", [])
        assert store.put(None, "resp_big", [{"role": "user", "content": "x" * 300}]) is False
        assert store.get(None, "resp_big") is None
        assert store.get(None, "resp_small") == []

    def test_a_flooding_principal_evicts_its_own_entries_first(self):
        store = R.ResponseStore(max_items=4)
        store.put("quiet", "resp_q", self.CONV)
        for i in range(10):
            store.put("loud", f"resp_l{i}", self.CONV)
        assert store.get("quiet", "resp_q") == self.CONV
        assert store.get("loud", "resp_l9") == self.CONV
        assert store.get("loud", "resp_l0") is None

    def test_a_large_entry_evicts_its_owner_not_the_others(self):
        store = R.ResponseStore(max_bytes=1000)
        small = [{"role": "user", "content": "s" * 100}]
        store.put("quiet", "resp_q", small)
        store.put("loud", "resp_l0", [{"role": "user", "content": "x" * 500}])
        kept = store.put("loud", "resp_l1", [{"role": "user", "content": "y" * 500}])
        assert kept is True and store.get("quiet", "resp_q") == small
        assert store.get("loud", "resp_l0") is None

    def test_put_reports_an_entry_dropped_to_make_room(self):
        store = R.ResponseStore(max_bytes=1000)
        store.put("owner", "resp_a", [{"role": "user", "content": "a" * 200}])
        store.put("owner", "resp_b", [{"role": "user", "content": "b" * 200}])
        assert store.put("other", "resp_c", [{"role": "user", "content": "c" * 600}]) is False
        assert store.get("owner", "resp_a") is not None

    def test_an_expired_response_is_gone(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(R.time, "monotonic", lambda: now[0])
        store = R.ResponseStore(ttl=10)
        store.put(None, "resp_a", self.CONV)
        now[0] += 9
        assert store.get(None, "resp_a") == self.CONV
        now[0] += 2
        assert store.get(None, "resp_a") is None

    def test_concurrent_use_keeps_the_bounds_and_the_accounting_exact(self):
        import sys
        import threading
        store = R.ResponseStore(max_items=40, max_bytes=8000)
        errors: list[BaseException] = []

        def worker(principal: str) -> None:
            try:
                for i in range(400):
                    rid = f"resp_{principal}_{i}"
                    store.put(principal, rid, [{"role": "user", "content": "x" * (i % 97)}])
                    store.get(principal, rid)
                    store.get(principal, f"resp_{principal}_{i // 2}")
            except BaseException as exc:
                errors.append(exc)

        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [threading.Thread(target=worker, args=(f"p{k}",)) for k in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sys.setswitchinterval(interval)
        assert errors == []
        entries = list(store._items.values())
        assert len(entries) <= 40
        assert store._bytes == sum(len(e.data) for e in entries) <= 8000
        usage: dict = {}
        for e in entries:
            counts = usage.setdefault(e.principal, [0, 0])
            counts[0] += 1
            counts[1] += len(e.data)
        assert usage == store._usage


def test_a_response_too_large_to_keep_says_store_false(home, monkeypatch):
    from localm.inference.routes import responses as routes
    routes.STORE = R.ResponseStore(max_bytes=50)
    served = Served(["A reply that is long enough to pass the tiny bound."])
    body = served.post(input="hello there").json()
    assert body["status"] == "completed" and body["store"] is False
    assert served.post(previous_response_id=body["id"]).status_code == 404


def test_a_conversation_with_an_image_is_continued(home):
    served = Served(["A red square."], ["Yes."])
    url = "data:image/png;base64,iVBORw0KGgo="
    first = served.post(input=[{"role": "user", "content": [
        {"type": "input_text", "text": "what is this"},
        {"type": "input_image", "image_url": url}]}]).json()
    served.post(input="Is it red?", previous_response_id=first["id"])
    user = [m for m in served.messages if m["role"] == "user"]
    assert {"type": "image_url", "image_url": {"url": url}} in user[0]["content"]
    assert user[-1]["content"] == "Is it red?"


def test_images_in_a_function_output_reach_the_model(home):
    served = Served(["It shows a cat."])
    url = "data:image/png;base64,iVBORw0KGgo="
    r = served.post(tools=[WEATHER], input=[
        {"role": "user", "content": "take a photo"},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": [
            {"type": "input_text", "text": "here: "},
            {"type": "input_image", "image_url": url},
            {"type": "input_text", "text": "done"}]}])
    assert r.status_code == 200, r.text
    tool_at = next(i for i, m in enumerate(served.messages) if m.get("origin") == "tool")
    assert "here: done" in served.messages[tool_at]["content"]
    assert served.messages[tool_at + 1]["role"] == "user"
    assert served.messages[tool_at + 1]["content"] == [{"type": "image_url",
                                                         "image_url": {"url": url}}]


def test_a_file_in_a_function_output_is_a_400(home):
    served = Served()
    r = served.post(input=[
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": [
            {"type": "input_file", "file_id": "f"}]}])
    assert r.status_code == 400 and "input_file" in r.json()["error"]["message"]
    assert served.calls == []


def test_localm_chat_fields_reach_the_model(home):
    served = Served()
    r = served.post(seed=7, stop=["zz"], top_k=5, min_p=0.1, repeat_penalty=1.2)
    assert r.status_code == 200, r.text
    kw = served.kwargs
    assert (kw["seed"], kw["top_k"], kw["repeat_penalty"]) == (7, 5, 1.2)
    assert kw["min_p"] == 0.1


def test_unknown_fields_are_named_in_the_debug_log(home, monkeypatch):
    import localm.debuglog as debuglog
    seen = []
    monkeypatch.setattr(debuglog.logger, "debug",
                        lambda msg, *a, **k: seen.append(str(msg) % a if a else str(msg)))
    served = Served()
    assert served.post(seed=1, frobnicate=2, prompt_cache_key="k").status_code == 200
    assert any("frobnicate" in m and "prompt_cache_key" in m and "seed" not in m for m in seen)
    assert R.ignored_fields(R.ResponsesRequest(input="x", seed=1, top_k=2, zz=3)) == ["zz"]


def test_a_capped_reply_closes_its_message_as_incomplete(home):
    served = Served(["abc"], ["abc"], finish="length")
    body = served.post().json()
    assert body["output"][0]["status"] == "incomplete"
    evs = events(served.post(stream=True))
    [done] = [e for e in evs if e["type"] == "response.output_item.done"]
    assert done["item"]["status"] == "incomplete"
    assert evs[-1]["type"] == "response.incomplete"


def test_auth_is_required_when_a_key_exists(home, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", KEY)
    served = Served()
    r = served.post()
    assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
    assert served.post(headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_openai_sdk_reads_the_reply_and_the_stream(home):
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    import asyncio
    served = Served(["Checking. ", call()], ["It is sunny in Oslo."])

    async def drive():
        client = openai.AsyncOpenAI(
            base_url="http://testserver/v1", api_key="x",
            http_client=httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=served.client.app),
                base_url="http://testserver"))
        first = await client.responses.create(model=MODEL, input="weather in Oslo?",
                                              tools=[WEATHER])
        fc = [i for i in first.output if i.type == "function_call"][0]
        stream = await client.responses.create(
            model=MODEL, previous_response_id=first.id, stream=True,
            input=[{"type": "function_call_output", "call_id": fc.call_id,
                    "output": "sunny"}])
        evs = [e async for e in stream]
        return first, fc, evs

    first, fc, evs = asyncio.run(drive())
    assert fc.name == "get_weather" and json.loads(fc.arguments) == {"city": "Oslo"}
    assert evs[-1].type == "response.completed"
    assert evs[-1].response.output_text == "It is sunny in Oslo."
