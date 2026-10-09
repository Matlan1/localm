# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tool calling for any chat model: request validation, the prompt and history
rendering, the grammars, and the reply parser."""

from __future__ import annotations

import copy
import json

import pytest

from localm.inference.gbnf import TOOL_CALL_TRIGGER, check_grammar_structure
from localm.inference.tool_calling import (
    MAX_TOOLS, CLOSE_TAG, OPEN_TAG, ToolCallStream, ToolChoice, ToolsError,
    extract_calls, has_tool_history, parse_tool_choice, render_messages, tool_grammar,
    tools_prompt, validate_tools,
)
from tests._gbnf_matcher import Grammar

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"enum": ["c", "f"]}},
                   "required": ["city"]}}}
TIME = {"type": "function", "function": {
    "name": "get_time", "parameters": {"type": "object", "properties": {"zone": {"type": "string"}},
                                       "required": ["zone"]}}}


def call_text(name, arguments):
    return f'{OPEN_TAG}\n{json.dumps({"name": name, "arguments": arguments})}\n{CLOSE_TAG}'


# ------------------------------------------------------------------ validation


def test_validate_tools_accepts_openai_function_tools():
    tools = validate_tools([WEATHER, TIME])
    assert [t.name for t in tools] == ["get_weather", "get_time"]
    assert tools[0].description == "Current weather for a city" and tools[1].description == ""
    assert validate_tools(None) == []


def test_a_tool_without_parameters_takes_none():
    tool = validate_tools([{"type": "function", "function": {"name": "ping"}}])[0]
    assert tool.parameters == {"type": "object", "properties": {}}


@pytest.mark.parametrize("bad", [
    "x", {"a": 1}, [1], [{"type": "retrieval"}], [{"type": "function"}],
    [{"type": "function", "function": "f"}],
    [{"type": "function", "function": {"name": ""}}],
    [{"type": "function", "function": {"name": "has space"}}],
    [{"type": "function", "function": {"name": "x" * 65}}],
    [{"type": "function", "function": {"name": 5}}],
    [{"type": "function", "function": {"name": "f", "description": 5}}],
    [{"type": "function", "function": {"name": "f", "parameters": "no"}}],
    [WEATHER, WEATHER],
])
def test_validate_tools_refuses_malformed_tools(bad):
    with pytest.raises(ToolsError):
        validate_tools(bad)


def test_validate_tools_limits():
    many = [{"type": "function", "function": {"name": f"f{i}"}} for i in range(MAX_TOOLS + 1)]
    with pytest.raises(ToolsError, match="at most"):
        validate_tools(many)
    huge = [{"type": "function", "function": {"name": "f", "parameters": {"description": "x" * 70000}}}]
    with pytest.raises(ToolsError, match="bytes"):
        validate_tools(huge)


def test_tool_choice():
    tools = validate_tools([WEATHER, TIME])
    assert parse_tool_choice(None, tools) == ToolChoice("auto")
    assert parse_tool_choice(None, []) == ToolChoice("none")
    for kind in ("auto", "none", "required"):
        assert parse_tool_choice(kind, tools) == ToolChoice(kind)
    assert parse_tool_choice({"type": "function", "function": {"name": "get_time"}}, tools) \
        == ToolChoice("function", "get_time")
    assert parse_tool_choice("none", []) == ToolChoice("none")


@pytest.mark.parametrize("bad", ["sometimes", 5, {"type": "function"}, {"type": "function", "function": {"name": "nope"}},
                                 {"type": "other", "function": {"name": "get_time"}}, ["auto"]])
def test_tool_choice_refuses_nonsense(bad):
    with pytest.raises(ToolsError):
        parse_tool_choice(bad, validate_tools([WEATHER, TIME]))


def test_required_without_tools_is_refused():
    with pytest.raises(ToolsError):
        parse_tool_choice("required", [])


# ------------------------------------------------------------------ prompt and history


def test_prompt_lists_every_tool_and_the_call_format():
    text = tools_prompt(validate_tools([WEATHER, TIME]), ToolChoice("auto"))
    assert "<tools>" in text and "</tools>" in text and "<tool_call>" in text
    lines = [json.loads(l) for l in text.splitlines() if l.startswith('{"type": "function"')]
    assert [l["function"]["name"] for l in lines] == ["get_weather", "get_time"]
    assert lines[0]["function"]["description"] == "Current weather for a city"
    assert "must call" not in text
    assert "must call at least one" in tools_prompt(validate_tools([WEATHER]), ToolChoice("required"))
    assert "must call the function get_time" in tools_prompt(
        validate_tools([WEATHER, TIME]), ToolChoice("function", "get_time"))


def test_the_prompt_goes_into_an_existing_system_message_or_a_new_one():
    tools = validate_tools([WEATHER])
    with_system = render_messages(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
        tools, ToolChoice("auto"))
    assert with_system[0]["content"].startswith("Be brief.\n\n# Tools")
    assert [m["role"] for m in with_system] == ["system", "user"]
    without = render_messages([{"role": "user", "content": "hi"}], tools, ToolChoice("auto"))
    assert [m["role"] for m in without] == ["system", "user"]
    assert without[0]["content"].startswith("# Tools")


def test_no_tools_prompt_when_the_choice_is_none_but_history_still_renders():
    tools = validate_tools([WEATHER])
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
    out = render_messages(messages, tools, ToolChoice("none"))
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert "# Tools" not in json.dumps(out)


def test_history_is_written_in_the_call_format():
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
            {"id": "b", "type": "function", "function": {"name": "get_time", "arguments": {"zone": "CET"}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "sunny"},
        {"role": "tool", "tool_call_id": "b", "content": "noon"},
        {"role": "assistant", "content": "Sunny at noon."},
    ]
    out = render_messages(messages, validate_tools([WEATHER, TIME]), ToolChoice("auto"))
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assistant = out[2]["content"]
    assert assistant.startswith("Let me check.\n")
    content, parsed = extract_calls(assistant)
    assert content == "Let me check." and [(c.name, c.arguments) for c in parsed] == [
        ("get_weather", {"city": "Paris"}), ("get_time", {"zone": "CET"})]
    results = out[3]["content"]
    assert results.count("<tool_response>") == 2 and "sunny" in results and "noon" in results
    assert results.index("sunny") < results.index("noon")


def test_arguments_that_are_not_json_are_kept_as_input():
    out = render_messages([{"role": "assistant", "tool_calls": [
        {"function": {"name": "f", "arguments": "not json"}}]}], [], ToolChoice("none"))
    _content, parsed = extract_calls(out[0]["content"])
    assert parsed[0].arguments == {"input": "not json"}


def test_render_does_not_change_its_input():
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "content": "r"}]
    before = copy.deepcopy(messages)
    render_messages(messages, validate_tools([WEATHER]), ToolChoice("auto"))
    assert messages == before


def test_has_tool_history():
    assert not has_tool_history([{"role": "user", "content": "hi"}])
    assert has_tool_history([{"role": "tool", "content": "x"}])
    assert has_tool_history([{"role": "assistant", "tool_calls": [{}]}])


# ------------------------------------------------------------------ grammar


def accepts(grammar, text):
    return Grammar(grammar).accepts(text)


def test_required_choice_forces_a_well_formed_call_from_the_first_token():
    grammar, lazy, triggers = tool_grammar(validate_tools([WEATHER]), ToolChoice("required"))
    assert lazy is False and triggers is None
    good = call_text("get_weather", {"city": "Paris"})
    assert accepts(grammar, good)
    assert accepts(grammar, good + "\n" + call_text("get_weather", {"city": "Rome", "unit": "c"}))
    assert accepts(grammar, good + "\n")
    assert not accepts(grammar, "sure " + good)
    assert not accepts(grammar, call_text("get_time", {"city": "Paris"}))
    assert not accepts(grammar, call_text("get_weather", {}))
    assert not accepts(grammar, call_text("get_weather", {"city": 5}))
    assert not accepts(grammar, call_text("get_weather", {"city": "Paris", "unit": "k"}))
    assert not accepts(grammar, call_text("get_weather", {"city": "Paris", "extra": 1}))
    assert not accepts(grammar, "")
    assert not accepts(grammar, OPEN_TAG + "\n{}\n" + CLOSE_TAG)


def test_auto_choice_is_lazy_and_triggered_by_the_tool_call_tag():
    grammar, lazy, triggers = tool_grammar(validate_tools([WEATHER]), ToolChoice("auto"))
    assert lazy is True and triggers == [TOOL_CALL_TRIGGER]
    assert accepts(grammar, call_text("get_weather", {"city": "Paris"}))


def test_several_tools_are_alternatives_and_a_named_function_narrows_them():
    tools = validate_tools([WEATHER, TIME])
    grammar, _lazy, _t = tool_grammar(tools, ToolChoice("required"))
    assert accepts(grammar, call_text("get_weather", {"city": "Paris"}))
    assert accepts(grammar, call_text("get_time", {"zone": "CET"}))
    assert not accepts(grammar, call_text("get_time", {"city": "Paris"}))
    only, _lazy, _t = tool_grammar(tools, ToolChoice("function", "get_time"))
    assert accepts(only, call_text("get_time", {"zone": "CET"}))
    assert not accepts(only, call_text("get_weather", {"city": "Paris"}))


def test_a_keyword_the_compiler_cannot_enforce_does_not_loosen_the_rest():
    tool = {"type": "function", "function": {"name": "grep", "parameters": {
        "type": "object", "properties": {"pattern": {"type": "string", "pattern": "^a"}}}}}
    grammar, _lazy, _t = tool_grammar(validate_tools([tool]), ToolChoice("required"))
    assert accepts(grammar, call_text("grep", {"pattern": "zzz"}))
    assert not accepts(grammar, call_text("grep", {"pattern": "zzz", "other": [1]}))
    assert not accepts(grammar, call_text("other", {}))


def test_every_tool_grammar_passes_the_server_structure_check():
    for choice in (ToolChoice("auto"), ToolChoice("required")):
        grammar, _l, _t = tool_grammar(validate_tools([WEATHER, TIME]), choice)
        check_grammar_structure(grammar)


def test_recursive_and_defs_based_parameter_schemas_work():
    tool = {"type": "function", "function": {"name": "tree", "parameters": {
        "$defs": {"Node": {"type": "object", "properties": {"v": {"type": "integer"}, "kids": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}, "required": ["v"]}},
        "type": "object", "properties": {"root": {"$ref": "#/$defs/Node"}}, "required": ["root"]}}}
    grammar, _l, _t = tool_grammar(validate_tools([tool]), ToolChoice("required"))
    assert accepts(grammar, call_text("tree", {"root": {"v": 1, "kids": [{"v": 2}]}}))
    assert not accepts(grammar, call_text("tree", {"root": {"kids": []}}))


# ------------------------------------------------------------------ parser


def stream(chunks, names=None):
    parser = ToolCallStream(names)
    events = []
    for chunk in chunks:
        events += parser.feed(chunk)
    events += parser.finish()
    return events


def text_of(events):
    return "".join(v for k, v in events if k == "text")


def calls_of(events):
    return [(c.name, c.arguments) for k, c in events if k == "call"]


def test_plain_text_passes_through():
    events = stream(["Hello ", "there."])
    assert text_of(events) == "Hello there." and calls_of(events) == []


def test_a_call_is_found_whole_or_split_at_any_boundary():
    full = "Checking.\n" + call_text("get_weather", {"city": "Paris"}) + "\n"
    for size in (1, 2, 3, 5, 7, 11, len(full)):
        chunks = [full[i:i + size] for i in range(0, len(full), size)]
        events = stream(chunks)
        assert calls_of(events) == [("get_weather", {"city": "Paris"})], size
        assert text_of(events).strip() == "Checking.", size


def test_several_calls_and_surrounding_text():
    full = ("A " + call_text("get_weather", {"city": "Paris"}) + "\n\n"
            + call_text("get_time", {"zone": "CET"}) + "\nDone.")
    events = stream([full])
    assert calls_of(events) == [("get_weather", {"city": "Paris"}), ("get_time", {"zone": "CET"})]
    assert text_of(events) == "A Done."


def test_text_is_released_before_the_call_arrives():
    parser = ToolCallStream()
    first = parser.feed("Looking it up. <tool")
    assert text_of(first) == "Looking it up. "
    assert parser.feed("_call>") == []


def test_a_partial_open_tag_that_never_completes_is_text():
    events = stream(["a <tool", "x"])
    assert text_of(events) == "a <toolx" and calls_of(events) == []


def test_a_block_left_open_at_the_end_is_a_call_when_its_body_is_valid():
    events = stream([OPEN_TAG + '\n{"name": "get_time", "arguments": {"zone": "CET"}}'])
    assert calls_of(events) == [("get_time", {"zone": "CET"})]
    broken = stream([OPEN_TAG + '\n{"name": "get_time", "argu'])
    assert calls_of(broken) == [] and text_of(broken).startswith(OPEN_TAG)


def test_an_invalid_block_stays_visible_as_text():
    events = stream([f"{OPEN_TAG}not json{CLOSE_TAG} after"])
    assert calls_of(events) == []
    assert OPEN_TAG in text_of(events) and "not json" in text_of(events) and "after" in text_of(events)


def test_a_call_to_an_unknown_function_stays_text():
    block = call_text("rm_rf", {})
    events = stream([block], names={"get_time"})
    assert calls_of(events) == [] and block in text_of(events)
    assert calls_of(stream([block])) == [("rm_rf", {})]


def test_argument_spellings():
    body = '{"name": "f", "parameters": {"a": 1}}'
    assert calls_of(stream([f"{OPEN_TAG}{body}{CLOSE_TAG}"])) == [("f", {"a": 1})]
    body = '{"name": "f", "args": {"a": 2}}'
    assert calls_of(stream([f"{OPEN_TAG}{body}{CLOSE_TAG}"])) == [("f", {"a": 2})]
    body = '{"name": "f", "arguments": "{\\"a\\": 3}"}'
    assert calls_of(stream([f"{OPEN_TAG}{body}{CLOSE_TAG}"])) == [("f", {"a": 3})]
    body = '{"name": "f"}'
    assert calls_of(stream([f"{OPEN_TAG}{body}{CLOSE_TAG}"])) == [("f", {})]
    body = '{"name": "f", "arguments": [1]}'
    assert calls_of(stream([f"{OPEN_TAG}{body}{CLOSE_TAG}"])) == []


def test_a_reply_that_is_only_a_json_call_object_is_a_call():
    names = {"get_weather"}
    events = stream(['{"name": "get_weather", ', '"parameters": {"city": "Rome"}}'], names)
    assert calls_of(events) == [("get_weather", {"city": "Rome"})] and text_of(events) == ""
    listed = stream(['[{"name": "get_weather", "arguments": {"city": "A"}}, {"name": "get_weather", "arguments": {"city": "B"}}]'], names)
    assert [a for _n, a in calls_of(listed)] == [{"city": "A"}, {"city": "B"}]


def test_a_json_reply_that_is_not_a_call_is_text():
    for reply in ('{"answer": 42}', '{"name": "other", "arguments": {}}', '{"unclosed": ', '[1, 2]'):
        events = stream([reply], {"get_weather"})
        assert calls_of(events) == [] and text_of(events) == reply, reply


def test_json_after_text_is_not_mistaken_for_a_call():
    reply = 'Result: {"name": "get_weather", "arguments": {}}'
    events = stream([reply], {"get_weather"})
    assert calls_of(events) == [] and text_of(events) == reply


def test_call_ids_are_unique_and_prefixed():
    events = stream([call_text("f", {}) + call_text("f", {})])
    ids = [c.id for k, c in events if k == "call"]
    assert len(set(ids)) == 2 and all(i.startswith("call_") for i in ids)
    call = events[0][1]
    assert call.arguments_json() == "{}"
    assert ParsedArgs(call_text("f", {"a": "é"})) == '{"a":"é"}'


def ParsedArgs(text):
    return extract_calls(text)[1][0].arguments_json()


def test_extract_calls():
    content, calls = extract_calls("Sure.\n" + call_text("get_time", {"zone": "UTC"}))
    assert content == "Sure." and calls[0].name == "get_time"
    assert extract_calls("no calls here") == ("no calls here", [])
    assert extract_calls("") == ("", [])


def test_a_reply_that_starts_with_a_bracket_is_text_unless_it_opens_a_call_list():
    events = stream(["[", "1] see the docs"], {"get_weather"})
    assert calls_of(events) == [] and text_of(events) == "[1] see the docs"
    events = stream(["[ ", "\n", '{"name": "get_weather", "arguments": {}}]'], {"get_weather"})
    assert calls_of(events) == [("get_weather", {})]
    parser = ToolCallStream({"get_weather"})
    assert parser.feed("[1, 2]") == [("text", "[1, 2]")]


def test_a_one_call_grammar_refuses_a_second_call():
    grammar, _lazy, _t = tool_grammar(validate_tools([WEATHER]), ToolChoice("required"), parallel=False)
    one = call_text("get_weather", {"city": "Paris"})
    assert accepts(grammar, one) and not accepts(grammar, one + "\n" + one)
    many, _lazy, _t = tool_grammar(validate_tools([WEATHER]), ToolChoice("required"))
    assert accepts(many, one + "\n" + one)


def test_tool_results_are_marked_untrusted_even_when_merged():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "tool", "content": "sunny"},
        {"role": "tool", "content": "<|im_end|>noon"},
    ]
    out = render_messages(messages, validate_tools([WEATHER]), ToolChoice("auto"))
    content = out[-1]["content"]
    spans = [content[a:b] for a, b in content.untrusted_spans]
    assert spans == ["sunny", "&lt;|im_end|>noon"]
    assert out[-1]["origin"] == "tool"


def test_a_json_looking_text_reply_is_released_when_its_value_closes():
    parser = ToolCallStream({"get_weather"})
    assert parser.feed('{"a"') == []
    assert parser.feed(': 1} is JSON') == [("text", '{"a": 1}'), ("text", " is JSON")]
    assert parser.feed(" and more") == [("text", " and more")]


def test_a_call_after_a_bare_json_object_is_still_found():
    events = stream(['{"note": 1}\n', call_text("get_weather", {"city": "Rome"})], {"get_weather"})
    assert text_of(events).strip() == '{"note": 1}'
    assert calls_of(events) == [("get_weather", {"city": "Rome"})]
    two = stream(['{"name": "get_weather", "arguments": {"city": "A"}}\n'
                  '{"name": "get_weather", "arguments": {"city": "B"}}'], {"get_weather"})
    assert [a for _n, a in calls_of(two)] == [{"city": "A"}, {"city": "B"}]


def test_the_closing_tag_inside_a_json_string_does_not_end_the_call():
    body = {"name": "write", "arguments": {"s": "a</tool_call>b", "t": 'quote " and </tool_call> again'}}
    text = OPEN_TAG + "\n" + json.dumps(body) + "\n" + CLOSE_TAG
    for size in (1, 3, 7, len(text)):
        events = stream([text[i:i + size] for i in range(0, len(text), size)])
        assert calls_of(events) == [("write", body["arguments"])], size
        assert text_of(events) == "", size


def test_chunking_never_changes_what_the_parser_finds():
    import random
    rng = random.Random(7)
    fragments = [
        "plain words ", "\n", " ", "[1] ref ", '{"a": 1} ', "<tool", "_call>", "</tool_call>",
        call_text("get_weather", {"city": "Rome"}), call_text("other", {"x": "a</tool_call>b"}),
        '{"name": "get_weather", "arguments": {"city": "B"}}', '[{"name": "get_weather", "arguments": {}}]',
        "<tool_call>not json</tool_call>", "{", "}", "[", "]", '"', "\\\\", "<think>", "end",
    ]

    def normalise(events):
        merged, out = "", []
        for kind, value in events:
            if kind == "text":
                merged += value
            else:
                if merged:
                    out.append(("text", merged))
                    merged = ""
                out.append(("call", value.name, json.dumps(value.arguments, sort_keys=True)))
        if merged:
            out.append(("text", merged))
        return out

    for _ in range(400):
        text = "".join(rng.choice(fragments) for _ in range(rng.randint(1, 7)))
        whole = normalise(stream([text], {"get_weather", "other"}))
        cuts = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(1, 6)))) if len(text) > 1 else []
        pieces_ = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)], strict=True)]
        assert normalise(stream(pieces_, {"get_weather", "other"})) == whole, (text, pieces_)


def schema_tool(name, props):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": props, "required": list(props)}}}


def test_an_unenforceable_keyword_is_dropped_and_the_rest_still_enforced():
    tool = schema_tool("set_temp", {
        "celsius": {"type": "number", "minimum": -50, "maximum": 100},
        "unit": {"enum": ["c", "f"]},
        "pattern": {"type": "string", "pattern": "^[a-z]+$"},
    })
    grammar, _lazy, _t = tool_grammar(validate_tools([tool]), ToolChoice("required"))
    good = {"celsius": 21.5, "unit": "c", "pattern": "abc"}
    assert accepts(grammar, call_text("set_temp", good))
    assert accepts(grammar, call_text("set_temp", {**good, "celsius": 900}))
    assert accepts(grammar, call_text("set_temp", {**good, "pattern": "NOT lower"}))
    assert not accepts(grammar, call_text("set_temp", {"celsius": 1, "pattern": "a"}))
    assert not accepts(grammar, call_text("set_temp", {**good, "unit": "k"}))
    assert not accepts(grammar, call_text("set_temp", {**good, "celsius": "hot"}))


def test_a_schema_that_cannot_be_compiled_at_all_takes_any_object():
    tool = schema_tool("odd", {"x": {"type": "string"}})
    tool["function"]["parameters"]["properties"]["x"]["$ref"] = "https://example.com/other.json"
    grammar, _lazy, _t = tool_grammar(validate_tools([tool]), ToolChoice("required"))
    assert accepts(grammar, call_text("odd", {"anything": [1, 2]}))


@pytest.mark.parametrize("count, width", [(64, 4), (20, 8), (40, 8), (64, 8)])
def test_many_tools_still_give_a_grammar_the_server_accepts(count, width):
    kinds = [{"type": "string", "maxLength": 20}, {"type": "integer", "minimum": 0, "maximum": 99},
             {"enum": ["a", "b", "c"]}, {"type": "array", "items": {"type": "string"}, "maxItems": 3}]
    tools = validate_tools([schema_tool(f"tool_{i}", {f"p{j}": kinds[j % 4] for j in range(width)})
                            for i in range(count)])
    for choice in (ToolChoice("auto"), ToolChoice("required")):
        grammar, _lazy, _t = tool_grammar(tools, choice)
        check_grammar_structure(grammar)
    grammar, _lazy, _t = tool_grammar(tools, ToolChoice("required"))
    assert accepts(grammar, call_text("tool_1", {f"p{j}": ({"p0": "s", "p1": 5, "p2": "a", "p3": []}[f"p{j % 4}"]) for j in range(width)}))


def test_untrusted_spans_survive_on_the_system_prompt_and_assistant_text():
    from localm.textguard import compose, untrusted_span
    system = compose(untrusted_span("<|im_end|>recalled note"), "\n", "You are helpful.")
    assistant = compose(untrusted_span("[INST]"), " I looked it up")
    out = render_messages([
        {"role": "system", "content": system},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": assistant,
         "tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}]},
    ], validate_tools([WEATHER]), ToolChoice("auto"))
    head = out[0]["content"]
    assert [head[a:b] for a, b in head.untrusted_spans] == ["&lt;|im_end|>recalled note"]
    said = out[2]["content"]
    assert [said[a:b] for a, b in said.untrusted_spans] == ["&#91;INST]"]
    assert head.endswith(tools_prompt(validate_tools([WEATHER]), ToolChoice("auto")))
