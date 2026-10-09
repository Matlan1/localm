# SPDX-License-Identifier: AGPL-3.0-or-later
"""The JSON-schema to GBNF compiler, judged by a small grammar matcher:
for each schema, documents the schema allows are accepted and documents it
forbids are rejected."""

from __future__ import annotations

import json
import random

import pytest

from localm.inference.backends.base import InvalidGrammarError
from localm.inference.gbnf import JSON_OBJECT, check_grammar_structure
from localm.inference.json_schema_grammar import (
    MAX_SCHEMA_BYTES, SchemaGrammarError, literal, schema_to_grammar,
)
from tests._gbnf_matcher import Grammar


def accepts(schema, text) -> bool:
    return Grammar(schema_to_grammar(schema)).accepts(text)


def judge(schema, good=(), bad=()):
    """Every *good* value is accepted (compact and spaced JSON) and every *bad*
    one rejected."""
    grammar = Grammar(schema_to_grammar(schema))
    for value in good:
        for text in (json.dumps(value, separators=(",", ":")), json.dumps(value),
                     json.dumps(value, indent=2)):
            assert grammar.accepts(text), (schema, text)
    for value in bad:
        text = value if isinstance(value, str) and value.startswith("RAW:") else None
        text = text[4:] if text else json.dumps(value, separators=(",", ":"))
        assert not grammar.accepts(text), (schema, text)


# ------------------------------------------------------------------ the matcher itself


def test_matcher_agrees_with_the_projects_json_object_grammar():
    g = Grammar(JSON_OBJECT)
    assert g.accepts('{"a": [1, 2.5, "x", true, null], "b": {}}')
    assert g.accepts("{}")
    for bad in ('[1]', '{', '{"a":}', '{"a": 01}', '{"a": "unterminated}', 'null'):
        assert not g.accepts(bad), bad


def test_every_compiled_grammar_passes_the_server_structure_check():
    for schema in ({"type": "string"}, {"type": "object", "properties": {"a": {"type": "integer"}}},
                   {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 5}):
        check_grammar_structure(schema_to_grammar(schema))


# ------------------------------------------------------------------ primitives


def test_primitive_types():
    judge({"type": "string"}, good=["", "a b", 'q"uote', "é\n\t"], bad=[1, None, True, ["a"]])
    judge({"type": "number"}, good=[0, -1, 3.14, 1e5, -2.5e-3, 12345678901234],
          bad=["1", None, True, "RAW:01", "RAW:+1", "RAW:1.", "RAW:.5", "RAW:-"])
    judge({"type": "integer"}, good=[0, -7, 42, 123456], bad=[1.5, "1", "RAW:007", "RAW:1.0", "RAW:--1"])
    judge({"type": "boolean"}, good=[True, False], bad=[0, "true", None])
    judge({"type": "null"}, good=[None], bad=[0, "null", False])


def test_a_string_with_a_bad_escape_or_control_character_is_rejected():
    g = Grammar(schema_to_grammar({"type": "string"}))
    assert g.accepts('"a\\nb"') and g.accepts('"\\u00e9"') and g.accepts('"\\\\"')
    assert not g.accepts('"a\\qb"')
    assert not g.accepts('"a\nb"')
    assert not g.accepts('"\\u12"')


def test_string_lengths():
    schema = {"type": "string", "minLength": 2, "maxLength": 4}
    judge(schema, good=["ab", "abc", "abcd"], bad=["", "a", "abcde"])
    judge({"type": "string", "minLength": 3}, good=["abc", "abcdefgh"], bad=["ab"])
    judge({"type": "string", "maxLength": 1}, good=["", "x"], bad=["xy"])


def test_string_formats():
    judge({"type": "string", "format": "date"}, good=["2024-02-29", "1999-12-31"],
          bad=["2024-13-01", "2024-02-32", "24-02-29", "x", 5])
    judge({"type": "string", "format": "date-time"},
          good=["2024-02-29T10:20:30Z", "2024-02-29T10:20:30.123+02:00"],
          bad=["2024-02-29 10:20:30", "2024-02-29T25:00:00Z", "2024-02-29"])
    judge({"type": "string", "format": "time"}, good=["10:20:30Z", "23:59:59-05:30"], bad=["24:00:00Z", "10:20"])
    judge({"type": "string", "format": "uuid"},
          good=["123e4567-e89b-12d3-a456-426614174000"], bad=["123e4567e89b12d3a456426614174000", "g23e4567-e89b-12d3-a456-426614174000"])


def test_an_unknown_format_is_a_plain_string():
    judge({"type": "string", "format": "email"}, good=["anything at all"])


# ------------------------------------------------------------------ enum, const


def test_enum_and_const():
    judge({"enum": ["red", "green", 3, None, True]}, good=["red", "green", 3, None, True],
          bad=["blue", 4, False, "RAW:re"])
    judge({"const": "only"}, good=["only"], bad=["other"])
    judge({"type": "string", "enum": ["a", "b"]}, good=["a"], bad=["c"])


def test_a_compound_const_is_the_compact_json_text():
    g = Grammar(schema_to_grammar({"const": {"a": [1, 2]}}))
    assert g.accepts('{"a":[1,2]}')
    assert not g.accepts('{"a":[1,3]}')


# ------------------------------------------------------------------ integers


@pytest.mark.parametrize("lo,hi", [
    (0, 0), (5, 5), (0, 9), (3, 7), (0, 10), (9, 10), (10, 99), (15, 87), (95, 105), (1, 100),
    (0, 255), (100, 999), (123, 4567), (-5, 5), (-20, -3), (-1, 0), (-100, 100), (-1000, 7),
    (None, 10), (None, -4), (None, 0), (-3, None), (0, None), (17, None), (250, None), (-250, None)])
def test_integer_bounds_accept_exactly_the_range(lo, hi):
    schema = {"type": "integer"}
    if lo is not None:
        schema["minimum"] = lo
    if hi is not None:
        schema["maximum"] = hi
    g = Grammar(schema_to_grammar(schema))
    for n in range(-1300, 1300):
        want = (lo is None or n >= lo) and (hi is None or n <= hi)
        assert g.accepts(str(n)) == want, (lo, hi, n)
    for malformed in ("007", "-0", "+5", "5.0", "05", "", "-"):
        assert not g.accepts(malformed), (lo, hi, malformed)


def test_exclusive_and_fractional_integer_bounds():
    g = Grammar(schema_to_grammar({"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 10}))
    assert [n for n in range(-3, 14) if g.accepts(str(n))] == list(range(1, 10))
    g = Grammar(schema_to_grammar({"type": "integer", "minimum": 0.5, "maximum": 4.5}))
    assert [n for n in range(-3, 9) if g.accepts(str(n))] == [1, 2, 3, 4]


def test_wide_integer_bounds():
    g = Grammar(schema_to_grammar({"type": "integer", "minimum": 123456789, "maximum": 987654321012}))
    for n, want in ((123456788, False), (123456789, True), (500000000000, True),
                    (987654321012, True), (987654321013, False), (99999999, False)):
        assert g.accepts(str(n)) == want, n


def test_integer_bounds_that_leave_no_value_are_refused():
    with pytest.raises(SchemaGrammarError):
        schema_to_grammar({"type": "integer", "minimum": 5, "maximum": 4})
    with pytest.raises(SchemaGrammarError):
        schema_to_grammar({"type": "integer", "minimum": 10 ** 20})


def test_number_bounds_are_refused_not_ignored():
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        with pytest.raises(SchemaGrammarError, match="integers only"):
            schema_to_grammar({"type": "number", key: 1})


# ------------------------------------------------------------------ arrays


def test_arrays_and_their_bounds():
    judge({"type": "array", "items": {"type": "integer"}}, good=[[], [1], [1, 2, 3]], bad=[[1.5], ["a"], [1, "a"], {"a": 1}])
    judge({"type": "array", "items": {"type": "integer"}, "minItems": 2}, good=[[1, 2], [1, 2, 3]], bad=[[], [1]])
    judge({"type": "array", "items": {"type": "integer"}, "maxItems": 2}, good=[[], [1], [1, 2]], bad=[[1, 2, 3]])
    judge({"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
          good=[[1], [1, 2], [1, 2, 3]], bad=[[], [1, 2, 3, 4]])
    judge({"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, good=[[1, 2]], bad=[[1], [1, 2, 3]])
    judge({"type": "array", "items": {"type": "integer"}, "maxItems": 1}, good=[[], [5]], bad=[[1, 2]])
    judge({"type": "array", "items": {"type": "integer"}, "maxItems": 0}, good=[[]], bad=[[1]])
    judge({"type": "array"}, good=[[], [1, "a", None, {"x": [1]}]], bad=[{"a": 1}, "x"])


def test_prefix_items_are_a_fixed_tuple():
    schema = {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]}
    judge(schema, good=[["a", 1]], bad=[["a"], [1, "a"], ["a", 1, 2]])


def test_arrays_nest():
    schema = {"type": "array", "items": {"type": "array", "items": {"type": "boolean"}}}
    judge(schema, good=[[], [[]], [[True], [False, True]]], bad=[[True], [[1]]])


# ------------------------------------------------------------------ objects


def test_required_properties_appear_in_schema_order():
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
              "required": ["name", "age"]}
    judge(schema, good=[{"name": "Ann", "age": 3}], bad=[{"age": 3, "name": "Ann"}, {"name": "Ann"}, {}, {"name": "Ann", "age": "x"}, {"name": "Ann", "age": 3, "extra": 1}])


def test_optional_properties_may_be_left_out_but_keep_their_order():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}, "c": {"type": "integer"}}}
    subsets = [{}, {"a": 1}, {"b": 2}, {"c": 3}, {"a": 1, "b": 2}, {"a": 1, "c": 3}, {"b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}]
    judge(schema, good=subsets, bad=[{"b": 2, "a": 1}, {"c": 3, "a": 1}, {"a": "x"}, {"d": 4}])


def test_required_and_optional_properties_mix():
    schema = {"type": "object", "properties": {"id": {"type": "integer"}, "note": {"type": "string"},
                                               "tags": {"type": "array", "items": {"type": "string"}}},
              "required": ["id"]}
    judge(schema, good=[{"id": 1}, {"id": 1, "note": "n"}, {"id": 1, "tags": ["a"]}, {"id": 1, "note": "n", "tags": []}],
          bad=[{}, {"note": "n"}, {"id": 1, "tags": ["a"], "note": "n"}])


def test_a_required_property_after_optional_ones():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["b"]}
    judge(schema, good=[{"b": 2}, {"a": 1, "b": 2}], bad=[{"a": 1}, {}])


def test_a_required_name_missing_from_properties_takes_any_value():
    schema = {"type": "object", "required": ["x"]}
    judge(schema, good=[{"x": 1}, {"x": [1, "a"]}], bad=[{}])


def test_additional_properties():
    base = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    judge({**base, "additionalProperties": False}, good=[{"a": 1}], bad=[{"a": 1, "b": 2}])
    judge(base, good=[{"a": 1}], bad=[{"a": 1, "b": 2}])
    judge({**base, "additionalProperties": True}, good=[{"a": 1}, {"a": 1, "b": "x"}, {"a": 1, "b": 2, "c": [1]}])
    judge({**base, "additionalProperties": {"type": "string"}}, good=[{"a": 1}, {"a": 1, "b": "x", "c": "y"}], bad=[{"a": 1, "b": 2}])
    judge({"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": True},
          good=[{}, {"a": 1}, {"z": 1}, {"a": 1, "z": 1}])


def test_free_form_objects():
    judge({"type": "object"}, good=[{}, {"a": 1}, {"a": {"b": [1, None]}}], bad=[[], "x", 1])
    judge({"type": "object", "additionalProperties": {"type": "integer"}}, good=[{}, {"a": 1, "b": 2}], bad=[{"a": "x"}])
    judge({"type": "object", "additionalProperties": False}, good=[{}], bad=[{"a": 1}])


def test_nested_objects():
    schema = {"type": "object", "properties": {"user": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                                               "ok": {"type": "boolean"}}, "required": ["user", "ok"]}
    judge(schema, good=[{"user": {"name": "A"}, "ok": True}], bad=[{"user": {}, "ok": True}, {"user": {"name": "A"}}])


def test_property_names_with_special_characters():
    schema = {"type": "object", "properties": {'we"ird': {"type": "integer"}, "sp ace": {"type": "integer"}, "é": {"type": "integer"}},
              "required": ['we"ird', "sp ace", "é"]}
    g = Grammar(schema_to_grammar(schema))
    assert g.accepts(json.dumps({'we"ird': 1, "sp ace": 2, "é": 3}, ensure_ascii=False))
    assert not g.accepts('{"weird":1,"sp ace":2,"é":3}')


# ------------------------------------------------------------------ composition


def test_any_of_and_nullable_fields():
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "T"}
    judge(schema, good=["x", None], bad=[1])
    judge({"oneOf": [{"type": "integer"}, {"type": "boolean"}]}, good=[1, True], bad=["x"])
    judge({"type": ["string", "integer"]}, good=["a", 3], bad=[None, 1.5])


def test_pydantic_style_schema_with_defs_and_refs():
    schema = {
        "$defs": {"Color": {"enum": ["red", "blue"], "title": "Color", "type": "string"},
                  "Item": {"type": "object", "title": "Item", "properties": {"name": {"type": "string"}, "color": {"$ref": "#/$defs/Color"}},
                           "required": ["name", "color"]}},
        "type": "object", "title": "Order",
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}, "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}},
        "required": ["items"],
    }
    judge(schema, good=[{"items": []}, {"items": [{"name": "a", "color": "red"}], "note": None},
                        {"items": [{"name": "a", "color": "blue"}, {"name": "b", "color": "red"}], "note": "hi"}],
          bad=[{}, {"items": [{"name": "a", "color": "green"}]}, {"items": [{"name": "a"}]}])


def test_recursive_schemas():
    schema = {"$defs": {"Node": {"type": "object", "properties": {"value": {"type": "integer"}, "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
                                 "required": ["value"]}}, "$ref": "#/$defs/Node"}
    judge(schema, good=[{"value": 1}, {"value": 1, "children": [{"value": 2, "children": [{"value": 3}]}]}],
          bad=[{"children": []}, {"value": 1, "children": [{"children": []}]}])
    selfref = {"type": "object", "properties": {"next": {"$ref": "#"}}}
    judge(selfref, good=[{}, {"next": {}}, {"next": {"next": {}}}], bad=[{"next": 1}])


def test_all_of_merges_object_schemas():
    schema = {"allOf": [{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
                        {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}]}
    judge(schema, good=[{"a": 1, "b": "x"}], bad=[{"a": 1}, {"b": "x"}])


def test_an_empty_or_true_schema_is_any_json():
    for schema in ({}, True, {"description": "anything"}):
        judge(schema, good=[1, "a", None, [1, {"x": 2}], {"a": [True]}])


def test_annotations_and_unique_items_are_accepted():
    schema = {"type": "array", "items": {"type": "integer", "description": "n", "examples": [1]}, "uniqueItems": True, "title": "T", "default": []}
    judge(schema, good=[[1, 2]], bad=[["a"]])


# ------------------------------------------------------------------ whitespace


def test_whitespace_is_bounded():
    g = Grammar(schema_to_grammar({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}))
    assert g.accepts('{ "a" : 1 }') and g.accepts('{\n  "a": 1\n}')
    assert not g.accepts("{" + " " * 5 + '"a": 1}')
    assert not g.accepts('{"a": 1}' + " " * 5)


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("schema,needle", [
    ({"type": "string", "pattern": "^a+$"}, "pattern"),
    ({"type": "number", "multipleOf": 2}, "multipleOf"),
    ({"not": {"type": "string"}}, "not"),
    ({"if": {}, "then": {}}, "if"),
    ({"type": "object", "patternProperties": {"^a": {}}}, "patternProperties"),
    ({"type": "object", "minProperties": 1}, "minProperties"),
    ({"type": "array", "contains": {}}, "contains"),
    ({"type": "string", "nullable": True}, "nullable"),
    (False, "false"),
    ("string", "object or a boolean"),
    ([{"type": "string"}], "object or a boolean"),
    ({"type": "widget"}, "unknown JSON schema type"),
    ({"type": []}, "empty list"),
    ({"enum": []}, "non-empty"),
    ({"anyOf": []}, "non-empty"),
    ({"anyOf": [{"type": "string"}], "type": "string"}, "next to anyOf"),
    ({"$ref": "http://example.com/x.json"}, "local"),
    ({"$ref": "#/$defs/Missing"}, "does not resolve"),
    ({"$ref": "#"}, "only a reference to itself"),
    ({"type": "array", "maxItems": 100000}, "maxItems"),
    ({"type": "array", "minItems": 3, "maxItems": 2}, "above"),
    ({"type": "string", "maxLength": 100000}, "maxLength"),
    ({"type": "object", "required": "a"}, "required"),
    ({"type": "object", "additionalProperties": 5}, "additionalProperties"),
    ({"type": "array", "prefixItems": [{"type": "string"}], "minItems": 2}, "prefixItems"),
    ({"allOf": [{"type": "string"}, {"type": "integer"}]}, "allOf"),
])
def test_what_the_grammar_cannot_enforce_is_refused_with_a_reason(schema, needle):
    with pytest.raises(SchemaGrammarError) as exc:
        schema_to_grammar(schema)
    assert needle in str(exc.value)


def test_limits():
    with pytest.raises(SchemaGrammarError, match="bytes"):
        schema_to_grammar({"description": "x" * (MAX_SCHEMA_BYTES + 1)})
    deep = {"type": "string"}
    for _ in range(40):
        deep = {"type": "array", "items": deep}
    with pytest.raises(SchemaGrammarError, match="levels deep"):
        schema_to_grammar(deep)
    wide = {"type": "object", "properties": {f"p{i}": {"enum": [1]} for i in range(1500)}}
    with pytest.raises(SchemaGrammarError, match="grammar rules"):
        schema_to_grammar(wide)
    with pytest.raises(SchemaGrammarError, match="not valid JSON"):
        schema_to_grammar({"x": object()})


def test_a_compiled_grammar_that_is_too_large_is_caught_by_the_server_check():
    schema = {"type": "object", "properties": {f"property_number_{i}_with_a_long_name": {"type": "string", "format": "date-time"} for i in range(250)}}
    with pytest.raises(InvalidGrammarError):
        check_grammar_structure(schema_to_grammar(schema))


# ------------------------------------------------------------------ literal


def test_literal_escapes():
    assert literal('a"b\\c\nd\te\x01é') == '"a\\"b\\\\c\\nd\\te\\x01é"'
    g = Grammar('root ::= ' + literal('a"b\\c\nd\x01'))
    assert g.accepts('a"b\\c\nd\x01') and not g.accepts("a")


# ------------------------------------------------------------------ randomised agreement


def _random_value(rng: random.Random, depth: int = 0):
    kinds = ["int", "float", "str", "bool", "null"] + (["list", "dict"] if depth < 3 else [])
    kind = rng.choice(kinds)
    if kind == "int":
        return rng.randint(-10 ** 6, 10 ** 6)
    if kind == "float":
        return rng.choice([0.5, -2.25, 3.0e5, 1e-3, 123.456])
    if kind == "str":
        return "".join(rng.choice('abc "\\\n\té😀') for _ in range(rng.randint(0, 8)))
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "null":
        return None
    if kind == "list":
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {f"k{i}": _random_value(rng, depth + 1) for i in range(rng.randint(0, 3))}


def _schema_for(value):
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        return {"type": "array", "prefixItems": [_schema_for(v) for v in value]} if value \
            else {"type": "array", "maxItems": 0}
    return {"type": "object", "properties": {k: _schema_for(v) for k, v in value.items()},
            "required": list(value), "additionalProperties": False}


def test_a_schema_derived_from_a_value_accepts_that_value_and_rejects_a_changed_one():
    rng = random.Random(20261009)
    for _ in range(150):
        value = _random_value(rng)
        schema = _schema_for(value)
        grammar = Grammar(schema_to_grammar(schema))
        assert grammar.accepts(json.dumps(value, ensure_ascii=False)), (schema, value)
        assert grammar.accepts(json.dumps(value, indent=1)), (schema, value)
        if not isinstance(value, (dict, list)):
            other = "x" if not isinstance(value, str) else 0
            assert not grammar.accepts(json.dumps(other)), (schema, value)
