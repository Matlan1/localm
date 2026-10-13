# SPDX-License-Identifier: AGPL-3.0-or-later
"""The JSON-schema-to-grammar compiler bounds its OUTPUT while it builds it, so a
schema inside every input cap cannot make it allocate a grammar far larger than
check_grammar_structure would ever accept."""

from __future__ import annotations

import json
import time
import tracemalloc

import pytest

from localm.inference import gbnf
from localm.inference import json_schema_grammar as jsg


def _bomb_schema(levels: int = 20, name_len: int = 3000) -> dict:
    node: dict = {"type": "integer", "minimum": -999999999999999,
                  "maximum": 999999999999999}
    for i in range(levels):
        node = {"type": "object", "properties": {f"p{i}" + "x" * name_len: node}}
    return node


def _wide_schema(n_props: int, name_len: int) -> dict:
    return {"type": "object",
            "properties": {f"p{i:04d}" + "x" * name_len: {"type": "string"}
                           for i in range(n_props)},
            "required": [f"p{i:04d}" + "x" * name_len for i in range(n_props)]}


def test_bomb_schema_is_refused_with_bounded_allocation():
    schema = _bomb_schema()
    assert len(json.dumps(schema)) <= jsg.MAX_SCHEMA_BYTES
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        with pytest.raises(jsg.SchemaGrammarError, match="byte") as ei:
            jsg.schema_to_grammar(schema)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert ei.value.keyword is None
    assert peak <= 8 * 1024 * 1024, f"peak {peak / 1e6:.1f} MB"
    assert time.perf_counter() - t0 < 10


def test_loosen_schema_stops_at_the_size_refusal(monkeypatch):
    schema = _bomb_schema(levels=10, name_len=3000)
    calls = {"n": 0}
    real = jsg.schema_to_grammar

    def counting(s):
        calls["n"] += 1
        return real(s)

    monkeypatch.setattr(jsg, "schema_to_grammar", counting)
    loosened, dropped, error = jsg.loosen_schema(schema)
    assert loosened is None and dropped == []
    assert isinstance(error, jsg.SchemaGrammarError)
    assert calls["n"] == 1


@pytest.mark.parametrize("n_props,name_len", [
    (10, 10), (100, 40), (200, 60), (300, 60), (400, 60), (600, 60), (1000, 60),
])
def test_budget_refuses_only_grammars_the_structure_check_would_refuse(
        monkeypatch, n_props, name_len):
    schema = _wide_schema(n_props, name_len)
    if len(json.dumps(schema)) > jsg.MAX_SCHEMA_BYTES:
        pytest.skip("over the input cap")
    monkeypatch.setattr(jsg, "MAX_GRAMMAR_BYTES", 10 ** 9)
    unbounded = jsg.schema_to_grammar(schema)
    monkeypatch.undo()
    try:
        bounded = jsg.schema_to_grammar(schema)
    except jsg.SchemaGrammarError:
        assert len(unbounded) > gbnf.MAX_GRAMMAR_BYTES
    else:
        assert bounded == unbounded
        gbnf.check_grammar_structure(bounded)


def test_ladder_crosses_the_limit():
    sizes = []
    for n in (300, 400, 600, 1000):
        schema = _wide_schema(n, 60)
        try:
            sizes.append(len(jsg.schema_to_grammar(schema)))
        except jsg.SchemaGrammarError:
            sizes.append(None)
    assert sizes[0] is not None and sizes[0] <= gbnf.MAX_GRAMMAR_BYTES
    assert sizes[-1] is None
