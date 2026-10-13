# SPDX-License-Identifier: AGPL-3.0-or-later
"""validate_trigger_patterns bounds the WHOLE list, not only each pattern: a
list longer than MAX_TRIGGER_PATTERNS is refused before any probe, and a list
whose validation runs past its wall-clock budget stops probing."""

from __future__ import annotations

import os
import threading
from unittest.mock import MagicMock

import pytest

from localm.inference import gbnf
from localm.inference.backends.base import (
    InvalidGrammarError,
    TriggerValidatorUnavailableError,
)
from localm.inference.tool_calling import MAX_TOOLS


@pytest.fixture
def probe_calls(monkeypatch):
    monkeypatch.setattr(gbnf, "_VALIDATED_TRIGGER_PATTERNS", {})
    calls: list[str] = []

    def fake_probe(pattern):
        calls.append(pattern)
        return gbnf._PROBE_SAFE, ""

    monkeypatch.setattr(gbnf, "_probe_pattern_is_safe", fake_probe)
    return calls


def _patterns(n: int, tag: str = "t") -> list[str]:
    return [f"(<{tag}{i}>)" for i in range(n)]


def test_one_trigger_per_tool_always_fits():
    assert gbnf.MAX_TRIGGER_PATTERNS == MAX_TOOLS


def test_list_at_the_cap_is_validated(probe_calls):
    gbnf.validate_trigger_patterns(_patterns(gbnf.MAX_TRIGGER_PATTERNS))
    assert len(probe_calls) == gbnf.MAX_TRIGGER_PATTERNS


@pytest.mark.parametrize("n", [gbnf.MAX_TRIGGER_PATTERNS + 1, 100_000])
def test_list_over_the_cap_is_refused_before_any_probe(probe_calls, n):
    with pytest.raises(InvalidGrammarError, match="over the limit") as ei:
        gbnf.validate_trigger_patterns(_patterns(n))
    assert not isinstance(ei.value, TriggerValidatorUnavailableError)
    assert probe_calls == []


def test_list_budget_stops_probing(monkeypatch):
    monkeypatch.setattr(gbnf, "_VALIDATED_TRIGGER_PATTERNS", {})
    monkeypatch.setattr(gbnf, "_TRIGGER_LIST_BUDGET_SECONDS", 0.05)
    calls: list[str] = []

    def slow_probe(pattern):
        calls.append(pattern)
        threading.Event().wait(0.1)
        return gbnf._PROBE_SAFE, ""

    monkeypatch.setattr(gbnf, "_probe_pattern_is_safe", slow_probe)
    with pytest.raises(TriggerValidatorUnavailableError, match="took more than"):
        gbnf.validate_trigger_patterns(_patterns(10))
    assert len(calls) == 1
    assert len(gbnf._VALIDATED_TRIGGER_PATTERNS) == 1


def test_cached_patterns_are_not_charged_against_the_budget(monkeypatch, probe_calls):
    patterns = _patterns(5)
    gbnf.validate_trigger_patterns(patterns)
    monkeypatch.setattr(gbnf, "_TRIGGER_LIST_BUDGET_SECONDS", -1.0)
    gbnf.validate_trigger_patterns(patterns)
    assert len(probe_calls) == 5


def _engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.chat_stream.side_effect = AssertionError("generation must not start")
    return engine


def test_route_answers_400_for_an_oversized_trigger_list(probe_calls):
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    os.environ.pop("LOCALM_API_KEY", None)
    engine = _engine()
    client = TestClient(create_app(engine), raise_server_exceptions=True)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "grammar": 'root ::= "x"',
        "grammar_lazy": True,
        "grammar_triggers": _patterns(gbnf.MAX_TRIGGER_PATTERNS + 1),
        "max_tokens": 4,
    })
    assert r.status_code == 400, (r.status_code, r.text)
    assert "over the limit" in r.text
    assert probe_calls == []
    engine.chat_stream.assert_not_called()
