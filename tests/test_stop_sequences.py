# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stop sequences: the request-value validation and the text filter."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from localm.inference.protocol import ChatRequest, CompletionRequest
from localm.inference.stop_sequences import (
    MAX_STOP_LENGTH, MAX_STOP_SEQUENCES, StopFilter, apply_stop, normalize_stop,
)

# ------------------------------------------------------------------ normalize_stop


def test_normalize_accepts_one_string_or_a_list_and_drops_empties():
    assert normalize_stop(None) is None
    assert normalize_stop("END") == ["END"]
    assert normalize_stop(["a", "", "b"]) == ["a", "b"]
    assert normalize_stop(("a",)) == ["a"]


@pytest.mark.parametrize("value", ["", [], [""], ["", ""]])
def test_normalize_turns_an_empty_value_into_no_stop(value):
    assert normalize_stop(value) is None


@pytest.mark.parametrize("value", [1, 1.5, True, {"a": 1}, [1, 2], ["a", 2], [None]])
def test_normalize_rejects_non_strings(value):
    with pytest.raises(ValueError):
        normalize_stop(value)


def test_normalize_enforces_the_limits():
    assert normalize_stop(["x"] * MAX_STOP_SEQUENCES) == ["x"] * MAX_STOP_SEQUENCES
    with pytest.raises(ValueError):
        normalize_stop(["x"] * (MAX_STOP_SEQUENCES + 1))
    assert normalize_stop("y" * MAX_STOP_LENGTH)
    with pytest.raises(ValueError):
        normalize_stop("y" * (MAX_STOP_LENGTH + 1))


@pytest.mark.parametrize("model", [ChatRequest, CompletionRequest])
def test_requests_normalise_stop_and_refuse_a_bad_one(model):
    body = {"messages": [{"role": "user", "content": "hi"}]} if model is ChatRequest \
        else {"prompt": "hi"}
    assert model(**body, stop="END").stop == ["END"]
    assert model(**body, stop=["a", "b"]).stop == ["a", "b"]
    assert model(**body).stop is None
    with pytest.raises(ValidationError):
        model(**body, stop=5)
    with pytest.raises(ValidationError):
        model(**body, stop=["x"] * (MAX_STOP_SEQUENCES + 1))


# ------------------------------------------------------------------ StopFilter


def test_filter_cuts_at_the_first_stop_and_emits_nothing_after():
    flt = StopFilter(["END"])
    assert flt.feed("hello ") == "hello "
    assert flt.feed("worldEND and more") == "world"
    assert flt.hit is True
    assert flt.feed("still more") == ""
    assert flt.flush() == ""


def test_filter_finds_a_stop_split_across_chunks():
    flt = StopFilter(["</s>"])
    pieces = [flt.feed(p) for p in ("ab<", "/", "s", ">cd")]
    assert "".join(pieces) == "ab"
    assert flt.hit is True


def test_filter_releases_a_held_tail_that_never_completed():
    flt = StopFilter(["</s>"])
    assert flt.feed("ab<") == "ab"
    assert flt.hit is False
    assert flt.flush() == "<"


def test_filter_picks_the_earliest_of_several_stops():
    flt = StopFilter(["two", "one"])
    assert flt.feed("zero one two") == "zero "
    assert flt.hit is True


def test_filter_without_stops_is_a_passthrough():
    flt = StopFilter([])
    assert flt.feed("anything") == "anything"
    assert flt.flush() == ""
    assert flt.hit is False


def test_filter_holds_the_longest_possible_prefix_across_several_stops():
    flt = StopFilter(["abcd", "xy"])
    assert flt.feed("12abc") == "12"
    assert flt.feed("x") == "abc"
    assert flt.flush() == "x"


def test_apply_stop():
    assert apply_stop("Hello world", ["wor"]) == ("Hello ", True)
    assert apply_stop("Hello world", ["xyz"]) == ("Hello world", False)
    assert apply_stop("ends with <", ["</s>"]) == ("ends with <", False)
    assert apply_stop("", ["a"]) == ("", False)
