# SPDX-License-Identifier: AGPL-3.0-or-later
"""Superlinear-backtracking bounds for the sites found by the repo-wide sweep."""

from __future__ import annotations

import re
import time

import pytest

from localm._log_digest import _LOG_LINE_RE
from localm.plugins.builtin.jobs.webtool import (
    _strip_think, _top_level_objects, parse_web_call, parse_web_calls)
from localm.plugins.coder.tools.files import (
    _WS_FALLBACK_MAX_TEXT, _WS_FALLBACK_MAX_TOKENS, _resolve_edit)

BUDGET = 0.5


def _timed(fn, *a, **kw):
    start = time.perf_counter()
    result = fn(*a, **kw)
    return result, time.perf_counter() - start


def _timed_cpu(fn, *a, **kw):
    """Like _timed, but measures CPU time (time.process_time), not wall-clock."""
    start = time.process_time()
    result = fn(*a, **kw)
    return result, time.process_time() - start


# ---------------------------------------------------------------------------
#  jobs/webtool.py - the second copy of the coder's tool-call patterns
# ---------------------------------------------------------------------------

# The witnesses are built by a FACTORY and the params are named. Parametrizing
# over the literal 50,000-character strings puts them in the pytest node id,
# which pytest then exports as PYTEST_CURRENT_TEST - and Windows refuses an
# environment variable over 32,767 characters, so every case ERRORS in teardown
# with its assertions never evaluated. Caught by CI on both legs after I had
# already fixed the identical thing in tests/test_redos_bounds.py and then
# reintroduced it here.
@pytest.mark.parametrize("name, make_witness", [
    # _THINK_RE: openers that never close. Pre-fix 1.22s.
    ("think_openers", lambda: "<r >" * 8_000),
    # _WRAP_RE: the adjacent-quantifier cubic. Pre-fix 30.0s at 4,000.
    ("wrap_spaces", lambda: "<tool_call>" + " " * 4_000),
    # _top_level_objects: unmatched braces, one full scan each. Pre-fix 2.84s.
    ("unmatched_braces", lambda: "{" * 16_000),
    # _FENCE_RE: the two adjacent [ \t]* quantifiers.
    ("fence_tabs", lambda: "```" + "\t" * 50_000),
    # Wrapper markers with no closer anywhere.
    ("wrap_openers", lambda: "<|tool_call>" * 5_000),
])
def test_parse_web_call_is_bounded_on_hostile_model_output(name, make_witness):
    """This runs on RAW MODEL OUTPUT in the jobs plugin, so every quantifier is reachable by whatever a poisoned page persuaded the model to emit."""
    _, elapsed = _timed(parse_web_call, make_witness())
    assert elapsed < BUDGET, f"parse_web_call took {elapsed:.2f}s on {name}"


@pytest.mark.parametrize("name, make_witness", [
    # The enumeration-level witnesses, re-run with a real call in FRONT of them.
    ("think_openers", lambda: "<r >" * 8_000),
    ("wrap_spaces", lambda: "<tool_call>" + " " * 4_000),
    ("unmatched_braces", lambda: "{" * 16_000),
    ("fence_tabs", lambda: "```" + "\t" * 50_000),
    ("wrap_openers", lambda: "<|tool_call>" * 5_000),
    # The one that is genuinely NEW, and the reason this test exists: 2,000
    # well-formed but unparseable wrapped bodies AFTER the real call. limit=1
    # never looks at them; limit=2 runs _lenient_json (4 regex passes) over
    # every one. MEASURED on this box at limit=2: 0.022s / 0.041s / 0.095s /
    # 0.248s at n = 500 / 1,000 / 2,000 / 4,000, i.e. ~linear (8x input, 11x
    # time) rather than the quadratic shape this file exists to catch. n=2,000
    # sits ~5x under BUDGET, which survives a loaded box without going soft.
    ("junk_bodies_after_a_real_call",
     lambda: ("<tool_call>" + "'a" * 200 + "</tool_call>") * 2_000),
])
def test_parse_web_calls_is_bounded_when_a_real_call_precedes_the_hostile_tail(
        name, make_witness):
    """``parse_web_call`` stops at the FIRST call, so on input that contains one the scan ends before any hostile tail is examined - the limit=1 bound above cannot speak for what follows a valid call."""
    valid = '<tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call>'
    calls, elapsed = _timed(parse_web_calls, valid + make_witness(), 2)
    assert elapsed < BUDGET, f"parse_web_calls took {elapsed:.2f}s on {name}"
    assert calls and calls[0]["name"] == "web_search", \
        "the bound must not have cost us the real call that precedes the junk"


@pytest.mark.parametrize("text, expected", [
    ("a<think>hidden</think>b", "ab"),
    ("x<reasoning>why</reasoning>y", "xy"),
    ("p<r>short</r>q", "pq"),
    ("<THINK>upper</THINK>kept", "kept"),          # IGNORECASE preserved
    ("<think a='1'>attrs</think>z", "z"),          # attributes preserved
    ("a<think>never closed", "a<think>never closed"),   # unterminated: left alone
    ("no markup at all", "no markup at all"),
])
def test_strip_think_still_strips_what_it_used_to(text, expected):
    assert _strip_think(text) == expected


@pytest.mark.parametrize("text, expected_name", [
    ('<|tool_call>call:web_search{"query": "x"}<tool_call|>', "web_search"),
    ('<|tool_call>{"name": "web_search", "args": {"query": "x"}}<|tool_call>', "web_search"),
    ('```json\n{"name": "fetch_url", "args": {"url": "http://x"}}\n```', "fetch_url"),
    ('{"name": "fetch_url", "args": {"url": "http://x"}}', "fetch_url"),
    ('<think>reasoning</think>{"name": "web_search", "args": {"query": "q"}}', "web_search"),
])
def test_real_web_calls_still_parse(text, expected_name):
    call = parse_web_call(text)
    assert call is not None, f"lost the call in {text!r}"
    assert call["name"] == expected_name


def test_top_level_objects_still_finds_real_objects():
    """The brace scanner's bound must not cost it a real object."""
    text = 'prose {"a": 1} more {"b": {"c": 2}} tail'
    assert list(_top_level_objects(text)) == ['{"a": 1}', '{"b": {"c": 2}}']
    # A trailing unmatched opener must not suppress the objects before it.
    assert list(_top_level_objects('{"a": 1} then {')) == ['{"a": 1}']


def test_top_level_objects_stays_linear_when_many_opens_never_balance():
    """The gap ``unmatched_braces`` above does NOT cover."""
    hostile = ("{" * 8_000) + "}"
    result, cpu_elapsed = _timed_cpu(parse_web_call, hostile)
    assert cpu_elapsed < 2.0, (
        f"parse_web_call used {cpu_elapsed:.2f}s of CPU time on 8,000 "
        "unbalanced opens plus one stray closing brace - the per-position "
        "balance rescan is quadratic again")
    assert result is None


# ---------------------------------------------------------------------------
#  rag/extract.py - reached by a crafted .docx, not by model output
# ---------------------------------------------------------------------------

_DOCX_ATTR_SUB = re.compile(r"<w:(?:tab|br|cr)\b[^<>]*/?>")
_DOCX_RUN_FIND = re.compile(r"<w:t\b[^<>]*>([^<]*)</w:t>")


@pytest.mark.parametrize("label, pattern, make_witness", [
    ("attr_sub_unclosed_br", _DOCX_ATTR_SUB, lambda: "<w:br" * 20_000),
    ("run_find_unclosed_t", _DOCX_RUN_FIND, lambda: "<w:t" * 20_000),
])
def test_docx_tag_scans_are_bounded(label, pattern, make_witness):
    """Pre-fix 16.21s and 1.69s."""
    witness = make_witness()
    _, elapsed = _timed(lambda: pattern.findall(witness))
    assert elapsed < BUDGET, f"{label} took {elapsed:.2f}s"


def test_well_formed_docx_xml_is_parsed_identically():
    """The bound must not narrow what a real document matches - attribute values with namespaces and spaces are ordinary in .docx."""
    xml = ('<w:p><w:r><w:t xml:space="preserve">Hello </w:t></w:r>'
           '<w:r><w:tab/><w:t>World</w:t></w:r></w:p>')
    assert _DOCX_ATTR_SUB.sub("\t", xml) == (
        '<w:p><w:r><w:t xml:space="preserve">Hello </w:t></w:r>'
        '<w:r>\t<w:t>World</w:t></w:r></w:p>')
    assert _DOCX_RUN_FIND.findall(xml) == ["Hello ", "World"]


# ---------------------------------------------------------------------------
#  _log_digest.py - bug-report and diagnostics parsing
# ---------------------------------------------------------------------------

def test_log_line_pattern_is_bounded_on_a_long_space_run():
    """``\\s+([^:]+)`` let both quantifiers claim the same run, because whitespace is a SUBSET of ``[^:]``."""
    line = "2026-01-01 00:00:00,000 X" + " " * 8_000 + "y" * 8_000
    _, elapsed = _timed(_LOG_LINE_RE.match, line)
    assert elapsed < BUDGET, f"log-line match took {elapsed:.2f}s"


@pytest.mark.parametrize("line, groups", [
    ("2026-07-29 06:12:03,123 INFO localm.server: started on 127.0.0.1",
     ("INFO", "localm.server", "started on 127.0.0.1")),
    # A message containing further colons must still be captured whole.
    ("2026-07-29 06:12:03,123 ERROR localm.rag.store: failed: nested: colons",
     ("ERROR", "localm.rag.store", "failed: nested: colons")),
    # A logger name containing a space is unusual but was accepted before.
    ("2026-07-29 06:12:03,123 WARNING my logger: message",
     ("WARNING", "my logger", "message")),
])
def test_real_log_lines_still_parse(line, groups):
    match = _LOG_LINE_RE.match(line)
    assert match is not None, f"lost a real log line: {line!r}"
    assert match.groups() == groups


# ---------------------------------------------------------------------------
#  files.py - the whitespace-tolerant edit fallback
# ---------------------------------------------------------------------------

def test_edit_fallback_caps_are_generous_enough_to_be_invisible():
    """A bound that excludes ordinary work is a broken feature, not a fix. 400 tokens is roughly a 40-line snippet and 512 KB is far past a single edit."""
    assert _WS_FALLBACK_MAX_TOKENS >= 200
    assert _WS_FALLBACK_MAX_TEXT >= 256 * 1024


def test_exact_edit_match_is_unaffected_by_the_cap():
    """The cap guards only the tolerant FALLBACK."""
    text = "x" * (_WS_FALLBACK_MAX_TEXT + 10_000) + "\nNEEDLE\n"
    result = _resolve_edit(text, "NEEDLE")
    assert result is not None, "the exact-match path must not be capped"
    start, end, count, tolerant = result
    assert text[start:end] == "NEEDLE"
    assert tolerant is False


def test_tolerant_edit_fallback_still_lands_below_the_cap():
    """The behaviour the cap must not break: a snippet the model reconstructed with different whitespace still matches, and is reported as tolerant."""
    text = "def f():\n    return    1\n"
    result = _resolve_edit(text, "return 1")
    assert result is not None, "the tolerant fallback stopped working"
    start, end, count, tolerant = result
    assert text[start:end] == "return    1"
    assert tolerant is True


def test_tolerant_edit_fallback_declines_a_pathological_input_quickly():
    """The defect: O(len(text) x len(pattern)) with BOTH sides model-supplied."""
    text = "a " * 100_000                      # 200 KB
    old = "a " * (_WS_FALLBACK_MAX_TOKENS + 50) + "b"   # over the token cap, no match
    result, elapsed = _timed(_resolve_edit, text, old)
    assert result is None, "an over-cap tolerant search must be declined"
    assert elapsed < BUDGET, f"declining still took {elapsed:.2f}s"
