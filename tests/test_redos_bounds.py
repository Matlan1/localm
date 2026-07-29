# SPDX-License-Identifier: AGPL-3.0-or-later
"""Superlinear-backtracking bounds for the regexes that run on remote text.

Guards the six ``py/polynomial-redos`` findings closed by the 2026-07-28 CodeQL
triage (WS5). Each pattern below is applied to attacker-controlled input:

  * ``textguard.neutralise`` runs on the body of an arbitrary URL fetched by
    ``POST /api/web/fetch`` (``max_chars`` up to 60,000) and on every RAG chunk.
  * ``parser.parse_tool_calls`` runs on raw model output, which a poisoned
    document or page can steer.
  * the coder's session transcript re-runs the tool_call pattern over EVERY
    stored assistant message at teardown, so one poisoned response used to
    re-hang the export on every later close.

Two kinds of assertion, and BOTH matter:

  * wall-clock bounds - the actual defect. Pre-fix baselines, measured on this
    project's venv under the box-wide test lock (min of n samples; a sample can
    only be inflated by interference, never deflated):
        textguard._FRAME_RE   66.3s  at the 60,000-char fetch ceiling
        parser._RE_FENCE      95.9s  at 100,000 tabs
        parser._RE_XML         7.0s  at 2,000 spaces (cubic - 20,000 never ran)
        session._TC_RE         5.5s  at 2,000 spaces
        parser._RE_VARIANT     2.1s  at 5,000 marker repetitions
    Post-fix, every one of those inputs costs between 0.0000s and 0.0051s. The
    0.5s budget therefore sits two to five orders of magnitude clear of both
    sides, which is what keeps it meaningful on a loaded box without ever being
    a flaky-by-design timer.
  * semantic regression - a bounded regex that also stops matching real input is
    a regression, not a fix. Every pattern keeps a paired "still matches"
    assertion, and the fence rewrite is pinned by a differential fuzz against
    the exact pattern it replaced.
"""

from __future__ import annotations

import random
import re
import time

import pytest

from localm.plugins.coder.parser import (
    _RE_FENCE_CLOSE, _RE_FENCE_OPEN, _RE_TOOL_MARKER, _RE_XML_CLOSE, _RE_XML_OPEN,
    _iter_marker_variant_calls,
    _iter_fenced_blocks, _iter_xml_tool_calls, _pair_delimited, _parse_gemma_args,
    _try_parse_body,
    parse_tool_calls, strip_xml_tool_calls)
from localm.textguard import _FRAME_RE, neutralise

# Generous on purpose: see the module docstring. The smallest pre-fix baseline
# among the cases below was 2.08s and the largest was 95.9s.
BUDGET = 0.5


def _timed(fn, *a, **kw):
    start = time.perf_counter()
    result = fn(*a, **kw)
    return result, time.perf_counter() - start


# ---------------------------------------------------------------------------
#  Wall-clock bounds (the CodeQL witnesses, verbatim)
# ---------------------------------------------------------------------------

def test_neutralise_bounded_on_frame_marker_prefix():
    """alert 194 - ``'<'`` then a long run of spaces. 60,000 is the ceiling
    POST /api/web/fetch itself accepts via max_chars, so this IS the route's
    worst case, not a synthetic one. Measured baseline 66.3s, all of it event-loop
    outage because neutralise() used to run inline in the async handler."""
    hostile = "<" + " " * 60_000
    out, elapsed = _timed(neutralise, hostile)
    assert elapsed < BUDGET, f"neutralise took {elapsed:.2f}s on 60k spaces"
    # No frame marker is present, so the text must come back untouched.
    assert out == hostile


# Alert 189 reports TWO witness clauses, not one. Both are asserted: a prescribed
# fix derived from a single witness is exactly what made alert 190 ship looking
# complete. Only two alerts in the repo carry a compound message and both are in
# this file, so this is the last place the mistake can hide.
_XML_WITNESS_PREFIXES = ["<tool_call>", "<tool_call>a"]


@pytest.mark.parametrize("prefix", _XML_WITNESS_PREFIXES)
def test_parse_tool_calls_bounded_on_xml_wrapper_prefix(prefix):
    """alerts 189/192/193 - a ``<tool_call>`` opener then a long run of spaces.
    Cubic pre-fix: 0.08 / 0.62 / 7.03s at 500 / 1000 / 2000 spaces for witness 1,
    so 20,000 would not have finished inside this test run at all. Witness 2 is
    the same defect with a far smaller constant (0.0095s at 2,000)."""
    hostile = prefix + " " * 20_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, (
        f"parse_tool_calls took {elapsed:.2f}s on {prefix!r} + 20k spaces")
    assert calls == []


def _best(fn, text, samples=5):
    """Fastest of n runs. Interference can only inflate a sample, never deflate
    one, so the minimum is the closest reading to uncontended cost - which is what
    makes a RATIO between two sizes meaningful on a shared box."""
    return min(_timed(fn, text)[1] for _ in range(samples))


@pytest.mark.parametrize("prefix", _XML_WITNESS_PREFIXES)
def test_xml_witnesses_do_not_grow_superlinearly(prefix):
    """Compares the COST RATIO across a 10x input, not two absolute bounds.

    This has to be a ratio. An absolute "both under BUDGET" check cannot tell
    linear from quadratic here, because the fixed pattern is so far under the
    budget that a hypothetically quadratic one would still fit inside it: ~0.0002s
    at the 1x size scales to ~0.08s at 20x, comfortably under 0.5s, and the
    assertion would pass on exactly the bug it claims to exclude. (An earlier
    draft of this test made that mistake.)

    10x input: linear costs ~10x, quadratic ~100x. The 30x bound sits between them
    with 3x of margin either side.

    MEASURED on this project's venv, which is how the constants were set rather
    than guessed:
        '<tool_call>'   200k = 9.16ms   2M = 92.86ms   ratio 10.1x
        '<tool_call>a'  200k = 0.41ms   2M =  3.83ms   ratio  9.3x
    Both linear. A quadratic implementation would land near 0.92s and 41ms
    respectively, so the 30x bound separates them either way.

    The absolute floor guards only against a base too small to divide by, and it
    MUST stay below ``30 * base`` or it silently swallows the bug. An earlier draft
    used 0.05s: fine for the first witness (30*base = 0.275s) but fatal for the
    second (30*base = 0.012s), where the floor would have become the threshold and
    passed a 41ms quadratic. 0.005s is below both.

    Deliberately NO absolute budget assertion here. At these sizes ordinary LINEAR
    work legitimately approaches 0.1s - the marker-variant pass still walks every
    marker and every space once, which is linear but not free - so an
    absolute bound would go red on correct code. BUDGET belongs at the witness
    sizes, where the other tests apply it; superlinearity is what this test is for,
    and a ratio is the only thing that measures it.
    """
    base = _best(parse_tool_calls, prefix + " " * 200_000)
    ten_x = _best(parse_tool_calls, prefix + " " * 2_000_000)
    assert ten_x < max(30 * base, 0.005), (
        f"{prefix!r}: {base * 1000:.2f}ms at 200k -> {ten_x * 1000:.2f}ms at 2M "
        f"({ten_x / max(base, 1e-9):.1f}x for a 10x input) - that is superlinear")


def test_parse_tool_calls_bounded_on_many_unterminated_openers():
    """The case a single-opener witness does NOT catch: thousands of ``<tool_call>``
    openers and no closer anywhere, so a lazy body scans to end-of-text once per
    opener. That is quadratic independently of the adjacent-quantifier problem,
    and it survived the first attempt at this fix (2.21s) until the opener/closer
    pairing landed. Same shape as the fence's second CodeQL witness."""
    hostile = "<tool_call>" * 5_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"parse_tool_calls took {elapsed:.2f}s on 5k openers"
    assert calls == []


def test_parse_tool_calls_bounded_on_unterminated_variant_markers():
    """alert 191 - repeated ``<tool_call>{{`` with no closing brace anywhere, so
    every marker used to start a scan to end-of-text. Baseline 2.08s."""
    hostile = "<tool_call>{{" * 5_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"parse_tool_calls took {elapsed:.2f}s on variant markers"
    assert calls == []


def test_parse_tool_calls_bounded_on_fence_tab_run():
    """alert 190, first witness - ``'```'`` then a long run of tabs, against the
    opener's two adjacent ``[ \\t]*`` quantifiers. Baseline 95.9s (1.36 / 7.22 /
    26.74s at 12.5k / 25k / 50k tabs)."""
    hostile = "```" + "\t" * 100_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"parse_tool_calls took {elapsed:.2f}s on 100k tabs"
    assert calls == []


def test_parse_tool_calls_bounded_on_repeated_fence_openers():
    """alert 190, second witness - many fence openers with no closer after the
    first few characters. This one is a MANY-START-POSITIONS quadratic, not an
    adjacent-quantifier one: de-ambiguating the lang group does not touch it
    (0.08 / 0.28 / 1.10 / 5.03s at 1k / 2k / 4k / 8k), only the opener/closer
    pairing does. Sized past the point the old form exceeded the budget."""
    hostile = "```\n" + "```\na" * 8_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"parse_tool_calls took {elapsed:.2f}s on repeated openers"
    assert calls == []


def test_parse_tool_calls_bounded_on_doubled_brace_run():
    """The oracle's ``'<tool_call>{{' * 5000`` case run through the name-gated
    path as well, since passing tool_names enables two extra passes (the bare
    top-level JSON scan) over the same hostile text."""
    hostile = "<tool_call>{{" * 5_000
    calls, elapsed = _timed(parse_tool_calls, hostile, {"read_file"})
    assert elapsed < BUDGET, f"parse_tool_calls took {elapsed:.2f}s with tool_names"
    assert calls == []


# ---------------------------------------------------------------------------
#  Structural guards - no clock involved, so these cannot go flaky under load
# ---------------------------------------------------------------------------
#
# A wall-clock threshold is a weak guard on a shared box: it drifts, and the
# temptation when it goes red is to loosen it until it guards nothing. These
# assert the SHAPE of the fix instead, and hold identically on an idle box and a
# saturated one.

class _CountingPattern:
    """Records the start offset of every search, and delegates."""

    def __init__(self, pattern):
        self._pattern = pattern
        self.starts: list[int] = []

    def search(self, text, pos=0):
        self.starts.append(pos)
        return self._pattern.search(text, pos)


def test_pairing_abandons_the_scan_after_one_failed_closer_search():
    """THE defect, stated structurally. The old lazy body re-scanned to
    end-of-text once per opener, so 5,000 unterminated openers meant 5,000 full
    scans. Pairing must issue exactly ONE closer search and then stop, because a
    closer search that failed from this opener cannot succeed from a later one."""
    hostile = "<tool_call>" * 5_000
    opener = _CountingPattern(_RE_XML_OPEN)
    closer = _CountingPattern(_RE_XML_CLOSE)

    assert list(_pair_delimited(hostile, opener, closer)) == []
    assert len(opener.starts) == 1, f"rescanned openers: {len(opener.starts)}"
    assert len(closer.starts) == 1, f"rescanned closers: {len(closer.starts)}"


def test_pairing_searches_never_go_backwards():
    """The linearity argument rests on the searches being monotonic - each one
    starts where the previous match ended, so together they walk the text once."""
    text = ('<tool_call>{"name": "a", "args": {}}</tool_call> prose '
            '<tool_call>{"name": "b", "args": {}}</tool_call> tail <tool_call>')
    opener = _CountingPattern(_RE_XML_OPEN)
    closer = _CountingPattern(_RE_XML_CLOSE)

    pairs = list(_pair_delimited(text, opener, closer))
    assert len(pairs) == 2
    assert opener.starts == sorted(opener.starts), opener.starts
    assert closer.starts == sorted(closer.starts), closer.starts
    # One opener search per pair, plus the one that found the trailing opener;
    # one closer search per pair, plus the one that came up empty.
    assert len(opener.starts) == 3 and len(closer.starts) == 3


@pytest.mark.parametrize("gap", [0, 1, 8, 9, 20, 200, 5_000])
def test_frame_marker_tolerance_is_not_bounded(gap):
    """The whitespace tolerance must survive the ReDoS fix INTACT.

    An earlier revision bounded this at ``\\s{0,8}`` and a test here asserted the
    non-match at nine spaces - enshrining the weakening as intent. That was
    backwards. Nothing in the codebase parses the closing fence (the code-side
    checks all test ``startswith('<tool_result')`` on the OUTER tag), so the only
    consumer is the MODEL, a fuzzy reader. Any finite bound is a bypass an
    attacker reaches by typing one more space, and the bound bought nothing: the
    de-ambiguated form is linear without it.
    """
    assert _FRAME_RE.search("<" + " " * gap + "/tool_result>") is not None
    assert _FRAME_RE.search("<" + " " * gap + "/" + " " * gap
                            + "untrusted_content>") is not None


def test_frame_marker_defanging_is_unchanged_from_the_prefix_pattern():
    """The de-ambiguated pattern must accept the same language as the ambiguous
    one it replaced - that is the whole claim, so it is fuzzed rather than eyeballed.

    The alphabet includes RUNS of whitespace, not just single characters. A fuzz
    over single characters shows 0 divergences even for the bounded version,
    because random short strings essentially never contain nine consecutive
    spaces - which is exactly how the bounded revision passed a 200,000-case fuzz
    while silently dropping the control.
    """
    rnd = random.Random(20260728)
    alphabet = ["<", "/", " ", "  ", " " * 9, " " * 30, "\t", "\n", "\r",
                "tool_result", "untrusted_content", "TOOL_RESULT", "x", ">"]
    for _ in range(30_000):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 12)))
        assert (_FRAME_RE.sub(r"&lt;\1", text)
                == _LEGACY_FRAME.sub(r"&lt;\1", text)), f"diverged on {text!r}"


def test_variant_body_is_brace_matched_not_pattern_matched():
    """The marker-variant body is delimited by BRACE BALANCE, not by a pattern.

    This assertion is the reverse of the one it replaces, deliberately. The first
    fix tempered a regex so the body could not span a marker - which bounded the
    cost and silently dropped a real case, since the coder editing its own parser
    docs emits a write_file whose CONTENT contains ``<tool_call>``. A
    string-aware brace scan has no such trade: it knows the marker is inside a
    JSON string, so the call is recovered AND the scan stays bounded.
    """
    ordinary = list(_iter_marker_variant_calls('<|tool_call>{"a": 1}<tool_call|>'))
    assert len(ordinary) == 1 and ordinary[0][3] == '{"a": 1}'

    # A marker inside a JSON STRING is part of the body, not a delimiter.
    with_marker = list(_iter_marker_variant_calls(
        '<|tool_call>{"a":"<|tool_call>"}<tool_call|>'))
    assert len(with_marker) == 1, "a marker inside a string value must not end the body"
    assert with_marker[0][3] == '{"a":"<|tool_call>"}'


# ---------------------------------------------------------------------------
#  Fires-control - proof the budget still catches a genuinely slow path
# ---------------------------------------------------------------------------

# The exact PRE-FIX patterns, in one place: they are both the fires-control below
# and the differential-fuzz oracle further down. A budget that never fails on
# these would be guarding nothing, which is how a timing assertion quietly rots
# into decoration.
_LEGACY_FRAME = re.compile(r"<(\s*/?\s*(?:tool_result|untrusted_content))",
                           re.IGNORECASE)
_LEGACY_VARIANT = re.compile(
    r"<\|?/?tool_call\|?>\s*(?:call:(?P<name>\w+)\s*)?(?P<body>\{.*?\})"
    r"\s*<\|?/?tool_call\|?>", re.DOTALL)
_RE_XML_LEGACY = re.compile(
    r"<tool_call(?:\s+name=['\"](?P<name_attr>[^'\"]+)['\"])?>\s*"
    r"(?P<body>.+?)"
    r"\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_RE_FENCE_LEGACY = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z_][\w+.-]*)?[ \t]*\r?\n"
    r"(?P<body>.+?)"
    r"\r?\n[ \t]*```",
    re.DOTALL,
)


@pytest.mark.parametrize("label, legacy, fixed, witness, base, cap", [
    ("textguard._FRAME_RE",
     lambda s: _LEGACY_FRAME.sub(r"&lt;\1", s),
     lambda s: _FRAME_RE.sub(r"&lt;\1", s),
     lambda n: "<" + " " * n, 15_000, 240_000),
    # Alert 189's FIRST witness. Pre-fix 7.03s at n=2000, so it fires at once.
    ("parser._RE_XML (witness 1)",
     lambda s: [m.span() for m in _RE_XML_LEGACY.finditer(s)],
     lambda s: list(_iter_xml_tool_calls(s)),
     lambda n: "<tool_call>" + " " * n, 1_500, 200_000),
    # Alert 189's SECOND witness. Asserted separately rather than assumed covered
    # by the first, and that turned out to matter: one character before the
    # whitespace run drives the legacy pattern down a completely different path.
    # MEASURED pre-fix at n=2000: witness 1 costs 7.03s, witness 2 costs 0.0095s -
    # a 740x difference. Both are genuinely superlinear (witness 2 grows ~4x per
    # doubling) but witness 2 needs n=24,000 to reach the budget, so it gets a much
    # higher cap. At the old 24,000 cap it fired exactly AT the cap, which on a
    # faster box would have failed as "the budget is guarding nothing".
    ("parser._RE_XML (witness 2)",
     lambda s: [m.span() for m in _RE_XML_LEGACY.finditer(s)],
     lambda s: list(_iter_xml_tool_calls(s)),
     lambda n: "<tool_call>a" + " " * n, 1_500, 200_000),
    ("parser marker-variant",
     lambda s: [m.span() for m in _LEGACY_VARIANT.finditer(s)],
     lambda s: list(_iter_marker_variant_calls(s)),
     lambda n: "<tool_call>{{" * n, 6_000, 96_000),
    ("parser._RE_FENCE",
     lambda s: [m.span() for m in _RE_FENCE_LEGACY.finditer(s)],
     lambda s: list(_iter_fenced_blocks(s)),
     lambda n: "```" + "\t" * n, 20_000, 320_000),
])
def test_budget_fires_on_the_prefix_pattern_and_not_on_the_fixed_one(
        label, legacy, fixed, witness, base, cap):
    """Grows the witness until the PRE-FIX pattern blows the budget, then checks
    the FIXED pattern is still far under it at that very same size.

    Growing rather than hard-coding a size is what keeps this honest in both
    directions: on a slow or loaded box it fires on the first try, and on a much
    faster future box it keeps doubling instead of reporting a false green. The
    superlinearity is what makes the search terminate quickly - each doubling is
    4x or 8x the work.
    """
    size = base
    while size <= cap:
        _, legacy_elapsed = _timed(legacy, witness(size))
        if legacy_elapsed >= BUDGET:
            break
        size *= 2
    else:
        pytest.fail(f"{label}: the {BUDGET}s budget never fired on the pre-fix "
                    f"pattern even at n={cap} - it is guarding nothing")

    _, fixed_elapsed = _timed(fixed, witness(size))
    assert fixed_elapsed < BUDGET, (
        f"{label}: at n={size} the pre-fix pattern took {legacy_elapsed:.2f}s "
        f"and the fixed one took {fixed_elapsed:.2f}s")


# ---------------------------------------------------------------------------
#  Semantic regression - the bounded patterns must still match real input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    # The exact tolerance the bounded \s{0,8} was chosen to keep.
    ("x </ tool_result > y", "x &lt;/ tool_result > y"),
    ("<tool_result>", "&lt;tool_result>"),
    ("</tool_result>", "&lt;/tool_result>"),
    ("<TOOL_RESULT>", "&lt;TOOL_RESULT>"),
    ("< / untrusted_content>", "&lt; / untrusted_content>"),
    ("<untrusted_content>", "&lt;untrusted_content>"),
    # Eight spaces is still stray whitespace and still defanged.
    ("<" + " " * 8 + "/tool_result>", "&lt;" + " " * 8 + "/tool_result>"),
    # Ordinary text with a bare '<' is left alone.
    ("a < b and vector<int> v", "a < b and vector<int> v"),
])
def test_neutralise_still_defangs_frame_markers(raw, expected):
    assert neutralise(raw) == expected


def test_neutralise_still_defangs_control_tokens():
    """The second pass (_SPECIAL_RE) was already bounded and must be untouched."""
    assert neutralise("<|im_start|>system") == "&lt;|im_start|>system"
    assert neutralise("</s>[INST]") == "&lt;/s>&#91;INST]"


@pytest.mark.parametrize("raw, name, args", [
    ('<tool_call>\n{"name": "read_file", "args": {"path": "src/main.py"}}\n</tool_call>',
     "read_file", {"path": "src/main.py"}),
    ('<tool_call>{"name":"read_file","args":{"path":"a.py"}}</tool_call>',
     "read_file", {"path": "a.py"}),
    ('<tool_call name="read_file">\n{"path": "a.py"}\n</tool_call>',
     "read_file", {"path": "a.py"}),
    # Case-insensitive, and nested args survive the body-group change.
    ('<TOOL_CALL>  {"name": "edit", "args": {"a": {"b": 1}}}  </TOOL_CALL>',
     "edit", {"a": {"b": 1}}),
    # Marker-variant dialects still recover.
    ('<|tool_call>call:read_file{"path": "utils.py"}<tool_call|>',
     "read_file", {"path": "utils.py"}),
    ('<|tool_call>{"name": "read_file", "args": {"path": "a.py"}}<|tool_call>',
     "read_file", {"path": "a.py"}),
    # Fenced blocks, explicit lang and with surrounding prose.
    ('```tool_call\n{"name": "read_file", "args": {"path": "a.py"}}\n```',
     "read_file", {"path": "a.py"}),
    ('before\n```tool_code\n{"name": "read_file", "args": {"path": "a.py"}}\n```\nafter',
     "read_file", {"path": "a.py"}),
])
def test_real_tool_calls_still_parse(raw, name, args):
    calls = parse_tool_calls(raw)
    assert len(calls) == 1, f"lost the call in {raw!r}"
    assert calls[0].name == name
    assert calls[0].args == args
    # The raw span must still cover the whole wrapper so split_response can
    # reconstruct the message with the result spliced in place.
    assert raw[calls[0].start:calls[0].end] == calls[0].raw
    assert calls[0].raw.strip().endswith(("</tool_call>", "</TOOL_CALL>", "```",
                                          "<tool_call|>", "<|tool_call>"))


def test_fence_with_inline_backticks_still_parses():
    """The fence body may contain ``` inline - a write_file call carrying
    markdown is ordinary coder traffic, and it is why the body is NOT tempered
    against the fence delimiter."""
    raw = ('```tool_call\n'
           '{"name": "write_file", "args": {"path": "R.md", '
           '"content": "see ``` for fences"}}\n'
           '```')
    calls = parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert "```" in calls[0].args["content"]


def test_name_gated_fences_still_require_a_known_tool():
    raw = '```json\n{"name": "read_file", "args": {"path": "a.py"}}\n```'
    assert parse_tool_calls(raw) == []
    calls = parse_tool_calls(raw, {"read_file"})
    assert len(calls) == 1 and calls[0].name == "read_file"


def test_multiple_fenced_blocks_are_all_found():
    """The pairing loop advances like finditer: non-overlapping, left to right."""
    raw = ('```tool_call\n{"name": "a", "args": {}}\n```\n'
           'prose\n'
           '```tool_call\n{"name": "b", "args": {}}\n```')
    calls = parse_tool_calls(raw)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].end <= calls[1].start


def test_session_transcript_uses_the_parser_splitter():
    """session.py must not carry its own copy of the tool_call regex - that
    duplicate was the reason a poisoned response re-hung the transcript export
    forever after, and two copies drift."""
    import pathlib

    from localm.plugins.coder import parser as parser_mod
    from localm.plugins.coder.agent import session as session_mod

    src = pathlib.Path(session_mod.__file__).read_text(encoding="utf-8")
    assert "_TC_RE" not in src, "session.py still defines a private tool_call regex"
    assert "re.compile" not in src and "_re.compile" not in src, \
        "session.py compiled a regex again"
    assert session_mod.strip_xml_tool_calls is parser_mod.strip_xml_tool_calls


@pytest.mark.parametrize("content, calls, clean", [
    ('a<tool_call>{"x": 1}</tool_call>b', [(None, '{"x": 1}')], "ab"),
    ('<tool_call>\n  {"x": 1}\n</tool_call>', [(None, '{"x": 1}')], ""),
    ('<tool_call name="w">{"p": 1}</tool_call>', [("w", '{"p": 1}')], ""),
    ("no calls here", [], "no calls here"),
    ('x<TOOL_CALL>{"a":1}</TOOL_CALL>y<tool_call>{"b":2}</tool_call>z',
     [(None, '{"a":1}'), (None, '{"b":2}')], "xyz"),
    # ZERO-LENGTH body. The replaced regexes used ``.*?``, so this was a real
    # match for them. Searching the closer from opener.end()+1 skipped it and
    # paired the opener with a LATER closer, swallowing the prose and the next
    # genuine call into one unparseable body.
    ("<tool_call></tool_call>tail", [(None, "")], "tail"),
    ('<tool_call></tool_call>prose<tool_call>{"a":1}</tool_call>',
     [(None, ""), (None, '{"a":1}')], "prose"),
    ("<tool_call> </tool_call>tail", [(None, "")], "tail"),
])
def test_strip_xml_tool_calls_splits_calls_from_prose(content, calls, clean):
    """What the transcript and the resume recap need: the same bodies findall()
    gave (stripped), the same remainder sub("") gave, and the name= attribute
    that the replaced regexes never matched at all."""
    assert strip_xml_tool_calls(content) == (calls, clean)


# The regex that actually lived in agent/session.py, with a STAR body. The XML
# fuzz above uses the PARSER's old pattern, which had a PLUS body - so it cannot
# see a zero-length-body divergence. That gap let a real transcript regression
# through a green suite: the one regex whose behaviour changed was the one the
# oracle did not model.
_LEGACY_SESSION_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def test_strip_matches_the_deleted_session_regex_under_fuzz():
    """Differential fuzz for the SESSION splitter specifically.

    Compared only on inputs without the ``name=`` form and without uppercase
    tags, because there the new splitter is a deliberate SUPERSET (the old
    private regex was case-sensitive and had no name= support, so those blocks
    used to survive as raw XML). Everywhere else it must agree exactly, and the
    zero-gap ``<tool_call></tool_call>`` is included in the alphabet on purpose.
    """
    rnd = random.Random(4242)
    alphabet = ["<tool_call>", "</tool_call>", "<tool_call></tool_call>",
                " ", "\n", "prose", '{"a":1}', "{", "}", "<", ">", "/"]
    checked = 0
    for _ in range(40_000):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 12)))
        legacy_bodies = _LEGACY_SESSION_TC.findall(text)
        legacy_clean = _LEGACY_SESSION_TC.sub("", text)
        got_calls, got_clean = strip_xml_tool_calls(text)
        assert [b for _n, b in got_calls] == [b.strip() for b in legacy_bodies], \
            f"bodies diverged on {text!r}"
        assert got_clean == legacy_clean, f"remainder diverged on {text!r}"
        checked += 1
    assert checked == 40_000


# ---------------------------------------------------------------------------
#  Differential fuzz - the fence rewrite must match the pattern it replaced
# ---------------------------------------------------------------------------

_FUZZ_ALPHABET = ["```", "`", "\n", "\r\n", " ", "\t", "a", "py", "json",
                  "{", "}", '"', "_", "-", "+", ".", "1"]


_XML_FUZZ_ALPHABET = ["<tool_call>", "</tool_call>", '<tool_call name="a">',
                      "<TOOL_CALL>", "<tool_call", ">", "<", "/", " ", "\n",
                      "\t", "{", "}", '"', "a", "1", "name=", "'"]


def test_xml_pairing_matches_the_legacy_regex_under_fuzz():
    """The rewrite is a COST change, and this pins how far that is literally true.

    The legacy pattern's leading ``\\s*`` was GREEDY, so when the text between an
    opener and its nearest closer was entirely whitespace it could not use that
    closer (the lazy body still needed one character), and it skipped ahead to a
    LATER closer, swallowing everything in between. The pairing takes the nearest
    closer. So two properties, both fuzzed, rather than one blanket equality:

      * absent a whitespace-only block, the spans are byte-identical - and spans
        are what split_response and the transcript depend on;
      * a call the legacy pattern recovered is ALWAYS still recovered. The
        divergent legacy match starts with ``</tool_call>`` and so never parsed
        as JSON anyway, while its oversized span could swallow a real call
        further on - which is why the pairing recovers strictly more, never less.

    Body text is compared stripped because the legacy pattern kept the flanking
    whitespace outside its body group; every consumer strips it (_try_parse_body's
    first line, and strip_xml_tool_calls).
    """
    rnd = random.Random(20260728)
    saw_blank = saw_calls = 0
    for _ in range(30_000):
        text = "".join(rnd.choice(_XML_FUZZ_ALPHABET)
                       for _ in range(rnd.randint(1, 22)))
        legacy = [(m.start(), m.end(), m.group("name_attr"),
                   m.group("body").strip())
                  for m in _RE_XML_LEGACY.finditer(text)]
        got = [(s, e, n, b.strip()) for s, e, n, b in _iter_xml_tool_calls(text)]

        if any(not body for _s, _e, _n, body in got):
            saw_blank += 1
        else:
            assert got == legacy, f"spans diverged on {text!r}"

        legacy_calls = [c for c in (_try_parse_body(b, n)
                                    for _s, _e, n, b in legacy) if c]
        got_calls = [c for c in (_try_parse_body(b, n)
                                 for _s, _e, n, b in got) if c]
        saw_calls += len(got_calls)
        assert all(c in got_calls for c in legacy_calls), \
            f"the pairing LOST a call the legacy pattern found: {text!r}"

    # The corpus must actually reach both branches, or the assertions above are
    # asserting nothing (a fuzz that never generates a tool call proves nothing).
    assert saw_blank > 0, "fuzz never produced a whitespace-only block"
    assert saw_calls > 0, "fuzz never produced a parseable tool call"


def test_fence_pairing_matches_the_legacy_regex_under_fuzz():
    """The rewrite is a COST change, not a behaviour change. Fixed seed so a
    failure is reproducible; the alphabet is biased to fence-relevant tokens so
    the cases actually exercise openers, closers and near-misses."""
    rnd = random.Random(20260728)
    for _ in range(30_000):
        text = "".join(rnd.choice(_FUZZ_ALPHABET)
                       for _ in range(rnd.randint(1, 40)))
        legacy = [(m.start(), m.end(), m.group("lang"), m.group("body"))
                  for m in _RE_FENCE_LEGACY.finditer(text)]
        assert list(_iter_fenced_blocks(text)) == legacy, f"diverged on {text!r}"


# The witnesses are built by a factory and the params are NAMED. Parametrizing
# over the literal 100,000-character strings puts them in the pytest node id,
# which the autouse tmp_path fixture then tries to turn into a directory name -
# 24 setup/teardown errors and a 3.1 MB log, from a test whose assertions were
# all fine.
@pytest.mark.parametrize("pattern_name, pattern", [
    ("fence_open", _RE_FENCE_OPEN), ("fence_close", _RE_FENCE_CLOSE),
    ("xml_open", _RE_XML_OPEN), ("xml_close", _RE_XML_CLOSE),
])
@pytest.mark.parametrize("witness_name, make_witness", [
    ("fence_tabs", lambda: "```" + "\t" * 100_000),
    ("xml_opener_spaces", lambda: "<tool_call" + " " * 100_000),
    ("bare_lt_spaces", lambda: "<" + " " * 100_000),
])
def test_pairing_halves_are_individually_linear(
        pattern_name, pattern, witness_name, make_witness):
    """Each half of a pairing must be cheap on its own, or the loop calling them
    is not linear either - the pairing only removes the cost of the BODY scan."""
    hostile = make_witness()
    _, elapsed = _timed(lambda: list(pattern.finditer(hostile)))
    assert elapsed < BUDGET, (
        f"{pattern_name} on {witness_name} took {elapsed:.2f}s")


# ---------------------------------------------------------------------------
#  The marker-variant body: brace-matched, not pattern-matched
# ---------------------------------------------------------------------------
#
# The `<|tool_call>` finetune dialect used to be one regex. Every regex form of
# it forced a choice between two failures, which is why it is now a scan:
#   - a lazy `\{.*?\}` body scans to end-of-text from EVERY marker when no
#     closing brace follows (quadratic: 2.08s on 5,000 reps of '<tool_call>{{'),
#   - and tempering it to stop at the next marker fixes that but silently drops
#     a body that legitimately CONTAINS a marker.
# A brace-matched, string-aware scan has neither problem: it knows the marker is
# inside a JSON string. The first attempt at that scan was ALSO quadratic (19.3s,
# worse than the regex) because it walked forward per marker; the single-pass
# brace map is what makes it linear.

_LEGACY_VARIANT = re.compile(
    r"<\|?/?tool_call\|?>\s*(?:call:(?P<name>\w+)\s*)?(?P<body>\{.*?\})"
    r"\s*<\|?/?tool_call\|?>", re.DOTALL)


@pytest.mark.parametrize("content", [
    "docs: open with <tool_call> and close with </tool_call>",
    "the |-dialect looks like <|tool_call> in the wild",
    "a closing marker alone: <tool_call|>",
])
def test_variant_body_may_contain_a_tool_call_marker(content):
    """The case the tempered regex dropped, and the reason it matters here.

    Ask the coder to edit parser.py's own module docstring - which really does
    carry these markers - or its test fixtures, and it emits a write_file whose
    CONTENT contains one. That is ordinary work in this repo, not an exotic
    input, so the parser has to survive it."""
    import json as _json
    body = _json.dumps({"path": "parser.py", "content": content})
    calls = parse_tool_calls(f"<|tool_call>call:write_file{body}<tool_call|>")
    assert len(calls) == 1, f"lost the call when the body contained a marker: {content!r}"
    assert calls[0].name == "write_file"
    assert calls[0].args["content"] == content


def test_variant_scan_never_loses_a_call_the_legacy_regex_recovered():
    r"""Differential fuzz against the pre-fix pattern, compared on ACCEPTED CALLS.

    Not on raw spans, and the difference is the finding rather than a convenience:
    the old regex's "body" was never a balanced object. Its lazy ``\{.*?\}`` ran
    from a ``{`` to whatever ``}`` happened to precede a marker, so it produced
    bodies like ``{"b": {"c": 2}}}`` (an extra brace) and ``{"a": 1} {"b": 2}``
    (two objects). Those never parsed as JSON, so they were never calls. Holding
    the replacement to span-equality would be demanding bug-compatibility with
    text the old code could not use either.

    The property that actually matters is that nothing REAL is lost, so that is
    what is asserted. The alphabet deliberately includes ``use {`` to open: an
    unmatched brace in prose is what broke an earlier global-brace-map attempt,
    and without that token the fuzz reported clean while losing calls.
    """
    rnd = random.Random(20260729)
    alphabet = ["<|tool_call>", "<tool_call|>", "<tool_call>", "</tool_call>",
                "call:write_file", "call:read_file", ' {"a": 1} ', '{"b": {"c": 2}}',
                "{", "}", " ", chr(10), "prose", '"', "{{", "use { to open"]

    def accepted(name, body):
        body = body.replace('<|"|>', '"')
        parsed = _try_parse_body(body, name)
        if parsed is None and name:
            args = _parse_gemma_args(body)
            parsed = (name, args) if args is not None else None
        return parsed

    compared = 0
    for _ in range(20_000):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 10)))
        legacy = [(m.group("name"), m.group("body"))
                  for m in _LEGACY_VARIANT.finditer(text)]
        if any(_RE_TOOL_MARKER.search(b or "") for _n, b in legacy):
            continue          # marker-in-body: the intended change, tested above
        legacy_calls = [c for c in (accepted(n, b) for n, b in legacy) if c]
        got_calls = [c for c in (accepted(n, b)
                                 for _s, _e, n, b in _iter_marker_variant_calls(text)) if c]
        for call in legacy_calls:
            assert call in got_calls, f"lost a real call on {text!r}: {call}"
        compared += 1
    assert compared > 1_000, f"fuzz compared only {compared} cases - alphabet is wrong"


@pytest.mark.parametrize("witness, size", [
    ("<tool_call>{{", 5_000),
    ("<tool_call>", 5_000),
    ("{", 200_000),
])
def test_brace_matched_variant_pass_stays_linear(witness, size):
    """The first brace-matched attempt was 19.3s here - WORSE than the 2.08s
    regex it replaced - because it walked forward from every marker. Guarding the
    property, not the implementation: a per-marker scan reintroduces it."""
    hostile = witness * size
    _, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"{witness!r}*{size} took {elapsed:.2f}s"


def test_marker_variant_stays_linear_when_none_of_many_markers_balance():
    """The gap none of the witnesses above actually exercise.

    Every witness in ``test_brace_matched_variant_pass_stays_linear`` hits
    ``_object_end_from``'s O(1) short-circuit: ``"<tool_call>{{" * n`` and
    ``"<tool_call>" * n`` both have NO closing brace anywhere, so
    ``last_close == -1`` and every marker's check returns instantly
    (``last_close < i``); ``"{" * n`` has no marker at all. None of them ever
    runs the real per-character balance scan more than once.

    This witness forces exactly that: many markers, each opening an object that
    never balances, with a single stray ``}`` far away at the very end so
    ``last_close`` is a large, real value. Each marker's balance scan then runs
    all the way through ``last_close`` and fails, and the pre-fix recovery
    (``pos = opener.end()``) retried the SAME scan from the very next marker -
    O(n) work per marker, O(n^2) overall. Measured pre-fix on this project's
    venv: 0.09s at n=500, 6.4s at n=4000 (~4x per doubling, i.e. quadratic).
    The fixed version should complete in roughly 0.02-0.05s, so 2.0s leaves a
    two-orders-of-magnitude margin against box load while still catching the
    6+s regression."""
    hostile = ("<tool_call>{" * 4_000) + "}"
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < 2.0, (
        f"parse_tool_calls took {elapsed:.2f}s on 4,000 unbalanced markers "
        "plus one stray closing brace - the per-marker balance rescan is "
        "quadratic again")
    assert calls == []
