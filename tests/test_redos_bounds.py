# SPDX-License-Identifier: AGPL-3.0-or-later
"""Superlinear-backtracking bounds for the regexes that run on remote text."""

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
    parse_tool_calls, strip_tool_calls, strip_xml_tool_calls)
from localm.textguard import _FRAME_RE, neutralise

# Generous on purpose: see the module docstring. The smallest pre-fix baseline
# among the cases below was 2.08s and the largest was 95.9s.
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
#  Wall-clock bounds (the CodeQL witnesses, verbatim)
# ---------------------------------------------------------------------------

def test_neutralise_bounded_on_frame_marker_prefix():
    """alert 194 - ``'<'`` then a long run of spaces. 60,000 is the ceiling POST /api/web/fetch itself accepts via max_chars, so this IS the route's worst case, not a synthetic one."""
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
    """alerts 189/192/193 - a ``<tool_call>`` opener then a long run of spaces."""
    hostile = prefix + " " * 20_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, (
        f"parse_tool_calls took {elapsed:.2f}s on {prefix!r} + 20k spaces")
    assert calls == []


def _best(fn, text, samples=5):
    """Fastest of n runs."""
    return min(_timed(fn, text)[1] for _ in range(samples))


@pytest.mark.parametrize("prefix", _XML_WITNESS_PREFIXES)
def test_xml_witnesses_do_not_grow_superlinearly(prefix):
    """Compares the COST RATIO across a 10x input, not two absolute bounds."""
    base = _best(parse_tool_calls, prefix + " " * 200_000)
    ten_x = _best(parse_tool_calls, prefix + " " * 2_000_000)
    assert ten_x < max(30 * base, 0.005), (
        f"{prefix!r}: {base * 1000:.2f}ms at 200k -> {ten_x * 1000:.2f}ms at 2M "
        f"({ten_x / max(base, 1e-9):.1f}x for a 10x input) - that is superlinear")


def test_parse_tool_calls_bounded_on_many_unterminated_openers():
    """The case a single-opener witness does NOT catch: thousands of ``<tool_call>`` openers and no closer anywhere, so a lazy body scans to end-of-text once per opener."""
    hostile = "<tool_call>" * 5_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"parse_tool_calls took {elapsed:.2f}s on 5k openers"
    assert calls == []


def test_parse_tool_calls_bounded_on_unterminated_variant_markers():
    """alert 191 - repeated ``<tool_call>{{`` with no closing brace anywhere, so every marker used to start a scan to end-of-text."""
    hostile = "<tool_call>{{" * 5_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"parse_tool_calls took {elapsed:.2f}s on variant markers"
    assert calls == []


def test_parse_tool_calls_bounded_on_fence_tab_run():
    """alert 190, first witness - ``'```'`` then a long run of tabs, against the opener's two adjacent ``[ \\t]*`` quantifiers."""
    hostile = "```" + "\t" * 100_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"parse_tool_calls took {elapsed:.2f}s on 100k tabs"
    assert calls == []


def test_parse_tool_calls_bounded_on_repeated_fence_openers():
    """alert 190, second witness - many fence openers with no closer after the first few characters."""
    hostile = "```\n" + "```\na" * 8_000
    calls, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"parse_tool_calls took {elapsed:.2f}s on repeated openers"
    assert calls == []


def test_parse_tool_calls_bounded_on_doubled_brace_run():
    """The oracle's ``'<tool_call>{{' * 5000`` case run through the name-gated path as well, since passing tool_names enables two extra passes (the bare top-level JSON scan) over the same hostile text."""
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
    """THE defect, stated structurally."""
    hostile = "<tool_call>" * 5_000
    opener = _CountingPattern(_RE_XML_OPEN)
    closer = _CountingPattern(_RE_XML_CLOSE)

    assert list(_pair_delimited(hostile, opener, closer)) == []
    assert len(opener.starts) == 1, f"rescanned openers: {len(opener.starts)}"
    assert len(closer.starts) == 1, f"rescanned closers: {len(closer.starts)}"


def test_pairing_searches_never_go_backwards():
    """The linearity argument rests on the searches being monotonic - each one starts where the previous match ended, so together they walk the text once."""
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
    """The whitespace tolerance must survive the ReDoS fix INTACT."""
    assert _FRAME_RE.search("<" + " " * gap + "/tool_result>") is not None
    assert _FRAME_RE.search("<" + " " * gap + "/" + " " * gap
                            + "untrusted_content>") is not None


def test_frame_marker_defanging_is_unchanged_from_the_prefix_pattern():
    """The de-ambiguated pattern must accept the same language as the ambiguous one it replaced - that is the whole claim, so it is fuzzed rather than eyeballed."""
    rnd = random.Random(20260728)
    alphabet = ["<", "/", " ", "  ", " " * 9, " " * 30, "\t", "\n", "\r",
                "tool_result", "untrusted_content", "TOOL_RESULT", "x", ">"]
    for _ in range(30_000):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 12)))
        assert (_FRAME_RE.sub(r"&lt;\1", text)
                == _LEGACY_FRAME.sub(r"&lt;\1", text)), f"diverged on {text!r}"


def test_variant_body_is_brace_matched_not_pattern_matched():
    """The marker-variant body is delimited by BRACE BALANCE, not by a pattern."""
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
    """Grows the witness until the PRE-FIX pattern blows the budget, then checks the FIXED pattern is still far under it at that very same size."""
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
    """The fence body may contain ``` inline - a write_file call carrying markdown is ordinary coder traffic, and it is why the body is NOT tempered against the fence delimiter."""
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
    """session.py must not carry its own copy of the tool_call regex - that duplicate was the reason a poisoned response re-hung the transcript export forever after, and two copies drift."""
    import pathlib

    from localm.plugins.coder import parser as parser_mod
    from localm.plugins.coder.agent import session as session_mod

    src = pathlib.Path(session_mod.__file__).read_text(encoding="utf-8")
    assert "_TC_RE" not in src, "session.py still defines a private tool_call regex"
    assert "re.compile" not in src and "_re.compile" not in src, \
        "session.py compiled a regex again"
    assert session_mod.strip_tool_calls is parser_mod.strip_tool_calls


def test_strip_tool_calls_recognises_a_fenced_json_call_with_tool_names():
    """The shape strip_xml_tool_calls never understood: a ```json fence naming a real tool."""
    text = 'Reading it now.\n```json\n{"name": "read_file", "args": {"path": "a.py"}}\n```'
    calls, clean, malformed = strip_tool_calls(text, tool_names={"read_file"})
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert clean.strip() == "Reading it now."
    assert malformed == 0

    calls2, clean2, malformed2 = strip_tool_calls(text)
    assert calls2 == []
    assert malformed2 == 0
    assert "```json" in clean2


def test_strip_tool_calls_recognises_a_bare_json_call_with_tool_names():
    text = '{"name": "write_file", "args": {"path": "out.py", "content": "x"}}'
    calls, clean, malformed = strip_tool_calls(text, tool_names={"write_file"})
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert clean == ""
    assert malformed == 0


def test_strip_tool_calls_counts_a_malformed_xml_block_and_still_cleans_it():
    text = 'Let me check.\n<tool_call>\n{"name": "read_file", "args": {"path": "a.py"\n</tool_call>'
    calls, clean, malformed = strip_tool_calls(text, tool_names={"read_file"})
    assert calls == []
    assert malformed == 1
    assert clean.strip() == "Let me check."
    assert "<tool_call>" not in clean


def test_strip_tool_calls_matches_strip_xml_tool_calls_on_xml_only_text():
    """Parity check for the case both functions have always handled."""
    text = '<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}}\n</tool_call>'
    xml_calls, xml_clean = strip_xml_tool_calls(text)
    calls, clean, malformed = strip_tool_calls(text)
    assert len(calls) == 1 and calls[0].name == "read_file"
    assert clean == xml_clean
    assert malformed == 0


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
    """What the transcript and the resume recap need: the same bodies findall() gave (stripped), the same remainder sub('') gave, and the name= attribute that the replaced regexes never matched at all."""
    assert strip_xml_tool_calls(content) == (calls, clean)


# The regex that actually lived in agent/session.py, with a STAR body. The XML
# fuzz above uses the PARSER's old pattern, which had a PLUS body - so it cannot
# see a zero-length-body divergence. That gap let a real transcript regression
# through a green suite: the one regex whose behaviour changed was the one the
# oracle did not model.
_LEGACY_SESSION_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def test_strip_matches_the_deleted_session_regex_under_fuzz():
    """Differential fuzz for the SESSION splitter specifically."""
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
    """The rewrite is a COST change, and this pins how far that is literally true."""
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
    """The rewrite is a COST change, not a behaviour change."""
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
    """Each half of a pairing must be cheap on its own, or the loop calling them is not linear either - the pairing only removes the cost of the BODY scan."""
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
    """The case the tempered regex dropped, and the reason it matters here."""
    import json as _json
    body = _json.dumps({"path": "parser.py", "content": content})
    calls = parse_tool_calls(f"<|tool_call>call:write_file{body}<tool_call|>")
    assert len(calls) == 1, f"lost the call when the body contained a marker: {content!r}"
    assert calls[0].name == "write_file"
    assert calls[0].args["content"] == content


def test_variant_scan_never_loses_a_call_the_legacy_regex_recovered():
    """Differential fuzz against the pre-fix pattern, compared on ACCEPTED CALLS."""
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
    """The first brace-matched attempt was 19.3s here - WORSE than the 2.08s regex it replaced - because it walked forward from every marker."""
    hostile = witness * size
    _, elapsed = _timed(parse_tool_calls, hostile)
    assert elapsed < BUDGET, f"{witness!r}*{size} took {elapsed:.2f}s"


def test_marker_variant_stays_linear_when_none_of_many_markers_balance():
    """The gap none of the witnesses above actually exercise."""
    hostile = ("<tool_call>{" * 4_000) + "}"
    calls, cpu_elapsed = _timed_cpu(parse_tool_calls, hostile)
    assert cpu_elapsed < 2.0, (
        f"parse_tool_calls used {cpu_elapsed:.2f}s of CPU time on 4,000 "
        "unbalanced markers plus one stray closing brace - the per-marker "
        "balance rescan is quadratic again")
    assert calls == []


# ---------------------------------------------------------------------------
#  Regression: a REAL call recovered by an earlier version of this scan must
#  still be recovered, not just "the scan stays fast"
# ---------------------------------------------------------------------------
#
# The fix above (rescan-budget cap replacing the unconditional "skip to
# last_close+1") was needed because the unconditional skip shipped a real
# correctness regression alongside the perf fix: a failed scan from an EARLIER
# marker running all the way through last_close does NOT prove no LATER
# marker can balance - a scan restarted at a later marker resets its own
# depth to 0, so it can reach 0 again well before last_close even though the
# earlier marker's scan, carrying that marker's own leftover depth, never
# did. Measured directly on this repo (bisected against 3a649bbb^, the commit
# immediately before #895): an unclosed canonical ``<tool_call>`` followed by
# a well-formed one later in the SAME response went from 1 call recovered to
# 0. Reported against qwen2.5-coder-1.5b-instruct - a "default" family model
# per prompts.py, taught ONLY the canonical ``<tool_call>`` XML wrapper (not
# the ``<|tool_call>`` finetune dialect _iter_marker_variant_calls exists
# for) - so this pass is also the safety net _iter_xml_tool_calls's strict
# opener/closer pairing relies on whenever an earlier unclosed tag steals a
# later call's closing tag (see _pair_delimited: a closer search that finds
# ANY later ``</tool_call>`` pairs with it regardless of which opener it
# structurally belongs to, so pass 1 alone mis-reads this exact shape).

def test_marker_variant_recovers_a_later_call_after_an_earlier_one_fails_to_balance():
    """The mangled ``<|tool_call>`` dialect: an earlier malformed attempt must not suppress a later, independent, well-formed one in the same response."""
    text = (
        '<|tool_call>call:write_file{"path": "x.py", "content": "def f():\\n'
        '    pass'  # deliberately truncated: opens more braces than it closes
        '\n\nLet me try that again.\n'
        '<|tool_call>call:run_tests{"runner": "pytest"}<tool_call|>\n'
    )
    calls = parse_tool_calls(text, tool_names={"write_file", "run_tests"})
    names = [c.name for c in calls]
    assert "run_tests" in names, (
        f"lost the later, well-formed call behind an earlier malformed one: {names}")


def test_xml_wrapper_recovers_via_marker_fallback_after_an_earlier_unclosed_tag():
    """The shape actually reported: a 'default'-family model (prompts.py) is taught ONLY the canonical ``<tool_call>`` XML wrapper, never the ``<|tool_call>`` dialect - so THIS is the path real small-model traffic exercises."""
    text = (
        "Let me read the file first.\n"
        "<tool_call>\n"
        '{"name": "read_file", "args": {"path": "strings_utils.py"'
        # truncated: no closing brace, no </tool_call> - this attempt is lost
        "\n\nNow I will run the tests.\n"
        "<tool_call>\n"
        '{"name": "run_tests", "args": {"runner": "pytest"}}\n'
        "</tool_call>\n"
    )
    calls = parse_tool_calls(text, tool_names={"read_file", "run_tests"})
    names = [c.name for c in calls]
    assert "run_tests" in names, (
        f"lost the later, well-formed <tool_call> behind an earlier unclosed one: {names}")


def test_marker_variant_rescan_budget_covers_realistic_repeated_failures():
    """The cap that keeps the above fix linear (_MAX_EXPENSIVE_MARKER_RESCANS) must not be so tight that an ordinary struggling small model - a handful of botched attempts before it gets the format right, not thousands - loses the call it finally got right. 10 failed attempts is already far more than _MAX_..."""
    failed_attempt = '<|tool_call>call:write_file{"path": "x.py", "content": "unterminated'
    text = ("\n\n".join([failed_attempt] * 10)
            + '\n\n<|tool_call>call:run_tests{"runner": "pytest"}<tool_call|>\n')
    calls = parse_tool_calls(text, tool_names={"write_file", "run_tests"})
    names = [c.name for c in calls]
    assert "run_tests" in names, (
        f"lost the well-formed call after only 10 realistic failed attempts: {names}")


def test_marker_variant_drops_a_call_beyond_the_rescan_budget_by_design():
    """States the boundary of the fix above explicitly, so it is a documented design property rather than a gap left for someone to discover later: _MAX_EXPENSIVE_MARKER_RESCANS is a BOUND, not a full fix."""
    from localm.plugins.coder.parser import _MAX_EXPENSIVE_MARKER_RESCANS
    failed_attempt = '<|tool_call>call:write_file{"path": "x.py", "content": "unterminated'
    n_failures = _MAX_EXPENSIVE_MARKER_RESCANS + 8
    text = ("\n\n".join([failed_attempt] * n_failures)
            + '\n\n<|tool_call>call:run_tests{"runner": "pytest"}<tool_call|>\n')
    calls = parse_tool_calls(text, tool_names={"write_file", "run_tests"})
    names = [c.name for c in calls]
    assert "run_tests" not in names, (
        "expected the budget-exhausted fallback to still drop this call - if "
        "it is now recovered, the fix's boundary moved and this test (and the "
        "PR description) need to say where the new boundary is")


def test_marker_variant_rescan_budget_keeps_cost_linear_not_quadratic():
    """Asserts the ARITHMETIC the fix relies on, not a literal."""
    small_n, big_n = 4_000, 16_000
    _, small_cpu = _timed_cpu(
        parse_tool_calls, ("<tool_call>{" * small_n) + "}")
    _, big_cpu = _timed_cpu(
        parse_tool_calls, ("<tool_call>{" * big_n) + "}")
    growth = big_n // small_n
    ratio = big_cpu / max(small_cpu, 1e-6)
    assert ratio < 8.0, (
        f"a {growth}x input increase cost {ratio:.1f}x more CPU time "
        f"({small_cpu:.3f}s -> {big_cpu:.3f}s) - closer to quadratic "
        f"({growth ** 2}x) than linear ({growth}x); the expensive-rescan "
        "budget is no longer bounding total cost to a constant number of "
        "full rescans")


# --------------------------------------------------------------------------- #
#  gbnf.TOOL_CALL_TRIGGER - the lazy-grammar trigger (GitHub #928, #833)
#
#  A SEVENTH pattern of the same class, and the reason it was not in the
#  2026-07-28 sweep is structural rather than an oversight: TOOL_CALL_TRIGGER is
#  never compiled or run through Python's `re` in production. It is a bare
#  string marshaled through ctypes into llama.cpp's native std::regex, so
#  CodeQL's `py/polynomial-redos` - a Python-level query - cannot see it at all.
#  Any regex string handed to native code is invisible to that whole query class.
#
#  llama.cpp appends every generated token to `grammar.trigger_buffer` with NO
#  SIZE CAP and re-runs the whole pattern over the whole buffer on EVERY token
#  until a trigger matches (src/llama-grammar.cpp, llama_grammar_accept_impl),
#  so a quadratic pattern makes the generation-total cost cubic. Measured on the
#  repetitive markdown-table output that provoked it in the field:
#
#      buffer     old pattern      this pattern
#       5,700       243 ms           0.061 ms
#      11,400       975 ms           0.008 ms
#      22,800     4,100 ms           0.007 ms
#
#  The separation is five to six orders of magnitude, so an ABSOLUTE budget is
#  the right instrument here (unlike the #895 test above, whose fixed-vs-broken
#  gap was small enough to need a ratio). CPU time, not wall clock: this is
#  pure CPU-bound regex work and targeted runs share the box.
# --------------------------------------------------------------------------- #

_TRIGGER_ROW = "| localm/inference/backends/llamacpp/_runner.py | 1234 |\n"


def _trigger_buffer(n_rows: int) -> str:
    """Unconstrained model output with NO trigger in it - the vulnerable state, where llama.cpp is still rescanning the whole buffer on every token."""
    return _TRIGGER_ROW * n_rows


def test_tool_call_trigger_does_not_backtrack_on_long_untriggered_output():
    from localm.inference.gbnf import TOOL_CALL_TRIGGER
    haystack = _trigger_buffer(1_800)          # ~100,000 chars
    pat = re.compile(TOOL_CALL_TRIGGER)
    _, cpu = _timed_cpu(pat.search, haystack)
    assert cpu < 0.5, (
        f"TOOL_CALL_TRIGGER took {cpu:.3f}s CPU on {len(haystack):,} chars of "
        "untriggered output. llama.cpp re-runs this on EVERY token against an "
        "uncapped buffer, so a superlinear pattern here hangs or crashes the "
        r"worker mid-generation (#928). A leading `[\s\S]*?` is the usual cause "
        "and is never needed - the native side matches with a search."
    )


def test_tool_call_trigger_still_captures_the_tag_itself():
    """The bound is worthless if the pattern stops meaning what it meant."""
    from localm.inference.gbnf import TOOL_CALL_TRIGGER
    body = '\n{"name": "read_file", "args": {"path": "a.py"}}\n</tool_call>'
    m = re.compile(TOOL_CALL_TRIGGER).search(_trigger_buffer(20) + "<tool_call>" + body)
    assert m is not None, "the trigger no longer matches a real tool call"
    assert m.group(1) == "<tool_call>" + body, (
        "capture group 1 must START AT the <tool_call> tag - the grammar is fed "
        "from it and its first literal is the tag itself"
    )
