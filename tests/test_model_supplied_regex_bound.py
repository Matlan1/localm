# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounds for the one place the REGEX ITSELF is attacker-supplied.

Every other ReDoS fix in this tree de-ambiguates a pattern WE wrote against
hostile text. `grep` and `search_replace` are the inverse: the model hands us
the pattern, and you cannot de-ambiguate one you did not write. So they are
bounded instead, with two mechanisms that are not interchangeable:

  * a per-match TIMEOUT, which is the only thing that can stop a match already
    running - stdlib ``re`` cannot be interrupted at any price;
  * hard INPUT CAPS, so the common path never reaches the timeout and an
    attacker cannot burn the full budget once per file in a glob.

Both are needed because stdlib ``re`` and ``regex`` have DIFFERENT
catastrophic sets, neither containing the other:

    (\\s*)*x  on 26 spaces   stdlib 7.0044s   regex 0.0001s
    (a|a)*$  on 30 a's       stdlib 2.3561s   regex 6.4100s

``regex`` is immune to the first and ~2.7x worse on the second, the alternation
shape a model writes when searching for alternatives. An engine swap alone moves
the vulnerability rather than removing it.

A THIRD failure mode: a pathological recursive or possessive pattern raises a
catchable ``MemoryError`` from ``regex``. ``_run_model_regex`` catches it
alongside ``regex.error`` and ``TimeoutError``, so it is attributed to the
PATTERN. Uncaught, it reaches each caller's generic handler and is misreported -
`grep` as an unreadable FILE, `search_replace` as a bare ``"Tool error: "``
(``str(MemoryError())`` is ``''``).
"""

from __future__ import annotations

import time

import pytest
import regex as _regex_module

from localm.plugins.coder.tools.files import (
    _MODEL_REGEX_MAX_LINE, _MODEL_REGEX_TIMEOUT, _ModelRegexTooExpensive,
    _ModelRegexTooSlow, _compile_model_pattern, _model_regex_flags,
    _run_model_regex, tool_grep, tool_search_replace)

# The witness the `regex` ENGINE is catastrophic on. Sized so the timeout fires
# well inside the budget rather than sitting on the boundary.
_ENGINE_KILLER = r"(a|a)*$"
_ENGINE_KILLER_INPUT = "a" * 60 + "!"

# The witness that exhausts MEMORY rather than time: a recursive possessive
# quantifier the `regex` engine (2026.7.19+) turns into an unbounded backtracking
# allocation instead of an infinite loop. On regex < 2026.7.19 this SEGFAULTS the
# process instead of raising, which the version guard below turns into a skip.
_MEMORY_KILLER = r"(?:a(?R)?b){e<=1}"
_MEMORY_KILLER_INPUT = "aabb"

_REGEX_VERSION = tuple(int(p) for p in _regex_module.__version__.split(".")[:3])
_regex_too_old_for_memory_probe = pytest.mark.skipif(
    _REGEX_VERSION < (2026, 7, 19),
    reason=(
        f"regex=={_regex_module.__version__} is older than 2026.7.19: on this "
        "engine _MEMORY_KILLER SEGFAULTS the process instead of raising "
        "MemoryError (the bug these tests exist to catch only became "
        "catchable in 2026.7.19, see #967). Sync the venv before running this "
        "test rather than risk crashing the whole run."
    ),
)


def _timed(fn, *a, **kw):
    start = time.perf_counter()
    result = fn(*a, **kw)
    return result, time.perf_counter() - start


# ---------------------------------------------------------------------------
#  The guard stops a runaway, through the real tool
# ---------------------------------------------------------------------------

@pytest.fixture
def hostile_repo(tmp_path):
    (tmp_path / "a.txt").write_text(_ENGINE_KILLER_INPUT, encoding="utf-8")
    return tmp_path


def test_grep_aborts_a_runaway_pattern_and_says_why(hostile_repo):
    """Without this, a guard that never fires and a broken guard are the same
    green run. The message has to name the SHAPE, because "too slow" alone
    leaves the model with no idea what to change."""
    result, elapsed = _timed(tool_grep, hostile_repo, _ENGINE_KILLER)
    assert result.ok is False, "a runaway pattern must not report success"
    assert elapsed < _MODEL_REGEX_TIMEOUT * 3, f"abort took {elapsed:.2f}s"
    message = (result.output or "") + (result.summary or "")
    assert "longer than" in message
    assert "quantifier" in message, "the message must name the shape to fix"


def test_search_replace_aborts_before_writing_anything(hostile_repo):
    """search_replace MUTATES, so a partial sweep is worse than none: half a glob
    rewritten and the rest not is a state nobody asked for and the caller cannot
    tell from success."""
    before = (hostile_repo / "a.txt").read_text(encoding="utf-8")
    result, elapsed = _timed(tool_search_replace, hostile_repo, _ENGINE_KILLER,
                             "REPLACED", glob="**/*")
    assert result.ok is False
    assert elapsed < _MODEL_REGEX_TIMEOUT * 3
    assert (hostile_repo / "a.txt").read_text(encoding="utf-8") == before, \
        "the file was modified despite the abort"


def test_the_timeout_is_what_stops_it_not_luck():
    """Asserts the MECHANISM. A test that only checked wall-clock would pass if
    the pattern happened to be fast for some unrelated reason."""
    rx = _compile_model_pattern(_ENGINE_KILLER, _model_regex_flags())
    with pytest.raises(_ModelRegexTooSlow):
        _run_model_regex(rx.search, _ENGINE_KILLER_INPUT)


# ---------------------------------------------------------------------------
#  Memory exhaustion is a DIFFERENT fact than a timeout, and is attributed to
#  the PATTERN rather than to an unreadable file or an empty "Tool error: ".
#  MemoryError is an Exception subclass, so it has to be caught specifically.
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_hostile_repo(tmp_path):
    (tmp_path / "a.txt").write_text(_MEMORY_KILLER_INPUT, encoding="utf-8")
    return tmp_path


@_regex_too_old_for_memory_probe
def test_grep_aborts_a_memory_exhausting_pattern_and_names_the_pattern(memory_hostile_repo):
    """Before the fix this fell through to the generic per-file handler and was
    reported as 'N file(s) could not be read and were not searched' - wrong and
    unactionable, since nothing is wrong with the file. It must also NOT reuse
    the timeout wording ('took longer than'/'timed out'): that would be false,
    since this is memory exhaustion on a bounded, non-timed-out match."""
    result = tool_grep(memory_hostile_repo, _MEMORY_KILLER)
    assert result.ok is False, "a memory-exhausting pattern must not report success"
    message = (result.output or "") + (result.summary or "")
    assert "could not be read" not in message, \
        "must not be misfiled as an unreadable file"
    assert "not searched" not in message
    assert "longer than" not in message, "must not claim a timeout that did not happen"
    assert "memory" in message.lower(), "the message must name what actually happened"
    assert "pattern" in message.lower(), "the message must name the PATTERN as the cause"


@_regex_too_old_for_memory_probe
def test_search_replace_aborts_before_writing_anything_on_memory_exhaustion(memory_hostile_repo):
    """Before the fix this escaped tool_search_replace's try block entirely (no
    handler matched MemoryError) and reached execution.py's generic
    `except Exception as e: ToolResult.error(f"Tool error: {e}")` - and
    str(MemoryError()) is '', so it surfaced as a bare, empty 'Tool error: '.
    search_replace MUTATES, so it must also abort before writing anything."""
    before = (memory_hostile_repo / "a.txt").read_text(encoding="utf-8")
    result = tool_search_replace(memory_hostile_repo, _MEMORY_KILLER, "REPLACED",
                                 glob="**/*")
    assert result.ok is False
    message = (result.output or "") + (result.summary or "")
    assert message.strip(), "must not surface as an empty 'Tool error: '"
    assert "memory" in message.lower()
    assert "pattern" in message.lower()
    assert (memory_hostile_repo / "a.txt").read_text(encoding="utf-8") == before, \
        "the file was modified despite the abort"


@_regex_too_old_for_memory_probe
def test_the_memoryerror_is_what_stops_it_not_luck():
    """Asserts the MECHANISM, mirroring test_the_timeout_is_what_stops_it_not_luck:
    a broad `except Exception` would also make this pass without actually
    distinguishing memory exhaustion from a timeout, which is exactly the bug
    (both misattributions passed every OTHER test in this file)."""
    rx = _compile_model_pattern(_MEMORY_KILLER, _model_regex_flags(ignore_case=True))
    with pytest.raises(_ModelRegexTooExpensive):
        _run_model_regex(rx.search, _MEMORY_KILLER_INPUT)


# ---------------------------------------------------------------------------
#  Two heap-buffer-overflow bugs in the ENGINE ITSELF (regex issues #611 and
#  #612), fixed in regex 2026.8.31. Neither is a slow or memory-hungry match,
#  so the timeout/cap machinery above does not and cannot guard against them -
#  the only guard is the engine version.
# ---------------------------------------------------------------------------

# Issue #611: a conditional group with no explicit `|no` branch reports
# itself empty via a boolean-precedence bug in `is_empty()`, so its whole
# subtree - including capture groups that already incremented the parser's
# group count - is dropped without rolling that count back. The C compiler
# then under-allocates relative to the inflated public group count, and
# mark_named_groups() writes past the allocation. regex.compile() returns
# normally either way; see the test below for what this specific pattern
# does and does not reveal about that from pure Python.
_GROUP_COUNT_OVERFLOW_PATTERN = ('((?!))(?(1)(?(?=' + '()' * 15 +
    '(?<a>)(?<b>)()(?<c>)()()(?<d>)(?<e>)' + ')))')

# Issue #612: BESTMATCH+POSIX fuzzy matching narrows the search slice for its
# final retry while a cached required-string position still points past the
# new slice end. The stale position feeds a reverse class test that reads
# past the subject buffer, and the byte it reads decides whether a match is
# returned.
_GHOST_MATCH_PATTERN = (r'(?bps)(?:qq){e<=2}fd~fecddabab(?:ff{d<=1}|'
    r'.{0,2100}+(?-i:(?<=[\x00-\x7d\x7f-\xff])))')
_GHOST_MATCH_SUBJECT = "~" * 6 + "fd~fecddabab" * 2 + "~" * 2000

_regex_too_old_for_heap_bounds_probe = pytest.mark.skipif(
    _REGEX_VERSION < (2026, 8, 31),
    reason=(
        f"regex=={_regex_module.__version__} is older than 2026.8.31: regex "
        "issues #611 (heap-buffer-overflow WRITE at compile time) and #612 "
        "(heap-buffer-overflow READ at search time), both reachable through "
        "a model-supplied pattern with no match needed to be interrupted, "
        "were fixed in that release. Sync the venv before running this "
        "test."
    ),
)


def test_the_group_count_overflow_pattern_compiles_and_never_matches():
    """Compiles _GROUP_COUNT_OVERFLOW_PATTERN through the real call site on
    any regex version. Deliberately NOT version-gated and does NOT assert on
    rx.groups or rx.groupindex: both are 24, with the same groupindex
    mapping, on every regex version this project supports, vulnerable or
    fixed - the regex 2026.8.31 fix corrects an internal C allocation to
    match the public group count, which was never wrong itself, rather than
    changing the count. So neither is a Python-visible discriminator for
    this bug, and asserting on either here would not detect a regression.
    What this test actually proves is narrower: the pattern does not raise
    or crash through this project's own compile and search path. The fix
    for regex issue #611 is verified independently, by diffing
    regex/_regex_core.py's LookAroundConditional.is_empty() against the
    upstream release that fixes it."""
    rx = _compile_model_pattern(_GROUP_COUNT_OVERFLOW_PATTERN, _model_regex_flags())
    result = _run_model_regex(rx.search, "anything")
    assert result is None, "the outer negative-lookahead group can never match"


@_regex_too_old_for_heap_bounds_probe
def test_the_stale_required_string_cache_no_longer_produces_a_ghost_match():
    """Below regex 2026.8.31, _GHOST_MATCH_PATTERN searched against
    _GHOST_MATCH_SUBJECT through the real call site returns a Match spanning
    (16, 2130): the OOB read hands a garbage byte to a reverse class test
    that should have failed, so the fuzzy-match retry reports success on a
    subject it should have rejected. Deterministic on both sides of the fix
    (regex issue #612), so this is a real behavioral regression guard, not
    merely a crash probe."""
    rx = _compile_model_pattern(_GHOST_MATCH_PATTERN, _model_regex_flags())
    result = _run_model_regex(rx.search, _GHOST_MATCH_SUBJECT)
    assert result is None, (
        "expected no match once the stale required-string cache is reset on "
        f"a narrowed slice; got a spurious match at "
        f"{result.span() if result else None} instead - the heap-read bug "
        "(regex issue #612) may have reappeared")


# ---------------------------------------------------------------------------
#  The bound must not cost the feature
# ---------------------------------------------------------------------------

def test_ordinary_grep_still_works(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return HAYSTACK\n", encoding="utf-8")
    result = tool_grep(tmp_path, "HAYSTACK")
    assert result.ok is True
    assert "1 match" in (result.summary or "")


@pytest.mark.parametrize("pattern", [
    r"def \w+\(",            # ordinary code search
    r"^\s*import ",          # anchored, with a quantifier
    r"TODO|FIXME",           # plain alternation, no nesting
    r"[A-Z_]{3,}",           # bounded repetition
])
def test_real_patterns_a_coder_writes_are_unaffected(tmp_path, pattern):
    (tmp_path / "a.py").write_text(
        "import os\n\ndef helper(x):\n    # TODO: fix\n    return CONSTANT_NAME\n",
        encoding="utf-8")
    result, elapsed = _timed(tool_grep, tmp_path, pattern)
    assert result.ok is True, f"{pattern!r} was rejected"
    assert elapsed < 1.0


def test_ordinary_search_replace_still_rewrites(tmp_path):
    (tmp_path / "a.py").write_text("old_name = 1\n", encoding="utf-8")
    result = tool_search_replace(tmp_path, r"old_name", "new_name", glob="**/*.py")
    assert result.ok is True
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "new_name = 1\n"


# ---------------------------------------------------------------------------
#  Input caps
# ---------------------------------------------------------------------------

def test_an_absurdly_long_line_is_skipped_not_searched(tmp_path):
    """The per-match budget would stop a runaway on one line, but paying it once
    per line of a large file is the accumulation the caps exist to prevent."""
    (tmp_path / "long.txt").write_text(
        "x" * (_MODEL_REGEX_MAX_LINE + 1000) + "\nNEEDLE\n", encoding="utf-8")
    result, elapsed = _timed(tool_grep, tmp_path, "NEEDLE")
    assert result.ok is True
    assert elapsed < 2.0
    # The short line is still searched - the cap skips a LINE, not the file.
    assert "1 match" in (result.summary or "")


def test_search_replace_honours_the_file_size_cap(tmp_path, monkeypatch):
    """`search_replace` honours the same per-file size cap as `grep`: one knob,
    one meaning.

    Patches ``_grep_config``, not the module default: ``_grep_cap`` resolves
    arg > CONFIG > default and the shipped config sets this key, so the default
    is never consulted.
    """
    import localm.plugins.coder.tools.files as files_mod
    monkeypatch.setattr(files_mod, "_grep_config",
                        lambda: {"coder_grep_max_file_bytes": 1024})
    (tmp_path / "big.txt").write_text("NEEDLE" + "x" * 4000, encoding="utf-8")
    (tmp_path / "small.txt").write_text("NEEDLE\n", encoding="utf-8")
    result = tool_search_replace(tmp_path, "NEEDLE", "FOUND", glob="**/*")
    assert result.ok is True
    assert (tmp_path / "big.txt").read_text(encoding="utf-8").startswith("NEEDLE"), \
        "the oversized file must be skipped, not rewritten"
    assert (tmp_path / "small.txt").read_text(encoding="utf-8") == "FOUND\n"


def test_the_size_cap_test_above_would_fail_without_the_cap(tmp_path, monkeypatch):
    """The control for the cap test: with the ceiling raised, the same oversized
    file IS rewritten. A cap test that patched the wrong layer would otherwise
    look identical to a working cap, since both leave the file untouched."""
    import localm.plugins.coder.tools.files as files_mod
    monkeypatch.setattr(files_mod, "_grep_config",
                        lambda: {"coder_grep_max_file_bytes": 10 * 1024 * 1024})
    (tmp_path / "big.txt").write_text("NEEDLE" + "x" * 4000, encoding="utf-8")
    result = tool_search_replace(tmp_path, "NEEDLE", "FOUND", glob="**/*")
    assert result.ok is True
    assert (tmp_path / "big.txt").read_text(encoding="utf-8").startswith("FOUND"), \
        "under a large cap the file must be rewritten - if not, the cap test proves nothing"


# ---------------------------------------------------------------------------
#  No silent downgrade
# ---------------------------------------------------------------------------

def test_regex_is_a_declared_core_dependency():
    """It arrives transitively via `transformers`, which is OPTIONAL - so a base
    install would have had no `regex` and this guard would have been absent
    exactly where nobody was looking. Declared directly, same lesson as certifi."""
    import pathlib
    import tomllib
    root = pathlib.Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    core = data["project"]["dependencies"]
    assert any(d.split(">")[0].split("=")[0].strip() == "regex" for d in core), \
        f"regex must be a CORE dependency, not transitive. core={core}"


def test_there_is_no_fallback_to_the_unbounded_engine():
    """A fallback to stdlib `re` when `regex` is missing would silently restore
    the unbounded path while every caller believed it was bounded."""
    import inspect

    import localm.plugins.coder.tools.files as files_mod
    source = inspect.getsource(files_mod._compile_model_pattern)
    assert "re.compile" not in source, "the guard must not fall back to stdlib re"
    assert "_ModelRegexUnavailable" in source, "a missing engine must refuse loudly"
