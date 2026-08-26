# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load-time gate on a pulled model's tokenizer.json.

Which backtracking engine a validator probes with decides what it can see: a
lookahead-in-a-loop pattern, ``((?=(a+))a)+b``, that Python `re` handles in
well under a second at N=2000 is ALREADY hung past 12 seconds in Oniguruma (the
engine ``tokenizers.Regex`` actually runs) on the identical input - the reverse
of the ``(a|a)*b`` direction, which is catastrophic in `re` and fast in
Oniguruma. A validator reusing `_trigger_probe.py`'s `re`-based probe would
score that pattern "safe".
"""

from __future__ import annotations

import json
import re
import time

import pytest

tokenizers = pytest.importorskip("tokenizers")

from localm.inference import hf_tokenizer_safety as safety  # noqa: E402
from localm.inference._hf_tokenizer_probe import _check_one  # noqa: E402

# A lookahead-in-a-loop pattern. re.compile accepts it and is still sub-second
# at N=2000, while Oniguruma is already hung there. 1800 is the input length
# used by the "must be rejected" tests below.
_UNTESTED_DIRECTION_PATTERN = r"((?=(a+))a)+b"
_UNTESTED_DIRECTION_INPUT = "a" * 1800

# The classic nested-quantifier shape: catastrophic (unbounded hang) in `re`,
# but hits Oniguruma's own internal backtrack ceiling almost immediately -
# surfacing as pyo3_runtime.PanicException, a BaseException, NOT an Exception.
_PANIC_PATTERN = r"(a+)+b"
_PANIC_INPUT = "a" * 60

# Same shape as _PANIC_PATTERN, keyed to a PUNCTUATION character instead of a
# letter, so it exercises _pattern_derived_probes on a non-alnum character.
# _PROBE_PUNCTUATION in _FIXED_PROBES does not catch it: it interleaves 16
# distinct punctuation characters, so no run of consecutive commas long enough
# to trip the ambiguity ever forms.
_PANIC_PATTERN_PUNCT = r"(,+)+b"

# The real, unmodified GPT-2 byte-level pre-tokenizer pattern (also used by
# GPT-NeoX and others), via \p{L}/\p{N} Unicode-property escapes that Python's
# `re` does not support at all.
_GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _timed(fn, *a, **kw):
    start = time.perf_counter()
    result = fn(*a, **kw)
    return result, time.perf_counter() - start


def _model_dir(tmp_path, name, tokenizer_json_obj=None, skip_tokenizer_json=False):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    if not skip_tokenizer_json:
        (d / "tokenizer.json").write_text(
            json.dumps(tokenizer_json_obj), encoding="utf-8")
    return d


def _with_pre_tokenizer_regex(pattern):
    return {"pre_tokenizer": {"type": "Split", "pattern": {"Regex": pattern},
                              "behavior": "Isolated", "invert": False}}


# ---------------------------------------------------------------------------
#  The untested direction: re is tolerable, Oniguruma is not
# ---------------------------------------------------------------------------

def test_the_untested_direction_is_real_re_tolerable_oniguruma_hung():
    """`re` completes on the witness well inside a budget a per-request check
    could plausibly use, while raw Oniguruma (bypassing this validator) is
    already far past it on the identical pattern and input."""
    _, re_elapsed = _timed(re.compile(_UNTESTED_DIRECTION_PATTERN).search,
                           _UNTESTED_DIRECTION_INPUT)
    assert re_elapsed < 2.0, f"re took {re_elapsed:.2f}s - weakens the contrast"

    regex = tokenizers.Regex(_UNTESTED_DIRECTION_PATTERN)
    splitter = tokenizers.pre_tokenizers.Split(regex, behavior="isolated")
    _, onig_elapsed = _timed(splitter.pre_tokenize_str, _UNTESTED_DIRECTION_INPUT)
    assert onig_elapsed > 5.0, (
        f"onig took only {onig_elapsed:.2f}s - the pattern is no longer a "
        "witness for the untested direction, update it")


def test_oniguruma_retry_limit_is_a_baseexception_not_an_exception():
    """States the exact type the probe MUST catch, independent of this
    validator's own code: pyo3_runtime.PanicException's MRO is
    (PanicException, BaseException, object), so a bare `except Exception:` does
    not catch it. If tokenizers ever makes it a normal Exception subclass, the
    assertion below fails rather than that going unnoticed."""
    regex = tokenizers.Regex(_PANIC_PATTERN)
    splitter = tokenizers.pre_tokenizers.Split(regex, behavior="isolated")
    with pytest.raises(BaseException) as exc_info:
        splitter.pre_tokenize_str(_PANIC_INPUT)
    assert not isinstance(exc_info.value, Exception), (
        f"{type(exc_info.value).__name__} is now an Exception subclass - "
        "the BaseException-specific catch in _hf_tokenizer_probe.py's "
        "_check_one may no longer be necessary, but re-verify before removing it")


def test_re_rejects_the_real_gpt2_pattern_oniguruma_does_not():
    """Beyond danger-detection: a re-based validator would refuse to load
    real, unmodified, widely-shipped tokenizers outright."""
    with pytest.raises(re.error):
        re.compile(_GPT2_PATTERN)
    regex = tokenizers.Regex(_GPT2_PATTERN)   # must not raise
    splitter = tokenizers.pre_tokenizers.Split(regex, behavior="isolated")
    splitter.pre_tokenize_str("The quick brown fox jumps.")   # must not raise


# ---------------------------------------------------------------------------
#  _pattern_derived_probes: punctuation must not be filtered out (isalnum())
# ---------------------------------------------------------------------------

def test_pattern_derived_probes_includes_punctuation_characters():
    """Every distinct character a pattern names is a candidate for a derived
    probe, not only the alnum() ones - so a catastrophic-backtracking
    ambiguity keyed to a PUNCTUATION character, e.g. the literal comma in
    ``(,+)+b``, still gets a probe for that character."""
    from localm.inference._hf_tokenizer_probe import _pattern_derived_probes

    probes = _pattern_derived_probes(_PANIC_PATTERN_PUNCT)
    derived_chars = {p[0] for p in probes}
    assert "," in derived_chars, (
        f"comma never got a derived probe - derived chars were {derived_chars!r}; "
        "a punctuation-keyed catastrophic pattern is invisible to this layer")


def test_pattern_derived_probes_still_bounded_with_many_punctuation_characters():
    """_MAX_DERIVED_PROBE_CHARS bounds the probe COUNT regardless of how many
    distinct punctuation characters a pattern names."""
    from localm.inference._hf_tokenizer_probe import (
        _MAX_DERIVED_PROBE_CHARS, _pattern_derived_probes)

    # ASCII 33-47 is fifteen consecutive punctuation characters, none alnum.
    many_punct = "".join(chr(c) for c in range(33, 48))
    assert not any(ch.isalnum() for ch in many_punct)
    probes = _pattern_derived_probes(many_punct)
    assert len(probes) == _MAX_DERIVED_PROBE_CHARS, (
        f"got {len(probes)} probes for {len(many_punct)} distinct punctuation "
        f"characters - the _MAX_DERIVED_PROBE_CHARS={_MAX_DERIVED_PROBE_CHARS} "
        "count bound no longer holds")


# ---------------------------------------------------------------------------
#  _check_one (the probe subprocess's per-pattern check)
# ---------------------------------------------------------------------------

def test_check_one_catches_the_panic_and_returns_a_bad_verdict():
    """The panic must not escape _check_one - proves the BaseException catch
    the test above motivates is actually in place and working, called
    in-process here (safe: this pattern panics fast, it does not hang)."""
    verdict = _check_one(_PANIC_PATTERN)
    assert verdict.startswith("BAD "), f"expected a BAD verdict, got {verdict!r}"
    assert "retry-limit" in verdict or "Panic" in verdict


def test_check_one_catches_ambiguous_nested_quantifier_keyed_to_punctuation():
    """Same shape as the panic pattern above, keyed to a PUNCTUATION character
    rather than a letter. _FIXED_PROBES' own punctuation corpus mixes 16
    distinct characters, so no run of consecutive commas long enough to trip
    the ambiguity forms there; the derived probe is what catches it."""
    verdict = _check_one(_PANIC_PATTERN_PUNCT)
    assert verdict.startswith("BAD "), f"expected a BAD verdict, got {verdict!r}"
    assert "retry-limit" in verdict or "Panic" in verdict


def test_check_one_accepts_the_real_gpt2_pattern():
    assert _check_one(_GPT2_PATTERN) == "OK"


def test_check_one_rejects_invalid_regex_syntax():
    verdict = _check_one("(unclosed")
    assert verdict.startswith("BAD invalid regex")


# ---------------------------------------------------------------------------
#  validate_tokenizer_json - the orchestration layer
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _short_probe_timeout(monkeypatch):
    """Shortens the probe budget from the production 8s to 3s, which is still
    above every safe-pattern and panic-pattern measurement, so only the
    genuinely-hanging case is affected and only in how long it takes to time
    out."""
    monkeypatch.setattr(safety, "_PROBE_TIMEOUT_SECONDS", 3.0)


def test_no_tokenizer_json_is_a_noop(tmp_path):
    d = _model_dir(tmp_path, "legacy", skip_tokenizer_json=True)
    safety.validate_tokenizer_json(str(d))   # must not raise


def test_real_gpt2_pretokenizer_pattern_is_accepted(tmp_path):
    d = _model_dir(tmp_path, "gpt2", _with_pre_tokenizer_regex(_GPT2_PATTERN))
    _, elapsed = _timed(safety.validate_tokenizer_json, str(d))
    assert elapsed < 2.0, f"a safe, real pattern took {elapsed:.2f}s to validate"


def test_plain_string_pattern_is_not_treated_as_regex(tmp_path):
    """A Split step whose pattern is a plain string (not wrapped in
    tokenizers.Regex) is a literal-substring split, never compiled as a
    regex - must not be flagged."""
    d = _model_dir(tmp_path, "literal", {
        "pre_tokenizer": {"type": "Split", "pattern": {"String": "|"},
                          "behavior": "Isolated", "invert": False},
    })
    safety.validate_tokenizer_json(str(d))   # must not raise


@pytest.mark.parametrize("pattern, reason_substr", [
    (_UNTESTED_DIRECTION_PATTERN, "Oniguruma safety probe"),
    (_PANIC_PATTERN, "retry-limit"),
    (_PANIC_PATTERN_PUNCT, "retry-limit"),
])
def test_dangerous_pre_tokenizer_pattern_is_refused(tmp_path, pattern, reason_substr):
    d = _model_dir(tmp_path, "dangerous", _with_pre_tokenizer_regex(pattern))
    with pytest.raises(RuntimeError) as exc_info:
        safety.validate_tokenizer_json(str(d))
    message = str(exc_info.value)
    # The message names the offending pattern AND says what is wrong with it,
    # not a generic "rejected" with no detail.
    assert pattern[:20] in message
    assert reason_substr in message


def test_dangerous_pattern_nested_in_sequence_normalizer_is_found(tmp_path):
    """The generic whole-document walk, not a hand-picked set of top-level
    keys: a Regex buried inside normalizer.Sequence.normalizers (a real,
    verified tokenizers.json shape - see hf_tokenizer_safety.py's module
    docstring) must be found exactly like a top-level pre_tokenizer one."""
    d = _model_dir(tmp_path, "nested", {
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "NFC"},
            {"type": "Replace", "pattern": {"Regex": _PANIC_PATTERN}, "content": "X"},
        ]},
        "pre_tokenizer": {"type": "Whitespace"},
    })
    with pytest.raises(RuntimeError, match="retry-limit"):
        safety.validate_tokenizer_json(str(d))


def test_dangerous_pattern_in_decoder_is_found(tmp_path):
    """decoder.Replace is a real, verified shape too (decode() runs on every
    generated token's text) - the same generic walk must reach it."""
    d = _model_dir(tmp_path, "decoder_danger", {
        "pre_tokenizer": {"type": "Whitespace"},
        "decoder": {"type": "Replace", "pattern": {"Regex": _PANIC_PATTERN},
                   "content": "X"},
    })
    with pytest.raises(RuntimeError, match="retry-limit"):
        safety.validate_tokenizer_json(str(d))


def test_malformed_tokenizer_json_is_refused_not_silently_accepted(tmp_path):
    d = tmp_path / "malformed"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (d / "tokenizer.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="valid JSON"):
        safety.validate_tokenizer_json(str(d))


def test_oversized_pattern_is_refused_without_spawning_the_probe(tmp_path, monkeypatch):
    """An oversized pattern is rejected by the CHEAP length check alone: the
    subprocess spawn is patched to raise if called at all, so the pattern
    never reaches the probe subprocess."""
    def _must_not_be_called(patterns):
        raise AssertionError("the probe subprocess must not be spawned for an "
                              "oversized pattern")
    monkeypatch.setattr(safety, "_run_probe_subprocess", _must_not_be_called)

    oversized = "a" * (safety.MAX_TOKENIZER_REGEX_BYTES + 1)
    d = _model_dir(tmp_path, "oversized", _with_pre_tokenizer_regex(oversized))
    with pytest.raises(RuntimeError, match="byte limit"):
        safety.validate_tokenizer_json(str(d))


def test_a_pattern_the_probe_cannot_verify_is_treated_as_unsafe(tmp_path, monkeypatch):
    """If the probe subprocess returns FEWER verdicts than patterns submitted
    (a hang or a crash), the first unverified pattern is refused rather than
    treated as fine because nothing said BAD."""
    monkeypatch.setattr(safety, "_run_probe_subprocess", lambda patterns: [])
    d = _model_dir(tmp_path, "unverifiable", _with_pre_tokenizer_regex("a+b"))
    with pytest.raises(RuntimeError, match="did not pass the Oniguruma safety probe"):
        safety.validate_tokenizer_json(str(d))


def test_probe_spawn_failure_is_treated_as_unsafe_with_an_accurate_message(
        tmp_path, monkeypatch):
    """A DIFFERENT unverifiable case: the probe process could not be started at
    all (returns None, not a short list). Still refuses, and the message does
    NOT claim a timeout was waited out, since a spawn failure is immediate."""
    monkeypatch.setattr(safety, "_run_probe_subprocess", lambda patterns: None)
    d = _model_dir(tmp_path, "unspawnable", _with_pre_tokenizer_regex("a+b"))
    with pytest.raises(RuntimeError) as exc_info:
        safety.validate_tokenizer_json(str(d))
    message = str(exc_info.value)
    assert "could not be started" in message
    assert "within" not in message, (
        "a spawn failure did not wait out any timeout - the message must not "
        "imply it did")


def test_tokenizers_not_installed_defers_to_the_transformers_error(monkeypatch, tmp_path):
    """When `tokenizers` cannot be found, this function is a silent no-op and
    HFBackend.load()'s next step, _require_transformers(), produces the
    refusal instead. Only "tokenizers" is faked absent; every other lookup is
    delegated to the real find_spec, since this patches the process-wide
    importlib.util module."""
    real_find_spec = safety.importlib.util.find_spec

    def _find_spec(name, *a, **kw):
        if name == "tokenizers":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(safety.importlib.util, "find_spec", _find_spec)
    d = _model_dir(tmp_path, "no_tokenizers_pkg", _with_pre_tokenizer_regex(_PANIC_PATTERN))
    safety.validate_tokenizer_json(str(d))   # must not raise


# ---------------------------------------------------------------------------
#  HFBackend.load() integration - the actual gate
# ---------------------------------------------------------------------------

def test_hfbackend_load_refuses_dangerous_tokenizer_before_require_torch(tmp_path):
    """The real integration point: load() must fail via THIS gate, proven by no
    isolated worker process ever having been spawned, rather than failing
    somewhere inside load() for another reason.

    HFBackend is a thin parent-side proxy, so the tokenizer and model objects
    live in an isolated child process. ``backend._runner`` is only ever set (to
    a real HFRunner, which spawns that child) AFTER both pre-flight gates -
    _check_custom_code_allowed and this one - pass."""
    pytest.importorskip("transformers", exc_type=ImportError)
    from localm.inference.backends.hf import HFBackend

    d = _model_dir(tmp_path, "dangerous_model", _with_pre_tokenizer_regex(_PANIC_PATTERN))
    backend = HFBackend(str(d))
    with pytest.raises(RuntimeError, match="Oniguruma safety probe"):
        backend.load()
    assert backend._runner is None, "no worker process must ever have been spawned"
