# SPDX-License-Identifier: AGPL-3.0-or-later
"""The grammar sampler works; the generation loop must never double-accept.

llama_sampler_sample() already ACCEPTS the sampled token into every stateful
sampler in the chain (upstream documents it as "sample and accept"). The loop
used to call llama_sampler_accept() again after it, which advanced the grammar
parser twice per token until its parse stacks emptied and it threw
std::runtime_error across the C ABI (WinError 0xe06d7363) - misdiagnosed for
months as "the bundled build's grammar sampler faults". It also double-counted
every token in the repetition-penalty window.

The unit tests pin the no-double-accept contract with the DLL never loaded;
the @integration test proves grammar-constrained generation end to end on a
real model (skipped unless the native runtime + network are available)."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp.llama import LlamaCpp

_FAKES: list = []


def _bare_llama() -> LlamaCpp:
    llm = LlamaCpp.__new__(LlamaCpp)
    llm._n_ctx = 4096
    llm._n_ctx_max = None
    llm._n_ctx_grow = 4096
    llm._seed = 1234
    llm._verbose = False
    llm._model_ptr = 111
    llm._ctx_ptr = 222
    llm._tokenizer = MagicMock()
    llm._cached_tokens = []
    llm._ctx_capacity = 4096
    llm._kv_supported = None
    llm._gen_lock = threading.RLock()
    llm._inference_lock = threading.Lock()
    llm._stop = threading.Event()
    _FAKES.append(llm)
    return llm


@pytest.fixture(autouse=True)
def _null_fake_pointers():
    # Null the fake pointers before GC so __del__ -> close() does not pass a
    # bogus int to the real llama_free.
    yield
    for llm in _FAKES:
        llm._model_ptr = None
        llm._ctx_ptr = None
    _FAKES.clear()


def test_build_sampler_rejects_invalid_grammar_never_adds_null():
    """An invalid GBNF string makes llama_sampler_init_grammar return NULL. The
    builder must raise InvalidGrammarError (and free the half-built chain) rather
    than add NULL to the chain - adding NULL NULL-derefs at sample time, which the
    GGUF backend catches by LATCHING _grammar_unsupported, silently stripping
    grammar from every later request (the poisoning bug)."""
    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.backends.llamacpp import llama as L

    mock_api = MagicMock()
    mock_api.llama_sampler_chain_init.return_value = 500
    mock_api.llama_sampler_init_grammar.return_value = None  # NULL == parse failure

    with patch.object(L, "api", mock_api):
        with pytest.raises(InvalidGrammarError):
            L._build_sampler(vocab=1, grammar='root ::= "x" (((', temperature=0.0)

    # The NULL sampler was NEVER added to the chain, and the chain was freed.
    mock_api.llama_sampler_chain_add.assert_not_called()
    mock_api.llama_sampler_free.assert_called_once_with(500)


def test_build_sampler_rejects_invalid_lazy_grammar():
    """Same guard for the lazy path: a NULL from init_grammar_lazy_patterns must
    raise, not be added to the chain."""
    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.backends.llamacpp import llama as L

    mock_api = MagicMock()
    mock_api.llama_sampler_chain_init.return_value = 500
    mock_api.has_lazy_grammar.return_value = True
    mock_api.llama_sampler_init_grammar_lazy_patterns.return_value = None

    with patch.object(L, "api", mock_api):
        with pytest.raises(InvalidGrammarError):
            L._build_sampler(vocab=1, grammar="bad", grammar_lazy=True,
                             grammar_triggers=["<tool_call>"], temperature=0.0)

    mock_api.llama_sampler_free.assert_called_once_with(500)


def test_build_sampler_accepts_valid_grammar():
    """A valid grammar (non-NULL init) is added to the chain, no raise, no free."""
    from localm.inference.backends.llamacpp import llama as L

    mock_api = MagicMock()
    mock_api.llama_sampler_chain_init.return_value = 500
    mock_api.llama_sampler_init_grammar.return_value = 777  # non-NULL == parsed
    mock_api.has_penalties_sampler.return_value = False

    with patch.object(L, "api", mock_api):
        L._build_sampler(vocab=1, grammar='root ::= "x"', temperature=0.0)

    added = [c.args for c in mock_api.llama_sampler_chain_add.call_args_list]
    assert (500, 777) in added, "the valid grammar sampler must be added to the chain"


def test_route_rejects_invalid_grammar_with_400_not_silent_200():
    """A malformed grammar is a clean 400 up front (both stream and non-stream),
    never a silent unconstrained 200 and never a 500. The engine is never asked to
    generate, so a bad grammar cannot poison later requests."""
    import os

    from fastapi.testclient import TestClient

    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.http_server import create_app

    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.validate_grammar.side_effect = InvalidGrammarError(
        "invalid GBNF grammar (native parser rejected it)")
    # If generation were reached this would stream - it must NOT be reached.
    engine.chat_stream.side_effect = AssertionError("generation must not start on a bad grammar")

    client = TestClient(create_app(engine), raise_server_exceptions=True)
    for stream in (False, True):
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": "root ::= (((",
            "stream": stream,
            "max_tokens": 4,
        })
        assert r.status_code == 400, (stream, r.status_code, r.text)
        assert "grammar" in r.text.lower(), r.text
    engine.chat_stream.assert_not_called()


def test_route_rejects_pathologically_nested_grammar_before_any_native_call():
    """LM-FZ-001 regression pin: a grammar built of deep unmatched-paren nesting
    (the exact minimized PoC that drove llama.cpp's native GBNF parser into a
    real stack overflow - caught exception in one run, a hard
    STATUS_ACCESS_VIOLATION process crash in another) must be rejected with a
    clean 400 by a pure-Python structural check BEFORE it ever reaches
    engine.validate_grammar() - not merely turned into a 400 *after* a native
    call, which would still crash a real backend. Proven here by asserting
    engine.validate_grammar is never even invoked, so the check protects the
    RunnerBusy-deferred path too (which otherwise skips up-front validation
    entirely and lets a bad grammar reach the native sampler at generation
    time instead)."""
    import os

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.chat_stream.side_effect = AssertionError(
        "generation must not start on a pathologically nested grammar")

    client = TestClient(create_app(engine), raise_server_exceptions=True)
    # Exact minimized repro from the fuzzing engagement (LM-FZ-001).
    pathological = "root ::= " + "(" * 5000 + '"a"'
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "grammar": pathological,
        "max_tokens": 4,
    })
    assert r.status_code == 400, (r.status_code, r.text)
    assert "grammar" in r.text.lower(), r.text
    engine.validate_grammar.assert_not_called(), (
        "the structural check must reject this BEFORE any call reaches the "
        "engine/native layer, not merely translate a native failure into a 400")
    engine.chat_stream.assert_not_called()


def test_check_grammar_structure_accepts_realistic_deeply_nested_grammar():
    """The structural guard must not be so tight it rejects real, legitimate
    grammars. TOOL_CALLS_ONLY (localm's own production tool-call grammar) and a
    moderately nested JSON-schema-derived grammar (nesting comparable to a
    real multi-level object schema) must both pass."""
    from localm.inference.gbnf import TOOL_CALLS_ONLY, check_grammar_structure

    check_grammar_structure(TOOL_CALLS_ONLY)  # must not raise

    moderately_nested = "root ::= " + "(" * 20 + '"leaf"' + ")" * 20
    check_grammar_structure(moderately_nested)  # must not raise


def test_check_grammar_structure_rejects_huge_repeat_count():
    """A `{100000}` repeat count is a second documented pathological shape
    (LM-FZ-001) distinct from paren-depth nesting - reject it too."""
    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import check_grammar_structure

    with pytest.raises(InvalidGrammarError):
        check_grammar_structure('root ::= "a"{100000}')


def test_generate_never_calls_accept_after_sample():
    llm = _bare_llama()
    mock_api = MagicMock()
    mock_api.llama_sampler_sample.side_effect = [11, 12, 13]
    mock_api.llama_decode.return_value = 0
    # Real ctypes-backed batch so _create_batch's native fill loop runs (the
    # mock-detection facade was removed from production).
    from tests._fake_batch import fake_batch_init
    mock_api.llama_batch_init.side_effect = fake_batch_init
    llm._tokenizer.is_eog.side_effect = lambda t: t == 13

    with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=999), \
         patch.object(llm, "_fit_generation_budget", return_value=8), \
         patch.object(llm, "_can_reuse_kv", return_value=True), \
         patch.object(llm, "_prefill_with_reuse", return_value=None):
        tokens = list(llm._generate([1, 2, 3], max_new_tokens=8, temperature=0.8,
                                    top_k=40, top_p=0.95, repeat_penalty=1.1))

    assert tokens == [11, 12], "sampled tokens stream until the EOG token"
    # THE regression pin: sample() accepts internally; a second accept faults
    # the grammar sampler and double-counts the penalties window.
    mock_api.llama_sampler_accept.assert_not_called()
    mock_api.llama_sampler_free.assert_called_once_with(999)


# --------------------------------------------------------------------------- #
# Real-model proof (same gating pattern as test_gguf_smoke_integration.py).
# --------------------------------------------------------------------------- #

_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"


@pytest.mark.integration
@pytest.mark.real_gguf
def test_grammar_constrains_real_generation():
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned: {e}")
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_REPO, filename=_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_REPO}/{_FILE}: {e}")

    from localm.inference.backends.gguf import GgufBackend
    backend = GgufBackend(path, n_ctx=1024)
    backend.load()
    try:
        out = "".join(backend.chat_stream(
            [{"role": "user", "content": "Answer with one word, yes or no: "
                                         "is water wet?"}],
            max_tokens=8, temperature=0.0,
            grammar='root ::= "yes" | "no"',
        ))
        assert out in ("yes", "no"), f"grammar must constrain the output, got {out!r}"
        # The soft-degrade flag must NOT have been tripped: the constraint was
        # actually enforced, not silently dropped.
        assert not getattr(backend, "_grammar_unsupported", False)
    finally:
        backend.unload()


@pytest.mark.integration
@pytest.mark.real_gguf
def test_invalid_grammar_does_not_poison_later_valid_grammars():
    """A single MALFORMED grammar must not disable grammar for later VALID requests.

    Regression pin for the poisoning bug (live-confirmed): an invalid grammar used
    to NULL-deref the native sampler; the OSError handler latched
    _grammar_unsupported and thereafter stripped grammar from EVERY request. The
    fix rejects a bad grammar up front (InvalidGrammarError) so the latch never
    trips and valid grammars keep constraining."""
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned: {e}")
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_REPO, filename=_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_REPO}/{_FILE}: {e}")

    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.backends.gguf import GgufBackend

    VALID = 'root ::= "yes" | "no"'

    def constrained() -> str:
        return "".join(backend.chat_stream(
            [{"role": "user", "content": "Answer with one word, yes or no: "
                                         "is water wet?"}],
            max_tokens=8, temperature=0.0, grammar=VALID))

    backend = GgufBackend(path, n_ctx=1024)
    backend.load()
    try:
        # A valid grammar constrains.
        assert constrained() in ("yes", "no")
        # Up-front validation rejects a malformed grammar as a typed error...
        with pytest.raises(InvalidGrammarError):
            backend.validate_grammar('root ::= "yes" (((')
        # ...and even driving generation with a bad grammar raises cleanly rather
        # than latching the silent-degrade flag.
        with pytest.raises(InvalidGrammarError):
            list(backend.chat_stream(
                [{"role": "user", "content": "hi"}],
                max_tokens=8, temperature=0.0, grammar='root ::= "yes" ((('))
        assert not getattr(backend, "_grammar_unsupported", False), \
            "a bad grammar must NOT latch the global soft-degrade flag"
        # The valid grammar STILL constrains (not poisoned).
        assert constrained() in ("yes", "no")
    finally:
        backend.unload()


#  Caller-supplied grammar_triggers validation (GitHub #928, #833, #933)
#
#  #933 fixed localm's OWN hardcoded TOOL_CALL_TRIGGER, which turned out to be
#  catastrophically backtracking-prone. But grammar_triggers is documented,
#  caller-supplied public API (docs/server-api.md), and nothing validated an
#  arbitrary caller's own trigger pattern before it reached the identical
#  native std::regex path. validate_trigger_patterns (gbnf.py) closes that -
#  these tests pin it at both the unit level (the function itself) and the
#  route level (matching test_route_rejects_pathologically_nested_grammar_
#  before_any_native_call's shape above, for the sibling LM-FZ-001 guard).

def _fast_trigger_probe_timeout(monkeypatch):
    """Shrink BOTH probe timeouts for tests that deliberately trigger a
    rejection, so the test does not need to wait out the production timeout
    to prove the same property - the spawn timeout matters too, since
    whichever test in the run happens to trigger the daemon's first spawn
    pays THAT bound, not the steady-state one. Production values are
    unaffected outside the test."""
    import localm.inference.gbnf as gbnf
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_TIMEOUT", 0.5)
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_SPAWN_TIMEOUT", 0.5)


#  Static shape pre-filter (the cheap layer in front of the daemon probe)
#
#  Added after a live measurement showed the probe ALONE is a DoS-
#  amplification vector: rejecting a pattern costs the FULL probe timeout by
#  construction, and every rejection also kills and respawns the daemon, so
#  N distinct attacker-supplied patterns serialize behind one lock at close
#  to the SPAWN timeout each (MEASURED: ~10s per pattern with a 1s
#  steady-state timeout configured, because every rejection forces the next
#  check - attacker or legitimate - to pay the slow spawn path). These tests
#  pin that the cheap filter actually removes the known shapes from that
#  queue, and that it does not become a false-positive trap.

def test_static_shape_rejection_accepts_legitimate_patterns():
    """The filter must not be so eager it rejects real, safe patterns -
    checked against every pattern the OTHER tests in this file rely on
    being accepted, so a regression here would show up as a false 400 on
    real traffic, not just fail its own test."""
    from localm.inference.gbnf import TOOL_CALL_TRIGGER, _static_shape_rejection

    for pattern in (
        TOOL_CALL_TRIGGER,
        r"(<function_call>[\s\S]*)",
        r"^<tool_call>",
        r"(<start>.*<end>)",   # a wildcard INSIDE a group - not the nested-
                                # quantifier shape (the group has no *internal*
                                # top-level quantifier of its own)
    ):
        assert _static_shape_rejection(pattern) is None, (
            f"{pattern!r} was rejected by the cheap filter - false positive")


def test_static_shape_rejection_catches_the_928_pattern_shape():
    """The exact historical #928/#833 shape - a leading unanchored wildcard -
    must be caught here, for free, without ever touching the daemon."""
    from localm.inference.gbnf import _static_shape_rejection

    reason = _static_shape_rejection(r"[\s\S]*?(<tool_call>[\s\S]*)")
    assert reason is not None and "leading" in reason


def test_static_shape_rejection_catches_nested_quantifiers():
    """The other textbook catastrophic-backtracking shape, independent of
    this codebase's own history: a group with its own top-level quantifier,
    itself immediately re-quantified."""
    from localm.inference.gbnf import _static_shape_rejection

    for pattern in (r"(a+)+", r"(a*)*b", r"(x{2,})+"):
        reason = _static_shape_rejection(pattern)
        assert reason is not None, f"{pattern!r} should have been flagged as nested-quantified"


def test_static_shape_rejection_catches_oversized_pattern():
    """No length cap existed anywhere on a caller-supplied trigger pattern
    before it reached re.compile() in the isolated daemon (_trigger_probe.py)
    - found in cross-session review after the grammar_triggers fix landed.
    An uncapped pattern still costs real CPU before any adversarial-shape
    logic runs: _static_shape_rejection's own paren-depth scan, and (had it
    passed the static filter) _pattern_derived_probes' character-frequency
    scan in the daemon, are both O(len(pattern)); the pattern string is also
    the cache key in _VALIDATED_TRIGGER_PATTERNS, so an oversized pattern is
    retained in memory up to _MAX_CACHED_TRIGGER_PATTERNS times. Real trigger
    patterns are tiny (TOOL_CALL_TRIGGER is ~30 chars); MAX_TRIGGER_PATTERN_
    BYTES is far above any legitimate need while still bounding the above."""
    from localm.inference.gbnf import MAX_TRIGGER_PATTERN_BYTES, _static_shape_rejection

    oversized = "a" * (MAX_TRIGGER_PATTERN_BYTES + 1)
    reason = _static_shape_rejection(oversized)
    assert reason is not None and (
        "length" in reason or "byte" in reason), reason


def test_static_shape_rejection_accepts_pattern_at_the_length_boundary():
    """The cap must not be off-by-one against itself: exactly
    MAX_TRIGGER_PATTERN_BYTES, in an otherwise-safe shape, must pass the
    length check (it may still legitimately fail some OTHER static check,
    but not this one - a benign flat literal run avoids that ambiguity)."""
    from localm.inference.gbnf import MAX_TRIGGER_PATTERN_BYTES, _static_shape_rejection

    at_boundary = "a" * MAX_TRIGGER_PATTERN_BYTES
    reason = _static_shape_rejection(at_boundary)
    assert reason is None, f"a pattern exactly at the cap was rejected: {reason}"


def test_oversized_pattern_rejected_without_reaching_the_probe():
    """The actual DoS-relevant property, proven directly (same technique as
    test_n_distinct_known_shape_attacks_never_reach_the_probe above): an
    oversized pattern must reject in CPU time that could only be explained
    by the cheap static filter, never by spawning or querying the daemon
    (which would cost at least the production spawn timeout)."""
    import time

    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import MAX_TRIGGER_PATTERN_BYTES, validate_trigger_patterns

    oversized = "a" * (MAX_TRIGGER_PATTERN_BYTES * 4)
    t0 = time.process_time()
    with pytest.raises(InvalidGrammarError):
        validate_trigger_patterns([oversized])
    elapsed = time.process_time() - t0
    assert elapsed < 0.5, (
        f"rejecting an oversized pattern took {elapsed:.2f}s CPU time - "
        "the length cap is not actually preventing daemon/probe cost")


def test_n_distinct_known_shape_attacks_never_reach_the_probe():
    """The actual DoS-amplification fix, proven directly: N distinct
    catastrophic patterns (same historical shape, different literals so
    none share a cache entry) must reject in a time that could ONLY be
    explained by the static filter - the production probe timeout alone
    (seconds each) would make N of them take N times that.

    CPU time (time.process_time()), not wall-clock: an upper-bound elapsed
    assertion on a shared box is exactly the flake shape this repo already
    hit and fixed the same night in test_redos_bounds.py's Unit A tests
    (dev-notes/FIX-LOOP-2026-07-29.md) - contention inflates wall-clock
    without inflating the actual (tiny, pure-Python) CPU work this loop
    does, so a wall-clock upper bound can fail under load for reasons that
    have nothing to do with the property under test."""
    import time

    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import validate_trigger_patterns

    patterns = [r"[\s\S]*?(<attack_marker_%d>[\s\S]*)" % i for i in range(20)]
    t0 = time.process_time()
    for p in patterns:
        with pytest.raises(InvalidGrammarError):
            validate_trigger_patterns([p])
    elapsed = time.process_time() - t0
    assert elapsed < 0.5, (
        f"20 distinct known-shape attack patterns took {elapsed:.2f}s CPU time - "
        "the static filter is not actually preventing the probe/lock DoS")


def test_pattern_derived_probes_are_single_character_not_interleaved():
    """Regression pin for a real bug caught in this session before it
    shipped: an earlier version concatenated ALL of a pattern's extracted
    characters into one repeated probe (e.g. "aabaabaab..." for a pattern
    naming 'a' and 'b'), which defeats itself - the interleaved 'b' acts as
    a periodic terminator that resets the backtracking search space before
    it can blow up, so the dangerous (a|a)*b pattern completed in ~2ms
    against that probe despite taking 131.9s against 30 PURE 'a' characters
    with no 'b' at all. Each derived probe must be ONE character alone,
    repeated - never a mix."""
    from localm.inference._trigger_probe import _pattern_derived_probes

    probes = _pattern_derived_probes(r"(a|a)*b")
    assert len(probes) == 2   # distinct alnum chars: 'a' and 'b'
    for probe in probes:
        assert len(set(probe)) == 1, (
            f"a derived probe must be a single character repeated, got "
            f"{len(set(probe))} distinct characters - interleaving defeats "
            "the alternation-ambiguity shape this probe exists to catch")


def test_pattern_derived_probes_catches_ambiguous_alternation_not_caught_by_static_filter(monkeypatch):
    """The other half of the composition: an ambiguous-alternation pattern
    like (a|a)*b is genuinely catastrophic (131.9s measured on 30 raw 'a'
    characters) but matches NEITHER static shape (no leading wildcard, and
    the group has no quantifier of its OWN - only alternation, which
    _static_shape_rejection does not and cannot cheaply analyze). It must
    still be caught, by the probe layer, using a probe DERIVED from the
    pattern's own characters (see _pattern_derived_probes) - proving the
    two layers are not redundant, each catches what the other cannot."""
    import time

    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import _static_shape_rejection, validate_trigger_patterns

    pattern = r"(a|a)*b"
    assert _static_shape_rejection(pattern) is None, (
        "this pattern is the test's whole point BECAUSE the static filter "
        "cannot see it - if this assertion fails, pick a different pattern "
        "that still bypasses the static filter")

    _fast_trigger_probe_timeout(monkeypatch)
    t0 = time.perf_counter()
    with pytest.raises(InvalidGrammarError):
        validate_trigger_patterns([pattern])
    elapsed = time.perf_counter() - t0
    # Must have gone through the (slow) probe path, not the static one - the
    # inverse assertion from the DoS-fix tests above.
    assert elapsed > 0.05, (
        f"rejected in {elapsed:.3f}s - suspiciously fast for a pattern "
        "that should only be catchable via the daemon probe")


def test_validate_trigger_patterns_accepts_the_fixed_tool_call_trigger():
    """Dogfood check: localm's OWN production trigger pattern (post-#933) must
    pass its own validator - proves the validator is not so tight it rejects
    the exact pattern this codebase ships, and stands as a regression guard
    against a FUTURE dangerous pattern being reintroduced into
    TOOL_CALL_TRIGGER without anyone noticing."""
    from localm.inference.gbnf import TOOL_CALL_TRIGGER, validate_trigger_patterns

    validate_trigger_patterns([TOOL_CALL_TRIGGER])  # must not raise


def test_validate_trigger_patterns_accepts_a_different_legitimate_pattern():
    """The validator must not be tuned so narrowly that it only accepts
    localm's own pattern - a caller-supplied, differently-shaped but SAFE
    pattern must also pass. Proves this is a real safety check, not a
    disguised allowlist of one string."""
    from localm.inference.gbnf import validate_trigger_patterns

    validate_trigger_patterns([r"(<function_call>[\s\S]*)"])  # must not raise


def test_validate_trigger_patterns_rejects_the_old_catastrophic_pattern():
    """Regression pin: the EXACT pre-#933 TOOL_CALL_TRIGGER pattern - the one
    that actually crashed #928/#833 - must be rejected by the validator, not
    merely fixed in localm's own copy. Proves the general defense catches the
    specific historical defect, not just a synthetic example of it.

    Also a regression pin for the DoS-amplification fix: this exact shape
    (leading unanchored wildcard) must be caught by _static_shape_rejection,
    the cheap in-process layer, NOT the daemon probe - proven by requiring
    the rejection to complete near-instantly with NO monkeypatched timeout
    (the production timeouts are seconds; a fast completion here can only
    mean the probe was never reached). MEASURED live before this static
    layer existed: rejecting this exact pattern took ~10s (the probe's
    SPAWN timeout, not even the steady-state one, because every prior
    rejection kills and respawns the daemon) - and N distinct such patterns
    serialize behind the probe's single lock at that cost EACH, which is
    the DoS this static layer exists to remove from the queue entirely."""
    import time

    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import validate_trigger_patterns

    old_pattern = r"[\s\S]*?(<tool_call>[\s\S]*)"
    t0 = time.perf_counter()
    with pytest.raises(InvalidGrammarError, match="rejected"):
        validate_trigger_patterns([old_pattern])
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, (
        f"rejection took {elapsed:.3f}s - too slow to have gone through the "
        "static shape filter; this pattern must never reach the daemon probe")


def test_validate_trigger_patterns_rejects_invalid_regex_syntax():
    """A syntactically invalid pattern must be a clean typed rejection, not an
    unhandled re.error escaping to the caller."""
    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import validate_trigger_patterns

    with pytest.raises(InvalidGrammarError, match="invalid regex"):
        validate_trigger_patterns(["(unclosed"])


def test_validate_trigger_patterns_caches_by_exact_pattern_string():
    """A pattern already validated safe in this process must not pay another
    daemon round-trip - proven by making a second validation of the SAME
    pattern complete near-instantly relative to a fresh (uncached) one, using
    a probe timeout short enough that a real round-trip would visibly cost
    more than a cache hit but long enough not to flake under shared-box load."""
    import time

    import localm.inference.gbnf as gbnf

    pattern = r"(<a_pattern_unique_to_this_test>[\s\S]*)"
    gbnf._VALIDATED_TRIGGER_PATTERNS.pop(pattern, None)  # ensure a clean start

    gbnf.validate_trigger_patterns([pattern])  # first call: real probe round-trip
    assert pattern in gbnf._VALIDATED_TRIGGER_PATTERNS

    t0 = time.perf_counter()
    gbnf.validate_trigger_patterns([pattern])  # second call: cache hit
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, (
        f"cached validation took {elapsed:.3f}s - too slow for a dict lookup, "
        "the cache is not actually being hit")


def test_route_rejects_catastrophic_grammar_trigger_before_any_native_call(monkeypatch):
    """The API-facing sibling of test_route_rejects_pathologically_nested_
    grammar_before_any_native_call above: a caller-supplied grammar_triggers
    pattern shaped like the actual #928/#833 defect must be rejected with a
    clean 400 by validate_trigger_patterns BEFORE it reaches the native
    sampler, proven by asserting chat_stream is never called - not merely
    turned into a 400 after a native call/crash, which would still be able to
    hang or take down a real backend."""
    import os

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    _fast_trigger_probe_timeout(monkeypatch)
    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.chat_stream.side_effect = AssertionError(
        "generation must not start on a caller-supplied catastrophic trigger pattern")

    client = TestClient(create_app(engine), raise_server_exceptions=True)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "grammar": 'root ::= "x"',
        "grammar_lazy": True,
        "grammar_triggers": [r"[\s\S]*?(<tool_call>[\s\S]*)"],
        "max_tokens": 4,
    })
    assert r.status_code == 400, (r.status_code, r.text)
    assert "trigger" in r.text.lower(), r.text
    engine.chat_stream.assert_not_called()


def test_route_accepts_safe_grammar_trigger_and_reaches_generation():
    """The positive case: a legitimate grammar_triggers pattern must NOT be
    blocked by the new validation - the request proceeds to generation
    exactly as it did before this defense existed. Guards against the
    validator becoming a false-positive trap that would violate "keep the
    feature working" as much as removing it would have."""
    import os

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.count_tokens.return_value = 2   # _tokens_per_sec needs a real int

    def _fake_stream(*a, **kw):
        yield "ok"   # a real generator, not a plain iterator - the non-stream
                     # response path calls gen.close() on this in a finally block

    engine.chat_stream.side_effect = _fake_stream

    client = TestClient(create_app(engine), raise_server_exceptions=True)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "grammar": 'root ::= "x"',
        "grammar_lazy": True,
        "grammar_triggers": [r"(<function_call>[\s\S]*)"],
        "max_tokens": 4,
        "stream": False,
    })
    assert r.status_code == 200, (r.status_code, r.text)
    engine.chat_stream.assert_called_once()
