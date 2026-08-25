# SPDX-License-Identifier: AGPL-3.0-or-later
"""The grammar sampler works; the generation loop must never double-accept."""

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
    """An invalid GBNF string makes llama_sampler_init_grammar return NULL."""
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
    """Same guard for the lazy path: a NULL from init_grammar_lazy_patterns must raise, not be added to the chain."""
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
    """A malformed grammar is a clean 400 up front (both stream and non-stream), never a silent unconstrained 200 and never a 500."""
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


def test_route_reports_worker_crash_during_grammar_check_as_a_fault_not_bad_grammar():
    """GitHub #964: a bare RuntimeError from engine.validate_grammar() means the isolated worker faulted (crashed, timed out, or replied unexpectedly - ModelRunner._simple_request has four such shapes), which has nothing to do with whether the caller's grammar is valid."""
    import os

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.validate_grammar.side_effect = RuntimeError(
        "The model process crashed (exit code -1073740791) while handling "
        "'check_grammar'.")
    engine.chat_stream.side_effect = AssertionError(
        "generation must not start when grammar validation itself faulted")

    client = TestClient(create_app(engine), raise_server_exceptions=True)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "grammar": 'root ::= "yes" | "no"',
        "max_tokens": 4,
    })
    assert r.status_code == 503, (r.status_code, r.text)
    detail = r.json()["detail"].lower()
    assert "invalid grammar" not in detail, (
        f"a worker crash must not be blamed on the caller's grammar: {r.text}")
    assert "worker" in detail or "crashed" in detail, r.text
    # Regression pin: only two of _simple_request's four RuntimeError shapes
    # actually kill the worker (see the route's comment) - "reload" must not
    # be promised unconditionally, since that claim is false for the other
    # two shapes and this same message text covers all of them.
    assert "reload" not in detail, (
        f"the message must not promise a reload it cannot guarantee for "
        f"every RuntimeError shape: {r.text}")
    engine.chat_stream.assert_not_called()


def test_completions_route_reports_worker_crash_during_grammar_check_as_a_fault():
    """Same fix, same shape, on /v1/completions - the two routes have byte-identical grammar-validation blocks and must not diverge."""
    import os

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.validate_grammar.side_effect = RuntimeError(
        "'check_grammar' timed out waiting for the model process.")
    engine.chat_stream.side_effect = AssertionError(
        "generation must not start when grammar validation itself faulted")

    client = TestClient(create_app(engine), raise_server_exceptions=True)
    r = client.post("/v1/completions", json={
        "model": "test-model",
        "prompt": "hi",
        "grammar": 'root ::= "yes" | "no"',
        "max_tokens": 4,
    })
    assert r.status_code == 503, (r.status_code, r.text)
    detail = r.json()["detail"].lower()
    assert "invalid grammar" not in detail, r.text
    assert "worker" in detail or "timed out" in detail, r.text
    assert "reload" not in detail, (
        f"the message must not promise a reload it cannot guarantee for "
        f"every RuntimeError shape: {r.text}")
    engine.chat_stream.assert_not_called()


def test_route_rejects_pathologically_nested_grammar_before_any_native_call():
    """LM-FZ-001 regression pin: a grammar built of deep unmatched-paren nesting (the exact minimized PoC that drove llama.cpp's native GBNF parser into a real stack overflow - caught exception in one run, a hard STATUS_ACCESS_VIOLATION process crash in another) must be rejected with a clean 400 by a pure-..."""
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
    """The structural guard must not be so tight it rejects real, legitimate grammars."""
    from localm.inference.gbnf import TOOL_CALLS_ONLY, check_grammar_structure

    check_grammar_structure(TOOL_CALLS_ONLY)  # must not raise

    moderately_nested = "root ::= " + "(" * 20 + '"leaf"' + ")" * 20
    check_grammar_structure(moderately_nested)  # must not raise


def test_check_grammar_structure_rejects_huge_repeat_count():
    """A `{100000}` repeat count is a second documented pathological shape (LM-FZ-001) distinct from paren-depth nesting - reject it too."""
    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import check_grammar_structure

    with pytest.raises(InvalidGrammarError):
        check_grammar_structure('root ::= "a"{100000}')


def test_check_grammar_structure_rejects_count_above_native_ceiling_margin():
    """MAX_GRAMMAR_REPEAT_COUNT must sit AT OR BELOW llama.cpp's real native GBNF parser ceiling (measured 1999 on the bundled runtime via a live check_grammar probe with a negative control - see gbnf.py's own comment), not merely below some arbitrary round number."""
    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import MAX_GRAMMAR_REPEAT_COUNT, check_grammar_structure

    assert MAX_GRAMMAR_REPEAT_COUNT <= 1999, (
        "the structural pre-check's ceiling must not sit above the measured "
        "native ceiling, or it cannot catch what the native parser rejects")

    with pytest.raises(InvalidGrammarError) as exc_info:
        check_grammar_structure('root ::= "a"{5000}')
    assert str(MAX_GRAMMAR_REPEAT_COUNT) in str(exc_info.value)


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
    """A single MALFORMED grammar must not disable grammar for later VALID requests."""
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
    """Shrink BOTH probe timeouts for tests that deliberately trigger a rejection, so the test does not need to wait out the production timeout to prove the same property - the spawn timeout matters too, since whichever test in the run happens to trigger the daemon's first spawn pays THAT bound, not the st..."""
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
    """The filter must not be so eager it rejects real, safe patterns - checked against every pattern the OTHER tests in this file rely on being accepted, so a regression here would show up as a false 400 on real traffic, not just fail its own test."""
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
    """The exact historical #928/#833 shape - a leading unanchored wildcard - must be caught here, for free, without ever touching the daemon."""
    from localm.inference.gbnf import _static_shape_rejection

    reason = _static_shape_rejection(r"[\s\S]*?(<tool_call>[\s\S]*)")
    assert reason is not None and "leading" in reason


def test_static_shape_rejection_catches_nested_quantifiers():
    """The other textbook catastrophic-backtracking shape, independent of this codebase's own history: a group with its own top-level quantifier, itself immediately re-quantified."""
    from localm.inference.gbnf import _static_shape_rejection

    for pattern in (r"(a+)+", r"(a*)*b", r"(x{2,})+"):
        reason = _static_shape_rejection(pattern)
        assert reason is not None, f"{pattern!r} should have been flagged as nested-quantified"


def test_static_shape_rejection_catches_oversized_pattern():
    """No length cap existed anywhere on a caller-supplied trigger pattern before it reached re.compile() in the isolated daemon (_trigger_probe.py) - found in cross-session review after the grammar_triggers fix landed."""
    from localm.inference.gbnf import MAX_TRIGGER_PATTERN_BYTES, _static_shape_rejection

    oversized = "a" * (MAX_TRIGGER_PATTERN_BYTES + 1)
    reason = _static_shape_rejection(oversized)
    assert reason is not None and (
        "length" in reason or "byte" in reason), reason


def test_static_shape_rejection_accepts_pattern_at_the_length_boundary():
    """The cap must not be off-by-one against itself: exactly MAX_TRIGGER_PATTERN_BYTES, in an otherwise-safe shape, must pass the length check (it may still legitimately fail some OTHER static check, but not this one - a benign flat literal run avoids that ambiguity)."""
    from localm.inference.gbnf import MAX_TRIGGER_PATTERN_BYTES, _static_shape_rejection

    at_boundary = "a" * MAX_TRIGGER_PATTERN_BYTES
    reason = _static_shape_rejection(at_boundary)
    assert reason is None, f"a pattern exactly at the cap was rejected: {reason}"


def test_oversized_pattern_rejected_without_reaching_the_probe():
    """The actual DoS-relevant property, proven directly (same technique as test_n_distinct_known_shape_attacks_never_reach_the_probe above): an oversized pattern must reject in CPU time that could only be explained by the cheap static filter, never by spawning or querying the daemon (which would cost at l..."""
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
    """The actual DoS-amplification fix, proven directly: N distinct catastrophic patterns (same historical shape, different literals so none share a cache entry) must reject in a time that could ONLY be explained by the static filter - the production probe timeout alone (seconds each) would make N of them..."""
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
    """Regression pin for a real bug caught in this session before it shipped: an earlier version concatenated ALL of a pattern's extracted characters into one repeated probe (e.g. 'aabaabaab...' for a pattern naming 'a' and 'b'), which defeats itself - the interleaved 'b' acts as a periodic terminator tha..."""
    from localm.inference._trigger_probe import _pattern_derived_probes

    probes = _pattern_derived_probes(r"(a|a)*b")
    # One probe per distinct character in the pattern (now every character,
    # not just alnum() ones - see the punctuation tests below), never fewer.
    assert len(probes) == len(set(r"(a|a)*b"))
    for probe in probes:
        assert len(set(probe)) == 1, (
            f"a derived probe must be a single character repeated, got "
            f"{len(set(probe))} distinct characters - interleaving defeats "
            "the alternation-ambiguity shape this probe exists to catch")


def test_pattern_derived_probes_catches_ambiguous_alternation_not_caught_by_static_filter(monkeypatch):
    """The other half of the composition: an ambiguous-alternation pattern like (a|a)*b is genuinely catastrophic (131.9s measured on 30 raw 'a' characters) but matches NEITHER static shape (no leading wildcard, and the group has no quantifier of its OWN - only alternation, which _static_shape_rejection do..."""
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


def test_pattern_derived_probes_includes_punctuation_characters():
    """The isalnum() filter (closed 2026-08-11) meant a pattern whose catastrophic-backtracking ambiguity is keyed to a PUNCTUATION character - e.g. the literal comma in ``(,|,)*b`` - never got a derived probe for that character at all, since isalnum() silently dropped it before the frequency count ever sa..."""
    from localm.inference._trigger_probe import _pattern_derived_probes

    probes = _pattern_derived_probes(r"(,|,)*b")
    derived_chars = {p[0] for p in probes}
    assert "," in derived_chars, (
        f"comma never got a derived probe - derived chars were {derived_chars!r}; "
        "a punctuation-keyed catastrophic pattern is invisible to this layer")


def test_pattern_derived_probes_catches_ambiguous_alternation_keyed_to_punctuation(monkeypatch):
    """Same defect class as test_..._not_caught_by_static_filter above, but keyed to a PUNCTUATION character rather than a letter - the isalnum() bypass this closes (2026-08-11): before the fix, _pattern_derived_probes silently dropped every non-alnum character from consideration, so a pattern whose ambigu..."""
    import time

    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import _static_shape_rejection, validate_trigger_patterns

    pattern = r"(,|,)*b"
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


def test_pattern_derived_probes_still_bounded_with_many_punctuation_characters():
    """Widening the character set to include punctuation must not create a new cost sink: _MAX_DERIVED_PROBE_CHARS still bounds probe COUNT regardless of how many distinct punctuation characters a pattern names - the internal per-pattern wall-clock budget (_PROBE_LOOP_BUDGET_SECONDS) is what bounds total c..."""
    from localm.inference._trigger_probe import _MAX_DERIVED_PROBE_CHARS, _pattern_derived_probes

    # ASCII 33-47 is fifteen consecutive punctuation characters, none alnum.
    many_punct = "".join(chr(c) for c in range(33, 48))
    assert not any(ch.isalnum() for ch in many_punct)
    probes = _pattern_derived_probes(many_punct)
    assert len(probes) == _MAX_DERIVED_PROBE_CHARS, (
        f"got {len(probes)} probes for {len(many_punct)} distinct punctuation "
        f"characters - the _MAX_DERIVED_PROBE_CHARS={_MAX_DERIVED_PROBE_CHARS} "
        "count bound no longer holds")


def test_validate_trigger_patterns_accepts_the_fixed_tool_call_trigger():
    """Dogfood check: localm's OWN production trigger pattern (post-#933) must pass its own validator - proves the validator is not so tight it rejects the exact pattern this codebase ships, and stands as a regression guard against a FUTURE dangerous pattern being reintroduced into TOOL_CALL_TRIGGER withou..."""
    from localm.inference.gbnf import TOOL_CALL_TRIGGER, validate_trigger_patterns

    validate_trigger_patterns([TOOL_CALL_TRIGGER])  # must not raise


def test_validate_trigger_patterns_accepts_a_different_legitimate_pattern():
    """The validator must not be tuned so narrowly that it only accepts localm's own pattern - a caller-supplied, differently-shaped but SAFE pattern must also pass."""
    from localm.inference.gbnf import validate_trigger_patterns

    validate_trigger_patterns([r"(<function_call>[\s\S]*)"])  # must not raise


def test_validate_trigger_patterns_rejects_the_old_catastrophic_pattern():
    """Regression pin: the EXACT pre-#933 TOOL_CALL_TRIGGER pattern - the one that actually crashed #928/#833 - must be rejected by the validator, not merely fixed in localm's own copy."""
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
    """A syntactically invalid pattern must be a clean typed rejection, not an unhandled re.error escaping to the caller."""
    from localm.inference.backends.base import InvalidGrammarError
    from localm.inference.gbnf import validate_trigger_patterns

    with pytest.raises(InvalidGrammarError, match="invalid regex"):
        validate_trigger_patterns(["(unclosed"])


def test_validate_trigger_patterns_caches_by_exact_pattern_string():
    """A pattern already validated safe in this process must not pay another daemon round-trip - proven by making a second validation of the SAME pattern complete near-instantly relative to a fresh (uncached) one, using a probe timeout short enough that a real round-trip would visibly cost more than a cach..."""
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
    """The API-facing sibling of test_route_rejects_pathologically_nested_ grammar_before_any_native_call above: a caller-supplied grammar_triggers pattern shaped like the actual #928/#833 defect must be rejected with a clean 400 by validate_trigger_patterns BEFORE it reaches the native sampler, proven by..."""
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
    """The positive case: a legitimate grammar_triggers pattern must NOT be blocked by the new validation - the request proceeds to generation exactly as it did before this defense existed."""
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


def test_probe_on_slot_joins_inflight_prewarm_without_duplicate_spawn():
    """PR #943's property, carried over to the per-slot probe pool: a caller that finds its slot empty must JOIN an in-flight pre-warm thread rather than start a duplicate spawn."""
    import localm.inference.gbnf as gbnf

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.stdout = MagicMock()
    fake_proc.stdin = MagicMock()

    spawn_called = False
    join_called = False
    start_prewarm = threading.Event()

    def mock_spawn():
        nonlocal spawn_called
        spawn_called = True
        return fake_proc

    slot = gbnf._ProbeSlot()

    with patch.object(gbnf, "_spawn_trigger_probe_daemon", side_effect=mock_spawn), \
         patch.object(gbnf, "_readline_with_timeout", return_value="OK"):

        def fake_prewarm():
            start_prewarm.wait()
            with slot.lock:
                slot.proc = fake_proc

        # daemon=True is not decoration: fake_prewarm blocks until join() sets
        # the event, so if the join under test is ever REMOVED the thread never
        # finishes, and a non-daemon thread blocks interpreter exit - pytest
        # would hang forever instead of reporting the failure. Found by running
        # exactly that break as a fires-control; the pre-existing version of
        # this test had the same shape.
        prewarm_thread = threading.Thread(target=fake_prewarm, daemon=True)
        slot.prewarm = prewarm_thread

        real_join = prewarm_thread.join
        def tracing_join(*a, **kw):
            nonlocal join_called
            join_called = True
            start_prewarm.set()
            return real_join(*a, **kw)

        prewarm_thread.join = tracing_join
        prewarm_thread.start()

        verdict, reason = gbnf._probe_on_slot(slot, r"^<tool_call>")

        assert verdict == gbnf._PROBE_SAFE, reason
        assert join_called is True, "join() must have been called on in-flight pre-warm thread"
        assert spawn_called is False, "should have joined in-flight prewarm instead of spawning duplicate"


#  Concurrency and admission control (the 2026-07-30 ruling's residual)
#
#  The ReDoS crash/hang vulnerability was closed by the static filter plus the
#  daemon probe. What the corrected arithmetic then identified as the REMAINING
#  cost was pure serialization:
#
#      18 x 2.0s steady-state probe timeout  = 36 s
#       2 x 10s daemon spawn                 = 20 s   (addressed by PR #943)
#
#  i.e. 36 of the 56 seconds were LEGITIMATE probe timeouts made serial by one
#  global lock held across the whole round trip. The ruling names the fix:
#  CONCURRENCY or ADMISSION CONTROL. These pin both.


class _HangingStream:
    """A stdout whose readline() never returns, which is EXACTLY what a catastrophic pattern does to the daemon: the process is alive and stuck inside re.search(), so the caller's own timeout is the only thing that can detect it (nothing inside one thread can interrupt a C-level match)."""

    def __init__(self) -> None:
        self._never = threading.Event()

    def readline(self):
        self._never.wait()      # never set; the reader thread is abandoned
        return ""


def _hanging_daemon() -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdout = _HangingStream()
    proc.stdin = MagicMock()
    return proc


def _fresh_pool(monkeypatch, size: int, waiters: "int | None" = None) -> None:
    """Replace the module-level pool with *size* fresh slots for one test."""
    import queue as _queue

    import localm.inference.gbnf as gbnf

    pool = _queue.LifoQueue()
    for _ in range(size):
        pool.put(gbnf._ProbeSlot())
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_POOL_SIZE", size)
    monkeypatch.setattr(gbnf, "_PROBE_SLOTS_FREE", pool)
    monkeypatch.setattr(gbnf, "_PROBE_WAITER_GATE", gbnf._WaiterGate())
    if waiters is not None:
        monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_MAX_WAITERS", waiters)


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll *predicate* to a deadline."""
    import time

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_concurrent_dangerous_patterns_do_not_serialize(monkeypatch):
    """THE residual, measured: N patterns that each hang their probe must cost about ceil(N / pool) timeouts of wall clock, not N of them."""
    import time

    import localm.inference.gbnf as gbnf

    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_TIMEOUT", 0.5)
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_SPAWN_TIMEOUT", 0.5)
    monkeypatch.setattr(gbnf, "_spawn_trigger_probe_daemon", _hanging_daemon)
    _fresh_pool(monkeypatch, 4)

    patterns = [f"^<hang_{i}_{time.time_ns()}>" for i in range(8)]
    verdicts: list = []
    lock = threading.Lock()

    def _one(p):
        v, _reason = gbnf._probe_pattern_is_safe(p)
        with lock:
            verdicts.append(v)

    threads = [threading.Thread(target=_one, args=(p,)) for p in patterns]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    # Every one of them was correctly judged dangerous - the concurrency must
    # not have been bought by skipping the check.
    assert verdicts == [gbnf._PROBE_UNSAFE] * 8, verdicts
    assert elapsed < 2.5, (
        f"8 hanging probes over a 4-slot pool took {elapsed:.2f}s; serial "
        "execution of 8 x 0.5s is ~4.0s, so they are still queueing behind "
        "each other")


def test_a_saturated_pool_refuses_fast_instead_of_queueing(monkeypatch):
    """ADMISSION CONTROL: a caller that cannot get a slot is refused promptly, and is refused as UNDETERMINED - a statement about the validator, not a verdict on its pattern."""
    import time

    import localm.inference.gbnf as gbnf

    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_TIMEOUT", 5.0)
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_SPAWN_TIMEOUT", 5.0)
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_SLOT_WAIT", 0.3)
    monkeypatch.setattr(gbnf, "_spawn_trigger_probe_daemon", _hanging_daemon)
    _fresh_pool(monkeypatch, 1)

    started = threading.Event()

    def _occupy():
        started.set()
        gbnf._probe_pattern_is_safe(f"^<occupier_{time.time_ns()}>")

    holder = threading.Thread(target=_occupy, daemon=True)
    holder.start()
    started.wait(timeout=5.0)
    time.sleep(0.15)     # let the holder actually take the only slot

    t0 = time.perf_counter()
    verdict, reason = gbnf._probe_pattern_is_safe(f"^<refused_{time.time_ns()}>")
    elapsed = time.perf_counter() - t0

    assert verdict == gbnf._PROBE_UNDETERMINED, (verdict, reason)
    assert "busy" in reason, reason
    # Refused after roughly the admission wait, NOT after the holder's 5.0s
    # probe timeout. The gap between those two is the whole point.
    assert elapsed < 2.0, (
        f"a refused caller waited {elapsed:.2f}s - it queued behind the "
        "in-flight probe instead of being admission-refused")
    holder.join(timeout=10.0)


def test_an_undetermined_verdict_is_never_cached(monkeypatch):
    """'I could not check this' must NEVER be remembered as 'this pattern is bad'."""
    import localm.inference.gbnf as gbnf
    from localm.inference.backends.base import TriggerValidatorUnavailableError

    # A FRESH, EMPTY pool is the precondition, not decoration. Patching
    # _spawn_trigger_probe_daemon simulates nothing unless a spawn is actually
    # needed, and _PROBE_SLOTS_FREE is module-level state that earlier tests in
    # the same run leave holding LIVE daemons - which answer "OK" without ever
    # reaching the patch. Measured: this test passed alone and failed in a
    # three-test selection for exactly that reason.
    _fresh_pool(monkeypatch, 2)

    pattern = r"^<undetermined_is_not_a_verdict>"
    gbnf._VALIDATED_TRIGGER_PATTERNS.pop(pattern, None)

    def _cannot_spawn():
        raise OSError("simulated: no subprocess available")

    with patch.object(gbnf, "_spawn_trigger_probe_daemon", side_effect=_cannot_spawn):
        with pytest.raises(TriggerValidatorUnavailableError):
            gbnf.validate_trigger_patterns([pattern])

    assert pattern not in gbnf._VALIDATED_TRIGGER_PATTERNS, (
        "a transient validator failure was cached as a verdict on the pattern")

    # And the proof that it matters: the identical pattern validates fine once
    # the validator is working again. Cached, this would raise forever.
    gbnf.validate_trigger_patterns([pattern])
    assert gbnf._VALIDATED_TRIGGER_PATTERNS[pattern] is None


def test_an_unsafe_verdict_is_still_cached(monkeypatch):
    """The other direction, so the fix above cannot be 'stop caching anything'."""
    import time

    import localm.inference.gbnf as gbnf
    from localm.inference.backends.base import InvalidGrammarError

    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_TIMEOUT", 0.5)
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_SPAWN_TIMEOUT", 0.5)
    monkeypatch.setattr(gbnf, "_spawn_trigger_probe_daemon", _hanging_daemon)
    _fresh_pool(monkeypatch, 2)

    pattern = f"^<proven_dangerous_{time.time_ns()}>"
    gbnf._VALIDATED_TRIGGER_PATTERNS.pop(pattern, None)

    with pytest.raises(InvalidGrammarError):
        gbnf.validate_trigger_patterns([pattern])
    assert gbnf._VALIDATED_TRIGGER_PATTERNS.get(pattern) is not None

    t0 = time.perf_counter()
    with pytest.raises(InvalidGrammarError):
        gbnf.validate_trigger_patterns([pattern])
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, (
        f"a re-sent proven-dangerous pattern cost {elapsed:.3f}s - it was "
        "re-probed instead of answered from the cache")


def test_route_answers_503_not_400_when_the_validator_cannot_check(monkeypatch):
    """WHO IS AT FAULT."""
    import os

    from fastapi.testclient import TestClient

    import localm.inference.gbnf as gbnf
    from localm.inference.http_server import create_app

    # See test_an_undetermined_verdict_is_never_cached: an empty pool is what
    # makes the spawn patch below bite at all.
    _fresh_pool(monkeypatch, 2)

    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.chat_stream.side_effect = AssertionError(
        "generation must not start on a pattern that was never validated")

    def _cannot_spawn():
        raise OSError("simulated: no subprocess available")

    pattern = r"^<route_503_probe>"
    gbnf._VALIDATED_TRIGGER_PATTERNS.pop(pattern, None)

    client = TestClient(create_app(engine), raise_server_exceptions=True)
    with patch.object(gbnf, "_spawn_trigger_probe_daemon", side_effect=_cannot_spawn):
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": 'root ::= "x"',
            "grammar_lazy": True,
            "grammar_triggers": [pattern],
            "max_tokens": 4,
        })

    assert r.status_code == 503, (r.status_code, r.text)
    assert "not checked" in r.text or "could not be" in r.text, r.text
    engine.chat_stream.assert_not_called()
    assert pattern not in gbnf._VALIDATED_TRIGGER_PATTERNS


def test_the_completions_route_also_answers_503(monkeypatch):
    """The same arm on /v1/completions."""
    import os

    from fastapi.testclient import TestClient

    import localm.inference.gbnf as gbnf
    from localm.inference.http_server import create_app

    _fresh_pool(monkeypatch, 2)

    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.chat_stream.side_effect = AssertionError(
        "generation must not start on a pattern that was never validated")

    def _cannot_spawn():
        raise OSError("simulated: no subprocess available")

    pattern = r"^<route_503_probe_completions>"
    gbnf._VALIDATED_TRIGGER_PATTERNS.pop(pattern, None)

    client = TestClient(create_app(engine), raise_server_exceptions=True)
    with patch.object(gbnf, "_spawn_trigger_probe_daemon", side_effect=_cannot_spawn):
        r = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "hi",
            "grammar": 'root ::= "x"',
            "grammar_lazy": True,
            "grammar_triggers": [pattern],
            "max_tokens": 4,
        })

    assert r.status_code == 503, (r.status_code, r.text)
    engine.chat_stream.assert_not_called()


def test_sequential_callers_keep_exactly_one_daemon_alive():
    """The pool must cost NOTHING in the ordinary case, and that is a claim about LIFO specifically, not about pools in general."""
    import localm.inference.gbnf as gbnf

    for i in range(6):
        verdict, reason = gbnf._probe_pattern_is_safe(rf"^<seq_daemon_probe_{i}>")
        assert verdict == gbnf._PROBE_SAFE, (verdict, reason)

    live = [s for s in list(gbnf._PROBE_SLOTS_FREE.queue)
            if s.proc is not None and s.proc.poll() is None]
    assert len(live) == 1, (
        f"{len(live)} daemons alive after 6 SEQUENTIAL probes - slots are not "
        "being reused, so a single-caller server pays for the whole pool")


def test_a_slot_is_returned_to_the_pool_even_when_the_probe_rejects(monkeypatch):
    """A leaked slot is a permanent, silent capacity loss: enough of them and every later request is refused as 'busy' forever, with nothing in the logs saying why."""
    import time

    import localm.inference.gbnf as gbnf

    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_TIMEOUT", 0.3)
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_SPAWN_TIMEOUT", 0.3)
    monkeypatch.setattr(gbnf, "_spawn_trigger_probe_daemon", _hanging_daemon)
    _fresh_pool(monkeypatch, 2)

    for _ in range(3):
        verdict, _reason = gbnf._probe_pattern_is_safe(f"^<leak_{time.time_ns()}>")
        assert verdict == gbnf._PROBE_UNSAFE

    assert gbnf._PROBE_SLOTS_FREE.qsize() == 2, (
        f"{2 - gbnf._PROBE_SLOTS_FREE.qsize()} slot(s) leaked after rejecting "
        "probes - the pool shrinks permanently on every rejection")


#  Bounding the WAITERS, not just the wait
#
#  _TRIGGER_PROBE_SLOT_WAIT bounds how long ONE caller queues. It says nothing
#  about how many callers may be queueing at the same time, and until
#  _TRIGGER_PROBE_MAX_WAITERS existed nothing else did: every waiter is a thread
#  parked on the asyncio loop's TRUE DEFAULT executor, which is shared with
#  engine.load, embedding, token counting and the isolated-runner RPCs. So an
#  unbounded queue here degrades work that has nothing to do with grammars.
#
#  These pin the cap in BOTH directions, because the easy way to get this wrong
#  is to bound the queue by accidentally bounding throughput as well.


def test_the_waiter_cap_leaves_room_for_the_pool_it_guards():
    """The RELATION between the two constants, asserted instead of the literal."""
    import localm.inference.gbnf as gbnf

    assert gbnf._TRIGGER_PROBE_MAX_WAITERS >= gbnf._TRIGGER_PROBE_POOL_SIZE, (
        f"waiter cap {gbnf._TRIGGER_PROBE_MAX_WAITERS} is below pool size "
        f"{gbnf._TRIGGER_PROBE_POOL_SIZE}: a burst the pool could absorb would "
        "be refused while its own slots were freeing up")


def test_a_caller_the_pool_can_serve_never_becomes_a_waiter(monkeypatch):
    """THE CAP MUST BOUND THE QUEUE, NEVER THE THROUGHPUT."""
    import localm.inference.gbnf as gbnf

    _fresh_pool(monkeypatch, 2, waiters=0)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.stdout = MagicMock()
    fake_proc.stdin = MagicMock()

    with patch.object(gbnf, "_spawn_trigger_probe_daemon", return_value=fake_proc), \
         patch.object(gbnf, "_readline_with_timeout", return_value="OK"):
        verdict, reason = gbnf._probe_pattern_is_safe(r"^<served_from_a_free_slot>")

    assert verdict == gbnf._PROBE_SAFE, (
        f"a caller with a free slot was refused ({verdict}: {reason}) with the "
        "waiter cap at 0 - it took a waiter permit it never needed, so the cap "
        "is bounding throughput rather than the queue")
    assert gbnf._PROBE_WAITER_GATE.waiting == 0, gbnf._PROBE_WAITER_GATE.waiting
    assert gbnf._PROBE_SLOTS_FREE.qsize() == 2


def test_a_full_waiter_queue_refuses_immediately_instead_of_parking_a_thread(monkeypatch):
    """THE LOAD-BEARING ONE: the refusal must come back in microseconds, NOT after _TRIGGER_PROBE_SLOT_WAIT."""
    import time

    import localm.inference.gbnf as gbnf

    # The holder's probe outlasts the slot wait, so the slot stays gone for the
    # whole test and the waiters really do wait.
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_TIMEOUT", 4.0)
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_SPAWN_TIMEOUT", 4.0)
    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_SLOT_WAIT", 3.0)
    monkeypatch.setattr(gbnf, "_spawn_trigger_probe_daemon", _hanging_daemon)
    _fresh_pool(monkeypatch, 1, waiters=2)

    gate = gbnf._PROBE_WAITER_GATE
    started = threading.Event()

    def _occupy():
        started.set()
        gbnf._probe_pattern_is_safe(f"^<occupier_{time.time_ns()}>")

    holder = threading.Thread(target=_occupy, daemon=True)
    holder.start()
    started.wait(timeout=5.0)
    assert _wait_until(lambda: gbnf._PROBE_SLOTS_FREE.qsize() == 0), (
        "the occupier never took the only slot, so nothing below is queueing")

    def _park(i):
        gbnf._probe_pattern_is_safe(f"^<waiter_{i}_{time.time_ns()}>")

    parked = [threading.Thread(target=_park, args=(i,), daemon=True) for i in range(2)]
    for t in parked:
        t.start()
    assert _wait_until(lambda: gate.waiting == 2), (
        f"only {gate.waiting} of 2 callers reached the queue, so the caller "
        "measured below is not actually arriving on a full one")

    t0 = time.perf_counter()
    verdict, reason = gbnf._probe_pattern_is_safe(f"^<over_capacity_{time.time_ns()}>")
    elapsed = time.perf_counter() - t0

    assert verdict == gbnf._PROBE_UNDETERMINED, (verdict, reason)
    assert "queue" in reason, (
        f"refused for capacity but reported the slot-wait reason: {reason}")
    assert elapsed < 0.25, (
        f"a caller arriving on a full queue took {elapsed:.3f}s to be refused; "
        "it parked a default-executor thread for the slot wait instead of "
        "being turned away at the gate")

    for t in parked:
        t.join(timeout=15.0)
    holder.join(timeout=20.0)
    assert gate.waiting == 0, (
        f"{gate.waiting} waiter permit(s) leaked - a leaked permit is a "
        "permanent capacity loss that refuses every later caller for a reason "
        "that has nothing to do with load")


def test_a_capacity_refusal_is_never_an_acceptance(monkeypatch):
    """NOTHING UNVALIDATED GETS THROUGH, on the new refusal path too."""
    import time

    import localm.inference.gbnf as gbnf
    from localm.inference.backends.base import TriggerValidatorUnavailableError

    monkeypatch.setattr(gbnf, "_TRIGGER_PROBE_SLOT_WAIT", 3.0)
    _fresh_pool(monkeypatch, 1, waiters=1)

    held = gbnf._PROBE_SLOTS_FREE.get_nowait()          # the pool is now empty
    assert gbnf._PROBE_WAITER_GATE.try_enter(1) is True  # and the queue is full

    def _must_not_spawn():
        raise AssertionError("a capacity-refused caller reached the daemon")

    pattern = f"^<capacity_refused_{time.time_ns()}>"
    gbnf._VALIDATED_TRIGGER_PATTERNS.pop(pattern, None)

    with patch.object(gbnf, "_spawn_trigger_probe_daemon", side_effect=_must_not_spawn):
        t0 = time.perf_counter()
        verdict, reason = gbnf._probe_pattern_is_safe(pattern)
        elapsed = time.perf_counter() - t0

        assert verdict != gbnf._PROBE_SAFE, (
            "a pattern that was never checked was returned as SAFE")
        assert verdict == gbnf._PROBE_UNDETERMINED, (verdict, reason)
        assert elapsed < 0.25, (
            f"refused after {elapsed:.3f}s - via the slot wait, not the gate")

        # And the same thing through the public entry point: a refusal for
        # capacity must raise the 503-shaped type, and must NOT be cached -
        # caching it would reject a good pattern for the life of the process
        # because the box was briefly busy.
        with pytest.raises(TriggerValidatorUnavailableError):
            gbnf.validate_trigger_patterns([pattern])
        assert pattern not in gbnf._VALIDATED_TRIGGER_PATTERNS, (
            "a transient capacity refusal was cached as a verdict on the pattern")

    gbnf._PROBE_WAITER_GATE.leave()
    gbnf._PROBE_SLOTS_FREE.put(held)


