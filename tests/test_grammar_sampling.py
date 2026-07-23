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
