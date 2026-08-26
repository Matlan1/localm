# SPDX-License-Identifier: AGPL-3.0-or-later
"""
On-device GGUF embeddings (localm.inference.embedder) + engine.embed dispatch.

CI-safe unit tests cover model-path resolution, the engine dispatch to the
dedicated embedder (without loading the chat model), and graceful degradation
when no embedding model is available. The native GGUFEmbedder itself is exercised
by a real-model test gated on LOCALM_TEST_EMBED_MODEL (a path to an embedding
GGUF) + the real_gguf runtime gate, so it runs on a real machine and skips in CI.
"""

from __future__ import annotations

import logging
import math
import os
import re
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm.inference import embedder as emb


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    emb.reset_embedder()
    yield
    emb.reset_embedder()


def _cfg(monkeypatch, **overrides):
    base = {"embedding_model": "bge-small-en-v1.5", "n_gpu_layers": 99,
            "net_mode": "off"}
    base.update(overrides)
    monkeypatch.setattr("localm.config.load_config", lambda: dict(base))


# --------------------------------------------------------------------------- #
#  resolve_embedding_model_path                                                #
# --------------------------------------------------------------------------- #

def test_resolve_explicit_path(tmp_path, monkeypatch):
    f = tmp_path / "my-embed.gguf"
    f.write_bytes(b"GGUF stub")
    _cfg(monkeypatch, embedding_model=str(f))
    assert emb.resolve_embedding_model_path() == str(f)


def test_resolve_unknown_returns_none(monkeypatch):
    _cfg(monkeypatch, embedding_model="not-a-real-model-xyz")
    assert emb.resolve_embedding_model_path() is None


def test_resolve_known_missing_offline_returns_none(monkeypatch):
    # a known key, but net is off and it is not downloaded -> None (lexical fallback)
    _cfg(monkeypatch, embedding_model="bge-small-en-v1.5", net_mode="off")
    assert emb.resolve_embedding_model_path() is None


def test_resolve_empty_returns_none(monkeypatch):
    _cfg(monkeypatch, embedding_model="")
    assert emb.resolve_embedding_model_path() is None


def test_resolve_unknown_records_last_error(monkeypatch):
    """A spec matching nothing at all must not just return None - last_error()
    is the GUI's only channel for this (#949), so it must name the spec."""
    _cfg(monkeypatch, embedding_model="not-a-real-model-xyz")
    assert emb.resolve_embedding_model_path() is None
    err = emb.last_error() or ""
    assert "not-a-real-model-xyz" in err


def test_resolve_bad_path_spec_does_not_leak_the_account_name(tmp_path, monkeypatch):
    """Regression (2026-08-04): the 'not a path, a registered model, or a known
    key' message used to build the spec into last_error() with {spec!r}. repr()
    doubles backslashes in a Windows path, so pathscrub's literal-prefix match
    (and its regex backstop, which requires exactly one separator right after
    the drive letter) never recognised the escaped form - the account name
    survived scrubbing untouched. A spec that IS an absolute path under the
    user's home, but resolves to nothing, must not leak that account name."""
    fake_home = tmp_path / "home" / "someaccount"
    fake_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    bad_spec = str(fake_home / "models" / "nonexistent-embedding-model.bin")
    _cfg(monkeypatch, embedding_model=bad_spec)

    assert emb.resolve_embedding_model_path() is None
    err = emb.last_error() or ""
    assert "someaccount" not in err, (
        f"account name leaked through last_error(): {err!r}")


def test_resolve_registered_directory_is_reported_as_hf_not_gguf(tmp_path, monkeypatch):
    """localm's own model pull can register a HuggingFace-format embedding model
    as a DIRECTORY of shards (get_model_info's contract is 'exists', not 'is a
    loadable GGUF'). #949: this used to fall through to the generic 'not a path,
    a registered model, or a known key' message - indistinguishable from a name
    that was never found at all, even though the model genuinely was found. Must
    now say specifically it is a directory, not a GGUF, via last_error()."""
    hf_dir = tmp_path / "bge-large-en-v1.5"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")
    _cfg(monkeypatch, embedding_model="bge-large-en-v1.5")

    import localm.model_manager.registry as registry
    monkeypatch.setattr(registry, "get_model_info",
                        lambda name, **k: (str(hf_dir), None))

    assert emb.resolve_embedding_model_path() is None
    err = emb.last_error() or ""
    assert "directory" in err and "HuggingFace" in err
    assert "not a path, a registered model, or a known key" not in err


def test_resolve_registered_unc_path_is_never_statted(monkeypatch):
    """A registered entry whose stored path is UNC/device-shaped (a hand-edited
    registry, never a legitimate `localm pull`/`add`) must be refused the same
    way step 0 refuses a UNC embedding_model spec directly - BEFORE any
    filesystem call, never after. A real stat reaches the Windows SMB
    redirector and can block for minutes on an unroutable host or
    auto-authenticate against a reachable one.

    Uses a side-effect counter, NOT a raise, to detect a regression: the
    registry-lookup block this guard sits in is wrapped in a broad
    'except Exception: pass' (to tolerate a broken registry), which would
    silently swallow an AssertionError raised from inside is_file()/is_dir()
    and make the test pass whether or not the guard actually fired - proven by
    running this exact scenario against the pre-fix code before adding the
    guard, which showed exactly that false pass. The spy never touches the
    real filesystem/network either way, so the test cannot hang."""
    _cfg(monkeypatch, embedding_model="sneaky-entry")

    unc = r"\\attacker-host\share\fake.gguf"
    import localm.model_manager.registry as registry
    monkeypatch.setattr(registry, "get_model_info", lambda name, **k: (unc, None))

    stat_calls = []

    class _SpyPath:
        def is_file(self):
            stat_calls.append("is_file")
            return False

        def is_dir(self):
            stat_calls.append("is_dir")
            return False

    real_path = emb.Path

    def _spy_path(*a, **k):
        if a and a[0] == unc:
            return _SpyPath()
        return real_path(*a, **k)

    monkeypatch.setattr(emb, "Path", _spy_path)
    assert emb.resolve_embedding_model_path() is None
    assert stat_calls == [], f"a UNC-shaped registered path was statted: {stat_calls}"


def test_resolve_success_clears_prior_last_error(tmp_path, monkeypatch):
    """A fixed (or always-fine) config must not keep reporting a stale reason
    from an earlier, unrelated failed resolve."""
    _cfg(monkeypatch, embedding_model="not-a-real-model-xyz")
    assert emb.resolve_embedding_model_path() is None
    assert emb.last_error() is not None

    f = tmp_path / "my-embed.gguf"
    f.write_bytes(b"GGUF stub")
    _cfg(monkeypatch, embedding_model=str(f))
    assert emb.resolve_embedding_model_path() == str(f)
    assert emb.last_error() is None


def test_load_failure_survives_a_later_resolve_success_probe(tmp_path, monkeypatch):
    """Regression: GET /api/rag/embedding's status handler calls
    resolve_embedding_model_path() directly (the same call get_embedder() makes)
    purely to compute 'installed'. A file that resolves fine at the path level
    but is NOT actually a usable embedding model (the real #949-adjacent case:
    wrong pooling, corrupt file, or - as here - a build that lacks the
    embeddings API) must not have its LOAD failure explanation wiped by that
    later resolve-only probe just because the path itself still exists. Only
    reset_embedder() may clear a latched load failure."""
    f = tmp_path / "not-an-embedder.gguf"
    f.write_bytes(b"GGUF stub")
    _cfg(monkeypatch, embedding_model=str(f))

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("this llama.dll build does not expose the embeddings API")

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Boom)
    assert emb.get_embedder() is None
    assert "embeddings API" in (emb.last_error() or "")

    # The status-poll probe: same file, resolves fine at the path level.
    assert emb.resolve_embedding_model_path() == str(f)
    assert "embeddings API" in (emb.last_error() or ""), (
        "a resolve-only probe must not erase a real load failure")


def test_download_gated_by_net_policy_records_last_error(monkeypatch):
    """A known key that is not yet downloaded, with auto-download blocked by
    policy (net_mode off), still explains itself via last_error() - INFO-level
    in the log (an expected state, not a defect: see _download_known), but not
    silent to the GUI status endpoint that reads last_error()."""
    _cfg(monkeypatch, embedding_model="bge-small-en-v1.5", net_mode="off")
    assert emb.resolve_embedding_model_path() is None
    err = emb.last_error() or ""
    assert "bge-small-en-v1.5" in err and "not auto-downloading" in err


def test_policy_declined_download_does_not_clobber_a_real_load_failure(monkeypatch):
    """Regression (found live via tests/test_disclosure.py's own scrub test,
    which poisons _LAST_ERROR + _LOAD_FAILED_SPEC to simulate a genuine load
    failure): the SAME latched protection given to _record_resolve_success()
    must also cover _download_known's two policy-decline branches. A "not
    auto-downloading" verdict fires only when the known-key file is ALSO
    absent from disk - a state that cannot coexist with a REAL load failure
    for the SAME spec (loading requires the file to have been found first) -
    so while a load failure is latched for the CURRENTLY CONFIGURED spec, a
    policy decline for that same spec is not new evidence and must leave
    last_error() alone, exactly like a bare resolve success already does."""
    _cfg(monkeypatch, embedding_model="bge-small-en-v1.5", net_mode="ask")
    monkeypatch.setattr(emb, "_LOAD_FAILED_SPEC", "bge-small-en-v1.5")
    monkeypatch.setattr(
        emb, "_LAST_ERROR",
        "failed to load embedding model: /some/path.gguf: not an embedding model")

    assert emb.resolve_embedding_model_path() is None
    err = emb.last_error() or ""
    assert "not an embedding model" in err
    assert "not auto-downloading" not in err


def test_resolve_failure_for_the_same_spec_does_not_clobber_a_real_load_failure(
        monkeypatch):
    """The unified choke point (_set_resolve_outcome) means the SAME guard now
    covers _record_resolve_failure too, not just _record_resolve_success and
    _download_known's policy branches - added after the coordinator's review
    named this as the third, still-unguarded write site sharing the identical
    "same reason, sibling path, guard not carried across" shape. A resolve
    FAILURE for the SAME spec that already has a latched load failure is
    suppressed - it CAN happen (the file can vanish between the load and the
    next poll, verified live 2026-08-05; it is not the contradiction an
    earlier version of this guard's own docstring claimed - see
    test_resolve_failure_for_a_different_spec_is_not_suppressed below for the
    case that IS live information and must NOT be suppressed), but a caller
    who already knows this exact spec failed to load learns nothing new from
    "and now it cannot even be found", so the more specific load-failure
    reason wins."""
    _cfg(monkeypatch, embedding_model="totally-unrelated-typo-xyz")
    monkeypatch.setattr(emb, "_LOAD_FAILED_SPEC", "totally-unrelated-typo-xyz")
    monkeypatch.setattr(
        emb, "_LAST_ERROR",
        "failed to load embedding model: /some/path.gguf: not an embedding model")

    assert emb.resolve_embedding_model_path() is None
    err = emb.last_error() or ""
    assert "not an embedding model" in err


def test_resolve_failure_for_a_different_spec_is_not_suppressed(monkeypatch):
    """Regression for the #1026 over-guard (coordinator's audit, 2026-08-04/05):
    the previous bare-bool ``_LOAD_FAILED`` choke point suppressed ANY
    resolve-side write while ANY load failure was latched, with no spec
    identity - so a currently-configured, ACTIVELY REFUSED spec (e.g. a UNC
    path just typed into embedding_model, or - as tested here - any other
    live resolve failure) had its own outcome silently swallowed, and
    last_error() kept reporting an old, unrelated reason instead. Confirmed
    live before this fix: resolve_embedding_model_path() DID call
    _record_resolve_failure for the new spec (proven via call-count
    instrumentation, not just the suppressed value), but last_error() never
    changed. A latched failure for spec A must never suppress a live resolve
    outcome for a DIFFERENT, currently-configured spec B."""
    _cfg(monkeypatch, embedding_model="totally-unrelated-typo-xyz")
    monkeypatch.setattr(emb, "_LOAD_FAILED_SPEC", "an-old-abandoned-spec.gguf")
    monkeypatch.setattr(
        emb, "_LAST_ERROR",
        "failed to load embedding model: /some/old-path.gguf: not an embedding model")

    assert emb.resolve_embedding_model_path() is None
    err = emb.last_error() or ""
    assert "totally-unrelated-typo-xyz" in err, (
        f"a live resolve failure for the currently-configured spec was "
        f"suppressed by a stale latch for a different, abandoned spec: {err!r}")
    assert "old-path.gguf" not in err


def test_resolve_failure_warns_once_then_quiets(monkeypatch, caplog):
    """resolve_embedding_model_path re-runs on every embed_texts() call while no
    embedder is loaded (get_embedder never caches a missing-model result). An
    UNCHANGED misconfiguration must warn once, not flood the log on every call -
    but last_error() must still carry the reason every time (checked above)."""
    _cfg(monkeypatch, embedding_model="not-a-real-model-xyz")
    caplog.set_level(logging.DEBUG, logger="localm")

    for _ in range(3):
        assert emb.resolve_embedding_model_path() is None

    matching = [r for r in caplog.records if "not-a-real-model-xyz" in r.getMessage()]
    warnings = [r for r in matching if r.levelno == logging.WARNING]
    debugs = [r for r in matching if r.levelno == logging.DEBUG]
    assert len(warnings) == 1
    assert len(debugs) == 2


# --------------------------------------------------------------------------- #
#  engine.embed dispatch                                                       #
# --------------------------------------------------------------------------- #

def _engine_with_backend(monkeypatch, backend):
    import localm.inference.engine as eng_mod
    monkeypatch.setattr(eng_mod, "create_backend", lambda *a, **k: backend)
    return eng_mod.Engine("dummy-model")


def test_engine_embed_dispatches_to_dedicated_embedder(monkeypatch):
    # a GGUF-style backend that cannot embed the chat model
    loaded = {"v": False}
    backend = types.SimpleNamespace(
        can_embed=False, loaded=False,
        load=lambda: loaded.__setitem__("v", True),
        embed=lambda t: (_ for _ in ()).throw(AssertionError("must not call backend.embed")))
    monkeypatch.setattr(emb, "embed_texts", lambda texts: [[1.0, 2.0]] * len(texts))
    engine = _engine_with_backend(monkeypatch, backend)
    out = engine.embed(["a", "b"])
    assert out == [[1.0, 2.0], [1.0, 2.0]]
    assert loaded["v"] is False              # the chat model was NOT loaded


def test_engine_embed_raises_when_no_embedder(monkeypatch):
    backend = types.SimpleNamespace(can_embed=False, loaded=False, load=lambda: None,
                                    embed=lambda t: [])
    monkeypatch.setattr(emb, "embed_texts", lambda texts: None)
    engine = _engine_with_backend(monkeypatch, backend)
    with pytest.raises(NotImplementedError):
        engine.embed(["x"])


def test_engine_embed_uses_backend_when_it_can_embed(monkeypatch):
    backend = types.SimpleNamespace(can_embed=True, loaded=True, load=lambda: None,
                                    embed=lambda t: [[9.0]] * len(t))
    # embed_texts must NOT be used when the backend can embed
    monkeypatch.setattr(emb, "embed_texts",
                        lambda texts: (_ for _ in ()).throw(AssertionError("no")))
    engine = _engine_with_backend(monkeypatch, backend)
    assert engine.embed(["x", "y"]) == [[9.0], [9.0]]


# --------------------------------------------------------------------------- #
#  A backend that only learns its embedding capability at LOAD time (HF)       #
# --------------------------------------------------------------------------- #

class _LoadTimeCapabilityBackend:
    """Mirrors HFBackend: whether the model is a genuine embedder is UNKNOWN
    until it is loaded, so ``can_embed`` answers True (= "load to find out")
    while unloaded and only tells the truth once the weights are in."""

    def __init__(self, embeds_for_real: bool):
        self._embeds_for_real = embeds_for_real
        self._loaded = False
        self.backend_embed_calls = 0

    @property
    def can_embed(self) -> bool:
        if not self._loaded:
            return True
        return self._embeds_for_real

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def embed(self, texts):
        self.backend_embed_calls += 1
        return [[7.0]] * len(texts)


def test_engine_embed_rechecks_can_embed_after_load(monkeypatch):
    """A chat decoder must NOT self-embed, even though it looked capable before load.

    The HF backend cannot know whether its model is a genuine embedder until the
    model is loaded (a causal LM's mean-pooled hidden states are not embeddings -
    see HFBackend.can_embed). Engine.embed therefore has to re-check can_embed
    AFTER the load. Checking only BEFORE it (the old behaviour) let a loaded chat
    decoder fall through to backend.embed() and silently return unusable vectors
    to /v1/embeddings and RAG.
    """
    backend = _LoadTimeCapabilityBackend(embeds_for_real=False)
    monkeypatch.setattr(emb, "embed_texts", lambda texts: [[0.5]] * len(texts))
    engine = _engine_with_backend(monkeypatch, backend)

    assert engine.embed(["a", "b"]) == [[0.5], [0.5]]
    assert backend.backend_embed_calls == 0     # never self-embedded


def test_engine_embed_uses_backend_that_can_embed_after_load(monkeypatch):
    """The mirror case: a genuine HF embedding model (an encoder, or one exposing
    .encode()) still embeds with the backend itself once loaded. The re-check must
    not push a real embedder onto the dedicated one."""
    backend = _LoadTimeCapabilityBackend(embeds_for_real=True)
    monkeypatch.setattr(emb, "embed_texts",
                        lambda texts: (_ for _ in ()).throw(
                            AssertionError("a real HF embedder must not be bypassed")))
    engine = _engine_with_backend(monkeypatch, backend)

    assert engine.embed(["x"]) == [[7.0]]
    assert backend.backend_embed_calls == 1


def test_engine_embed_chat_decoder_without_embedder_raises(monkeypatch):
    """No dedicated embedder + a chat decoder -> the actionable error, NOT the chat
    model's own vectors. Silently returning unusable vectors is the defect (rule 5);
    this is the same path the GGUF backend already takes, and RAG catches it and
    degrades to lexical-only with a warning."""
    backend = _LoadTimeCapabilityBackend(embeds_for_real=False)
    monkeypatch.setattr(emb, "embed_texts", lambda texts: None)
    engine = _engine_with_backend(monkeypatch, backend)

    with pytest.raises(NotImplementedError, match="setup-embeddings"):
        engine.embed(["x"])
    assert backend.backend_embed_calls == 0


# --------------------------------------------------------------------------- #
#  HFWorker.can_embed - honest capability reporting                            #
# --------------------------------------------------------------------------- #
# Tests HFWorker (_hf_worker.py), not the HFBackend proxy (hf.py): the
# can_embed COMPUTATION logic below runs only in the isolated child process
# now (see the thread-pool-exhaustion fix) - HFBackend.can_embed just returns
# a value this same logic already computed once at load time and cached. The
# proxy's own "unloaded -> True" contract is covered separately by
# test_hf_embed_integration.py::test_unloaded_hf_backend_reports_unknown_as_capable.

def _hf_backend(model=None):
    from localm.inference.backends._hf_worker import HFWorker
    # Safe today ONLY because __init__ and can_embed have zero torch/
    # transformers imports (checked here, not just assumed - see
    # _hf_worker.py). _hf_worker.py's load/embed/chat_stream methods DO have
    # unguarded function-local torch/transformers imports - production code
    # relies entirely on process isolation (HFWorker only ever runs in a
    # spawned child) for safety, not a runtime check. If this helper starts
    # calling one of THOSE methods, it needs the same native_lib_loaded()
    # guard as test_hf_prompt_tokenization.py/test_hf_runner_isolation.py.
    be = HFWorker("does-not-need-to-exist")
    be._model = model
    return be


def test_hf_can_embed_false_for_generative_decoder():
    """A causal/chat LM reports can_embed=False: mean-pooling its last hidden
    states yields vectors that cannot separate related from unrelated text
    (measured 2026-07-15: Qwen2.5-0.5B max-unrelated cosine 0.7523 EXCEEDS its
    min-related 0.7518), so they must never stand in for a real embedder."""
    assert _hf_backend(types.SimpleNamespace(can_generate=lambda: True)).can_embed is False


def test_hf_can_embed_true_for_encoder():
    """A non-generative encoder (AutoModel/BERT-family) is a legitimate embedder:
    mean-pooling its last hidden states is the standard recipe."""
    model = types.SimpleNamespace(
        can_generate=lambda: False,
        config=types.SimpleNamespace(architectures=["BertModel"]))
    assert _hf_backend(model).can_embed is True


def test_hf_can_embed_trusts_the_declared_arch_over_the_loaded_class():
    """A REAL encoder checkpoint answers can_generate() True, so the declared
    architecture - not the loaded class - has to decide.

    load() tries AutoModelForCausalLM BEFORE AutoModel, and transformers registers
    the encoder families as causal LMs (5.12.1: bert -> BertLMHeadModel, roberta,
    xlm-roberta, electra). So bge-small / all-MiniLM / e5, which declare
    ["BertModel"], load as BertLMHeadModel and report can_generate() True while
    being perfectly good embedders. Reading can_generate() alone therefore
    misroutes localm's OWN default embedding model to the dedicated embedder (or
    to a 422 when none is installed). Pins the real shape: can_generate() True but
    a non-generative DECLARED architecture -> still an embedder.
    """
    model = types.SimpleNamespace(
        can_generate=lambda: True,                       # what BertLMHeadModel says
        config=types.SimpleNamespace(architectures=["BertModel"]))
    assert _hf_backend(model).can_embed is True


def test_hf_can_embed_false_for_declared_causal_lm():
    """The mirror: a chat checkpoint declares a generative architecture, so it is
    not an embedder even though nothing else about the object says so."""
    model = types.SimpleNamespace(
        can_generate=lambda: True,
        config=types.SimpleNamespace(architectures=["Qwen2ForCausalLM"]))
    assert _hf_backend(model).can_embed is False


def test_hf_can_embed_falls_back_when_nothing_is_declared():
    """No declared architecture -> the loaded class's own answer is all there is."""
    no_arch = types.SimpleNamespace(architectures=None)
    assert _hf_backend(types.SimpleNamespace(
        can_generate=lambda: True, config=no_arch)).can_embed is False
    assert _hf_backend(types.SimpleNamespace(
        can_generate=lambda: False, config=no_arch)).can_embed is True


def test_hf_can_embed_covers_the_generative_head_names():
    """Every generative task head transformers names is caught, and the bare
    encoder ``*Model`` names are not. (The suffix list itself is pinned against
    transformers' own GenerationMixin by the integration test.)"""
    def _embeds(arch):
        return _hf_backend(types.SimpleNamespace(
            can_generate=lambda: True,
            config=types.SimpleNamespace(architectures=[arch]))).can_embed

    for arch in ("Qwen2ForCausalLM", "GPT2LMHeadModel", "BertLMHeadModel",
                 "T5ForConditionalGeneration", "LlamaForCausalLM"):
        assert _embeds(arch) is False, f"{arch} generates; it is not an embedder"
    for arch in ("BertModel", "XLMRobertaModel", "NomicBertModel",
                 "T5EncoderModel", "DistilBertModel", "MPNetModel"):
        assert _embeds(arch) is True, f"{arch} is an encoder; it embeds"


def test_hf_can_embed_never_imports_transformers(monkeypatch):
    """can_embed must not drag in transformers (hence torch).

    Importing torch in a process that already loaded the bundled llama.dll dies
    with OSError [WinError 127] (rocm_sdk.preload_libraries; reproduced
    2026-07-15) - and an import guard would swallow that and answer WRONGLY,
    which is how the encoder case regressed. It never needs the import: it only
    runs on an already-loaded model. Fails the import outright to prove the
    property never reaches for it.
    """
    import builtins
    real_import = builtins.__import__

    def _no_transformers(name, *a, **k):
        if name.split(".")[0] == "transformers":
            raise AssertionError(f"can_embed must not import {name}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_transformers)
    model = types.SimpleNamespace(
        can_generate=lambda: True,
        config=types.SimpleNamespace(architectures=["BertModel"]))
    assert _hf_backend(model).can_embed is True


def test_hf_can_embed_true_for_sentence_transformer():
    """A sentence-transformer exposes .encode(); that is a purpose-built embedding
    path and wins regardless of what can_generate() says."""
    model = types.SimpleNamespace(encode=lambda t, **k: [], can_generate=lambda: True)
    assert _hf_backend(model).can_embed is True


def test_hf_can_embed_unknown_before_load_is_true():
    """Unloaded, the capability is genuinely unknown. Answer True so callers still
    load the model and find out (routes/chat.py force-loads on this), rather than
    silently skipping a real HF embedding model. Engine.embed re-checks after load."""
    assert _hf_backend(None).can_embed is True


def test_hf_can_embed_false_when_capability_unprovable():
    """An exotic model object that does not answer can_generate() is NOT proof of
    an embedder. Fail towards the dedicated embedder rather than silently pooling
    something that may be a chat decoder (rule 5: never assume 'probably fine')."""
    class _Odd:
        def can_generate(self):
            raise RuntimeError("no idea")
    assert _hf_backend(_Odd()).can_embed is False


# --------------------------------------------------------------------------- #
#  Pooling: honour the model, never silently mis-pool it                       #
# --------------------------------------------------------------------------- #

def test_pooling_setting_resolution():
    """Every documented embedding_pooling choice maps to its llama.cpp value."""
    assert emb.resolve_pooling_setting("mean") == emb._POOLING_MEAN
    assert emb.resolve_pooling_setting("cls") == emb._POOLING_CLS
    assert emb.resolve_pooling_setting("last") == emb._POOLING_LAST
    assert emb.resolve_pooling_setting("none") == emb._POOLING_NONE
    assert emb.resolve_pooling_setting("auto") == emb.POOLING_AUTO
    assert emb.resolve_pooling_setting("  LAST ") == emb._POOLING_LAST   # tolerant


def test_pooling_setting_defaults_to_unset():
    """Unset/blank/bogus all resolve to the internal UNSET sentinel - resolved
    per-model at _effective_pooling time, not pinned to a bare int here. A
    BOGUS value must not fail the load, but must not pass silently either (it
    is logged by resolve_*)."""
    assert emb.resolve_pooling_setting(None) == emb._POOLING_UNSET
    assert emb.resolve_pooling_setting("") == emb._POOLING_UNSET
    assert emb.resolve_pooling_setting("nonsense") == emb._POOLING_UNSET


def test_default_pooling_is_mean_not_the_declared_type():
    """Guards the deliberate choice NOT to unconditionally follow the model by
    default.

    bge-small (the default embedder) declares CLS, yet every existing index was
    built with MEAN at the same 384 dims. Following the declaration by default
    would silently invalidate those indexes with no dim guard to catch it, so
    MEAN stays the default for CLS/unspecified declarations - only a model
    declaring LAST specifically (see test_unset_default_resolves_last_for_a_
    last_declaring_model) gets the correction, since there is no existing
    mean-built index of THAT shape to protect."""
    assert emb.resolve_pooling_setting(
        emb_config_default("embedding_pooling")) == emb._POOLING_UNSET
    assert emb._effective_pooling(emb._POOLING_UNSET, emb._POOLING_CLS) == emb._POOLING_MEAN
    assert emb._effective_pooling(emb._POOLING_UNSET, None) == emb._POOLING_MEAN


def test_unset_default_resolves_last_for_a_last_declaring_model():
    """The fix: nothing configured (the common case - no one manually sets
    embedding_pooling) + a model that declares LAST (a decoder-based embedder
    like Qwen3-Embedding) now correctly resolves to LAST, not MEAN - so it
    works out of the box with no setting to discover."""
    assert emb._effective_pooling(emb._POOLING_UNSET, emb._POOLING_LAST) == emb._POOLING_LAST


def emb_config_default(key):
    from localm.config import DEFAULT_CONFIG
    return DEFAULT_CONFIG[key]


def test_auto_honours_the_declared_pooling():
    """auto = use what the GGUF declares. Qwen3-Embedding declares LAST (verified
    2026-07-15: qwen3.pooling_type=3); forcing MEAN on it is the defect."""
    assert emb._effective_pooling(emb.POOLING_AUTO, emb._POOLING_LAST) == emb._POOLING_LAST
    assert emb._effective_pooling(emb.POOLING_AUTO, emb._POOLING_CLS) == emb._POOLING_CLS
    assert emb._effective_pooling(emb.POOLING_AUTO, emb._POOLING_MEAN) == emb._POOLING_MEAN


def test_declared_unspecified_reads_as_not_declared(monkeypatch):
    """A GGUF that declares UNSPECIFIED (-1) has declared nothing usable; it must
    read the same as an absent key so auto falls back to MEAN."""
    class _Api:
        @staticmethod
        def has_model_meta_api():
            return True

        @staticmethod
        def llama_model_meta_val_str(model, key):
            return {"general.architecture": "bert",
                    "bert.pooling_type": "-1"}.get(key)

    assert emb.declared_pooling_type(object(), _Api) is None


def test_declared_pooling_survives_a_stripped_or_broken_dll():
    """No metadata API, or a junk value, must not fail an otherwise fine load -
    the caller just keeps its configured pooling (debug-logged, never silent)."""
    class _NoMeta:
        @staticmethod
        def has_model_meta_api():
            return False

    class _Junk:
        @staticmethod
        def has_model_meta_api():
            return True

        @staticmethod
        def llama_model_meta_val_str(model, key):
            return "bert" if key == "general.architecture" else "not-an-int"

    assert emb.declared_pooling_type(object(), _NoMeta) is None
    assert emb.declared_pooling_type(object(), _Junk) is None


def test_auto_falls_back_to_mean_when_nothing_usable_is_declared():
    """A model declaring nothing (gte-Qwen2, chat GGUFs) or NONE must not be left
    NONE-pooled: llama_get_embeddings_seq then returns NULL and every embed call
    fails. MEAN is the rescue that made it the historical default."""
    assert emb._effective_pooling(emb.POOLING_AUTO, None) == emb._POOLING_MEAN
    assert emb._effective_pooling(emb.POOLING_AUTO, emb._POOLING_NONE) == emb._POOLING_MEAN


def test_explicit_pooling_is_never_overridden_by_the_model():
    """An explicit user choice wins over the declaration - never silently
    'corrected' (hard-won rule: do not override an explicit selection)."""
    assert emb._effective_pooling(emb._POOLING_MEAN, emb._POOLING_LAST) == emb._POOLING_MEAN
    assert emb._effective_pooling(emb._POOLING_LAST, emb._POOLING_CLS) == emb._POOLING_LAST


def _mispool_probe(monkeypatch, declared, effective):
    """An IsolatedEmbedder with the pooling facts a load would have reported,
    without spawning a worker; returns the warnings it emitted."""
    warnings = []
    monkeypatch.setattr("localm.debuglog.logger.warning",
                        lambda msg, *a: warnings.append(msg % a if a else msg))
    probe = emb.IsolatedEmbedder.__new__(emb.IsolatedEmbedder)
    probe.model_path = "Qwen3-Embedding-0.6B-Q8_0.gguf"
    probe.declared_pooling = declared
    probe.effective_pooling = effective
    probe._warn_if_mispooled()
    return warnings


def test_warns_when_a_last_pooling_model_is_mean_pooled(monkeypatch):
    """THE defect-2 surfacing: a decoder-based embedder pooled against its own
    training still returns healthy normalised vectors, so nothing else would ever
    tell the user. Warn, name the model, and name the fix (rule 5)."""
    warnings = _mispool_probe(monkeypatch, emb._POOLING_LAST, emb._POOLING_MEAN)
    assert len(warnings) == 1
    text = warnings[0]
    assert "Qwen3-Embedding-0.6B-Q8_0.gguf" in text
    assert "embedding_pooling" in text          # names the fix
    assert "re-index" in text                   # and its consequence


def test_no_warning_when_the_model_gets_the_pooling_it_declares(monkeypatch):
    assert _mispool_probe(monkeypatch, emb._POOLING_LAST, emb._POOLING_LAST) == []


def test_no_warning_for_the_default_bge_setup(monkeypatch):
    """bge declares CLS and is pooled MEAN, which measures fine (+0.29 margin)
    and matches every existing index. Warning on the DEFAULT setup would be noise
    on every user's box, not signal - so this stays quiet (debug only)."""
    assert _mispool_probe(monkeypatch, emb._POOLING_CLS, emb._POOLING_MEAN) == []
    assert _mispool_probe(monkeypatch, None, emb._POOLING_MEAN) == []


# --------------------------------------------------------------------------- #
#  singleton get_embedder / embed_texts                                        #
# --------------------------------------------------------------------------- #

def test_embed_texts_none_when_no_model(monkeypatch):
    """No embedding model -> None (lexical fallback), and the network auto-download
    is attempted AT MOST once - a batch of embed calls must not re-download per
    chunk. (The filesystem is still re-checked each call; only the download probe
    is latched.)"""
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5",
                                 "n_gpu_layers": 99, "net_mode": "ask"})
    downloads = {"n": 0}

    def _resolve(*, allow_download=None):
        if allow_download is not False:       # the download-permitted probe
            downloads["n"] += 1
        return None

    monkeypatch.setattr(emb, "resolve_embedding_model_path", _resolve)
    assert emb.embed_texts(["a"]) is None
    assert emb.get_embedder() is None
    assert emb.get_embedder() is None
    assert downloads["n"] == 1                 # auto-download probed once, not per call


def test_get_embedder_picks_up_model_installed_mid_session(monkeypatch):
    """A model installed into a RUNNING server (``localm setup-embeddings``) is
    picked up on the NEXT call, without a restart. Regression: get_embedder latched
    the 'no model' result for the whole process lifetime, so embeddings stayed dead
    (RAG/memory 422 -> lexical) until a restart even right after setup."""
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5",
                                 "n_gpu_layers": 99, "net_mode": "ask"})
    state = {"path": None}
    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None: state["path"])

    class _FakeEmbedder:
        dim = 3

        def __init__(self, path, **kw):
            self.model_path = path

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", _FakeEmbedder)

    # No model yet -> lexical fallback; the negative result must NOT be latched.
    assert emb.get_embedder() is None
    # setup-embeddings installs the model; the next call must find it, no reset().
    state["path"] = "/home/models/embeddings/bge-small.gguf"
    e = emb.get_embedder()
    assert e is not None and e.model_path.endswith("bge-small.gguf")


def test_get_embedder_on_progress_announces_stages_on_success(monkeypatch):
    """ADR-0004 Unit B: on_progress (used by the 'warm up now' job) receives
    coarse stage announcements around the existing load steps - purely additive,
    the isolated child's own IPC protocol is untouched (this test only fakes the
    PARENT-side IsolatedEmbedder construction)."""
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5",
                                 "n_gpu_layers": 99, "net_mode": "off"})
    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None: "/models/bge-small.gguf")

    class _Ok:
        dim = 5

        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Ok)

    messages: list[str] = []
    e = emb.get_embedder(on_progress=messages.append)

    assert e is not None
    assert any("resolv" in m.lower() for m in messages), messages
    assert any("loading" in m.lower() for m in messages), messages
    assert any("ready" in m.lower() and "5" in m for m in messages), messages


_MINUTE_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "fifteen": 15, "twenty": 20, "thirty": 30}


def _stated_minutes(msg):
    """Minutes claimed by a progress line ('up to five minutes', 'up to 90
    seconds'), or None when it states no bound. Accepts both a spelled-out word
    and a digit so a reword does not fail this for the wrong reason."""
    m = re.search(r"up to (\w+) (minutes?|seconds?)", msg, re.I)
    if not m:
        return None
    word, unit = m.group(1), m.group(2).lower()
    n = _MINUTE_WORDS.get(word.lower())
    if n is None:
        try:
            n = float(word)
        except ValueError:
            return None
    return float(n) / 60.0 if unit.startswith("second") else float(n)


def test_get_embedder_progress_states_the_real_load_bound(monkeypatch):
    """The 'loading' stage quotes a ceiling, and it must be the ceiling this stage
    genuinely runs under: _embedder_runner.LOAD_TIMEOUT_DEFAULT, which applies
    because _reload's spawn_and_load() passes no override. It used to promise "up
    to a minute" against a 300s deadline, so a slow-but-perfectly-healthy first
    load looked hung to anyone watching it.

    Pinned to the CONSTANT, never to a literal sentence, so raising the timeout
    without touching the copy fails HERE rather than misinforming a user. The
    sibling stage test above only asserts a 'loading' line exists at all, which
    is exactly why the wrong figure survived in it."""
    from localm.inference._embedder_runner import LOAD_TIMEOUT_DEFAULT

    _cfg(monkeypatch)
    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None: "/models/bge-small.gguf")

    class _Ok:
        dim = 5

        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Ok)

    messages: list[str] = []
    assert emb.get_embedder(on_progress=messages.append) is not None

    loading = [m for m in messages if "loading" in m.lower()]
    assert loading, messages
    stated = _stated_minutes(loading[0])
    assert stated is not None, f"the loading stage states no bound: {loading[0]!r}"
    assert stated == pytest.approx(LOAD_TIMEOUT_DEFAULT / 60.0), (
        f"progress promises {stated} min but the real ceiling is "
        f"{LOAD_TIMEOUT_DEFAULT / 60.0} min: {loading[0]!r}")


def test_get_embedder_on_progress_reports_no_model_available(monkeypatch):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5",
                                 "n_gpu_layers": 99, "net_mode": "off"})
    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None: None)

    messages: list[str] = []
    assert emb.get_embedder(on_progress=messages.append) is None
    assert any("no embedding model" in m.lower() for m in messages), messages


def test_get_embedder_on_progress_reports_load_failure(monkeypatch):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "x", "n_gpu_layers": 99,
                                 "net_mode": "off"})
    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None: "/models/not-an-embedder.gguf")

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("this llama.dll build does not expose the embeddings API")

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Boom)

    messages: list[str] = []
    assert emb.get_embedder(on_progress=messages.append) is None
    assert any("load failed" in m.lower() and "embeddings api" in m.lower()
              for m in messages), messages


def test_get_embedder_recovers_after_the_config_changes_to_a_good_spec(monkeypatch):
    """Regression for the permanent-breakage bug (coordinator's audit,
    2026-08-04/05): the previous bare-bool ``_LOAD_FAILED`` short-circuited at
    the TOP of get_embedder(), before even calling
    resolve_embedding_model_path() again - so once ANY load failed, EVERY
    later call returned None immediately, forever, regardless of the user
    fixing ``embedding_model`` to point at a perfectly good model, until an
    explicit reset_embedder(). Confirmed live via call-count instrumentation
    before this fix: resolve_embedding_model_path was never even called on
    the second attempt, no matter what embedding_model was changed to.

    ``_LOAD_FAILED_SPEC`` must be compared against the CURRENT config value,
    so a genuinely different spec clears the stale latch and gets a real
    attempt - not just a correctly-worded decline."""
    cfg = {"embedding_model": "broken-model.gguf", "n_gpu_layers": 99,
           "net_mode": "off"}
    monkeypatch.setattr("localm.config.load_config", lambda: dict(cfg))
    paths = {"broken-model.gguf": "/models/broken-model.gguf",
             "good-model.gguf": "/models/good-model.gguf"}
    monkeypatch.setattr(
        emb, "resolve_embedding_model_path",
        lambda *, allow_download=None: paths.get(cfg["embedding_model"]))

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("corrupt GGUF header")

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Boom)
    assert emb.get_embedder() is None
    assert "corrupt GGUF header" in (emb.last_error() or "")

    # The user fixes embedding_model to point at a working model.
    cfg["embedding_model"] = "good-model.gguf"

    class _Ok:
        dim = 5

        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Ok)
    e = emb.get_embedder()
    assert e is not None and e.dim == 5, (
        "get_embedder() stayed permanently dead after the earlier load "
        "failure, even though embedding_model was changed to a working model")
    assert emb.last_error() is None


def test_get_embedder_does_not_retry_the_load_for_the_same_still_broken_spec(
        monkeypatch):
    """The optimization this replaces the bare bool for is still real: a
    genuinely UNCHANGED spec that already failed to load must not pay for
    a second (expensive, native) load attempt on every call - only the
    permanent-breakage bug (a config CHANGE being ignored) was wrong, not
    the original "stop retrying a known-broken spec" behavior."""
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "broken-model.gguf",
                                 "n_gpu_layers": 99, "net_mode": "off"})
    monkeypatch.setattr(
        emb, "resolve_embedding_model_path",
        lambda *, allow_download=None: "/models/broken-model.gguf")

    attempts = []

    class _Boom:
        def __init__(self, *a, **k):
            attempts.append(1)
            raise RuntimeError("corrupt GGUF header")

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Boom)
    assert emb.get_embedder() is None
    assert emb.get_embedder() is None
    assert emb.get_embedder() is None
    assert len(attempts) == 1, (
        f"a still-broken, UNCHANGED spec should only ever attempt the "
        f"expensive native load once, not on every call: {len(attempts)} attempts")


def test_get_embedder_on_progress_raising_sink_does_not_abort_load(monkeypatch):
    """The '_emit never aborts a load' contract (mirrors
    managed_comfy_provision._emit): a broken progress sink must not turn a
    successful load into a failure."""
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5",
                                 "n_gpu_layers": 99, "net_mode": "off"})
    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None: "/models/bge-small.gguf")

    class _Ok:
        dim = 5

        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Ok)

    def _raising_sink(msg):
        raise RuntimeError("boom")

    e = emb.get_embedder(on_progress=_raising_sink)
    assert e is not None and e.dim == 5


def test_loaded_dim_and_last_error_track_state(monkeypatch):
    """loaded_dim()/last_error() power the GUI picker: a load FAILURE records why
    (so the user learns a wrong pick is not an embedding model) and reports no dim;
    a success clears the error and reports the dimension."""
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "x", "n_gpu_layers": 99,
                                 "net_mode": "off"})
    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None: "/models/not-an-embedder.gguf")
    assert emb.loaded_dim() is None and emb.last_error() is None

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("this llama.dll build does not expose the embeddings API")

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Boom)
    assert emb.get_embedder() is None
    assert "embeddings API" in (emb.last_error() or "")
    assert emb.loaded_dim() is None                 # no dim on failure

    class _Ok:
        dim = 7

        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", _Ok)
    emb.reset_embedder()                            # clears the recorded error
    assert emb.last_error() is None
    assert emb.get_embedder() is not None
    assert emb.loaded_dim() == 7 and emb.last_error() is None


def test_reset_embedder_force_semantics():
    """Direct test against the REAL reset_embedder(force=...), not a test
    double standing in for it. The shape of bug a review caught DURING this
    function's own development - an early "if _EMBEDDER is None: return
    False" that accidentally skipped clearing the negative-cache flags - had
    no test against the real function; every force=False test elsewhere
    (http_server.py's unload paths) monkeypatches reset_embedder itself with
    a fake, which by construction cannot catch a bug INSIDE reset_embedder."""
    class _Fake:
        def __init__(self):
            self.active_requests = 0
            self.closed = False

        def close(self):
            self.closed = True

    # force=False, nothing loaded, but a cached load failure - must still
    # clear the negative caches (the exact regression this test locks in).
    emb._EMBEDDER = None
    emb._LOAD_FAILED_SPEC = "some-spec.gguf"
    emb._TRIED_DOWNLOAD = True
    emb._LAST_ERROR = "boom"
    assert emb.reset_embedder(force=False) is False
    assert emb._LOAD_FAILED_SPEC is None
    assert emb._TRIED_DOWNLOAD is False
    assert emb._LAST_ERROR is None

    # force=False, idle embedder - clears it and reports True.
    fake = _Fake()
    emb._EMBEDDER = fake
    assert emb.reset_embedder(force=False) is True
    assert fake.closed is True
    assert emb._EMBEDDER is None

    # force=False, BUSY embedder - a full no-op: not closed, negative caches
    # untouched, _EMBEDDER survives (AUDIT-CRIT-1 for the embedder).
    fake = _Fake()
    fake.active_requests = 1
    emb._EMBEDDER = fake
    emb._LOAD_FAILED_SPEC = "some-spec.gguf"
    emb._LAST_ERROR = "still cached"
    assert emb.reset_embedder(force=False) is False
    assert fake.closed is False, "a busy embedder must not be closed"
    assert emb._EMBEDDER is fake, "a busy embedder must survive force=False"
    assert emb._LOAD_FAILED_SPEC == "some-spec.gguf", (
        "a no-op decline (busy, force=False) must not touch the negative "
        "caches either - nothing actually changed")
    assert emb._LAST_ERROR == "still cached"

    # force=True (the default) always clears, even a busy embedder.
    fake = _Fake()
    fake.active_requests = 1
    emb._EMBEDDER = fake
    assert emb.reset_embedder() is True
    assert fake.closed is True
    assert emb._EMBEDDER is None


# --------------------------------------------------------------------------- #
#  IsolatedEmbedder - preflight dispatch + auto-respawn (parent-side only, no  #
#  real subprocess - EmbedderRunner is stubbed, mirroring how                  #
#  test_vram_preflight.py patches ModelRunner.spawn_and_load for GgufBackend)  #
# --------------------------------------------------------------------------- #

class _StubRunner:
    """A fake EmbedderRunner that never actually spawns a process."""
    instances = []

    def __init__(self):
        self.alive = True
        self.loaded_params = None
        self.embed_calls = []
        self.shutdown_calls = 0
        type(self).instances.append(self)

    def is_alive(self):
        return self.alive

    def spawn_and_load(self, params, timeout=None):
        self.loaded_params = params
        return {"dim": 5, "n_ctx": params.get("n_ctx") or 512}

    def embed(self, texts, timeout=None):
        self.embed_calls.append(list(texts))
        return [[1.0] * 5 for _ in texts]

    def shutdown(self, grace=5.0):
        self.shutdown_calls += 1
        self.alive = False


@pytest.fixture(autouse=True)
def _reset_stub_runner():
    _StubRunner.instances = []
    yield


def _isolated_embedder(monkeypatch, *, split_devices=0, check_vram=None, n_gpu_layers=99):
    monkeypatch.setattr("localm.inference._embedder_runner.EmbedderRunner", _StubRunner)
    # The embedder gates on applied_split_device_count (loader truth), not the
    # detected/labelling split_device_count - see GPU-SPLIT-VKINDEX.
    monkeypatch.setattr("localm.discover.applied_split_device_count", lambda cfg: split_devices)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr(emb.IsolatedEmbedder, "_check_vram",
                        check_vram if check_vram is not None else (lambda self: None))
    return emb.IsolatedEmbedder("model.gguf", n_gpu_layers=n_gpu_layers)


def test_isolated_embedder_single_gpu_runs_check_vram_preflight(monkeypatch):
    """The single-GPU case (no split configured) must run the SAME
    VramSizingMixin._check_vram() preflight the chat backend uses - the gap
    this refactor closes (gpu_split_shortfall alone was a no-op here)."""
    calls = {"n": 0}

    def _check_vram(self):
        calls["n"] += 1

    e = _isolated_embedder(monkeypatch, split_devices=0, check_vram=_check_vram)
    assert calls["n"] == 1
    assert e.dim == 5


def test_isolated_embedder_single_gpu_preflight_refusal_skips_spawn(monkeypatch):
    """A _check_vram() refusal must raise BEFORE a child is ever spawned - no
    process-spawn cost paid for a load that can never fit (mirrors
    GgufBackend.load()'s fail-fast-before-spawn contract)."""
    def _check_vram(self):
        raise RuntimeError("Context too large for available VRAM")

    with pytest.raises(RuntimeError, match="too large"):
        _isolated_embedder(monkeypatch, split_devices=0, check_vram=_check_vram)
    assert _StubRunner.instances == []          # never spawned


def test_isolated_embedder_multi_gpu_uses_split_shortfall_not_check_vram(monkeypatch, tmp_path):
    """>= 2 configured split devices: gpu_split_shortfall() gates instead of
    _check_vram() (which only reasons about the single main GPU device)."""
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * 1000)
    calls = {"check_vram": 0, "shortfall": 0}

    def _check_vram(self):
        calls["check_vram"] += 1

    def _shortfall(vram, cfg):
        calls["shortfall"] += 1
        return []

    monkeypatch.setattr(emb.IsolatedEmbedder, "_check_vram", _check_vram)
    monkeypatch.setattr("localm.inference._embedder_runner.EmbedderRunner", _StubRunner)
    monkeypatch.setattr("localm.discover.applied_split_device_count", lambda cfg: 2)
    monkeypatch.setattr("localm.discover.gpu_split_shortfall", _shortfall)
    monkeypatch.setattr("localm.config.load_config", lambda: {})

    e = emb.IsolatedEmbedder(str(f), n_gpu_layers=99)
    assert calls["shortfall"] == 1
    assert calls["check_vram"] == 0             # NOT called in the split path
    assert e.dim == 5


def test_isolated_embedder_vulkan_split_skips_per_device_and_still_loads(
        monkeypatch, tmp_path, caplog):
    """On the vulkan build a CONFIGURED 2-way split must (a) route through the
    split branch - applied_split_device_count is loader-truth so it does NOT
    collapse the way split_device_count does when list_gpus() is Vulkan-blind
    (GPU-SPLIT-VKINDEX) - and (b) SKIP the per-device VRAM preflight HONESTLY
    (gpu_split_shortfall self-skips on vulkan, logging at INFO) instead of running
    the wrong-index-space check or falling back to the single-GPU _check_vram().

    Drives REAL discover (only _native_backend_has_vulkan / list_gpus / load_config
    are stubbed), so it exercises the actual fix end to end, not a mock of it.

    MUTATION A: revert the embedder gate to split_device_count -> the Vulkan-blind
    list_gpus collapses it to < 2 -> the single-GPU _check_vram() branch runs ->
    calls["check_vram"] != 0 -> RED.
    MUTATION B: revert gpu_split_shortfall's vulkan guard -> the one tiny-free
    device it CAN see is flagged -> RuntimeError -> the no-raise / dim==5 line RED."""
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * 100_000)
    calls = {"check_vram": 0}

    def _check_vram(self):
        calls["check_vram"] += 1

    # Vulkan-blind: list_gpus (torch/nvidia-smi) sees only ONE tiny-free device;
    # the second split device is Vulkan-only and invisible to it. This single
    # detected-but-starved device is what makes BOTH mutations above fire.
    blind = [{"index": 0, "total": 16_000_000_000, "free": 100}]
    monkeypatch.setattr(emb.IsolatedEmbedder, "_check_vram", _check_vram)
    monkeypatch.setattr("localm.inference._embedder_runner.EmbedderRunner", _StubRunner)
    monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: True)
    monkeypatch.setattr("localm.discover.list_gpus", lambda *a, **k: list(blind))
    monkeypatch.setattr("localm.config.load_config", lambda: {"gpu_split_indices": [0, 1]})

    with caplog.at_level(logging.INFO, logger="localm"):
        e = emb.IsolatedEmbedder(str(f), n_gpu_layers=99)   # must NOT raise
    assert e.dim == 5                                        # child spawned + loaded
    assert calls["check_vram"] == 0                          # split branch, not the single-GPU fallback
    assert any(r.levelno == logging.INFO and "GPU-SPLIT-VKINDEX" in r.getMessage()
               for r in caplog.records), \
        "the honest-unknown per-device skip must be surfaced at INFO (reaches a bug report)"


def test_isolated_embedder_split_preflight_admits_on_stale_probe(monkeypatch, tmp_path):
    """Consumer-level oracle, no consumer edit: the embedder split preflight is
    gpu_split_shortfall's OTHER figure-quoting caller. This drives the REAL
    gpu_split_shortfall (list_gpus patched, not the function stubbed) with a TIMEOUT
    probe whose frozen reading reads too small. The source fix returns [] instead of
    fabricating a shortfall, so _preflight_vram must NOT raise a 'Not enough VRAM ...
    {stale MB} free' RuntimeError, and the child spawns. Reverting the discover.py
    fix makes the real gpu_split_shortfall refuse from the stale reading -> RED."""
    from localm.discover import GPU_PROBE_TIMEOUT
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * 4000)          # needed = int(4800 * ratio) per device
    # A frozen last-known-good reading that reads far too small for even this tiny
    # ask: the OLD code would quote its stale "free" and refuse both devices.
    stale = [{"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 1000},
             {"index": 1, "name": "B", "total": 16 * 1024 ** 3, "free": 1000}]

    def _list_gpus(*a, return_status=False, **k):
        return (stale, GPU_PROBE_TIMEOUT) if return_status else stale

    monkeypatch.setattr("localm.inference._embedder_runner.EmbedderRunner", _StubRunner)
    # The embedder now gates on applied_split_device_count (loader truth); pin it to
    # 2 to enter the split branch, and pin non-vulkan so the REAL gpu_split_shortfall
    # reaches the stale-probe path under test here, not the vulkan honest-unknown skip.
    monkeypatch.setattr("localm.discover.applied_split_device_count", lambda cfg: 2)
    monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
    monkeypatch.setattr("localm.discover.list_gpus", _list_gpus)
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]})

    e = emb.IsolatedEmbedder(str(f), n_gpu_layers=99)   # must NOT raise
    assert e.dim == 5
    assert len(_StubRunner.instances) == 1              # the child was spawned


def test_isolated_embedder_embed_respawns_after_prior_crash(monkeypatch):
    """A crash on a PRIOR embed() call must not permanently disable the
    embedder: the NEXT call transparently respawns + reloads (so one
    transient native fault does not disable embeddings for the process's
    whole remaining life)."""
    e = _isolated_embedder(monkeypatch)
    runner1 = _StubRunner.instances[-1]
    runner1.alive = False                       # simulate: died since last call

    out = e.embed(["hello"])
    assert out == [[1.0] * 5]
    assert len(_StubRunner.instances) == 2      # a second runner was spawned
    assert _StubRunner.instances[-1].embed_calls == [["hello"]]


def test_isolated_embedder_embed_crash_clears_runner_for_next_call(monkeypatch):
    """A crash DURING this call still raises to the caller (never silently
    swallowed, rule 5) - but clears the runner so the NEXT call auto-reloads
    instead of repeatedly hitting the same dead child.

    n_gpu_layers=0: this is the generic dead-worker/drop-and-reload contract,
    kept CPU-configured so the separate GPU-crash-fallback path (which retries
    INLINE rather than waiting for the next call - see
    test_isolated_embedder_gpu_crash_falls_back_to_cpu_inline below) never
    engages here."""
    e = _isolated_embedder(monkeypatch, n_gpu_layers=0)
    runner1 = _StubRunner.instances[-1]

    def _boom(texts, timeout=None):
        # A real crash is DETECTED by the child being gone: _wait raises this
        # exact error only after proc.is_alive() came back False. The double must
        # model that, or it is really a LIVE worker returning a clean error - and
        # discarding one of those orphans it (it keeps the model resident in VRAM
        # with nothing left able to reach it; see test_embedder_runner_isolation
        # .py::TestCleanEmbedErrorKeepsTheWorker).
        runner1.alive = False
        raise RuntimeError("The embedding worker process crashed (exit code -6)")
    runner1.embed = _boom

    with pytest.raises(RuntimeError, match="crashed"):
        e.embed(["hello"])
    assert e._runner is None

    out = e.embed(["hello again"])              # next call transparently reloads
    assert out == [[1.0] * 5]
    assert len(_StubRunner.instances) == 2


def test_isolated_embedder_gpu_crash_falls_back_to_cpu_inline(monkeypatch):
    """A GPU-offloaded worker crash retries INLINE on CPU within the SAME
    call - unlike the generic case above, the caller gets a real result, not
    an exception, so a batch already in progress does not go lexical-only for
    the rest of its run. Falls back only ONCE per embedder instance."""
    e = _isolated_embedder(monkeypatch, n_gpu_layers=99)
    assert e.gpu_fallback_reason is None
    runner1 = _StubRunner.instances[-1]

    def _boom(texts, timeout=None):
        runner1.alive = False
        raise RuntimeError("The embedding worker process crashed (exit code -6)")
    runner1.embed = _boom

    out = e.embed(["hello"])                    # recovers inline, no exception
    assert out == [[1.0] * 5]
    assert len(_StubRunner.instances) == 2       # the CPU respawn
    assert e.n_gpu_layers == 0                   # dropped to CPU
    assert e.gpu_fallback_reason is not None
    assert "CPU" in e.gpu_fallback_reason
    # n_gpu_layers=0 alone does not guarantee no GPU backend involvement (a
    # large enough model's matmul can still dispatch to a REGISTERED vendor
    # backend regardless - issue #749) - the respawned child must be told to
    # hide GPU devices from the runtime entirely, not just skip weight offload.
    assert _StubRunner.instances[-1].loaded_params.get("cpu_only") is True

    # A SECOND crash (now already on CPU) must NOT try to fall back again -
    # it propagates normally, matching the generic dead-worker contract.
    runner2 = _StubRunner.instances[-1]

    def _boom_again(texts, timeout=None):
        runner2.alive = False
        raise RuntimeError("The embedding worker process crashed (exit code -6)")
    runner2.embed = _boom_again
    with pytest.raises(RuntimeError, match="crashed"):
        e.embed(["hello again"])
    assert e._runner is None


def test_isolated_embedder_close_shuts_down_runner(monkeypatch):
    e = _isolated_embedder(monkeypatch)
    runner = _StubRunner.instances[-1]
    e.close()
    assert runner.shutdown_calls == 1
    assert e._runner is None


# --------------------------------------------------------------------------- #
#  Native embedder (real model) - gated                                        #
# --------------------------------------------------------------------------- #

_EMBED_MODEL = os.environ.get("LOCALM_TEST_EMBED_MODEL")


@pytest.mark.real_gguf
@pytest.mark.skipif(not _EMBED_MODEL,
                    reason="set LOCALM_TEST_EMBED_MODEL to a real embedding GGUF")
def test_real_gguf_embeddings_are_semantic():
    e = emb.GGUFEmbedder(_EMBED_MODEL)
    try:
        assert e.dim > 0
        V = e.embed(["a cat", "a kitten", "a car", "quantum chromodynamics"])
        assert len(V) == 4 and all(len(v) == e.dim for v in V)
        for v in V:                           # L2-normalised
            assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-3

        def cos(a, b):
            return sum(x * y for x, y in zip(a, b))
        # semantic ordering: kitten closest to cat, then car, then unrelated
        assert cos(V[0], V[1]) > cos(V[0], V[2]) > cos(V[0], V[3])
        # deterministic WITHIN one decode path: two separate single-item
        # calls for the identical text match tightly (confirmed exactly 0.0
        # cosine-distance live, not just "close").
        s1 = e.embed(["a cat"])[0]
        s2 = e.embed(["a cat"])[0]
        assert abs(cos(s1, s2) - 1.0) < 1e-4
        # "a cat" ABOVE was embedded as part of a 4-text BATCH (V[0]); here it
        # is embedded ALONE, a DIFFERENT native decode path (_decode_batch vs
        # _decode_single - see GGUFEmbedder.embed()'s own docstring). These
        # are only NEAR-identical, not bit-identical: llama.cpp's batched
        # multi-sequence compute graph orders floating-point operations
        # differently than a single-sequence one, a well-known and benign
        # source of batch-size-dependent numerical variance in native
        # inference libraries. Measured live: 0.000674 cosine-distance
        # between the two paths for the identical text - real, deterministic
        # (reproduced exactly on repeat), and far below any threshold that
        # would affect retrieval ranking (this project's own memory-recall
        # gate uses REL_COS_MIN=0.55). 1e-3 gives headroom above the
        # measured value while still being a meaningfully tight bound - the
        # same order of magnitude as the batched-vs-serial correctness bar
        # used when this batching was first proven correct (cosine > 0.999).
        assert abs(cos(s1, V[0]) - 1.0) < 1e-3
    finally:
        e.close()


@pytest.mark.real_gguf
@pytest.mark.skipif(not _EMBED_MODEL,
                    reason="set LOCALM_TEST_EMBED_MODEL to a real embedding GGUF")
def test_real_gguf_embeddings_via_isolated_embedder(monkeypatch):
    """The isolation-wrapped path (IsolatedEmbedder / get_embedder(), running
    the real native load in a CHILD process) must produce the same real,
    semantically-correct embeddings as the raw GGUFEmbedder class above -
    proving the subprocess boundary does not silently corrupt or degrade
    output."""
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": _EMBED_MODEL, "n_gpu_layers": 99,
                                 "net_mode": "off"})
    e = emb.get_embedder()
    assert e is not None and e.dim > 0
    V = e.embed(["a cat", "a kitten", "quantum chromodynamics"])
    assert len(V) == 3 and all(len(v) == e.dim for v in V)

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))
    assert cos(V[0], V[1]) > cos(V[0], V[2])       # kitten closer to cat than QCD


# --------------------------------------------------------------------------- #
#  Over-long input truncation (memory-audit 2026-07-02: every >n_ctx-token     #
#  text embedded to ONE identical garbage vector because the DLL writes        #
#  nothing on tokenize overflow and the zero buffer decoded as 512x token 0)   #
# --------------------------------------------------------------------------- #

class _OverflowApi:
    """Stub reproducing the REAL llama.dll overflow contract (probe-verified in
    the audit): llama_tokenize returns -(needed) and writes NOTHING when the
    buffer is too small; a full-size buffer gets the tokens. One token per
    input byte keeps token sequences content-dependent."""

    def __init__(self):
        self.decoded_tokens = None

    def _tokens_for(self, raw):
        return [(b % 250) + 1 for b in raw]

    def llama_tokenize(self, vocab, raw, ln, buf, cap, add_special, parse_special):
        toks = self._tokens_for(raw[:ln])
        if len(toks) > cap:
            return -len(toks)               # writes NOTHING, like the real DLL
        for i, t in enumerate(toks):
            buf[i] = t
        return len(toks)

    def llama_batch_get_one(self, arr, n):
        self.decoded_tokens = list(arr[:n])
        return ("batch", n)

    def llama_decode(self, ctx, batch):
        return 0

    def llama_get_embeddings_seq(self, ctx, seq):
        toks = self.decoded_tokens or []
        # Content-dependent 4-dim "embedding": identical token sequences (the
        # pre-fix zero buffer) give identical vectors, real sequences differ.
        return [float(sum(toks) % 9973), float(toks[0] if toks else 0),
                float(toks[-1] if toks else 0), float(len(toks))]


def _stub_embedder(n_ctx=8, n_seq_max=emb._EMBED_BATCH_TARGET):
    import ctypes
    import threading
    e = emb.GGUFEmbedder.__new__(emb.GGUFEmbedder)
    e._api = _OverflowApi()
    e._llama_token = ctypes.c_int
    e._lock = threading.RLock()
    e.n_ctx = n_ctx
    e._mem = None
    e._vocab = None
    e._ctx = object()                        # truthy: "loaded"
    e.dim = 4
    e.model_path = "<stub>"
    # A real GGUFEmbedder always has this set by __init__ (see
    # _choose_n_seq_max) - matched here so a stub-based test that embeds
    # MORE than one text at once behaves like the real thing, rather than
    # accidentally working only because a single-item call's grouping loop
    # never actually dereferences it (Python's `and` short-circuits on the
    # first, empty-`current` iteration).
    e._n_seq_max = n_seq_max
    return e


class TestGGUFEmbedderLoadStderrWrapping:
    """The isolated child's native model-load calls (llama_load_model_from_file /
    llama_init_from_model) must run inside debuglog.dedup_native_stderr(), not
    with the child's raw inherited fd 2 left unmanaged - the exact gap #951's
    embedder-load spew came from (create_tensor/load_tensors/sched_reserve/
    graph_reserve landing raw, ungrouped, with no live view and no repeat
    collapsing at all, unlike every load path in llamacpp/llama.py).

    Mocked at the _api boundary rather than a real GGUF, mirroring
    test_lazy_grammar.py's pattern for the equivalent llama.py decision - the
    real load path itself is covered by the @real_gguf tests elsewhere in
    this file."""

    def _patch_native_calls(self, monkeypatch, events):
        import localm.inference.backends.llamacpp._api as api_module

        def _load(path, mp):
            events.append("load")
            return "model_ptr"

        def _init(model, cp):
            events.append("init")
            return "ctx_ptr"

        monkeypatch.setattr(api_module, "has_embeddings_api", lambda: True)
        monkeypatch.setattr(api_module, "llama_backend_init", lambda: None)
        monkeypatch.setattr(api_module, "llama_model_default_params",
                            lambda: types.SimpleNamespace())
        monkeypatch.setattr(api_module, "llama_load_model_from_file", _load)
        monkeypatch.setattr(api_module, "llama_model_get_vocab", lambda model: "vocab")
        monkeypatch.setattr(api_module, "llama_model_n_embd", lambda model: 384)
        monkeypatch.setattr(api_module, "llama_model_n_ctx_train", lambda model: 512)
        monkeypatch.setattr(api_module, "has_model_meta_api", lambda: False)
        monkeypatch.setattr(api_module, "llama_context_default_params",
                            lambda: types.SimpleNamespace())
        monkeypatch.setattr(api_module, "llama_init_from_model", _init)
        monkeypatch.setattr(api_module, "has_memory_api", lambda: False)
        # close()/__del__ safety net: the fake pointers above are plain
        # strings, not real ctypes handles, so the REAL native free functions
        # must never see them - a GGUFEmbedder that outlives this test's
        # monkeypatch teardown (garbage-collected later, calling __del__ with
        # the real api restored) would otherwise pass "ctx_ptr"/"model_ptr"
        # into the actual native free() and crash the interpreter. The test
        # itself also closes explicitly (below) while these are still active,
        # so this is a backstop, not the primary guard.
        monkeypatch.setattr(api_module, "llama_free", lambda ctx: None)
        monkeypatch.setattr(api_module, "llama_free_model", lambda model: None)
        monkeypatch.setattr("localm.discover.apply_main_gpu", lambda *a, **k: None)
        monkeypatch.setattr("localm.discover.apply_gpu_split", lambda *a, **k: None)

    def test_load_and_init_run_inside_one_dedup_native_stderr_scope(self, monkeypatch):
        import contextlib

        events = []
        self._patch_native_calls(monkeypatch, events)

        @contextlib.contextmanager
        def spy_dedup():
            events.append("enter")
            yield
            events.append("exit")

        monkeypatch.setattr(emb, "dedup_native_stderr", spy_dedup)

        embedder = emb.GGUFEmbedder("<stub-path>", n_gpu_layers=0)
        try:
            # ONE contiguous scope covering BOTH native calls, not two
            # separate entries and not left unwrapped - re-entering
            # dedup_native_stderr per call would pay its background-thread
            # cost twice for one load.
            assert events == ["enter", "load", "init", "exit"], events
        finally:
            # Clear the fake pointers while llama_free/llama_free_model are
            # still mocked (see _patch_native_calls) - a later __del__, after
            # monkeypatch has reverted to the real native functions, must
            # find nothing left to free.
            embedder.close()


class TestResolveEmbedCtx:
    """The embedding window auto-sizing decision (_resolve_embed_ctx), in
    isolation from the native load path: a model's own declared training
    context is honoured up to a ceiling, never a flat guess (see the
    constants' own docstrings in embedder.py)."""

    def test_native_window_under_ceiling_is_used_as_is(self):
        assert emb._resolve_embed_ctx(512) == 512

    def test_native_window_over_ceiling_is_capped(self):
        assert emb._resolve_embed_ctx(32768) == emb._EMBED_CTX_CEILING

    def test_native_window_exactly_at_ceiling(self):
        assert emb._resolve_embed_ctx(emb._EMBED_CTX_CEILING) == emb._EMBED_CTX_CEILING

    def test_zero_native_window_falls_back(self):
        """A model that does not usefully declare its own window (a build too
        old for llama_model_n_ctx_train, or genuinely absent metadata) still
        gets exactly the flat default that shipped before this fix, not a
        broken (0-sized) or arbitrary window."""
        assert emb._resolve_embed_ctx(0) == emb._EMBED_CTX_FALLBACK

    def test_negative_native_window_falls_back(self):
        assert emb._resolve_embed_ctx(-1) == emb._EMBED_CTX_FALLBACK


class TestChooseNSeqMax:
    """The n_seq_max sizing decision, in isolation from the native load path.

    Root cause (measured 2026-08-05 via subprocess-isolated bisection across
    two models of different dim, three n_ctx values, CPU and GPU - see
    dev-notes/FINDING-embedder-serial-batching-2026-08-04.md): n_seq_max must
    evenly divide n_ubatch or llama_init_from_model hard-crashes
    (uncatchable native GGML_ASSERT abort()) during its own internal
    graph-reserve warmup. This is a property of the (n_seq_max, n_ubatch)
    PAIR, not of the model, so the fix is a search, not a constant."""

    def test_real_ctx_ceiling_lands_on_the_target(self):
        # 2048 (_EMBED_CTX_CEILING) is a power of two, so the search finds
        # the target on the first try - the common case for most real models.
        assert emb._choose_n_seq_max(2048) == emb._EMBED_BATCH_TARGET

    def test_real_ctx_fallback_lands_on_the_target(self):
        # 512 (_EMBED_CTX_FALLBACK) likewise.
        assert emb._choose_n_seq_max(512) == emb._EMBED_BATCH_TARGET

    def test_non_power_of_two_ctx_finds_a_smaller_divisor(self):
        # 48 = 2^4*3: 32 does not divide it evenly (48/32=1.5), so the search
        # must step down to 16 (48/16=3, exact) rather than falling all the
        # way to 1 - proves the search finds an intermediate divisor, not
        # just the two trivial endpoints (the target or the safe floor).
        assert emb._choose_n_seq_max(48) == 16

    def test_odd_ctx_falls_back_to_one(self):
        # No power of two > 1 divides an odd number - the search must reach
        # its safe floor, exactly today's pre-batching single-sequence
        # behavior, proven safe by every model this embedder has ever loaded.
        assert emb._choose_n_seq_max(513) == 1

    def test_result_always_evenly_divides_the_input(self):
        """The actual safety property, checked directly rather than via a
        handful of examples: for a wide range of n_ubatch values, the chosen
        n_seq_max must be a genuine divisor - the property a hardcoded
        constant could never guarantee for an arbitrary future model."""
        for n_ubatch in range(1, 4096):
            n_seq_max = emb._choose_n_seq_max(n_ubatch)
            assert n_ubatch % n_seq_max == 0, (
                f"n_seq_max={n_seq_max} does not evenly divide "
                f"n_ubatch={n_ubatch} - this exact combination is the "
                f"measured native GGML_ASSERT crash trigger")

    def test_never_exceeds_target_max(self):
        for n_ubatch in (1, 2, 100, 2048, 100000):
            assert emb._choose_n_seq_max(n_ubatch) <= emb._EMBED_BATCH_TARGET

    def test_custom_target_max_is_honoured(self):
        assert emb._choose_n_seq_max(2048, target_max=8) == 8
        assert emb._choose_n_seq_max(48, target_max=8) == 8


def test_overlong_inputs_do_not_collide():
    """Two DIFFERENT over-long texts must not embed identically (pre-fix they
    all decoded the zero-filled buffer and returned one constant vector)."""
    e = _stub_embedder(n_ctx=8)
    v1 = e.embed(["alpha beta gamma delta epsilon zeta"])[0]
    v2 = e.embed(["one two three four five six seven eight nine"])[0]
    assert v1 != v2


def test_overlong_truncation_keeps_final_token():
    """Truncation keeps the full sequence's FINAL token in the last slot (the
    BERT [SEP] with add_special=True) and exactly n_ctx tokens are decoded."""
    e = _stub_embedder(n_ctx=8)
    text = "abcdefghijklmnopqrstuvwxyz"       # 26 tokens in the stub
    e.embed([text])
    toks = e._api.decoded_tokens
    full = e._api._tokens_for(text.encode("utf-8"))
    assert len(toks) == 8
    assert toks[-1] == full[-1]               # SEP preserved
    assert toks[:7] == full[:7]               # head preserved


def test_short_input_unchanged_by_overflow_fix():
    e = _stub_embedder(n_ctx=8)
    e.embed(["abc"])
    assert e._api.decoded_tokens == e._api._tokens_for(b"abc")


class TestEmbedBatchDispatch:
    """embed()'s own grouping/dispatch logic - which of _decode_single vs
    _decode_batch gets called, for which group sizes - tested in isolation
    from native decode correctness (covered separately by the @real_gguf
    tests below) by spying on both methods on a stub embedder. _OverflowApi
    tokenizes roughly one token per input byte (see its own docstring), so a
    text's token count is easy to control by its length."""

    def _spied(self, monkeypatch, n_ctx=64, n_seq_max=4):
        e = _stub_embedder(n_ctx=n_ctx, n_seq_max=n_seq_max)
        calls = {"single": [], "batch": []}

        def fake_single(tokens):
            calls["single"].append(list(tokens))
            return [0.0] * e.dim

        def fake_batch(token_lists):
            calls["batch"].append([list(t) for t in token_lists])
            return [[0.0] * e.dim for _ in token_lists]

        monkeypatch.setattr(e, "_decode_single", fake_single)
        monkeypatch.setattr(e, "_decode_batch", fake_batch)
        return e, calls

    def test_single_text_uses_the_fast_path(self, monkeypatch):
        """Measured (GPU, real model): a batched call for exactly ONE
        sequence is SLOWER than the single-sequence path (0.60x) - the
        multi-sequence batch machinery's own setup cost loses when there is
        nothing to amortize it across. A lone embed (a chat memory query,
        most /v1/embeddings calls in practice) must keep using the cheap
        path, not the new one."""
        e, calls = self._spied(monkeypatch)
        out = e.embed(["hello"])
        assert len(out) == 1
        assert len(calls["single"]) == 1
        assert len(calls["batch"]) == 0

    def test_multiple_short_texts_use_the_batched_path(self, monkeypatch):
        e, calls = self._spied(monkeypatch)
        out = e.embed(["a", "b", "c"])
        assert len(out) == 3
        assert len(calls["batch"]) == 1
        assert len(calls["batch"][0]) == 3
        assert len(calls["single"]) == 0

    def test_group_size_is_capped_at_n_seq_max(self, monkeypatch):
        e, calls = self._spied(monkeypatch, n_seq_max=4)
        texts = [f"t{i}" for i in range(10)]     # 2 tokens each - count-bound
        out = e.embed(texts)
        assert len(out) == 10
        assert all(len(g) <= 4 for g in calls["batch"]), calls["batch"]
        assert sum(len(g) for g in calls["batch"]) + len(calls["single"]) == 10

    def test_group_shrinks_to_respect_the_token_budget(self, monkeypatch):
        """A group's SUMMED token count must never exceed n_ctx (n_ubatch) -
        llama.cpp's own hard 'encoder requires n_ubatch >= n_tokens'
        constraint for this non-causal architecture (it cannot chunk one
        micro-batch across multiple internal passes the way causal
        generation can). Long texts must pack FEWER per group even when
        n_seq_max would allow more by count alone."""
        e, calls = self._spied(monkeypatch, n_ctx=20, n_seq_max=8)
        texts = ["abcdefgh"] * 8   # 8 tokens each (1/byte) - token-bound, not count-bound
        out = e.embed(texts)
        assert len(out) == 8
        assert len(calls["batch"]) > 1, (
            "8 texts * 8 tokens = 64 tokens must not fit in ONE 20-token group")
        for group in calls["batch"]:
            total = sum(len(g) for g in group)
            assert total <= 20, f"group exceeded the n_ctx token budget: {total}"

    def test_a_text_too_long_to_share_a_group_gets_its_own(self, monkeypatch):
        """Two texts that are each individually within n_ctx but together
        exceed it must NOT be forced into one batched group - they fall back
        to two single-sequence groups, exactly the pre-batching behavior for
        this case (this is also exercised for real in
        test_real_gguf_overlong_texts_not_identical below, with an actual
        model instead of a spy)."""
        e, calls = self._spied(monkeypatch, n_ctx=10, n_seq_max=8)
        texts = ["abcdefghij", "klmnopqrst"]      # 10 tokens each = n_ctx exactly
        out = e.embed(texts)
        assert len(out) == 2
        assert len(calls["batch"]) == 0, (
            "two n_ctx-length texts together (20 tokens) exceed the "
            "10-token budget and must each get their own single-sequence "
            "group, not be forced into one batched call")
        assert len(calls["single"]) == 2

    def test_a_text_that_fails_to_tokenize_needs_no_native_call(self, monkeypatch):
        """The pre-existing zero-token fallback (a text that could not be
        tokenized at all, per _tokenize's own docstring) must stay a pure
        Python short-circuit - it never had a native call to make even
        before batching existed, and grouping must not accidentally route it
        into either decode path."""
        e, calls = self._spied(monkeypatch)
        monkeypatch.setattr(e, "_tokenize", lambda text: [])
        out = e.embed(["anything"])
        assert out == [[0.0] * e.dim]
        assert len(calls["single"]) == 0
        assert len(calls["batch"]) == 0


@pytest.mark.real_gguf
@pytest.mark.skipif(not _EMBED_MODEL,
                    reason="set LOCALM_TEST_EMBED_MODEL to a real embedding GGUF")
class TestBatchedDecodeFailureGranularity:
    """AGENTS.md rule 5: a partial batch failure must NEVER report as
    success. Faults injected at the native-call boundary (llama_decode /
    llama_get_embeddings_seq) on a REAL loaded model and a REAL
    llama_batch_init-built batch - only the OUTCOME of the native call is
    faked, not the batch-construction machinery itself, so this exercises
    the actual ctypes cast/pack code the measurement unit proved correct."""

    def test_a_nonzero_decode_return_raises_and_returns_nothing(self, monkeypatch):
        e = emb.GGUFEmbedder(_EMBED_MODEL)
        try:
            real_decode = e._api.llama_decode
            monkeypatch.setattr(e._api, "llama_decode", lambda ctx, batch: 1)
            with pytest.raises(RuntimeError, match="batched embedding decode failed"):
                e.embed(["alpha", "beta", "gamma"])
            # restore for e.close()'s own (real, single-call) teardown path -
            # close() does not decode, but keep the fixture honest regardless
            monkeypatch.setattr(e._api, "llama_decode", real_decode)
        finally:
            e.close()

    def test_one_null_sequence_in_an_otherwise_ok_batch_raises(self, monkeypatch):
        """The genuinely dangerous partial-success shape: llama_decode
        returns 0 (the call as a whole 'succeeded') but ONE sequence's
        pooled output comes back NULL. Must still raise - a caller must
        never receive a fabricated or all-zero vector for the sequence that
        actually failed while believing the whole batch succeeded."""
        e = emb.GGUFEmbedder(_EMBED_MODEL)
        try:
            real_get = e._api.llama_get_embeddings_seq

            def poisoned_get(ctx, seq):
                if seq == 1:
                    return None
                return real_get(ctx, seq)

            monkeypatch.setattr(e._api, "llama_get_embeddings_seq", poisoned_get)
            with pytest.raises(RuntimeError, match="null embedding for sequence 1"):
                e.embed(["alpha", "beta", "gamma"])
        finally:
            e.close()

    def test_a_group_failure_does_not_leak_partial_results(self, monkeypatch):
        """embed() must not return SOME real vectors and silently drop
        others when a later group in the same call fails - the whole call
        raises, exactly like today's pre-existing single-text serial loop
        already does when any one text fails (Python's list-comprehension
        exception propagation - no caller anywhere expects a partial
        list)."""
        e = emb.GGUFEmbedder(_EMBED_MODEL, n_ctx=16)
        try:
            # n_ctx=16, n_seq_max chosen from that - force at least 2 groups
            # by using MORE texts than n_seq_max allows in one group.
            n_seq_max = e._n_seq_max
            texts = [f"t{i}" for i in range(n_seq_max + 2)]
            call_count = 0
            real_decode = e._api.llama_decode

            def fail_second_call(ctx, batch):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    return 1
                return real_decode(ctx, batch)

            monkeypatch.setattr(e._api, "llama_decode", fail_second_call)
            with pytest.raises(RuntimeError):
                e.embed(texts)
            assert call_count == 2, (
                "the fault must have been reached on the SECOND group, not "
                "before it or not at all - otherwise this test proves nothing")
        finally:
            e.close()


@pytest.mark.real_gguf
@pytest.mark.skipif(not _EMBED_MODEL,
                    reason="set LOCALM_TEST_EMBED_MODEL to a real embedding GGUF")
def test_real_gguf_overlong_texts_not_identical():
    """Audit repro against the real DLL: two different multi-thousand-token
    texts had cosine 1.0 pre-fix."""
    e = emb.GGUFEmbedder(_EMBED_MODEL)
    try:
        long_a = ("the greenhouse controller regulates temperature and "
                  "humidity with a hysteresis loop ") * 120
        long_b = ("quantum chromodynamics binds quarks through gluon "
                  "exchange in the strong interaction ") * 120
        va, vb = e.embed([long_a, long_b])
        cos = sum(x * y for x, y in zip(va, vb))
        assert cos < 0.999, f"over-long texts still collide (cos={cos})"
    finally:
        e.close()


# --------------------------------------------------------------------------
# THE EMBEDDING CONTEXT MUST USE ONE SHARED KV CACHE.
#
# Reported live against 0.1.5rc3 (issue #1320): every re-embed returned 503,
#
#     find_slot: n_tokens = 284 > size = 256
#     decode: failed to find a memory slot for batch of size 1988
#
# llama.cpp's default carves n_ctx into n_seq_max private slices, so at
# n_ctx=2048 / n_seq_max=32 each sequence gets 64 tokens (padded to 256 cells).
# _pack_groups bounds a group by its SUMMED tokens against n_ctx, which is the
# right budget for a SHARED cache and far too generous for a sliced one - so any
# ordinary RAG chunk (300-650 tokens) overflowed its own slice while the group
# still looked legal.
#
# WHY NO EXISTING TEST CAUGHT IT, and why this one asserts on the PARAMS rather
# than by embedding: the bundled default (bge-small) is a BERT-style ENCODER.
# llama.cpp routes it through encode(), which has no KV cache, so the slicing
# cannot bite and every test against it passes. Reproducing the failure for real
# needs a DECODER-based embedding model - a 531 MB download, which does not
# belong in this suite. The context parameter is the durable, free check, and it
# is the one that would have caught this.

def test_embedding_context_requests_a_shared_kv_cache():
    """Drives the pure params function, so it needs NO native runtime.

    The first version of this test drove the whole GGUFEmbedder and passed on a
    machine with llama.cpp provisioned while failing on CI, where __init__ raised
    before it ever reached context creation and the pytest.raises() around it hid
    that. A parameter check must not depend on whether a GPU runtime exists.
    """
    from localm.inference.embedder import configure_embed_context

    class _CP:
        pass

    cp = configure_embed_context(_CP(), n_ctx=2048, n_seq_max=32, pooling_type=1)

    assert cp.kv_unified is True, (
        "the embedding context must share ONE KV cache across the sequences of a "
        "batch. Without it llama.cpp gives each sequence n_ctx/n_seq_max tokens "
        f"(here {2048 // 32}), and every RAG chunk overflows its own slice while "
        "_pack_groups still considers the group legal - issue #1320."
    )
    # The rest of the shape the batcher relies on, pinned in the same place.
    assert cp.n_ctx == cp.n_batch == cp.n_ubatch == 2048, (
        "non-causal encode needs n_ubatch >= the longest sequence, so these move "
        "together"
    )
    assert cp.n_seq_max == 32
    assert cp.embeddings is True


# --------------------------------------------------------------------------
# THE REQUEST ABOVE CAN SILENTLY NOT TAKE.
#
# #1320 (just above) fixed the REQUEST (kv_unified=True). This covers whether
# the loaded runtime actually HONOURS it - a field report (2026-08-25) of
# native embed batches failing to decode with kv_unified requested, traced to
# the loaded context coming back with kv_unified=false and a much smaller
# effective n_ctx_seq. llama.cpp exposes no getter for kv_unified once a
# context exists, so llama_n_ctx_seq is the closest observable proxy: sliced
# (kv_unified not honoured) hands each sequence n_ctx/n_seq_max tokens instead
# of the full shared n_ctx. This is a DIAGNOSTIC ONLY (a log line) - it cannot
# fix or explain a drift, only make it loud instead of silent.

class _FakeCtxApi:
    """Stands in for the llamacpp _api module: only the two calls
    _warn_if_context_config_drifted makes."""

    def __init__(self, n_ctx, n_ctx_seq):
        self._n_ctx = n_ctx
        self._n_ctx_seq = n_ctx_seq

    def llama_n_ctx(self, ctx):
        return self._n_ctx

    def llama_n_ctx_seq(self, ctx):
        return self._n_ctx_seq


def test_context_drift_check_silent_when_honoured(caplog):
    caplog.set_level(logging.WARNING, logger="localm")
    emb._warn_if_context_config_drifted(_FakeCtxApi(2048, 2048), object(), 2048)
    assert caplog.records == []


def test_context_drift_check_warns_when_kv_unified_not_honoured(caplog):
    """The exact field-reported shape: n_ctx round-trips fine, but n_ctx_seq
    comes back sliced (n_ctx/n_seq_max-sized) instead of the full window."""
    caplog.set_level(logging.WARNING, logger="localm")
    emb._warn_if_context_config_drifted(_FakeCtxApi(2048, 64), object(), 2048)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "kv_unified" in warnings[0].getMessage()
    assert "64" in warnings[0].getMessage()


def test_context_drift_check_warns_on_n_ctx_mismatch(caplog):
    """Isolates the n_ctx check from the kv_unified one: n_ctx_seq (500) stays
    above half of the ACTUAL n_ctx's own value is irrelevant here - it must
    stay above half of what was REQUESTED (600) so only the n_ctx-mismatch
    warning fires, not both."""
    caplog.set_level(logging.WARNING, logger="localm")
    emb._warn_if_context_config_drifted(_FakeCtxApi(512, 500), object(), 600)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "n_ctx=512" in warnings[0].getMessage()


def test_context_drift_check_silent_on_older_build_with_no_n_ctx_seq(caplog):
    """llama_n_ctx_seq returns None on a build too old to export it - a
    missing accessor is not evidence of drift, so this must not warn."""
    caplog.set_level(logging.WARNING, logger="localm")
    emb._warn_if_context_config_drifted(_FakeCtxApi(2048, None), object(), 2048)
    assert caplog.records == []


def test_context_drift_check_silent_on_non_int_probe_result(caplog):
    """A probe result that is neither an int nor None (an unconfigured test
    double, or any other caller supplying a non-conforming api) must not
    raise and must not warn - the margin comparison only makes sense for a
    real integer."""
    caplog.set_level(logging.WARNING, logger="localm")
    emb._warn_if_context_config_drifted(
        _FakeCtxApi(2048, MagicMock()), object(), 2048)
    assert caplog.records == []


def test_context_drift_check_never_raises(caplog):
    class _BrokenApi:
        def llama_n_ctx(self, ctx):
            raise RuntimeError("native call failed")

    caplog.set_level(logging.DEBUG, logger="localm")
    emb._warn_if_context_config_drifted(_BrokenApi(), object(), 2048)   # must not raise


def test_llama_n_ctx_seq_returns_none_on_older_build(monkeypatch):
    """Optional accessor (like llama_model_has_mtp): absent on a build old
    enough to predate it must degrade to None, never raise."""
    from localm.inference.backends.llamacpp import _api
    mock_dll = MagicMock(spec=[])            # exports no llama_n_ctx_seq
    monkeypatch.setattr(_api, "load_lib", lambda: mock_dll)
    assert _api.llama_n_ctx_seq(object()) is None


def test_llama_n_ctx_seq_reads_the_native_value(monkeypatch):
    from localm.inference.backends.llamacpp import _api
    mock_dll = MagicMock()
    mock_dll.llama_n_ctx_seq.return_value = 64
    monkeypatch.setattr(_api, "load_lib", lambda: mock_dll)
    assert _api.llama_n_ctx_seq(object()) == 64


# --------------------------------------------------------------------------- #
#  Lock order vs engine._LOAD_LOCK (2026-08-18 whole-server deadlock)          #
# --------------------------------------------------------------------------- #

_LOCK_ORDER_SCENARIO = r'''
import os
import sys
import threading
import time

import localm
from localm.inference import embedder as emb
from localm.inference import engine as eng

# Provenance first, in every outcome: the parent asserts this child imported
# the SAME tree it is testing (an editable-install .pth can silently resolve
# a different checkout).
print("localm-file:", localm.__file__)
sys.stdout.flush()

# Force get_embedder down its slow (load) path with every external effect
# stubbed out: no filesystem, no network, no VRAM probe, no child process.
# The two REAL locks and the real control flow between them are the subject.
emb._EMBEDDER = None
emb._LOAD_FAILED_SPEC = None
emb._TRIED_DOWNLOAD = True
emb.resolve_embedding_model_path = lambda **k: "/models/fake-embed.gguf"
emb._maybe_swap_for_embedder = lambda *a, **k: None
emb._choose_embedder_gpu_layers = lambda *a, **k: (0, None)

import localm.config as _cfgmod
_cfgmod.load_config = lambda: {"embedding_model": "fake", "net_mode": "off"}


class _Ok:
    dim = 5
    model_path = "/models/fake-embed.gguf"

    def __init__(self, *a, **k):
        pass

    def close(self):
        pass


emb.IsolatedEmbedder = _Ok

real_load_lock = eng._LOAD_LOCK
wants_load_lock = threading.Event()
engine_holds_load_lock = threading.Event()


class _SignallingLoadLock:
    """Same underlying lock the engine thread holds, but announces the moment
    get_embedder tries to take it (get_embedder re-imports engine._LOAD_LOCK
    on every call, so this module-attribute swap reaches it)."""

    def __enter__(self):
        wants_load_lock.set()
        real_load_lock.acquire()

    def __exit__(self, *exc):
        real_load_lock.release()


eng._LOAD_LOCK = _SignallingLoadLock()

res = {}


def embedder_side():
    res["embedder"] = emb.get_embedder()


def engine_side():
    # What Engine.load() does for the whole duration of a model load...
    with real_load_lock:
        engine_holds_load_lock.set()
        if not wants_load_lock.wait(10):
            res["harness"] = "get_embedder never tried to take _LOAD_LOCK"
            return
        # ...including calling this from ctx sizing (#767,
        # _sizing.embedder_ctx_reservation_bytes). Pre-fix, get_embedder is
        # holding _LOCK right now while waiting for _LOAD_LOCK, so this call
        # never returns: the 2026-08-18 ABBA deadlock.
        t0 = time.monotonic()
        res["loaded_path"] = emb.loaded_path()
        res["loaded_path_seconds"] = time.monotonic() - t0


eng_t = threading.Thread(target=engine_side, daemon=True)
emb_t = threading.Thread(target=embedder_side, daemon=True)
eng_t.start()
if not engine_holds_load_lock.wait(5):
    print("VERDICT: HARNESS-BROKEN engine thread never took _LOAD_LOCK")
    sys.stdout.flush()
    os._exit(2)
emb_t.start()

eng_t.join(12)
if eng_t.is_alive():
    # loaded_path() has been blocked for 12s while get_embedder holds _LOCK
    # and waits for _LOAD_LOCK: the deadlock this test exists to forbid.
    print("VERDICT: DEADLOCK loaded_path() wedged against get_embedder "
          "(_LOCK -> _LOAD_LOCK inversion)")
    sys.stdout.flush()
    os._exit(1)
if res.get("harness"):
    print("VERDICT: HARNESS-BROKEN", res["harness"])
    sys.stdout.flush()
    os._exit(2)
emb_t.join(10)

ok = (
    not emb_t.is_alive()
    and res.get("embedder") is not None
    # The discriminating outcome ONLY the fixed order can produce: sizing's
    # loaded_path() answered (None: nothing published yet) instantly while a
    # get_embedder was queued waiting for the load lock. The broken order
    # cannot produce this - it wedges instead.
    and "loaded_path" in res
    and res["loaded_path"] is None
    and res.get("loaded_path_seconds", 99.0) < 5.0
)
print("VERDICT:", "OK" if ok else "UNEXPECTED %r" % (res,))
sys.stdout.flush()
os._exit(0 if ok else 3)
'''


def test_get_embedder_lock_order_cannot_deadlock_a_concurrent_engine_load(tmp_path):
    """Regression for the 2026-08-18 whole-server hang: get_embedder() used to
    acquire engine._LOAD_LOCK while holding embedder._LOCK (#313), while the
    chat-load path holds _LOAD_LOCK and calls loaded_path() - which takes
    _LOCK - from its ctx sizing (#767). A model preload racing a first embed
    (the memory plugin's startup migration, in the live incident) deadlocked
    both threads permanently, and every other _LOCK caller wedged behind
    them until the GUI's whole connection pool was dead. Neither acquire has
    a timeout, so the wedge held for over an hour at 0% CPU until the
    process was killed.

    Reconstructs that exact geometry with the REAL locks and the real
    get_embedder()/loaded_path() control flow (externals stubbed), in a
    SUBPROCESS: the pre-fix failure mode is a genuine permanent deadlock
    holding module-level locks, which in-process would wedge this suite's
    own teardown (reset_embedder takes _LOCK) instead of failing cleanly.
    """
    import subprocess
    import sys as _sys

    script = tmp_path / "lock_order_scenario.py"
    script.write_text(_LOCK_ORDER_SCENARIO, encoding="utf-8")
    repo_root = Path(emb.__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [_sys.executable, str(script)], capture_output=True, text=True,
        timeout=60, env=env,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    # The child must have imported THIS tree's localm, or the run proved
    # nothing about the code under test.
    assert str(repo_root).lower() in out.lower(), out
    assert proc.returncode == 0 and "VERDICT: OK" in out, out
