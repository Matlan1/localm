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

import math
import os
import types

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
        return {"dim": 5}

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


def _isolated_embedder(monkeypatch, *, split_devices=0, check_vram=None):
    monkeypatch.setattr("localm.inference._embedder_runner.EmbedderRunner", _StubRunner)
    monkeypatch.setattr("localm.discover.split_device_count", lambda cfg: split_devices)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr(emb.IsolatedEmbedder, "_check_vram",
                        check_vram if check_vram is not None else (lambda self: None))
    return emb.IsolatedEmbedder("model.gguf", n_gpu_layers=99)


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
    monkeypatch.setattr("localm.discover.split_device_count", lambda cfg: 2)
    monkeypatch.setattr("localm.discover.gpu_split_shortfall", _shortfall)
    monkeypatch.setattr("localm.config.load_config", lambda: {})

    e = emb.IsolatedEmbedder(str(f), n_gpu_layers=99)
    assert calls["shortfall"] == 1
    assert calls["check_vram"] == 0             # NOT called in the split path
    assert e.dim == 5


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
    instead of repeatedly hitting the same dead child."""
    e = _isolated_embedder(monkeypatch)
    runner1 = _StubRunner.instances[-1]

    def _boom(texts, timeout=None):
        raise RuntimeError("The embedding worker process crashed (exit code -6)")
    runner1.embed = _boom

    with pytest.raises(RuntimeError, match="crashed"):
        e.embed(["hello"])
    assert e._runner is None

    out = e.embed(["hello again"])              # next call transparently reloads
    assert out == [[1.0] * 5]
    assert len(_StubRunner.instances) == 2


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
        # deterministic
        assert abs(cos(e.embed(["a cat"])[0], V[0]) - 1.0) < 1e-4
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


def _stub_embedder(n_ctx=8):
    import ctypes
    import threading
    e = emb.GGUFEmbedder.__new__(emb.GGUFEmbedder)
    e._api = _OverflowApi()
    e._llama_token = ctypes.c_int
    e._lock = threading.RLock()
    e._n_ctx = n_ctx
    e._mem = None
    e._vocab = None
    e._ctx = object()                        # truthy: "loaded"
    e.dim = 4
    e.model_path = "<stub>"
    return e


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
