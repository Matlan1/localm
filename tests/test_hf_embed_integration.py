# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL proof that a HuggingFace CHAT model never silently self-embeds.

No mocks: a tiny ungated causal LM (sshleifer/tiny-gpt2) is loaded for real
through ``HFBackend``, and we assert that the backend reports ``can_embed`` False
and that ``Engine.embed`` routes to the dedicated on-device embedder instead of
mean-pooling the chat model's own hidden states.

A stub backend cannot prove this: the whole defect was that a REAL transformers
causal LM answers ``hasattr(model, "encode")`` False (so hf.embed()'s only
"is this a real embedder" check missed it) while ``can_generate()`` answers True.
Only a real checkpoint exercises that distinction.

Why it matters (measured 2026-07-15 through localm's own IsolatedEmbedder,
CPU-only): a chat decoder's mean-pooled vectors look healthy but cannot separate
related from unrelated text - Qwen2.5-0.5B's max UNRELATED cosine (0.7523)
EXCEEDS its min RELATED cosine (0.7518), versus bge-small's +0.29 margin. Those
vectors reached /v1/embeddings and RAG.

Marked @integration so the default `pytest -m "not integration"` skips it (it
downloads ~2.5 MB on first run and needs torch + transformers).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

_MODEL = "sshleifer/tiny-gpt2"
# A real BERT ENCODER checkpoint (architectures: ["BertModel"]), the same shape as
# bge-small / all-MiniLM / e5 - i.e. localm's own default embedding model.
_ENCODER_MODEL = "hf-internal-testing/tiny-random-BertModel"


def _load(model_id):
    # exc_type=ImportError: see hf_chat_backend below.
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(model_id)
    except Exception as e:                       # offline / hub unreachable
        pytest.skip(f"could not fetch {model_id}: {e}")

    from localm.inference.backends.hf import HFBackend

    be = HFBackend(local_dir, device="cpu")
    be.load()
    return be


@pytest.fixture(scope="module")
def hf_encoder_backend():
    be = _load(_ENCODER_MODEL)
    yield be
    be.unload()


def test_declared_arch_suffixes_match_transformers_own_generation_mixin():
    """Pin _GENERATIVE_ARCH_SUFFIXES against transformers' OWN answer.

    can_embed classifies a checkpoint by the NAME of its declared architecture,
    because resolving the class would import transformers (and torch, which cannot
    be imported alongside the bundled llama.dll - see _declared_generative). That
    naming convention is an assumption about transformers, so it gets checked
    against the real classes HERE, where importing transformers is legitimate:
    GenerationMixin is transformers' own definition of "this generates". If
    upstream renames a task head, this fails loudly instead of letting a model be
    silently misrouted.
    """
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    import transformers as tr
    from transformers.generation import GenerationMixin

    from localm.inference.backends.hf import _GENERATIVE_ARCH_SUFFIXES

    # Real declared architectures: the encoders localm must keep embedding with,
    # and the generative heads it must route to the dedicated embedder.
    for arch in ("BertModel", "XLMRobertaModel", "NomicBertModel", "T5EncoderModel",
                 "DistilBertModel", "MPNetModel", "GPT2LMHeadModel",
                 "Qwen2ForCausalLM", "LlamaForCausalLM", "BertLMHeadModel",
                 "T5ForConditionalGeneration"):
        cls = getattr(tr, arch, None)
        if not isinstance(cls, type):
            continue                             # not in this transformers build
        by_name = arch.endswith(_GENERATIVE_ARCH_SUFFIXES)
        by_class = issubclass(cls, GenerationMixin)
        assert by_name == by_class, (
            f"{arch}: the name-suffix rule says generative={by_name} but "
            f"transformers' GenerationMixin says {by_class}. The naming "
            f"convention _GENERATIVE_ARCH_SUFFIXES relies on has changed; "
            f"update it or can_embed will misroute this model.")


def test_real_bert_encoder_is_still_embedding_capable(hf_encoder_backend):
    """A REAL encoder embedding model must NOT be misrouted to the dedicated
    embedder: it embeds well itself, and refusing it would regress a working path.

    This is the case a mocked model cannot express. HFBackend.load() tries
    AutoModelForCausalLM before AutoModel, and transformers registers bert as a
    causal LM, so this pure encoder loads as BertLMHeadModel and answers
    can_generate() True. Asserting on the real object is what proves can_embed
    reads the DECLARED architecture instead.
    """
    model = hf_encoder_backend._model
    assert model.config.architectures == ["BertModel"]      # a pure encoder
    assert model.can_generate() is True, (
        "if this ever goes False, transformers stopped registering bert as a "
        "causal LM and can_embed's declared-architecture check is why this "
        "encoder was rescued - see HFBackend._declared_generative")
    assert not hasattr(model, "encode")
    assert hf_encoder_backend.can_embed is True

    vecs = hf_encoder_backend.embed(["hello world"])
    assert vecs and len(vecs[0]) > 1                          # a real BERT pool


def test_engine_embed_uses_a_real_encoder_rather_than_the_dedicated_embedder(
        hf_encoder_backend, monkeypatch):
    """End to end: a genuine HF embedding model keeps serving its OWN vectors."""
    import localm.inference.engine as eng_mod
    from localm.inference import embedder as emb

    monkeypatch.setattr(eng_mod, "create_backend", lambda *a, **k: hf_encoder_backend)
    monkeypatch.setattr(emb, "embed_texts", lambda texts: (_ for _ in ()).throw(
        AssertionError("a real encoder must not be sent to the dedicated embedder")))

    engine = eng_mod.Engine("dummy-encoder")
    assert engine.embed(["hello world"]) == hf_encoder_backend.embed(["hello world"])


@pytest.fixture(scope="module")
def hf_chat_backend():
    # exc_type=ImportError: transformers/torch can raise a plain ImportError for
    # a reason other than "not installed" (an internal version gate, a clashed
    # build). pytest 9.1 narrowed importorskip's default to ModuleNotFoundError,
    # so without this those hard-error instead of skipping. Mirrors
    # test_hf_grammar_integration.py.
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(_MODEL)
    except Exception as e:                       # offline / hub unreachable
        pytest.skip(f"could not fetch {_MODEL}: {e}")

    from localm.inference.backends.hf import HFBackend

    be = HFBackend(local_dir, device="cpu")
    be.load()
    yield be
    be.unload()


def test_real_causal_lm_reports_it_cannot_embed(hf_chat_backend):
    """The real checkpoint is a generative decoder, so can_embed is False."""
    model = hf_chat_backend._model
    # Ground the test in what the real model actually is, so a transformers
    # change that breaks the discriminator fails HERE with a clear reason.
    assert model.can_generate() is True
    assert not hasattr(model, "encode"), (
        "the pre-fix .encode() probe was the only embedder check, and a causal "
        "LM does not expose it - that is exactly why it missed this case")
    assert hf_chat_backend.can_embed is False


def test_unloaded_hf_backend_reports_unknown_as_capable():
    """Unloaded, capability is unknown -> True, so routes/chat.py still loads a
    genuine HF embedding model rather than skipping it. Engine.embed re-checks."""
    from localm.inference.backends.hf import HFBackend

    assert HFBackend("does-not-need-to-exist").can_embed is True


def test_engine_embed_does_not_self_embed_a_real_chat_model(hf_chat_backend,
                                                            monkeypatch):
    """End to end: a REAL loaded chat model + Engine.embed -> the dedicated
    embedder is used and the chat model's own mean-pooled vectors never escape."""
    import localm.inference.engine as eng_mod
    from localm.inference import embedder as emb

    monkeypatch.setattr(eng_mod, "create_backend", lambda *a, **k: hf_chat_backend)
    monkeypatch.setattr(emb, "embed_texts", lambda texts: [[0.25]] * len(texts))

    # What the chat model WOULD have returned pre-fix, computed for real, so the
    # assertion below is "not that", not merely "some list".
    chat_vectors = hf_chat_backend.embed(["hello world"])
    assert chat_vectors and len(chat_vectors[0]) > 1     # healthy-looking, non-zero

    engine = eng_mod.Engine("dummy-model")
    out = engine.embed(["hello world"])

    assert out == [[0.25]]                                # the dedicated embedder
    assert out != chat_vectors                            # NOT the chat model's own


def test_engine_embed_real_chat_model_without_embedder_raises(hf_chat_backend,
                                                              monkeypatch):
    """No embedding model installed -> the actionable error, NOT chat vectors.
    RAG catches this and degrades to lexical-only BM25 with a warning, which beats
    blending unusable vectors into its 50/50 lexical+vector score."""
    import localm.inference.engine as eng_mod
    from localm.inference import embedder as emb

    monkeypatch.setattr(eng_mod, "create_backend", lambda *a, **k: hf_chat_backend)
    monkeypatch.setattr(emb, "embed_texts", lambda texts: None)

    engine = eng_mod.Engine("dummy-model")
    with pytest.raises(NotImplementedError, match="setup-embeddings"):
        engine.embed(["hello world"])
