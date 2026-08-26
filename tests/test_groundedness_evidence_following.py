# SPDX-License-Identifier: AGPL-3.0-or-later
"""Groundedness ("evidence-following") probe for the two paths that put
retrieved text into a model's context: RAG retrieval and memory recall.

WHAT IS MEASURED. Retrieval returning the right chunk and the ANSWER being
grounded in that chunk are different properties. This file measures the second:
does THIS model, at THIS prompt, with THESE chunks, actually use them.

THE METHOD. Index (or remember) an INVENTED fact that no model can know from
pretraining, ask about it, then INVERT the stored fact and ask again. A grounded
answer flips with the evidence; an ungrounded one does not. The invented-fact
framing is load-bearing: with a real-world fact a correct answer is ambiguous
between "read the chunk" and "already knew it".

The construction is Evidence-Following Accuracy from "Do Modules Stay in Their
Lane? Role Drift in Compound LLM Systems" (Cao et al., arXiv 2607.21627), scoped
to the static question above. localm does not train models, so this file does
NOT measure role drift.

A REAL MODEL. The property under test IS model behaviour, so a mock proves only
that the harness passes strings around. Marked @integration + @real_gguf, so the
default `pytest -m "not integration"` run is unaffected and conftest.py skips
(never fails) when the native runtime is absent. Run it with:
    pytest -m real_gguf tests/test_groundedness_evidence_following.py -v

Each direction has a paired ABLATION test that asks the identical question with
the evidence REMOVED and asserts neither invented token appears. If grounding
breaks so the model stops reading chunks, the follow tests go red; if the probe
goes vacuous (the fact leaking into the question, say), the ablation tests go
red.
"""

from __future__ import annotations

import types

import pytest

from localm import scopes
from localm.memory import MemoryRecord
from localm.plugins.builtin.memory import plug

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

# The chat model these tests drive.
_CHAT_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
_CHAT_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# The invented fact. Non-relational, and the codenames are unguessable, so a
# model cannot emit them by luck.
_SUBJECT = "the Zelmara Project"
_QUESTION = "What is the current release codename of the Zelmara Project?"
_ORIGINAL = "VESPERTINE"
_INVERTED = "CALDERON"


def _fact(codename: str) -> str:
    return ("Release notes. The Zelmara Project is an internal tool. "
            f"The current release codename of {_SUBJECT} is {codename}. "
            "It replaced the previous internal build numbering.")


# Two "no answer here" shapes.
#
# OFF_TOPIC shares no scoreable term with the question, so retrieval returns
# NOTHING and the consumer is handed nothing rather than junk.
#
# ON_TOPIC_NO_ANSWER names the subject, so it DOES retrieve and looks relevant,
# but it does not contain the codename.
_OFF_TOPIC = (
    "Cafeteria notice. The staff canteen now opens at 07:30 on weekdays. "
    "Hot food service ends at 14:00. The salad bar is unaffected by this change."
)
_ON_TOPIC_NO_ANSWER = (
    f"Facilities notice for {_SUBJECT}. The staff canteen now opens at 07:30 on "
    f"weekdays, and hot food service ends at 14:00. This notice covers building "
    f"access only and does not record any software release information."
)


@pytest.fixture(scope="module")
def native_runtime():
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned "
                    f"(run 'localm setup-llama'): {e}")


@pytest.fixture(scope="module")
def chat_backend(native_runtime):
    """CPU-only, matching test_memory_longitudinal_harness.py: this file tests
    whether the model READS supplied context, not GPU inference, and a 0.5B model
    is fast enough on CPU for a handful of short prompts. CPU-only also keeps the
    probe off the box's single shared GPU entirely."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_CHAT_REPO, filename=_CHAT_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_CHAT_REPO}/{_CHAT_FILE}: {e}")

    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend(path, n_ctx=4096, n_gpu_layers=0)
    try:
        be.load()
    except Exception as e:
        pytest.skip(f"chat GGUF failed to load on this machine: {e}")
    yield be
    be.unload()


_GROUNDED_PREFIX = (
    "Answer the question using ONLY the CONTEXT below. If the context does not "
    "contain the answer, say you do not know.\n\nCONTEXT:\n"
)
_UNGROUNDED_SYSTEM = (
    "Answer the question. If you do not know the answer, say you do not know."
)


def _ask(backend, context: str) -> str:
    """One deterministic turn with *context* as the system message, or with no
    context at all when *context* is empty (the ablation arm).

    localm has NO automatic RAG-into-chat inlet: retrieval is an API whose hits
    reach a model as a coder TOOL RESULT (plugins/coder/tools/rag.py), an HTTP
    response, or CLI output. So the RAG half of this file supplies the prompt a
    consumer would build, around REAL retrieved text. The memory half needs no
    such scaffolding: recall genuinely does inject server-side, and those tests
    drive the real inlet and read back what it produced."""
    system = (_GROUNDED_PREFIX + context) if context else _UNGROUNDED_SYSTEM
    return "".join(backend.chat_stream(
        [{"role": "system", "content": system},
         {"role": "user", "content": _QUESTION}],
        max_tokens=96, temperature=0.0, seed=7)).strip()


def _index_and_retrieve(tmp_path, body: str, name: str,
                        require_hits: bool = True) -> str:
    """REAL ingest and REAL retrieval, then the REAL neutralise() the coder tool
    applies before untrusted chunk text reaches a model. Returns the retrieved
    text framed as the consumer would frame it."""
    from localm.rag.store import Collection
    from localm.textguard import neutralise

    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = tmp_path / f"{name}.txt"
    doc.write_text(body, encoding="utf-8")
    coll = Collection(name, base=tmp_path / "collections").create()
    coll.add_paths([str(doc)])

    hits = coll.query(_QUESTION, k=4)
    if require_hits:
        assert hits, "retrieval returned nothing, so this probe would test nothing"
    return "\n\n".join(neutralise(str(h.get("text", ""))) for h in hits)


def _memory_context(home, monkeypatch, codename: str) -> str:
    """Drive the REAL recall inlet (plug._memory_inlet) and return the system
    message it produced, so the probe reads what the product actually injects."""
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plug, "_home", lambda: home)
    monkeypatch.setenv("LOCALM_MODE", "log")
    plug._chat_store(None).add(MemoryRecord(
        text=f"The current release codename of {_SUBJECT} is {codename}.",
        source="user", importance=0.9))

    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": _QUESTION}]
    ctx = types.SimpleNamespace(model_id="", principal="owner-key-hash",
                                stream=False, request_id="r1", state={},
                                scopes=(scopes.ADMIN,))
    out = plug._memory_inlet(messages, ctx)
    assert out is messages, \
        "the recall inlet injected nothing, so this probe would test nothing"
    injected = messages[0]["content"]
    assert codename in injected, \
        f"the inlet did not carry {codename!r} into the system message"
    return injected


# --------------------------------------------------------------------------- #
#  RAG retrieval                                                              #
# --------------------------------------------------------------------------- #

def test_rag_answer_follows_the_indexed_chunk(tmp_path, chat_backend):
    """Invert the indexed fact and the answer must invert with it."""
    original_ctx = _index_and_retrieve(tmp_path / "orig", _fact(_ORIGINAL),
                                       "kb_original")
    assert _ORIGINAL in original_ctx, "the original codename was not retrieved"
    answer = _ask(chat_backend, original_ctx).upper()
    assert _ORIGINAL in answer, \
        f"answer did not follow the retrieved chunk: {answer!r}"

    # Same question, OPPOSITE indexed fact, freshly ingested and retrieved.
    inverted_ctx = _index_and_retrieve(tmp_path / "inv", _fact(_INVERTED),
                                       "kb_inverted")
    assert _INVERTED in inverted_ctx, "the inverted codename was not retrieved"
    inverted_answer = _ask(chat_backend, inverted_ctx).upper()
    assert _INVERTED in inverted_answer, \
        f"answer did not follow the INVERTED chunk: {inverted_answer!r}"
    assert _ORIGINAL not in inverted_answer, \
        (f"answer kept the ORIGINAL codename after the evidence was inverted, "
         f"which is answering from parametric memory rather than from the "
         f"retrieved chunk: {inverted_answer!r}")


def test_rag_ablation_answer_is_ungrounded_without_the_retrieved_chunk(
        chat_backend):
    """The permanent control: identical question, evidence REMOVED. Neither
    invented codename can appear - they are unguessable, so one appearing means
    the fact reached the model some other way and the follow test above is
    vacuous."""
    answer = _ask(chat_backend, "").upper()
    assert _ORIGINAL not in answer and _INVERTED not in answer, \
        (f"an invented codename appeared with NO evidence supplied, so the "
         f"follow test proves nothing: {answer!r}")


def test_rag_irrelevant_passages_do_not_manufacture_the_invented_fact(
        tmp_path, chat_backend):
    """Retrieved passages that do not bear on the question must not produce the
    invented fact.

    THE SCORING DIRECTION IS INVERTED HERE. On irrelevant passages a confident,
    specific answer is the FAILING direction and an abstention is the PASSING
    one.

    The assertion is the ABSENCE of the invented tokens, not the presence of a
    refusal: a small model can emit a fabricated version string rather than
    declining, which is an instruction-following limit of the model rather than
    a property of the system under test."""
    # A wholly off-topic document shares no scoreable term with the question, so
    # retrieval hands the consumer NOTHING rather than a low-scoring chunk.
    off_topic = _index_and_retrieve(tmp_path / "off", _OFF_TOPIC, "kb_off_topic",
                                    require_hits=False)
    assert off_topic == "", \
        (f"a wholly off-topic document was retrieved for this question, so the "
         f"consumer would be handed irrelevant context: {off_topic!r}")

    # Names the subject, so it DOES retrieve and looks relevant, but carries no
    # codename.
    ctx = _index_and_retrieve(tmp_path / "irr", _ON_TOPIC_NO_ANSWER,
                              "kb_on_topic_no_answer")
    assert _ORIGINAL not in ctx and _INVERTED not in ctx, \
        "the decoy document must not contain either codename"
    answer = _ask(chat_backend, ctx).upper()
    assert _ORIGINAL not in answer and _INVERTED not in answer, \
        (f"the model produced an invented codename from a passage that never "
         f"contained one: {answer!r}")


# --------------------------------------------------------------------------- #
#  Memory recall                                                              #
# --------------------------------------------------------------------------- #

def test_memory_recall_answer_follows_the_injected_memory(
        tmp_path, monkeypatch, chat_backend):
    """Same property one path over: recall injects server-side into the system
    message every chat turn, so an answer that ignores it is the same failure."""
    original_ctx = _memory_context(tmp_path / "mem_a", monkeypatch, _ORIGINAL)
    answer = _ask(chat_backend, original_ctx).upper()
    assert _ORIGINAL in answer, \
        f"answer did not follow the injected memory: {answer!r}"

    inverted_ctx = _memory_context(tmp_path / "mem_b", monkeypatch, _INVERTED)
    inverted_answer = _ask(chat_backend, inverted_ctx).upper()
    assert _INVERTED in inverted_answer, \
        f"answer did not follow the INVERTED memory: {inverted_answer!r}"
    assert _ORIGINAL not in inverted_answer, \
        (f"answer kept the ORIGINAL codename after the remembered fact was "
         f"inverted: {inverted_answer!r}")


def test_memory_ablation_answer_is_ungrounded_without_the_injected_memory(
        chat_backend):
    """The memory path's own ablation control, matching the RAG one."""
    answer = _ask(chat_backend, "").upper()
    assert _ORIGINAL not in answer and _INVERTED not in answer, \
        (f"an invented codename appeared with no memory injected, so the "
         f"memory follow test proves nothing: {answer!r}")
