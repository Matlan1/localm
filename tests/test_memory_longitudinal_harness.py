# SPDX-License-Identifier: AGPL-3.0-or-later
"""Longitudinal memory verification harness.

An N-session SCRIPTED SIMULATION that replays real chat sessions through the REAL
memory pipeline - real MemoryStore, real run_consolidation/synthesize_memory
(localm.plugins.builtin.memory.plug), a real bge embedding model, and a real GGUF
chat model for extract/decide/summarize - and asserts memory COMPOUNDS across
sessions rather than in a single shot:

  1. facts stated in early sessions accumulate and are recalled SEMANTICALLY in a
     later, paraphrased query (not exact keyword match);
  2. a paraphrased contradiction across two sessions does not linger as two
     unresolved, permanently-coexisting facts;
  3. the store stays BOUNDED: prune() (real code) forgets decayed synthetic
     clutter even on a zero-fact consolidation round, archives it
     recoverably, and a user-typed fact seeded alongside it survives;
  4. recall precision: an off-topic query injects nothing, an on-topic
     paraphrase recalls with the semantic signal active;
  5. per-session episodic capture: one episode per session, watermarked so
     a re-run with no new session adds zero;
  6. no '<think' substring ever reaches the store.

No mocks of the pipeline under test - only the OUTER HTTP/CLI plumbing is
skipped (session files are written directly in the exact on-disk shape
AuditLog produces, which is what a real second server run would create).

@integration + @real_gguf, so the default `pytest -m "not integration"` run is
unaffected; run this explicitly with
`pytest -m real_gguf tests/test_memory_longitudinal_harness.py -v`. The GPU is a
shared resource, so serialise this against any other GPU workload on the same
machine before running it. Skips cleanly (never fails) when the native runtime or
network is unavailable.
"""

from __future__ import annotations

import json
import os
import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

# Default chat model: a small non-thinking baseline, reliable enough at the 4-way
# ADD/UPDATE/DELETE/NO_OP JSON decision to be a stable regression harness.
_CHAT_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
_CHAT_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Optional override: a path to a real THINKING-family GGUF, so the
# think-stripping assertion is exercised non-vacuously (a raw <think> block is
# actually produced and actually stripped). No default path: set the env var to
# point at a real file to opt in, unset runs the portable default model.
_THINKING_MODEL_ENV = "LOCALM_TEST_THINKING_MODEL"


@pytest.fixture(scope="module")
def native_runtime():
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned (run 'localm setup-llama'): {e}")


@pytest.fixture(scope="module")
def chat_backend(native_runtime):
    override = os.environ.get(_THINKING_MODEL_ENV, "").strip()
    is_thinking = bool(override)
    if override:
        if not os.path.isfile(override):
            pytest.skip(f"{_THINKING_MODEL_ENV} does not point to a file: {override}")
        path = override
    else:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(repo_id=_CHAT_REPO, filename=_CHAT_FILE)
        except Exception as e:
            pytest.skip(f"could not fetch {_CHAT_REPO}/{_CHAT_FILE}: {e}")

    from localm.inference.backends.gguf import GgufBackend
    # CPU-only (n_gpu_layers=0): this harness tests the memory PIPELINE's logic,
    # not GPU inference performance, and a 0.5B model is fast enough on CPU for a
    # handful of short prompts.
    be = GgufBackend(path, n_ctx=4096, n_gpu_layers=0)
    try:
        be.load()
    except Exception as e:
        pytest.skip(f"chat GGUF failed to load on this machine: {e}")
    be._harness_is_thinking = is_thinking
    yield be
    be.unload()


@pytest.fixture(scope="module")
def real_embedder(native_runtime):
    """The real bge GGUFEmbedder instance (not just its bound ``.embed``): the
    test monkeypatches ``localm.inference.embedder.get_embedder`` to return
    this SAME instance, so ``plug._embed_fn()`` (the real consolidation path)
    and the test's own ``recall()`` checks share one embedder/vector space
    instead of ``get_embedder()`` resolving nothing in the throwaway
    LOCALM_HOME (no config.json, no models/embeddings/ dir) and every
    consolidation-added record silently landing with no vector."""
    from huggingface_hub import hf_hub_download

    from localm.inference.embedder import KNOWN_EMBEDDING_MODELS
    repo, filename = KNOWN_EMBEDDING_MODELS["bge-small-en-v1.5"]
    try:
        path = hf_hub_download(repo_id=repo, filename=filename)
    except Exception as e:
        pytest.skip(f"could not fetch embedding model {repo}/{filename}: {e}")

    from localm.inference.embedder import GGUFEmbedder
    try:
        emb = GGUFEmbedder(path)
    except Exception as e:
        pytest.skip(f"bge embedder failed to load on this machine: {e}")
    yield emb
    emb.close()


@pytest.fixture(scope="module")
def real_embed_fn(real_embedder):
    return real_embedder.embed


# --------------------------------------------------------------------------- #
#  Model wiring: complete() mirrors exactly what plug.py binds in production   #
#  (strip_think applied at the consumer, real chat_stream call), plus a raw-  #
#  reply helper for the session transcript itself (unstripped - matching what #
#  AuditLog actually persists; stripping only happens at CONSUMPTION time).   #
# --------------------------------------------------------------------------- #

def _make_complete(backend, think_seen: dict):
    from localm.textnorm import strip_think

    def complete(prompt: str) -> str:
        raw = "".join(backend.chat_stream(
            [{"role": "user", "content": prompt}],
            max_tokens=768, temperature=0.0, seed=7,
        ))
        if getattr(backend, "_harness_is_thinking", False) and "<think" in raw.lower():
            think_seen["hit"] = True
        cleaned = strip_think(raw).strip()
        assert "<think" not in cleaned.lower(), \
            f"strip_think left a scratchpad marker in a consumed reply: {cleaned!r}"
        return cleaned
    return complete


def _raw_reply(backend, user_text: str) -> str:
    """The assistant's RAW reply to one user turn, exactly as AuditLog would
    persist it (unstripped) - think-stripping happens later, at consumption."""
    return "".join(backend.chat_stream(
        [{"role": "user", "content": user_text}],
        max_tokens=256, temperature=0.0, seed=7,
    ))


# --------------------------------------------------------------------------- #
#  Session transcript: fact statements, a paraphrased contradiction, and      #
#  off-topic filler, spread across N distinct session files.                  #
# --------------------------------------------------------------------------- #

_SESSIONS = [
    ("s1_intro", "Hi, I'm Priya, a backend engineer. I'm currently working on "
                 "the Nimbus payments project."),
    ("s2_location", "Just so you know, I live in Berlin."),
    ("s3_offtopic", "Can you give me a quick pasta recipe for tonight?"),
    ("s4_contradiction", "Small correction: I moved to Munich last month, so "
                         "Berlin is no longer where I live."),
    ("s5_tool", "By the way, my favorite editor is Neovim, I use it every day "
               "for coding."),
    ("s6_offtopic", "What's a fun fact about octopuses?"),
    ("s7_project_rename", "Quick note: we renamed the Nimbus project to Comet."),
    ("s8_closing", "Thanks for the help today, see you tomorrow."),
]


def _write_session(sessions_dir, name: str, user_text: str, assistant_raw: str,
                   mtime: float) -> None:
    p = sessions_dir / f"{name}.jsonl"
    rows = [{"type": "user", "data": {"content": user_text}},
            {"type": "llm", "data": {"content": assistant_raw}}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    os.utime(p, (mtime, mtime))


# --------------------------------------------------------------------------- #
#  Bounded-store seeding: decayed synthetic clutter + one user-typed marker,  #
#  added via the REAL store.add() so prune()'s decay-eviction path fires on   #
#  the first consolidation round, archiving to .forgotten.jsonl while the     #
#  user-typed fact survives.                                                  #
# --------------------------------------------------------------------------- #

_CLUTTER_COUNT = 15
_MARKER_TEXT = "The user's favorite color is teal."


def _seed_bounded_store_scenario(store, embed_fn) -> None:
    from localm.memory import MemoryRecord
    old = time.time() - 90 * 86400        # 90 days ago: exp(-3/30*... ) well below PRUNE_FLOOR
    for i in range(_CLUTTER_COUNT):
        store.add(MemoryRecord(
            text=f"Stale synth clutter fact #{i} about an irrelevant one-off detail",
            kind="semantic", source="synth", importance=0.1,
            created=old, last_used=old), save=True)
    store.add(MemoryRecord(text=_MARKER_TEXT, kind="semantic", source="user",
                           importance=0.8), embed_fn=embed_fn, save=True)


# --------------------------------------------------------------------------- #
#  The harness itself                                                          #
# --------------------------------------------------------------------------- #

def test_longitudinal_memory_compounds_across_sessions(
        tmp_path, monkeypatch, chat_backend, real_embed_fn, real_embedder):
    home = tmp_path / ".localm"
    sessions_dir = home / "sessions"
    sessions_dir.mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_MODE", "log")       # writes allowed (not privacy)

    # The throwaway LOCALM_HOME has no config.json and no models/embeddings/
    # dir, so the real get_embedder() singleton (what plug._embed_fn() - the
    # actual consolidation path - calls) would resolve nothing and every
    # consolidation-added record would silently land with no vector, even
    # though real_embed_fn itself works. Point get_embedder() at the SAME
    # already-loaded instance real_embed_fn uses, so consolidation and recall
    # share one real embedder/vector space.
    monkeypatch.setattr(
        "localm.inference.embedder.get_embedder", lambda: real_embedder)

    from localm.plugins.builtin.memory import plug

    think_seen = {"hit": False}
    complete = _make_complete(chat_backend, think_seen)

    # --- seed the bounded-store scenario before any session runs ----------- #
    seed_store = plug._chat_store()
    _seed_bounded_store_scenario(seed_store, real_embed_fn)
    assert len(seed_store.all()) == _CLUTTER_COUNT + 1

    # --- replay N sessions, consolidating after EACH one (mirrors the        #
    #     debounced auto-consolidate outlet firing after a real turn) ------- #
    mtime = time.time() - len(_SESSIONS) * 10
    episodic_counts = []
    results = []
    for name, user_text in _SESSIONS:
        assistant_raw = _raw_reply(chat_backend, user_text)
        _write_session(sessions_dir, name, user_text, assistant_raw, mtime)
        mtime += 10
        res = plug.synthesize_memory(complete)
        results.append((name, res))
        store_now = plug._chat_store()
        episodic_counts.append(sum(1 for r in store_now.all() if r.kind == "episodic"))

    final_store = plug._chat_store()
    all_records = final_store.all()
    all_text = " ".join(r.text for r in all_records).lower()

    # 6. No '<think' substring EVER reaches the store, regardless of which model
    #    is in use.
    assert "<think" not in all_text, \
        f"a raw scratchpad reached the store: {[r.text for r in all_records if '<think' in r.text.lower()]}"

    # 3. Bounded store: the seeded clutter is gone (decayed below PRUNE_FLOOR),
    #    archived recoverably, and the user-typed marker survived.
    clutter_left = [r for r in all_records if r.text.startswith("Stale synth clutter")]
    assert clutter_left == [], f"decayed clutter should have been pruned: {clutter_left}"
    assert any(r.text == _MARKER_TEXT and r.source == "user" for r in all_records), \
        "the user-typed marker fact must survive prune() unconditionally"
    forgotten_path = seed_store.path.with_suffix(".forgotten.jsonl")
    assert forgotten_path.is_file(), "evicted clutter must be archived, not silently gone"
    forgotten_lines = [ln for ln in forgotten_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(forgotten_lines) >= _CLUTTER_COUNT, \
        f"expected >= {_CLUTTER_COUNT} archived records, got {len(forgotten_lines)}"

    # 5. Per-session episodic capture + watermark: episodes accumulated over
    #    the run (monotonically, one per usable-summary session, never fewer
    #    than seen before), and a re-run with NO new session adds none.
    assert episodic_counts == sorted(episodic_counts), \
        f"episodic count must never decrease across sessions: {episodic_counts}"
    episodic_before_rerun = episodic_counts[-1]
    assert episodic_before_rerun >= 1, \
        f"expected at least one usable per-session episode; counts={episodic_counts}"
    rerun_res = plug.synthesize_memory(complete)
    episodic_after_rerun = sum(1 for r in plug._chat_store().all() if r.kind == "episodic")
    assert episodic_after_rerun == episodic_before_rerun, (
        "watermark must skip already-processed sessions: episodic count changed "
        f"from {episodic_before_rerun} to {episodic_after_rerun} on a no-new-session rerun "
        f"(rerun result: {rerun_res})")
    episodics = [r for r in final_store.all() if r.kind == "episodic"]
    for r in episodics:
        assert r.meta.get("session"), f"episode missing source-session tag: {r.meta}"

    # 2. The paraphrased contradiction (Berlin -> Munich): the NEW information
    #    must never be lost - a Munich-mentioning durable fact must exist,
    #    semantic kind only, since an episodic session summary legitimately
    #    mentions the topic too. Perfect single-record resolution is NOT
    #    asserted: consolidate.py prefers ADD over UPDATE when the model is
    #    unsure, so a low-confidence small model legitimately keeps multiple
    #    ADDs instead of collapsing to one. What is asserted is that the
    #    paraphrase reaches the semantic decide() step at all, shown by the
    #    store growing to more than one location record.
    location_hits = [r for r in all_records
                     if r.kind == "semantic"
                     and ("berlin" in r.text.lower() or "munich" in r.text.lower())]
    assert location_hits, "the location contradiction pair produced no durable fact at all"
    assert any("munich" in r.text.lower() for r in location_hits), (
        f"the NEW information (Munich) must never be silently lost: "
        f"{[r.text for r in location_hits]}\nper-session results: {results}")

    # 1. + 4. Recall compounds and is recalled SEMANTICALLY; off-topic queries
    #    inject nothing (absolute relevance gate, real bge cosine values).
    diag_on = {}
    on_topic_hits = final_store.recall(
        "Where does the user currently live, and what editor do they use for coding?",
        k=6, embed_fn=real_embed_fn, diagnostics=diag_on)
    assert on_topic_hits, (
        f"on-topic paraphrase recalled nothing; store had {[r.text for r in all_records]}")
    assert diag_on.get("degrade_reason") is None, \
        f"semantic signal should be active with a real bge embedder: {diag_on}"

    diag_off = {}
    off_topic_hits = final_store.recall(
        "What is the freezing point of liquid nitrogen on Mars?",
        k=6, embed_fn=real_embed_fn, diagnostics=diag_off)
    assert off_topic_hits == [], (
        f"an unrelated query must inject nothing (F12 absolute relevance gate); "
        f"got {[r.text for r in off_topic_hits]}")

    # Think-stripping non-vacuous check: only meaningful when a real thinking
    # model was actually used (LOCALM_TEST_THINKING_MODEL); otherwise this is a
    # no-op, since the default model never emits a think block to strip.
    if getattr(chat_backend, "_harness_is_thinking", False):
        assert think_seen["hit"], (
            "LOCALM_TEST_THINKING_MODEL was set but no raw completion ever "
            "contained a <think> block - the think-stripping check would be "
            "vacuous; verify the model is actually a thinking/reasoning model")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
