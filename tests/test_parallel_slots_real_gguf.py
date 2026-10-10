# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL parallel slots: a small GGUF loaded through GgufBackend with more than
one slot, so the isolated worker, the multiplexed runner protocol and the slot
scheduler all run on the native runtime.

@integration: needs the native runtime (localm setup-llama) and the model on
disk or reachable; their absence is a skip, any later failure is real.

Greedy replies decoded in the same batch as other replies are not
bit-identical to the same reply decoded alone (batch-size dependent kernels),
so these tests compare a reply only with one decoded under the same batch
shape.
"""
from __future__ import annotations

import threading
import time

import pytest

from tests._real_gguf import fetch_gguf, require_native_runtime

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"

_HYBRID_REPO = "unsloth/Qwen3.5-0.8B-MTP-GGUF"
_HYBRID_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"


def _load(path, slots, n_ctx=2048):
    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend(path, n_ctx=n_ctx, parallel_slots=slots)
    be.load()
    return be


@pytest.fixture(scope="module")
def model_path():
    require_native_runtime()
    return fetch_gguf(_REPO, _FILE)


@pytest.fixture(scope="module")
def slots4(model_path):
    be = _load(model_path, 4)
    yield be
    be.unload()


def _ask(be, prompt, max_tokens=48, **kw):
    return be.chat_stream([{"role": "user", "content": prompt}], max_tokens=max_tokens,
                          temperature=0.0, seed=1, **kw)


def _run(be, prompt, out, key, max_tokens=48, start=None, **kw):
    if start is not None:
        start.wait(10)
    t0 = time.perf_counter()
    stamps = []
    pieces = []
    for piece in _ask(be, prompt, max_tokens, **kw):
        stamps.append(time.perf_counter() - t0)
        pieces.append(piece)
    out[key] = {"text": "".join(pieces), "finish": be.last_finish_reason,
                "first": t0 + (stamps[0] if stamps else 0.0),
                "last": t0 + (stamps[-1] if stamps else 0.0)}


def _together(be, prompts, max_tokens=48, **kw):
    out = {}
    start = threading.Event()
    threads = [threading.Thread(target=_run, args=(be, p, out, i, max_tokens, start),
                                kwargs=kw, daemon=True)
               for i, p in enumerate(prompts)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(120)
        assert not t.is_alive(), "a reply did not finish"
    return [out[i] for i in range(len(prompts))]


LONG = ["Count from 1 to 200, separated by spaces.",
        "List the even numbers from 2 to 400, separated by spaces."]


def test_the_load_reports_its_slots(slots4):
    assert slots4.parallel_slots == 4
    assert slots4._runner.multiplexed is True


def test_two_replies_decode_at_the_same_time(slots4):
    a, b = _together(slots4, LONG, max_tokens=64)
    assert a["text"] and b["text"]
    assert a["first"] < b["last"] and b["first"] < a["last"], \
        "the replies did not overlap"


def test_identical_requests_at_once_get_identical_replies(slots4):
    a, b = _together(slots4, [LONG[0], LONG[0]])
    assert a["text"] == b["text"] and a["text"]


def test_a_reply_does_not_see_the_reply_beside_it(slots4):
    # Same batch shape both times: the neighbours have the same token count and
    # both run to the budget, so only their content differs.
    first = "Count from 1 to 200, separated by spaces."
    neighbours = ["Count from 300 to 500, separated by spaces.",
                  "Count from 600 to 800, separated by spaces."]
    lens = [slots4.count_messages_tokens([{"role": "user", "content": n}])
            for n in neighbours]
    assert lens[0] == lens[1], f"neighbour prompts differ in length: {lens}"
    a1, n1 = _together(slots4, [first, neighbours[0]], max_tokens=32)
    a2, n2 = _together(slots4, [first, neighbours[1]], max_tokens=32)
    assert n1["finish"] == "length" and n2["finish"] == "length"
    assert n1["text"] != n2["text"]
    assert a1["text"] == a2["text"]


def test_a_grammar_applies_to_its_own_reply_only(slots4):
    grammar = 'root ::= "yes" | "no"'
    out = {}
    start = threading.Event()
    t1 = threading.Thread(target=_run, args=(slots4, "Is the sky blue?", out, "g", 8, start),
                          kwargs={"grammar": grammar}, daemon=True)
    t2 = threading.Thread(target=_run, args=(slots4, LONG[0], out, "free", 48, start),
                          daemon=True)
    t1.start()
    t2.start()
    start.set()
    t1.join(120)
    t2.join(120)
    assert out["g"]["text"].strip() in ("yes", "no")
    assert len(out["free"]["text"].split()) > 5


def test_cancelling_one_reply_leaves_the_other_to_finish(slots4):
    out = {}
    t = threading.Thread(target=_run, args=(slots4, LONG[1], out, "keep", 64), daemon=True)
    victim = _ask(slots4, LONG[0], max_tokens=400)
    next(victim)
    t.start()
    next(victim)
    victim.close()
    t.join(120)
    assert out["keep"]["finish"] == "length"
    assert slots4.loaded


def test_a_token_count_during_a_reply_is_exact(slots4):
    text = "The quick brown fox jumps over the lazy dog, twice over."
    idle = slots4.count_tokens(text)
    gen = _ask(slots4, LONG[0], max_tokens=200)
    next(gen)
    busy = slots4.count_tokens(text)
    gen.close()
    assert busy == idle
    assert idle != max(1, len(text) // 4)


def test_a_reply_alone_matches_the_one_slot_model(model_path, slots4):
    prompt = "Name three colors of the rainbow."
    with_slots = "".join(_ask(slots4, prompt))
    slots4.unload()
    try:
        single = _load(model_path, 1)
        try:
            assert single.parallel_slots == 1
            alone = "".join(_ask(single, prompt))
        finally:
            single.unload()
    finally:
        slots4.load()
    assert with_slots == alone and alone


@pytest.fixture(scope="module")
def hybrid_path():
    require_native_runtime()
    return fetch_gguf(_HYBRID_REPO, _HYBRID_FILE)


def test_a_hybrid_recurrent_model_decodes_replies_together(hybrid_path):
    be = _load(hybrid_path, 2, n_ctx=4096)
    try:
        assert be.parallel_slots == 2
        a, b = _together(be, LONG, max_tokens=32)
        assert a["text"] and b["text"]
        assert a["first"] < b["last"] and b["first"] < a["last"]
        follow = [{"role": "user", "content": LONG[0]},
                  {"role": "assistant", "content": a["text"]},
                  {"role": "user", "content": "Now count backwards from 20."}]
        again = "".join(be.chat_stream(follow, max_tokens=24, temperature=0.0, seed=1))
        assert again.strip()
    finally:
        be.unload()
