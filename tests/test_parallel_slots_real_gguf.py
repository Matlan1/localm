# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL parallel slots: a small GGUF loaded through GgufBackend with more than
one slot, so the isolated worker, the multiplexed runner protocol and the slot
scheduler all run on the native runtime.

@integration: needs the native runtime (localm setup-llama) and the model on
disk or reachable; their absence is a skip, any later failure is real.

Text is compared only for a reply decoded alone from a clean cache; the same
batch can round differently run to run on a GPU. Isolation between replies is
checked on their logits, against that run-to-run noise and against a control
that does leak.
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


def _isolated_logits(llm, ops, own, neighbour, *, shared_seq=False):
    """The logits row of *own*'s last token after one decode that also holds
    *neighbour*, from a cleared cache. With *shared_seq* the neighbour's tokens
    come first in the SAME sequence, so *own* attends to them (the leak
    control)."""
    import ctypes as _ct

    from localm.inference.backends.llamacpp import _api as api
    with llm._gen_lock:
        ops.clear_memory()
        if shared_seq:
            entries = ([(t, i, 0, False) for i, t in enumerate(neighbour)]
                       + [(t, len(neighbour) + i, 0, i == len(own) - 1)
                          for i, t in enumerate(own)])
        else:
            entries = ([(t, i, 0, i == len(own) - 1) for i, t in enumerate(own)]
                       + [(t, i, 1, i == len(neighbour) - 1) for i, t in enumerate(neighbour)])
        assert ops.decode(entries) == 0
        row = next(i for i, e in enumerate(entries) if e[3] and e[2] == 0)
        n_vocab = api.llama_vocab_n_tokens(llm._tokenizer._vocab)
        ptr = api.llama_get_logits_ith(llm._ctx_ptr, row)
        values = _ct.cast(ptr, _ct.POINTER(_ct.c_float * n_vocab)).contents
        out = list(values)
        ops.clear_memory()
    return out


def test_a_reply_does_not_see_the_reply_beside_it(model_path):
    # A neighbour's content moves this reply's logits by no more than the
    # run-to-run noise of the same batch, and far less than seeing the
    # neighbour's tokens does.
    from localm.inference.backends.llamacpp._slots import LlamaSlotOps
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    llm = LlamaCpp(model_path, n_ctx=2048, n_gpu_layers=99, n_parallel=4)
    try:
        assert llm.n_parallel == 4
        ops = LlamaSlotOps(llm)

        def toks(text):
            return llm._tokenizer.encode(text, add_bos=True)

        own = toks("The capital city of France is")
        b = toks("Seven times eight equals fifty six, and")
        c = b[:1] + b[:0:-1]
        assert len(b) == len(c) and b != c
        with_b = _isolated_logits(llm, ops, own, b)
        with_b_again = _isolated_logits(llm, ops, own, b)
        with_c = _isolated_logits(llm, ops, own, c)
        leaked = _isolated_logits(llm, ops, own, b, shared_seq=True)

        def dist(x, y):
            return max(abs(p - q) for p, q in zip(x, y, strict=True))

        noise = dist(with_b, with_b_again)
        neighbour = dist(with_b, with_c)
        leak = dist(with_b, leaked)
        assert leak > 1.0, f"the leak control moved the logits by only {leak}"
        assert neighbour < leak / 10, (noise, neighbour, leak)
        assert neighbour <= max(10 * noise, 0.05), (noise, neighbour, leak)
    finally:
        llm.close()


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


def test_a_reply_alone_matches_the_one_slot_model(model_path):
    # Both models are loaded fresh for this comparison.
    prompt = "Name three colors of the rainbow."
    with_slots = _load(model_path, 4)
    try:
        assert with_slots.parallel_slots == 4
        sloted = "".join(_ask(with_slots, prompt))
    finally:
        with_slots.unload()
    single = _load(model_path, 1)
    try:
        assert single.parallel_slots == 1
        alone = "".join(_ask(single, prompt))
    finally:
        single.unload()
    assert sloted == alone and alone


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


def test_a_grammar_check_during_a_reply_is_answered(slots4):
    from localm.inference.backends.base import InvalidGrammarError
    gen = _ask(slots4, LONG[0], max_tokens=200)
    next(gen)
    try:
        slots4.validate_grammar('root ::= "yes" | "no"')
        with pytest.raises(InvalidGrammarError):
            slots4.validate_grammar("root ::= (")
    finally:
        gen.close()


_VLM_REPO = "ggml-org/SmolVLM-256M-Instruct-GGUF"
_VLM_FILE = "SmolVLM-256M-Instruct-Q8_0.gguf"
_VLM_MMPROJ = "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"


def _red_png_data_url():
    import base64
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 20, 20)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_an_image_turn_waits_for_the_text_reply_in_flight():
    require_native_runtime()
    model = fetch_gguf(_VLM_REPO, _VLM_FILE)
    mmproj = fetch_gguf(_VLM_REPO, _VLM_MMPROJ)
    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend(model, mmproj_path=mmproj, n_ctx=4096, parallel_slots=2)
    be.load()
    try:
        assert be.parallel_slots == 2 and be.supports_images
        out = {}
        text_started = threading.Event()

        def text():
            pieces = []
            last = None
            for piece in _ask(be, LONG[0], max_tokens=600):
                text_started.set()
                last = time.perf_counter()
                pieces.append(piece)
            out["text"] = {"text": "".join(pieces), "last": last}

        def image():
            text_started.wait(30)
            t0 = time.perf_counter()
            pieces = []
            stamps = []
            for piece in be.chat_stream(
                    [{"role": "user", "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {"type": "image_url", "image_url": {"url": _red_png_data_url()}}]}],
                    max_tokens=24, temperature=0.0, seed=1):
                stamps.append(time.perf_counter())
                pieces.append(piece)
            out["image"] = {"text": "".join(pieces), "first": stamps[0] if stamps else None,
                            "start": t0}

        t_text = threading.Thread(target=text, daemon=True)
        t_text.start()
        t_image = threading.Thread(target=image, daemon=True)
        t_image.start()
        t_text.join(120)
        t_image.join(120)
        assert out["text"]["text"] and out["image"]["text"].strip()
        assert out["image"]["first"] >= out["text"]["last"], \
            "the image turn decoded while the text reply was still running"
        after = "".join(_ask(be, "Say hello.", max_tokens=12))
        assert after.strip()
    finally:
        be.unload()
