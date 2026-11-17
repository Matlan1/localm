# SPDX-License-Identifier: AGPL-3.0-or-later
"""llama_decode and llama_encode run batches through the extended batch API
(llama_batch_ext_* and llama_process) where the runtime exports it, and through
the runtime's own llama_decode / llama_encode only where it does not.

Upstream deprecated llama_decode, llama_encode, llama_batch_init,
llama_batch_get_one and llama_batch_free at b12470; the runtime's own
llama_decode is a thin wrapper that converts a llama_batch to a
llama_batch_ext (llama_batch_compat) and decodes that. localm does the same
conversion, so a batch means the same thing on either path:

* a batch with no positions continues each sequence from its memory;
* a batch with no output flags outputs only its last entry;
* an entry may belong to several sequences;
* a token entry that also carries an embedding row gives the row as the
  entry's state (the draft head of an MTP model).

The real-runtime tests at the end decode the same prompt through both paths on
the provisioned runtime and require the same logits.
"""
from __future__ import annotations

import ctypes
import logging

import pytest

from localm.inference.backends.llamacpp import _api


# --------------------------------------------------------------------------- #
#  llama_batch_init / llama_batch_free / llama_batch_get_one: localm-side     #
# --------------------------------------------------------------------------- #

def _ints(addr, n):
    return list(ctypes.cast(addr, ctypes.POINTER(ctypes.c_int32))[:n])


def test_batch_init_lays_out_a_token_batch_like_the_runtime():
    b = _api.llama_batch_init(3, 0, 2)
    try:
        assert b.n_tokens == 0
        assert b.token and not b.embd
        assert b.pos and b.n_seq_id and b.seq_id and b.logits
        seq = ctypes.cast(b.seq_id, ctypes.POINTER(ctypes.c_void_p))
        rows = [seq[i] for i in range(3)]
        assert len(set(rows)) == 3 and all(rows)
        assert seq[3] is None                      # null-terminated, as natively
        # two sequence ids fit per entry and do not overlap the next entry
        for i, row in enumerate(rows):
            p = ctypes.cast(row, ctypes.POINTER(ctypes.c_int32))
            p[0], p[1] = 10 * i, 10 * i + 1
        assert [_ints(r, 2) for r in rows] == [[0, 1], [10, 11], [20, 21]]
    finally:
        _api.llama_batch_free(b)


def test_batch_init_with_embd_allocates_embedding_rows_not_tokens():
    b = _api.llama_batch_init(2, 4, 1)
    try:
        assert b.embd and not b.token
        rows = ctypes.cast(b.embd, ctypes.POINTER(ctypes.c_float))
        rows[7] = 1.5                              # the last float of 2 x 4
        assert rows[7] == 1.5
    finally:
        _api.llama_batch_free(b)


def test_batch_free_releases_what_batch_init_kept():
    before = len(_api._BATCH_ARRAYS)
    b = _api.llama_batch_init(4, 0, 1)
    assert len(_api._BATCH_ARRAYS) == before + 1
    _api.llama_batch_free(b)
    assert len(_api._BATCH_ARRAYS) == before
    _api.llama_batch_free(b)                       # a second free is harmless
    assert len(_api._BATCH_ARRAYS) == before


def test_batch_get_one_points_at_the_callers_tokens_only():
    arr = (ctypes.c_int32 * 3)(5, 6, 7)
    b = _api.llama_batch_get_one(arr, 3)
    assert b.n_tokens == 3
    assert _ints(b.token, 3) == [5, 6, 7]
    assert not (b.embd or b.pos or b.n_seq_id or b.seq_id or b.logits)


# --------------------------------------------------------------------------- #
#  The conversion to the extended batch                                        #
# --------------------------------------------------------------------------- #

class _Recorder:
    """An extended-batch API that records the entries it is given."""

    def __init__(self, *, seq_pos_max=None, state_ok=True, refuse_at=None, rope=0):
        self.rope = rope
        self.entries: list = []
        self.seq_pos_max_by_seq = seq_pos_max or {}
        self.state_ok = state_ok
        self.refuse_at = refuse_at
        self.processed = None
        self.freed = []

    def init(self, ctx):
        return 0xBA7C

    def free(self, handle):
        self.freed.append(handle)

    def _add(self, seq, **kw):
        if self.refuse_at is not None and len(self.entries) == self.refuse_at:
            return -2
        self.entries.append({"seqs": [seq], "pos": None, "out": False, **kw})
        return len(self.entries) - 1

    def add_token(self, handle, seq, tok):
        return self._add(seq, token=tok)

    def add_embd(self, handle, seq, embd):
        return self._add(seq, embd=list(ctypes.cast(
            embd.data, ctypes.POINTER(ctypes.c_float))[:embd.n_rows * embd.n_embd]))

    def add_seq(self, handle, idx, seq):
        self.entries[idx]["seqs"].append(seq)
        return True

    def set_embd_state(self, handle, idx, embd):
        if not self.state_ok:
            return False
        self.entries[idx]["state"] = list(ctypes.cast(
            embd.data, ctypes.POINTER(ctypes.c_float))[:embd.n_embd])
        return True

    def set_embd_token(self, handle, idx, embd):
        self.entries[idx]["token_embd"] = list(ctypes.cast(
            embd.data, ctypes.POINTER(ctypes.c_float))[:embd.n_embd])
        return True

    def set_pos(self, handle, idx, addr):
        self.entries[idx]["pos"] = _ints(addr, 4)
        return True

    def set_output(self, handle, idx, value):
        self.entries[idx]["out"] = bool(value)
        return True

    def get_memory(self, ctx):
        return 0x3E3

    def seq_pos_max(self, mem, seq):
        return self.seq_pos_max_by_seq.get(seq, -1)

    def get_model(self, ctx):
        return 0x30DE1

    def rope_type(self, model):
        return self.rope

    def process(self, ctx, kind, handle):
        self.processed = kind
        return 0


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(_api, "_batch_ext", lambda: rec)
    return rec


def _token_batch(tokens, pos, seqs, outs):
    b = _api.llama_batch_init(len(tokens), 0, max(len(s) for s in seqs))
    tok = ctypes.cast(b.token, ctypes.POINTER(ctypes.c_int32))
    p = ctypes.cast(b.pos, ctypes.POINTER(ctypes.c_int32))
    nsi = ctypes.cast(b.n_seq_id, ctypes.POINTER(ctypes.c_int32))
    sid = ctypes.cast(b.seq_id, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)))
    lg = ctypes.cast(b.logits, ctypes.POINTER(ctypes.c_int8))
    for i, t in enumerate(tokens):
        tok[i], p[i], nsi[i], lg[i] = t, pos[i], len(seqs[i]), outs[i]
        for j, s in enumerate(seqs[i]):
            sid[i][j] = s
    b.n_tokens = len(tokens)
    return b


def test_decode_carries_tokens_positions_sequences_and_outputs(recorder):
    b = _token_batch([11, 12, 13], [4, 5, 6], [[0], [0, 2], [1]], [0, 1, 1])
    try:
        assert _api.llama_decode(0xC7, b) == 0
    finally:
        _api.llama_batch_free(b)
    assert recorder.processed == _api.LLAMA_PROCESS_TYPE_DECODE
    assert [(e["token"], e["pos"][0], e["seqs"], e["out"]) for e in recorder.entries] == [
        (11, 4, [0], False), (12, 5, [0, 2], True), (13, 6, [1], True)]
    assert recorder.freed == [0xBA7C]


def test_encode_runs_the_encode_process_type(recorder):
    arr = (ctypes.c_int32 * 2)(3, 4)
    assert _api.llama_encode(0xC7, _api.llama_batch_get_one(arr, 2)) == 0
    assert recorder.processed == _api.LLAMA_PROCESS_TYPE_ENCODE


def test_a_batch_without_positions_continues_from_memory_and_outputs_the_last(
        recorder):
    recorder.seq_pos_max_by_seq = {0: 9}
    arr = (ctypes.c_int32 * 3)(1, 2, 3)
    assert _api.llama_decode(0xC7, _api.llama_batch_get_one(arr, 3)) == 0
    assert [(e["token"], e["pos"][0], e["seqs"], e["out"]) for e in recorder.entries] == [
        (1, 10, [0], False), (2, 11, [0], False), (3, 12, [0], True)]


def test_an_empty_memory_starts_at_position_zero(recorder):
    arr = (ctypes.c_int32 * 2)(1, 2)
    _api.llama_decode(0xC7, _api.llama_batch_get_one(arr, 2))
    assert [e["pos"][0] for e in recorder.entries] == [0, 1]


def test_a_token_entry_with_an_embedding_row_gives_it_as_state(recorder):
    b = _api.llama_batch_init(2, 3, 1)
    holder = (ctypes.c_int32 * 2)(7, 8)
    original = b.token
    try:
        b.token = ctypes.cast(holder, ctypes.c_void_p)
        b.n_tokens = 2
        rows = ctypes.cast(b.embd, ctypes.POINTER(ctypes.c_float))
        for k in range(6):
            rows[k] = float(k)
        p = ctypes.cast(b.pos, ctypes.POINTER(ctypes.c_int32))
        nsi = ctypes.cast(b.n_seq_id, ctypes.POINTER(ctypes.c_int32))
        lg = ctypes.cast(b.logits, ctypes.POINTER(ctypes.c_int8))
        for i in range(2):
            p[i], nsi[i], lg[i] = 20 + i, 1, i
        assert _api.llama_decode(0xC7, b) == 0
    finally:
        b.token = original
        _api.llama_batch_free(b)
    assert [(e["token"], e["state"], e["pos"][0]) for e in recorder.entries] == [
        (7, [0.0, 1.0, 2.0], 20), (8, [3.0, 4.0, 5.0], 21)]


def test_the_row_becomes_the_token_embedding_where_the_context_takes_no_state(
        recorder):
    recorder.state_ok = False
    b = _api.llama_batch_init(1, 2, 1)
    holder = (ctypes.c_int32 * 1)(7)
    original = b.token
    try:
        b.token = ctypes.cast(holder, ctypes.c_void_p)
        b.n_tokens = 1
        ctypes.cast(b.n_seq_id, ctypes.POINTER(ctypes.c_int32))[0] = 1
        assert _api.llama_decode(0xC7, b) == 0
    finally:
        b.token = original
        _api.llama_batch_free(b)
    assert recorder.entries[0]["token_embd"] == [0.0, 0.0]
    assert "state" not in recorder.entries[0]


def test_an_embedding_batch_adds_one_row_per_entry(recorder):
    b = _api.llama_batch_init(2, 2, 1)
    try:
        b.n_tokens = 2
        rows = ctypes.cast(b.embd, ctypes.POINTER(ctypes.c_float))
        for k in range(4):
            rows[k] = float(k + 1)
        p = ctypes.cast(b.pos, ctypes.POINTER(ctypes.c_int32))
        nsi = ctypes.cast(b.n_seq_id, ctypes.POINTER(ctypes.c_int32))
        for i in range(2):
            p[i], nsi[i] = i, 1
        assert _api.llama_decode(0xC7, b) == 0
    finally:
        _api.llama_batch_free(b)
    assert [(e["embd"], e["pos"]) for e in recorder.entries] == [
        ([1.0, 2.0], [0, 0, 0, 0]), ([3.0, 4.0], [1, 0, 0, 0])]


def test_an_mrope_embedding_batch_takes_its_positions_section_major(recorder):
    """The caller of an M-RoPE embedding batch points ``pos`` at its own
    4 x n array, as with the runtime's llama_batch_init."""
    recorder.rope = 40                      # LLAMA_ROPE_TYPE_IMROPE
    b = _api.llama_batch_init(2, 2, 1)
    own_pos = (ctypes.c_int32 * 8)(10, 11, 20, 21, 30, 31, 40, 41)
    try:
        b.n_tokens = 2
        b.pos = ctypes.cast(own_pos, ctypes.c_void_p)
        nsi = ctypes.cast(b.n_seq_id, ctypes.POINTER(ctypes.c_int32))
        nsi[0] = nsi[1] = 1
        assert _api.llama_decode(0xC7, b) == 0
    finally:
        _api.llama_batch_free(b)
    assert [e["pos"] for e in recorder.entries] == [[10, 20, 30, 40], [11, 21, 31, 41]]


def test_an_embedding_batch_without_positions_continues_from_memory(recorder):
    recorder.seq_pos_max_by_seq = {0: 4}
    b = _api.llama_batch_init(2, 2, 1)
    try:
        b.n_tokens = 2
        b.pos = None
        b.n_seq_id = None
        b.seq_id = None
        b.logits = None
        assert _api.llama_decode(0xC7, b) == 0
    finally:
        _api.llama_batch_free(b)
    assert [(e["pos"][0], e["seqs"], e["out"]) for e in recorder.entries] == [
        (5, [0], False), (6, [0], True)]


def test_free_finds_a_batch_whose_pos_the_caller_repointed():
    before = len(_api._BATCH_ARRAYS)
    b = _api.llama_batch_init(2, 2, 1)
    b.pos = None
    _api.llama_batch_free(b)
    assert len(_api._BATCH_ARRAYS) == before


def test_a_refused_entry_fails_the_batch_without_processing_it(recorder, caplog):
    caplog.set_level(logging.WARNING, logger="localm")
    recorder.refuse_at = 1
    b = _token_batch([1, 999999, 3], [0, 1, 2], [[0], [0], [0]], [0, 0, 1])
    try:
        assert _api.llama_decode(0xC7, b) == -1
    finally:
        _api.llama_batch_free(b)
    assert recorder.processed is None
    assert recorder.freed == [0xBA7C]
    assert "entry 1 was refused" in caplog.text


# --------------------------------------------------------------------------- #
#  Which native functions are called                                           #
# --------------------------------------------------------------------------- #

class _Lib:
    """A loaded runtime exporting exactly the functions given."""

    def __init__(self, **fns):
        for name, fn in fns.items():
            setattr(self, name, fn)


def _native(ret, calls, name):
    def fn(*args):
        calls.append(name)
        return ret
    return fn


def test_a_runtime_without_the_extended_api_uses_its_llama_decode(monkeypatch):
    calls: list = []
    lib = _Lib(llama_decode=_native(0, calls, "llama_decode"),
               llama_encode=_native(0, calls, "llama_encode"))
    monkeypatch.setattr(_api, "load_lib", lambda: lib)
    arr = (ctypes.c_int32 * 1)(1)
    assert not _api.has_batch_ext_api()
    assert _api.llama_decode(None, _api.llama_batch_get_one(arr, 1)) == 0
    assert _api.llama_encode(None, _api.llama_batch_get_one(arr, 1)) == 0
    assert calls == ["llama_decode", "llama_encode"]


def test_a_runtime_with_the_extended_api_never_calls_llama_decode(monkeypatch):
    calls: list = []
    names = ("llama_batch_ext_init", "llama_batch_ext_free",
             "llama_batch_ext_add_token", "llama_batch_ext_add_seq",
             "llama_batch_ext_add_embd", "llama_batch_ext_set_embd_token",
             "llama_batch_ext_set_embd_state", "llama_batch_ext_set_pos",
             "llama_batch_ext_set_output_logits", "llama_process",
             "llama_get_memory", "llama_memory_seq_pos_max", "llama_get_model",
             "llama_model_rope_type")
    fns = {n: _native(1 if n != "llama_memory_seq_pos_max" else -1, calls, n)
           for n in names}
    fns["llama_batch_ext_add_token"] = _native(0, calls, "llama_batch_ext_add_token")
    fns["llama_process"] = _native(0, calls, "llama_process")
    fns["llama_decode"] = _native(0, calls, "llama_decode")
    lib = _Lib(**fns)
    monkeypatch.setattr(_api, "load_lib", lambda: lib)
    arr = (ctypes.c_int32 * 1)(1)
    assert _api.has_batch_ext_api()
    assert _api.llama_decode(None, _api.llama_batch_get_one(arr, 1)) == 0
    assert "llama_process" in calls and "llama_decode" not in calls
    assert calls[-1] == "llama_batch_ext_free"


def test_a_runtime_with_neither_says_so(monkeypatch):
    monkeypatch.setattr(_api, "load_lib", lambda: _Lib())
    arr = (ctypes.c_int32 * 1)(1)
    with pytest.raises(RuntimeError, match="neither llama_process nor llama_decode"):
        _api.llama_decode(None, _api.llama_batch_get_one(arr, 1))
    assert not _api.has_encode()


# --------------------------------------------------------------------------- #
#  The real runtime: both paths give the same logits                           #
# --------------------------------------------------------------------------- #

_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"


@pytest.fixture(scope="module")
def llm():
    from tests._real_gguf import fetch_gguf, require_native_runtime
    require_native_runtime()
    path = fetch_gguf(_REPO, _FILE)
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    model = LlamaCpp(path, n_gpu_layers=0, n_ctx=512, verbose=False)
    yield model
    model.close()


def _last_logits(llm, tokens, use_ext: bool, monkeypatch) -> list:
    if not use_ext:
        monkeypatch.setattr(_api, "_batch_ext", lambda: None)
    try:
        _api.llama_memory_clear(_api.llama_get_memory(llm._ctx_ptr), True)
        batch = llm._create_batch(tokens, 0, logits_at_last_only=True)
        try:
            assert _api.llama_decode(llm._ctx_ptr, batch) == 0
        finally:
            _api.llama_batch_free(batch)
        n_vocab = _api.llama_vocab_n_tokens(llm._loaded_tokenizer()._vocab)
        return list(_api.llama_get_logits_ith(llm._ctx_ptr, -1)[:n_vocab])
    finally:
        monkeypatch.undo()


@pytest.mark.integration
@pytest.mark.real_gguf
def test_the_runtime_has_the_extended_api_and_both_paths_agree(llm, monkeypatch):
    assert _api.has_batch_ext_api(), "the provisioned runtime predates llama_process"
    tokens = llm.tokenize("The sky on a clear day is", add_bos=True)
    via_ext = _last_logits(llm, tokens, True, monkeypatch)
    if not hasattr(_api.load_lib(), "llama_decode"):
        pytest.skip("this runtime no longer exports llama_decode to compare against")
    via_legacy = _last_logits(llm, tokens, False, monkeypatch)
    assert len(via_ext) == len(via_legacy) > 0
    assert max(abs(a - b) for a, b in zip(via_ext, via_legacy, strict=True)) < 1e-3
    assert via_ext.index(max(via_ext)) == via_legacy.index(max(via_legacy))
