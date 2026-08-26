# SPDX-License-Identifier: AGPL-3.0-or-later
"""The mtmd_input_text ABI drift and its two consequences.

llama.cpp commit 4114ba18b inserted ``size_t text_len`` as the SECOND field of
``mtmd_input_text``. Passing the older 16-byte layout to a build that has it
makes the callee read ``text_len`` out of the two flag bytes plus padding: with
both flags true that is 257, so EVERY image prompt is silently truncated to 257
bytes, dropping the media marker and failing with "number of media markers in
text (0) does not match number of bitmaps (1)". The cliff sits exactly at marker
offset 257.
"""

import ctypes
import os

import pytest

from localm.inference.backends.base import (
    UnsupportedInputError,
    VisionInputError,
)
from localm.inference.backends.llamacpp import mtmd as lmtmd


# --------------------------------------------------------------------------- #
#  The layouts themselves                                                      #
# --------------------------------------------------------------------------- #

def test_v1_flag_bytes_are_read_as_text_len_257_by_a_v2_build():
    """Pins the exact mechanism: the V1 flag bytes plus padding are read as a
    text_len of 257 by a V2 build, so the consequence is stated rather than
    left as an opaque size assertion."""
    v1 = lmtmd._MtmdInputTextV1(b"x" * 1000, True, True)
    raw = ctypes.string_at(ctypes.byref(v1), ctypes.sizeof(v1))
    assert ctypes.sizeof(v1) == 16
    assert int.from_bytes(raw[8:16], "little") == 257

    # And the flags a V2 build would read live at 16/17, i.e. past the end.
    assert ctypes.sizeof(lmtmd._MtmdInputTextV2) == 24
    assert lmtmd._MtmdInputTextV2.add_special.offset == 16
    assert lmtmd._MtmdInputTextV2.parse_special.offset == 17


def test_make_input_text_sets_text_len_only_on_v2():
    raw = b"hello world"
    v2 = lmtmd._make_input_text(lmtmd._MtmdInputTextV2, raw, True, False)
    assert v2.text_len == len(raw)
    assert (bool(v2.add_special), bool(v2.parse_special)) == (True, False)

    v1 = lmtmd._make_input_text(lmtmd._MtmdInputTextV1, raw, True, False)
    assert not hasattr(v1, "text_len")
    assert (bool(v1.add_special), bool(v1.parse_special)) == (True, False)


def test_probe_payloads_keep_add_special_false_for_a_v1_reader():
    """The 256-byte length is load-bearing, not cosmetic.

    A V1 build reads add_special from byte 8, which under the V2 struct is the low
    byte of text_len. At 256 that reads 0 (False). If it read nonzero, a V1 build
    would prepend BOS to the empty string it sees and return 1 token instead of 0,
    and the discriminator would silently invert."""
    assert len(lmtmd._PROBE_CONTROL) == 256
    assert len(lmtmd._PROBE_EMBEDDED_NUL) == 256
    assert lmtmd._PROBE_EMBEDDED_NUL[0] == 0        # the leading NUL is the whole point
    assert lmtmd._PROBE_CONTROL[0] != 0

    probe = lmtmd._make_input_text(
        lmtmd._MtmdInputTextV2, lmtmd._PROBE_EMBEDDED_NUL, False, True)
    raw = ctypes.string_at(ctypes.byref(probe), ctypes.sizeof(probe))
    assert raw[8] == 0, "a V1 build would read add_special=True and emit a BOS token"


# --------------------------------------------------------------------------- #
#  The layout probe                                                            #
# --------------------------------------------------------------------------- #

class _FakeMtmd:
    """A stand-in mtmd honouring either text_len or strlen.

    Deliberately reads the caller's struct out of real memory by address, exactly
    as the native side does, so the test exercises the actual marshalling rather
    than a Python-level mock of it."""

    def __init__(self, honours_text_len: bool, *, tokenize_rc: int = 0,
                 tokens_per_byte: int = 1):
        self.honours_text_len = honours_text_len
        self.tokenize_rc = tokenize_rc
        self.tokens_per_byte = tokens_per_byte
        self._n = 0

    def mtmd_input_chunks_init(self):
        return 0x1000

    def mtmd_input_chunks_free(self, chunks):
        pass

    def mtmd_tokenize(self, ctx, chunks, itext_addr, bitmaps, n_bitmaps):
        if self.tokenize_rc:
            return self.tokenize_rc
        ptr = ctypes.cast(itext_addr, ctypes.POINTER(ctypes.c_uint8 * 24)).contents
        raw = bytes(ptr)
        text_ptr = int.from_bytes(raw[0:8], "little")
        if self.honours_text_len:
            n_bytes = int.from_bytes(raw[8:16], "little")
        else:
            n_bytes = len(ctypes.string_at(text_ptr))   # strlen: stops at NUL
        self._n = n_bytes * self.tokens_per_byte
        return 0

    def mtmd_helper_get_n_tokens(self, chunks):
        return self._n


def test_probe_detects_a_text_len_honouring_build():
    assert lmtmd._detect_input_text_class(_FakeMtmd(True), 1) is lmtmd._MtmdInputTextV2


def test_probe_detects_a_strlen_build():
    assert lmtmd._detect_input_text_class(_FakeMtmd(False), 1) is lmtmd._MtmdInputTextV1


def test_probe_is_inconclusive_when_the_control_cannot_fire():
    """The fires-control for the instrument itself. A build where tokenize yields
    no tokens at all must read as "cannot tell", never as "uses strlen" - the two
    are indistinguishable from the discriminator alone, and guessing wrong
    silently truncates every image prompt."""
    assert lmtmd._detect_input_text_class(_FakeMtmd(True, tokens_per_byte=0), 1) is None
    assert lmtmd._detect_input_text_class(_FakeMtmd(True, tokenize_rc=1), 1) is None


# --------------------------------------------------------------------------- #
#  A checked mtmd failure must NOT be a process-killing fault                  #
# --------------------------------------------------------------------------- #

def test_vision_input_error_is_not_a_runtime_error():
    """The classification IS the fix.

    GgufBackend.chat_stream catches RuntimeError from the runner and unloads the
    model; _runner.py lets any non-InvalidGrammarError escape and kill the worker
    process. A checked status code from a native call that returned normally must
    be neither of those."""
    assert issubclass(VisionInputError, UnsupportedInputError)
    assert issubclass(VisionInputError, ValueError)
    assert not issubclass(VisionInputError, RuntimeError)


def test_eval_into_raises_vision_input_error_on_a_nonzero_tokenize_rc():
    """Sited on eval_into, which is where the rc is turned into an exception."""
    ctx = lmtmd.MtmdContext.__new__(lmtmd.MtmdContext)
    ctx._ctx = 0x2000
    ctx._input_text_class = lmtmd._MtmdInputTextV2

    class _M:
        def mtmd_bitmap_init(self, w, h, rgb):
            return 0x3000

        def mtmd_bitmap_free(self, b):
            pass

        def mtmd_input_chunks_init(self):
            return 0x4000

        def mtmd_input_chunks_free(self, c):
            pass

        def mtmd_tokenize(self, *a):
            return 2

    ctx._m = _M()

    import localm.inference.backends.llamacpp._api as api
    orig = api.llama_n_ctx
    api.llama_n_ctx = lambda _c: 4096
    try:
        with pytest.raises(VisionInputError):
            ctx.eval_into(0x5000, "prompt <__media__>", [(4, 4, b"\0" * 48)],
                          add_special=True)
    finally:
        api.llama_n_ctx = orig


# --------------------------------------------------------------------------- #
#  The projector runs on the GPU, and degrades only on a real failure          #
# --------------------------------------------------------------------------- #

class _ParamsRecorder:
    """Captures the params buffer mtmd_init_from_file is handed, so the two
    offsets this module writes can be asserted as BYTES rather than trusted."""

    def __init__(self, init_returns=0x7000):
        self.seen = []
        self._init_returns = init_returns

    def mtmd_context_params_default(self):
        return lmtmd._MtmdParams()

    def mtmd_init_from_file(self, path, model, params):
        raw = ctypes.string_at(ctypes.byref(params), ctypes.sizeof(params))
        self.seen.append({"use_gpu": raw[0],
                          "n_threads": int.from_bytes(raw[4:8], "little")})
        return self._init_returns

    def mtmd_free(self, ctx):
        pass


def _bare_ctx(m):
    c = lmtmd.MtmdContext.__new__(lmtmd.MtmdContext)
    c._m = m
    c._mmproj_path = "x.gguf"
    c._model_ptr = 1
    return c


def test_open_asks_for_the_gpu_and_a_real_thread_count():
    """The regression this guards: the projector was pinned to the CPU for every
    user on every GPU, and n_threads was left at mtmd's flat default of 4, so a
    screenshot took ten minutes with the card idle."""
    rec = _ParamsRecorder()
    c = _bare_ctx(rec)
    c._open(use_gpu=True)
    assert rec.seen[-1]["use_gpu"] == 1
    assert rec.seen[-1]["n_threads"] == lmtmd._encode_threads()
    assert rec.seen[-1]["n_threads"] != 4 or (os.cpu_count() or 4) == 5


def test_open_can_still_be_asked_for_cpu():
    rec = _ParamsRecorder()
    c = _bare_ctx(rec)
    c._open(use_gpu=False)
    assert rec.seen[-1]["use_gpu"] == 0


def test_env_escape_hatch_skips_the_gpu_attempt(monkeypatch):
    monkeypatch.setenv("LOCALM_MTMD_CPU", "1")
    rec = _ParamsRecorder()
    c = _bare_ctx(rec)
    assert c._open(use_gpu=True) is None
    assert rec.seen == [], "the GPU attempt must not reach mtmd_init_from_file"


def test_encode_threads_leaves_a_core_and_never_returns_zero():
    n = lmtmd._encode_threads()
    assert n >= 1
    assert n <= (os.cpu_count() or 4)


def test_retry_on_cpu_reopens_once_and_refuses_to_loop():
    rec = _ParamsRecorder()
    c = _bare_ctx(rec)
    c.on_gpu = True
    c._ctx = 0x7000
    assert c.retry_on_cpu() is True
    assert c.on_gpu is False
    assert rec.seen[-1]["use_gpu"] == 0
    # Already on the CPU: a second call declines.
    assert c.retry_on_cpu() is False


def test_a_gpu_encode_failure_is_retryable_but_a_cpu_one_is_not():
    """The type carries the retry decision, so the caller (which owns the KV the
    failed evaluation dirtied) can reset and try once on the CPU."""
    assert issubclass(lmtmd.MtmdGpuEncodeFailed, VisionInputError)

    import localm.inference.backends.llamacpp._api as api

    class _M:
        def mtmd_bitmap_init(self, w, h, rgb):
            return 0x3000

        def mtmd_bitmap_free(self, b):
            pass

        def mtmd_input_chunks_init(self):
            return 0x4000

        def mtmd_input_chunks_free(self, c):
            pass

        def mtmd_tokenize(self, *a):
            return 0

        def mtmd_helper_eval_chunks(self, *a):
            return 5                      # a native eval failure

    orig = api.llama_n_ctx
    api.llama_n_ctx = lambda _c: 4096
    try:
        for on_gpu, expected in ((True, lmtmd.MtmdGpuEncodeFailed),
                                 (False, VisionInputError)):
            c = _bare_ctx(_M())
            c._ctx = 0x2000
            c._input_text_class = lmtmd._MtmdInputTextV2
            c.on_gpu = on_gpu
            with pytest.raises(expected):
                c.eval_into(0x5000, "p <__media__>", [(4, 4, b"\0" * 48)],
                            add_special=True)
            if not on_gpu:
                # and it must NOT be the retryable subclass
                try:
                    c.eval_into(0x5000, "p <__media__>", [(4, 4, b"\0" * 48)],
                                add_special=True)
                except Exception as e:
                    assert not isinstance(e, lmtmd.MtmdGpuEncodeFailed)
    finally:
        api.llama_n_ctx = orig


def test_runner_dispatch_survives_an_unprocessable_image(monkeypatch):
    """THE crash regression, sited on the dispatch loop where the defect lives.

    _runner_main's chat_stream branch catching only InvalidGrammarError lets
    everything else escape and kill the worker process. A test of eval_into
    alone cannot see that: the collapse is in the ARRANGEMENT of except clauses
    here, not in the raising code. So drive the real dispatch loop and assert
    BOTH halves - it reports an error envelope, and it is still alive to serve
    the next command."""
    import queue
    import threading

    import localm._mp_spawn as mp_spawn
    from localm.inference.backends.llamacpp import _runner, _worker

    monkeypatch.setattr(mp_spawn, "install_parent_death_watchdog", lambda *a: None)
    monkeypatch.setattr(mp_spawn, "suppress_native_error_dialogs", lambda *a: None)

    class _FakeWorker:
        last_finish_reason = "stop"
        grammar_unsupported_this_call = False
        chatml_fallback_reason = None

        def __init__(self, **kw):
            pass

        def load(self):
            return {"ok": True}

        def chat_stream(self, **kw):
            raise VisionInputError("the vision projector could not process this image")

        def count_tokens(self, payload):
            return 42

        def close(self):
            pass

    monkeypatch.setattr(_worker, "GgufWorker", _FakeWorker)

    req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    died: list = []

    def _run():
        try:
            _runner._runner_main(req_q, resp_q, ctrl_q)
        except BaseException as e:      # noqa: BLE001 - escaping = the worker dies
            died.append(e)

    t = threading.Thread(target=_run, name="dispatch-under-test", daemon=True)
    t.start()

    req_q.put(("load", {}))
    assert resp_q.get(timeout=5)[0] == "ok"

    req_q.put(("chat_stream", {"messages": [{"role": "user", "content": "hi"}]}))
    envelope = resp_q.get(timeout=5)
    assert envelope[0] == "error", f"expected a clean error envelope, got {envelope!r}"
    assert envelope[2] == "UnsupportedInputError", (
        f"untagged errors become RuntimeError, which makes GgufBackend unload the "
        f"model; got tag {envelope[2:]!r}")

    # The worker must still be serving.
    req_q.put(("count_tokens", "still alive?"))
    assert resp_q.get(timeout=5) == ("ok", 42)
    assert not died, f"the dispatch loop died instead of reporting: {died!r}"

    req_q.put(None)
    t.join(timeout=5)


def test_parent_maps_the_tag_back_to_a_non_runtime_error():
    """The other half: GgufBackend.chat_stream catches RuntimeError from here and
    UNLOADS the model, so an untagged envelope would still evict a healthy model
    from VRAM even once the worker survives."""
    import queue

    from localm.inference.backends.llamacpp._runner import ModelRunner

    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()

    class _Proc:
        exitcode = None

        def is_alive(self):
            return True

    r._proc = _Proc()
    r._resp_q.put(("error", "could not process this image", "UnsupportedInputError"))

    with pytest.raises(UnsupportedInputError):
        list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
