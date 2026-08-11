# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-process multimodal (vision) for the GGUF backend, via the bundled mtmd.dll
(llama.cpp ``libmtmd``).

Loads an mmproj (vision projector) alongside the text model and evaluates an
image+text prompt straight into the llama KV cache, so the GGUF backend can answer
about images instead of refusing them (issue C1).

ABI strategy (the bundled runtime ships NO headers and the mtmd C ABI has drifted
across llama.cpp versions, so this binding avoids version-specific struct layouts):

* ``mtmd_context_params`` is treated as an OVER-ALLOCATED opaque buffer. We call the
  exported ``mtmd_context_params_default()`` and pass it through UNMODIFIED except
  two leading fields whose offsets were MEASURED against the shipped runtime
  (``use_gpu`` at byte 0, ``n_threads`` at byte 4 - the default params read
  ``01 01 00 00 04 00 00 00``). On Win64 a struct that large is passed by hidden
  pointer, so an over-sized buffer is safe regardless of the real field layout.
* THE PROJECTOR RUNS ON THE GPU. It used to be forced onto the CPU unconditionally,
  on this reasoning: "CPU clip is the universally-safe path: gfx1030 / RDNA2 hipBLAS
  fails a BF16 GEMM (CUBLAS_STATUS_INTERNAL_ERROR) on a BF16 mmproj." That failure is
  real, but it is specific to a BF16 projector, and the override was applied to every
  user, every GPU and every mmproj - a safety net promoted into the design. Measured
  cost of that on a 1600x928 screenshot (920 image tokens, Qwen3-VL-8B + an f16
  mmproj): the encode ran for TEN MINUTES on 4 of 12 cores and very nearly hit the
  900s first-token timeout, with the GPU idle throughout.
  So: GPU first, and fall back to CPU only when the GPU path ACTUALLY fails, saying
  so plainly in the log (AGENTS.md rule 5). ``n_threads`` is also set now - mtmd
  defaults it to 4 regardless of the machine, which is what made the CPU path so
  much worse than it had to be.
* the image is decoded to raw RGB by the caller and passed to the clean-signature
  ``mtmd_bitmap_init(w, h, rgb)`` - NOT ``mtmd_helper_bitmap_init_from_buf``, whose
  return type drifted to a by-value wrapper in newer builds.
* ``mtmd_input_text`` DID drift and cannot be avoided (it is the one struct this
  module must pass by value), so both layouts are bound and the live one is
  MEASURED at load time - see :func:`_detect_input_text_class`.

Verified end-to-end on gfx1030 with gemma-4 + mmproj-BF16: a test image was
described correctly. See dev-notes for the standalone probe this was lifted from.
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Optional, Tuple

from ..base import VisionInputError
from . import _api as api


class _MtmdParams(ctypes.Structure):
    # Over-allocated opaque buffer (the real struct is well under this); 8-byte
    # aligned via c_uint64. Only byte 0 (use_gpu) is ever touched - see module docs.
    _fields_ = [("_buf", ctypes.c_uint64 * 32)]   # 256 bytes


class _MtmdInputTextV1(ctypes.Structure):
    """``mtmd_input_text`` BEFORE llama.cpp 4114ba18b (#25548, 2026-07-12).

    The text is NUL-terminated: the tokenizer did ``input_text = text->text``."""

    _fields_ = [("text", ctypes.c_char_p),
                ("add_special", ctypes.c_bool),
                ("parse_special", ctypes.c_bool)]


class _MtmdInputTextV2(ctypes.Structure):
    """``mtmd_input_text`` FROM #25548 onward: an explicit ``text_len``.

    That commit ("mtmd: fix silent prompt truncation on embedded NUL") inserted
    ``size_t text_len`` as the SECOND field and switched the tokenizer to
    ``input_text.assign(text->text, text->text_len)``. Passing the V1 layout to a
    V2 build is silently catastrophic rather than merely wrong: the callee reads
    ``text_len`` out of V1's ``add_special``/``parse_special`` bytes plus padding,
    so with both flags true it reads 257 and TRUNCATES EVERY PROMPT TO 257 BYTES -
    which drops the image marker for any prompt with a system preamble, yielding
    "number of media markers in text (0) does not match number of bitmaps (1)".
    It also reads the two flags from offsets 16/17, past the end of V1's 16 bytes.
    See dev-notes/mtmd-input-text-abi-drift-2026-08-06.md."""

    _fields_ = [("text", ctypes.c_char_p),
                ("text_len", ctypes.c_size_t),
                ("add_special", ctypes.c_bool),
                ("parse_special", ctypes.c_bool)]


def _make_input_text(cls, raw: bytes, add_special: bool, parse_special: bool):
    """Build *cls* for *raw*, supplying ``text_len`` only where the layout has it."""
    if cls is _MtmdInputTextV2:
        return cls(raw, len(raw), add_special, parse_special)
    return cls(raw, add_special, parse_special)


_lib: Optional[ctypes.CDLL] = None

# Which mtmd_input_text layout the LOADED mtmd honours. Resolved once per process
# by _detect_input_text_class (a property of the library, not of the model).
_input_text_class: Optional[type] = None


class MtmdUnavailable(RuntimeError):
    """Raised when mtmd.dll or the mmproj cannot be loaded - the GGUF backend then
    stays text-only rather than crashing."""


class MtmdGpuEncodeFailed(VisionInputError):
    """A GPU projector encode failed at runtime. Distinct from a plain
    :class:`VisionInputError` purely so the caller knows a CPU retry is worth one
    attempt (it owns the KV cache, which the failed evaluation dirtied, so the
    retry cannot happen inside ``eval_into``)."""


def _encode_threads() -> int:
    """Threads for the projector. mtmd defaults to a flat 4 regardless of the
    machine; leave one core for the rest of the server rather than taking the box.
    Falls back to mtmd's own default when the CPU count is unknown."""
    n = os.cpu_count() or 4
    return max(1, n - 1)


def _load_lib() -> ctypes.CDLL:
    """Load mtmd.dll from the same runtime dir as llama.dll and bind the minimal
    API surface. Cached. The llama/ggml deps must already be loaded (they are - the
    GGUF backend loads the model first), and the runtime dir is on the DLL search
    path via the loader."""
    global _lib
    if _lib is not None:
        return _lib
    from . import _loader
    binary_dir = _loader.runtime_binary_dir()
    if binary_dir is None:
        raise MtmdUnavailable("native runtime not provisioned")
    name = "mtmd.dll" if os.name == "nt" else "libmtmd.so"
    path = binary_dir / name
    if not path.exists():
        raise MtmdUnavailable(f"{name} not found (runtime has no multimodal support)")
    try:
        m = ctypes.CDLL(str(path))
    except OSError as e:
        raise MtmdUnavailable(f"could not load {name}: {e}")

    m.mtmd_context_params_default.restype = _MtmdParams
    m.mtmd_default_marker.restype = ctypes.c_char_p
    m.mtmd_init_from_file.restype = ctypes.c_void_p
    m.mtmd_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p, _MtmdParams]
    m.mtmd_free.argtypes = [ctypes.c_void_p]
    m.mtmd_support_vision.restype = ctypes.c_bool
    m.mtmd_support_vision.argtypes = [ctypes.c_void_p]
    m.mtmd_bitmap_init.restype = ctypes.c_void_p
    m.mtmd_bitmap_init.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p]
    m.mtmd_bitmap_free.argtypes = [ctypes.c_void_p]
    m.mtmd_input_chunks_init.restype = ctypes.c_void_p
    m.mtmd_input_chunks_free.argtypes = [ctypes.c_void_p]
    m.mtmd_tokenize.restype = ctypes.c_int32
    # The mtmd_input_text pointer is bound as an untyped void* on purpose: which
    # of the two layouts is live is only known after _detect_input_text_class
    # runs, and re-pointing argtypes per call would mutate shared state on this
    # CDLL's function object. Callers pass ctypes.addressof(struct).
    m.mtmd_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    # Exported by both ABI eras (checked against the 2026-06-04 and 2026-08-04
    # builds); used by the layout probe below.
    m.mtmd_helper_get_n_tokens.restype = ctypes.c_size_t
    m.mtmd_helper_get_n_tokens.argtypes = [ctypes.c_void_p]
    m.mtmd_helper_eval_chunks.restype = ctypes.c_int32
    m.mtmd_helper_eval_chunks.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool,
        ctypes.POINTER(ctypes.c_int32)]
    _lib = m
    return m


# Probe payloads for _detect_input_text_class. Both are EXACTLY 256 bytes, and
# that length is load-bearing rather than arbitrary: a V1 build reads add_special
# from byte 8 and parse_special from byte 9, which under the V2 struct are the low
# two bytes of text_len. At 256 those read as add_special=False, parse_special=True.
# add_special MUST come out False, or a V1 build would prepend BOS to the empty
# string it sees and return 1 token instead of 0, destroying the discriminator.
#
# llama.cpp's own tokenizer trace echoes whatever it is handed via an
# "add_text: <text>" line on stderr, so a reader of a captured/debug load log who
# sees "add_text: aaaa..." followed by an empty "add_text: " is looking at these
# two probe calls, in order - not a leaked user prompt. Both are text-only calls
# with no marker/bitmaps, so nothing else runs during them (see _probe_n_tokens's
# own docstring); llama.py's caller wraps this whole constructor call so neither
# line reaches the console.
_PROBE_CONTROL = b"a" * 256
_PROBE_EMBEDDED_NUL = b"\x00" + b"a" * 255


def _probe_n_tokens(m: ctypes.CDLL, ctx: int, cls: type, raw: bytes) -> Optional[int]:
    """Tokenize *raw* with the *cls* layout and return the token count, or None if
    the call itself failed.

    Text only: no marker, no bitmaps. So nothing is image-preprocessed, no llama
    context is touched (``mtmd_tokenize`` only fills a chunk list; only
    ``mtmd_helper_eval_chunks`` writes KV), and mtmd logs nothing - 0 markers
    against 0 bitmaps is a match, so both eras return rc 0 and the probe is
    silent in the native log on the healthy path."""
    chunks = m.mtmd_input_chunks_init()
    if not chunks:
        return None
    try:
        itext = _make_input_text(cls, raw, False, True)
        rc = m.mtmd_tokenize(ctx, chunks, ctypes.addressof(itext), None, 0)
        if rc != 0:
            return None
        return int(m.mtmd_helper_get_n_tokens(chunks))
    except Exception:   # noqa: BLE001 - a probe failure must not condemn the lib
        return None
    finally:
        m.mtmd_input_chunks_free(chunks)


def _detect_input_text_class(m: ctypes.CDLL, ctx: int) -> Optional[type]:
    """Which ``mtmd_input_text`` layout the loaded mtmd honours, or None if the
    probe could not decide.

    Measures the exact property depended on - does this build read ``text_len``
    or ``strlen`` - rather than correlating with a symbol. #25548 added no new
    export, so any symbol probe (the approach ``_abi.py`` can use for
    ``llama_model_params``, where the marker symbols landed in the SAME commit as
    the reorder) would have a window of builds where it is simply wrong.

    Two calls that differ only in a leading NUL byte:

    * CONTROL ``"a"*256`` must tokenize to > 0 tokens. This is the fires-control
      for the instrument: without it, a build where tokenize always yields 0
      would be silently read as "uses strlen" instead of "the probe is broken".
    * DISCRIMINATOR ``"\\0" + "a"*255``, same 256 bytes. A build that honours
      text_len tokenizes all 256 and returns > 0; a build using strlen stops at
      the leading NUL, tokenizes nothing and returns 0.

    Inconclusive is a real answer and is NOT resolved by guessing: the caller
    keeps the model text-only with a logged reason. Guessing V2 on a V1 build
    would leave add_special/parse_special reading out of the low bytes of
    text_len, i.e. a prompt tokenized with the wrong special-token handling and
    no error anywhere - silent wrong output, which is worse than no vision."""
    control = _probe_n_tokens(m, ctx, _MtmdInputTextV2, _PROBE_CONTROL)
    if not control:
        return None
    embedded = _probe_n_tokens(m, ctx, _MtmdInputTextV2, _PROBE_EMBEDDED_NUL)
    if embedded is None:
        return None
    return _MtmdInputTextV2 if embedded > 0 else _MtmdInputTextV1


class MtmdContext:
    """A loaded mmproj bound to a text model, able to evaluate image prompts into
    that model's llama context."""

    # Always overwritten by __init__ with the PROBED layout (and __init__ refuses
    # to construct at all when the probe is inconclusive, so this default is
    # unreachable in production). It exists so an instance built by other means -
    # tests bypass the native-loading __init__ deliberately - marshals with the
    # current upstream layout instead of raising AttributeError.
    _input_text_class: type = _MtmdInputTextV2

    # Same reason as the class default above: __init__ always sets it, but an
    # instance built without __init__ must not AttributeError. False is the
    # conservative default - it means "do not suggest a CPU retry".
    on_gpu: bool = False

    def __init__(self, mmproj_path: str, model_ptr: int) -> None:
        self._m = _load_lib()
        self._mmproj_path = mmproj_path
        self._model_ptr = model_ptr
        self.on_gpu = True
        self._ctx = self._open(use_gpu=True)
        if not self._ctx:
            # A GPU-side refusal at INIT (the backend cannot take this projector at
            # all) is exactly the case the old blanket CPU override existed for, so
            # degrade here rather than losing vision - but say so, and only after
            # the real path was actually tried.
            from localm.debuglog import logger
            logger.warning(
                "mtmd: the vision projector could not be loaded onto the GPU; "
                "falling back to CPU encoding, which is much slower on large "
                "images. Set LOCALM_MTMD_CPU=1 to skip the GPU attempt entirely.")
            self.on_gpu = False
            self._ctx = self._open(use_gpu=False)
        if not self._ctx:
            raise MtmdUnavailable(
                f"mtmd_init_from_file returned NULL for {mmproj_path} "
                "(mmproj incompatible with this model or build)")
        self.supports_vision = bool(self._m.mtmd_support_vision(self._ctx))
        self.marker = self._m.mtmd_default_marker().decode("utf-8")

        # Resolve the mtmd_input_text layout once per process. Needs a live
        # context (mtmd_tokenize takes one), so it happens here rather than in
        # _load_lib; the answer is a property of the LIBRARY, so it is cached
        # globally and later contexts reuse it.
        global _input_text_class
        if _input_text_class is None:
            _input_text_class = _detect_input_text_class(self._m, self._ctx)
            if _input_text_class is None:
                self.free()
                raise MtmdUnavailable(
                    "could not determine this build's mtmd_input_text layout "
                    "(the text-length probe was inconclusive); refusing to guess, "
                    "because guessing wrong silently truncates every image prompt")
            from localm.debuglog import logger
            logger.debug("mtmd input_text layout: %s", _input_text_class.__name__)
        self._input_text_class = _input_text_class

    def _open(self, *, use_gpu: bool) -> Optional[int]:
        """Create the native mtmd context, on GPU or CPU.

        Only two fields of the opaque params buffer are touched, both at offsets
        measured against the shipped runtime (see the module docstring):
        ``use_gpu`` at byte 0 and ``n_threads`` at byte 4.

        ``n_threads`` matters even on the GPU path (parts of preprocessing stay on
        the host) and matters enormously on the CPU one: mtmd's own default is a
        flat 4 regardless of the machine, so a 12-thread box was using a third of
        itself. ``LOCALM_MTMD_CPU=1`` is an escape hatch for a build/GPU where the
        GPU encode is broken in a way that only shows up mid-encode."""
        if use_gpu and os.environ.get("LOCALM_MTMD_CPU"):
            return None
        params = self._m.mtmd_context_params_default()
        buf = ctypes.cast(ctypes.byref(params), ctypes.POINTER(ctypes.c_uint8))
        buf[0] = 1 if use_gpu else 0
        ctypes.cast(ctypes.byref(params, 4),
                    ctypes.POINTER(ctypes.c_int32))[0] = _encode_threads()
        return self._m.mtmd_init_from_file(
            self._mmproj_path.encode("utf-8"), self._model_ptr, params)

    def retry_on_cpu(self) -> bool:
        """Rebuild this context on the CPU after a GPU encode failed at RUNTIME.

        The gfx1030 / RDNA2 hipBLAS BF16 GEMM failure the old blanket override was
        written for does NOT show up at init - it surfaces mid-encode - so an
        init-time fallback alone would not cover it. Returns False when already on
        the CPU (so the caller reports the real error instead of looping)."""
        if not self.on_gpu:
            return False
        from localm.debuglog import logger
        logger.warning(
            "mtmd: the GPU vision encode failed; rebuilding the projector on the "
            "CPU and retrying. Image replies will be slower until the model is "
            "reloaded.")
        try:
            self._m.mtmd_free(self._ctx)
        except Exception:   # noqa: BLE001 - freeing a wedged context must not mask the retry
            pass
        self._ctx = None
        self.on_gpu = False
        self._ctx = self._open(use_gpu=False)
        return bool(self._ctx)

    def eval_into(self, llama_ctx: int, prompt: str,
                  images: List[Tuple[int, int, bytes]], *,
                  add_special: bool, n_batch: Optional[int] = None) -> int:
        """Tokenize *prompt* (which contains one ``self.marker`` per image, in
        order) together with *images* (each ``(width, height, rgb_bytes)``) and
        evaluate the resulting text+image chunks into *llama_ctx*'s KV cache from
        position 0. Returns the new n_past (with logits at the last position, ready
        for sampling). Raises RuntimeError on a tokenize/eval failure.

        RAG-VISION-1: *n_batch* defaults to the LIVE context's own configured
        batch size (via ``llama_n_ctx``, capped the same way llama.py's own
        context construction caps it) rather than a fixed 512 - the caller's
        real context can be configured larger (up to 2048), and asking mtmd to
        micro-batch smaller than what the context was built for is a latent
        mismatch, not just a performance nit. An explicit *n_batch* still wins,
        for a caller that knows its own real batch size precisely."""
        m = self._m
        ctx_n_ctx = api.llama_n_ctx(llama_ctx)
        if n_batch is None:
            n_batch = min(ctx_n_ctx, 2048) if ctx_n_ctx else 512
        bitmaps = []
        try:
            for (w, h, rgb) in images:
                bmp = m.mtmd_bitmap_init(w, h, rgb)
                if not bmp:
                    raise VisionInputError("mtmd_bitmap_init failed (bad image buffer)")
                bitmaps.append(bmp)
            chunks = m.mtmd_input_chunks_init()
            if not chunks:
                raise VisionInputError("mtmd_input_chunks_init failed")
            try:
                raw = prompt.encode("utf-8")
                itext = _make_input_text(
                    self._input_text_class, raw, add_special, True)
                arr = (ctypes.c_void_p * len(bitmaps))(*bitmaps)
                rc = m.mtmd_tokenize(self._ctx, chunks, ctypes.addressof(itext),
                                     arr, len(bitmaps))
                if rc != 0:
                    raise VisionInputError(
                        f"the vision projector could not process this image "
                        f"(mtmd_tokenize rc={rc}). See the debug log for the "
                        f"native reason.")
                new_n_past = ctypes.c_int32(0)
                rc2 = m.mtmd_helper_eval_chunks(
                    self._ctx, llama_ctx, chunks, 0, 0, n_batch, True,
                    ctypes.byref(new_n_past))
                if rc2 != 0:
                    # On the GPU path this is the shape the documented gfx1030 /
                    # RDNA2 hipBLAS BF16 failure takes, so tell the caller a CPU
                    # retry is worth one attempt rather than failing the request.
                    exc = MtmdGpuEncodeFailed if self.on_gpu else VisionInputError
                    raise exc(
                        f"the vision projector could not evaluate this image "
                        f"(mtmd_helper_eval_chunks rc={rc2})")
                pos = int(new_n_past.value)
                # RAG-VISION-1: the generation loop trusts this position as the
                # base for every subsequent single-token decode with zero prior
                # sanity check - if a native call under/over-reports how many KV
                # positions the image actually consumed (a real risk: idefics3
                # and llava-family projectors emit very different image-token
                # counts, and this binding's ABI note above already flags mtmd's
                # struct layout as unverified across llama.cpp versions),
                # generation would silently continue from a corrupted position
                # instead of failing loudly. A garbage pos manifests exactly like
                # the degenerate/repeated-token output this check exists to catch.
                if pos <= 0 or (ctx_n_ctx and pos > ctx_n_ctx):
                    raise VisionInputError(
                        f"mtmd image eval returned an implausible position "
                        f"(new_n_past={pos}, context size={ctx_n_ctx}) - refusing "
                        f"to generate from a likely-corrupted KV state")
                return pos
            finally:
                m.mtmd_input_chunks_free(chunks)
        finally:
            for bmp in bitmaps:
                m.mtmd_bitmap_free(bmp)

    def free(self) -> None:
        if getattr(self, "_ctx", None):
            try:
                self._m.mtmd_free(self._ctx)
            except Exception:
                pass
            self._ctx = None
