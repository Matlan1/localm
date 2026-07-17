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
  byte 0 - ``use_gpu`` (the first field) - which we force to 0 so the clip encode
  runs on CPU. On Win64 a struct that large is passed by hidden pointer, so an
  over-sized buffer is safe regardless of the real field layout.
* CPU clip is the universally-safe path: gfx1030 / RDNA2 hipBLAS fails a BF16 GEMM
  (CUBLAS_STATUS_INTERNAL_ERROR) on a BF16 mmproj. The text model still runs on the
  GPU; only the (one-off, per-image) projector encode is on CPU.
* the image is decoded to raw RGB by the caller and passed to the clean-signature
  ``mtmd_bitmap_init(w, h, rgb)`` - NOT ``mtmd_helper_bitmap_init_from_buf``, whose
  return type drifted to a by-value wrapper in newer builds.

Verified end-to-end on gfx1030 with gemma-4 + mmproj-BF16: a test image was
described correctly. See dev-notes for the standalone probe this was lifted from.
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Optional, Tuple

from . import _api as api


class _MtmdParams(ctypes.Structure):
    # Over-allocated opaque buffer (the real struct is well under this); 8-byte
    # aligned via c_uint64. Only byte 0 (use_gpu) is ever touched - see module docs.
    _fields_ = [("_buf", ctypes.c_uint64 * 32)]   # 256 bytes


class _MtmdInputText(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p),
                ("add_special", ctypes.c_bool),
                ("parse_special", ctypes.c_bool)]


_lib: Optional[ctypes.CDLL] = None


class MtmdUnavailable(RuntimeError):
    """Raised when mtmd.dll or the mmproj cannot be loaded - the GGUF backend then
    stays text-only rather than crashing."""


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
    m.mtmd_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(_MtmdInputText),
                                ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    m.mtmd_helper_eval_chunks.restype = ctypes.c_int32
    m.mtmd_helper_eval_chunks.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool,
        ctypes.POINTER(ctypes.c_int32)]
    _lib = m
    return m


class MtmdContext:
    """A loaded mmproj bound to a text model, able to evaluate image prompts into
    that model's llama context."""

    def __init__(self, mmproj_path: str, model_ptr: int) -> None:
        self._m = _load_lib()
        params = self._m.mtmd_context_params_default()
        # Force use_gpu (first field, byte 0) off -> CPU clip (see module docs).
        ctypes.cast(ctypes.byref(params), ctypes.POINTER(ctypes.c_uint8))[0] = 0
        self._ctx = self._m.mtmd_init_from_file(
            mmproj_path.encode("utf-8"), model_ptr, params)
        if not self._ctx:
            raise MtmdUnavailable(
                f"mtmd_init_from_file returned NULL for {mmproj_path} "
                "(mmproj incompatible with this model or build)")
        self.supports_vision = bool(self._m.mtmd_support_vision(self._ctx))
        self.marker = self._m.mtmd_default_marker().decode("utf-8")

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
                    raise RuntimeError("mtmd_bitmap_init failed (bad image buffer)")
                bitmaps.append(bmp)
            chunks = m.mtmd_input_chunks_init()
            if not chunks:
                raise RuntimeError("mtmd_input_chunks_init failed")
            try:
                itext = _MtmdInputText(prompt.encode("utf-8"), add_special, True)
                arr = (ctypes.c_void_p * len(bitmaps))(*bitmaps)
                rc = m.mtmd_tokenize(self._ctx, chunks, ctypes.byref(itext),
                                     arr, len(bitmaps))
                if rc != 0:
                    raise RuntimeError(f"mtmd_tokenize failed (rc={rc})")
                new_n_past = ctypes.c_int32(0)
                rc2 = m.mtmd_helper_eval_chunks(
                    self._ctx, llama_ctx, chunks, 0, 0, n_batch, True,
                    ctypes.byref(new_n_past))
                if rc2 != 0:
                    raise RuntimeError(f"mtmd image eval failed (rc={rc2})")
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
                    raise RuntimeError(
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
