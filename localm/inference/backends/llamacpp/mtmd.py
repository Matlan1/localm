# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-process multimodal (vision) for the GGUF backend, via the bundled mtmd.dll
(llama.cpp ``libmtmd``).

Loads an mmproj (vision projector) alongside the text model and evaluates an
image+text prompt straight into the llama KV cache, so the GGUF backend can answer
about images.

ABI strategy - the bundled runtime ships NO headers and the mtmd C ABI has drifted
across llama.cpp versions, so this binding avoids version-specific struct layouts:

* ``mtmd_context_params`` is treated as an OVER-ALLOCATED opaque buffer. The
  exported ``mtmd_context_params_default()`` is called and passed through
  UNMODIFIED except two leading fields (``use_gpu`` at byte 0, ``n_threads`` at
  byte 4). On Win64 a struct that large is passed by hidden pointer, so an
  over-sized buffer is safe regardless of the real field layout.
* THE PROJECTOR RUNS ON THE GPU, falling back to the CPU only when the GPU path
  actually fails, and saying so in the log. ``n_threads`` is set explicitly;
  mtmd defaults it to 4 regardless of the machine.
* the image is decoded to raw RGB by the caller and passed to the clean-signature
  ``mtmd_bitmap_init(w, h, rgb)``, NOT ``mtmd_helper_bitmap_init_from_buf``, whose
  return type drifted to a by-value wrapper in newer builds.
* ``mtmd_input_text`` is the one struct this module must pass by value and it did
  drift, so both layouts are bound and the live one is detected at load time -
  see :func:`_detect_input_text_class`.
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Optional, Tuple

from ..base import VisionInputError
from . import _api as api


class _MtmdParams(ctypes.Structure):
    # Over-allocated opaque buffer (the real struct is well under this); 8-byte
    # aligned via c_uint64. Only byte 0 (use_gpu) is ever touched.
    _fields_ = [("_buf", ctypes.c_uint64 * 32)]   # 256 bytes


class _MtmdInputTextV1(ctypes.Structure):
    """``mtmd_input_text`` in the layout with no ``text_len``.

    The text is NUL-terminated: the tokenizer does ``input_text = text->text``."""

    _fields_ = [("text", ctypes.c_char_p),
                ("add_special", ctypes.c_bool),
                ("parse_special", ctypes.c_bool)]


class _MtmdInputTextV2(ctypes.Structure):
    """``mtmd_input_text`` in the layout with an explicit ``text_len``.

    ``size_t text_len`` is the SECOND field, and the tokenizer does
    ``input_text.assign(text->text, text->text_len)``. Passing the V1 layout to
    a V2 build is silently catastrophic: the callee reads ``text_len`` out of
    V1's ``add_special``/``parse_special`` bytes plus padding, so with both
    flags true it reads 257 and truncates every prompt to 257 bytes, dropping
    the image marker for any prompt with a system preamble. It also reads the
    two flags from offsets 16/17, past the end of V1's 16 bytes."""

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
    """Raised when mtmd.dll or the mmproj cannot be loaded. The GGUF backend
    then stays text-only."""


class MtmdGpuEncodeFailed(VisionInputError):
    """A GPU projector encode failed at runtime. Distinct from a plain
    :class:`VisionInputError` so the caller knows a CPU retry is worth one
    attempt. The caller owns the KV cache, which the failed evaluation dirtied,
    so the retry cannot happen inside ``eval_into``."""


def _encode_threads() -> int:
    """Threads for the projector: the CPU count minus one, leaving a core for
    the rest of the server. Falls back to mtmd's own default of 4 when the CPU
    count is unknown."""
    n = os.cpu_count() or 4
    return max(1, n - 1)


_MTMD_DEVICE_ENV = "MTMD_BACKEND_DEVICE"


def _resolve_backend_device_name(gpu_index: int) -> Optional[str]:
    """The ggml device NAME (e.g. ``"Vulkan1"``) to pin the projector to for
    llama.cpp GPU-list index *gpu_index*, or None when localm cannot determine it
    UNAMBIGUOUSLY - in which case the caller leaves ``MTMD_BACKEND_DEVICE`` unset
    and clip keeps today's behaviour (the first GPU-type device).

    ``mtmd_context_params`` has no device field, so the only selector is the
    process environment variable ``MTMD_BACKEND_DEVICE``, read in clip_ctx's
    constructor. Unset, clip takes
    ``ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)``
    unconditionally, with no awareness of tensor_split or main_gpu.

    THE INDEX SPACE. *gpu_index* is ``mp.main_gpu``, which indexes llama.cpp's
    OWN ``model->devices``, whereas ``compute_devices()`` reports ggml's FULL
    registry. Those two sequences are NOT interchangeable:
    ``llama_prepare_model_devices`` hoists RPC devices to the front,
    deduplicates GPUs by device_id, SKIPS ACCEL entirely, and admits iGPUs only
    when no discrete GPU was found (and then at most one). A META device is not
    skipped but ``GGML_ABORT``s the load outright, so it never reaches that list
    either. Against the shipped runtime:

    * RPC devices need an explicit ``ggml_backend_rpc_add_server(endpoint)``;
      localm never calls it, so shipping ggml-rpc registers no device.
    * ggml-vulkan dedups one physical GPU seen under two drivers, by
      deviceUUID/deviceLUID, so a device_id duplicate cannot reach the registry
      from a single-backend build.
    * An INTEGRATED GPU can and routinely does appear: ggml-vulkan enumerates it
      and types it ``GGML_BACKEND_DEVICE_TYPE_IGPU``, while llama.cpp drops it
      whenever any discrete GPU exists. The registry then contains a device
      llama.cpp's list does not, and every index past it is wrong.

    So this refuses unless EVERY non-CPU device is a plain ``GPU``. Under that
    condition the two sequences are identical and ``non_cpu[gpu_index]`` is
    exact.

    Index 0 returns None: clip's own default already picks that device."""
    if gpu_index <= 0:
        return None      # already clip's default; nothing to change
    from localm.debuglog import logger
    try:
        from . import _loader
        devices = _loader.compute_devices()
    except Exception as e:      # noqa: BLE001 - a probe failure must not lose vision
        logger.info(
            "mtmd: could not read the ggml device registry (%s); leaving the "
            "vision projector on the default GPU device", type(e).__name__)
        return None

    non_cpu = [(name, dev_type) for (name, dev_type) in devices
               if dev_type != _loader.GGML_DEV_TYPE_CPU]
    reason: Optional[str] = None
    if not non_cpu:
        reason = "the runtime registers no GPU device"
    elif any(t != _loader.GGML_DEV_TYPE_GPU for _, t in non_cpu):
        # The iGPU/ACCEL case above: llama.cpp's device list is a filtered
        # subsequence of this one, so the index cannot be mapped by position.
        reason = ("the device registry mixes device types (%s), so localm cannot "
                  "map a device index to a name unambiguously"
                  % ", ".join(f"{n}:type{t}" for n, t in non_cpu))
    elif gpu_index >= len(non_cpu):
        reason = (f"device index {gpu_index} is out of range "
                  f"({len(non_cpu)} GPU device(s) registered)")
    elif not non_cpu[gpu_index][0]:
        reason = f"device index {gpu_index} reported an empty name"
    if reason is not None:
        # The user asked for a non-default device and the projector is NOT going
        # there. INFO, so the always-on ring buffer carries it into a bug report.
        logger.info(
            "mtmd: leaving the vision projector on the default GPU device "
            "(device 0) rather than the configured device %d - %s",
            gpu_index, reason)
        return None
    return non_cpu[gpu_index][0]


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
    # The mtmd_input_text pointer is bound as an untyped void*: which of the two
    # layouts is live is only known after _detect_input_text_class runs, and
    # re-pointing argtypes per call would mutate shared state on this CDLL's
    # function object. Callers pass ctypes.addressof(struct).
    m.mtmd_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    # Exported by both ABI eras; used by the layout probe below.
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
# that length is load-bearing: a V1 build reads add_special from byte 8 and
# parse_special from byte 9, which under the V2 struct are the low two bytes of
# text_len. At 256 those read as add_special=False, parse_special=True.
# add_special must come out False, or a V1 build prepends BOS to the empty
# string it sees and returns 1 token instead of 0, destroying the discriminator.
#
# llama.cpp's tokenizer trace echoes whatever it is handed via an
# "add_text: <text>" line on stderr, so "add_text: aaaa..." followed by an empty
# "add_text: " in a debug load log is these two probe calls, in order - not a
# leaked user prompt. llama.py's caller wraps this constructor call so neither
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

    Measures the property directly - does this build read ``text_len`` or
    ``strlen`` - rather than correlating with a symbol; the layout change added
    no new export.

    Two calls that differ only in a leading NUL byte:

    * CONTROL ``"a"*256`` must tokenize to > 0 tokens, so a build where
      tokenize always yields 0 reads as a broken probe, not as "uses strlen".
    * DISCRIMINATOR ``"\\0" + "a"*255``, same 256 bytes. A build that honours
      text_len tokenizes all 256 and returns > 0; a build using strlen stops at
      the leading NUL, tokenizes nothing and returns 0.

    Returns None when inconclusive; the caller then keeps the model text-only
    with a logged reason and never guesses."""
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

    # Always overwritten by __init__ with the PROBED layout; __init__ refuses to
    # construct when the probe is inconclusive. The default lets an instance
    # built without __init__ marshal with the current upstream layout instead of
    # raising AttributeError.
    _input_text_class: type = _MtmdInputTextV2

    # __init__ always sets it. False means "do not suggest a CPU retry".
    on_gpu: bool = False

    # __init__ always sets it. 0 means "the device clip would have picked
    # anyway", so an instance built without __init__ pins nothing.
    _gpu_index: int = 0

    def __init__(self, mmproj_path: str, model_ptr: int,
                 gpu_index: int = 0) -> None:
        self._m = _load_lib()
        self._mmproj_path = mmproj_path
        self._model_ptr = model_ptr
        # The text model's resolved primary device (llama_model_params.main_gpu,
        # after discover.apply_main_gpu/apply_gpu_split have validated it and
        # forced it inside any configured split). Keeps the projector off a
        # device the user's configuration excluded.
        try:
            self._gpu_index = int(gpu_index)
        except (TypeError, ValueError):
            # Degrade to 0, leaving clip's own choice alone, and say so.
            from localm.debuglog import logger
            logger.warning(
                "mtmd: ignoring an unusable projector device index %r; the vision "
                "projector will use the default GPU device", gpu_index)
            self._gpu_index = 0
        self.on_gpu = True
        self._ctx = self._open(use_gpu=True)
        if not self._ctx:
            # A GPU-side refusal at INIT: degrade to CPU rather than losing
            # vision, and say so.
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

        # Resolve the mtmd_input_text layout once per process. It needs a live
        # context (mtmd_tokenize takes one), so it happens here rather than in
        # _load_lib. The answer is a property of the LIBRARY, so it is cached
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

        Only two fields of the opaque params buffer are touched: ``use_gpu`` at
        byte 0 and ``n_threads`` at byte 4. ``n_threads`` applies on both paths;
        mtmd's own default is a flat 4 regardless of the machine.
        ``LOCALM_MTMD_CPU=1`` skips the GPU attempt entirely.

        DEVICE PLACEMENT is not a params field: clip reads the process
        environment variable ``MTMD_BACKEND_DEVICE``, so it is set around THIS
        CALL ONLY and restored in a ``finally``. Clip gates the read on
        ``use_gpu``, so the CPU attempt and :meth:`retry_on_cpu` never consult
        it, and an already-set value belongs to the USER and is never
        overwritten.

        That set/restore is NOT serialised. Two concurrent ``_open`` calls in
        ONE process would race on the variable, so a caller that loads two
        mmprojs at once needs a lock."""
        if use_gpu and os.environ.get("LOCALM_MTMD_CPU"):
            return None
        params = self._m.mtmd_context_params_default()
        buf = ctypes.cast(ctypes.byref(params), ctypes.POINTER(ctypes.c_uint8))
        buf[0] = 1 if use_gpu else 0
        ctypes.cast(ctypes.byref(params, 4),
                    ctypes.POINTER(ctypes.c_int32))[0] = _encode_threads()

        # An explicitly exported MTMD_BACKEND_DEVICE is the user's own choice and
        # is never corrected. Resolved only on the GPU attempt, the one path clip
        # reads the variable on.
        #
        # PRESENCE, not truthiness: an exported-but-EMPTY value is still the
        # user's variable, and clip does not treat it as unset (it takes the
        # getenv branch, fails to init by that name and warns). The key is only
        # ever set when it was absent, so the pop below cannot delete a value
        # somebody else owned.
        device_name = None
        if use_gpu and _MTMD_DEVICE_ENV not in os.environ:
            device_name = _resolve_backend_device_name(self._gpu_index)
        if device_name is not None:
            from localm.debuglog import logger
            # INFO reaches the always-on ring buffer, so a bug report carries
            # which device the projector landed on without --debug.
            logger.info(
                "mtmd: pinning the vision projector to ggml device %r (the text "
                "model's configured primary device %d) via %s",
                device_name, self._gpu_index, _MTMD_DEVICE_ENV)
            os.environ[_MTMD_DEVICE_ENV] = device_name
        try:
            return self._m.mtmd_init_from_file(
                self._mmproj_path.encode("utf-8"), self._model_ptr, params)
        finally:
            # Only ever unsets what THIS call set: the branch above does not run
            # when the variable already had a value.
            if device_name is not None:
                os.environ.pop(_MTMD_DEVICE_ENV, None)

    def retry_on_cpu(self) -> bool:
        """Rebuild this context on the CPU after a GPU encode failed at RUNTIME.

        Some GPU encode failures (the RDNA2 hipBLAS BF16 GEMM one, for example)
        surface mid-encode rather than at init, so the init-time fallback does
        not cover them. Returns False when already on the CPU, so the caller
        reports the real error instead of looping."""
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

        *n_batch* defaults to the LIVE context's own configured batch size (via
        ``llama_n_ctx``, capped at 2048 the same way llama.py's own context
        construction caps it), falling back to 512 when that cannot be read. An
        explicit *n_batch* wins."""
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
                    # On the GPU path, raise the type that tells the caller a CPU
                    # retry is worth one attempt.
                    exc = MtmdGpuEncodeFailed if self.on_gpu else VisionInputError
                    raise exc(
                        f"the vision projector could not evaluate this image "
                        f"(mtmd_helper_eval_chunks rc={rc2})")
                pos = int(new_n_past.value)
                # The generation loop uses this position as the base for every
                # subsequent single-token decode, so an under- or over-reported
                # count is refused here rather than generating from a corrupted
                # KV state.
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
