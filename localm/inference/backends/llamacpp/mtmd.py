# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-process multimodal (vision) for the GGUF backend, via the bundled mtmd.dll
(llama.cpp ``libmtmd``).

Loads an mmproj (vision projector) alongside the text model and evaluates an
image+text prompt straight into the llama KV cache, so the GGUF backend can answer
about images instead of refusing them.

ABI strategy (the bundled runtime ships NO headers and the mtmd C ABI has drifted
across llama.cpp versions, so this binding avoids version-specific struct layouts):

* ``mtmd_context_params`` is treated as an OVER-ALLOCATED opaque buffer. We call the
  exported ``mtmd_context_params_default()`` and pass it through UNMODIFIED except
  two leading fields: ``use_gpu`` at byte 0 and ``n_threads`` at byte 4. On Win64 a
  struct that large is passed by hidden pointer, so an over-sized buffer is safe
  regardless of the real field layout.
* THE PROJECTOR RUNS ON THE GPU, falling back to CPU only when the GPU path
  ACTUALLY fails, which is logged plainly. A GPU failure is specific to
  the projector in use - gfx1030 / RDNA2 hipBLAS fails a BF16 GEMM
  (CUBLAS_STATUS_INTERNAL_ERROR) on a BF16 mmproj - so the fallback is per
  context, not a blanket override: it reopens this context on the CPU and stays
  there until the model is reloaded. ``n_threads`` is set explicitly: mtmd defaults it to 4
  regardless of the machine, which makes the CPU path far slower than it needs to
  be.
* the image is decoded to raw RGB by the caller and passed to the clean-signature
  ``mtmd_bitmap_init(w, h, rgb)`` - NOT ``mtmd_helper_bitmap_init_from_buf``, whose
  return type drifted to a by-value wrapper in newer builds.
* ``mtmd_input_text`` DID drift and cannot be avoided (it is the one struct this
  module must pass by value), so both layouts are bound and the live one is
  detected at load time - see :func:`_detect_input_text_class`.
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
    """``mtmd_input_text`` BEFORE llama.cpp 4114ba18b.

    The text is NUL-terminated: the tokenizer does ``input_text = text->text``."""

    _fields_ = [("text", ctypes.c_char_p),
                ("add_special", ctypes.c_bool),
                ("parse_special", ctypes.c_bool)]


class _MtmdInputTextV2(ctypes.Structure):
    """``mtmd_input_text`` FROM llama.cpp 4114ba18b onward: an explicit ``text_len``.

    That commit inserted ``size_t text_len`` as the SECOND field and switched the
    tokenizer to ``input_text.assign(text->text, text->text_len)``. Passing the V1
    layout to a V2 build is silently catastrophic rather than merely wrong: the
    callee reads ``text_len`` out of V1's ``add_special``/``parse_special`` bytes
    plus padding, so with both flags true it reads 257 and TRUNCATES EVERY PROMPT
    TO 257 BYTES - which drops the image marker for any prompt with a system
    preamble, yielding "number of media markers in text (0) does not match number
    of bitmaps (1)". It also reads the two flags from offsets 16/17, past the end
    of V1's 16 bytes."""

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


_MTMD_DEVICE_ENV = "MTMD_BACKEND_DEVICE"


def _resolve_backend_device_name(gpu_index: int) -> Optional[str]:
    """The ggml device NAME (e.g. ``"Vulkan1"``) to pin the projector to for
    llama.cpp GPU-list index *gpu_index*, or None when localm cannot determine it
    UNAMBIGUOUSLY - in which case the caller leaves ``MTMD_BACKEND_DEVICE`` unset
    and clip keeps today's behaviour (the first GPU-type device).

    ``mtmd_context_params`` has no device field, so the only selector is the
    process environment variable ``MTMD_BACKEND_DEVICE``, read in clip_ctx's
    constructor (upstream ``tools/mtmd/clip.cpp:184-195`` at b10361). Unset, clip
    takes ``ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)``
    unconditionally, with zero awareness of tensor_split or main_gpu, so the
    projector weights and its compute buffer land on device 0 even when the
    configured split excludes that card.

    THE INDEX SPACE, WHICH THE TYPE CHECK BELOW IS THE WHOLE GUARD FOR. *gpu_index*
    is ``mp.main_gpu``, which indexes llama.cpp's OWN ``model->devices``, whereas
    ``compute_devices()`` reports ggml's FULL registry. Those two sequences are NOT
    interchangeable: ``llama_prepare_model_devices`` (upstream ``src/llama.cpp``
    :149-296) hoists RPC devices to the front, deduplicates GPUs by device_id,
    SKIPS ACCEL entirely, and admits iGPUs only when no discrete GPU was found
    (and then at most one). A META device is not skipped but ``GGML_ABORT``s the
    load outright, so it can never reach that list either. Against the shipped
    runtime, two of those cannot arise here and one can:

    * RPC devices need an explicit ``ggml_backend_rpc_add_server(endpoint)``
      (``ggml-rpc.cpp:1949-1951``); localm never calls it, so merely shipping
      ggml-rpc registers no device.
    * ggml-vulkan already dedups one physical GPU seen under two drivers, by
      deviceUUID/deviceLUID (``ggml-vulkan.cpp:7446-7470``), so a device_id
      duplicate cannot reach the registry from a single-backend build.
    * An INTEGRATED GPU can and routinely does: ggml-vulkan enumerates it
      (``:7444``) and types it ``GGML_BACKEND_DEVICE_TYPE_IGPU`` (``:17878``),
      while llama.cpp drops it whenever any discrete GPU exists. On a laptop, or
      any desktop with iGPU-bearing silicon, the registry therefore contains a
      device llama.cpp's list does not - and every index past it is wrong.

    So: refuse unless EVERY non-CPU device is a plain ``GPU``. Under that condition
    the two sequences are identical and ``non_cpu[gpu_index]`` is exact. Refusing
    leaves the projector on clip's own default.

    Index 0 returns None rather than resolving to the same device: clip's own
    default already picks it, so there is nothing to correct."""
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
        # Surface the decision: the user asked for a non-default device and the
        # projector is NOT going there. INFO so the always-on ring buffer carries
        # it into a bug report, rather than a WARNING the user cannot act on.
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
# The filler carries punctuation every few bytes, so its longest unbroken run of
# one character class is 2. A 256-byte run of plain letters is above the length
# at which several pre-tokenizer regexes abort the process (see
# pretokenizer_guard), and this probe runs before any caller text exists, so the
# guard cannot cover it. Only three things about these two values are
# load-bearing: equal byte length, a control that tokenizes to more than 0
# tokens, and a leading NUL on the discriminator.
_PROBE_FILLER = b"ab. " * 64
_PROBE_CONTROL = _PROBE_FILLER
_PROBE_EMBEDDED_NUL = b"\x00" + _PROBE_FILLER[:255]


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
    or ``strlen`` - rather than correlating with a symbol. The upstream change
    added no new export, so a symbol probe (the approach ``_abi.py`` can use for
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
    # a test that bypasses the native-loading __init__ - marshals with the
    # current upstream layout instead of raising AttributeError.
    _input_text_class: type = _MtmdInputTextV2

    # Same reason as the class default above: __init__ always sets it, but an
    # instance built without __init__ must not AttributeError. False is the
    # conservative default - it means "do not suggest a CPU retry".
    on_gpu: bool = False

    # Same reason as the two defaults above: __init__ always sets it, and 0 is the
    # conservative value - it means "the device clip would have picked anyway", so
    # an instance built without __init__ pins nothing.
    _gpu_index: int = 0

    def __init__(self, mmproj_path: str, model_ptr: int,
                 gpu_index: int = 0) -> None:
        self._m = _load_lib()
        self._mmproj_path = mmproj_path
        self._model_ptr = model_ptr
        # The text model's resolved primary device (llama_model_params.main_gpu,
        # after discover.apply_main_gpu/apply_gpu_split have validated it and
        # forced it inside any configured split). Used to keep the projector off a
        # device the user's configuration excluded - see _resolve_backend_device_name.
        try:
            self._gpu_index = int(gpu_index)
        except (TypeError, ValueError):
            # Unreachable from the one production caller (llama.py passes an int
            # derived from mp.main_gpu), so this is a programming error, not a
            # runtime condition. Degrade to 0 (= leave clip's own choice alone)
            # rather than costing the user vision over a placement hint, and warn
            # rather than swallowing the bad input.
            from localm.debuglog import logger
            logger.warning(
                "mtmd: ignoring an unusable projector device index %r; the vision "
                "projector will use the default GPU device", gpu_index)
            self._gpu_index = 0
        self.on_gpu = True
        _cpu_requested = bool(os.environ.get("LOCALM_MTMD_CPU"))
        self._ctx = self._open(use_gpu=True)
        if not self._ctx:
            # A GPU-side refusal at INIT (the backend cannot take this projector
            # at all) degrades to CPU rather than losing vision, and is logged,
            # only after the GPU path was actually tried. _open() returns None
            # without trying anything when LOCALM_MTMD_CPU is set: a requested
            # skip, not a failure, so it gets a different message.
            from localm.debuglog import logger
            if _cpu_requested:
                logger.info(
                    "mtmd: using CPU encoding for the vision projector, as "
                    "requested by LOCALM_MTMD_CPU - much slower on large images.")
            else:
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

        Only two fields of the opaque params buffer are touched (see the module
        docstring): ``use_gpu`` at byte 0 and ``n_threads`` at byte 4.

        ``n_threads`` applies on the GPU path too (parts of preprocessing stay on
        the host) and dominates the CPU one: mtmd's own default is a flat 4
        regardless of the machine. ``LOCALM_MTMD_CPU=1`` is an escape hatch for a
        build/GPU where the GPU encode is broken in a way that only shows up
        mid-encode.

        DEVICE PLACEMENT is not a params field at all - clip reads the process
        environment variable ``MTMD_BACKEND_DEVICE`` instead - so it is set around
        THIS CALL ONLY and restored in a ``finally``: it is process-global state
        that would otherwise leak into every later library call, clip gates the
        read on ``use_gpu`` so the CPU attempt and :meth:`retry_on_cpu` never
        consult it, and an already-set value belongs to the USER and is never
        overwritten (see below).

        That set/restore is NOT serialised. It holds because the projector is
        loaded once, inline, during a model load in a worker that is doing nothing
        else at the time. Two concurrent ``_open`` calls in ONE process would race
        on the variable (the second's restore could drop the first's value), so a
        caller that loads two mmprojs at once needs a lock."""
        if use_gpu and os.environ.get("LOCALM_MTMD_CPU"):
            return None
        params = self._m.mtmd_context_params_default()
        buf = ctypes.cast(ctypes.byref(params), ctypes.POINTER(ctypes.c_uint8))
        buf[0] = 1 if use_gpu else 0
        ctypes.cast(ctypes.byref(params, 4),
                    ctypes.POINTER(ctypes.c_int32))[0] = _encode_threads()

        # An explicitly exported MTMD_BACKEND_DEVICE is the user's own choice and
        # outranks anything derived from config: it is never overwritten. Only
        # resolved at all on the GPU attempt, the one path clip reads the
        # variable on.
        #
        # PRESENCE, not truthiness: an exported-but-EMPTY value is still the user's
        # variable, and it is not equivalent to unset for clip either (it takes the
        # getenv branch, fails to init by that name and warns). Testing membership
        # also makes the pop below provably safe - we only ever set the key when it
        # was absent, so unsetting it cannot delete a value somebody else owned.
        device_name = None
        if use_gpu and _MTMD_DEVICE_ENV not in os.environ:
            device_name = _resolve_backend_device_name(self._gpu_index)
        if device_name is not None:
            from localm.debuglog import logger
            # Surface which device the projector landed on. INFO reaches the
            # always-on ring buffer, so a bug report carries it without --debug.
            logger.info(
                "mtmd: pinning the vision projector to ggml device %r (the text "
                "model's configured primary device %d) via %s",
                device_name, self._gpu_index, _MTMD_DEVICE_ENV)
            os.environ[_MTMD_DEVICE_ENV] = device_name
        try:
            return self._m.mtmd_init_from_file(
                self._mmproj_path.encode("utf-8"), self._model_ptr, params)
        finally:
            # Only ever unset what THIS call set: the branch above does not run
            # when the variable already had a value, so there is nothing to
            # restore and a user's own export is never clobbered.
            if device_name is not None:
                os.environ.pop(_MTMD_DEVICE_ENV, None)

    def retry_on_cpu(self) -> bool:
        """Rebuild this context on the CPU after a GPU encode failed at RUNTIME.

        The gfx1030 / RDNA2 hipBLAS BF16 GEMM failure does NOT show up at init -
        it surfaces mid-encode - so an init-time fallback alone does not cover it.
        Returns False when already on the CPU (so the caller reports the real
        error instead of looping)."""
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
        ``llama_n_ctx``, capped the same way llama.py's own context construction
        caps it) rather than a fixed 512, so mtmd never micro-batches smaller than
        what the context was built for. An explicit *n_batch* wins, for a caller
        that knows its own real batch size precisely."""
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
                    # On the GPU path this is the shape the gfx1030 / RDNA2
                    # hipBLAS BF16 failure takes, so tell the caller a CPU retry
                    # is worth one attempt rather than failing the request.
                    exc = MtmdGpuEncodeFailed if self.on_gpu else VisionInputError
                    raise exc(
                        f"the vision projector could not evaluate this image "
                        f"(mtmd_helper_eval_chunks rc={rc2})")
                pos = int(new_n_past.value)
                # The generation loop trusts this position as the base for every
                # subsequent single-token decode with no further sanity check, so
                # a native call that under/over-reports how many KV positions the
                # image consumed would let generation continue from a corrupted
                # position instead of failing loudly.
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
