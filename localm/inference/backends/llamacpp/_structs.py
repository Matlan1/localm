# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ctypes Structure definitions for the llama.cpp C API.

upstream appends fields to the params structs several times a quarter with no ABI
or soname bump, so localm OVER-allocates both by-value params structs (a named
trailing field for what we know plus a reserved pad) and round-trips
``*_default_params()`` - we only overwrite the fields we name, so any field we do
not know keeps its native default and a newer build never reads past our buffer
in ``llama_load_model_from_file`` / ``llama_init_from_model``. A trailing append
is therefore harmless.

A mid-struct REORDER is NOT harmless, and upstream did one:

TWO TAG NAMESPACES, AND THEY COLLIDE
------------------------------------
``b1288`` / ``b1307`` are **lemonade-sdk/llamacpp-rocm** tags: the bundled AMD
build. ``b9xxx`` / ``b10xxx`` are **ggml-org/llama.cpp** tags: upstream itself.
They are different numbering schemes and they OVERLAP on a real string -
``b1288`` also exists upstream as an unrelated release. So resolving a lemonade
tag against upstream does not give you a 404, it gives you a plausible WRONG
ARTIFACT with no signal that anything is off. Every tag below therefore names
its repository.

TWO llama_model_params LAYOUTS EXIST, BOTH 72 BYTES
---------------------------------------------------
``llama_model_params`` was reordered in place between llama.cpp `7c158fbb4aec`
(lemonade b1288, ggml 0.13.1) and `07132750825a` (lemonade b1307, ggml 0.18.1).
The change landed in upstream release **b10105**. Upstream b10103 is the last
release with the old layout; b10104 was never tagged. ``sizeof`` is 72 on BOTH
sides, so nothing about the size trips - the fields simply moved:

    offset   V1 (lemonade b1288,       V2 (lemonade b1307,
             upstream <= b10103)        upstream >= b10105)
    [16]     n_gpu_layers                      n_gpu_layers
    [20]     split_mode                        split_mode
    [24]     main_gpu                          load_mode      <-- INSERTED
    [28]     (pad)                             main_gpu       <-- MOVED
    [64]     vocab_only                        vocab_only
    [65]     use_mmap                          check_tensors  <-- use_mmap,
    [66]     use_direct_io                     use_extra_bufts    use_direct_io
    [67]     use_mlock                         no_host            and use_mlock
    [68]     check_tensors                     no_alloc           DELETED; mmap /
    [69]     use_extra_bufts                   load_mtp           mlock / direct-io
    [70]     no_host                           (pad)              are now the
    [71]     no_alloc                          (pad)              load_mode enum

Writing a V1 ``main_gpu`` into a V2 build lands in ``load_mode`` and silently
drops the user's GPU selection while changing how the weights are mapped; writing
``check_tensors`` at the V1 offset lands in V2's ``no_alloc``, which loads
metadata and no weights at all. None of it raises.

localm ships BOTH layouts and picks one per loaded library at load time
(``_abi.detect_model_params_layout``). There is NO bare ``LlamaModelParams``
name: a caller must go through ``_abi.model_params_class()`` /
``_api.llama_model_default_params()``, so the wrong one cannot be constructed by
habit.

Fields at offsets that did NOT move (``devices``, ``tensor_buft_overrides``,
``n_gpu_layers``, ``split_mode``, ``tensor_split``, the callbacks,
``kv_overrides``, ``vocab_only``) are named identically in both classes, so call
sites can set them directly. ``main_gpu`` is also named in both, at its own
correct offset in each, so ``mp.main_gpu = i`` is right once the class is right.
The ONE field with no V2 counterpart is ``use_mmap``; use :func:`set_use_mmap`.

Verified NATIVE sizes:
    llama_model_params   = 72 bytes (V1 and V2 alike)
    llama_context_params = 152 bytes on lemonade b1288; 160 bytes on
                           upstream b9682+ / lemonade b1307
                           (adds a trailing ``ctx_other`` pointer). Trailing
                           append only - no mid-struct movement.
    llama_batch          = 56 bytes (7 pointers + 1 int32 + padding)

The ``sizeof`` asserts below guard against editing these definitions wrong; they
do NOT validate against the loaded DLL (a struct is the same size whatever the
DLL contains).  The runtime cross-check against the ACTUAL native layout lives in
``_abi.py`` (``verify_abi``), called from ``_loader.load_lib`` on first load.
"""

from __future__ import annotations

import ctypes

# Primitive aliases

llama_token   = ctypes.c_int32   # token id
llama_pos     = ctypes.c_int32   # position in sequence
llama_seq_id  = ctypes.c_int32   # sequence id


# enum llama_load_mode  (V2 builds only)
#
# Replaces V1's separate use_mmap / use_mlock / use_direct_io booleans with a
# single enum. AUTO is negative: llama_model_default_params() returns -1 on a
# build whose default is AUTO and 1 (MMAP) on an older one. The surrounding
# llama_model_params layout is the same either way, so only the value differs.
LLAMA_LOAD_MODE_AUTO       = -1  # let the build decide from device capabilities
LLAMA_LOAD_MODE_NONE       = 0   # no special loading mode (i.e. no mmap)
LLAMA_LOAD_MODE_MMAP       = 1   # memory map the model
LLAMA_LOAD_MODE_MLOCK      = 2   # keep in RAM, no swap/compress
LLAMA_LOAD_MODE_MMAP_MLOCK = 3   # both
LLAMA_LOAD_MODE_DIRECT_IO  = 4   # direct I/O where available

# This tuple MIRRORS llama.h's enumerators. It is not a policy knob and not a
# tolerance: _abi.evaluate reads it as a misaligned-read tripwire, so widening
# it disables that check.
_VALID_LOAD_MODES = (
    LLAMA_LOAD_MODE_AUTO,
    LLAMA_LOAD_MODE_NONE,
    LLAMA_LOAD_MODE_MMAP,
    LLAMA_LOAD_MODE_MLOCK,
    LLAMA_LOAD_MODE_MMAP_MLOCK,
    LLAMA_LOAD_MODE_DIRECT_IO,
)


# llama_model_params V1  (72 bytes)
#
# Native defaults from llama_model_default_params():
#   - [0-7]   ptr  devices                = NULL
#   - [8-15]  ptr  tensor_buft_overrides  = NULL
#   - [16]    i32  n_gpu_layers           = -1 (default: all layers)
#   - [20]    i32  split_mode             = 1 (LLAMA_SPLIT_MODE_LAYER)
#   - [24]    i32  main_gpu               = 0
#   - [28]    pad
#   - [32-39] ptr  tensor_split           = NULL
#   - [40-47] ptr  progress_callback      = NULL
#   - [48-55] ptr  progress_callback_user_data = NULL
#   - [56-63] ptr  kv_overrides           = NULL
#   - [64]    bool vocab_only             = False
#   - [65]    bool use_mmap               = True
#   - [66]    bool use_direct_io          = False
#   - [67]    bool use_mlock              = False
#   - [68]    bool check_tensors          = False
#   - [69]    bool use_extra_bufts        = True
#   - [70]    bool no_host                = False
#   - [71]    bool no_alloc               = False

# llama_model_tensor_buft_override
#
#     struct llama_model_tensor_buft_override {
#         const char * pattern;
#         ggml_backend_buffer_type_t buft;
#     };
#
# An ARRAY of these, terminated by an entry whose pattern is NULL, is what
# llama_model_params.tensor_buft_overrides points at. Each entry sends any
# tensor whose name matches the pattern to that buffer type instead of where
# the layer assignment would have put it, the same effect as llama.cpp's own
# --override-tensor / --cpu-moe / --n-cpu-moe flags.

class LlamaModelTensorBuftOverride(ctypes.Structure):
    _fields_ = [
        ("pattern", ctypes.c_char_p),
        ("buft",    ctypes.c_void_p),    # ggml_backend_buffer_type_t
    ]


class LlamaModelParamsV1(ctypes.Structure):
    _fields_ = [
        ("devices",                     ctypes.c_void_p),    # ggml_backend_dev_t**
        ("tensor_buft_overrides",       ctypes.c_void_p),
        ("n_gpu_layers",                ctypes.c_int32),
        ("split_mode",                  ctypes.c_int32),
        ("main_gpu",                    ctypes.c_int32),
        ("_pad0",                       ctypes.c_int32),
        ("tensor_split",                ctypes.c_void_p),    # const float*
        ("progress_callback",           ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides",                ctypes.c_void_p),
        ("vocab_only",                  ctypes.c_bool),
        ("use_mmap",                    ctypes.c_bool),
        ("use_direct_io",               ctypes.c_bool),
        ("use_mlock",                   ctypes.c_bool),
        ("check_tensors",               ctypes.c_bool),
        ("use_extra_bufts",             ctypes.c_bool),
        ("no_host",                     ctypes.c_bool),
        ("no_alloc",                    ctypes.c_bool),
        # Forward-compat headroom. The native struct ends at no_alloc (72
        # bytes); over-allocating keeps a newer build's appended trailing
        # fields inside our buffer.
        ("_reserved",                   ctypes.c_uint8 * 32),
    ]


# llama_model_params V2  (72 bytes)
#
# Native defaults from llama_model_default_params():
#   - [16]    i32  n_gpu_layers    = -1
#   - [20]    i32  split_mode      = 1 (LLAMA_SPLIT_MODE_LAYER)
#   - [24]    i32  load_mode       = 1 (LLAMA_LOAD_MODE_MMAP)
#   - [28]    i32  main_gpu        = 0
#   - [64]    bool vocab_only      = False
#   - [65]    bool check_tensors   = False
#   - [66]    bool use_extra_bufts = True
#   - [67]    bool no_host         = False
#   - [68]    bool no_alloc        = False
#   - [69]    bool load_mtp        = False
# main_gpu at 28 needs no explicit pad after it: 28 + 4 == 32, already aligned
# for the tensor_split pointer.

class LlamaModelParamsV2(ctypes.Structure):
    _fields_ = [
        ("devices",                     ctypes.c_void_p),    # ggml_backend_dev_t**
        ("tensor_buft_overrides",       ctypes.c_void_p),
        ("n_gpu_layers",                ctypes.c_int32),
        ("split_mode",                  ctypes.c_int32),
        ("load_mode",                   ctypes.c_int32),     # enum llama_load_mode
        ("main_gpu",                    ctypes.c_int32),
        ("tensor_split",                ctypes.c_void_p),    # const float*
        ("progress_callback",           ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides",                ctypes.c_void_p),
        ("vocab_only",                  ctypes.c_bool),
        ("check_tensors",               ctypes.c_bool),
        ("use_extra_bufts",             ctypes.c_bool),
        ("no_host",                     ctypes.c_bool),
        ("no_alloc",                    ctypes.c_bool),
        ("load_mtp",                    ctypes.c_bool),
        ("_pad0",                       ctypes.c_uint8 * 2),
        # Forward-compat headroom, same as V1.
        ("_reserved",                   ctypes.c_uint8 * 32),
    ]


# Self-consistency guards ONLY (these do NOT validate against the DLL - that is
# _abi.verify_abi). 72 native bytes + 32 reserved = 104, for BOTH layouts.
assert ctypes.sizeof(LlamaModelParamsV1) == 104, (
    f"LlamaModelParamsV1 size mismatch: {ctypes.sizeof(LlamaModelParamsV1)} != 104"
)
assert ctypes.sizeof(LlamaModelParamsV2) == 104, (
    f"LlamaModelParamsV2 size mismatch: {ctypes.sizeof(LlamaModelParamsV2)} != 104"
)
# Assert the field offsets each layout is required to have.
for _cls, _off in (
    (LlamaModelParamsV1, {"n_gpu_layers": 16, "split_mode": 20, "main_gpu": 24,
                          "vocab_only": 64, "use_mmap": 65, "use_direct_io": 66,
                          "use_mlock": 67, "check_tensors": 68,
                          "use_extra_bufts": 69, "no_host": 70, "no_alloc": 71}),
    (LlamaModelParamsV2, {"n_gpu_layers": 16, "split_mode": 20, "load_mode": 24,
                          "main_gpu": 28, "vocab_only": 64, "check_tensors": 65,
                          "use_extra_bufts": 66, "no_host": 67, "no_alloc": 68,
                          "load_mtp": 69}),
):
    for _name, _want in _off.items():
        _got = getattr(_cls, _name).offset
        assert _got == _want, f"{_cls.__name__}.{_name} at {_got}, expected {_want}"
del _cls, _off, _name, _want, _got


def set_use_mmap(mp, enabled: bool) -> None:
    """Express "memory-map the weights (or do not)" on EITHER layout.

    V1 has a ``use_mmap`` bool; V2 folded mmap / mlock / direct-io into the
    ``load_mode`` enum, so there is no field of that name to assign and a plain
    ``mp.use_mmap = False`` on a V2 struct would raise (or, if the wrong class
    were bound, silently write into ``check_tensors``). Call sites use this
    instead of naming either field.

    Any mlock the caller already asked for is preserved across the flip:
    mapping "no mmap" onto a bare LLAMA_LOAD_MODE_NONE would drop it.
    """
    if isinstance(mp, LlamaModelParamsV2):
        keep_mlock = mp.load_mode in (LLAMA_LOAD_MODE_MLOCK,
                                      LLAMA_LOAD_MODE_MMAP_MLOCK)
        if enabled:
            mp.load_mode = (LLAMA_LOAD_MODE_MMAP_MLOCK if keep_mlock
                            else LLAMA_LOAD_MODE_MMAP)
        else:
            mp.load_mode = (LLAMA_LOAD_MODE_MLOCK if keep_mlock
                            else LLAMA_LOAD_MODE_NONE)
        return
    mp.use_mmap = enabled


def get_use_mmap(mp) -> bool:
    """Read back whether the weights will be memory-mapped, on either layout.

    DIRECT_IO is NOT mmap: upstream documents it as taking precedence over
    mmap, and V1 carried it as its own separate flag.

    AUTO reports False, and that is a KNOWN IMPRECISION rather than an answer:
    under LLAMA_LOAD_MODE_AUTO the build picks at load time from device
    capabilities, so whether the weights end up mapped is not knowable from the
    params at all, and a bool cannot say "undetermined". Every localm call site
    writes an explicit mode through set_use_mmap before any read, so AUTO does
    not reach here in practice; a caller reading unmodified default params on a
    b10373-or-newer build must not read False as "mmap is off".
    """
    if isinstance(mp, LlamaModelParamsV2):
        return mp.load_mode in (LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_MMAP_MLOCK)
    return bool(mp.use_mmap)


# TWO llama_context_params LAYOUTS EXIST, BOTH 224 BYTES (152/160 native + pad)
# -------------------------------------------------------------------------
# V2 carries an extra uint32_t field, n_outputs_max_per_seq, directly before
# n_threads. Every field from n_threads onward sits 4 bytes later as a result:
#
#     offset   V1                               V2
#     [20]     n_outputs_max                    n_outputs_max
#     [24]     n_threads                        n_outputs_max_per_seq
#     [28]     n_threads_batch                  n_threads
#     [32]     ctx_type                         n_threads_batch
#     [36]     rope_scaling_type                ctx_type
#     [40]     pooling_type                     rope_scaling_type
#     [44]     attention_type                   pooling_type
#     [48]     flash_attn_type                  attention_type
#     [52]     rope_freq_base                   flash_attn_type
#     [84]     _pad1 (alignment filler)         (none: defrag_thold ends at 88,
#                                                already 8-aligned for cb_eval)
#     [88]     cb_eval                          cb_eval
#
# localm ships BOTH layouts and picks one per loaded library at load time
# (`_abi.detect_context_params_layout`), same as the model_params V1/V2 split
# above. There is NO bare `LlamaContextParams` name: a caller must go through
# `_abi.context_params_class()` / `_api.llama_context_default_params()`.
#
# Fields that did not move (everything before n_threads, and everything from
# cb_eval onward) are named identically in both classes.
#
# enum llama_context_type
LLAMA_CONTEXT_TYPE_DEFAULT = 0
LLAMA_CONTEXT_TYPE_MTP     = 1


class LlamaContextParamsV1(ctypes.Structure):
    _fields_ = [
        # --- batch / sequence limits ---
        ("n_ctx",             ctypes.c_uint32),   # [0]
        ("n_batch",           ctypes.c_uint32),   # [4]
        ("n_ubatch",          ctypes.c_uint32),   # [8]
        ("n_seq_max",         ctypes.c_uint32),   # [12]
        ("n_rs_seq",          ctypes.c_uint32),   # [16] recurrent-state snapshots
        ("n_outputs_max",     ctypes.c_uint32),   # [20] max outputs per ubatch
        # --- threading ---
        ("n_threads",         ctypes.c_int32),    # [24]
        ("n_threads_batch",   ctypes.c_int32),    # [28]
        # --- encoding type enums ---
        ("ctx_type",          ctypes.c_int32),    # [32] LLAMA_CONTEXT_TYPE_*
        ("rope_scaling_type", ctypes.c_int32),    # [36] default -1
        ("pooling_type",      ctypes.c_int32),    # [40] default -1
        ("attention_type",    ctypes.c_int32),    # [44] default -1
        ("flash_attn_type",   ctypes.c_int32),    # [48] default -1
        # --- RoPE / YaRN floats ---
        ("rope_freq_base",    ctypes.c_float),    # [52]
        ("rope_freq_scale",   ctypes.c_float),    # [56]
        ("yarn_ext_factor",   ctypes.c_float),    # [60] -1 = from model
        ("yarn_attn_factor",  ctypes.c_float),    # [64]
        ("yarn_beta_fast",    ctypes.c_float),    # [68]
        ("yarn_beta_slow",    ctypes.c_float),    # [72]
        ("yarn_orig_ctx",     ctypes.c_uint32),   # [76]
        ("defrag_thold",      ctypes.c_float),    # [80]
        # --- backend eval callback ---
        ("_pad1",             ctypes.c_uint32),   # [84] alignment pad
        ("cb_eval",           ctypes.c_void_p),   # [88]
        ("cb_eval_user_data", ctypes.c_void_p),   # [96]
        # --- KV cache types ---
        ("type_k",            ctypes.c_int32),    # [104] ggml_type
        ("type_v",            ctypes.c_int32),    # [108] ggml_type
        # --- abort callback ---
        ("abort_callback",      ctypes.c_void_p), # [112]
        ("abort_callback_data", ctypes.c_void_p), # [120]
        # --- boolean flags (kept together per llama.h comment) ---
        ("embeddings",  ctypes.c_bool),           # [128]
        ("offload_kqv", ctypes.c_bool),           # [129]
        ("no_perf",     ctypes.c_bool),           # [130]
        ("op_offload",  ctypes.c_bool),           # [131]
        ("swa_full",    ctypes.c_bool),            # [132]
        ("kv_unified",  ctypes.c_bool),           # [133]
        ("_pad2",       ctypes.c_uint8 * 2),      # [134-135]
        # --- sampler chain hooks ---
        ("samplers",    ctypes.c_void_p),         # [136]
        ("n_samplers",  ctypes.c_uint64),         # [144]
        # ctx_other is a trailing field older builds lack; they ignore it.
        # Naming it keeps the round-trip through
        # llama_context_default_params() correct on newer builds.
        ("ctx_other",   ctypes.c_void_p),         # [152] struct llama_context*
        # Forward-compat headroom: future trailing fields land here, keep their
        # native default via the default_params round-trip, and never cause
        # llama_init_from_model to read past us.
        ("_reserved",   ctypes.c_uint8 * 64),     # [160]
    ]


class LlamaContextParamsV2(ctypes.Structure):
    _fields_ = [
        # --- batch / sequence limits ---
        ("n_ctx",             ctypes.c_uint32),   # [0]
        ("n_batch",           ctypes.c_uint32),   # [4]
        ("n_ubatch",          ctypes.c_uint32),   # [8]
        ("n_seq_max",         ctypes.c_uint32),   # [12]
        ("n_rs_seq",          ctypes.c_uint32),   # [16] recurrent-state snapshots
        ("n_outputs_max",     ctypes.c_uint32),   # [20] max outputs per ubatch
        ("n_outputs_max_per_seq", ctypes.c_uint32), # [24] max outputs per sequence
        # --- threading ---
        ("n_threads",         ctypes.c_int32),    # [28]
        ("n_threads_batch",   ctypes.c_int32),    # [32]
        # --- encoding type enums ---
        ("ctx_type",          ctypes.c_int32),    # [36] LLAMA_CONTEXT_TYPE_*
        ("rope_scaling_type", ctypes.c_int32),    # [40] default -1
        ("pooling_type",      ctypes.c_int32),    # [44] default -1
        ("attention_type",    ctypes.c_int32),    # [48] default -1
        ("flash_attn_type",   ctypes.c_int32),    # [52] default -1
        # --- RoPE / YaRN floats ---
        ("rope_freq_base",    ctypes.c_float),    # [56]
        ("rope_freq_scale",   ctypes.c_float),    # [60]
        ("yarn_ext_factor",   ctypes.c_float),    # [64] -1 = from model
        ("yarn_attn_factor",  ctypes.c_float),    # [68]
        ("yarn_beta_fast",    ctypes.c_float),    # [72]
        ("yarn_beta_slow",    ctypes.c_float),    # [76]
        ("yarn_orig_ctx",     ctypes.c_uint32),   # [80]
        ("defrag_thold",      ctypes.c_float),    # [84]
        # --- backend eval callback ---
        # No manual pad here: defrag_thold ends at byte 88, already 8-byte
        # aligned for the pointer below.
        ("cb_eval",           ctypes.c_void_p),   # [88]
        ("cb_eval_user_data", ctypes.c_void_p),   # [96]
        # --- KV cache types ---
        ("type_k",            ctypes.c_int32),    # [104] ggml_type
        ("type_v",            ctypes.c_int32),    # [108] ggml_type
        # --- abort callback ---
        ("abort_callback",      ctypes.c_void_p), # [112]
        ("abort_callback_data", ctypes.c_void_p), # [120]
        # --- boolean flags (kept together per llama.h comment) ---
        ("embeddings",  ctypes.c_bool),           # [128]
        ("offload_kqv", ctypes.c_bool),           # [129]
        ("no_perf",     ctypes.c_bool),           # [130]
        ("op_offload",  ctypes.c_bool),           # [131]
        ("swa_full",    ctypes.c_bool),            # [132]
        ("kv_unified",  ctypes.c_bool),           # [133]
        ("_pad2",       ctypes.c_uint8 * 2),      # [134-135]
        # --- sampler chain hooks ---
        ("samplers",    ctypes.c_void_p),         # [136]
        ("n_samplers",  ctypes.c_uint64),         # [144]
        ("ctx_other",   ctypes.c_void_p),         # [152] struct llama_context*
        # Forward-compat headroom: future trailing fields land here, keep their
        # native default via the default_params round-trip, and never cause
        # llama_init_from_model to read past us.
        ("_reserved",   ctypes.c_uint8 * 64),     # [160]
    ]

# Self-consistency guards ONLY (these do NOT validate against the DLL - that
# is _abi.verify_abi). Both layouts land at 224 bytes: V2's extra 4-byte
# n_outputs_max_per_seq takes the place of V1's 4-byte alignment pad before
# cb_eval.
assert ctypes.sizeof(LlamaContextParamsV1) == 224, (
    f"LlamaContextParamsV1 size mismatch: {ctypes.sizeof(LlamaContextParamsV1)} != 224"
)
assert ctypes.sizeof(LlamaContextParamsV2) == 224, (
    f"LlamaContextParamsV2 size mismatch: {ctypes.sizeof(LlamaContextParamsV2)} != 224"
)
for _cls, _off in (
    (LlamaContextParamsV1, {"n_outputs_max": 20, "n_threads": 24,
                            "n_threads_batch": 28, "ctx_type": 32,
                            "rope_scaling_type": 36, "pooling_type": 40,
                            "attention_type": 44, "flash_attn_type": 48,
                            "cb_eval": 88, "ctx_other": 152}),
    (LlamaContextParamsV2, {"n_outputs_max": 20, "n_outputs_max_per_seq": 24,
                            "n_threads": 28, "n_threads_batch": 32,
                            "ctx_type": 36, "rope_scaling_type": 40,
                            "pooling_type": 44, "attention_type": 48,
                            "flash_attn_type": 52, "cb_eval": 88,
                            "ctx_other": 152}),
):
    for _name, _want in _off.items():
        _got = getattr(_cls, _name).offset
        assert _got == _want, f"{_cls.__name__}.{_name} at {_got}, expected {_want}"
del _cls, _off, _name, _want, _got


# llama_sampler_chain_params  (1 byte + padding)

class LlamaSamplerChainParams(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool),
    ]


# llama_batch  (56 bytes)
#
# struct llama_batch {
#     int32_t     n_tokens;
#     llama_token * token;
#     float       * embd;
#     llama_pos   * pos;
#     int32_t     * n_seq_id;
#     llama_seq_id** seq_id;
#     int8_t      * logits;
# };
#
# Layout (64-bit):
#   [0]   int32  n_tokens
#   [4]   [4 pad]
#   [8]   ptr    token
#   [16]  ptr    embd
#   [24]  ptr    pos
#   [32]  ptr    n_seq_id
#   [40]  ptr    seq_id   (llama_seq_id**)
#   [48]  ptr    logits
# Total: 56 bytes

# llama_chat_message  (used by llama_chat_apply_template)

class LlamaChatMessage(ctypes.Structure):
    """
    typedef struct llama_chat_message {
        const char * role;
        const char * content;
    } llama_chat_message;
    """
    _fields_ = [
        ("role",    ctypes.c_char_p),
        ("content", ctypes.c_char_p),
    ]


# llama_batch  (56 bytes)
#
class LlamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens",  ctypes.c_int32),
        ("_pad",      ctypes.c_int32),
        ("token",     ctypes.c_void_p),    # llama_token*
        ("embd",      ctypes.c_void_p),    # float*
        ("pos",       ctypes.c_void_p),    # llama_pos*
        ("n_seq_id",  ctypes.c_void_p),    # int32_t*
        ("seq_id",    ctypes.c_void_p),    # llama_seq_id**
        ("logits",    ctypes.c_void_p),    # int8_t*
    ]
