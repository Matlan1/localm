# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ctypes Structure definitions for the llama.cpp C API.

These layouts were derived by probing the prebuilt llama.dll with known default
values and cross-referencing against llama.h.  The prebuilt is from a commit
that sits between the introduction of llama_init_from_model and the addition of
the 'devices' / 'tensor_buft_overrides' fields to llama_model_params.

The ``sizeof`` asserts below guard against editing these definitions wrong; they
do NOT validate against the loaded DLL (a struct is the same size whatever the
DLL contains).  The runtime cross-check against the ACTUAL native layout lives in
``_abi.py`` (``verify_abi``), called from ``_loader.load_lib`` on first load.

Verified NATIVE sizes (against the cpu / vulkan / amd-rocm prebuilts, b1288..b9740):
    llama_model_params   = 72 bytes
    llama_context_params = 152 bytes on b1288; 160 bytes on b9682+ (adds a
                           trailing ``ctx_other`` pointer)
    llama_batch          = 56 bytes (7 pointers + 1 int32 + padding)

upstream appends fields to the params structs several times a quarter with no ABI
or soname bump, so localm OVER-allocates both by-value params structs (a named
trailing field for what we know plus a reserved pad) and round-trips
``*_default_params()`` - we only overwrite the fields we name, so any field we do
not know keeps its native default and a newer build never reads past our buffer
in ``llama_load_model_from_file`` / ``llama_init_from_model``. A trailing append
is therefore harmless; a mid-struct REORDER is caught at load time by
``_abi.verify_abi``.
"""

from __future__ import annotations

import ctypes

# Primitive aliases

llama_token   = ctypes.c_int32   # token id
llama_pos     = ctypes.c_int32   # position in sequence
llama_seq_id  = ctypes.c_int32   # sequence id


# llama_model_params  (72 bytes - probed)
#
# This is the NEW layout that includes 'devices' and 'tensor_buft_overrides'
# at the beginning (added in recent llama.cpp).  Verified by probing the
# prebuilt llama.dll:
#   - [0-7]   ptr  devices                = NULL
#   - [8-15]  ptr  tensor_buft_overrides  = NULL
#   - [16]    i32  n_gpu_layers           = -1 (default: all layers)
#   - [20]    i32  split_mode             = 1 (LLAMA_SPLIT_MODE_LAYER)
#   - [24]    i32  main_gpu               = 0
#   - [28]    pad
#   - [32-39] ptr  tensor_split           = <static default array>
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

class LlamaModelParams(ctypes.Structure):
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
        # Forward-compat headroom (see the module docstring). The native struct
        # ends at no_alloc (72 bytes today); we over-allocate so a newer build
        # that appends trailing fields never reads past our buffer.
        ("_reserved",                   ctypes.c_uint8 * 32),
    ]

# Self-consistency guard ONLY (this does NOT validate against the DLL - that is
# _abi.verify_abi). 72 native bytes + 32 reserved = 104.
assert ctypes.sizeof(LlamaModelParams) == 104, (
    f"LlamaModelParams size mismatch: {ctypes.sizeof(LlamaModelParams)} != 104"
)


# llama_context_params  (152 bytes - probed)
#
# Added vs the old layout:
#   n_rs_seq, n_outputs_max, ctx_type, flash_attn_type,
#   op_offload, swa_full, kv_unified, samplers, n_samplers

class LlamaContextParams(ctypes.Structure):
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
        # ctx_other was appended upstream after the b1288 build localm's layout
        # was first probed (present b9682+; absent on older builds, which simply
        # ignore this trailing field). Naming it keeps the round-trip through
        # llama_context_default_params() correct on newer builds.
        ("ctx_other",   ctypes.c_void_p),         # [152] struct llama_context*
        # Forward-compat headroom (see the module docstring): future trailing
        # fields land here, keep their native default via the default_params
        # round-trip, and never cause llama_init_from_model to read past us.
        ("_reserved",   ctypes.c_uint8 * 64),     # [160]
    ]

# Self-consistency guard ONLY (NOT a check against the DLL - that is
# _abi.verify_abi). 152 native bytes + 8 (ctx_other) + 64 reserved = 224.
assert ctypes.sizeof(LlamaContextParams) == 224, (
    f"LlamaContextParams size mismatch: {ctypes.sizeof(LlamaContextParams)} != 224"
)


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
