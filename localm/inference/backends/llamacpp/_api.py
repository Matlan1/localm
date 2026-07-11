# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Low-level ctypes bindings for the llama.cpp C API.

All functions are bound lazily on first access via the module-level ``lib``
property so the DLL is not loaded until something actually imports this module.

Naming convention mirrors the C API exactly (llama_xyz → llama_xyz).
"""

from __future__ import annotations

import ctypes
from typing import Optional

from ._loader import load_lib
from ._structs import (
    LlamaBatch,
    LlamaChatMessage,
    LlamaContextParams,
    LlamaModelParams,
    LlamaSamplerChainParams,
    llama_token,
)

# Opaque handle types
LlamaModel   = ctypes.c_void_p   # struct llama_model*
LlamaContext = ctypes.c_void_p   # struct llama_context*
LlamaSampler = ctypes.c_void_p   # struct llama_sampler*
LlamaVocab   = ctypes.c_void_p   # struct llama_vocab*


def _bind(fn_name: str, restype, *argtypes):
    """Retrieve a function from the DLL and set its signature."""
    lib = load_lib()
    fn = getattr(lib, fn_name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


# ---------------------------------------------------------------------------
#  Backend lifecycle
# ---------------------------------------------------------------------------

def llama_backend_init() -> None:
    _bind("llama_backend_init", None)()


def llama_backend_free() -> None:
    _bind("llama_backend_free", None)()


# ---------------------------------------------------------------------------
#  Default params
# ---------------------------------------------------------------------------

def llama_model_default_params() -> LlamaModelParams:
    fn = _bind("llama_model_default_params", LlamaModelParams)
    return fn()


def llama_context_default_params() -> LlamaContextParams:
    fn = _bind("llama_context_default_params", LlamaContextParams)
    return fn()


def llama_sampler_chain_default_params() -> LlamaSamplerChainParams:
    fn = _bind("llama_sampler_chain_default_params", LlamaSamplerChainParams)
    return fn()


# ---------------------------------------------------------------------------
#  Model loading / freeing
# ---------------------------------------------------------------------------

def llama_load_model_from_file(
    path: str,
    params: LlamaModelParams,
) -> Optional[ctypes.c_void_p]:
    fn = _bind(
        "llama_load_model_from_file",
        LlamaModel,
        ctypes.c_char_p,
        LlamaModelParams,
    )
    result = fn(path.encode(), params)
    return result if result else None


def llama_free_model(model: ctypes.c_void_p) -> None:
    _bind("llama_free_model", None, LlamaModel)(model)


# ---------------------------------------------------------------------------
#  Context creation / freeing
# ---------------------------------------------------------------------------

def llama_init_from_model(
    model: ctypes.c_void_p,
    params: LlamaContextParams,
) -> Optional[ctypes.c_void_p]:
    fn = _bind(
        "llama_init_from_model",
        LlamaContext,
        LlamaModel,
        LlamaContextParams,
    )
    result = fn(model, params)
    return result if result else None


def llama_free(ctx: ctypes.c_void_p) -> None:
    _bind("llama_free", None, LlamaContext)(ctx)


# ---------------------------------------------------------------------------
#  Context / model accessors
# ---------------------------------------------------------------------------

def llama_get_model(ctx: ctypes.c_void_p) -> ctypes.c_void_p:
    return _bind("llama_get_model", LlamaModel, LlamaContext)(ctx)


def llama_n_ctx(ctx: ctypes.c_void_p) -> int:
    return _bind("llama_n_ctx", ctypes.c_uint32, LlamaContext)(ctx)


def llama_n_vocab(ctx: ctypes.c_void_p) -> int:
    return _bind("llama_n_vocab", ctypes.c_int32, LlamaContext)(ctx)


def llama_model_n_ctx_train(model: ctypes.c_void_p) -> int:
    return _bind("llama_model_n_ctx_train", ctypes.c_int32, LlamaModel)(model)


def llama_model_n_embd(model: ctypes.c_void_p) -> int:
    return _bind("llama_model_n_embd", ctypes.c_int32, LlamaModel)(model)


def llama_model_n_layer(model: ctypes.c_void_p) -> int:
    return _bind("llama_model_n_layer", ctypes.c_int32, LlamaModel)(model)


def has_kv_head_api() -> bool:
    """True when this llama.dll exports llama_model_n_head + llama_model_n_head_kv,
    so the KV-cache size per token can be computed EXACTLY from the model's
    attention shape (heads x head_dim x layers) instead of estimated from file
    size. Every current build exports them (probed on the shipped cpu / vulkan /
    amd-rocm runtimes); the guard lets an exotic stripped build fall back to the
    size-class heuristic instead of raising AttributeError - same pattern as
    has_memory_api()/has_penalties_sampler()."""
    lib = load_lib()
    return all(hasattr(lib, fn)
               for fn in ("llama_model_n_head", "llama_model_n_head_kv"))


def llama_model_n_head(model: ctypes.c_void_p) -> int:
    """Number of attention (query) heads. Only call after has_kv_head_api()."""
    return _bind("llama_model_n_head", ctypes.c_int32, LlamaModel)(model)


def llama_model_n_head_kv(model: ctypes.c_void_p) -> int:
    """Number of key/value heads - fewer than n_head under grouped-query
    attention, which is exactly what makes the KV cache smaller than a naive
    n_head estimate. Only call after has_kv_head_api()."""
    return _bind("llama_model_n_head_kv", ctypes.c_int32, LlamaModel)(model)


# ---------------------------------------------------------------------------
#  Vocabulary / tokenisation
# ---------------------------------------------------------------------------

def llama_model_get_vocab(model: ctypes.c_void_p) -> ctypes.c_void_p:
    return _bind("llama_model_get_vocab", LlamaVocab, LlamaModel)(model)


def llama_vocab_n_tokens(vocab: ctypes.c_void_p) -> int:
    return _bind("llama_vocab_n_tokens", ctypes.c_int32, LlamaVocab)(vocab)


def llama_tokenize(
    vocab: ctypes.c_void_p,
    text: bytes,
    n_text: int,
    tokens_out: ctypes.Array,
    n_tokens_max: int,
    add_special: bool,
    parse_special: bool,
) -> int:
    """
    Tokenise *text* into *tokens_out*.

    Returns the number of tokens written (positive) or the negative of the
    number needed if the buffer was too small.
    """
    fn = _bind(
        "llama_tokenize",
        ctypes.c_int32,
        LlamaVocab,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(llama_token),
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool,
    )
    return fn(vocab, text, n_text, tokens_out, n_tokens_max, add_special, parse_special)


def llama_token_to_piece(
    vocab: ctypes.c_void_p,
    token: int,
    buf: ctypes.Array,
    length: int,
    lstrip: int,
    special: bool,
) -> int:
    """Convert a single token id to its text representation."""
    fn = _bind(
        "llama_token_to_piece",
        ctypes.c_int32,
        LlamaVocab,
        llama_token,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_bool,
    )
    return fn(vocab, token, buf, length, lstrip, special)


def llama_detokenize(
    vocab: ctypes.c_void_p,
    tokens: ctypes.Array,
    n_tokens: int,
    text_out: ctypes.Array,
    text_size: int,
    remove_special: bool,
    unparse_special: bool,
) -> int:
    fn = _bind(
        "llama_detokenize",
        ctypes.c_int32,
        LlamaVocab,
        ctypes.POINTER(llama_token),
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool,
    )
    return fn(vocab, tokens, n_tokens, text_out, text_size, remove_special, unparse_special)


# ---------------------------------------------------------------------------
#  Special token helpers
# ---------------------------------------------------------------------------

def llama_token_bos(vocab: ctypes.c_void_p) -> int:
    return _bind("llama_token_bos", llama_token, LlamaVocab)(vocab)


def llama_token_eos(vocab: ctypes.c_void_p) -> int:
    return _bind("llama_token_eos", llama_token, LlamaVocab)(vocab)


def llama_vocab_is_eog(vocab: ctypes.c_void_p, token: int) -> bool:
    return bool(_bind("llama_vocab_is_eog", ctypes.c_bool, LlamaVocab, llama_token)(vocab, token))


def llama_token_is_eog(vocab: ctypes.c_void_p, token: int) -> bool:
    return bool(_bind("llama_token_is_eog", ctypes.c_bool, LlamaVocab, llama_token)(vocab, token))


# ---------------------------------------------------------------------------
#  Chat template
# ---------------------------------------------------------------------------

def llama_model_chat_template(model: ctypes.c_void_p, name: Optional[bytes] = None) -> Optional[str]:
    fn = _bind("llama_model_chat_template", ctypes.c_char_p, LlamaModel, ctypes.c_char_p)
    result = fn(model, name)
    return result.decode(errors="replace") if result else None


def llama_chat_apply_template(
    tmpl: Optional[bytes],
    messages: ctypes.Array,   # LlamaChatMessage[]
    n_messages: int,
    add_assistant: bool,
    buf: ctypes.Array,
    length: int,
) -> int:
    """
    Apply a Jinja-ish chat template to a list of messages.

    Parameters
    ----------
    tmpl:
        Raw template string obtained from ``llama_model_chat_template()``.
        Pass *None* to fall back to the built-in "chatml" template.
    messages:
        A ctypes array of ``LlamaChatMessage`` structs.
    n_messages:
        Number of elements in *messages*.
    add_assistant:
        If True, append the assistant-turn start tokens so the model knows
        it should continue generating.
    buf:
        Output buffer.
    length:
        Size of *buf*.  If the required output is larger, the function returns
        the total bytes needed (positive) so you can reallocate and retry.

    Returns
    -------
    int
        Number of bytes written.  If > *length*, the buffer was too small -
        reallocate and call again.
    """
    fn = _bind(
        "llama_chat_apply_template",
        ctypes.c_int32,
        ctypes.c_char_p,                         # tmpl
        ctypes.POINTER(LlamaChatMessage),         # chat
        ctypes.c_size_t,                          # n_msg
        ctypes.c_bool,                            # add_ass
        ctypes.c_char_p,                          # buf
        ctypes.c_int32,                           # length
    )
    return fn(tmpl, messages, n_messages, add_assistant, buf, length)


# ---------------------------------------------------------------------------
#  Batch
# ---------------------------------------------------------------------------

def llama_batch_get_one(
    tokens: ctypes.Array,
    n_tokens: int,
) -> LlamaBatch:
    fn = _bind(
        "llama_batch_get_one",
        LlamaBatch,
        ctypes.POINTER(llama_token),
        ctypes.c_int32,
    )
    return fn(tokens, n_tokens)


def llama_batch_init(n_tokens: int, embd: int, n_seq_max: int) -> LlamaBatch:
    fn = _bind(
        "llama_batch_init",
        LlamaBatch,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    )
    return fn(n_tokens, embd, n_seq_max)


def llama_batch_free(batch: LlamaBatch) -> None:
    _bind("llama_batch_free", None, LlamaBatch)(batch)


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------

def llama_decode(ctx: ctypes.c_void_p, batch: LlamaBatch) -> int:
    """
    Run the model on *batch*.  Returns 0 on success, 1 if no KV slot available,
    negative on error.
    """
    return _bind("llama_decode", ctypes.c_int32, LlamaContext, LlamaBatch)(ctx, batch)


# ---------------------------------------------------------------------------
#  Logits
# ---------------------------------------------------------------------------

def llama_get_logits_ith(ctx: ctypes.c_void_p, i: int) -> ctypes.Array:
    """Return a pointer to the logit row for batch position *i*."""
    fn = _bind("llama_get_logits_ith", ctypes.POINTER(ctypes.c_float), LlamaContext, ctypes.c_int32)
    return fn(ctx, i)


def llama_get_logits(ctx: ctypes.c_void_p) -> ctypes.Array:
    """Return a pointer to the full logit matrix."""
    fn = _bind("llama_get_logits", ctypes.POINTER(ctypes.c_float), LlamaContext)
    return fn(ctx)


# ---------------------------------------------------------------------------
#  Embeddings (probe before use - present on any real llama.cpp, but keep the
#  binding guarded so a stripped build degrades cleanly instead of AttributeError)
# ---------------------------------------------------------------------------

def has_embeddings_api() -> bool:
    """True when this llama.dll exports the embedding accessors. Every mainline
    llama.cpp build does; the probe lets a caller fall back (to lexical-only
    retrieval) rather than crash on an exotic stripped build."""
    lib = load_lib()
    return all(hasattr(lib, fn)
               for fn in ("llama_get_embeddings_seq", "llama_get_embeddings_ith"))


def llama_get_embeddings_seq(ctx: ctypes.c_void_p, seq_id: int) -> ctypes.Array:
    """Pooled embedding for sequence *seq_id* (a pointer to ``n_embd`` floats).

    Valid after ``llama_decode`` on a context created with ``embeddings=True`` and
    a pooling type other than NONE. Returns a NULL pointer when pooling is off or
    the sequence has no output."""
    fn = _bind("llama_get_embeddings_seq", ctypes.POINTER(ctypes.c_float),
               LlamaContext, ctypes.c_int32)
    return fn(ctx, seq_id)


def llama_get_embeddings_ith(ctx: ctypes.c_void_p, i: int) -> ctypes.Array:
    """Per-token embedding for batch position *i* (used when pooling is NONE)."""
    fn = _bind("llama_get_embeddings_ith", ctypes.POINTER(ctypes.c_float),
               LlamaContext, ctypes.c_int32)
    return fn(ctx, i)


# ---------------------------------------------------------------------------
#  Sampler chain
# ---------------------------------------------------------------------------

def llama_sampler_chain_init(params: LlamaSamplerChainParams) -> ctypes.c_void_p:
    return _bind(
        "llama_sampler_chain_init",
        LlamaSampler,
        LlamaSamplerChainParams,
    )(params)


def llama_sampler_chain_add(chain: ctypes.c_void_p, sampler: ctypes.c_void_p) -> None:
    _bind("llama_sampler_chain_add", None, LlamaSampler, LlamaSampler)(chain, sampler)


def llama_sampler_free(sampler: ctypes.c_void_p) -> None:
    _bind("llama_sampler_free", None, LlamaSampler)(sampler)


def llama_sampler_sample(sampler: ctypes.c_void_p, ctx: ctypes.c_void_p, idx: int) -> int:
    return _bind(
        "llama_sampler_sample",
        llama_token,
        LlamaSampler,
        LlamaContext,
        ctypes.c_int32,
    )(sampler, ctx, idx)


def llama_sampler_accept(sampler: ctypes.c_void_p, token: int) -> None:
    _bind("llama_sampler_accept", None, LlamaSampler, llama_token)(sampler, token)


# --- individual sampler constructors ---

def llama_sampler_init_greedy() -> ctypes.c_void_p:
    return _bind("llama_sampler_init_greedy", LlamaSampler)()


def llama_sampler_init_dist(seed: int) -> ctypes.c_void_p:
    return _bind("llama_sampler_init_dist", LlamaSampler, ctypes.c_uint32)(seed)


def llama_sampler_init_top_k(k: int) -> ctypes.c_void_p:
    return _bind("llama_sampler_init_top_k", LlamaSampler, ctypes.c_int32)(k)


def llama_sampler_init_top_p(p: float, min_keep: int) -> ctypes.c_void_p:
    return _bind("llama_sampler_init_top_p", LlamaSampler, ctypes.c_float, ctypes.c_size_t)(p, min_keep)


def llama_sampler_init_min_p(p: float, min_keep: int) -> ctypes.c_void_p:
    return _bind("llama_sampler_init_min_p", LlamaSampler, ctypes.c_float, ctypes.c_size_t)(p, min_keep)


def llama_sampler_init_temp(temp: float) -> ctypes.c_void_p:
    return _bind("llama_sampler_init_temp", LlamaSampler, ctypes.c_float)(temp)


def has_penalties_sampler() -> bool:
    """True when this llama.dll exports llama_sampler_init_penalties."""
    try:
        getattr(load_lib(), "llama_sampler_init_penalties")
        return True
    except AttributeError:
        return False


def llama_sampler_init_penalties(
    penalty_last_n: int,
    penalty_repeat: float,
    penalty_freq: float = 0.0,
    penalty_present: float = 0.0,
) -> ctypes.c_void_p:
    """Repetition penalty sampler (current 4-argument signature)."""
    return _bind(
        "llama_sampler_init_penalties", LlamaSampler,
        ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    )(penalty_last_n, penalty_repeat, penalty_freq, penalty_present)


def llama_sampler_init_grammar(
    vocab: ctypes.c_void_p,
    grammar_str: bytes,
    grammar_root: bytes,
) -> ctypes.c_void_p:
    """
    Create a grammar sampler that constrains token selection to outputs
    matching the given GBNF grammar.

    The sampler masks logits for tokens that would violate the grammar at the
    current parse position, so only structurally valid continuations survive
    into the temperature / dist stage.

    Parameters
    ----------
    vocab:
        Vocabulary pointer from ``llama_model_get_vocab()``.
    grammar_str:
        GBNF grammar source, UTF-8 encoded.
    grammar_root:
        Name of the root rule, e.g. ``b"root"``.
    """
    return _bind(
        "llama_sampler_init_grammar",
        LlamaSampler,
        LlamaVocab,
        ctypes.c_char_p,
        ctypes.c_char_p,
    )(vocab, grammar_str, grammar_root)


def has_lazy_grammar() -> bool:
    """True when this llama.dll exports llama_sampler_init_grammar_lazy_patterns."""
    try:
        getattr(load_lib(), "llama_sampler_init_grammar_lazy_patterns")
        return True
    except AttributeError:
        return False


def llama_sampler_init_grammar_lazy_patterns(
    vocab: ctypes.c_void_p,
    grammar_str: bytes,
    grammar_root: bytes,
    trigger_patterns: list,
) -> ctypes.c_void_p:
    """
    Create a LAZY grammar sampler: generation is unconstrained until the
    accumulated output matches one of *trigger_patterns* (regex, full-match
    against the generated text; the grammar is fed from capture group 1),
    then the GBNF grammar enforces from that point on.

    This is the "text-or-tool" mechanism: thinking and prose flow freely, a
    started structured block must be valid.

    Parameters
    ----------
    vocab:
        Vocabulary pointer from ``llama_model_get_vocab()``.
    grammar_str:
        GBNF grammar source, UTF-8 encoded.
    grammar_root:
        Name of the root rule, e.g. ``b"root"``.
    trigger_patterns:
        Regex patterns as ``bytes``, e.g. ``[rb"[\\s\\S]*?(<tool_call>[\\s\\S]*)"]``.
    """
    pats = (ctypes.c_char_p * len(trigger_patterns))(*trigger_patterns)
    return _bind(
        "llama_sampler_init_grammar_lazy_patterns",
        LlamaSampler,
        LlamaVocab,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(llama_token),
        ctypes.c_size_t,
    )(vocab, grammar_str, grammar_root, pats, len(trigger_patterns), None, 0)


# ---------------------------------------------------------------------------
#  KV cache / memory management (newer builds only - probe before use)
# ---------------------------------------------------------------------------

LlamaMemory = ctypes.c_void_p   # llama_memory_t


def has_memory_api() -> bool:
    """
    True when this llama.cpp build exports the llama_memory_* family
    (introduced mid-2025). Older DLLs lack it - callers must fall back to
    recreating the context to clear the KV cache.
    """
    lib = load_lib()
    return all(
        hasattr(lib, fn)
        for fn in ("llama_get_memory", "llama_memory_clear", "llama_memory_seq_rm")
    )


def llama_get_memory(ctx: ctypes.c_void_p) -> ctypes.c_void_p:
    """Return the memory (KV cache) handle for a context."""
    return _bind("llama_get_memory", LlamaMemory, LlamaContext)(ctx)


def llama_memory_clear(mem: ctypes.c_void_p, data: bool = True) -> None:
    """Clear the KV cache. data=True also zeroes the buffers."""
    _bind("llama_memory_clear", None, LlamaMemory, ctypes.c_bool)(mem, data)


def llama_memory_seq_rm(
    mem: ctypes.c_void_p, seq_id: int, p0: int, p1: int
) -> bool:
    """
    Remove cached tokens of sequence *seq_id* in position range [p0, p1).
    p0 < 0 means from the start; p1 < 0 means to the end.
    Returns False when a partial removal is not possible.
    """
    fn = _bind(
        "llama_memory_seq_rm",
        ctypes.c_bool,
        LlamaMemory,
        ctypes.c_int32,   # llama_seq_id
        ctypes.c_int32,   # llama_pos p0
        ctypes.c_int32,   # llama_pos p1
    )
    return bool(fn(mem, seq_id, p0, p1))


# ---------------------------------------------------------------------------
#  System info (useful for diagnostics)
# ---------------------------------------------------------------------------

def llama_print_system_info() -> str:
    fn = _bind("llama_print_system_info", ctypes.c_char_p)
    return fn().decode(errors="replace")


# ---------------------------------------------------------------------------
#  Device capacity (multi-GPU tensor-split)
# ---------------------------------------------------------------------------

def has_max_devices() -> bool:
    """True when this llama.dll exports llama_max_devices(). Every build with
    tensor_split support has exported this for years, but it is probed (not
    assumed) the same way has_memory_api()/has_penalties_sampler() are, so an
    exotic stripped build degrades to a documented fallback instead of an
    AttributeError. See discover.apply_gpu_split for how the fallback is used."""
    try:
        getattr(load_lib(), "llama_max_devices")
        return True
    except AttributeError:
        return False


def llama_max_devices() -> int:
    """Capacity of the tensor_split array this build's native loader will read
    from (a const float* with no length parameter of its own - the caller must
    match this exactly: too short is a real out-of-bounds read). Only call
    after has_max_devices() is True."""
    fn = _bind("llama_max_devices", ctypes.c_size_t)
    return int(fn())
