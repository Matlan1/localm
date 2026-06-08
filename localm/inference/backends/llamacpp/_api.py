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
        Number of bytes written.  If > *length*, the buffer was too small —
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


# ---------------------------------------------------------------------------
#  System info (useful for diagnostics)
# ---------------------------------------------------------------------------

def llama_print_system_info() -> str:
    fn = _bind("llama_print_system_info", ctypes.c_char_p)
    return fn().decode(errors="replace")
