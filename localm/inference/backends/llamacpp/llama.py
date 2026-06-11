"""
High-level LlamaCpp class — a pure-Python / ctypes replacement for the
llama-cpp-python ``Llama`` class.

Implements only the subset used by GgufBackend:
    llm = LlamaCpp(model_path, n_ctx=4096, n_gpu_layers=99, verbose=False)
    for chunk in llm.create_chat_completion(messages, max_tokens=1024,
                                             temperature=0.8, stream=True):
        token = chunk["choices"][0]["delta"]["content"]
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import time
import uuid
from typing import Dict, Generator, Iterable, Iterator, List, Optional

from . import _api as api
from ._structs import llama_token, LlamaChatMessage, LlamaContextParams, LlamaModelParams


@contextlib.contextmanager
def _quiet_stderr():
    """
    Redirect fd 2 (stderr) away from the terminal for the duration of the block.

    llama.cpp writes model-loading noise (create_tensor, llama_kv_cache,
    sched_reserve, …) directly via fprintf(stderr, …), bypassing Python's
    logging system entirely.  The only reliable way to silence it is to
    redirect the file descriptor at the OS level.

    In debug mode the stream goes into the debug log file instead of
    /dev/null — native abort messages (the reason for a hard crash) land
    there, which is the difference between a diagnosable crash and a
    silent one.
    """
    from localm.debuglog import native_stderr_target
    target_fd = native_stderr_target()
    if target_fd is None:
        target_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(target_fd, 2)
    os.close(target_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)

# LLAMA_DEFAULT_SEED from llama.h
_DEFAULT_SEED = 0xFFFF_FFFF


def _make_chunk_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
#  Tokenisation helpers
# ---------------------------------------------------------------------------

class _Tokenizer:
    """Thin wrapper for the vocab / tokenisation layer."""

    def __init__(self, model_ptr: int, ctx_ptr: int) -> None:
        self._vocab = api.llama_model_get_vocab(model_ptr)
        self._ctx   = ctx_ptr

    # -- encode ----------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        raw = text.encode("utf-8", errors="replace")
        # First call: find required size (returns negative if buffer too small)
        n_max = len(raw) + 128
        buf = (llama_token * n_max)()
        n = api.llama_tokenize(
            self._vocab, raw, len(raw), buf, n_max,
            add_special=add_bos, parse_special=True,
        )
        if n < 0:
            # buffer too small — reallocate and retry
            n_max = -n + 64
            buf = (llama_token * n_max)()
            n = api.llama_tokenize(
                self._vocab, raw, len(raw), buf, n_max,
                add_special=add_bos, parse_special=True,
            )
        if n < 0:
            raise RuntimeError(f"Tokenisation failed (returned {n})")
        return [buf[i] for i in range(n)]

    # -- decode ----------------------------------------------------------------

    def token_to_piece(self, token: int) -> str:
        buf = ctypes.create_string_buffer(256)
        n = api.llama_token_to_piece(self._vocab, token, buf, 256, 0, True)
        if n < 0:
            buf = ctypes.create_string_buffer(-n + 4)
            n = api.llama_token_to_piece(self._vocab, token, buf, len(buf), 0, True)
        return buf.raw[:n].decode("utf-8", errors="replace")

    def is_eog(self, token: int) -> bool:
        return api.llama_vocab_is_eog(self._vocab, token)


# ---------------------------------------------------------------------------
#  Stop strings — supplement llama_vocab_is_eog()
#
#  Some models don't register their end-of-turn token in the vocabulary's EOG
#  list.  Checking the decoded text of each token against this set handles the
#  remaining cases gracefully.
# ---------------------------------------------------------------------------

_STOP_STRINGS: frozenset = frozenset({
    "<|im_end|>",       # ChatML  (Mistral, Qwen, etc.)
    "<end_of_turn>",    # Gemma
    "<|eot_id|>",       # Llama 3
    "</s>",             # LLaMA 1/2
    "<|endoftext|>",    # GPT-2 / StarCoder
    "[/INST]",          # Mistral v1 instruct
    "<|end|>",          # Phi
})


# ---------------------------------------------------------------------------
#  Chat-template helpers
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Return the plain-text portion of a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "")
            for p in content
            if p.get("type") == "text"
        )
    return str(content)


def _format_chatml(messages: List[Dict]) -> str:
    """Render messages as a ChatML-formatted prompt string (fallback)."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_text(msg.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _apply_model_template(model_ptr: int, messages: List[Dict]) -> str:
    """
    Format *messages* using the model's own embedded Jinja chat template.

    Falls back to :func:`_format_chatml` if:
    * The model has no embedded template (``llama_model_chat_template`` returns None).
    * The template call fails for any reason.
    """
    tmpl_str = api.llama_model_chat_template(model_ptr)
    if not tmpl_str:
        return _format_chatml(messages)

    tmpl_bytes = tmpl_str.encode()

    # Build C-array of llama_chat_message structs
    n = len(messages)
    chat_arr = (LlamaChatMessage * n)()
    for i, msg in enumerate(messages):
        chat_arr[i].role    = _extract_text(msg.get("role", "user")).encode()
        chat_arr[i].content = _extract_text(msg.get("content", "")).encode()

    # First call with a small buffer to get the required size
    buf_size = sum(len(_extract_text(m.get("content", ""))) for m in messages) * 3 + 512
    buf = ctypes.create_string_buffer(buf_size)
    needed = api.llama_chat_apply_template(tmpl_bytes, chat_arr, n, True, buf, buf_size)

    if needed < 0:
        # Template not supported — fall back
        return _format_chatml(messages)

    if needed > buf_size:
        # Reallocate and retry
        buf = ctypes.create_string_buffer(needed + 64)
        needed = api.llama_chat_apply_template(tmpl_bytes, chat_arr, n, True, buf, len(buf))
        if needed < 0:
            return _format_chatml(messages)

    return buf.raw[:needed].decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
#  Streaming stop-string filter
#
#  Many models signal end-of-turn with a token sequence like <|im_end|> that
#  is spread across multiple vocabulary tokens (e.g. '<', '|', 'im', '_',
#  'end', '|>').  A per-token text check can never catch these, so we filter
#  the accumulated text stream instead.
# ---------------------------------------------------------------------------

_MAX_STOP_LEN: int = max(len(s) for s in _STOP_STRINGS)


def _filtered_stream(pieces: Iterator[str]) -> Iterator[str]:
    """
    Pass text pieces through, halting the stream the moment any ``_STOP_STRINGS``
    entry appears in the accumulated output.

    We buffer the last ``_MAX_STOP_LEN - 1`` characters because a stop string
    may straddle two consecutive pieces.  The safe prefix is yielded immediately;
    the buffer is held back and only flushed if no stop string materialises.
    """
    buf = ""
    hold = _MAX_STOP_LEN - 1   # max chars that could be a partial stop prefix

    for piece in pieces:
        buf += piece

        # Check for any complete stop string in the buffer
        stop_idx = -1
        for stop in _STOP_STRINGS:
            idx = buf.find(stop)
            if idx != -1 and (stop_idx == -1 or idx < stop_idx):
                stop_idx = idx

        if stop_idx != -1:
            if stop_idx > 0:
                yield buf[:stop_idx]
            return  # discard the stop string and everything after

        # Yield the part of the buffer that can't be a stop-string prefix
        safe = max(0, len(buf) - hold)
        if safe > 0:
            yield buf[:safe]
            buf = buf[safe:]

    # Stream ended without a stop string — flush remaining buffer
    if buf:
        yield buf


# ---------------------------------------------------------------------------
#  Internal-marker scrubbing
#
#  Some finetunes emit their training-format control markers as plain text:
#  harmony-style channel tags (<|channel|>analysis … <|message|>), mangled
#  variants of them, or reserved vocabulary placeholders (<unused7>).  These
#  are model internals, not content — they are stripped from normal chat
#  output.  Debug mode (LOCALM_DEBUG) shows them raw for analysis.
# ---------------------------------------------------------------------------

import re as _re

_MARKER_RE = _re.compile(
    r"<\|?channel\|?>(thought|analysis|final|commentary)?"   # channel tags, incl. mangled
    r"|<\|message\|>"
    r"|<\|start\|>(assistant|user|system)?"
    r"|<\|return\|>"
    r"|<unused\d+>?"                                          # Gemma reserved tokens
)

# Longest text a partial marker could span across two stream pieces
_MARKER_HOLD = 24


def _scrub_stream(pieces: Iterator[str]) -> Iterator[str]:
    """
    Remove internal model markers from a text stream.

    Buffers the trailing ``_MARKER_HOLD`` characters because a marker can
    straddle two pieces; flushes the remainder when the stream ends.
    """
    buf = ""
    for piece in pieces:
        buf += piece
        buf = _MARKER_RE.sub("", buf)
        safe = max(0, len(buf) - _MARKER_HOLD)
        if safe > 0:
            yield buf[:safe]
            buf = buf[safe:]
    buf = _MARKER_RE.sub("", buf)
    if buf:
        yield buf


# ---------------------------------------------------------------------------
#  KV cache helpers
# ---------------------------------------------------------------------------

# Suffix tokens are prefilled in chunks of this size (matches n_batch ceiling)
_PREFILL_CHUNK = 2048


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    """Length of the longest common prefix of two token lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ---------------------------------------------------------------------------
#  Sampler chain builder
# ---------------------------------------------------------------------------

def _build_sampler(
    vocab: int,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.05,
    repeat_penalty: float = 1.0,
    seed: int = _DEFAULT_SEED,
    grammar: Optional[str] = None,
) -> int:
    """
    Construct a sampler chain:
        [grammar] → [penalties] → top_k → top_p → min_p → temperature → dist

    The optional grammar sampler sits first so it masks invalid tokens before
    any scoring or sampling stage sees them.  The repetition-penalty stage is
    added when ``repeat_penalty != 1.0`` and the DLL exports it — without it
    models prone to looping repeat the same marker lines until max_tokens.
    For temperature ≤ 0 greedy sampling replaces the stochastic stages.

    Parameters
    ----------
    vocab:
        Vocabulary pointer from ``llama_model_get_vocab()``.  Required when
        *grammar* is provided; unused otherwise.
    grammar:
        GBNF grammar string.  When supplied, only token sequences that match
        this grammar at the current parse position are eligible for sampling.
        Pass ``None`` (the default) to skip grammar-constrained sampling.
    """
    chain_params = api.llama_sampler_chain_default_params()
    chain_params.no_perf = True
    chain = api.llama_sampler_chain_init(chain_params)

    # Grammar sampler masks logits before any scoring stage touches them
    if grammar:
        api.llama_sampler_chain_add(
            chain,
            api.llama_sampler_init_grammar(vocab, grammar.encode(), b"root"),
        )

    # Repetition penalty applies to greedy and stochastic sampling alike
    if repeat_penalty and repeat_penalty != 1.0 and api.has_penalties_sampler():
        api.llama_sampler_chain_add(
            chain,
            api.llama_sampler_init_penalties(64, repeat_penalty, 0.0, 0.0),
        )

    if temperature <= 0.0:
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_greedy())
    else:
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_top_k(top_k))
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_top_p(top_p, 1))
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_min_p(min_p, 1))
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_temp(temperature))
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_dist(seed))

    return chain


# ---------------------------------------------------------------------------
#  LlamaCpp — main public class
# ---------------------------------------------------------------------------

class LlamaCpp:
    """
    In-process GGUF inference backed by the native llama.dll.

    This is a drop-in replacement for ``llama_cpp.Llama`` for the subset of
    the API used by :class:`~localm.inference.backends.gguf.GgufBackend`.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        verbose: bool = False,
        seed: int = _DEFAULT_SEED,
        n_threads: Optional[int] = None,
        n_ctx_max: Optional[int] = None,
        n_ctx_grow: int = 4096,
        **_ignored,
    ) -> None:
        self._n_ctx       = n_ctx
        # Dynamic context window: starts at n_ctx, grows in n_ctx_grow steps
        # up to n_ctx_max when a conversation outgrows it. None/0 = unlimited
        # (the pre-dynamic behaviour: grow exactly as far as needed).
        # An explicitly requested base larger than the ceiling wins — the
        # user asked for it, the cap only governs automatic growth.
        self._n_ctx_max   = max(n_ctx_max, n_ctx) if n_ctx_max else None
        self._n_ctx_grow  = max(256, n_ctx_grow)
        self._seed        = seed
        self._verbose     = verbose
        self._model_ptr   = None   # type: ignore[assignment]
        self._ctx_ptr     = None   # type: ignore[assignment]
        self._tokenizer   = None   # type: ignore[assignment]
        # Persistent KV cache bookkeeping (prefix reuse across calls)
        self._cached_tokens: List[int] = []   # tokens currently in the KV cache
        self._ctx_capacity  = n_ctx           # n_ctx of the live context
        self._kv_supported: Optional[bool] = None   # lazy llama_memory_* probe

        _ctx = _quiet_stderr if not verbose else contextlib.nullcontext

        with _ctx():
            api.llama_backend_init()

        # --- load model ---
        mp = api.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers

        with _ctx():
            self._model_ptr = api.llama_load_model_from_file(model_path, mp)
        if not self._model_ptr:
            raise RuntimeError(f"Failed to load model: {model_path}")

        # --- create context ---
        cp = api.llama_context_default_params()
        cp.n_ctx             = n_ctx
        cp.n_batch           = min(n_ctx, 2048)
        cp.n_ubatch          = cp.n_batch   # match micro-batch to batch
        cp.offload_kqv       = True
        cp.flash_attn_type   = -1  # keep default (unspecified)
        if n_threads is not None:
            cp.n_threads       = n_threads
            cp.n_threads_batch = n_threads

        with _ctx():
            self._ctx_ptr = api.llama_init_from_model(self._model_ptr, cp)
        if not self._ctx_ptr:
            api.llama_free_model(self._model_ptr)
            raise RuntimeError("Failed to create llama context")

        self._tokenizer = _Tokenizer(self._model_ptr, self._ctx_ptr)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Release GPU/CPU memory held by this instance."""
        self._cached_tokens = []
        if not (self._ctx_ptr or self._model_ptr):
            return
        # Suppress the ROCm lazy-buffer verification chatter the native
        # destructors write to stderr ("~llama_context: ... compute buffer
        # size ... matches expectation") — internal noise, not user output.
        try:
            _ctx = _quiet_stderr if not self._verbose else contextlib.nullcontext
            with _ctx():
                self._free_native()
        except Exception:
            # Interpreter shutdown can break the fd redirection — free anyway
            self._free_native()

    def _free_native(self) -> None:
        if self._ctx_ptr:
            api.llama_free(self._ctx_ptr)
            self._ctx_ptr = None
        if self._model_ptr:
            api.llama_free_model(self._model_ptr)
            self._model_ptr = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Tokenisation (public helpers used by tests / introspection)        #
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        return self._tokenizer.encode(text, add_bos=add_bos)

    def detokenize(self, tokens: Iterable[int]) -> str:
        return "".join(self._tokenizer.token_to_piece(t) for t in tokens)

    # ------------------------------------------------------------------ #
    #  Core generation loop                                               #
    # ------------------------------------------------------------------ #

    def _generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repeat_penalty: float,
        grammar: Optional[str] = None,
    ) -> Iterator[int]:
        """
        Yield generated token ids one at a time.

        KV cache strategy: when this llama.cpp build exports the
        llama_memory_* API and the request fits in the live context, the
        common token prefix shared with the previous call is kept in the KV
        cache and only the new suffix is prefilled (fast follow-up turns in
        a chat).  Otherwise the context is recreated from scratch — the
        behaviour of older builds without KV-management functions.
        """
        if not self._model_ptr:
            raise RuntimeError("Model not loaded")

        n_prompt = len(prompt_tokens)
        if n_prompt == 0:
            return

        # Dynamic window: shrink the generation budget to fit under the
        # ceiling rather than blowing past it; fail clearly when even a
        # minimal reply cannot fit any more.
        max_new_tokens = self._fit_generation_budget(n_prompt, max_new_tokens)

        _ctx = _quiet_stderr if not self._verbose else contextlib.nullcontext
        needed = n_prompt + max_new_tokens + 64

        # One contiguous suppression scope covering context work and prefill.
        # The ROCm lazy-buffer verification messages fire asynchronously
        # after llama_init_from_model returns but before the first
        # llama_decode completes, so separate windows leave a gap.
        with _ctx():
            if self._can_reuse_kv(needed):
                self._prefill_with_reuse(prompt_tokens)
            else:
                self._prefill_fresh_context(prompt_tokens, needed)

        # Build sampler
        sampler = _build_sampler(
            vocab=self._tokenizer._vocab,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=self._seed,
            grammar=grammar,
        )

        pos = n_prompt
        try:
            for _ in range(max_new_tokens):
                if self._ctx_ptr is None:
                    # The context was freed (unload/close) while we were
                    # generating. Stop cleanly instead of passing NULL into
                    # the native library, which crashes the GPU driver.
                    raise RuntimeError(
                        "Model was unloaded during generation — request aborted."
                    )
                token = api.llama_sampler_sample(sampler, self._ctx_ptr, -1)
                api.llama_sampler_accept(sampler, token)

                # Stop when the model signals end-of-generation via the vocabulary
                if self._tokenizer.is_eog(token):
                    break

                yield token

                # Feed the new token back for next step
                tok_one = (llama_token * 1)(token)
                batch = api.llama_batch_get_one(tok_one, 1)
                with _ctx():
                    ret = api.llama_decode(self._ctx_ptr, batch)
                if ret != 0:
                    # KV cache full or error
                    break
                self._cached_tokens.append(token)
                pos += 1
        finally:
            api.llama_sampler_free(sampler)

    # ------------------------------------------------------------------ #
    #  Dynamic context window                                              #
    # ------------------------------------------------------------------ #

    def _fit_generation_budget(self, n_prompt: int, max_new_tokens: int) -> int:
        """
        Clamp the generation budget so prompt + reply fits under n_ctx_max.

        Raises RuntimeError when the prompt alone leaves no usable room —
        the conversation has genuinely outgrown the configured ceiling.
        """
        if not self._n_ctx_max:
            return max_new_tokens
        room = self._n_ctx_max - n_prompt - 64
        if room < 32:
            raise RuntimeError(
                f"Conversation ({n_prompt} tokens) has outgrown the maximum "
                f"context window (n_ctx_max={self._n_ctx_max}). Start a new "
                f"chat, or raise it:  localm config n_ctx_max 32768  "
                f"(or set ctx_auto true to size it from free VRAM)."
            )
        return min(max_new_tokens, room)

    def _target_ctx(self, needed: int) -> int:
        """
        Context size to create for a request needing *needed* tokens:
        grow in n_ctx_grow steps (avoids a rebuild on every turn), never
        below the configured base, capped at n_ctx_max when one is set.
        """
        grow = self._n_ctx_grow
        target = ((needed + grow - 1) // grow) * grow
        target = max(self._n_ctx, target)
        if self._n_ctx_max:
            # _fit_generation_budget guarantees needed <= n_ctx_max here
            target = min(target, self._n_ctx_max)
        return target

    # ------------------------------------------------------------------ #
    #  KV cache management                                                 #
    # ------------------------------------------------------------------ #

    def _memory_api_available(self) -> bool:
        """Probe once for the llama_memory_* function family."""
        if self._kv_supported is None:
            try:
                self._kv_supported = api.has_memory_api()
            except Exception:
                self._kv_supported = False
        return self._kv_supported

    def _can_reuse_kv(self, needed_tokens: int) -> bool:
        """True when the live context and its KV cache can serve this call."""
        return (
            self._ctx_ptr is not None
            and needed_tokens <= self._ctx_capacity
            and self._memory_api_available()
        )

    def _prefill_with_reuse(self, prompt_tokens: List[int]) -> None:
        """
        Prefill keeping the common prefix with the previous call in the KV
        cache: remove diverging cached tokens, decode only the new suffix.
        """
        mem = api.llama_get_memory(self._ctx_ptr)

        prefix = _common_prefix_len(self._cached_tokens, prompt_tokens)
        # The model must decode at least the final prompt token so the
        # logits for sampling position -1 are fresh.
        if prefix == len(prompt_tokens):
            prefix -= 1

        if prefix < len(self._cached_tokens):
            # Drop cached tokens past the common prefix
            if not api.llama_memory_seq_rm(mem, 0, prefix, -1):
                # Partial removal unsupported (e.g. SWA cache) — start over
                api.llama_memory_clear(mem, True)
                prefix = 0

        suffix = prompt_tokens[prefix:]
        for i in range(0, len(suffix), _PREFILL_CHUNK):
            if self._ctx_ptr is None:
                self._cached_tokens = []
                raise RuntimeError(
                    "Model was unloaded during prefill — request aborted."
                )
            chunk = suffix[i:i + _PREFILL_CHUNK]
            tok_arr = (llama_token * len(chunk))(*chunk)
            batch = api.llama_batch_get_one(tok_arr, len(chunk))
            ret = api.llama_decode(self._ctx_ptr, batch)
            if ret != 0:
                # Cache state is now unknown — wipe it so the next call
                # starts clean rather than trusting a half-decoded prefix
                self._cached_tokens = []
                try:
                    api.llama_memory_clear(mem, True)
                except Exception:
                    pass
                raise RuntimeError(f"llama_decode failed during prefill (code {ret})")

        self._cached_tokens = list(prompt_tokens)

    def _prefill_fresh_context(self, prompt_tokens: List[int], needed: int) -> None:
        """Recreate the context (empty KV cache) and prefill the full prompt."""
        if self._ctx_ptr:
            api.llama_free(self._ctx_ptr)
            self._ctx_ptr = None
        self._cached_tokens = []

        cp = api.llama_context_default_params()
        cp.n_ctx       = self._target_ctx(needed)
        cp.n_batch     = min(cp.n_ctx, 2048)
        cp.n_ubatch    = cp.n_batch   # micro-batch must match so prefill fits in one call
        cp.offload_kqv = True

        self._ctx_ptr = api.llama_init_from_model(self._model_ptr, cp)
        if not self._ctx_ptr:
            raise RuntimeError("Failed to (re)create llama context")
        self._ctx_capacity = cp.n_ctx
        # Update the tokenizer's ctx reference
        self._tokenizer._ctx = self._ctx_ptr

        # Prefill in n_batch-sized chunks. A single llama_decode call with
        # more tokens than n_batch does not return an error — it aborts the
        # whole process inside the native library. Long chat histories
        # (prompt > 2048 tokens) land here whenever the context is recreated.
        n_batch = cp.n_batch
        for i in range(0, len(prompt_tokens), n_batch):
            chunk = prompt_tokens[i:i + n_batch]
            tok_arr = (llama_token * len(chunk))(*chunk)
            batch = api.llama_batch_get_one(tok_arr, len(chunk))
            ret = api.llama_decode(self._ctx_ptr, batch)
            if ret != 0:
                self._cached_tokens = []
                raise RuntimeError(f"llama_decode failed during prefill (code {ret})")

        self._cached_tokens = list(prompt_tokens)

    # ------------------------------------------------------------------ #
    #  Public API compatible with llama-cpp-python                        #
    # ------------------------------------------------------------------ #

    def create_chat_completion(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stream: bool = False,
        grammar: Optional[str] = None,
        **_ignored,
    ):
        """
        Generate a chat completion.

        With ``stream=True`` yields dicts matching the llama-cpp-python
        streaming format:
            {"choices": [{"delta": {"content": "<token>"}}]}

        With ``stream=False`` returns a single completion dict.
        """
        # Use the model's embedded chat template when available (Gemma, Llama3,
        # Mistral, etc.) so we don't force ChatML on every model.
        prompt = _apply_model_template(self._model_ptr, messages)

        # If the template already encodes a BOS marker (e.g. Gemma's "<bos>"),
        # parse_special=True (used inside encode) will convert it to the BOS
        # token, so we must NOT also ask for add_special=True to avoid doubling.
        # Otherwise keep add_bos=True so the tokenizer prepends BOS normally.
        bos_markers = ("<bos>", "<s>", "﻿")
        add_bos = not any(prompt.startswith(m) for m in bos_markers)
        tokens = self._tokenizer.encode(prompt, add_bos=add_bos)

        gen = self._generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            grammar=grammar,
        )

        if stream:
            return self._stream_chunks(gen)
        else:
            full_text = "".join(self._decode_stream(gen))
            return {
                "id": _make_chunk_id(),
                "object": "chat.completion",
                "created": int(time.time()),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }
                ],
            }

    def _decode_stream(self, gen: Iterator[int]) -> Iterator[str]:
        """Token-ID stream → text stream: stop-string filter, then marker scrub
        (skipped in debug mode so raw model output is observable)."""
        from localm.debuglog import debug_enabled
        raw = (self._tokenizer.token_to_piece(t) for t in gen)
        stream = _filtered_stream(raw)
        if not debug_enabled():
            stream = _scrub_stream(stream)
        yield from stream

    def _stream_chunks(self, gen: Iterator[int]) -> Generator:
        chunk_id = _make_chunk_id()
        created  = int(time.time())
        for text in self._decode_stream(gen):
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
        # final chunk with finish_reason
        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
