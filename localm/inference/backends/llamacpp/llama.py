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
    Redirect fd 2 (stderr) to /dev/null for the duration of the block.

    llama.cpp writes model-loading noise (create_tensor, llama_kv_cache,
    sched_reserve, …) directly via fprintf(stderr, …), bypassing Python's
    logging system entirely.  The only reliable way to silence it is to
    redirect the file descriptor at the OS level.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
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
#  Sampler chain builder
# ---------------------------------------------------------------------------

def _build_sampler(
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.05,
    seed: int = _DEFAULT_SEED,
) -> int:
    """
    Construct a sampler chain:
        top_k → top_p → min_p → temperature → dist (random draw)

    For temperature ≤ 0 we use greedy sampling instead.
    """
    chain_params = api.llama_sampler_chain_default_params()
    chain_params.no_perf = True
    chain = api.llama_sampler_chain_init(chain_params)

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
        **_ignored,
    ) -> None:
        self._n_ctx       = n_ctx
        self._seed        = seed
        self._verbose     = verbose
        self._model_ptr   = None   # type: ignore[assignment]
        self._ctx_ptr     = None   # type: ignore[assignment]
        self._tokenizer   = None   # type: ignore[assignment]

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
    ) -> Iterator[int]:
        """
        Yield generated token ids one at a time.

        The prompt is fed in a single prefill batch; then tokens are generated
        one-by-one in decode mode using the sampler chain.

        Note: this implementation creates a fresh context for each call by
        re-using the same context handle.  Since there are no kv_cache_clear
        functions in this build, we reconstruct the context before each run.
        The cost is a small (~ms) context reinitialisation.
        """
        if not self._model_ptr:
            raise RuntimeError("Model not loaded")

        # Re-create a fresh context so the KV cache is empty
        # (llama_kv_self_clear is not present in this build)
        if self._ctx_ptr:
            api.llama_free(self._ctx_ptr)

        cp = api.llama_context_default_params()
        cp.n_ctx       = max(self._n_ctx, len(prompt_tokens) + max_new_tokens + 64)
        cp.n_batch     = min(cp.n_ctx, 2048)
        cp.n_ubatch    = cp.n_batch   # micro-batch must match so prefill fits in one call
        cp.offload_kqv = True

        n_prompt = len(prompt_tokens)
        if n_prompt == 0:
            return

        _ctx = _quiet_stderr if not self._verbose else contextlib.nullcontext

        # One contiguous suppression scope covering both context creation and
        # prefill.  The ROCm lazy-buffer verification messages
        # ("~llama_context: ROCm0 compute buffer size …") fire asynchronously
        # after llama_init_from_model returns but before the first llama_decode
        # completes, so separate per-call windows leave a gap.  Bridging them
        # into a single scope closes it.
        with _ctx():
            self._ctx_ptr = api.llama_init_from_model(self._model_ptr, cp)
            if not self._ctx_ptr:
                raise RuntimeError("Failed to (re)create llama context")
            # Update the tokenizer's ctx reference
            self._tokenizer._ctx = self._ctx_ptr

            # --- prefill prompt ---
            tok_arr = (llama_token * n_prompt)(*prompt_tokens)
            batch = api.llama_batch_get_one(tok_arr, n_prompt)
            ret = api.llama_decode(self._ctx_ptr, batch)

        if ret != 0:
            raise RuntimeError(f"llama_decode failed during prefill (code {ret})")

        # Build sampler
        sampler = _build_sampler(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=self._seed,
        )

        pos = n_prompt
        try:
            for _ in range(max_new_tokens):
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
                pos += 1
        finally:
            api.llama_sampler_free(sampler)

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
        """Convert a token-ID stream to a text-piece stream, then filter stop strings."""
        raw = (self._tokenizer.token_to_piece(t) for t in gen)
        yield from _filtered_stream(raw)

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
