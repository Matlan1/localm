# SPDX-License-Identifier: AGPL-3.0-or-later
"""
High-level LlamaCpp class - a pure-Python / ctypes replacement for the
llama-cpp-python ``Llama`` class.

Implements only the subset used by GgufBackend:
    llm = LlamaCpp(model_path, n_ctx=4096, n_gpu_layers=99, verbose=False)
    for chunk in llm.create_chat_completion(messages, max_tokens=1024,
                                             temperature=0.8, stream=True):
        token = chunk["choices"][0]["delta"]["content"]
"""

from __future__ import annotations

import codecs
import contextlib
import ctypes
import os
import tempfile
import threading
import time
import uuid
from typing import Dict, Generator, Iterable, Iterator, List, Optional

from . import _api as api
from ._structs import llama_token, LlamaChatMessage, LlamaBatch


_stderr_lock = threading.Lock()
_devnull_fd: Optional[int] = None


@contextlib.contextmanager
def _quiet_stderr():
    """
    Redirect fd 2 (stderr) away from the terminal for the duration of the block.

    llama.cpp writes model-loading noise (create_tensor, llama_kv_cache,
    sched_reserve, …) directly via fprintf(stderr, …), bypassing Python's
    logging system entirely.  The only reliable way to silence it is to
    redirect the file descriptor at the OS level.

    In debug mode the stream goes into the debug log file instead of
    /dev/null - native abort messages (the reason for a hard crash) land
    there, which is the difference between a diagnosable crash and a
    silent one.
    """
    global _devnull_fd
    with _stderr_lock:
        from localm.debuglog import native_stderr_target
        target_fd = native_stderr_target()
        should_close = False
        if target_fd is None:
            if _devnull_fd is None:
                _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            target_fd = _devnull_fd
        else:
            should_close = True
        saved_fd = os.dup(2)
        os.dup2(target_fd, 2)
        if should_close:
            os.close(target_fd)
        try:
            yield
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


class _CapturedStderr:
    """Holder yielded by _capture_stderr; .tail() reads the captured native text."""

    def __init__(self, path: str) -> None:
        self._path = path

    def tail(self, max_chars: int = 1500) -> str:
        # Best-effort read of the captured native stderr (the OOM / no-backends /
        # bad-quant reason); never raise from a diagnostics helper.
        try:
            with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            return ""
        text = text.strip()
        return text[-max_chars:] if len(text) > max_chars else text


@contextlib.contextmanager
def _capture_stderr():
    """
    Redirect fd 2 (native stderr) into a temp file for the duration of the block
    so the load-failure reason is retainable even when chat output must stay
    clean. On success the temp file is simply discarded (stderr stays clean);
    on failure the caller surfaces .tail() in the raised error. In debug mode
    the native stream still also lands in the debug log via _quiet_stderr at
    other sites, so this only adds the failure-diagnostic capture, not noise.
    """
    fd, path = tempfile.mkstemp(prefix="localm_load_", suffix=".log")
    saved_fd = os.dup(2)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield _CapturedStderr(path)
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        with contextlib.suppress(OSError):
            os.unlink(path)


# LLAMA_DEFAULT_SEED from llama.h
_DEFAULT_SEED = 0xFFFF_FFFF


def _make_chunk_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:12]


class _Tokenizer:
    """Thin wrapper for the vocab / tokenisation layer."""

    def __init__(self, model_ptr: int, ctx_ptr: int) -> None:
        self._vocab = api.llama_model_get_vocab(model_ptr)
        self._ctx   = ctx_ptr

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
            # buffer too small - reallocate and retry
            n_max = -n + 64
            buf = (llama_token * n_max)()
            n = api.llama_tokenize(
                self._vocab, raw, len(raw), buf, n_max,
                add_special=add_bos, parse_special=True,
            )
        if n < 0:
            raise RuntimeError(f"Tokenisation failed (returned {n})")
        return [buf[i] for i in range(n)]

    def token_to_piece_bytes(self, token: int) -> bytes:
        """Raw UTF-8 bytes of a single token, UNDECODED. A multibyte character
        can straddle two tokens, so callers that stream or join multiple tokens
        must accumulate bytes and decode the run as a whole (see ``detokenize``
        and ``_utf8_pieces``); decoding each token's bytes in isolation produces
        U+FFFD replacement characters at the split (R46)."""
        buf = ctypes.create_string_buffer(256)
        n = api.llama_token_to_piece(self._vocab, token, buf, 256, 0, True)
        if n < 0:
            buf = ctypes.create_string_buffer(-n + 4)
            n = api.llama_token_to_piece(self._vocab, token, buf, len(buf), 0, True)
            if n < 0:
                # The retry buffer is sized from the first call's answer, so a
                # correct runtime cannot land here; a still-negative n means the
                # decode genuinely failed. Slicing buf.raw[:n] with a negative n
                # would silently return garbage bytes instead (rule 5).
                raise RuntimeError(
                    f"llama_token_to_piece failed for token {token} (returned {n})"
                )
        return buf.raw[:n]

    def token_to_piece(self, token: int) -> str:
        """Single token decoded to text. Safe for whole tokens; for multi-token
        runs use ``detokenize``/``_utf8_pieces`` so a character split across a
        token boundary is not mangled."""
        return self.token_to_piece_bytes(token).decode("utf-8", errors="replace")

    def is_eog(self, token: int) -> bool:
        return api.llama_vocab_is_eog(self._vocab, token)


# Stop strings supplement llama_vocab_is_eog(): some models don't register their
# end-of-turn token in the vocab EOG list, so we also check each token's text.
_STOP_STRINGS: frozenset = frozenset({
    "<|im_end|>",       # ChatML  (Mistral, Qwen, etc.)
    "<end_of_turn>",    # Gemma 1-3
    "<turn|>",          # Gemma 4
    "<|eot_id|>",       # Llama 3
    "</s>",             # LLaMA 1/2
    "<|endoftext|>",    # GPT-2 / StarCoder
    "[/INST]",          # Mistral v1 instruct
    "<|end|>",          # Phi
})


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

    if needed <= 0:
        # Template not supported (< 0) or it rendered nothing (== 0): an empty
        # prompt would silently generate from thin air - fall back
        return _format_chatml(messages)

    if needed > buf_size:
        # Reallocate and retry
        buf = ctypes.create_string_buffer(needed + 64)
        needed = api.llama_chat_apply_template(tmpl_bytes, chat_arr, n, True, buf, len(buf))
        if needed <= 0:
            # Same guard as above: a failed or empty render falls back
            return _format_chatml(messages)

    return buf.raw[:needed].decode("utf-8", errors="replace")


# UTF-8-safe token-bytes -> text stream (R46): a multibyte character is often
# emitted across two or more tokens, so its bytes straddle a token boundary.
# Decoding each token in isolation yields U+FFFD at the split (mid-word mojibake);
# an incremental decoder buffers an incomplete trailing sequence until the next
# token's bytes complete it.

def _utf8_pieces(token_bytes: Iterator[bytes]) -> Iterator[str]:
    """Decode a stream of per-token byte pieces into text, never splitting a
    multibyte UTF-8 character across a token boundary. A character whose bytes
    are not yet complete is held back until the following token supplies the
    rest; a genuinely truncated tail at end-of-stream surfaces as U+FFFD via the
    final flush rather than being silently dropped."""
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    for b in token_bytes:
        out = dec.decode(b)
        if out:
            yield out
    tail = dec.decode(b"", final=True)
    if tail:
        yield tail


# Streaming stop-string filter: an end-of-turn marker like <|im_end|> is often
# spread across multiple tokens ('<','|','im','_','end','|>'), so a per-token
# check can never catch it; we filter the accumulated text stream instead.

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

    # Stream ended without a stop string - flush remaining buffer
    if buf:
        yield buf


# Internal-marker scrubbing: some finetunes emit training-format control markers
# as plain text - harmony channel tags (<|channel|>analysis ... <|message|>), the
# Gemma 4 turn/tool dialect (<|turn>model ... <turn|>, <|tool_call> ... <tool_call|>,
# <|"|> quote tokens), reserved vocab placeholders (<unused7>). These are model
# internals, not content, so chat output is ALWAYS scrubbed; debug mode
# (LOCALM_DEBUG) also writes the raw unscrubbed text to the debug log. Thinking-
# channel markers are not dropped but normalised to canonical <think> ... </think>.

# Marker scrubbing now lives in a shared module so every backend normalises the
# same way and the engine can apply it once for all of them. It is re-imported
# here (under the original private names) for the GGUF decode pipeline below; a
# second pass at the engine layer is idempotent.
from localm.inference.textnorm import scrub_stream as _scrub_stream  # noqa: E402


# Suffix tokens are prefilled in chunks of this size (matches n_batch ceiling)
_PREFILL_CHUNK = 2048


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    """Length of the longest common prefix of two token lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _build_sampler(
    vocab: int,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.05,
    repeat_penalty: float = 1.0,
    seed: int = _DEFAULT_SEED,
    grammar: Optional[str] = None,
    grammar_lazy: bool = False,
    grammar_triggers: Optional[List[str]] = None,
) -> int:
    """
    Construct a sampler chain:
        [grammar] → [penalties] → top_k → top_p → min_p → temperature → dist

    The optional grammar sampler sits first so it masks invalid tokens before
    any scoring or sampling stage sees them.  The repetition-penalty stage is
    added when ``repeat_penalty != 1.0`` and the DLL exports it - without it
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
    grammar_lazy:
        With *grammar_triggers*, generation stays UNCONSTRAINED until the
        output matches a trigger pattern; the grammar enforces from there
        (text-or-tool). When the runtime lacks the lazy export, or no
        triggers are given, the grammar is skipped entirely - a lazy request
        must never silently become a strict constraint (a strict grammar
        stalls thinking models, live-verified 2026-07-02).
    """
    chain_params = api.llama_sampler_chain_default_params()
    chain_params.no_perf = True
    chain = api.llama_sampler_chain_init(chain_params)

    # Grammar sampler masks logits before any scoring stage touches them
    if grammar and grammar_lazy:
        if grammar_triggers and api.has_lazy_grammar():
            api.llama_sampler_chain_add(
                chain,
                api.llama_sampler_init_grammar_lazy_patterns(
                    vocab, grammar.encode(), b"root",
                    [t.encode() for t in grammar_triggers],
                ),
            )
        else:
            from localm.debuglog import logger as _dbg
            _dbg.debug(
                "lazy grammar requested but %s; generating unconstrained",
                "no trigger patterns were given" if not grammar_triggers
                else "this llama build lacks llama_sampler_init_grammar_lazy_patterns")
    elif grammar:
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
        mmproj_path: Optional[str] = None,
        cancel_event: Optional["threading.Event"] = None,
        **_ignored,
    ) -> None:
        self._n_ctx       = n_ctx
        # Dynamic context window: starts at n_ctx, grows in n_ctx_grow steps
        # up to n_ctx_max when a conversation outgrows it. None/0 = unlimited
        # (the pre-dynamic behaviour: grow exactly as far as needed).
        # An explicitly requested base larger than the ceiling wins - the
        # user asked for it, the cap only governs automatic growth.
        self._n_ctx_max   = max(n_ctx_max, n_ctx) if n_ctx_max else None
        self._n_ctx_grow  = max(256, n_ctx_grow)
        self._seed        = seed
        self._verbose     = verbose
        self._model_ptr   = None   # type: ignore[assignment]
        self._ctx_ptr     = None   # type: ignore[assignment]
        self._mmproj_path = mmproj_path
        self._mtmd        = None   # MtmdContext (vision) when an mmproj is loaded
        self._tokenizer   = None   # type: ignore[assignment]
        # Serialize native calls (prefill/decode/free) against unload. Without
        # this, an unload on another thread can llama_free the context between
        # the generator's None-check and its next native call - a use-after-
        # free that crashes the GPU driver. The decode loop holds _gen_lock
        # around each native step; close()/_free_native take it too, after
        # setting _stop so an in-flight generation bails at its next step.
        self._gen_lock    = threading.RLock()
        self._stop        = threading.Event()
        self._inference_lock = threading.Lock()
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
        if n_gpu_layers >= 99:
            mp.use_mmap = False
        # Multi-GPU: honour the configured main_gpu_index (validated against
        # the devices actually visible right now); leaves the native default
        # (device 0) untouched when unset. See discover.apply_main_gpu.
        from localm.discover import apply_gpu_split, apply_main_gpu
        apply_main_gpu(mp)
        # Multi-GPU tensor-split: spreads the model across 2+ configured
        # devices when gpu_split_indices is set (see discover.apply_gpu_split).
        # The returned buffer must stay alive through llama_load_model_from_file
        # below - it is read once at load time, not held as a live pointer.
        _tensor_split_keepalive = apply_gpu_split(mp)

        # Preemptive model switching: wire llama.cpp's native load-progress
        # callback so a load can be ABORTED mid-flight. The callback returns false
        # once `cancel_event` is set, at which point llama_load_model_from_file
        # stops and returns NULL - so a model the user has already switched away
        # from does not run its (slow) load to completion. Keep the CFUNCTYPE
        # object alive on self for the whole load span; ctypes would otherwise GC
        # it and the native side would call freed memory. The callback must NEVER
        # raise: a Python exception inside a ctypes callback is reported as a
        # false return, which would abort a load we did not mean to cancel - so it
        # is fully guarded and defaults to "continue" (return True).
        self._cancel_event = cancel_event
        self._load_progress_cb = None   # keep-alive ref for the native callback
        if cancel_event is not None:
            @ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)
            def _load_progress(_progress, _user_data, _ev=cancel_event):
                try:
                    return not _ev.is_set()
                except Exception:
                    return True
            self._load_progress_cb = _load_progress
            mp.progress_callback = ctypes.cast(_load_progress, ctypes.c_void_p)

        # Capture native stderr for the load span (non-verbose only) so a NULL
        # return still carries its cause (OOM / no-backends / bad-quant); else the
        # only native diagnostic is discarded to devnull and the error is blind
        # (rule 5). Success path stays clean (captured text discarded). Verbose mode
        # leaves it (nullcontext); the native stream already reaches terminal/debug.
        _load_ctx = _capture_stderr if not verbose else contextlib.nullcontext
        with _load_ctx() as captured:
            self._model_ptr = api.llama_load_model_from_file(model_path, mp)
        if not self._model_ptr:
            # A NULL return when we asked to cancel is an ABORT, not a failure:
            # the load was superseded by a newer model selection. Report it as
            # such so the caller does not surface it as a load error.
            if cancel_event is not None and cancel_event.is_set():
                from localm.inference.backends.base import ModelLoadCancelled
                raise ModelLoadCancelled(
                    f"Model load aborted (superseded): {model_path}")
            detail = captured.tail() if captured is not None else ""
            hint = ("" if detail
                    else " (run with LOCALM_DEBUG=1 for the native load log)")
            suffix = f"\n{detail}" if detail else ""
            raise RuntimeError(
                f"Failed to load model: {model_path}{hint}{suffix}")

        # REC-GPULAYERS-CLAMP: llama.cpp already offloads min(n_gpu_layers, actual),
        # so an over-large value is harmless - but silently clamping a SPECIFIC
        # number is confusing, so surface a message. 99 = "offload all", so skip it.
        if 0 < n_gpu_layers < 99:
            try:
                actual = api.llama_model_n_layer(self._model_ptr)
                if actual and n_gpu_layers > actual:
                    from localm.debuglog import logger
                    logger.info(
                        "n_gpu_layers=%d exceeds the model's %d layers; "
                        "offloading all %d (the extra has no effect)",
                        n_gpu_layers, actual, actual)
            except Exception:
                pass  # introspection is best-effort; never block a successful load

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

        # Optional in-process vision (C1): load the mmproj via mtmd so image
        # messages can be answered. Best-effort - any failure (no mtmd.dll, an
        # incompatible mmproj) leaves the model text-only rather than breaking it.
        if mmproj_path:
            try:
                from .mtmd import MtmdContext
                mt = MtmdContext(mmproj_path, self._model_ptr)
                if mt.supports_vision:
                    self._mtmd = mt
                else:
                    mt.free()
            except Exception as exc:
                from localm.debuglog import logger
                logger.debug("mmproj load failed (%s); model stays text-only", exc)
                self._mtmd = None

    @property
    def supports_images(self) -> bool:
        """True when an mmproj is loaded and the projector supports vision."""
        return getattr(self, "_mtmd", None) is not None

    def close(self) -> None:
        """Release GPU/CPU memory held by this instance.

        Signals any in-flight generation to stop, then frees under _gen_lock
        so the free can never land between a generator's stop-check and its
        next native call (which would be a use-after-free GPU crash)."""
        self._stop.set()
        with self._gen_lock:
            self._cached_tokens = []
            if not (self._ctx_ptr or self._model_ptr):
                return
            # Suppress the ROCm lazy-buffer verification chatter the native
            # destructors write to stderr ("~llama_context: ... compute buffer
            # size ... matches expectation") - internal noise, not user output.
            try:
                _ctx = _quiet_stderr if not self._verbose else contextlib.nullcontext
                with _ctx():
                    self._free_native()
            except Exception:
                # Interpreter shutdown can break the fd redirection - free anyway
                self._free_native()

    def _free_native(self) -> None:
        if getattr(self, "_mtmd", None) is not None:
            self._mtmd.free()
            self._mtmd = None
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

    # Tokenisation (public helpers used by tests / introspection)

    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        return self._tokenizer.encode(text, add_bos=add_bos)

    def detokenize(self, tokens: Iterable[int]) -> str:
        # Join the raw token bytes first, then decode once, so a multibyte
        # character that straddles a token boundary is not split into U+FFFD
        # (R46). Per-token decode would mangle exactly those boundaries.
        raw = b"".join(self._tokenizer.token_to_piece_bytes(t) for t in tokens)
        return raw.decode("utf-8", errors="replace")

    def _create_batch(self, tokens: List[int], start_pos: int, logits_at_last_only: bool = True) -> LlamaBatch:
        n = len(tokens)
        batch = api.llama_batch_init(n, 0, 1)
        batch.n_tokens = n
        
        # cast pointers
        token_ptr = ctypes.cast(batch.token, ctypes.POINTER(llama_token))

        pos_ptr = ctypes.cast(batch.pos, ctypes.POINTER(ctypes.c_int32))
        n_seq_id_ptr = ctypes.cast(batch.n_seq_id, ctypes.POINTER(ctypes.c_int32))
        seq_id_ptr = ctypes.cast(batch.seq_id, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)))
        logits_ptr = ctypes.cast(batch.logits, ctypes.POINTER(ctypes.c_int8))
        
        for idx, tok in enumerate(tokens):
            token_ptr[idx] = tok
            pos_ptr[idx] = start_pos + idx
            n_seq_id_ptr[idx] = 1
            seq_id_ptr[idx][0] = 0
            if logits_at_last_only:
                logits_ptr[idx] = 1 if idx == n - 1 else 0
            else:
                logits_ptr[idx] = 1
                
        return batch

    def _generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repeat_penalty: float,
        grammar: Optional[str] = None,
        grammar_lazy: bool = False,
        grammar_triggers: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[int]:
        """
        Yield generated token ids one at a time.

        KV cache strategy: when this llama.cpp build exports the
        llama_memory_* API and the request fits in the live context, the
        common token prefix shared with the previous call is kept in the KV
        cache and only the new suffix is prefilled (fast follow-up turns in
        a chat).  Otherwise the context is recreated from scratch - the
        behaviour of older builds without KV-management functions.
        """
        with self._inference_lock:
            if not self._model_ptr:
                raise RuntimeError("Model not loaded")

            n_prompt = len(prompt_tokens)
            if n_prompt == 0:
                return

            # Dynamic window: shrink the generation budget to fit under the
            # ceiling rather than blowing past it; fail clearly when even a
            # minimal reply cannot fit any more.
            max_new_tokens = self._fit_generation_budget(n_prompt, max_new_tokens)

            if self._verbose:
                _ctx = contextlib.nullcontext
            elif grammar or grammar_lazy:
                _ctx = _quiet_stderr
            else:
                from localm.debuglog import dedup_native_stderr
                _ctx = dedup_native_stderr

            # If unlimited (<= 0), allocate a modest chunk up front and grow later
            initial_budget = max_new_tokens if max_new_tokens > 0 else 512
            needed = n_prompt + initial_budget + 64

            # One contiguous suppression scope covering context work and prefill.
            # The ROCm lazy-buffer verification messages fire asynchronously
            # after llama_init_from_model returns but before the first
            # llama_decode completes, so separate windows leave a gap.
            # Prefill (re)creates/decodes into the context, so it must hold the
            # lock against a concurrent unload too.
            with self._gen_lock:
                if self._stop.is_set():
                    return
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
                # A per-request seed (when provided) overrides the instance default
                # so temperature>0 sampling is reproducible; masked to uint32 to match
                # llama_sampler_init_dist's c_uint32 binding.
                seed=self._seed if seed is None else (seed & 0xFFFFFFFF),
                grammar=grammar,
                grammar_lazy=grammar_lazy,
                grammar_triggers=grammar_triggers,
            )

            pos = n_prompt
            # Why generation ended, read by callers as self.last_finish_reason.
            # Default "stop" - it must cover every early exit (EOG token, a
            # stop-string match in _filtered_stream abandoning this generator,
            # client abort). Only a genuinely exhausted token budget is "length".
            self.last_finish_reason = "stop"
            tokens_generated = 0
            try:
                # ONE contiguous _ctx() scope for the whole streaming loop, not
                # re-entered per native call: dedup_native_stderr() spins up a
                # background reader thread, so re-entering it per-token would
                # both reset its dedup state every time (defeating grouping
                # across tokens) and pay thread-creation cost per token. The
                # native calls below run unwrapped inside this single scope;
                # the yield in between is safe to leave wrapped too, since
                # inference is already serialized process-wide.
                with _ctx():
                    while max_new_tokens <= 0 or tokens_generated < max_new_tokens:
                        # --- locked native region 1: sample the next token ---
                        with self._gen_lock:
                            if self._stop.is_set() or self._ctx_ptr is None:
                                # The context was freed (unload) while we were
                                # generating. Stop cleanly instead of passing NULL
                                # into the native library, which crashes the driver.
                                self.last_finish_reason = "error"
                                break
                            # llama_sampler_sample() already ACCEPTS the sampled token
                            # into every stateful sampler in the chain (documented
                            # upstream as "sample and accept"). A second explicit
                            # accept here advanced the grammar parser twice per token,
                            # emptying its parse stacks and throwing std::runtime_error
                            # across the C ABI (WinError 0xe06d7363) - the "grammar
                            # sampler fault" that kept grammar enforcement dormant. It
                            # also double-counted tokens in the repetition-penalty
                            # window. Do NOT re-add an accept after sample.
                            token = api.llama_sampler_sample(sampler, self._ctx_ptr, -1)
                            eog = self._tokenizer.is_eog(token)

                        # Stop when the model signals end-of-generation via the vocabulary
                        if eog:
                            break   # last_finish_reason stays "stop"

                        yield token   # consumer runs here; an unload can interleave

                        # --- locked native region 2: feed the token back ---
                        with self._gen_lock:
                            if self._stop.is_set() or self._ctx_ptr is None:
                                self.last_finish_reason = "error"
                                break
                            batch = self._create_batch([token], pos, logits_at_last_only=True)
                            try:
                                ret = api.llama_decode(self._ctx_ptr, batch)
                                if ret != 0:
                                    # KV cache full or error.
                                    # Attempt mid-generation context growth if there is headroom.
                                    current_needed = pos + 512
                                    target = self._target_ctx(current_needed)
                                    if target > self._ctx_capacity:
                                        # We can grow! Re-prefill the context. Free the
                                        # old batch (its layout matches the OLD context)
                                        # BEFORE the re-prefill, because
                                        # _prefill_fresh_context can raise (NULL context,
                                        # a decode failure, an unload) and the native
                                        # batch must not leak if it does. AUDIT: a
                                        # llama_batch_init allocation is freed only by
                                        # llama_batch_free.
                                        api.llama_batch_free(batch)
                                        batch = None
                                        prompt_and_gen = self._cached_tokens.copy()
                                        self._prefill_fresh_context(prompt_and_gen, current_needed)
                                        # Retry decode on the newly grown context.
                                        batch = self._create_batch([token], pos, logits_at_last_only=True)
                                        ret = api.llama_decode(self._ctx_ptr, batch)

                                    if ret != 0:
                                        # The reply was cut short and we cannot grow further.
                                        # The cache bookkeeping has diverged from native KV
                                        # state, so invalidate it.
                                        self.last_finish_reason = "length"
                                        self._cached_tokens = []
                                        break
                                self._cached_tokens.append(token)
                                pos += 1
                                tokens_generated += 1
                            finally:
                                # Always release the native batch - including when
                                # _prefill_fresh_context above raises mid-growth.
                                if batch is not None:
                                    api.llama_batch_free(batch)
                    else:
                        # Budget exhausted without the model finishing its turn
                        self.last_finish_reason = "length"
            finally:
                api.llama_sampler_free(sampler)

    @staticmethod
    def _messages_with_markers(messages: List[Dict], marker: str):
        """Return (text_messages, images): a copy of *messages* where each image
        content part is replaced by *marker* in the text, plus the decoded RGB
        images (``(w, h, rgb_bytes)``) in marker order. The templated text_messages
        carry the marker so mtmd_tokenize can splice each image in at its place."""
        from localm.inference.media import decode_image_url
        out: List[Dict] = []
        images: List = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue
            parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    pil = decode_image_url(url).convert("RGB")
                    images.append((pil.width, pil.height, pil.tobytes()))
                    parts.append(marker)
                elif part.get("type") == "text":
                    parts.append(part.get("text", ""))
            new_msg = dict(msg)
            new_msg["content"] = "\n".join(p for p in parts if p)
            out.append(new_msg)
        return out, images

    def _generate_image(
        self,
        messages: List[Dict],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repeat_penalty: float,
        seed: Optional[int] = None,
    ) -> Iterator[int]:
        """Yield generated token ids for a chat whose prompt includes image(s).

        The image+text prompt is evaluated into the KV cache by mtmd (CPU clip);
        sampling then continues exactly like the text loop. Grammar is not applied
        on the image path. mtmd fills the KV from scratch, so the text KV-reuse
        cache is invalidated afterwards."""
        with self._inference_lock:
            if not self._model_ptr or getattr(self, "_mtmd", None) is None:
                raise RuntimeError("vision is not available on this model")

            text_messages, images = self._messages_with_markers(
                messages, self._mtmd.marker)
            prompt = _apply_model_template(self._model_ptr, text_messages)
            bos_markers = ("<bos>", "<s>", "﻿")
            add_special = not any(prompt.startswith(m) for m in bos_markers)

            _ctx = _quiet_stderr if not self._verbose else contextlib.nullcontext
            self.last_finish_reason = "stop"
            with self._gen_lock:
                if self._stop.is_set() or self._ctx_ptr is None:
                    return
                with _ctx():
                    # Clear any prior turn's KV so the mtmd prefill from position 0 is
                    # valid on a reused context, then evaluate the image+text prompt.
                    self._reset_kv_for_image()
                    pos = self._mtmd.eval_into(self._ctx_ptr, prompt, images,
                                               add_special=add_special)

            sampler = _build_sampler(
                vocab=self._tokenizer._vocab,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=self._seed if seed is None else (seed & 0xFFFFFFFF),
                grammar=None,
            )
            try:
                for _ in range(max_new_tokens):
                    with self._gen_lock:
                        if self._stop.is_set() or self._ctx_ptr is None:
                            self.last_finish_reason = "error"
                            break
                        # No explicit accept: llama_sampler_sample() accepts
                        # internally (see the why-comment in _generate).
                        token = api.llama_sampler_sample(sampler, self._ctx_ptr, -1)
                        eog = self._tokenizer.is_eog(token)
                    if eog:
                        break
                    yield token
                    with self._gen_lock:
                        if self._stop.is_set() or self._ctx_ptr is None:
                            self.last_finish_reason = "error"
                            break
                        batch = self._create_batch([token], pos, logits_at_last_only=True)
                        with _ctx():
                            ret = api.llama_decode(self._ctx_ptr, batch)
                        if ret != 0:
                            self.last_finish_reason = "length"
                            api.llama_batch_free(batch)
                            break
                        api.llama_batch_free(batch)
                        pos += 1
                else:
                    self.last_finish_reason = "length"
            finally:
                api.llama_sampler_free(sampler)

    def _fit_generation_budget(self, n_prompt: int, max_new_tokens: int) -> int:
        """
        Clamp the generation budget so prompt + reply fits under n_ctx_max.

        Raises RuntimeError when the prompt alone leaves no usable room -
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

        # Drop cached tokens past the common prefix. The empty-bookkeeping case
        # (``not self._cached_tokens``) is NOT redundant with ``prefix < len(...)``:
        # an image turn (_generate_image never appends its tokens) and a mid-generate
        # decode failure both leave the NATIVE KV populated while _cached_tokens is [].
        # Without this branch the guard is 0 < 0 (False), the wipe is skipped, and the
        # new prompt decodes onto stale KV at shifted positions (U-1: "sees earlier
        # text out of order"). When empty, prefix is 0, so seq_rm(0, 0, -1) drops the
        # residual and the suffix decodes cleanly from position 0.
        if prefix < len(self._cached_tokens) or not self._cached_tokens:
            if not api.llama_memory_seq_rm(mem, 0, prefix, -1):
                # Partial removal unsupported (e.g. SWA cache) - start over
                api.llama_memory_clear(mem, True)
                prefix = 0

        suffix = prompt_tokens[prefix:]
        for i in range(0, len(suffix), _PREFILL_CHUNK):
            if self._ctx_ptr is None:
                self._cached_tokens = []
                raise RuntimeError(
                    "Model was unloaded during prefill - request aborted."
                )
            chunk = suffix[i:i + _PREFILL_CHUNK]
            batch = self._create_batch(chunk, prefix + i, logits_at_last_only=True)
            ret = api.llama_decode(self._ctx_ptr, batch)
            api.llama_batch_free(batch)
            if ret != 0:
                # Cache state is now unknown - wipe it so the next call
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
        # more tokens than n_batch does not return an error - it aborts the
        # whole process inside the native library. Long chat histories
        # (prompt > 2048 tokens) land here whenever the context is recreated.
        n_batch = cp.n_batch
        for i in range(0, len(prompt_tokens), n_batch):
            chunk = prompt_tokens[i:i + n_batch]
            batch = self._create_batch(chunk, i, logits_at_last_only=True)
            ret = api.llama_decode(self._ctx_ptr, batch)
            api.llama_batch_free(batch)
            if ret != 0:
                self._cached_tokens = []
                raise RuntimeError(f"llama_decode failed during prefill (code {ret})")

        self._cached_tokens = list(prompt_tokens)

    def _reset_kv_for_image(self) -> None:
        """Empty the KV cache so a multimodal eval (which prefills from position 0)
        is valid on a REUSED context. mtmd_helper_eval_chunks does its own prefill
        at n_past=0, so a prior turn's tokens must be cleared first - otherwise a
        second image chat evaluates over stale KV and faults. Uses the memory API
        when present, else recreates an empty context (older builds)."""
        self._cached_tokens = []
        if self._memory_api_available():
            try:
                mem = api.llama_get_memory(self._ctx_ptr)
                api.llama_memory_clear(mem, True)
                return
            except Exception:
                pass
        self._prefill_fresh_context([], self._n_ctx)   # empty fresh context

    # Public API compatible with llama-cpp-python

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
        grammar_lazy: bool = False,
        grammar_triggers: Optional[List[str]] = None,
        seed: Optional[int] = None,
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

        from localm.inference.backends.base import messages_contain_image
        if getattr(self, "_mtmd", None) is not None and messages_contain_image(messages):
            # Image present + an mmproj is loaded: evaluate the image+text via mtmd
            # instead of the text-only prefill. The text path below is untouched.
            gen = self._generate_image(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=seed,
            )
        else:
            gen = self._generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                grammar=grammar,
                grammar_lazy=grammar_lazy,
                grammar_triggers=grammar_triggers,
                seed=seed,
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
                        "finish_reason": getattr(self, "last_finish_reason", "stop"),
                    }
                ],
            }

    def _decode_stream(self, gen: Iterator[int]) -> Iterator[str]:
        """Token-ID stream → text stream: stop-string filter, then marker
        scrub.  Chat output is always scrubbed; in debug mode the raw
        pre-scrub text is additionally written to the debug log."""
        from localm.debuglog import debug_enabled, logger
        # Decode token BYTES through one UTF-8-safe stream so a character split
        # across a token boundary is reassembled, not turned into U+FFFD (R46).
        raw = _utf8_pieces(self._tokenizer.token_to_piece_bytes(t) for t in gen)
        if debug_enabled():
            captured: list = []

            def _tee(pieces):
                for p in pieces:
                    captured.append(p)
                    yield p

            try:
                yield from _scrub_stream(_filtered_stream(_tee(raw)))
            finally:
                if captured:
                    logger.debug("raw model output:\n%s", "".join(captured))
        else:
            yield from _scrub_stream(_filtered_stream(raw))

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
        # final chunk with finish_reason ("length" = max_tokens budget ran out)
        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "choices": [{"index": 0, "delta": {},
                         "finish_reason": getattr(self, "last_finish_reason", "stop")}],
        }
