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
import re
import tempfile
import threading
import time
import uuid
from typing import Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

from localm.inference import pretokenizer_guard

from . import _api as api
from ._structs import (
    llama_token, LlamaChatMessage, LlamaBatch, LlamaModelTensorBuftOverride,
    set_use_mmap)


_stderr_lock = threading.Lock()
_devnull_fd: Optional[int] = None

# Surfaced (as InvalidGrammarError) when the native GBNF parser rejects a grammar.
_INVALID_GRAMMAR_MSG = "invalid GBNF grammar (the native parser could not parse it)"


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


def _stderr_ctx_for_generate(verbose: bool):
    """Which stderr-handling context manager _generate() wraps its (single,
    contiguous) prefill + decode-loop scope in. A pure function, isolated from
    grammar/grammar_lazy entirely, so the decision is directly unit-testable
    without constructing a real LlamaCpp/native model.

    verbose already lets native output straight through unfiltered, so there
    is nothing here to suppress or group - nullcontext. Otherwise this is
    always ``dedup_native_stderr`` (debuglog.py), for grammar-constrained
    requests exactly the same as plain ones: the lazy grammar sampler logs
    "Grammar still awaiting trigger after token N" for every token until the
    trigger fires, and _LineGrouper's repeat-count collapsing (which also
    tolerates a repeating multi-line CYCLE, not just one line) handles that
    volume without hiding real grammar diagnostics.

    Sharing dedup_native_stderr is safe here specifically because BOTH of
    _generate()'s call sites for this context manager wrap one whole
    prefill/decode-loop scope, never re-entered per token - the one
    requirement dedup_native_stderr's own docstring imposes."""
    if verbose:
        return contextlib.nullcontext
    from localm.debuglog import dedup_native_stderr
    return dedup_native_stderr


# llama.cpp's own load-time report of where each backend's share of the model's
# weights ended up, e.g. "load_tensors:        ROCm0 model buffer size =   3.35 MiB"
# or "load_tensors:    ROCm_Host model buffer size =   3.20 MiB". This is the ONLY
# place that per-backend split is ever reported - llama.h exposes no API for it
# (no buffer/tensor-size introspection function is bound in _api.py, and none
# exists to bind), it is a printf inside llama.cpp's own model-loading code.
#
# The backend-name group must stay [A-Za-z0-9_]+, never \S+: this text comes from
# captured native stderr, which a hostile GGUF could in principle influence (an
# embedded string surfacing near a "load_tensors:" line), and \S+ is
# polynomial-time on adversarial input - a string with many "load_tensors:"
# restart points each failing to complete lets \S+ backtrack across the whole
# remaining text at every one. Every real backend name (ROCm0, ROCm_Host, CUDA0,
# CUDA_Host, Vulkan0, Metal, CPU, CPU_Mapped, ...) is plain
# alphanumeric/underscore, which shares no characters with "load_tensors:" (the
# colon) or the literal " model buffer size" (the leading space) that follows, so
# the class has a clean boundary and a failed attempt terminates immediately with
# no backtracking.
_MODEL_BUFFER_RE = re.compile(
    r"load_tensors:\s*([A-Za-z0-9_]+) model buffer size\s*=\s*([\d.]+)\s*MiB")


class _CapturedStderr:
    """Holder yielded by _capture_stderr; .tail() reads the captured native text."""

    def __init__(self, path: str) -> None:
        self._path = path

    def _read(self) -> str:
        try:
            with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    def tail(self, max_chars: int = 1500) -> str:
        # Best-effort read of the captured native stderr (the OOM / no-backends /
        # bad-quant reason); never raise from a diagnostics helper.
        text = self._read().strip()
        return text[-max_chars:] if len(text) > max_chars else text

    def model_buffers(self) -> list:
        """Every ``load_tensors: <backend> model buffer size = N MiB`` line from
        the captured native load log, as ``[{"backend", "mib", "is_ram"}, ...]``.

        ``is_ram`` classifies llama.cpp/ggml's own backend naming: a bare "CPU"
        or "CPU_*" buffer, or any GPU backend's "*_Host" pinned-transfer buffer,
        is system RAM; a plain device name (ROCm0, CUDA0, Vulkan0, Metal, ...) is
        that device's VRAM. Best-effort: [] on any read failure, or when this
        llama.cpp build's output does not match (a future format change) - a
        caller must treat an empty list as "not reported", never as "0 bytes
        everywhere"."""
        out = []
        for m in _MODEL_BUFFER_RE.finditer(self._read()):
            name = m.group(1)
            out.append({
                "backend": name,
                "mib": float(m.group(2)),
                "is_ram": name == "CPU" or name.startswith("CPU_")
                          or name.endswith("_Host"),
            })
        return out


@contextlib.contextmanager
def _capture_stderr():
    """
    Redirect fd 2 (native stderr) into a temp file for the duration of the block
    so the load report is retainable even when chat output must stay clean:
    the failure reason (OOM / no-backends / bad-quant) on a NULL return, and the
    per-backend weight placement (see _MODEL_BUFFER_RE) on success.

    The temp file is removed when the block exits, so a caller that wants
    .tail()/.model_buffers() MUST read them from inside the ``with`` block, not
    after it - reading after exit silently returns "" / [].

    When debug mode is on, the full captured text is ALSO appended to the debug
    log before removal, matching _quiet_stderr's "debug mode sees the native
    stream" contract at its other call sites - this capture is the one span
    _quiet_stderr does not cover (see its docstring), so without this the load's
    own native report would be invisible even under LOCALM_DEBUG=1.
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
        from localm.debuglog import native_stderr_target
        target_fd = native_stderr_target()
        if target_fd is not None:
            try:
                with open(path, "rb") as src:
                    os.write(target_fd, src.read())
            except OSError:
                pass
            finally:
                os.close(target_fd)
        with contextlib.suppress(OSError):
            os.unlink(path)


class _CapturedStdio:
    """Holder yielded by _capture_stdio; .tail() reads whatever native text
    landed on EITHER stream while it was open. Distinct from _CapturedStderr
    above (fd 2 only, used for llama.cpp's structured load_tensors report) -
    this one exists purely to keep an uncategorised native banner off the
    terminal and, on failure, off the floor entirely."""

    def __init__(self, out_path: str, err_path: str) -> None:
        self._out_path = out_path
        self._err_path = err_path

    def _read(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    def tail(self, max_chars: int = 1500) -> str:
        # Best-effort read of whatever landed on stdout/stderr; never raise
        # from a diagnostics helper. Order between the two streams is not
        # preserved (they are separate files) - good enough for "was there
        # anything at all", which is all a caller needs to decide whether to
        # surface it.
        text = "\n".join(s for s in (self._read(self._out_path),
                                      self._read(self._err_path)) if s).strip()
        return text[-max_chars:] if len(text) > max_chars else text


@contextlib.contextmanager
def _capture_stdio():
    """Redirect BOTH fd 1 (stdout) and fd 2 (stderr) into temp files for the
    duration of the block.

    Unlike _capture_stderr above (fd 2 only - llama.cpp's structured
    load_tensors report is always on stderr), this exists for native output
    whose stream is not documented and not worth trusting either way: the
    ggml/backend-registration banner a GPU build prints while its native
    library loads (e.g. "ggml_cuda_init: found 1 ROCm devices..."), which
    load_lib() (_loader.py) triggers with no capture scope of its own. Left
    unredirected it lands mid-line in whatever this process's inherited
    console is currently rendering - a parent-owned live Rich load spinner,
    on the one caller (GgufWorker.load) this was written for.

    Always pair with debuglog.suppress_console_mirror() around the SAME
    scope: this only handles the OS-level fd redirect, and load_lib() also
    calls logger.warning (e.g. "no ggml compute backends registered") - in
    debug mode that reaches the terminal through the console mirror, which
    is BY DESIGN immune to an fd redirect (see suppress_console_mirror's own
    docstring; _capture_stderr's caller in this same module hit the exact
    same gap first, for the exact same reason).

    The temp files are removed when the block exits, so a caller that wants
    .tail() MUST read it from inside the ``with`` block, same contract as
    _capture_stderr above - see its docstring for why (reading after exit
    silently returns "").
    """
    out_fd, out_path = tempfile.mkstemp(prefix="localm_loadlib_", suffix=".out.log")
    err_fd, err_path = tempfile.mkstemp(prefix="localm_loadlib_", suffix=".err.log")
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    os.dup2(out_fd, 1)
    os.dup2(err_fd, 2)
    os.close(out_fd)
    os.close(err_fd)
    try:
        yield _CapturedStdio(out_path, err_path)
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        from localm.debuglog import native_stderr_target
        target_fd = native_stderr_target()
        if target_fd is not None:
            try:
                for p in (out_path, err_path):
                    with open(p, "rb") as src:
                        os.write(target_fd, src.read())
            except OSError:
                pass
            finally:
                os.close(target_fd)
        for p in (out_path, err_path):
            with contextlib.suppress(OSError):
                os.unlink(p)


# LLAMA_DEFAULT_SEED from llama.h
_DEFAULT_SEED = 0xFFFF_FFFF


def _make_chunk_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:12]


class _Tokenizer:
    """Thin wrapper for the vocab / tokenisation layer."""

    def __init__(self, model_ptr: int, ctx_ptr: int) -> None:
        self._vocab = api.llama_model_get_vocab(model_ptr)
        self._ctx   = ctx_ptr
        # Read once per load, not per encode: the value cannot change while the
        # model is loaded, and encode() runs on every request.
        self._pre_type = pretokenizer_guard.read_pre_type(model_ptr, api)
        if pretokenizer_guard.policy_for(self._pre_type) is not None:
            from localm.debuglog import logger
            logger.warning(
                "tokenizer: this model declares tokenizer.ggml.pre=%r, whose "
                "pre-tokenizer regex aborts the process on long unbroken runs "
                "of one character class. Such input will be refused rather "
                "than tokenised.", self._pre_type)

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        pretokenizer_guard.check_text(self._pre_type, text)
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
        U+FFFD replacement characters at the split."""
        buf = ctypes.create_string_buffer(256)
        n = api.llama_token_to_piece(self._vocab, token, buf, 256, 0, True)
        if n < 0:
            buf = ctypes.create_string_buffer(-n + 4)
            n = api.llama_token_to_piece(self._vocab, token, buf, len(buf), 0, True)
            if n < 0:
                # The retry buffer is sized from the first call's answer, so a
                # correct runtime cannot land here; a still-negative n means the
                # decode genuinely failed. Slicing buf.raw[:n] with a negative n
                # would silently return garbage bytes instead.
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


def _warn_chatml_fallback(reason: str) -> None:
    """Log that a model's own chat template could not be used and generic ChatML
    was substituted.

    ``llama_chat_apply_template`` is not a real Jinja engine - it pattern-matches
    the model's template string against a fixed list of about 54 hardcoded
    signatures in llama.cpp's own ``llm_chat_apply_template`` and returns -1 for
    anything it does not recognize. A model whose real dialect is not among those
    - most non-mainstream VLMs, e.g. moondream2 - falls back to generic ChatML,
    feeding chat AND vision requests alike an out-of-distribution prompt the model
    was never fine-tuned on, which shows up as degenerate or hallucinated output.
    Surfacing it does not fix that model's output quality (the real fix is routing
    through llama.cpp's own full chat-template engine), but it stops the mismatch
    from being invisible."""
    from localm.debuglog import logger
    logger.warning(
        "chat template not recognized by llama.cpp's built-in matcher (%s) - "
        "falling back to a generic ChatML prompt this model may not "
        "understand; chat and vision output quality may be degraded", reason)


def _apply_model_template(model_ptr: int, messages: List[Dict]) -> Tuple[str, Optional[str]]:
    """
    Format *messages* using the model's own embedded Jinja chat template.

    Falls back to :func:`_format_chatml` if:
    * The model has no embedded template (``llama_model_chat_template`` returns None).
    * The template call fails for any reason.

    Returns ``(prompt, fallback_reason)``. *fallback_reason* is ``None`` on a
    normal templated render, or the reason ChatML was substituted instead
    (already passed to :func:`_warn_chatml_fallback` for the debug log). This
    function runs inside the isolated worker process and must never
    console.print (see check_hygiene.py's child-process list) - a caller that
    reaches a real generation request is responsible for propagating the
    reason to a channel visible without ``--debug`` once it is back in the
    parent process; see GgufBackend's ``_chatml_fallback`` latch in gguf.py.
    """
    tmpl_str = api.llama_model_chat_template(model_ptr)
    if not tmpl_str:
        reason = "model has no embedded chat template"
        _warn_chatml_fallback(reason)
        return _format_chatml(messages), reason

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
        reason = "embedded template not recognized/rendered nothing"
        _warn_chatml_fallback(reason)
        return _format_chatml(messages), reason

    if needed > buf_size:
        # Reallocate and retry
        buf = ctypes.create_string_buffer(needed + 64)
        needed = api.llama_chat_apply_template(tmpl_bytes, chat_arr, n, True, buf, len(buf))
        if needed <= 0:
            # Same guard as above: a failed or empty render falls back
            reason = "embedded template not recognized/rendered nothing"
            _warn_chatml_fallback(reason)
            return _format_chatml(messages), reason

    return buf.raw[:needed].decode("utf-8", errors="replace"), None


# UTF-8-safe token-bytes -> text stream: a multibyte character is often emitted
# across two or more tokens, so its bytes straddle a token boundary. Decoding each
# token in isolation yields U+FFFD at the split (mid-word mojibake); an
# incremental decoder buffers an incomplete trailing sequence until the next
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
from localm.textnorm import scrub_stream as _scrub_stream  # noqa: E402


# Suffix tokens are prefilled in chunks of this size (matches n_batch ceiling)
_PREFILL_CHUNK = 2048

# Coarse decode-progress heartbeat: every N generated tokens, never per token
# (a per-token line would flood the shared debug log and the bug-report
# digest's benign-record budget - see debuglog.py's ring-buffer docstring).
# Logged at DEBUG, not INFO - unlike the boundary markers (prefill start/
# complete, decode entered, complete/aborted), this one recurs every N tokens
# for the life of a generation, and the always-on ring buffer is a fixed 400
# records shared with everything else the server logs; an INFO line here
# would be spent forever, evicting unrelated diagnostics. It still reaches
# the shared debug-log file once --debug is on. 50 gives several checkpoints
# even on a short reply while keeping a stalled or crashed generation
# localized to within ~50 tokens of decode time.
_DECODE_PROGRESS_INTERVAL = 50


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
        triggers are given, the request is REFUSED with
        :class:`GrammarUnsupportedError` - a lazy request must never silently
        become a strict constraint (a strict grammar stalls thinking models),
        and it must never silently become NO constraint either (the caller is
        told the reply matches a grammar it was never sampled against).
    """
    from localm.inference.backends.base import (
        GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE,
        GRAMMAR_LAZY_UNSUPPORTED_MESSAGE,
        GrammarUnsupportedError,
        InvalidGrammarError,
    )

    chain_params = api.llama_sampler_chain_default_params()
    chain_params.no_perf = True
    chain = api.llama_sampler_chain_init(chain_params)

    # Grammar sampler masks logits before any scoring stage touches them.
    # llama_sampler_init_grammar[_lazy_patterns] returns NULL (ctypes -> None) when
    # the native GBNF parser rejects the grammar. Adding that NULL to the chain
    # NULL-derefs at sample time (a native access violation): the GGUF backend
    # CATCHES that fault and latches _grammar_unsupported, silently stripping
    # grammar from EVERY later request (valid ones too) until reload - one bad
    # grammar poisoned the whole feature for all clients. So check the return and
    # raise a typed error the request path can turn into a clean 400, instead of
    # letting a malformed grammar reach the crash-and-latch path.
    if grammar and grammar_lazy:
        if grammar_triggers and api.has_lazy_grammar():
            gsampler = api.llama_sampler_init_grammar_lazy_patterns(
                vocab, grammar.encode(), b"root",
                [t.encode() for t in grammar_triggers],
            )
            if gsampler is None:
                api.llama_sampler_free(chain)
                raise InvalidGrammarError(_INVALID_GRAMMAR_MSG)
            api.llama_sampler_chain_add(chain, gsampler)
        else:
            # REFUSE, do not drop. Dropping the grammar here would answer the
            # request with a normal 200 of unconstrained text that the caller had
            # every reason to believe was grammar-conformant - and the coder acts
            # on that reply by parsing it for tool calls. A typed refusal costs
            # the caller one clean 400 naming what to do instead
            # (http_server.py's _BACKEND_ERROR_STATUS maps
            # GrammarUnsupportedError -> 400; _runner.py carries the type across
            # the worker IPC as a tagged envelope so it does not degrade into the
            # worker-faulted RuntimeError that would evict the loaded model).
            #
            # Two DISTINCT messages because the two recoveries are opposite: the
            # caller fixes a missing trigger list by sending one, and can only fix
            # a build without the native export by dropping grammar_lazy. Reusing
            # one string here would also mis-latch the coder's session-wide
            # lazy-grammar disable - see GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE.
            #
            # `not grammar_triggers` is checked FIRST, mirroring the short-circuit
            # in the `if` above: with no triggers, has_lazy_grammar() was never
            # called, so this branch does not know whether the export exists and
            # must not imply one. A caller who is BOTH missing triggers AND on an
            # old build therefore learns it in two steps.
            #
            # Free the chain first, exactly like the two InvalidGrammarError arms:
            # nothing else owns it yet, so raising past it would leak the native
            # allocation on every refusal.
            api.llama_sampler_free(chain)
            raise GrammarUnsupportedError(
                GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE if not grammar_triggers
                else GRAMMAR_LAZY_UNSUPPORTED_MESSAGE)
    elif grammar:
        gsampler = api.llama_sampler_init_grammar(vocab, grammar.encode(), b"root")
        if gsampler is None:
            api.llama_sampler_free(chain)
            raise InvalidGrammarError(_INVALID_GRAMMAR_MSG)
        api.llama_sampler_chain_add(chain, gsampler)

    # Repetition penalty applies to greedy and stochastic sampling alike.
    # Newer builds take the vocabulary size as a leading argument; _api dispatches
    # on the build, but it needs the real n_vocab to pass - a 0 there would
    # under-allocate the sampler's frequency counters.
    if repeat_penalty and repeat_penalty != 1.0 and api.has_penalties_sampler():
        n_vocab = api.llama_vocab_n_tokens(vocab) if vocab else 0
        if not n_vocab and api.penalties_needs_n_vocab():
            from localm.debuglog import logger
            logger.warning(
                "skipping the repetition-penalty sampler: this llama build "
                "needs the vocabulary size and no vocab pointer was available.")
        else:
            api.llama_sampler_chain_add(
                chain,
                api.llama_sampler_init_penalties(
                    64, repeat_penalty, 0.0, 0.0, n_vocab=n_vocab),
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


# The exact, user-facing text for each way _apply_cpu_moe can decline to
# apply the override - a SINGLE source of truth read by both the (debug-log)
# caller inside the isolated child and the PARENT, which is the only process
# allowed to render it. Keyed by the same short reason string _apply_cpu_moe
# returns, so the parent-side renderer cannot drift from the child-side log line
# describing the identical fact.
MOE_SKIP_MESSAGES = {
    "no_experts": (
        "[yellow]  n_cpu_moe:[/yellow] this model has no experts (it is not a "
        "Mixture-of-Experts model), so the setting does nothing here. Loading "
        "normally."),
    "buffer_unresolved": (
        "[yellow]  n_cpu_moe:[/yellow] the CPU buffer type could not be "
        "resolved from this llama runtime, so MoE experts were NOT moved to "
        "system RAM. Loading normally instead."),
}


def _apply_cpu_moe(mp, n_layers: int, model_path: str):
    """Keep the first *n_layers* layers' EXPERT weights in system RAM.

    Points ``mp.tensor_buft_overrides`` at a NULL-terminated override array and
    returns ``(keepalive, skip_reason)``: *keepalive* is the ``(array,
    patterns)`` pair that must stay alive across
    ``llama_load_model_from_file`` (ctypes does not own those strings), or
    ``None`` if the override could not be applied; *skip_reason* is ``None``
    on success, or a key into ``MOE_SKIP_MESSAGES`` naming why not.

    Runs inside the ISOLATED WORKER CHILD (this whole class is loaded only
    there - see the module docstring). A child must NEVER ``console.print``:
    its stdout is not the server's own console, and a call here garbles the
    parent's Rich spinner output mid-line. *This function only reports the
    FACT* - via ``skip_reason``, carried out through
    ``LlamaCpp.moe_skip_reason`` and ``GgufWorker.load()``'s returned metadata,
    the same channel this feature uses for the placement report (see
    ``GgufBackend._load_native()``, which renders
    ``MOE_SKIP_MESSAGES[skip_reason]`` from the PARENT). It still logs locally
    (``_dbg.info``/``.warning``) for the debug log.

    A skip is LOUD, never silent: the user asked for a specific placement, and
    quietly loading with a DIFFERENT one would report success for something that
    did not happen. The load still proceeds, because a normal load is a working
    load - the user just has to be TOLD it happened, and only the parent can
    tell them without garbling its own output."""
    from ._loader import cpu_buffer_type
    from localm.debuglog import logger as _dbg
    # The FUSED per-layer expert weights, as llama.cpp's converters name them:
    # blk.<i>.ffn_gate_exps / ffn_down_exps / ffn_up_exps. Only these move. The
    # router (ffn_gate_inp) and any SHARED expert stay wherever the layer
    # assignment put them: they are read for EVERY token and they are tiny, so
    # moving them to system RAM would cost per-token bandwidth for almost no
    # VRAM back.
    #
    # SINGLE SOURCE OF TRUTH, imported rather than redefined here: the VRAM
    # preflight (llamacpp/_sizing.py, via model_manager.gguf.gguf_moe_pinned_
    # expert_bytes) needs to know EXACTLY which tensors this pins, before the
    # load, to charge them correctly - a second, independently-maintained copy
    # of this pattern could silently disagree with what actually gets pinned
    # here.
    # A dense model has no expert tensors, so every pattern below would match
    # nothing and the setting would silently do nothing. Say so instead: a
    # control that appears to apply but cannot is a silent no-op.
    from localm.model_manager.gguf import (
        _MOE_TENSOR_PREFIX, _MOE_TENSOR_SUFFIX, gguf_expert_count)
    from pathlib import Path as _Path
    if gguf_expert_count(_Path(model_path)) == 0:
        _dbg.info("n_cpu_moe=%d ignored: %s reports no experts",
                  n_layers, model_path)
        return None, "no_experts"

    buft = cpu_buffer_type()
    if not buft:
        _dbg.warning("n_cpu_moe=%d requested but cpu_buffer_type() returned None; "
                     "loading without a tensor placement override", n_layers)
        return None, "buffer_unresolved"

    patterns = [(_MOE_TENSOR_PREFIX + str(i) + _MOE_TENSOR_SUFFIX).encode("ascii")
                for i in range(n_layers)]
    array = (LlamaModelTensorBuftOverride * (len(patterns) + 1))()
    for i, pattern in enumerate(patterns):
        array[i].pattern = pattern
        array[i].buft = buft
    # NULL-pattern sentinel: how the native side finds the end of the array.
    array[len(patterns)].pattern = None
    array[len(patterns)].buft = None
    mp.tensor_buft_overrides = ctypes.cast(array, ctypes.c_void_p)
    _dbg.info("n_cpu_moe=%d: expert weights of layers 0-%d pinned to system RAM",
              n_layers, n_layers - 1)
    return (array, patterns), None


class LlamaCpp:
    """
    In-process GGUF inference backed by the native llama.dll.

    This is a drop-in replacement for ``llama_cpp.Llama`` for the subset of
    the API used by :class:`~localm.inference.backends.gguf.GgufBackend`.
    """

    # Class-level so it exists on instances built with object.__new__ (the test
    # helpers) as well as through __init__. Cleared the first time a rejected
    # draft's KV cell cannot be removed; speculation needs that rewind, so it
    # stays off for the rest of the model's life.
    _mtp_usable = True
    _mtp_ctx_capacity = 0        # the draft context's own n_ctx, 0 until created
    _mtp_wants_h = False         # True once both contexts expose the next-n state
    mtp_active_this_call = False # whether THIS generation actually speculated
    _pending_h = None            # the hidden state the next draft will read
    _h_buf = None                # reusable copy target for it
    _n_embd = 0

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
        vram_check: Optional[Callable[[int, int], Optional[bool]]] = None,
        gpu_split_ratios: Optional[list] = None,
        n_cpu_moe: int = 0,
        mtp_enabled: bool = False,
        **_ignored,
    ) -> None:
        self._n_ctx       = n_ctx
        self._mtp_enabled = mtp_enabled
        # Optional preflight consulted by _prefill_fresh_context() before
        # (re)creating a BIGGER context (conversation growth, not just the
        # initial load already guarded by the caller's own preflight). Called
        # with (target_n_ctx, current_ctx_capacity); returns True to keep the KV
        # cache in VRAM, False to place it in system RAM (a degrade, never an
        # abort), or None when VRAM is unmeasurable (keep the default). None (the
        # attribute) = no check (only a NULL-pointer check on the result after).
        self._vram_check  = vram_check
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
        self._mtp_ctx_ptr = None   # Multi-Token Prediction draft context
        self.supports_mtp = False  # True when MTP heads and draft context are active
        self.mtp_status   = "not-initialised"  # short token: why MTP is or is not active
        self._mtp_usable = True
        self._mtp_wants_h = False   # True once both contexts expose the next-n state
        self._pending_h   = None    # the hidden state the next draft will read
        self._h_buf       = None    # reusable copy target for it
        self._n_embd      = 0
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
        # Where the live context's KV cache actually lives: True = VRAM (offload_kqv),
        # False = system RAM (a prior grow found VRAM too tight). The initial context
        # below is created with offload_kqv=True. _prefill_fresh_context updates this,
        # and GgufBackend._check_context_fit reads it so a further grow charges the KV
        # correctly (full target vs net delta) - otherwise a RAM-resident KV would be
        # under-charged and wrongly flipped back to VRAM, overflowing and aborting.
        self._offload_kqv   = True
        self._kv_supported: Optional[bool] = None   # lazy llama_memory_* probe
        # Non-None once _apply_model_template has had to substitute a generic
        # ChatML prompt because this model's own embedded template could not
        # be used (RAG-VISION-1). Sticky for the life of this instance - the
        # underlying template never changes for a loaded model. Read by
        # GgufWorker (via the "done" envelope) so the PARENT process can
        # surface the degrade once, outside --debug (see gguf.py).
        self.chat_template_fallback_reason: Optional[str] = None

        # --- load model ---
        mp = api.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        if hasattr(mp, "load_mtp") and mtp_enabled:
            mp.load_mtp = True
        if n_gpu_layers >= 99:
            # Newer builds replaced use_mmap/use_mlock/use_direct_io with a
            # single load_mode enum at a DIFFERENT offset; set_use_mmap writes
            # whichever this build has. Assigning mp.use_mmap directly would
            # land in check_tensors on those builds - same size, no error.
            set_use_mmap(mp, False)
        # Multi-GPU: honour the configured main_gpu_index (validated against
        # the devices actually visible right now); leaves the native default
        # (device 0) untouched when unset. See discover.apply_main_gpu.
        from localm.discover import apply_gpu_split, apply_main_gpu
        apply_main_gpu(mp)
        # Multi-GPU tensor-split: spreads the model across 2+ configured
        # devices when gpu_split_indices is set (see discover.apply_gpu_split).
        # gpu_split_ratios carries the PARENT's already-resolved effective
        # ratios (auto free-VRAM-proportional distribution,
        # discover.resolve_auto_split_ratios) into this isolated worker, which
        # must not probe for them itself - see that function's docstring.
        # The returned buffer must stay alive through llama_load_model_from_file
        # below - it is read once at load time, not held as a live pointer.
        _tensor_split_keepalive = apply_gpu_split(
            mp, ratios_override=gpu_split_ratios)
        # Read AFTER apply_gpu_split, which is what makes this value meaningful:
        # it forces main_gpu inside the configured split set (substituting the
        # first split device, with a warning, when the configured main_gpu_index
        # is not one of them). So this is the load's RESOLVED primary device, not
        # the raw config value. The vision projector follows it (below) so it does
        # not land on a card the user's split excludes.
        self._main_gpu_index = int(mp.main_gpu)

        # MoE expert placement (opt-in, n_cpu_moe > 0): keep the EXPERT weights of
        # the first N layers in system RAM while everything else follows the normal
        # layer assignment. This is llama.cpp's own --n-cpu-moe, driven through
        # llama_model_params.tensor_buft_overrides.
        #
        # It buys VRAM FOOTPRINT, not throughput. On a 64-expert/8-active MoE at
        # MATCHED VRAM it is throughput-neutral, but it reaches a given speed in
        # far less VRAM. Sparsity already applies under layer offload too (a
        # CPU-resident layer's experts are still only 8-of-64 read per token), so
        # the throughput is a wash and this stays default-off: a footprint dial,
        # not a free speed-up.
        #
        # The array must stay alive across llama_load_model_from_file - it is read
        # at load time, exactly like tensor_split above - and every `pattern` bytes
        # object with it, hence keeping the list, not just the array.
        self._moe_override_keepalive = None
        # Why the override did not apply (a key into MOE_SKIP_MESSAGES), or
        # None on success / when n_cpu_moe was never requested - carried out
        # to the parent (GgufWorker.load()'s returned metadata), which is the
        # only process allowed to render it. See _apply_cpu_moe's own
        # docstring for why this cannot be a console.print here. The CALL
        # itself moved below, inside the merged native-call scope - see that
        # scope's own comment for why.
        self.moe_skip_reason: Optional[str] = None

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
            # _progress is DISCARDED: it looks like a ready-made load percentage
            # and it is not one. The VALUE is well-behaved - 0.0 -> 1.0, strictly
            # increasing - but the TIMING is unusable, because the count is
            # per-TENSOR, not per-unit-time: the same number of calls fires for a
            # 0.5B and a 7B regardless of whether the load took one second or
            # thirteen, and most of a load elapses BEFORE the first call. Rendering
            # it as a bar would sit dead for most of the wait and then flash
            # 0->100%, claiming what it does not know. If load progress is wanted,
            # report a PHASE over _runner.py's non-terminal "progress" envelope; a
            # percentage is only meaningful INSIDE the tensor-upload phase and must
            # be labelled as such.
            @ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)
            def _load_progress(_progress, _user_data, _ev=cancel_event):
                try:
                    return not _ev.is_set()
                except Exception:
                    return True
            self._load_progress_cb = _load_progress
            mp.progress_callback = ctypes.cast(_load_progress, ctypes.c_void_p)

        # ONE CONTIGUOUS scope for the whole native-call span, from backend init
        # through the model load. A separate _quiet_stderr() scope around
        # llama_backend_init() would leave a gap over this span's Python-only
        # setup (GPU split/main_gpu, and _apply_cpu_moe) where fd 2 is back to
        # whatever the CHILD inherited from the PARENT at spawn, i.e. the SAME
        # terminal the parent's Rich load spinner renders to. With one scope,
        # nothing between llama_backend_init() and llama_load_model_from_file()
        # can reach fd 2 unredirected. Including the Python-only setup calls is
        # safe - none of them writes to fd 2 (apply_main_gpu/apply_gpu_split's
        # only native call, llama_max_devices(), is a compile-time-constant
        # getter, not a device probe).
        #
        # THAT ALONE IS NOT ENOUGH, so suppress_console_mirror() is paired with
        # it below. _apply_cpu_moe's own _dbg.info/.warning calls go through
        # Python's logging module, not fd 2, and in debug mode debuglog.py's
        # console mirror is BY DESIGN immune to this exact fd-2 redirect (see
        # _stable_console_stream), so it writes straight to the terminal from
        # this child process, invisible to the parent's Rich Live region,
        # desyncing its cursor bookkeeping and stranding an orphaned spinner
        # frame on screen. Widening this redirect scope can never cover that:
        # the mirror survives an fd-2 redirect by design, so it needs its OWN,
        # separate gate every time this scope is touched.
        #
        # Capture (not just quiet) for the whole span, non-verbose only, so a
        # NULL return still carries its cause (OOM / no-backends / bad-quant),
        # and a successful return still carries llama.cpp's own load_tensors
        # report of where each backend's share of the weights actually landed
        # (the only source for that - see _MODEL_BUFFER_RE). Both reads happen
        # INSIDE the ``with`` block - _capture_stderr unlinks its temp file the
        # moment the block exits, so reading after exit silently returns ""
        # / [] (see that function's docstring). Verbose mode leaves this
        # untouched (nullcontext, captured=None; mirror also left alone): the
        # native stream already reaches terminal/debug directly, and there is
        # nothing captured to parse.
        from localm.debuglog import suppress_console_mirror
        _load_ctx = _capture_stderr if not verbose else contextlib.nullcontext
        _mirror_ctx = suppress_console_mirror if not verbose else contextlib.nullcontext
        self.weight_placement: list = []
        _load_failure_detail = ""
        with _mirror_ctx(), _load_ctx() as captured:
            api.llama_backend_init()
            if n_cpu_moe > 0:
                self._moe_override_keepalive, self.moe_skip_reason = _apply_cpu_moe(
                    mp, n_cpu_moe, model_path)
            self._model_ptr = api.llama_load_model_from_file(model_path, mp)
            if captured is not None:
                if self._model_ptr:
                    self.weight_placement = captured.model_buffers()
                else:
                    _load_failure_detail = captured.tail()
        if not self._model_ptr:
            # A NULL return when we asked to cancel is an ABORT, not a failure:
            # the load was superseded by a newer model selection. Report it as
            # such so the caller does not surface it as a load error.
            if cancel_event is not None and cancel_event.is_set():
                from localm.inference.backends.base import ModelLoadCancelled
                raise ModelLoadCancelled(
                    f"Model load aborted (superseded): {model_path}")
            hint = ("" if _load_failure_detail
                    else " (run with LOCALM_DEBUG=1 for the native load log)")
            suffix = f"\n{_load_failure_detail}" if _load_failure_detail else ""
            raise RuntimeError(
                f"Failed to load model: {model_path}{hint}{suffix}")

        # Model's true transformer layer count, read once here from the loaded
        # model. This is the only place it is currently EXPOSED, which is NOT the
        # same as the only place it is knowable: model_manager/gguf.py parses the
        # header before any load, and gguf_kv_bytes_per_token already reads
        # <arch>.block_count off it, consuming the value for KV arithmetic instead
        # of surfacing it. The GGUF backend caches this one (localm.model_meta) so
        # later loads and the GUI VRAM estimate can size a partial GPU offload
        # precisely; the clamp note below reuses it.
        self.n_layers: Optional[int] = None
        try:
            actual = api.llama_model_n_layer(self._model_ptr)
            if actual and actual > 0:
                self.n_layers = int(actual)
        except Exception:
            pass  # introspection is best-effort; never block a successful load

        # Architecture-accurate KV-cache size PER TOKEN, in bytes, read once here
        # from the model's attention shape (see _read_kv_bytes_per_token). This is
        # what a full-context KV cache costs per token in VRAM; the grow-time
        # decision (GgufBackend._check_context_fit, which reads this attribute) uses
        # it instead of a file-size heuristic that under-counted wide-KV models by
        # ~2.6x.
        self.kv_bytes_per_token: int = self._read_kv_bytes_per_token()

        # llama.cpp already offloads min(n_gpu_layers, actual), so an over-large
        # value is harmless - but silently clamping a SPECIFIC number is
        # confusing, so surface a message. 99 = "offload all", so skip it.
        if 0 < n_gpu_layers < 99 and self.n_layers and n_gpu_layers > self.n_layers:
            from localm.debuglog import logger
            logger.info(
                "n_gpu_layers=%d exceeds the model's %d layers; "
                "offloading all %d (the extra has no effect)",
                n_gpu_layers, self.n_layers, self.n_layers)

        # --- create context ---
        cp = api.llama_context_default_params()
        cp.n_ctx             = n_ctx
        cp.n_batch           = min(n_ctx, 2048)
        cp.n_ubatch          = cp.n_batch   # match micro-batch to batch
        cp.offload_kqv       = True
        # Speculation writes a draft token into the cache and takes it back out
        # when the target rejects it. A recurrent cache cannot be truncated at
        # all UNLESS it is keeping per-token state snapshots, and it keeps none
        # by default, so a rejected draft leaves the sequence unrewindable and
        # every later batch is refused. One snapshot covers a one-token draft;
        # two leaves room. Costs nothing on a model with no recurrent layers.
        # See test_recurrent_rollback_is_requested_when_mtp_is_enabled.
        if self._mtp_enabled and hasattr(cp, "n_rs_seq"):
            cp.n_rs_seq = max(int(getattr(cp, "n_rs_seq", 0) or 0), 2)
        cp.flash_attn_type   = -1  # keep default (unspecified)
        if n_threads is not None:
            cp.n_threads       = n_threads
            cp.n_threads_batch = n_threads

        # A separate, later native call than the merged load scope above (it
        # needs the just-loaded self._model_ptr) - kept on plain _quiet_stderr,
        # not the mirror-gated/capture scope: nothing runs here that logs via
        # the isolated child's console mirror, so there is no equivalent gap
        # to close, and there is nothing to capture/parse from this call.
        _ctx = _quiet_stderr if not verbose else contextlib.nullcontext
        with _ctx():
            self._ctx_ptr = api.llama_init_from_model(self._model_ptr, cp)
        if not self._ctx_ptr:
            api.llama_free_model(self._model_ptr)
            raise RuntimeError("Failed to create llama context")

        # Multi-Token Prediction (MTP) draft context initialization
        if not self._mtp_enabled:
            self.mtp_status = "disabled"
        else:
            try:
                eligible, self.mtp_status = api.llama_model_mtp_support(self._model_ptr)
                if eligible and not self._cache_can_drop_a_speculative_token():
                    # Ask before allocating a draft context this model can never
                    # use: speculation needs to take a rejected token back out.
                    self.mtp_status = "rewind-unsupported"
                    self._mtp_usable = False
                    eligible = False
                if eligible and not api.mtp_hidden_state_available():
                    # Without this the draft head reads only the token embedding,
                    # and its drafts cost more per token than they save. Refusing
                    # is the better answer than drafting badly.
                    self.mtp_status = "no-hidden-state-api"
                    eligible = False
                cp_mtp = api.llama_context_default_params() if eligible else None
                if cp_mtp is not None and not hasattr(cp_mtp, "ctx_type"):
                    # Without ctx_type this build cannot be ASKED for an MTP
                    # context, so llama_init_from_model would hand back a second
                    # ordinary decoder with its own uncharged KV cache.
                    self.mtp_status = "no-ctx-type-field"
                    cp_mtp = None
                if cp_mtp is not None:
                    from ._structs import LLAMA_CONTEXT_TYPE_MTP
                    cp_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP
                    cp_mtp.n_ctx = min(n_ctx, 2048)
                    cp_mtp.n_batch = cp_mtp.n_ctx
                    cp_mtp.n_ubatch = cp_mtp.n_batch
                    cp_mtp.offload_kqv = True
                    if n_threads is not None:
                        cp_mtp.n_threads = n_threads
                        cp_mtp.n_threads_batch = n_threads
                    with _ctx():
                        self._mtp_ctx_ptr = api.llama_init_from_model(self._model_ptr, cp_mtp)
                    if self._mtp_ctx_ptr:
                        self._mtp_ctx_capacity = cp_mtp.n_ctx
                        self._n_embd = api.llama_model_n_embd(self._model_ptr)
                        # The target exposes its hidden state; the draft consumes
                        # it masked. Both must take, or the head is starved and
                        # drafting is worse than not drafting.
                        exposed = api.llama_set_embeddings_nextn(self._ctx_ptr, True, False)
                        consumed = api.llama_set_embeddings_nextn(self._mtp_ctx_ptr, True, True)
                        self._mtp_wants_h = bool(exposed and consumed and self._n_embd > 0)
                        if not self._mtp_wants_h:
                            api.llama_free(self._mtp_ctx_ptr)
                            self._mtp_ctx_ptr = None
                            self.mtp_status = "hidden-state-refused"
                        else:
                            self.supports_mtp = True
                    else:
                        self.mtp_status = "context-refused"
            except Exception as exc:
                self._mtp_ctx_ptr = None
                self.supports_mtp = False
                self.mtp_status = f"error:{type(exc).__name__}"
        from localm.debuglog import logger as _mtp_log
        _mtp_log.info("MTP: active=%s status=%s", self.supports_mtp, self.mtp_status)

        self._tokenizer = _Tokenizer(self._model_ptr, self._ctx_ptr)

        # Optional in-process vision (C1): load the mmproj via mtmd so image
        # messages can be answered. Best-effort - any failure (no mtmd.dll, an
        # incompatible mmproj) leaves the model text-only rather than breaking it.
        if mmproj_path:
            self._load_mmproj(mmproj_path, verbose)

    def _load_mmproj(self, mmproj_path: str, verbose: bool) -> None:
        """Load *mmproj_path* via mtmd and set self._mtmd, or leave it None on
        any failure. Pulled out of __init__ (same reasoning as
        _stderr_ctx_for_generate above) so the wrap is directly unit-testable
        without a real native model.

        Wrapped in the SAME mirror+capture scope as the main model load in
        __init__: MtmdContext.__init__ makes several native calls of its own
        (mtmd_init_from_file for the CLIP/vision projector, then the
        mtmd_tokenize-based ABI probe in _detect_input_text_class) and none of
        them were ever redirected, so the projector's tensor-by-tensor load
        report and the ABI probe's raw text payload both landed on the real
        console unfiltered. _capture_stderr also means a failure still carries
        its native reason instead of losing it: MtmdContext.__init__ RAISES
        rather than returning NULL (unlike the main model load in __init__), so
        the detail is grabbed INSIDE the ``with`` block, before its temp file
        is unlinked on exit (see _capture_stderr's own docstring)."""
        from localm.debuglog import suppress_console_mirror
        _mtmd_load_ctx = _capture_stderr if not verbose else contextlib.nullcontext
        _mtmd_mirror_ctx = suppress_console_mirror if not verbose else contextlib.nullcontext
        _mtmd_detail = ""
        try:
            with _mtmd_mirror_ctx(), _mtmd_load_ctx() as captured:
                try:
                    from .mtmd import MtmdContext
                    # getattr, not self._main_gpu_index: this method is unit-tested
                    # directly against instances that never ran __init__ (see this
                    # docstring's note on why it was pulled out), and 0 is exactly
                    # the "leave clip's own default alone" value.
                    mt = MtmdContext(
                        mmproj_path, self._model_ptr,
                        gpu_index=getattr(self, "_main_gpu_index", 0))
                except Exception:
                    if captured is not None:
                        _mtmd_detail = captured.tail()
                    raise
            if mt.supports_vision:
                self._mtmd = mt
            else:
                mt.free()
        except Exception as exc:
            from localm.debuglog import logger
            suffix = f"\n{_mtmd_detail}" if _mtmd_detail else ""
            # WARNING, not debug: a vision model that silently drops to
            # text-only is a real capability loss the user asked for and did not
            # get - it must reach a level they will actually see, not only
            # LOCALM_DEBUG=1.
            logger.warning(
                "mmproj load failed (%s); model stays text-only%s", exc, suffix)
            self._mtmd = None

    def _read_kv_bytes_per_token(self) -> int:
        """Architecture-accurate KV-cache size PER TOKEN in bytes, from the loaded
        model's attention shape, or 0 when it cannot be determined (a stripped DLL
        without the head accessors, or unreadable metadata) so callers fall back to
        the size-class estimate.

        K and V cache = n_layers x n_head_kv x head_dim, times 2 (K and V) and
        times 2 bytes/element (the f16 KV cache, llama.cpp's default type_k/type_v).
        head_dim = n_embd / n_head. n_head_kv (fewer than n_head under grouped-query
        attention) is exactly why the true KV cost is smaller than a naive n_head
        estimate - and why estimating it from file size alone is unreliable.

        REFUSES on a hybrid/recurrent stack. That formula assumes every layer holds
        a KV cache, which is false for Qwen3-Next, Granite 4 H, LFM2, Jamba and the
        rest, where most layers keep a FIXED-size recurrent state instead and cost
        no per-token KV at all. The exported llama_model_n_head_kv reports LAYER 0
        only (upstream n_head_kv() defaults il=0), so there is nothing here to sum
        over - and answering anyway over-charges by the ratio of attending layers
        to all layers. Returning 0 hands the question to the caller's next source,
        the GGUF header probe, which CAN read the exact per-layer array."""
        try:
            if self.n_layers and api.has_kv_head_api():
                if api.has_hybrid_api() and (
                        api.llama_model_is_hybrid(self._model_ptr)
                        or api.llama_model_is_recurrent(self._model_ptr)):
                    return 0
                n_embd    = int(api.llama_model_n_embd(self._model_ptr))
                n_head    = int(api.llama_model_n_head(self._model_ptr))
                n_head_kv = int(api.llama_model_n_head_kv(self._model_ptr))
                if n_embd > 0 and n_head > 0 and n_head_kv > 0:
                    head_dim = n_embd // n_head
                    return self.n_layers * n_head_kv * head_dim * 2 * 2
        except Exception as exc:
            # A genuine failure here (NOT the expected has_kv_head_api-False path,
            # which skips the block and returns 0 cleanly) silently drops back to
            # the under-counting size heuristic - which can re-enable the very
            # Vulkan crash this figure exists to avoid. Logged so that regression
            # is discoverable rather than invisible: surface, then degrade.
            from localm.debuglog import logger as _dbg
            _dbg.debug("kv_bytes_per_token computation failed (%s); falling back to "
                       "the size-class estimate", type(exc).__name__)
        return 0

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
        if getattr(self, "_mtp_ctx_ptr", None) is not None:
            try:
                api.llama_free(self._mtp_ctx_ptr)
            except Exception:
                pass
            self._mtp_ctx_ptr = None
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
        # character that straddles a token boundary is not split into U+FFFD.
        # Per-token decode would mangle exactly those boundaries.
        raw = b"".join(self._tokenizer.token_to_piece_bytes(t) for t in tokens)
        return raw.decode("utf-8", errors="replace")

    def check_grammar(self, grammar: str) -> None:
        """Raise :class:`InvalidGrammarError` if *grammar* is not a parseable GBNF
        string, WITHOUT running any generation. A cheap native parse:
        ``llama_sampler_init_grammar`` returns NULL for a malformed grammar. Lets
        the request path reject a bad grammar with a clean 400 up front instead of
        letting it reach the sample-time NULL-deref (which the GGUF backend catches
        by latching the silent _grammar_unsupported degrade). No-op for an empty
        grammar or when the model is not loaded (no vocab to parse against)."""
        from localm.inference.backends.base import InvalidGrammarError

        if not grammar or not self._model_ptr:
            return
        # The native parser prints "failed to parse grammar" to stderr on rejection;
        # keep that off the terminal (it still lands in the debug log via _quiet_stderr).
        with _quiet_stderr():
            sampler = api.llama_sampler_init_grammar(
                self._tokenizer._vocab, grammar.encode(), b"root")
        if sampler is None:
            raise InvalidGrammarError(_INVALID_GRAMMAR_MSG)
        api.llama_sampler_free(sampler)

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

            # grammar/grammar_lazy no longer pick a different stderr context here -
            # see _stderr_ctx_for_generate's docstring for why the grammar path
            # was folded into the same one as plain generation.
            _ctx = _stderr_ctx_for_generate(self._verbose)

            # If unlimited (<= 0), allocate a modest chunk up front and grow later
            initial_budget = max_new_tokens if max_new_tokens > 0 else 512
            needed = n_prompt + initial_budget + 64

            # BOUNDARY LOGGING: between "model loaded" and either a token or a
            # corpse, this worker would otherwise emit nothing at any level, so a
            # native crash mid-generation could not be placed in prefill vs
            # decode. INFO, following discover.py's resolve_auto_split_ratios
            # precedent (the always-on ring buffer is INFO+, so a bug report shows
            # what was decided) - though here that only reaches a bug report once
            # --debug is on, since this method runs inside the isolated worker
            # process and the parent's ring buffer is process-local. Per-token
            # detail never lands here - see _DECODE_PROGRESS_INTERVAL.
            from localm.debuglog import logger
            logger.info("gguf generate: prefill starting, %d prompt token(s)", n_prompt)
            _t0 = time.monotonic()
            tokens_generated = 0
            in_decode = False
            # Both handles are freed in the finally below, which is reachable
            # before either is bound: the _stop check right after the lock, and
            # any raise out of prefill, both exit early. Binding one inside the
            # try loses the real error to an UnboundLocalError.
            sampler = None
            draft_sampler = None
            # Carries a rejected speculation's replacement token into the next
            # loop iteration; set only by the reject branch below.
            pending_token = None
            try:
                # One contiguous suppression scope covering context work and
                # prefill. The ROCm lazy-buffer verification messages fire
                # asynchronously after llama_init_from_model returns but before
                # the first llama_decode completes, so separate windows leave a
                # gap. Prefill (re)creates/decodes into the context, so it must
                # hold the lock against a concurrent unload too.
                with self._gen_lock:
                    if self._stop.is_set():
                        return
                    reuse = self._can_reuse_kv(needed)
                    with _ctx():
                        if reuse:
                            self._prefill_with_reuse(prompt_tokens)
                        else:
                            self._prefill_fresh_context(prompt_tokens, needed)

                logger.info("gguf generate: prefill complete in %.2fs (kv_reuse=%s)",
                            time.monotonic() - _t0, reuse)

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
                # Drafting proposes tokens greedily off the MTP context; the
                # request's own sampler below decides what is emitted. Constrained
                # requests take the single-token path instead: draft_sampler stays
                # None whenever a grammar is active, so no speculation ever runs
                # with a grammar sampler in the chain, where a mis-sequenced accept
                # throws across the C ABI.
                # See test_mtp_drafting_is_disabled_while_a_grammar_is_active.
                draft_sampler = (
                    api.llama_sampler_init_greedy()
                    if self._mtp_ctx_ptr is not None and grammar is None
                    and self._mtp_usable
                    else None
                )

                self.mtp_active_this_call = draft_sampler is not None
                pos = n_prompt
                # Why generation ended, read by callers as self.last_finish_reason.
                # Default "stop" - it must cover every early exit (EOG token, a
                # stop-string match in _filtered_stream abandoning this generator,
                # client abort). Only a genuinely exhausted token budget is "length".
                self.last_finish_reason = "stop"
                in_decode = True
                logger.info("gguf generate: entering decode loop")
                _decode_t0 = time.monotonic()
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
                            #
                            # A pending token was already sampled from *sampler*
                            # by the rejected speculation below, so re-sampling
                            # here would both discard it and read a logits row
                            # that no longer matches the KV cache.
                            if pending_token is not None:
                                token, pending_token = pending_token, None
                            else:
                                token = api.llama_sampler_sample(sampler, self._ctx_ptr, -1)
                            eog = self._tokenizer.is_eog(token)

                        # Stop when the model signals end-of-generation via the vocabulary
                        if eog:
                            break   # last_finish_reason stays "stop"

                        yield token   # consumer runs here; an unload can interleave
                        tokens_generated += 1

                        # Coarse heartbeat, OUTSIDE the lock above (never add
                        # work to a native-call-holding region). DEBUG, not
                        # INFO: the file-side ring-buffer precedent this whole
                        # scheme follows (discover.py) is explicit that INFO is
                        # for a decision made once per call, not a recurring
                        # tick - the always-on ring buffer holds 400 records
                        # SHARED across everything the server logs, and an
                        # INFO line here is spent on every generation forever.
                        # Only the phase BOUNDARIES (prefill start/complete,
                        # decode entered, complete/aborted - roughly four per
                        # generation) are affordable at that level; this one
                        # still reaches the shared debug-log file once --debug
                        # is on, which is where a stalled-vs-hung decode is
                        # actually diagnosed.
                        if (tokens_generated
                                and tokens_generated % _DECODE_PROGRESS_INTERVAL == 0):
                            logger.debug(
                                "gguf generate: decode progress, %d token(s) in %.2fs",
                                tokens_generated, time.monotonic() - _decode_t0)

                        if max_new_tokens > 0 and tokens_generated >= max_new_tokens:
                            # The while/else below only runs when the loop exits by
                            # CONDITION, and this break skips it, so last_finish_reason
                            # is set here too. Callers read it to tell a reply that ran
                            # out of budget from one the model chose to end.
                            self.last_finish_reason = "length"
                            # Final token budget reached, update KV cache bookkeeping
                            with self._gen_lock:
                                if not (self._stop.is_set() or self._ctx_ptr is None):
                                    batch = self._create_batch([token], pos, logits_at_last_only=True)
                                    try:
                                        api.llama_decode(self._ctx_ptr, batch)
                                        self._cached_tokens.append(token)
                                        pos += 1
                                    except Exception:
                                        pass
                                    finally:
                                        if batch is not None:
                                            api.llama_batch_free(batch)
                            break

                        # --- Speculative MTP drafting (if draft context is active) ---
                        draft_token = None
                        if (self._mtp_ctx_ptr is not None and draft_sampler is not None
                                and self._mtp_usable):
                            with self._gen_lock:
                                if not (self._stop.is_set() or self._ctx_ptr is None):
                                    try:
                                        if self._pending_h is not None:
                                            d_batch, d_orig, _hold = self._create_draft_batch(token, pos)
                                            try:
                                                d_ret = api.llama_decode(self._mtp_ctx_ptr, d_batch)
                                            finally:
                                                self._free_draft_batch(d_batch, d_orig)
                                            if d_ret == 0:
                                                draft_token = api.llama_sampler_sample(draft_sampler, self._mtp_ctx_ptr, -1)
                                    except Exception:
                                        draft_token = None

                        if draft_token is not None and not self._tokenizer.is_eog(draft_token):
                            # Multi-token verification on main context: decode [token, draft_token]
                            with self._gen_lock:
                                if self._stop.is_set() or self._ctx_ptr is None:
                                    self.last_finish_reason = "error"
                                    break
                                batch = self._create_batch([token, draft_token], pos, logits_at_last_only=False)
                                try:
                                    ret = api.llama_decode(self._ctx_ptr, batch)
                                    if ret == 0:
                                        # The target model's own continuation of
                                        # *token*, drawn through the REQUEST's
                                        # sampler (temperature, top_k, top_p and
                                        # the repetition window), from logits row
                                        # 0 - the row produced after *token*.
                                        # llama_sampler_sample accepts what it
                                        # returns, and verified_token is emitted
                                        # on both branches below, so the sampler
                                        # is never advanced past a token that was
                                        # not produced.
                                        verified_token = api.llama_sampler_sample(sampler, self._ctx_ptr, 0)
                                        if verified_token == draft_token:
                                            # Draft MATCHED / ACCEPTED! The sample
                                            # above already accepted it; a second
                                            # accept here is the double accept
                                            # that threw across the C ABI.
                                            # Keep MTP context KV cache in sync with accepted draft token
                                            self._capture_h(1)
                                            if self._mtp_ctx_ptr is not None and self._mtp_usable:
                                                try:
                                                    d_acc, a_orig, _ah = self._create_draft_batch(draft_token, pos + 1)
                                                    try:
                                                        api.llama_decode(self._mtp_ctx_ptr, d_acc)
                                                    finally:
                                                        self._free_draft_batch(d_acc, a_orig)
                                                except Exception:
                                                    pass

                                            self._cached_tokens.extend([token, draft_token])
                                            pos += 2
                                            yield draft_token
                                            tokens_generated += 1
                                            if self._tokenizer.is_eog(draft_token):
                                                break
                                            continue
                                        else:
                                            # Draft REJECTED: remove the speculative token slot at pos + 1
                                            removed = api.llama_kv_cache_seq_rm(self._ctx_ptr, 0, pos + 1, -1)
                                            if self._mtp_ctx_ptr is not None and self._mtp_usable:
                                                api.llama_kv_cache_seq_rm(self._mtp_ctx_ptr, 0, pos + 1, -1)
                                            self._capture_h(0)
                                            self._cached_tokens.append(token)
                                            pos += 1
                                            if not removed:
                                                # This memory module cannot drop the
                                                # rejected cell, so it still holds
                                                # position pos and llama.cpp refuses
                                                # every later batch as having
                                                # inconsistent sequence positions.
                                                # Rebuild from the tokens actually
                                                # emitted and stop speculating.
                                                # See test_a_stuck_draft_cell_disables_mtp_and_keeps_generating.
                                                self._mtp_usable = False
                                                self.supports_mtp = False
                                                self.mtp_status = "rewind-unsupported"
                                                from localm.debuglog import logger as _dbg_rewind
                                                _dbg_rewind.warning(
                                                    "MTP: this model's KV cache cannot drop a rejected "
                                                    "draft token; speculation disabled for this model")
                                                if not self._rebuild_kv_after_stuck_draft():
                                                    self.last_finish_reason = "error"
                                                    self._cached_tokens = []
                                                    break
                                            # verified_token is the target's own
                                            # continuation of *token* and is what
                                            # this position emits. Carry it to the
                                            # loop head rather than re-sampling:
                                            # the last logits row belongs to the
                                            # draft token just removed from the KV
                                            # cache.
                                            pending_token = verified_token
                                    else:
                                        # Decode failed, fall back to single token
                                        api.llama_batch_free(batch)
                                        batch = self._create_batch([token], pos, logits_at_last_only=True)
                                        ret = api.llama_decode(self._ctx_ptr, batch)
                                        if ret == 0:
                                            self._cached_tokens.append(token)
                                            pos += 1
                                        else:
                                            self.last_finish_reason = "error"
                                            self._cached_tokens = []
                                            break
                                finally:
                                    if batch is not None:
                                        api.llama_batch_free(batch)
                        else:
                            # --- locked native region 2: feed single token back ---
                            with self._gen_lock:
                                if self._stop.is_set() or self._ctx_ptr is None:
                                    self.last_finish_reason = "error"
                                    break
                                batch = self._create_batch([token], pos, logits_at_last_only=True)
                                try:
                                    ret = api.llama_decode(self._ctx_ptr, batch)
                                    if ret == 0:
                                        self._capture_h(-1)
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
                                finally:
                                    # Always release the native batch - including when
                                    # _prefill_fresh_context above raises mid-growth.
                                    if batch is not None:
                                        api.llama_batch_free(batch)
                    else:
                        # Budget exhausted without the model finishing its turn
                        self.last_finish_reason = "length"
                logger.info(
                    "gguf generate: complete, %d token(s) in %.2fs, finish_reason=%s",
                    tokens_generated, time.monotonic() - _decode_t0, self.last_finish_reason)
            except GeneratorExit:
                logger.info(
                    "gguf generate: aborted (cancelled) during %s, %d token(s) generated",
                    "decode" if in_decode else "prefill", tokens_generated)
                raise
            except Exception:
                logger.info(
                    "gguf generate: aborted (exception) during %s, %d token(s) generated",
                    "decode" if in_decode else "prefill", tokens_generated)
                raise
            finally:
                if sampler is not None:
                    api.llama_sampler_free(sampler)
                if draft_sampler is not None:
                    api.llama_sampler_free(draft_sampler)

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
        cache is invalidated afterwards.

        BOUNDARY LOGGING: same scheme as _generate (prefill start/complete,
        decode entered, complete/aborted with phase and token count). The
        "vision" tag on every line distinguishes it from _generate's in a shared
        debug log.

        Structural note: unlike _generate's single contiguous scope, this
        method's _ctx()/dedup_native_stderr usage is re-entered per native call
        - see the comment at that call site before restructuring it."""
        with self._inference_lock:
            if not self._model_ptr or getattr(self, "_mtmd", None) is None:
                raise RuntimeError("vision is not available on this model")

            from localm.debuglog import logger
            # An image turn never speculates: this loop has no draft context, and
            # upstream's own driver skips vision batches for the same reason - the
            # draft head reads its hidden state from a batch's embd slot and image
            # embeddings arrive in that same slot.
            self.mtp_active_this_call = False
            logger.info("gguf generate (vision): prefill starting")
            _t0 = time.monotonic()
            tokens_generated = 0
            in_decode = False
            sampler = None
            try:
                text_messages, images = self._messages_with_markers(
                    messages, self._mtmd.marker)
                prompt, fallback_reason = _apply_model_template(self._model_ptr, text_messages)
                if fallback_reason:
                    self.chat_template_fallback_reason = fallback_reason
                bos_markers = ("<bos>", "<s>", "﻿")
                add_special = not any(prompt.startswith(m) for m in bos_markers)

                # Stays on _quiet_stderr rather than _generate()'s
                # dedup_native_stderr: below, _ctx() is entered once for the mtmd
                # prefill AND AGAIN INSIDE THE PER-TOKEN LOOP (the llama_decode
                # call further down), never hoisted to one contiguous scope the
                # way _generate() is. dedup_native_stderr spins up a background
                # reader thread per entry, so re-entering it per token would both
                # reset its dedup grouping on every token (defeating it) and pay
                # real thread-creation cost per token, which is the anti-pattern
                # dedup_native_stderr's own docstring warns against. Widening this
                # path needs the same per-call-not-per-token restructuring
                # _generate() has.
                _ctx = _quiet_stderr if not self._verbose else contextlib.nullcontext
                self.last_finish_reason = "stop"
                with self._gen_lock:
                    if self._stop.is_set() or self._ctx_ptr is None:
                        return
                    with _ctx():
                        # Clear any prior turn's KV so the mtmd prefill from position 0 is
                        # valid on a reused context, then evaluate the image+text prompt.
                        self._reset_kv_for_image()
                        from .mtmd import MtmdGpuEncodeFailed
                        try:
                            pos = self._mtmd.eval_into(self._ctx_ptr, prompt, images,
                                                       add_special=add_special)
                        except MtmdGpuEncodeFailed:
                            # The projector runs on the GPU now; a GPU encode that fails
                            # mid-flight (the documented gfx1030 / RDNA2 hipBLAS BF16
                            # case) is worth exactly one CPU retry before the request
                            # fails. The KV must be reset again first: the failed
                            # evaluation already wrote into it. Rebuilding is latched in
                            # the MtmdContext, so this costs one retry per model load,
                            # not one per image.
                            if not self._mtmd.retry_on_cpu():
                                raise
                            self._reset_kv_for_image()
                            pos = self._mtmd.eval_into(self._ctx_ptr, prompt, images,
                                                       add_special=add_special)

                logger.info(
                    "gguf generate (vision): prefill complete in %.2fs, "
                    "%d image(s)", time.monotonic() - _t0, len(images))

                sampler = _build_sampler(
                    vocab=self._tokenizer._vocab,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    seed=self._seed if seed is None else (seed & 0xFFFFFFFF),
                    grammar=None,
                )
                in_decode = True
                logger.info("gguf generate (vision): entering decode loop")
                _decode_t0 = time.monotonic()
                for _ in range(max_new_tokens):
                    with self._gen_lock:
                        if self._stop.is_set() or self._ctx_ptr is None:
                            self.last_finish_reason = "error"
                            break
                        # No explicit accept: llama_sampler_sample() accepts
                        # internally (see the note in _generate).
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
                        tokens_generated += 1
                    # Coarse heartbeat - DEBUG not INFO, see _DECODE_PROGRESS_INTERVAL.
                    if (tokens_generated
                            and tokens_generated % _DECODE_PROGRESS_INTERVAL == 0):
                        logger.debug(
                            "gguf generate (vision): decode progress, %d "
                            "token(s) in %.2fs",
                            tokens_generated, time.monotonic() - _decode_t0)
                else:
                    self.last_finish_reason = "length"
                logger.info(
                    "gguf generate (vision): complete, %d token(s) in %.2fs, "
                    "finish_reason=%s", tokens_generated,
                    time.monotonic() - _decode_t0, self.last_finish_reason)
            except GeneratorExit:
                logger.info(
                    "gguf generate (vision): aborted (cancelled) during %s, "
                    "%d token(s) generated",
                    "decode" if in_decode else "prefill", tokens_generated)
                raise
            except Exception:
                logger.info(
                    "gguf generate (vision): aborted (exception) during %s, "
                    "%d token(s) generated",
                    "decode" if in_decode else "prefill", tokens_generated)
                raise
            finally:
                if sampler is not None:
                    api.llama_sampler_free(sampler)

    def _fit_generation_budget(self, n_prompt: int, max_new_tokens: int) -> int:
        """
        Clamp the generation budget so prompt + reply fits under n_ctx_max.

        Raises ContextCapacityExceededError when the prompt alone leaves no usable room -
        the conversation has genuinely outgrown the configured ceiling.
        """
        if not self._n_ctx_max:
            return max_new_tokens
        room = self._n_ctx_max - n_prompt - 64
        if room < 32:
            from localm.inference.backends.base import ContextCapacityExceededError
            raise ContextCapacityExceededError(
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
        if (
            self._ctx_ptr is None
            or needed_tokens > self._ctx_capacity
            or not self._memory_api_available()
        ):
            return False
        # M-RoPE models use multi-dimensional RoPE coordinate grids that cannot be
        # partially rewound by sequence removal; always start clean from fresh context.
        if api.llama_model_has_mrope(self._model_ptr):
            return False
        return True

    def _cache_can_drop_a_speculative_token(self) -> bool:
        """Whether the main cache can drop one trailing position.

        Speculation writes a draft token into the cache and removes it again
        when the target rejects it. A hybrid or recurrent cache cannot be
        truncated at all - measured on qwen35, where removal succeeds only for
        the whole sequence - so speculation there ends every rejection in a full
        rebuild. Asking two tokens' worth of question at load costs far less
        than discovering it mid-reply.

        Answers True when the probe itself cannot run: an unanswered question is
        not evidence of inability, and the rejection path handles the failure.
        """
        try:
            mem = api.llama_get_memory(self._ctx_ptr)
            if not mem:
                return True
            batch = self._create_batch([0, 0], 0, logits_at_last_only=True)
            try:
                if api.llama_decode(self._ctx_ptr, batch) != 0:
                    return True
            finally:
                api.llama_batch_free(batch)
            can = bool(api.llama_memory_seq_rm(mem, 0, 1, -1))
            api.llama_memory_clear(mem, True)
            return can
        except Exception:
            return True

    def _capture_h(self, row: int = -1) -> bool:
        """Copy the main context's next-n hidden state for *row* into _pending_h.

        The MTP head at a position consumes the hidden state from the position
        BEFORE it, so the draft that follows this decode needs the state this
        decode just produced. The pointer llama.cpp returns is into its own
        buffer and is overwritten by the next decode, hence the copy.
        """
        if not self._mtp_wants_h:
            return False
        ptr = (api.llama_get_embeddings_nextn(self._ctx_ptr) if row < 0
               else api.llama_get_embeddings_nextn_ith(self._ctx_ptr, row))
        if not ptr:
            self._pending_h = None
            return False
        if self._h_buf is None:
            self._h_buf = (ctypes.c_float * self._n_embd)()
        ctypes.memmove(self._h_buf, ptr, self._n_embd * ctypes.sizeof(ctypes.c_float))
        self._pending_h = self._h_buf
        return True

    def _create_draft_batch(self, token: int, pos: int):
        """A one-token batch carrying the hidden state the draft head reads.

        ``llama_batch_init`` allocates token OR embd, never both, so embd comes
        from the library and the token array is attached here. The original
        pointer is restored before the batch is freed, so the library's own
        free() never sees an allocation it did not make.
        """
        batch = api.llama_batch_init(1, self._n_embd, 1)
        original_token = batch.token
        holder = (llama_token * 1)(token)
        batch.token = ctypes.cast(holder, ctypes.c_void_p)
        batch.n_tokens = 1
        ctypes.cast(batch.pos, ctypes.POINTER(ctypes.c_int32))[0] = pos
        ctypes.cast(batch.n_seq_id, ctypes.POINTER(ctypes.c_int32))[0] = 1
        ctypes.cast(batch.seq_id, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)))[0][0] = 0
        ctypes.cast(batch.logits, ctypes.POINTER(ctypes.c_int8))[0] = 1
        ctypes.memmove(batch.embd, self._pending_h,
                       self._n_embd * ctypes.sizeof(ctypes.c_float))
        return batch, original_token, holder

    @staticmethod
    def _free_draft_batch(batch, original_token) -> None:
        """Detach the caller-owned token array, then free what the library owns."""
        batch.token = original_token
        api.llama_batch_free(batch)

    def _disable_mtp(self, status: str, detail: str) -> None:
        """Turn speculation off for the rest of this model's life, and say why."""
        self._mtp_usable = False
        self.supports_mtp = False
        self.mtp_status = status
        from localm.debuglog import logger as _dbg
        _dbg.info("MTP: speculation disabled - %s", detail)

    def _prefill_mtp(self, tokens: List[int], base_pos: int) -> None:
        """Mirror a prefill into the MTP draft cache.

        The draft context is created once, with its own smaller n_ctx, and is
        never resized while the main context grows. A conversation that outgrows
        it can no longer be drafted for, so stop here rather than paying a
        failing decode per token for the rest of the session. A draft decode
        that fails for any other reason leaves the draft cache out of step with
        the main one, which is the same dead end.
        """
        if not tokens:
            return
        cap = self._mtp_ctx_capacity
        if cap and base_pos + len(tokens) > cap:
            self._disable_mtp(
                "draft-context-full",
                "the conversation outgrew the %d-token draft context" % cap)
            return
        for i in range(0, len(tokens), _PREFILL_CHUNK):
            batch = self._create_batch(tokens[i:i + _PREFILL_CHUNK],
                                       base_pos + i, logits_at_last_only=True)
            try:
                ret = api.llama_decode(self._mtp_ctx_ptr, batch)
            except Exception as exc:
                self._disable_mtp("draft-prefill-error:%s" % type(exc).__name__,
                                  "the draft prefill raised %s" % type(exc).__name__)
                return
            finally:
                api.llama_batch_free(batch)
            if ret != 0:
                self._disable_mtp("draft-prefill-failed:%d" % ret,
                                  "the draft prefill returned %d" % ret)
                return

    def _rebuild_kv_after_stuck_draft(self) -> bool:
        """Re-decode the emitted tokens into a cleared main KV cache.

        Called when a rejected draft token could not be removed from the cache,
        which leaves a cell at the position the next batch wants to write.
        ``_cached_tokens`` holds exactly the tokens already emitted, so decoding
        them from position 0 restores the state the caller's ``pos`` describes.
        Returns False when the rebuild itself fails.
        """
        tokens = list(self._cached_tokens)
        try:
            api.llama_memory_clear(api.llama_get_memory(self._ctx_ptr), True)
            for i in range(0, len(tokens), _PREFILL_CHUNK):
                batch = self._create_batch(tokens[i:i + _PREFILL_CHUNK], i,
                                           logits_at_last_only=True)
                try:
                    if api.llama_decode(self._ctx_ptr, batch) != 0:
                        return False
                finally:
                    api.llama_batch_free(batch)
        except Exception as exc:
            from localm.debuglog import logger as _dbg
            _dbg.debug("MTP: KV rebuild after a stuck draft cell failed (%s)",
                       type(exc).__name__)
            return False
        return True

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
        # text out of order"). A zero prefix clears the memory outright instead of
        # removing a range: recurrent and M-RoPE state cannot be partially rewound,
        # so a range removal can leave them stale.
        mtp_needs_full_prefill = False
        if prefix == 0:
            api.llama_memory_clear(mem, True)
            if self._mtp_ctx_ptr is not None and self._mtp_usable:
                try:
                    mem_mtp = api.llama_get_memory(self._mtp_ctx_ptr)
                    api.llama_memory_clear(mem_mtp, True)
                except Exception:
                    pass
        elif prefix < len(self._cached_tokens) or not self._cached_tokens:
            if api.llama_memory_seq_rm(mem, 0, prefix, -1):
                # The suffix below is decoded into BOTH caches at prefix + i, so
                # the draft cache has to drop the same range the main one did.
                # A draft cache that cannot be trimmed is cleared and refilled
                # from the whole prompt instead of the suffix alone.
                if self._mtp_ctx_ptr is not None and self._mtp_usable:
                    try:
                        mem_mtp = api.llama_get_memory(self._mtp_ctx_ptr)
                        if not api.llama_memory_seq_rm(mem_mtp, 0, prefix, -1):
                            api.llama_memory_clear(mem_mtp, True)
                            mtp_needs_full_prefill = True
                    except Exception as exc:
                        # The draft cache's state is now unknown, so refilling it
                        # would guess. Stop drafting instead.
                        self._disable_mtp(
                            "draft-trim-error:%s" % type(exc).__name__,
                            "trimming the draft cache raised %s" % type(exc).__name__)
            else:
                # Partial removal unsupported (e.g. SWA cache / M-RoPE) - start over
                api.llama_memory_clear(mem, True)
                if self._mtp_ctx_ptr is not None and self._mtp_usable:
                    try:
                        mem_mtp = api.llama_get_memory(self._mtp_ctx_ptr)
                        api.llama_memory_clear(mem_mtp, True)
                    except Exception:
                        pass
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
                # If partial reuse failed (e.g. M-RoPE position mismatch or recurrent state conflict),
                # perform a full clean prefill from position 0
                self._cached_tokens = []
                try:
                    api.llama_memory_clear(mem, True)
                    if self._mtp_ctx_ptr is not None and self._mtp_usable:
                        mem_mtp = api.llama_get_memory(self._mtp_ctx_ptr)
                        api.llama_memory_clear(mem_mtp, True)
                except Exception:
                    pass
                if prefix > 0:
                    for j in range(0, len(prompt_tokens), _PREFILL_CHUNK):
                        full_chunk = prompt_tokens[j:j + _PREFILL_CHUNK]
                        full_batch = self._create_batch(full_chunk, j, logits_at_last_only=True)
                        full_ret = api.llama_decode(self._ctx_ptr, full_batch)
                        api.llama_batch_free(full_batch)
                        if full_ret != 0:
                            raise RuntimeError(f"llama_decode failed during prefill (code {full_ret})")
                    break
                else:
                    raise RuntimeError(f"llama_decode failed during prefill (code {ret})")

        self._capture_h(-1)
        if self._mtp_ctx_ptr is not None and self._mtp_usable:
            mtp_tokens = prompt_tokens if mtp_needs_full_prefill else suffix
            mtp_base = 0 if mtp_needs_full_prefill else prefix
            self._prefill_mtp(mtp_tokens, mtp_base)

        self._cached_tokens = list(prompt_tokens)

    def _prefill_fresh_context(self, prompt_tokens: List[int], needed: int) -> None:
        """Recreate the context (empty KV cache) and prefill the full prompt.

        Consults ``self._vram_check`` (when set) with the target n_ctx BEFORE
        freeing the live context, so a refusal leaves the old, still-working
        context and its cache intact instead of destroying it first and only
        then discovering the bigger replacement cannot fit. This is the same
        "will it fit" question the caller's own preflight already answered for
        the INITIAL load; growth (e.g. the very first prompt, since a request
        needing more than the base n_ctx forces a grow here) got no such check
        until this hook - only a NULL-pointer check on the result, after the
        native call already ran.
        """
        target = self._target_ctx(needed)
        offload_kqv = True
        vram_check = getattr(self, "_vram_check", None)
        if vram_check is not None:
            # Ask WHERE this context's KV cache must live. The check reads
            # self._offload_kqv (the CURRENT placement) to charge correctly: the net
            # growth when the old KV is in VRAM and reclaimed by the free below, or the
            # full target when the old KV is already in system RAM (nothing to reclaim).
            # If it does not fit VRAM, keep the FULL window but put the KV cache in
            # system RAM (slower) rather than shrinking the window or refusing - a
            # degrade, not an abort, so a model that can run always runs.
            decision = vram_check(target, self._ctx_capacity)
            if decision is False:
                offload_kqv = False

        if self._ctx_ptr:
            api.llama_free(self._ctx_ptr)
            self._ctx_ptr = None
        self._cached_tokens = []

        cp = api.llama_context_default_params()
        cp.n_ctx       = target
        cp.n_batch     = min(cp.n_ctx, 2048)
        cp.n_ubatch    = cp.n_batch   # micro-batch must match so prefill fits in one call
        cp.offload_kqv = offload_kqv  # False -> KV cache in system RAM (VRAM was tight)
        # The grown context must keep the rollback snapshots too, or speculation
        # stops working the moment a conversation outgrows its first context.
        if self._mtp_enabled and hasattr(cp, "n_rs_seq"):
            cp.n_rs_seq = max(int(getattr(cp, "n_rs_seq", 0) or 0), 2)

        self._ctx_ptr = api.llama_init_from_model(self._model_ptr, cp)
        if not self._ctx_ptr:
            # The native context could not be created. Report HONESTLY where the KV
            # was placed: "even in system RAM" only when we actually chose RAM;
            # if we judged it fit VRAM and it still failed, say so - do not claim
            # a RAM fallback that was never attempted.
            where = ("even with the KV cache in system RAM"
                     if not offload_kqv else "with the KV cache in VRAM")
            raise RuntimeError(
                f"Not enough memory to create a {target:,}-token context, {where}. "
                f"Start a new chat, lower n_ctx_max, or free some memory."
            )
        self._ctx_capacity = cp.n_ctx
        self._offload_kqv = offload_kqv   # record the new context's KV placement
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

        self._capture_h(-1)
        if self._mtp_ctx_ptr is not None and self._mtp_usable:
            self._prefill_mtp(prompt_tokens, 0)

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
                if self._mtp_ctx_ptr is not None and self._mtp_usable:
                    try:
                        mem_mtp = api.llama_get_memory(self._mtp_ctx_ptr)
                        api.llama_memory_clear(mem_mtp, True)
                    except Exception:
                        pass
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
        prompt, fallback_reason = _apply_model_template(self._model_ptr, messages)
        if fallback_reason:
            self.chat_template_fallback_reason = fallback_reason

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
        scrub.  Chat output is always scrubbed; in debug mode the raw pre-scrub
        text is additionally written to the debug log - EXCEPT in privacy mode,
        where chat content is never persisted (debug_content_enabled)."""
        from localm.debuglog import debug_content_enabled, logger
        # Decode token BYTES through one UTF-8-safe stream so a character split
        # across a token boundary is reassembled, not turned into U+FFFD.
        raw = _utf8_pieces(self._tokenizer.token_to_piece_bytes(t) for t in gen)
        if debug_content_enabled():
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
