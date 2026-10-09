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
import collections
import contextlib
import ctypes
import functools
import os
import re
import statistics
import tempfile
import threading
import time
import uuid
from typing import Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

from localm.inference import pretokenizer_guard
from localm.textguard import (
    content_spans_via_sentinels, map_untrusted_ranges, split_by_trust,
    untrusted_spans_of,
)

from . import _api as api
from ._drafting import (
    SPEC_MTP, SPEC_NGRAM, SPEC_OFF, DraftSource, MtpSource, resolve_spec_source)
from ._ngram import NgramSource, ngram_draft_cap, ngram_rs_seq
from ._structs import (
    llama_token, LlamaChatMessage, LlamaBatch, LlamaModelTensorBuftOverride,
    set_use_mmap)


# Held by _quiet_stderr for its whole block, and by generation's
# dedup_native_stderr only while it redirects or restores fd 2. Lock order: a
# LlamaCpp's _gen_lock is taken before _stderr_lock, never after; no block that
# holds it contains a yield; it is not reentrant, so neither scope is entered
# inside a _quiet_stderr block. See
# test_close_during_a_suspended_grammar_generation_does_not_deadlock and
# test_abandoning_a_generation_during_close_restores_fd2_in_order.
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
    """Return the context manager used to wrap generation stderr:
    nullcontext when verbose is True, otherwise dedup_native_stderr with
    _stderr_lock as its swap lock, with or without a grammar."""
    if verbose:
        return contextlib.nullcontext
    from localm.debuglog import dedup_native_stderr
    return functools.partial(dedup_native_stderr, swap_lock=_stderr_lock)


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

# llama.cpp's load-time report of how it reads the weights: "(load_mode = mmap)"
# on builds with the load_mode enum (builds that resolve AUTO print mmap or none
# instead), "(mmap = true)" on older ones. The value classes share no character
# with the closing ")", so a failed match ends without backtracking.
_LOAD_MODE_RE = re.compile(r"\(load_mode = ([a-z+]+)\)|\(mmap = (true|false)\)")
_MAPPED_LOAD_MODES = {"mmap": True, "mmap+mlock": True, "none": False,
                      "mlock": False, "dio": False}
_MMAP_UNSUPPORTED_TEXT = "mmap is not supported on this platform"


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

    def mapped(self) -> Optional[bool]:
        """Whether the captured native load memory-mapped the model file: False
        when the log says mmap is not supported on this platform; else the
        last ``(load_mode = X)`` / ``(mmap = X)`` report when it names a mode
        (``auto`` names none); else True when a ``CPU_Mapped`` model buffer
        was reported; else None (not reported)."""
        text = self._read()
        if _MMAP_UNSUPPORTED_TEXT in text:
            return False
        reported = None
        for m in _LOAD_MODE_RE.finditer(text):
            if m.group(1) is not None:
                reported = _MAPPED_LOAD_MODES.get(m.group(1), reported)
            else:
                reported = m.group(2) == "true"
        if reported is not None:
            return reported
        if any(b["backend"] == "CPU_Mapped" for b in self.model_buffers()):
            return True
        return None


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
        hazard = pretokenizer_guard.hazard_note(self._pre_type)
        if hazard is not None:
            from localm.debuglog import logger
            logger.warning(
                "tokenizer: this model declares tokenizer.ggml.pre=%r, whose "
                "pre-tokenizer regex %s. Such input will be refused rather "
                "than tokenised.", self._pre_type, hazard)

    def encode(self, text: str, add_bos: bool = True, untrusted_ranges=()) -> List[int]:
        """Tokenise *text*, parsing control tokens everywhere except untrusted ranges.

        *untrusted_ranges* are ``(start, end)`` character offsets into *text*
        holding content that came from outside. Those are tokenised with
        ``parse_special=False``, so a control token spelled inside them stays
        ordinary text and cannot forge a role boundary. Everything else, the
        chat template's own role markers included, keeps ``parse_special=True``.

        SCOPE: this covers the tokens the native tokenizer treats as special,
        which is how a chat template's role markers are registered. Whether it
        also covers every non-control added token has not been established here.

        With no ranges this issues the single call it always did.
        """
        pretokenizer_guard.check_text(self._pre_type, text)
        if not untrusted_ranges:
            return self._encode_segment(text, add_bos, True)

        tokens: List[int] = []
        want_bos = add_bos
        for segment, is_untrusted in split_by_trust(text, untrusted_ranges):
            if not segment:
                continue
            tokens.extend(self._encode_segment(segment, want_bos, not is_untrusted))
            want_bos = False
        if want_bos and add_bos:
            return self._encode_segment(text, add_bos, True)
        return tokens

    def _encode_segment(self, text: str, add_special: bool, parse_special: bool) -> List[int]:
        raw = text.encode("utf-8", errors="replace")
        # First call: find required size (returns negative if buffer too small)
        n_max = len(raw) + 128
        buf = (llama_token * n_max)()
        n = api.llama_tokenize(
            self._vocab, raw, len(raw), buf, n_max,
            add_special=add_special, parse_special=parse_special,
        )
        if n < 0:
            # buffer too small - reallocate and retry
            n_max = -n + 64
            buf = (llama_token * n_max)()
            n = api.llama_tokenize(
                self._vocab, raw, len(raw), buf, n_max,
                add_special=add_special, parse_special=parse_special,
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


_ENCODER_ROLE_LABELS = {"user": "User", "assistant": "Assistant", "tool": "Tool"}


def _encoder_message_text(message: Dict) -> str:
    """The text an encoder-decoder prompt takes from one message: its text
    parts (``_extract_text``), "" for absent or None content."""
    return _extract_text(message.get("content") or "")


def _flatten_for_encoder(messages: List[Dict]) -> str:
    """The single text an encoder-decoder model (T5) reads for *messages*.

    Such a model has no chat template. Each message becomes one line, in
    message order, and the lines are joined with a newline:

    * a system message: its text;
    * when the messages hold exactly one non-system message and it is a user
      message: that message's text, unchanged;
    * otherwise each non-system message: ``"<Label>: <text>"``, the label being
      ``User``, ``Assistant``, ``Tool``, or the role name capitalised, followed
      by a last line ``"Assistant:"``.

    Only the text parts of a content list are read.
    """
    turns = [m for m in messages if m.get("role", "user") != "system"]
    labelled = not (len(turns) == 1 and turns[0].get("role", "user") == "user")
    lines = []
    for message in messages:
        role = message.get("role", "user")
        text = _encoder_message_text(message)
        if role == "system" or not labelled:
            lines.append(text)
        else:
            label = _ENCODER_ROLE_LABELS.get(role, str(role).capitalize())
            lines.append(f"{label}: {text}")
    if labelled:
        lines.append(f"{_ENCODER_ROLE_LABELS['assistant']}:")
    return "\n".join(lines)


def _encoder_untrusted_ranges(messages: List[Dict], prompt: str) -> Tuple[Tuple[int, int], ...]:
    """Untrusted character ranges of *prompt*, the ``_flatten_for_encoder``
    text of *messages*, in prompt coordinates. Empty when no message carries
    an annotation, and when the contents cannot be located exactly (logged)."""
    per_message = [untrusted_spans_of(m.get("content")) for m in messages]
    if not any(per_message):
        return ()
    contents = [_encoder_message_text(m) for m in messages]

    def render(sentinels):
        return _flatten_for_encoder(
            [dict(m, content=s) for m, s in zip(messages, sentinels)])

    spans = content_spans_via_sentinels(contents, render, prompt)
    if spans is None:
        from localm.debuglog import logger
        logger.warning(
            "textguard: could not locate message content in the encoder prompt, "
            "so untrusted spans are tokenised with special-token parsing ON; "
            "only the text-level defang applies to this request")
        return ()
    return map_untrusted_ranges(spans, per_message)


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
    prompt, reason = _render_template(model_ptr, messages)
    if reason:
        _warn_chatml_fallback(reason)
    return prompt, reason


def _render_template(model_ptr: int, messages: List[Dict]) -> Tuple[str, Optional[str]]:
    """Render *messages* exactly as :func:`_apply_model_template` does, without logging.

    Returns ``(prompt, fallback_reason)``. Split out so the untrusted-span probe
    can render a second, sentinel-carrying message list without emitting a
    duplicate chat-template warning for the same request.
    """
    tmpl_str = api.llama_model_chat_template(model_ptr)
    if not tmpl_str:
        reason = "model has no embedded chat template"
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
        return _format_chatml(messages), reason

    if needed > buf_size:
        # Reallocate and retry
        buf = ctypes.create_string_buffer(needed + 64)
        needed = api.llama_chat_apply_template(tmpl_bytes, chat_arr, n, True, buf, len(buf))
        if needed <= 0:
            # Same guard as above: a failed or empty render falls back
            reason = "embedded template not recognized/rendered nothing"
            return _format_chatml(messages), reason

    return buf.raw[:needed].decode("utf-8", errors="replace"), None


def _content_spans_in_prompt(
    model_ptr: int,
    messages: List[Dict],
    prompt: str,
    fallback_reason: Optional[str],
) -> Optional[List[Tuple[int, int, int]]]:
    """Character ranges in *prompt* holding each message's content, or ``None``.

    Renders *messages* a second time with every content replaced by a unique
    sentinel, then substitutes the real contents back into that skeleton and
    requires the result to equal *prompt*. The ranges are only returned when
    that equality holds, so a template that escapes, reorders, drops or
    duplicates content yields ``None`` rather than a wrong offset. Nothing is
    searched for inside the rendered output. A template that trims a content's
    surrounding whitespace (the built-in llama3 and gemma formatters) is
    located with the trimmed text; each item carries the number of leading
    characters stripped, see ``textguard.content_spans_via_sentinels``.
    """
    contents = [_extract_text(m.get("content", "")) for m in messages]

    def render(sentinels):
        probe = [dict(m, content=s) for m, s in zip(messages, sentinels)]
        skeleton, probe_reason = _render_template(model_ptr, probe)
        if probe_reason != fallback_reason:
            return None
        return skeleton

    return content_spans_via_sentinels(contents, render, prompt)


def _untrusted_prompt_ranges(
    model_ptr: int,
    messages: List[Dict],
    prompt: str,
    fallback_reason: Optional[str],
) -> Tuple[Tuple[int, int], ...]:
    """Untrusted character ranges of *prompt*, in prompt coordinates.

    Empty when no message carries an annotation (the common case, which costs no
    extra render) and also when the spans cannot be located exactly, in which
    case the caller tokenises exactly as it did before this seam existed and a
    warning records that the stronger defence did not apply.
    """
    per_message = [untrusted_spans_of(m.get("content")) for m in messages]
    if not any(per_message):
        return ()

    spans = _content_spans_in_prompt(model_ptr, messages, prompt, fallback_reason)
    if spans is None:
        from localm.debuglog import logger
        logger.warning(
            "textguard: could not locate message content in the rendered prompt, "
            "so untrusted spans are tokenised with special-token parsing ON; "
            "only the text-level defang applies to this request")
        return ()

    return map_untrusted_ranges(spans, per_message)


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

# Draft tokens one MTP speculation step may propose, and the default.
MTP_DRAFT_TOKENS_MAX = 3
MTP_DRAFT_TOKENS_DEFAULT = 1


def mtp_rs_seq(default_n_rs_seq, draft_tokens) -> int:
    """The ``n_rs_seq`` an MTP-enabled context is created with: the runtime's
    own default, but at least 2 and at least the (clamped) draft-token count."""
    draft_max = max(1, min(int(draft_tokens), MTP_DRAFT_TOKENS_MAX))
    return max(int(default_n_rs_seq or 0), 2, draft_max)

# Accepted tokens queued for the draft cache before they are decoded on their own.
_MTP_QUEUED_ROWS_MAX = 32


def _greedy_chain():
    """A sampler chain holding one greedy sampler; it reuses its candidate
    buffer from one sample to the next."""
    params = api.llama_sampler_chain_default_params()
    params.no_perf = True
    chain = api.llama_sampler_chain_init(params)
    api.llama_sampler_chain_add(chain, api.llama_sampler_init_greedy())
    return chain


class _DraftPacer:
    """Keeps MTP speculation on only while it is measured to cost less time per
    emitted token than decoding one token at a time.

    One per loaded model. ``record`` takes the seconds a step took and the
    tokens it made available; the last ``window`` speculative and plain steps
    are kept. The cost per token of speculating is the median seconds of a
    speculative step divided by the mean tokens one made available, and of
    plain decoding the median seconds of a plain step, so a single slow step
    does not decide anything and every accepted draft counts. While
    speculating, one step in every ``probe_every`` (every ``bootstrap_every``
    until there are ``min_samples`` plain figures) runs plain so the plain
    figure stays current. When both sides have ``min_samples`` figures and
    speculation is the slower one, ``speculate`` answers False for the next
    ``pause_steps`` steps, doubling on each consecutive pause up to
    ``max_pause_steps``; speculation is measured afresh after a pause.
    """

    def __init__(self, probe_every: int = 24, bootstrap_every: int = 3,
                 min_samples: int = 6, window: int = 16, pause_steps: int = 32,
                 max_pause_steps: int = 256) -> None:
        self.probe_every = probe_every
        self.bootstrap_every = bootstrap_every
        self.min_samples = min_samples
        self.first_pause = pause_steps
        self.max_pause_steps = max_pause_steps
        self._spec = collections.deque(maxlen=window)         # seconds per speculative step
        self._spec_tokens = collections.deque(maxlen=window)  # tokens each one made available
        self._plain = collections.deque(maxlen=window)        # seconds per plain step
        self.pauses = 0
        self._since_plain = 0
        self._pause_left = 0
        self._pause_len = pause_steps

    @property
    def spec_cost(self) -> Optional[float]:
        """Seconds per token of recent speculative steps (median step time over
        mean tokens per step), or None."""
        if not self._spec:
            return None
        return statistics.median(self._spec) / statistics.fmean(self._spec_tokens)

    @property
    def plain_cost(self) -> Optional[float]:
        """Median seconds of recent plain one-token steps, or None."""
        return statistics.median(self._plain) if self._plain else None

    @property
    def n_spec(self) -> int:
        return len(self._spec)

    @property
    def n_plain(self) -> int:
        return len(self._plain)

    @property
    def paused(self) -> bool:
        """True while speculation is paused for being the slower path."""
        return self._pause_left > 0

    def speculate(self) -> bool:
        """Whether the next step should draft; consumes one paused step."""
        if self._pause_left > 0:
            self._pause_left -= 1
            if self._pause_left == 0:
                self._spec.clear()
                self._spec_tokens.clear()
            return False
        every = (self.bootstrap_every if self.n_plain < self.min_samples
                 else self.probe_every)
        if self._since_plain >= every - 1:
            self._since_plain = 0
            return False
        self._since_plain += 1
        return True

    def record(self, speculative: bool, seconds: float, tokens: int) -> None:
        """Account one step: *seconds* spent, *tokens* made available."""
        seconds = max(0.0, seconds)
        if speculative:
            self._spec.append(seconds)
            self._spec_tokens.append(max(1, tokens))
        else:
            self._plain.append(seconds / max(1, tokens))
        if (speculative and self.n_spec >= self.min_samples
                and self.n_plain >= self.min_samples):
            if self.spec_cost > self.plain_cost:
                self._pause_left = self._pause_len
                self._pause_len = min(self._pause_len * 2, self.max_pause_steps)
                self.pauses += 1
            else:
                self._pause_len = self.first_pause


def _address(ptr) -> Optional[int]:
    """The integer address behind a ctypes pointer or array, or None for NULL."""
    if ptr is None:
        return None
    if isinstance(ptr, ctypes.Array):
        return ctypes.addressof(ptr)
    return ctypes.cast(ptr, ctypes.c_void_p).value or None


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
    _mtp_ctx_ptr = None          # the MTP draft context, None until created
    _mtp_ctx_capacity = 0        # the draft context's own n_ctx, 0 until created
    _mtp_wants_h = False         # True once both contexts expose the next-n state
    mtp_active_this_call = False # whether THIS generation actually speculated
    mtp_call_status = ""         # why THIS generation stopped speculating, "" if it did not
    _mtp_draft_stale = False     # the draft cache missed tokens the main cache holds
    _mtp_draft_max = 1           # draft tokens one speculation step may propose
    _mtp_drafting = False        # THIS generation is still proposing drafts
    _mtp_backend_chain = None    # sampler chain attached to the draft context, or None
    mtp_drafted = 0              # draft tokens THIS generation sent to verification
    mtp_accepted = 0             # how many of those the target model accepted
    mtp_steps = 0                # verification batches THIS generation decoded
    mtp_paused_steps = 0         # steps THIS generation ran plain because drafting was slower
    mtp_skipped = ""             # why THIS generation could not draft at all: "image" or ""
    is_encoder_decoder = False   # the model runs llama_encode before decoding (T5)
    encoder_input_limit = 0      # most tokens one llama_encode call takes, 0 unless encoder-decoder
    _draft_pacer = None          # _DraftPacer for this model, created on first use
    _source = None               # the DraftSource for this model, created on first use
    _spec_source_name = SPEC_MTP # the configured draft source: off, mtp or ngram
    _ngram_draft_max = 0         # draft tokens per n-gram step, 0 unless the source is ngram
    _clock = time.perf_counter
    _draft_pos = 0               # the draft cache holds positions [0, _draft_pos)
    _queued_tokens: Tuple[int, ...] = ()  # tokens at _draft_pos.. not yet in the draft cache
    _queued_h = None             # their hidden-state rows, one per queued token
    _n_threads = None
    _pending_h = None            # the hidden state the next draft will read
    _pending_h_pos = -1          # the position whose hidden state _pending_h holds
    _h_buf = None                # reusable copy target for it
    _n_embd = 0
    # The prompt the image path last evaluated into the KV cache, one
    # (key, n_pos) pair per unit: a text token id with n_pos 1, or a media
    # chunk's key with its n_pos. The reply decoded after it is not recorded.
    # None when the KV cache does not start with such a prompt.
    _vision_kv: Optional[List[Tuple[object, int]]] = None

    @property
    def _cached_tokens(self) -> List[int]:
        """Tokens the text path holds in the KV cache.

        Assigning it also drops the image path's record (``_vision_kv``)."""
        return self._text_kv_tokens

    @_cached_tokens.setter
    def _cached_tokens(self, tokens: List[int]) -> None:
        self._text_kv_tokens = tokens
        self._vision_kv = None

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
        main_gpu: Optional[int] = None,
        mtp_draft_tokens: int = MTP_DRAFT_TOKENS_DEFAULT,
        spec_source: Optional[str] = None,
        spec_draft_tokens: Optional[int] = None,
        use_mmap: Optional[bool] = None,
        **_ignored,
    ) -> None:
        self._n_ctx       = n_ctx
        # spec_source names the draft source (off, mtp, ngram); None follows
        # mtp_enabled. MTP is enabled exactly when the source is mtp.
        self._spec_source_name = resolve_spec_source(spec_source, mtp_enabled)
        self._mtp_enabled = self._spec_source_name == SPEC_MTP
        self._mtp_draft_max = max(1, min(int(mtp_draft_tokens), MTP_DRAFT_TOKENS_MAX))
        self._n_threads = n_threads
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
        self._pending_h_pos = -1    # the position whose hidden state _pending_h holds
        self._h_buf       = None    # reusable copy target for it
        self._n_embd      = 0
        self._draft_pos   = 0       # the draft cache holds positions [0, _draft_pos)
        self._queued_tokens = []    # tokens at _draft_pos.. not yet in the draft cache
        self._mmproj_path = mmproj_path
        self._mtmd        = None   # MtmdContext (vision) when an mmproj is loaded
        self._tokenizer   = None   # type: ignore[assignment]
        # Serialize native calls (prefill/decode/free) against unload. Without
        # this, an unload on another thread can llama_free the context between
        # the generator's None-check and its next native call - a use-after-
        # free that crashes the GPU driver. The decode loop holds _gen_lock
        # around each native step; close()/_free_native take it too, after
        # setting _stop so an in-flight generation bails at its next step.
        # Lock order: _gen_lock before the module-level _stderr_lock.
        # _gen_lock is held around native calls, never across a yield.
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
        if hasattr(mp, "load_mtp") and self._mtp_enabled:
            mp.load_mtp = True
        # use_mmap True or False forces that load mode; None keeps the build's
        # default (mmap on every device that supports it).
        if use_mmap is not None:
            # Newer builds replaced use_mmap/use_mlock/use_direct_io with a
            # single load_mode enum at a DIFFERENT offset; set_use_mmap writes
            # whichever this build has. Assigning mp.use_mmap directly would
            # land in check_tensors on those builds - same size, no error.
            set_use_mmap(mp, bool(use_mmap))
        # Multi-GPU: honour the configured main_gpu_index (validated against
        # the devices actually visible right now), or the parent's main_gpu in
        # llama.cpp's device numbering when given; leaves the native default
        # (device 0) untouched when unset. See discover.apply_main_gpu.
        from localm.discover import apply_gpu_split, apply_main_gpu
        if main_gpu is not None:
            apply_main_gpu(mp, slot=main_gpu)
        else:
            apply_main_gpu(mp)
        # Multi-GPU tensor-split: spreads the model across 2+ configured
        # devices when gpu_split_indices is set, or loads it on the one device
        # a 1-entry ratios mapping from the parent names (see
        # discover.apply_gpu_split).
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
        # Whether the load memory-mapped the weights, from the native load log
        # (_CapturedStderr.mapped); None when not reported or not captured.
        self.mmap_mapped: Optional[bool] = None
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
                    self.mmap_mapped = captured.mapped()
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

        # Refuse a model whose declared pre-tokenizer cannot hold a
        # conversation, before a context is allocated for it.
        refusal = pretokenizer_guard.load_refusal(
            pretokenizer_guard.read_pre_type(self._model_ptr, api))
        if refusal is not None:
            api.llama_free_model(self._model_ptr)
            self._model_ptr = None
            raise pretokenizer_guard.PretokenizerUnusableModelError(refusal)

        try:
            self.is_encoder_decoder = self._detect_encoder_decoder()
        except Exception:
            api.llama_free_model(self._model_ptr)
            self._model_ptr = None
            raise
        if self.is_encoder_decoder:
            # No draft source runs on an encoder-decoder model, and nothing
            # below may decode on its context before llama_encode has run.
            self._spec_source_name = SPEC_OFF
            self._mtp_enabled = False

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
        # every later batch is refused. Rolling back r positions needs r
        # snapshots, and a step that proposes k drafts can reject all k.
        # Costs nothing on a model with no recurrent layers.
        # See test_recurrent_rollback_is_requested_when_mtp_is_enabled.
        self._apply_initial_spec_params(cp, spec_draft_tokens)
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
        if self.is_encoder_decoder:
            self.encoder_input_limit = self._read_encoder_input_limit(cp)

        # Multi-Token Prediction (MTP) draft context initialization
        if self.is_encoder_decoder:
            self.mtp_status = "encoder-decoder"
        elif not self._mtp_enabled:
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
                if eligible:
                    failure = self._create_mtp_context(n_ctx, True, n_threads, _ctx)
                    if failure:
                        self.mtp_status = failure
                    else:
                        self.supports_mtp = True
            except Exception as exc:
                self._mtp_ctx_ptr = None
                self.supports_mtp = False
                self.mtp_status = f"error:{type(exc).__name__}"
        from localm.debuglog import logger as _mtp_log
        _mtp_log.info("MTP: active=%s status=%s", self.supports_mtp, self.mtp_status)
        if self._spec_source_name == SPEC_NGRAM:
            source = self._draft_source()
            if not self._cache_can_drop_a_speculative_token():
                source.usable = False
                source.status = "rewind-unsupported"
            _mtp_log.info("n-gram drafting: status=%s draft_max=%d",
                          source.status, self._ngram_draft_max)

        self._tokenizer = _Tokenizer(self._model_ptr, self._ctx_ptr)

        # Optional in-process vision (C1): load the mmproj via mtmd so image
        # messages can be answered. Best-effort - any failure (no mtmd.dll, an
        # incompatible mmproj) leaves the model text-only rather than breaking it.
        if mmproj_path and self.is_encoder_decoder:
            _mtp_log.warning(
                "vision: an encoder-decoder model reads text only, so the "
                "projector %s was not loaded", os.path.basename(mmproj_path))
        elif mmproj_path:
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
                    from .mtmd import MtmdContext, compatible_mmproj_path
                    # getattr, not self._main_gpu_index: this method is unit-tested
                    # directly against instances that never ran __init__ (see this
                    # docstring's note on why it was pulled out), and 0 is exactly
                    # the "leave clip's own default alone" value.
                    mt = MtmdContext(
                        compatible_mmproj_path(mmproj_path), self._model_ptr,
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
        the GGUF header probe, which CAN read the exact per-layer array.

        An encoder-decoder model is sized by ``_decoder_kv_bytes_per_token``
        when its metadata states the head width."""
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
                    if self.is_encoder_decoder:
                        decoder_kv = self._decoder_kv_bytes_per_token(n_head_kv)
                        if decoder_kv:
                            return decoder_kv
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

    def _architecture(self) -> Optional[str]:
        """The loaded model's general.architecture, or None when this build
        cannot read metadata or the key is absent."""
        if not api.has_model_meta_api():
            return None
        return api.llama_model_meta_val_str(self._model_ptr, "general.architecture")

    def _meta_int(self, key: str) -> int:
        """The loaded model's metadata value under *key* as a positive int, or 0
        when it is absent, unreadable or not a positive integer."""
        if not api.has_model_meta_api():
            return 0
        raw = api.llama_model_meta_val_str(self._model_ptr, key)
        try:
            value = int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            return 0
        return value if value > 0 else 0

    def _decoder_kv_bytes_per_token(self, n_head_kv: int) -> int:
        """f16 KV-cache bytes per token of an encoder-decoder model's decoder
        stack: ``<arch>.decoder_block_count`` layers (defaulting to n_layers) x
        *n_head_kv* x (``key_length`` + ``value_length``) x 2 bytes. 0 when the
        metadata does not state both head widths."""
        arch = self._architecture()
        if not arch:
            return 0
        k_len = self._meta_int(f"{arch}.attention.key_length")
        v_len = self._meta_int(f"{arch}.attention.value_length")
        if not (k_len and v_len):
            return 0
        layers = self._meta_int(f"{arch}.decoder_block_count") or self.n_layers or 0
        return layers * n_head_kv * (k_len + v_len) * 2

    def _detect_encoder_decoder(self) -> bool:
        """True when the loaded model has both an encoder and a decoder stack
        (T5); False for every other model, an encoder-only one included.

        Raises RuntimeError when this build does not export the encoder API and
        the model's architecture is an encoder-decoder one, since such a model
        cannot generate without ``llama_encode``."""
        if api.has_encoder_api():
            return (api.llama_model_has_encoder(self._model_ptr)
                    and api.llama_model_has_decoder(self._model_ptr))
        arch = self._architecture()
        from localm.model_manager.gguf import _GGUF_ENCODER_DECODER_ARCHITECTURES
        if arch in _GGUF_ENCODER_DECODER_ARCHITECTURES:
            raise RuntimeError(
                f"This model's architecture ('{arch}') is an encoder-decoder "
                "model, and the loaded llama runtime does not export "
                "llama_encode, so it cannot run it. Update the runtime with  "
                "localm setup-llama")
        return False

    def _read_encoder_input_limit(self, cp) -> int:
        """Most tokens one ``llama_encode`` call takes on the live context: its
        n_ubatch as llama.cpp reports it, or, on a build without that accessor,
        *cp*'s n_ubatch clamped as llama.cpp clamps it (to n_batch, itself
        clamped to n_ctx)."""
        native = api.llama_n_ubatch(self._ctx_ptr)
        if native:
            return int(native)
        return int(min(cp.n_ubatch, cp.n_batch, cp.n_ctx))

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
        self._free_backend_draft_sampler()
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
        on_status: Optional[Callable[[str], None]] = None,
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

            if on_status:
                on_status("Processing prompt...")

            # Dynamic window: shrink the generation budget to fit under the
            # ceiling rather than blowing past it; fail clearly when even a
            # minimal reply cannot fit any more.
            max_new_tokens = self._fit_generation_budget(n_prompt, max_new_tokens)

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
            source = None
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
                # A draft source never hands its proposals to the request's
                # sampler: every emitted token, with or without a grammar in that
                # sampler, is one the sampler itself sampled from a verification
                # or decode row, in emission order.
                # See test_a_grammar_reply_drafts_and_its_sampler_sees_only_emitted_tokens.
                source = self._draft_source()
                self.mtp_skipped = ""
                self.mtp_active_this_call = False
                self.mtp_call_status = ""
                self.mtp_drafted = 0
                self.mtp_accepted = 0
                self.mtp_steps = 0
                self.mtp_paused_steps = 0
                if source.begin_call() and self._draft_pacer is None:
                    self._draft_pacer = _DraftPacer()
                pacer = self._draft_pacer
                clock = self._clock
                # The step whose cost is still being measured: (start, drafted,
                # tokens it makes available); its time ends when the next token
                # is in hand and excludes time spent in the consumer.
                step = None
                consumer_s = 0.0
                pos = n_prompt
                # Why generation ended, read by callers as self.last_finish_reason.
                # Default "stop" - it must cover every early exit (EOG token, a
                # stop-string match in _filtered_stream abandoning this generator,
                # client abort). Only a genuinely exhausted token budget is "length".
                self.last_finish_reason = "stop"
                in_decode = True
                logger.info("gguf generate: entering decode loop")
                if on_status:
                    on_status("Generating response...")
                _decode_t0 = time.monotonic()
                # ONE contiguous _ctx() scope for the whole streaming loop, not
                # re-entered per native call: dedup_native_stderr() spins up a
                # background reader thread, so re-entering it per-token would
                # both reset its dedup state every time (defeating grouping
                # across tokens) and pay thread-creation cost per token. The
                # native calls below run unwrapped inside this single scope,
                # and so does the yield: dedup_native_stderr holds _stderr_lock
                # only while it redirects or restores fd 2.
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

                        if step is not None:
                            pacer.record(step[1], clock() - step[0] - consumer_s, step[2])
                            step = None

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
                                        decoded = api.llama_decode(self._ctx_ptr, batch) == 0
                                        self._cached_tokens.append(token)
                                        pos += 1
                                        if decoded:
                                            source.after_single_token(token, pos - 1)
                                    except Exception as exc:
                                        from localm.debuglog import logger as _dbg
                                        _dbg.debug("gguf generate: final-token bookkeeping raised %s",
                                                   type(exc).__name__)
                                    finally:
                                        if batch is not None:
                                            api.llama_batch_free(batch)
                            break

                        # --- Speculative drafting (while the source drafts) ---
                        drafts: List[int] = []
                        accepted: List[int] = []
                        timed = speculate = False
                        if source.drafting():
                            if pacer.paused:
                                timed = True
                                pacer.speculate()
                                source.on_paused_step()
                            elif source.ready(pos):
                                n_max = source.budget(
                                    pos, max_new_tokens - tokens_generated
                                    if max_new_tokens > 0 else None)
                                if n_max > 0:
                                    timed = True
                                    speculate = pacer.speculate()
                            step_t0 = clock()
                            consumer_s = 0.0
                            if speculate:
                                with self._gen_lock:
                                    if not (self._stop.is_set() or self._ctx_ptr is None):
                                        try:
                                            drafts = source.propose(token, pos, n_max)
                                        except Exception as exc:
                                            drafts = []
                                            source.stop_this_call(
                                                "draft-decode-error:%s" % type(exc).__name__)

                        if drafts:
                            # One main decode verifies [token, d1..dk]. Row i holds
                            # the target's continuation after the token at pos + i.
                            with self._gen_lock:
                                if self._stop.is_set() or self._ctx_ptr is None:
                                    self.last_finish_reason = "error"
                                    break
                                batch = self._create_batch([token] + drafts, pos, logits_at_last_only=False)
                                try:
                                    ret = api.llama_decode(self._ctx_ptr, batch)
                                    if ret == 0:
                                        # Each verification sample goes through the
                                        # REQUEST's sampler, which accepts what it
                                        # returns, and every token sampled here is
                                        # emitted: the matching drafts below, the
                                        # first mismatching token as pending_token.
                                        replacement = None
                                        for i, draft in enumerate(drafts):
                                            verified = api.llama_sampler_sample(sampler, self._ctx_ptr, i)
                                            if verified != draft:
                                                replacement = verified
                                                break
                                            accepted.append(draft)
                                        n_acc = len(accepted)
                                        source.on_verify(len(drafts), n_acc)
                                        removed = True
                                        if n_acc < len(drafts):
                                            # Rejected drafts leave the main cache.
                                            removed = api.llama_kv_cache_seq_rm(
                                                self._ctx_ptr, 0, pos + n_acc + 1, -1)
                                        source.after_verify(accepted, pos)
                                        self._cached_tokens.extend([token] + accepted)
                                        pos += n_acc + 1
                                        if not removed:
                                            # The rejected cells are still in the
                                            # cache and llama.cpp refuses every later
                                            # batch as having inconsistent sequence
                                            # positions. Rebuild from the tokens
                                            # emitted so far and stop speculating.
                                            # See test_a_stuck_draft_cell_disables_mtp_and_keeps_generating.
                                            source.rewind_unsupported()
                                            if not self._rebuild_kv_after_stuck_draft():
                                                self.last_finish_reason = "error"
                                                self._cached_tokens = []
                                                break
                                        if replacement is not None:
                                            # The target's own token for the first
                                            # rejected position, emitted at the loop
                                            # head without a second sample.
                                            pending_token = replacement
                                    else:
                                        # Decode failed, fall back to single token
                                        api.llama_batch_free(batch)
                                        batch = self._create_batch([token], pos, logits_at_last_only=True)
                                        ret = api.llama_decode(self._ctx_ptr, batch)
                                        if ret == 0:
                                            source.after_single_token(token, pos)
                                            self._cached_tokens.append(token)
                                            pos += 1
                                        else:
                                            self.last_finish_reason = "error"
                                            self._cached_tokens = []
                                            break
                                finally:
                                    if batch is not None:
                                        api.llama_batch_free(batch)
                            if timed:
                                step = (step_t0, True, 1 + len(accepted))
                            for draft in accepted:
                                yield_t0 = clock()
                                yield draft
                                consumer_s += clock() - yield_t0
                                tokens_generated += 1
                            continue
                        else:
                            # --- locked native region 2: feed single token back ---
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
                                    source.after_single_token(token, pos)
                                    self._cached_tokens.append(token)
                                    pos += 1
                                    if timed:
                                        step = (step_t0, speculate and not source.free_miss, 1)
                                finally:
                                    # Always release the native batch - including when
                                    # _prefill_fresh_context above raises mid-growth.
                                    if batch is not None:
                                        api.llama_batch_free(batch)
                    else:
                        # Budget exhausted without the model finishing its turn
                        self.last_finish_reason = "length"
                    with self._gen_lock:
                        if not (self._stop.is_set() or self._ctx_ptr is None):
                            source.finish()
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
                if source is not None:
                    source.end_call()

    def encoder_tokens(self, messages: List[Dict]) -> List[int]:
        """The encoder input of an encoder-decoder model for *messages*.

        The ``_flatten_for_encoder`` text, tokenized with control tokens parsed
        everywhere except inside untrusted spans, with the vocabulary's BOS
        prepended when it asks for one and its EOS appended unless it says not
        to (a T5 vocabulary: EOS appended, no BOS)."""
        prompt = _flatten_for_encoder(messages)
        ranges = _encoder_untrusted_ranges(messages, prompt)
        tokens = self._tokenizer.encode(prompt, add_bos=False, untrusted_ranges=ranges)
        vocab = self._tokenizer._vocab
        if api.llama_vocab_get_add_bos(vocab):
            bos = api.llama_token_bos(vocab)
            if bos != api.LLAMA_TOKEN_NULL:
                tokens.insert(0, bos)
        if api.llama_vocab_get_add_eos(vocab) is not False:
            eos = api.llama_token_eos(vocab)
            if eos != api.LLAMA_TOKEN_NULL:
                tokens.append(eos)
        return tokens

    def _clear_decoder_memory(self) -> None:
        """Empty the KV cache before an encoder-decoder request: llama_memory_clear
        when this build has the memory API, else a fresh context of the same
        size. Caller holds _gen_lock."""
        self._cached_tokens = []
        if self._memory_api_available():
            api.llama_memory_clear(api.llama_get_memory(self._ctx_ptr), True)
            return
        self._prefill_fresh_context([], self._ctx_capacity)

    def _decoder_start_token(self) -> int:
        """The token an encoder-decoder model's decoder starts from: the model's
        declared decoder-start token, else its BOS. Raises RuntimeError when the
        model declares neither."""
        token = api.llama_model_decoder_start_token(self._model_ptr)
        if token == api.LLAMA_TOKEN_NULL:
            token = api.llama_token_bos(self._tokenizer._vocab)
        if token == api.LLAMA_TOKEN_NULL:
            raise RuntimeError(
                "This encoder-decoder model declares neither a decoder start "
                "token nor a BOS token, so its decoder has nothing to start from.")
        return int(token)

    def _generate_encoder_decoder(
        self,
        messages: List[Dict],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repeat_penalty: float,
        grammar: Optional[str] = None,
        grammar_lazy: bool = False,
        grammar_triggers: Optional[List[str]] = None,
        seed: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Iterator[int]:
        """Yield the token ids an encoder-decoder model (T5) generates for
        *messages*, one at a time.

        Every call starts from nothing: the KV cache is cleared, the whole
        ``encoder_tokens(messages)`` input is encoded by one ``llama_encode``
        call, and the decoder starts from ``_decoder_start_token()`` at
        position 0. No state carries over between calls, so no prefix is reused
        and no draft source runs.

        The reply stops at an end-of-generation token, after *max_new_tokens*
        tokens when that is positive, and after the context's n_ctx tokens
        (decoder positions) in any case; the last two set
        ``last_finish_reason`` to "length".

        Raises ContextCapacityExceededError, before any native call, when the
        encoder input is longer than ``encoder_input_limit``; RuntimeError when
        the model is not loaded or ``llama_encode`` or the first decoder step
        fails.
        """
        from localm.debuglog import logger
        from localm.inference.backends.base import ContextCapacityExceededError

        with self._inference_lock:
            if not self._model_ptr:
                raise RuntimeError("Model not loaded")
            if on_status:
                on_status("Processing prompt...")
            enc_tokens = self.encoder_tokens(messages)
            n_enc = len(enc_tokens)
            if n_enc > self.encoder_input_limit:
                raise ContextCapacityExceededError(
                    f"This prompt is {n_enc} tokens, and this encoder-decoder "
                    f"model reads at most {self.encoder_input_limit} tokens of "
                    f"prompt in one pass. Shorten the message or start a new chat.")
            budget = self._ctx_capacity
            if max_new_tokens > 0:
                budget = min(budget, max_new_tokens)

            self.mtp_skipped = ""
            self.mtp_active_this_call = False
            self.mtp_call_status = ""
            self.mtp_drafted = 0
            self.mtp_accepted = 0
            self.mtp_steps = 0
            self.mtp_paused_steps = 0
            self.last_finish_reason = "stop"

            _ctx = _stderr_ctx_for_generate(self._verbose)
            logger.info("gguf generate: encoding %d prompt token(s)", n_enc)
            _t0 = time.monotonic()
            tokens_generated = 0
            in_decode = False
            sampler = None
            try:
                with self._gen_lock:
                    if self._stop.is_set() or self._ctx_ptr is None:
                        self.last_finish_reason = "error"
                        return
                    with _ctx():
                        self._clear_decoder_memory()
                        enc_arr = (llama_token * n_enc)(*enc_tokens)
                        ret = api.llama_encode(
                            self._ctx_ptr, api.llama_batch_get_one(enc_arr, n_enc))
                    if ret != 0:
                        raise RuntimeError(f"llama_encode failed (code {ret})")
                    token = self._decoder_start_token()
                logger.info("gguf generate: encode complete in %.2fs",
                            time.monotonic() - _t0)

                sampler = _build_sampler(
                    vocab=self._tokenizer._vocab,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    seed=self._seed if seed is None else (seed & 0xFFFFFFFF),
                    grammar=grammar,
                    grammar_lazy=grammar_lazy,
                    grammar_triggers=grammar_triggers,
                )
                in_decode = True
                if on_status:
                    on_status("Generating response...")
                _decode_t0 = time.monotonic()
                pos = 0
                with _ctx():
                    while True:
                        with self._gen_lock:
                            if self._stop.is_set() or self._ctx_ptr is None:
                                self.last_finish_reason = "error"
                                break
                            batch = self._create_batch([token], pos, logits_at_last_only=True)
                            try:
                                ret = api.llama_decode(self._ctx_ptr, batch)
                            finally:
                                api.llama_batch_free(batch)
                            if ret != 0:
                                if pos == 0:
                                    raise RuntimeError(
                                        f"llama_decode failed on the decoder start token (code {ret})")
                                logger.warning(
                                    "gguf generate: decoder step failed at position %d "
                                    "(code %d); ending the reply", pos, ret)
                                self.last_finish_reason = "error"
                                break
                            pos += 1
                            token = api.llama_sampler_sample(sampler, self._ctx_ptr, -1)
                            eog = self._tokenizer.is_eog(token)
                        if eog:
                            break
                        yield token
                        tokens_generated += 1
                        if (tokens_generated
                                and tokens_generated % _DECODE_PROGRESS_INTERVAL == 0):
                            logger.debug(
                                "gguf generate: decode progress, %d token(s) in %.2fs",
                                tokens_generated, time.monotonic() - _decode_t0)
                        if tokens_generated >= budget:
                            self.last_finish_reason = "length"
                            break
                logger.info(
                    "gguf generate: complete, %d token(s) in %.2fs, finish_reason=%s",
                    tokens_generated, time.monotonic() - _decode_t0, self.last_finish_reason)
            except GeneratorExit:
                logger.info(
                    "gguf generate: aborted (cancelled) during %s, %d token(s) generated",
                    "decode" if in_decode else "encode", tokens_generated)
                raise
            except Exception:
                logger.info(
                    "gguf generate: aborted (exception) during %s, %d token(s) generated",
                    "decode" if in_decode else "encode", tokens_generated)
                raise
            finally:
                if sampler is not None:
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
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Iterator[int]:
        """Yield generated token ids for a chat whose prompt includes image(s).

        The image+text prompt is evaluated into the KV cache by
        :meth:`_prefill_vision`, which keeps the part of the cache an earlier
        image turn left that still matches; sampling then continues exactly like
        the text loop. Grammar is not applied on the image path. The text path's
        KV record (``_cached_tokens``) is left empty, so the next text turn
        prefills from scratch.

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
            self.mtp_skipped = ("image" if self._mtp_ctx_ptr is not None and self._mtp_usable
                                else "")
            self.mtp_active_this_call = False
            self.mtp_call_status = ""
            self.mtp_drafted = 0
            self.mtp_accepted = 0
            self.mtp_steps = 0
            self.mtp_paused_steps = 0
            self._draft_source().skip_call("image")
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
                # mtmd_tokenize runs the same pre-tokenizer over the text parts
                # of this prompt, so the vision path needs the same check the
                # text path gets in _Tokenizer.encode.
                pretokenizer_guard.check_text(self._tokenizer._pre_type, prompt)

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
                with self._gen_lock:
                    if self._stop.is_set() or self._ctx_ptr is None:
                        return
                    with _ctx():
                        vprompt = self._mtmd.tokenize(prompt, images, add_special=add_special)
                        try:
                            # Sizes the context from the tokenized prompt before
                            # any evaluation; a prompt over n_ctx_max raises
                            # ContextCapacityExceededError here.
                            n_prompt = vprompt.n_tokens
                            max_new_tokens = self._fit_generation_budget(n_prompt, max_new_tokens)
                            encoded_before = self._mtmd.encode_count
                            self.last_finish_reason = "stop"
                            # If unlimited (<= 0), reserve the same modest chunk
                            # _generate does rather than sizing for a runaway reply.
                            initial_budget = max_new_tokens if max_new_tokens > 0 else 512
                            needed = n_prompt + initial_budget + 64
                            from .mtmd import MtmdGpuEncodeFailed
                            try:
                                pos, reused = self._prefill_vision(vprompt, needed, on_status)
                            except MtmdGpuEncodeFailed:
                                # One CPU retry per model load: retry_on_cpu()
                                # returns False once the projector is on the CPU.
                                # The retry re-tokenizes and evaluates from an
                                # empty KV cache.
                                if not self._mtmd.retry_on_cpu():
                                    raise
                                if on_status:
                                    from localm.inference.backends.base import VISION_CPU_FALLBACK_STATUS
                                    on_status(VISION_CPU_FALLBACK_STATUS)
                                vprompt.free()
                                vprompt = self._mtmd.tokenize(
                                    prompt, images, add_special=add_special)
                                self._vision_kv = None
                                pos, reused = self._prefill_vision(vprompt, needed)
                            encoded = self._mtmd.encode_count - encoded_before
                        finally:
                            vprompt.free()

                logger.info(
                    "gguf generate (vision): prefill complete in %.2fs, "
                    "%d image(s), %d image chunk(s) encoded, %d of %d position(s) "
                    "reused", time.monotonic() - _t0, len(images),
                    encoded, reused, pos)
                if on_status:
                    on_status("Generating response...")

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
                # max_new_tokens <= 0 is this codebase's "unlimited" sentinel
                # (see _generate's identical while condition, and
                # _fit_generation_budget's docstring) - a `for _ in
                # range(max_new_tokens)` loop treats 0/negative as "generate
                # nothing" instead, which used to make a vision reply end
                # silently with zero tokens whenever max_tokens was set to
                # unlimited.
                while max_new_tokens <= 0 or tokens_generated < max_new_tokens:
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
                            self._vision_kv = None
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
                    if max_new_tokens > 0 and tokens_generated >= max_new_tokens:
                        self.last_finish_reason = "length"
                        break
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

    def _prefill_vision(self, vprompt, needed: int,
                        on_status: Optional[Callable[[str], None]] = None) -> Tuple[int, int]:
        """Evaluate the tokenized image prompt *vprompt* (an ``MtmdPrompt``) into
        the KV cache and return ``(n_past, reused)``.

        Keeps the longest prefix of the cache that ``_vision_kv`` shows matches
        *vprompt* (text tokens by id, media chunks by key), stopping at least one
        token or image short of the whole prompt, and evaluates only the rest; *reused* is the
        number of positions kept. Nothing is kept when the context has to grow to
        *needed* tokens, when ``_can_reuse_kv`` refuses, when there is no record,
        or when the cache cannot drop its tail. Media chunks go through
        ``MtmdContext.eval_media_chunk``, so an image whose embeddings are cached
        is not encoded again, and the cache then keeps only this prompt's images.

        Emits ``"Encoding image (GPU)..."`` or ``"Encoding image (CPU)..."``
        through *on_status* when an image has to be encoded, else
        ``"Processing prompt..."``. Raises ``MtmdGpuEncodeFailed`` or
        ``VisionInputError`` when evaluation fails, leaving ``_vision_kv`` None.
        Caller must hold ``_gen_lock``."""
        from localm.inference.backends.base import VisionInputError

        units: List[Tuple[object, int]] = []
        starts: List[int] = []
        for chunk in vprompt.chunks:
            starts.append(len(units))
            if chunk.tokens is not None:
                units.extend((tok, 1) for tok in chunk.tokens)
            else:
                key = chunk.key if chunk.key is not None else object()
                units.append((key, chunk.n_pos))

        keep = 0
        if needed > self._ctx_capacity:
            self._prefill_fresh_context([], needed)
        elif self._vision_kv is not None and self._can_reuse_kv(needed):
            keep = min(_common_prefix_len(self._vision_kv, units), len(units) - 1)
            if keep > 0:
                keep_pos = sum(n_pos for _, n_pos in units[:keep])
                mem = api.llama_get_memory(self._ctx_ptr)
                if not api.llama_memory_seq_rm(mem, 0, keep_pos, -1):
                    keep = 0
            if keep <= 0:
                keep = 0
                self._reset_kv_for_image()
        else:
            self._reset_kv_for_image()
        self._vision_kv = None

        if on_status:
            encoding = any(
                chunk.tokens is None and start >= keep
                and not self._mtmd.has_embedding(chunk.key)
                for chunk, start in zip(vprompt.chunks, starts))
            if not encoding:
                on_status("Processing prompt...")
            elif self._mtmd.on_gpu:
                on_status("Encoding image (GPU)...")
            else:
                on_status("Encoding image (CPU)...")

        n_ctx = api.llama_n_ctx(self._ctx_ptr)
        n_batch = min(n_ctx, 2048) if n_ctx else 512
        reused = sum(n_pos for _, n_pos in units[:keep])
        pos = reused
        for chunk, start in zip(vprompt.chunks, starts):
            if chunk.tokens is not None:
                rest = chunk.tokens[min(max(keep - start, 0), len(chunk.tokens)):]
                if rest:
                    pos = self._decode_vision_text(rest, pos, n_batch)
            elif start >= keep:
                pos = self._mtmd.eval_media_chunk(self._ctx_ptr, chunk, pos, n_batch)
        if pos <= 0 or (n_ctx and pos > n_ctx):
            raise VisionInputError(
                f"the image prompt ended at an implausible position "
                f"(n_past={pos}, context size={n_ctx}) - refusing to generate "
                f"from a likely-corrupted KV state")
        self._vision_kv = units
        self._mtmd.retain_embeddings(
            chunk.key for chunk in vprompt.chunks if chunk.key is not None)
        return pos, reused

    def _decode_vision_text(self, tokens, pos: int, n_batch: int) -> int:
        """Decode the prompt *tokens* at positions starting at *pos*, *n_batch* at
        a time, and return the position after them. A failed decode raises
        ``MtmdGpuEncodeFailed`` while the projector is on the GPU, else
        ``VisionInputError``."""
        from localm.inference.backends.base import VisionInputError

        from .mtmd import MtmdGpuEncodeFailed
        for i in range(0, len(tokens), n_batch):
            piece = list(tokens[i:i + n_batch])
            batch = self._create_batch(piece, pos, logits_at_last_only=True)
            try:
                ret = api.llama_decode(self._ctx_ptr, batch)
            finally:
                api.llama_batch_free(batch)
            if ret != 0:
                exc = MtmdGpuEncodeFailed if self._mtmd.on_gpu else VisionInputError
                raise exc(f"the image prompt could not be evaluated "
                          f"(llama_decode rc={ret})")
            pos += len(piece)
        return pos

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

    def _capture_h(self, row: int, pos: int) -> bool:
        """Copy the main context's next-n hidden state for batch row *row*, the
        token at *pos*, into _pending_h.

        The MTP head at a position consumes the hidden state from the position
        before it, so the draft that follows this decode needs the state this
        decode just produced. The pointer llama.cpp returns is into its own
        buffer and is overwritten by the next decode, hence the copy.
        """
        if not self._mtp_wants_h:
            return False
        ptr = api.llama_get_embeddings_nextn_ith(self._ctx_ptr, row)
        if not ptr:
            self._pending_h = None
            self._pending_h_pos = -1
            return False
        if self._h_buf is None:
            self._h_buf = (ctypes.c_float * self._n_embd)()
        ctypes.memmove(self._h_buf, ptr, self._n_embd * ctypes.sizeof(ctypes.c_float))
        self._pending_h = self._h_buf
        self._pending_h_pos = pos
        return True

    def _pending_h_addr(self, pos: int) -> Optional[int]:
        """The address of the hidden state for position pos - 1, or None when
        there is none (pos 0) or _pending_h holds some other position's."""
        if pos <= 0 or self._pending_h is None or self._pending_h_pos != pos - 1:
            return None
        return _address(self._pending_h)

    def _main_h_rows(self, n: int) -> List[Optional[int]]:
        """Addresses of the main context's next-n rows 0..n-1 of the last batch."""
        return api.llama_get_embeddings_nextn_rows(self._ctx_ptr, 0, n)

    def _decode_draft(self, tokens: List[int], pos0: int,
                      h_rows: List[Optional[int]], output_last: bool = True) -> int:
        """Decode *tokens* at positions pos0.. into the draft context.

        Row i carries the hidden state at address ``h_rows[i]``, or zeros for
        None. Only the last row produces an output, and only when *output_last*.
        ``llama_batch_init`` allocates token OR embd, never both, so embd comes
        from the library and the token array is attached here; the original
        pointer is restored before the batch is freed. Returns llama_decode's
        result.
        """
        n = len(tokens)
        row = self._n_embd * ctypes.sizeof(ctypes.c_float)
        batch = api.llama_batch_init(n, self._n_embd, 1)
        original_token = batch.token
        holder = (llama_token * n)(*tokens)
        try:
            batch.token = ctypes.cast(holder, ctypes.c_void_p)
            batch.n_tokens = n
            pos_p = ctypes.cast(batch.pos, ctypes.POINTER(ctypes.c_int32))
            n_seq_p = ctypes.cast(batch.n_seq_id, ctypes.POINTER(ctypes.c_int32))
            seq_p = ctypes.cast(batch.seq_id, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)))
            logits_p = ctypes.cast(batch.logits, ctypes.POINTER(ctypes.c_int8))
            embd = ctypes.cast(batch.embd, ctypes.c_void_p).value
            for i in range(n):
                pos_p[i] = pos0 + i
                n_seq_p[i] = 1
                seq_p[i][0] = 0
                logits_p[i] = 1 if (output_last and i == n - 1) else 0
                src = h_rows[i]
                if src:
                    ctypes.memmove(embd + i * row, src, row)
                else:
                    ctypes.memset(embd + i * row, 0, row)
            return api.llama_decode(self._mtp_ctx_ptr, batch)
        finally:
            batch.token = original_token
            api.llama_batch_free(batch)

    def _draft_tracking(self) -> bool:
        """Whether the draft cache is being kept in step with the main cache."""
        return (self._mtp_ctx_ptr is not None and self._mtp_usable
                and self._mtp_wants_h and not self._mtp_draft_stale)

    def _queued_row_addrs(self) -> List[int]:
        """Addresses of the queued hidden-state rows, in queue order."""
        row = self._n_embd * ctypes.sizeof(ctypes.c_float)
        base = ctypes.addressof(self._queued_h) if self._queued_h is not None else 0
        return [base + i * row for i in range(len(self._queued_tokens))]

    def _queue_draft_rows(self, tokens: List[int], h_rows: List[Optional[int]]) -> bool:
        """Queue *tokens*, the positions right after what the draft cache holds,
        with their hidden-state rows for the next draft decode.

        The rows are copied, since they point into buffers the next decode
        overwrites. A full queue is decoded on its own first. Returns False when
        that decode fails, which stops drafting for this call.
        """
        if len(self._queued_tokens) + len(tokens) > _MTP_QUEUED_ROWS_MAX:
            if not self._flush_queued_rows():
                return False
        if self._queued_h is None:
            self._queued_h = (ctypes.c_float * (self._n_embd * _MTP_QUEUED_ROWS_MAX))()
        queued = list(self._queued_tokens)
        row = self._n_embd * ctypes.sizeof(ctypes.c_float)
        base = ctypes.addressof(self._queued_h)
        for token, src in zip(tokens, h_rows):
            dst = base + len(queued) * row
            if src:
                ctypes.memmove(dst, src, row)
            else:
                ctypes.memset(dst, 0, row)
            queued.append(token)
        self._queued_tokens = queued
        return True

    def _flush_queued_rows(self) -> bool:
        """Decode the queued rows into the draft cache, with no output.

        Returns False, and stops drafting for this call, when the decode fails.
        """
        n = len(self._queued_tokens)
        if not n:
            return True
        try:
            ret = self._decode_draft(list(self._queued_tokens), self._draft_pos,
                                     self._queued_row_addrs(), output_last=False)
        except Exception as exc:
            self._stop_drafting_this_call("draft-catchup-error:%s" % type(exc).__name__)
            return False
        if ret != 0:
            self._stop_drafting_this_call("draft-catchup-failed:%d" % ret)
            return False
        self._draft_pos += n
        self._queued_tokens = []
        return True

    def _finish_draft_tracking(self) -> None:
        """At the end of a generation, decode the queued rows so the draft cache
        holds everything the main cache does."""
        if (self._draft_tracking() and self._queued_tokens
                and self._draft_pos + len(self._queued_tokens) == len(self._cached_tokens)):
            self._flush_queued_rows()

    def _after_main_token(self, token: int, pos: int) -> None:
        """Record a token the main context just decoded alone at *pos*.

        Queues it for the draft cache, paired with the hidden state of the
        position before it, unless a draft step already put it there, and keeps
        its own hidden state for the next draft. While the pacer has drafting
        paused in a call that drafts, nothing is recorded; the first token after
        the pause first flushes the rows queued before it and mirrors the
        skipped positions, the first of them with the hidden state held from
        before the pause and the rest with zeros. A failing mirror, or a draft
        cache that is past *pos* by more than one token, stops drafting for this
        call.
        """
        pacer = self._draft_pacer
        if pacer is not None and pacer.paused and self._mtp_drafting:
            return
        if self._draft_tracking():
            covered = self._draft_pos + len(self._queued_tokens)
            if covered < pos and self._flush_queued_rows():
                self._prefill_mtp(self._cached_tokens[covered:pos], covered, mid_reply=True)
                covered = self._draft_pos
            if covered == pos:
                self._queue_draft_rows([token], [self._pending_h_addr(pos)])
            elif covered != pos + 1 and self._draft_tracking():
                self._stop_drafting_this_call("draft-out-of-step")
        self._capture_h(0, pos)

    def _after_verify(self, accepted: List[int], pos: int) -> None:
        """Record a verification batch decoded at *pos* of which the drafts in
        *accepted* were kept.

        Accepted draft j, at pos + j, is queued for the draft cache with the
        hidden state of verification row j - 1, and row ``len(accepted)``
        becomes the state the next draft reads.
        """
        n = len(accepted)
        if n and self._draft_tracking():
            self._queue_draft_rows(accepted, self._main_h_rows(n))
        self._capture_h(n, pos + n)

    def _draft_source(self) -> DraftSource:
        """The draft source the decode loop drives for this model: an
        NgramSource when the configured source is ngram, else the MtpSource,
        which drafts only when an MTP draft context exists."""
        if self._source is None:
            if self._spec_source_name == SPEC_NGRAM:
                self._source = NgramSource(self, self._ngram_draft_max or 1)
            else:
                self._source = MtpSource(self)
        return self._source

    def speculation_report(self) -> dict:
        """The configured draft source as ``source`` plus its
        ``DraftSource.report()``: the model's speculation state and the
        figures of the reply that finished last."""
        return {"source": self._spec_source_name, **self._draft_source().report()}

    def _spec_rollback_wanted(self) -> bool:
        """Whether contexts keep recurrent-state snapshots for rejected drafts:
        True while the configured source drafts (mtp or ngram)."""
        return self._spec_source_name == SPEC_NGRAM or self._mtp_enabled

    def _spec_rollback_snapshots(self, cp) -> int:
        """Recurrent-state snapshots a context keeps so a step whose drafts are
        all rejected can still be rolled back, for the configured source."""
        if self._spec_source_name == SPEC_NGRAM:
            return ngram_rs_seq(getattr(cp, "n_rs_seq", 0), self._ngram_draft_max)
        return self._mtp_rollback_snapshots(cp)

    def _apply_initial_spec_params(self, cp, spec_draft_tokens: Optional[int]) -> None:
        """Set the n-gram draft cap for the loaded model, then the recurrent
        snapshots the first context keeps for the configured source."""
        if self._spec_source_name == SPEC_NGRAM:
            self._ngram_draft_max = ngram_draft_cap(
                spec_draft_tokens, self._model_has_recurrent_layers() is not False)
        if self._spec_rollback_wanted() and hasattr(cp, "n_rs_seq"):
            cp.n_rs_seq = self._spec_rollback_snapshots(cp)

    def _model_has_recurrent_layers(self) -> Optional[bool]:
        """Whether the loaded model has recurrent layers (fully recurrent or
        hybrid); None when the runtime cannot say. A caller sizing recurrent
        snapshots treats None as recurrent, which the VRAM estimate also
        assumes."""
        try:
            if not api.has_hybrid_api():
                return None
            return bool(api.llama_model_is_recurrent(self._model_ptr)
                        or api.llama_model_is_hybrid(self._model_ptr))
        except Exception as exc:
            from localm.debuglog import logger
            logger.debug("recurrent-layer probe failed (%s); capping n-gram "
                         "drafts as for a recurrent model", type(exc).__name__)
            return None

    def _mtp_draft_budget(self, pos: int, tokens_left: Optional[int]) -> int:
        """How many drafts the step at *pos* may propose: the configured count,
        capped by the tokens left in the reply (None for no limit) and by room
        in both caches for the verification batch."""
        n = self._mtp_draft_max
        if tokens_left is not None:
            n = min(n, tokens_left)
        room = self._ctx_capacity - pos - 1
        if self._mtp_ctx_capacity:
            room = min(room, self._mtp_ctx_capacity - pos - 1)
        return max(0, min(n, room))

    def _mtp_rollback_snapshots(self, cp) -> int:
        """Recurrent-state snapshots a context needs so a step whose drafts are
        all rejected can still be rolled back."""
        return mtp_rs_seq(getattr(cp, "n_rs_seq", 0), self._mtp_draft_max)

    def _propose_drafts(self, token: int, pos: int, n_max: int, draft_sampler) -> List[int]:
        """Propose up to *n_max* tokens to follow *token* at *pos*.

        One draft decode carries the queued rows and *token* paired with the
        hidden state of pos - 1; each further draft is decoded with the draft
        head's own next-n row. Drafting stops at an end-of-generation token,
        which is never proposed. The draft cache ends holding positions up to
        and including *pos*. Caller holds _gen_lock. A failure stops drafting
        for this call and returns [].
        """
        if self._draft_pos + len(self._queued_tokens) != pos:
            self._stop_drafting_this_call("draft-out-of-step")
            return []
        tokens = list(self._queued_tokens) + [token]
        h_rows = self._queued_row_addrs() + [self._pending_h_addr(pos)]
        ret = self._decode_draft(tokens, self._draft_pos, h_rows)
        if ret != 0:
            self._stop_drafting_this_call("draft-decode-failed:%d" % ret)
            return []
        self._draft_pos = pos + 1
        self._queued_tokens = []
        drafts: List[int] = []
        row = len(tokens) - 1
        extended = False
        while True:
            draft = api.llama_sampler_sample(draft_sampler, self._mtp_ctx_ptr, row)
            if self._tokenizer.is_eog(draft):
                break
            drafts.append(draft)
            if len(drafts) >= n_max:
                break
            h = _address(api.llama_get_embeddings_nextn_ith(self._mtp_ctx_ptr, row))
            if h is None:
                break
            ret = self._decode_draft([draft], pos + len(drafts), [h])
            extended = True
            if ret != 0:
                self._stop_drafting_this_call("draft-decode-failed:%d" % ret)
                return []
            row = 0
        if extended and not api.llama_memory_seq_rm(
                api.llama_get_memory(self._mtp_ctx_ptr), 0, pos + 1, -1):
            self._stop_drafting_this_call("draft-trim-failed")
            return []
        return drafts

    def _create_mtp_context(self, n_ctx: int, offload_kqv: bool = True,
                            n_threads: Optional[int] = None, quiet=None) -> str:
        """Create the MTP draft context at *n_ctx* tokens and wire the hidden-state
        exchange between it and the main context.

        The draft context is sized like the main one, so a position the main
        context can hold is one the draft context can hold. Returns "" on
        success, otherwise the status naming why no draft context exists:
        "no-mtp-graph", "no-ctx-type-field", "context-refused" or
        "hidden-state-refused".
        """
        if not api.llama_model_mtp_support(self._model_ptr)[0]:
            return "no-mtp-graph"
        cp_mtp = api.llama_context_default_params()
        if not hasattr(cp_mtp, "ctx_type"):
            # Without ctx_type this build cannot be ASKED for an MTP context, so
            # llama_init_from_model would hand back a second ordinary decoder
            # with its own uncharged KV cache.
            return "no-ctx-type-field"
        from ._structs import LLAMA_CONTEXT_TYPE_MTP
        cp_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP
        cp_mtp.n_ctx = n_ctx
        cp_mtp.n_batch = min(n_ctx, _PREFILL_CHUNK)
        cp_mtp.n_ubatch = cp_mtp.n_batch
        cp_mtp.offload_kqv = offload_kqv
        if n_threads is not None:
            cp_mtp.n_threads = n_threads
            cp_mtp.n_threads_batch = n_threads
        if quiet is None:
            quiet = contextlib.nullcontext
        with quiet():
            self._mtp_ctx_ptr = api.llama_init_from_model(self._model_ptr, cp_mtp)
        if not self._mtp_ctx_ptr:
            self._mtp_ctx_ptr = None
            self._mtp_ctx_capacity = 0
            return "context-refused"
        self._mtp_ctx_capacity = cp_mtp.n_ctx
        self._n_embd = api.llama_model_n_embd(self._model_ptr)
        # The target exposes its hidden state; the draft consumes it masked. Both
        # must take, or the head is starved and drafting is worse than not
        # drafting.
        exposed = api.llama_set_embeddings_nextn(self._ctx_ptr, True, False)
        consumed = api.llama_set_embeddings_nextn(self._mtp_ctx_ptr, True, True)
        self._mtp_wants_h = bool(exposed and consumed and self._n_embd > 0)
        if not self._mtp_wants_h:
            api.llama_free(self._mtp_ctx_ptr)
            self._mtp_ctx_ptr = None
            self._mtp_ctx_capacity = 0
            return "hidden-state-refused"
        self._draft_pos = 0
        self._queued_tokens = []
        self._attach_backend_draft_sampler()
        return ""

    def _attach_backend_draft_sampler(self) -> None:
        """Have the draft context pick its greedy draft inside llama_decode.

        The chain is attached to sequence 0 of the draft context and freed with
        it. A runtime that cannot run it on the backend keeps sampling drafts on
        the CPU, which gives the same tokens.
        """
        self._mtp_backend_chain = None
        if not api.has_backend_sampling():
            return
        chain = _greedy_chain()
        if api.llama_set_sampler(self._mtp_ctx_ptr, 0, chain):
            self._mtp_backend_chain = chain
            return
        api.llama_sampler_free(chain)
        from localm.debuglog import logger as _dbg
        _dbg.debug("MTP: the draft sampler runs on the CPU (backend sampling refused)")

    def _free_backend_draft_sampler(self) -> None:
        """Free the draft context's backend sampler chain; call after freeing the context."""
        chain = self._mtp_backend_chain
        self._mtp_backend_chain = None
        if chain is not None:
            api.llama_sampler_free(chain)

    def _rebuild_mtp_context(self, n_ctx: int, offload_kqv: bool) -> None:
        """Replace the draft context with one sized to the freshly created main
        context, which starts with an empty KV cache like the draft one.

        Called right after the main context is recreated and before anything is
        decoded into it, so the main context exposes its hidden state from its
        first decode. A draft context that cannot be recreated stops
        speculation for the model with the status naming why.
        """
        if self._mtp_ctx_ptr is not None:
            api.llama_free(self._mtp_ctx_ptr)
            self._mtp_ctx_ptr = None
        self._free_backend_draft_sampler()
        self._mtp_ctx_capacity = 0
        self._mtp_draft_stale = False
        self._draft_pos = 0
        self._queued_tokens = []
        failure = self._create_mtp_context(n_ctx, offload_kqv, self._n_threads)
        if failure:
            self._disable_mtp(failure, "the draft context could not be recreated "
                                       "at %d tokens (%s)" % (n_ctx, failure))

    def _stop_drafting_this_call(self, status: str) -> None:
        """Stop speculating for the rest of this generation and record why.

        The model keeps its speculation capability: the next request drafts
        again. The draft cache may now miss tokens the main cache holds, so the
        next prefill refills it from the whole prompt.
        """
        self.mtp_active_this_call = False
        self._mtp_drafting = False
        self.mtp_call_status = status
        self._mtp_draft_stale = True
        self._queued_tokens = []
        from localm.debuglog import logger as _dbg
        _dbg.info("MTP: speculation stopped for this reply - %s", status)

    def _disable_mtp(self, status: str, detail: str) -> None:
        """Turn speculation off for the rest of this model's life, and say why."""
        self._mtp_usable = False
        self.supports_mtp = False
        self.mtp_status = status
        from localm.debuglog import logger as _dbg
        _dbg.info("MTP: speculation disabled - %s", detail)

    def _prefill_mtp(self, tokens: List[int], base_pos: int,
                     h_rows: Optional[List[Optional[int]]] = None,
                     mid_reply: bool = False) -> None:
        """Mirror prefilled *tokens*, at base_pos.., into the MTP draft cache.

        Token i is paired with the hidden state at ``h_rows[i]``, which belongs
        to the position before it. Without *h_rows* the first token gets the
        held state for base_pos - 1 when there is one and every other token gets
        zeros. No row produces an output. On success the draft cache holds
        positions up to base_pos + len(tokens).

        The draft context is recreated at the main context's size whenever the
        main one is, so a position the main cache holds fits the draft cache. A
        prompt that still would not fit stops speculation here rather than
        paying a failing decode per token. A draft decode that fails leaves the
        draft cache out of step with the main one, which is the same dead end;
        with *mid_reply* it stops drafting for this call only, and the next
        prefill refills the draft cache.
        """
        if not tokens:
            return
        cap = self._mtp_ctx_capacity
        if cap and base_pos + len(tokens) > cap:
            self._disable_mtp(
                "draft-context-full",
                "the conversation outgrew the %d-token draft context" % cap)
            return
        if h_rows is None:
            h_rows = [self._pending_h_addr(base_pos)] + [None] * (len(tokens) - 1)
        for i in range(0, len(tokens), _PREFILL_CHUNK):
            piece = tokens[i:i + _PREFILL_CHUNK]
            try:
                ret = self._decode_draft(piece, base_pos + i,
                                         h_rows[i:i + _PREFILL_CHUNK], output_last=False)
            except Exception as exc:
                if mid_reply:
                    self._stop_drafting_this_call("draft-catchup-error:%s" % type(exc).__name__)
                else:
                    self._disable_mtp("draft-prefill-error:%s" % type(exc).__name__,
                                      "the draft prefill raised %s" % type(exc).__name__)
                return
            if ret != 0:
                if mid_reply:
                    self._stop_drafting_this_call("draft-catchup-failed:%d" % ret)
                else:
                    self._disable_mtp("draft-prefill-failed:%d" % ret,
                                      "the draft prefill returned %d" % ret)
                return
            self._draft_pos = base_pos + i + len(piece)

    def _after_prefill_chunk(self, chunk: List[int], base: int) -> None:
        """Pair a main prefill chunk just decoded at *base* with its hidden states.

        Mirrors the chunk into the draft cache, each token paired with the
        hidden state of the position before it, and keeps the chunk's last row
        for the first draft.
        """
        n = len(chunk)
        if not n or not self._mtp_wants_h:
            return
        if self._draft_tracking() and self._draft_pos == base:
            rows = [self._pending_h_addr(base)] + (self._main_h_rows(n - 1) if n > 1 else [])
            self._prefill_mtp(chunk, base, rows)
        self._capture_h(n - 1, base + n - 1)

    def _sync_draft_cache(self, prefix: int, tokens: List[int]) -> None:
        """Make the draft cache hold exactly tokens[:prefix] before a suffix
        prefill.

        Keeps what it already shares with the main cache and drops the rest; a
        stale draft cache is cleared. Positions the main cache keeps but the
        draft cache lacks are mirrored without hidden states, since the main
        context does not decode them again. A draft cache whose state cannot be
        established stops speculation for the model.
        """
        self._queued_tokens = []
        if self._pending_h_pos >= prefix:
            self._pending_h_pos = -1
        if self._mtp_ctx_ptr is None or not self._mtp_usable:
            return
        keep = 0 if self._mtp_draft_stale else min(prefix, self._draft_pos)
        self._mtp_draft_stale = False
        try:
            mem_mtp = api.llama_get_memory(self._mtp_ctx_ptr)
            if keep == 0:
                api.llama_memory_clear(mem_mtp, True)
            elif not api.llama_memory_seq_rm(mem_mtp, 0, keep, -1):
                api.llama_memory_clear(mem_mtp, True)
                keep = 0
        except Exception as exc:
            self._disable_mtp(
                "draft-trim-error:%s" % type(exc).__name__,
                "trimming the draft cache raised %s" % type(exc).__name__)
            return
        self._draft_pos = keep
        if keep < prefix:
            self._prefill_mtp(tokens[keep:prefix], keep)

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
        self._vision_kv = None
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
        # removing a range: recurrent state cannot be partially rewound,
        # so a range removal can leave it stale.
        if prefix == 0:
            api.llama_memory_clear(mem, True)
        elif prefix < len(self._cached_tokens) or not self._cached_tokens:
            if not api.llama_memory_seq_rm(mem, 0, prefix, -1):
                # Partial removal unsupported (e.g. SWA cache / recurrent state) - start over
                api.llama_memory_clear(mem, True)
                prefix = 0
        # The suffix below is mirrored into the draft cache at prefix + i, so the
        # draft cache has to end at prefix too.
        self._sync_draft_cache(prefix, prompt_tokens)

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
            if ret == 0:
                self._after_prefill_chunk(chunk, prefix + i)
                continue
            # If partial reuse failed (e.g. a recurrent state conflict),
            # perform a full clean prefill from position 0
            self._cached_tokens = []
            api.llama_memory_clear(mem, True)
            self._sync_draft_cache(0, prompt_tokens)
            if prefix > 0:
                for j in range(0, len(prompt_tokens), _PREFILL_CHUNK):
                    full_chunk = prompt_tokens[j:j + _PREFILL_CHUNK]
                    full_batch = self._create_batch(full_chunk, j, logits_at_last_only=True)
                    full_ret = api.llama_decode(self._ctx_ptr, full_batch)
                    api.llama_batch_free(full_batch)
                    if full_ret != 0:
                        raise RuntimeError(f"llama_decode failed during prefill (code {full_ret})")
                    self._after_prefill_chunk(full_chunk, j)
                break
            else:
                raise RuntimeError(f"llama_decode failed during prefill (code {ret})")

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
        self._pending_h_pos = -1
        had_draft_context = self._mtp_ctx_ptr is not None
        if had_draft_context:
            api.llama_free(self._mtp_ctx_ptr)
            self._mtp_ctx_ptr = None
            self._free_backend_draft_sampler()
            self._mtp_ctx_capacity = 0

        cp = api.llama_context_default_params()
        cp.n_ctx       = target
        cp.n_batch     = min(cp.n_ctx, 2048)
        cp.n_ubatch    = cp.n_batch   # micro-batch must match so prefill fits in one call
        cp.offload_kqv = offload_kqv  # False -> KV cache in system RAM (VRAM was tight)
        # The grown context must keep the rollback snapshots too, or speculation
        # stops working the moment a conversation outgrows its first context.
        if self._spec_rollback_wanted() and hasattr(cp, "n_rs_seq"):
            cp.n_rs_seq = self._spec_rollback_snapshots(cp)

        self._ctx_ptr = api.llama_init_from_model(self._model_ptr, cp)
        if not self._ctx_ptr:
            if had_draft_context:
                self._disable_mtp("context-refused",
                                  "the main context could not be recreated at %d tokens" % target)
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
        if had_draft_context and self._mtp_usable:
            self._rebuild_mtp_context(cp.n_ctx, offload_kqv)

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
            self._after_prefill_chunk(chunk, i)

        self._cached_tokens = list(prompt_tokens)

    def _reset_kv_for_image(self) -> None:
        """Empty the KV cache (and the MTP draft cache) so an image prefill can
        start at position 0 on a REUSED context. Uses the memory API when
        present, else recreates an empty context (older builds)."""
        self._cached_tokens = []
        self._pending_h_pos = -1
        self._queued_tokens = []
        if self._memory_api_available():
            try:
                mem = api.llama_get_memory(self._ctx_ptr)
                api.llama_memory_clear(mem, True)
                if self._mtp_ctx_ptr is not None and self._mtp_usable:
                    try:
                        mem_mtp = api.llama_get_memory(self._mtp_ctx_ptr)
                        api.llama_memory_clear(mem_mtp, True)
                        self._draft_pos = 0
                    except Exception as exc:
                        # The next text prefill clears a stale draft cache again.
                        self._mtp_draft_stale = True
                        from localm.debuglog import logger as _dbg
                        _dbg.info("MTP: clearing the draft cache for an image turn raised %s",
                                  type(exc).__name__)
                return
            except Exception:
                pass
        self._prefill_fresh_context([], self._ctx_capacity)   # empty, same size

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
        on_status: Optional[Callable[[str], None]] = None,
        thinking: Optional[bool] = None,
        **_ignored,
    ):
        """
        Generate a chat completion.

        With ``stream=True`` yields dicts matching the llama-cpp-python
        streaming format:
            {"choices": [{"delta": {"content": "<token>"}}]}

        With ``stream=False`` returns a single completion dict.

        ``thinking=False`` starts a text-only reply with an empty reasoning
        block (``no_think_prompt``); an image request is unaffected.

        An encoder-decoder model generates through
        ``_generate_encoder_decoder``; ``thinking`` does not apply to it and is
        ignored.
        """
        if self.is_encoder_decoder:
            return self._completion_result(self._generate_encoder_decoder(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                grammar=grammar,
                grammar_lazy=grammar_lazy,
                grammar_triggers=grammar_triggers,
                seed=seed,
                on_status=on_status,
            ), stream)

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
        from localm.inference.backends.base import messages_contain_image
        going_to_vision = (getattr(self, "_mtmd", None) is not None
                           and messages_contain_image(messages))
        if going_to_vision:
            untrusted_ranges = ()
            if any(untrusted_spans_of(m.get("content")) for m in messages):
                from localm.debuglog import logger
                logger.warning(
                    "textguard: this request takes the vision path, which "
                    "tokenises the whole prompt through mtmd in one call, so "
                    "untrusted spans keep special-token parsing ON; only the "
                    "text-level defang applies to this request")
        else:
            untrusted_ranges = _untrusted_prompt_ranges(
                self._model_ptr, messages, prompt, fallback_reason)
        if thinking is False and not going_to_vision:
            from localm.inference.backends.base import no_think_prompt
            prompt = no_think_prompt(
                prompt, api.llama_model_chat_template(self._model_ptr))
        tokens = self._tokenizer.encode(
            prompt, add_bos=add_bos, untrusted_ranges=untrusted_ranges)

        if going_to_vision:
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
                on_status=on_status,
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
                on_status=on_status,
            )

        return self._completion_result(gen, stream)

    def _completion_result(self, gen: Iterator[int], stream: bool):
        """The ``create_chat_completion`` result for the token generator *gen*:
        the streaming chunk generator when *stream*, else the whole completion
        dict."""
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
