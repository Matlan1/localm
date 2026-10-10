# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OpenAI request fields localm does not serve, refused by name.

A field that would change what the client gets back (a second choice, token
log probabilities, audio, a deprecated function-call shape) is refused with a
message naming it rather than ignored. Fields that are hints only (``user``,
``metadata``, ``store``, ``service_tier``, ``prediction``, ``verbosity``,
``prompt_cache_key``, ``safety_identifier``) are accepted and have no effect.

Pure Python: no engine, no web framework."""

from __future__ import annotations

from typing import Any, Optional

from localm.inference.protocol import ChatRequest, CompletionRequest


class UnsupportedFieldError(ValueError):
    """A request field localm cannot honour; the message names it."""


def _check_common(req: ChatRequest | CompletionRequest) -> None:
    if req.n is not None and req.n != 1:
        raise UnsupportedFieldError(
            f"n={req.n} is not supported: localm returns one choice per request; "
            "send n=1 or omit it")
    if req.logit_bias:
        raise UnsupportedFieldError(
            "logit_bias is not supported: token ids differ between models; omit it")
    if req.stream_options is not None:
        opts = req.stream_options
        if not isinstance(opts, dict):
            raise UnsupportedFieldError("stream_options must be an object")
        include = opts.get("include_usage")
        if include is not None and not isinstance(include, bool):
            raise UnsupportedFieldError("stream_options.include_usage must be a boolean")


def check_chat_request(req: ChatRequest) -> None:
    """Raise :class:`UnsupportedFieldError` when *req* sets a field localm does
    not serve."""
    _check_common(req)
    if req.functions is not None or req.function_call is not None:
        raise UnsupportedFieldError(
            "functions and function_call are not supported: send tools and "
            "tool_choice instead")
    if req.logprobs or req.top_logprobs:
        raise UnsupportedFieldError(
            "logprobs and top_logprobs are not supported; omit them")
    if req.audio is not None:
        raise UnsupportedFieldError("audio output is not supported; omit audio")
    if req.modalities is not None:
        mods = req.modalities
        if not isinstance(mods, list) or any(m != "text" for m in mods):
            raise UnsupportedFieldError(
                f"modalities {mods!r} is not supported: only [\"text\"] is served")
    if req.web_search_options is not None:
        raise UnsupportedFieldError(
            "web_search_options is not supported: localm does not search the web "
            "for API requests")
    if (req.max_tokens is not None and req.max_completion_tokens is not None
            and req.max_tokens != req.max_completion_tokens):
        raise UnsupportedFieldError(
            "max_tokens and max_completion_tokens disagree; send one of them")


def check_completion_request(req: CompletionRequest) -> str:
    """The one prompt *req* carries. Raises :class:`UnsupportedFieldError` when
    *req* sets a field localm does not serve."""
    _check_common(req)
    if req.best_of is not None and req.best_of != 1:
        raise UnsupportedFieldError(
            f"best_of={req.best_of} is not supported: send best_of=1 or omit it")
    if req.logprobs is not None:
        raise UnsupportedFieldError("logprobs is not supported; omit it")
    if req.suffix:
        raise UnsupportedFieldError("suffix (fill-in-the-middle) is not supported")
    prompt: Any = req.prompt
    if isinstance(prompt, list):
        if len(prompt) != 1 or not isinstance(prompt[0], str):
            raise UnsupportedFieldError(
                "prompt must be one string (or a list holding one string); a batch "
                "of prompts or token ids is not supported")
        prompt = prompt[0]
    return prompt


def max_tokens_of(req: ChatRequest) -> Optional[int]:
    """The reply cap *req* asks for, from ``max_completion_tokens`` or
    ``max_tokens``."""
    return req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens


def include_usage(req: ChatRequest | CompletionRequest) -> bool:
    """Whether a streamed reply ends with a usage-only chunk."""
    opts = req.stream_options
    return bool(req.stream and isinstance(opts, dict) and opts.get("include_usage"))


def thinking_of(req: ChatRequest) -> Optional[bool]:
    """``enable_thinking`` for *req*: ``chat_template_kwargs.enable_thinking``
    when set, else False for ``reasoning_effort: "none"``, else ``None``."""
    flag = (req.chat_template_kwargs or {}).get("enable_thinking")
    if flag is not None:
        return flag
    if isinstance(req.reasoning_effort, str) and req.reasoning_effort.strip().lower() == "none":
        return False
    return None
