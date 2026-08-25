# SPDX-License-Identifier: AGPL-3.0-or-later
"""HTTP backend - connects to any OpenAI-compatible inference endpoint."""

from __future__ import annotations

import os
import time
from typing import Callable, Iterator, Optional

import requests

from .base import BaseLLMBackend

# Retry policy for rate limits and transient server errors (mainly relevant
# for the opt-in cloud providers; a local server never returns 429).
_RETRY_STATUSES = {429, 500, 502, 503, 529}
_MAX_RETRIES = 4
_BACKOFF_BASE_S = 2.0

# 503 from OUR OWN local server is never retried at all (#964) - unlike
# 429/500/502/529, which keep the full budget above unchanged (a cloud
# provider's genuine rate-limit/overload signal). This is not "we cannot
# tell 503 apart from a real error", it is TRACED: every 503 reachable from
# this backend's own call path (/v1/chat/completions, /v1/completions, both
# resolve their engine via inference/http_server.py's get_engine(), which
# always calls switch_engine(..., preempt=False)) is either deterministic or
# has already exhausted its own resolution window server-side before the
# response ever reaches this client:
#   - "Model load was superseded by a newer request" (http_server.py) is
#     UNREACHABLE here: superseding only cancels a preempt=True load (an
#     explicit GUI/CLI model switch), and switch_engine's own comment says so
#     plainly - "API-routed loads (preempt=False) run to completion, never
#     cancelled by a concurrent different-model load". A preempt=False
#     load's cancel token is created but never wired to the global one that
#     .set() actually triggers, so ModelLoadCancelled cannot fire for it.
#   - A VRAM-refusal 503 already waited on wait_for_vram_release() (vram.py,
#     5s default) SERVER-SIDE before refusing - a client retry on top adds
#     latency without improving the odds, since the server already tried.
#   - "No model loaded"/"No engine initialised" need an explicit user action
#     (load a model), not a passing few seconds.
#   - The grammar-check/embedding worker-fault 503 (inference/routes/chat.py,
#     #991) is a deterministic isolated-worker crash that will almost
#     certainly recur on the identical retried request.
# Before this, EVERY local 503 got the full 4-retry/~30s budget, so the
# deterministic case cost the user up to 30s of blind waiting before an
# already-informative error message (see CoderServerError/_raise_for_status
# above) ever reached them, for zero real chance of success. A NON-local
# OpenAI-compatible endpoint (cloud or otherwise) is unaffected - its 503
# causes were never traced here and keep the existing behaviour.


class CoderAuthError(RuntimeError):
    """The inference server rejected the request for auth reasons (401/403)."""


class CoderServerError(RuntimeError):
    """A non-2xx response the server explained (a FastAPI ``{'detail': '...'}`` body, or plain text), folded into the message - unlike ``resp.raise_for_status()``, which only ever reports the status line ('503 Server Error: Service Unavailable for url: ...') and never reads the body, so a server-side diagn..."""


def _response_detail(resp) -> str:
    """Best-effort server-provided detail text from *resp*, or '' if none is extractable."""
    try:
        data = resp.json()
    except Exception:
        data = None
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()[:500]
    try:
        text = resp.text
    except Exception:
        return ""
    if isinstance(text, str) and text.strip():
        return text.strip()[:500]
    return ""


def _raise_for_status(resp) -> None:
    """Turn a 401/403 into a CoderAuthError whose message tells the user how to supply an API key."""
    if resp.status_code in (401, 403):
        raise CoderAuthError(
            f"Authentication failed (HTTP {resp.status_code}) for {resp.url}. "
            "This server requires an API key. Run `localm key show --reveal` to "
            "view the current key (or `localm key generate` to mint one), then "
            "pass it with `--api-key <key>` or set the LOCALM_API_KEY environment "
            "variable."
        )
    if resp.status_code < 400:
        return
    detail = _response_detail(resp)
    if detail:
        raise CoderServerError(
            f"HTTP {resp.status_code} error from {resp.url}: {detail}")
    resp.raise_for_status()


def _retry_delay(response, attempt: int) -> float:
    """Honour Retry-After when present, else exponential backoff."""
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return min(float(retry_after), 120.0)
    return min(_BACKOFF_BASE_S * (2 ** attempt), 60.0)


def _post_with_retry(url: str, *, headers: dict, json_body: dict,
                     timeout: int, stream: bool = False,
                     verify=True, retry_503: bool = True) -> requests.Response:
    """POST with retry on 429/5xx."""
    last = None
    for attempt in range(_MAX_RETRIES + 1):
        resp = requests.post(url, headers=headers, json=json_body,
                             timeout=timeout, stream=stream, verify=verify)
        if resp.status_code not in _RETRY_STATUSES or attempt == _MAX_RETRIES:
            return resp
        if resp.status_code == 503 and not retry_503:
            return resp
        delay = _retry_delay(resp, attempt)
        resp.close()
        last = resp
        time.sleep(delay)
    return last  # unreachable, keeps type checkers happy


class HTTPBackend(BaseLLMBackend):
    """OpenAI /v1/chat/completions compatible backend."""

    # Anthropic requires max_tokens on every request; OpenAI-compat does not.
    _ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
    _ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "localm",
        timeout: int = 300,
        native_tools: bool = False,
        anthropic: bool = False,
        localm_server: bool = False,
        verify=None,
        **extra_params,
    ) -> None:
        self._base_url     = base_url.rstrip("/")
        self._model        = model
        self._api_key      = api_key
        self._timeout      = timeout
        self._extra        = extra_params
        # This backend's OWN local server, as opposed to a cloud/remote
        # OpenAI-compatible endpoint - gates the 503 retry carve-out (see the
        # comment above _RETRY_STATUSES). Stored directly rather than reusing
        # supports_grammar below: that flag is ALSO False whenever anthropic
        # is set, which would wrongly re-enable full 503 retries for the
        # (never actually constructed) localm_server=True, anthropic=True
        # combination.
        self._is_local_server = bool(localm_server)
        # TLS verification for the POST. A localm network bind serves HTTPS via
        # its own local CA, so a loopback self-call must trust that CA; external
        # HTTPS (OpenAI/Anthropic) keeps normal public verification. verify=None
        # derives this from base_url; an explicit value overrides (tests).
        if verify is None:
            from localm import tls
            verify = tls.requests_verify(base_url)
        self._verify = verify
        self._last_usage: dict = {}
        # Most recent call's reasoning text (H4 `reasoning_content`), the
        # non-streaming counterpart of chat_stream's on_reasoning callback - see
        # last_reasoning. Never mixed into chat()/chat_stream()'s returned/yielded
        # text (AUD-HIGH-17-3).
        self._last_reasoning: str = ""
        self.native_tools  = native_tools
        # Anthropic speaks the Messages API (/v1/messages, x-api-key,
        # anthropic-version, content-block responses) - NOT the OpenAI
        # /chat/completions + Bearer shape. When True, chat()/chat_stream()
        # use the Messages translation below.
        self.anthropic     = anthropic
        self._tool_defs: list = []   # OpenAI-format tool definitions

        # GBNF grammar sampling is only supported by our own local server, so
        # only a backend EXPLICITLY constructed for one advertises it. The old
        # blacklist (not api.openai.com/api.anthropic.com) mislabelled every
        # third-party OpenAI-compatible server (LM Studio, vLLM, a remote URL)
        # as grammar-capable, which mattered once the coder started sending
        # grammar kwargs BY DEFAULT - unknown body fields can 400 there. A
        # remote localm reached via a hand-typed --backend URL loses grammar
        # (conservative); the attach flow and self-connections pass the flag.
        self.supports_grammar = bool(localm_server) and not anthropic
        self._ctx_capacity_cached = False
        self._ctx_capacity: Optional[int] = None

    @property
    def supports_native_tools(self) -> bool:
        """Can the CONNECTED server actually honour ``native_tools``?"""
        return not self._is_local_server

    def context_capacity(self) -> Optional[int]:
        """The loaded model's RESOLVED context ceiling from the server's /v1/config (VRAM-derived under ctx_auto), cached after the first successful fetch."""
        if self._ctx_capacity_cached:
            return self._ctx_capacity
        self._ctx_capacity_cached = True
        try:
            resp = requests.get(f"{self._base_url}/config",
                                headers=self._headers(), timeout=15,
                                verify=self._verify)
            if resp.ok:
                v = resp.json().get("effective_ctx_max")
                if isinstance(v, int) and v > 0:
                    self._ctx_capacity = v
        except Exception:
            self._ctx_capacity = None
        return self._ctx_capacity

    @property
    def model_id(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        """Repoint this backend at a different model NAME, in place."""
        self._model = model

    @property
    def last_usage(self) -> dict:
        """Usage dict from the most recent call: {prompt_tokens, completion_tokens, total_tokens}."""
        return dict(self._last_usage)

    @property
    def last_reasoning(self) -> str:
        """The most recent call's full reasoning text (H4 ``reasoning_content``), or ``''`` when the model/server did not emit one. ``chat()`` populates this directly; ``chat_stream()`` accumulates it from the same deltas passed to ``on_reasoning`` as they arrive."""
        return self._last_reasoning

    def set_tools(self, tool_defs: list) -> None:
        """Register OpenAI-format tool definitions for native function calling."""
        self._tool_defs = tool_defs

    def _flush_tool_calls_as_xml(self, tc_buf: dict) -> Optional[str]:
        """The trailing '\\n' + XML block for accumulated streaming tool-call chunks (index -> {'name', 'arguments'}), or None if none were accumulated."""
        if not tc_buf:
            return None
        ordered = [tc_buf[i] for i in sorted(tc_buf)]
        xml = self._tool_calls_to_xml([
            {"function": {"name": t["name"], "arguments": t["arguments"]}}
            for t in ordered
        ])
        return "\n" + xml

    @staticmethod
    def _tool_calls_to_xml(tool_calls: list) -> str:
        """Convert an OpenAI tool_calls list to our internal XML format."""
        import json as _json
        parts = []
        for tc in tool_calls:
            fn   = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = _json.loads(fn.get("arguments", "{}") or "{}")
            except Exception:
                args = {}
            parts.append(
                f'<tool_call>\n{_json.dumps({"name": name, "args": args})}\n</tool_call>'
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------ #

    def _headers(self) -> dict:
        if self.anthropic:
            # Anthropic Messages API auth: x-api-key + anthropic-version,
            # never a Bearer Authorization header.
            return {
                "x-api-key": self._api_key,
                "anthropic-version": self._ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Client-Id": "coder",
        }

    def _chat_url(self) -> str:
        """The completion endpoint for this backend's protocol."""
        if self.anthropic:
            return f"{self._base_url}/messages"
        return f"{self._base_url}/chat/completions"

    def _body(self, messages: list[dict], stream: bool, **kwargs) -> dict:
        if self.anthropic:
            return self._anthropic_body(messages, stream, **kwargs)
        body = {
            "model":    self._model,
            "messages": messages,
            "stream":   stream,
            **self._extra,
            **kwargs,
        }
        if self.native_tools and self._tool_defs:
            body["tools"] = self._tool_defs
            body["tool_choice"] = "auto"
        return {k: v for k, v in body.items() if v is not None}

    def _anthropic_body(self, messages: list[dict], stream: bool, **kwargs) -> dict:
        """Translate an OpenAI-style message list into the Anthropic Messages API request shape:."""
        system_parts: list[str] = []
        convo: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                if content:
                    system_parts.append(str(content))
            else:
                convo.append(m)

        body: dict = {
            "model":      self._model,
            "messages":   convo,
            "stream":     stream,
            **self._extra,
            **kwargs,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if not body.get("max_tokens"):
            body["max_tokens"] = self._ANTHROPIC_DEFAULT_MAX_TOKENS
        if self.native_tools and self._tool_defs:
            # Anthropic's tool schema differs from OpenAI's nested form;
            # flatten {"function": {...}} entries to top-level name/schema.
            body["tools"] = [self._anthropic_tool(t) for t in self._tool_defs]
        return {k: v for k, v in body.items() if v is not None}

    @staticmethod
    def _anthropic_tool(tool_def: dict) -> dict:
        fn = tool_def.get("function", tool_def)
        return {
            "name":         fn.get("name", ""),
            "description":  fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        }

    def _parse_anthropic_response(self, data: dict) -> str:
        """Extract assistant text (and any tool_use blocks) from a Messages response."""
        usage = data.get("usage") or {}
        if usage:
            in_tok  = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            self._last_usage = {
                "prompt_tokens":     in_tok,
                "completion_tokens": out_tok,
                "total_tokens":      in_tok + out_tok,
            }
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in data.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                import json as _json
                tool_calls.append({
                    "function": {
                        "name":      block.get("name", ""),
                        "arguments": _json.dumps(block.get("input", {})),
                    }
                })
        text = "".join(text_parts)
        if tool_calls:
            xml = self._tool_calls_to_xml(tool_calls)
            text = (text + "\n" + xml).strip() if text else xml
        return text

    # ------------------------------------------------------------------ #

    def chat(self, messages: list[dict], **kwargs) -> str:
        self._last_usage = {}
        self._last_reasoning = ""
        resp = _post_with_retry(
            self._chat_url(),
            headers=self._headers(),
            json_body=self._body(messages, stream=False, **kwargs),
            timeout=self._timeout,
            verify=self._verify,
            retry_503=not self._is_local_server,
        )
        _raise_for_status(resp)
        data = resp.json()
        if self.anthropic:
            return self._parse_anthropic_response(data)
        if data.get("usage"):
            self._last_usage = data["usage"]
        message = data["choices"][0]["message"]
        text    = message.get("content") or ""
        # H4: the server splits a thinking model's reasoning into its own
        # `reasoning_content` field, kept OUT of `text` (AUD-HIGH-17-3) - unlike
        # HttpEngine's chat REPL, the coder loop has no downstream splitter, so
        # inlining it here would leak raw <think> tags into the visible answer,
        # the audit log, and conversation history. Callers that want it read
        # last_reasoning after this call returns.
        self._last_reasoning = message.get("reasoning_content") or ""
        # Native tool calls: convert to our XML format and append
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            xml = self._tool_calls_to_xml(tool_calls)
            text = (text + "\n" + xml).strip() if text else xml
        return text

    def chat_stream(self, messages: list[dict],
                    on_reasoning: Optional[Callable[[str], None]] = None,
                    **kwargs) -> Iterator[str]:
        if self.anthropic:
            # Anthropic extended-thinking events are a distinct shape this
            # backend does not translate; not in scope here (AUD-HIGH-17-3).
            yield from self._anthropic_stream(messages, **kwargs)
            return
        import json as _json
        self._last_usage = {}
        self._last_reasoning = ""
        _reasoning_parts: list[str] = []
        # Accumulate streaming native tool_calls: index → {name, arguments_buf}
        _tc_buf: dict[int, dict] = {}

        with _post_with_retry(
            self._chat_url(),
            headers=self._headers(),
            json_body=self._body(messages, stream=True, **kwargs),
            timeout=self._timeout,
            stream=True,
            verify=self._verify,
            retry_503=not self._is_local_server,
        ) as resp:
            _raise_for_status(resp)
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode() if isinstance(line, bytes) else line
                if text.startswith("data: "):
                    text = text[6:]
                if text in ("[DONE]", ""):
                    break
                try:
                    chunk = _json.loads(text)
                except Exception:
                    continue
                # Capture usage from the final stop chunk (sent by localm server)
                if chunk.get("usage"):
                    self._last_usage = chunk["usage"]
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                # H4 reasoning delta: routed to on_reasoning (a SEPARATE channel
                # from the yielded content), never yielded inline - see chat()'s
                # comment and BaseLLMBackend.chat_stream's docstring.
                reasoning = delta.get("reasoning_content") or ""
                if reasoning:
                    _reasoning_parts.append(reasoning)
                    if on_reasoning is not None:
                        try:
                            on_reasoning(reasoning)
                        except Exception:
                            pass  # a broken sink must not kill the stream
                # Regular content tokens
                piece = delta.get("content") or ""
                if piece:
                    yield piece
                # Accumulate native tool_call chunks
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in _tc_buf:
                        _tc_buf[idx] = {"name": "", "arguments": ""}
                    fn = tc_delta.get("function") or {}
                    if fn.get("name"):
                        _tc_buf[idx]["name"] += fn["name"]
                    if fn.get("arguments"):
                        _tc_buf[idx]["arguments"] += fn["arguments"]

        self._last_reasoning = "".join(_reasoning_parts)

        # Emit accumulated tool calls as XML after the stream ends
        flushed = self._flush_tool_calls_as_xml(_tc_buf)
        if flushed is not None:
            yield flushed

    def _anthropic_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """Stream from the Anthropic Messages API (SSE)."""
        import json as _json
        self._last_usage = {}
        in_tok = 0
        # index -> {"name": str, "arguments": str}
        _tc_buf: dict[int, dict] = {}

        with _post_with_retry(
            self._chat_url(),
            headers=self._headers(),
            json_body=self._body(messages, stream=True, **kwargs),
            timeout=self._timeout,
            stream=True,
            verify=self._verify,
            retry_503=not self._is_local_server,
        ) as resp:
            _raise_for_status(resp)
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode() if isinstance(line, bytes) else line
                if not text.startswith("data:"):
                    continue   # skip "event:" lines and blank separators
                text = text[len("data:"):].strip()
                if not text or text == "[DONE]":
                    continue
                try:
                    evt = _json.loads(text)
                except Exception:
                    continue
                etype = evt.get("type")
                if etype == "message_start":
                    usage = (evt.get("message") or {}).get("usage") or {}
                    in_tok = usage.get("input_tokens", 0)
                elif etype == "content_block_start":
                    block = evt.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        _tc_buf[evt.get("index", 0)] = {
                            "name": block.get("name", ""), "arguments": ""
                        }
                elif etype == "content_block_delta":
                    delta = evt.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        piece = delta.get("text") or ""
                        if piece:
                            yield piece
                    elif delta.get("type") == "input_json_delta":
                        idx = evt.get("index", 0)
                        _tc_buf.setdefault(idx, {"name": "", "arguments": ""})
                        _tc_buf[idx]["arguments"] += delta.get("partial_json", "")
                elif etype == "message_delta":
                    usage = evt.get("usage") or {}
                    out_tok = usage.get("output_tokens", 0)
                    self._last_usage = {
                        "prompt_tokens":     in_tok,
                        "completion_tokens": out_tok,
                        "total_tokens":      in_tok + out_tok,
                    }
                elif etype == "message_stop":
                    break

        flushed = self._flush_tool_calls_as_xml(_tc_buf)
        if flushed is not None:
            yield flushed


# ------------------------------------------------------------------ #
#  Convenience constructors
# ------------------------------------------------------------------ #

def make_localm_backend(model: str, port: int = 8642, *, host: str = "",
                        api_key: str = "", **kw) -> HTTPBackend:
    """Backend for a local ``localm serve``."""
    from localm.auth import get_api_key
    from localm.bindhost import self_connect_host, url_host
    key = api_key or get_api_key() or "localm"
    # *host* is the address the target server BOUND. Blank keeps the IPv4
    # loopback every pre-IPv6 caller expects; an IPv6-bound server needs its
    # own loopback, since it is not listening on 127.0.0.1 at all.
    _h = url_host(self_connect_host(host)) if host else "127.0.0.1"
    return HTTPBackend(f"http://{_h}:{port}/v1", model, api_key=key,
                       localm_server=True, **kw)


def make_openai_backend(model: str = "gpt-4o", **kw) -> HTTPBackend:
    key = os.environ.get("OPENAI_API_KEY", "")
    return HTTPBackend("https://api.openai.com/v1", model, api_key=key, native_tools=True, **kw)


def make_anthropic_backend(model: str = "claude-opus-4-5", **kw) -> HTTPBackend:
    """Anthropic Messages API backend."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return HTTPBackend(
        "https://api.anthropic.com/v1",
        model,
        api_key=key,
        anthropic=True,
        native_tools=True,
        **kw,
    )
