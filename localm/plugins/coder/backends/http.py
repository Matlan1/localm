# SPDX-License-Identifier: AGPL-3.0-or-later
"""
HTTP backend - connects to any OpenAI-compatible inference endpoint.

Covers:
  - localm serve (offline, local model)
  - OpenAI API          (online, requires OPENAI_API_KEY)
  - Anthropic Messages  (online, requires ANTHROPIC_API_KEY) via openai-compat shim
  - Ollama              (offline, localhost:11434)
  - LM Studio, llama.cpp server, etc.
"""

from __future__ import annotations

import os
import time
from typing import Iterator

import requests

from .base import BaseLLMBackend

# Retry policy for rate limits and transient server errors (mainly relevant
# for the opt-in cloud providers; a local server never returns 429).
_RETRY_STATUSES = {429, 500, 502, 503, 529}
_MAX_RETRIES = 4
_BACKOFF_BASE_S = 2.0


class CoderAuthError(RuntimeError):
    """The inference server rejected the request for auth reasons (401/403).

    Carries an actionable hint (how to find/set the API key) instead of the bare
    '401 Client Error' that requests.raise_for_status would raise.
    """


def _raise_for_status(resp) -> None:
    """Turn a 401/403 into a CoderAuthError whose message tells the user how to
    supply an API key; otherwise behave like resp.raise_for_status()."""
    if resp.status_code in (401, 403):
        raise CoderAuthError(
            f"Authentication failed (HTTP {resp.status_code}) for {resp.url}. "
            "This server requires an API key. Run `localm key show --reveal` to "
            "view the current key (or `localm key generate` to mint one), then "
            "pass it with `--api-key <key>` or set the LOCALM_API_KEY environment "
            "variable."
        )
    resp.raise_for_status()


def _retry_delay(response, attempt: int) -> float:
    """Honour Retry-After when present, else exponential backoff."""
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return min(float(retry_after), 120.0)
    return min(_BACKOFF_BASE_S * (2 ** attempt), 60.0)


def _post_with_retry(url: str, *, headers: dict, json_body: dict,
                     timeout: int, stream: bool = False,
                     verify=True) -> requests.Response:
    """POST with retry on 429/5xx. Returns the first non-retryable response."""
    last = None
    for attempt in range(_MAX_RETRIES + 1):
        resp = requests.post(url, headers=headers, json=json_body,
                             timeout=timeout, stream=stream, verify=verify)
        if resp.status_code not in _RETRY_STATUSES or attempt == _MAX_RETRIES:
            return resp
        delay = _retry_delay(resp, attempt)
        resp.close()
        last = resp
        time.sleep(delay)
    return last  # unreachable, keeps type checkers happy


class HTTPBackend(BaseLLMBackend):
    """
    OpenAI /v1/chat/completions compatible backend.

    Parameters
    ----------
    base_url:
        Root URL of the API, e.g. ``http://127.0.0.1:8080/v1``.
    model:
        Model identifier to pass in the request body.
    api_key:
        Bearer token (use ``"localm"`` or any placeholder for local servers).
    timeout:
        Per-request timeout in seconds.
    extra_params:
        Additional fields merged into every request body (e.g. ``top_k``).
    """

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
        # TLS verification for the POST. A localm network bind serves HTTPS via
        # its own local CA, so a loopback self-call must trust that CA; external
        # HTTPS (OpenAI/Anthropic) keeps normal public verification. verify=None
        # derives this from base_url; an explicit value overrides (tests).
        if verify is None:
            from localm import tls
            verify = tls.requests_verify(base_url)
        self._verify = verify
        self._last_usage: dict = {}
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

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def last_usage(self) -> dict:
        """Usage dict from the most recent call: {prompt_tokens, completion_tokens, total_tokens}."""
        return dict(self._last_usage)

    def set_tools(self, tool_defs: list) -> None:
        """
        Register OpenAI-format tool definitions for native function calling.

        Each entry should be a dict of the form::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {"type": "object", "properties": {...}, "required": [...]}
                }
            }
        """
        self._tool_defs = tool_defs

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
        """
        Translate an OpenAI-style message list into the Anthropic Messages
        API request shape:

          - system-role messages are concatenated into a top-level ``system``
            field (Anthropic does not accept a ``system`` role in ``messages``).
          - ``max_tokens`` is required, so a default is supplied when absent.
        """
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
        resp = _post_with_retry(
            self._chat_url(),
            headers=self._headers(),
            json_body=self._body(messages, stream=False, **kwargs),
            timeout=self._timeout,
            verify=self._verify,
        )
        _raise_for_status(resp)
        data = resp.json()
        if self.anthropic:
            return self._parse_anthropic_response(data)
        if data.get("usage"):
            self._last_usage = data["usage"]
        message = data["choices"][0]["message"]
        text    = message.get("content") or ""
        # Native tool calls: convert to our XML format and append
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            xml = self._tool_calls_to_xml(tool_calls)
            text = (text + "\n" + xml).strip() if text else xml
        return text

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        if self.anthropic:
            yield from self._anthropic_stream(messages, **kwargs)
            return
        import json as _json
        self._last_usage = {}
        # Accumulate streaming native tool_calls: index → {name, arguments_buf}
        _tc_buf: dict[int, dict] = {}

        with _post_with_retry(
            self._chat_url(),
            headers=self._headers(),
            json_body=self._body(messages, stream=True, **kwargs),
            timeout=self._timeout,
            stream=True,
            verify=self._verify,
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

        # Emit accumulated tool calls as XML after the stream ends
        if _tc_buf:
            ordered = [_tc_buf[i] for i in sorted(_tc_buf)]
            xml = self._tool_calls_to_xml([
                {"function": {"name": t["name"], "arguments": t["arguments"]}}
                for t in ordered
            ])
            yield "\n" + xml

    def _anthropic_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """
        Stream from the Anthropic Messages API (SSE).

        Anthropic emits typed events (message_start, content_block_delta with
        ``text_delta``/``input_json_delta``, message_delta with usage). We yield
        text deltas and accumulate tool_use blocks, emitting them as our XML
        format after the stream ends - mirroring the OpenAI path.
        """
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

        if _tc_buf:
            ordered = [_tc_buf[i] for i in sorted(_tc_buf)]
            xml = self._tool_calls_to_xml([
                {"function": {"name": t["name"], "arguments": t["arguments"]}}
                for t in ordered
            ])
            yield "\n" + xml


# ------------------------------------------------------------------ #
#  Convenience constructors
# ------------------------------------------------------------------ #

def make_localm_backend(model: str, port: int = 8642, *,
                        api_key: str = "", **kw) -> HTTPBackend:
    """Backend for a local ``localm serve``. The bearer key resolves to *api_key*
    if given, else ``$LOCALM_API_KEY``, else ``"localm"`` (open-mode servers
    accept any non-empty string). A hardcoded ``"localm"`` would 401 against a
    ``require_auth`` server - the C1 keystone bug this resolves."""
    key = api_key or os.environ.get("LOCALM_API_KEY") or "localm"
    return HTTPBackend(f"http://127.0.0.1:{port}/v1", model, api_key=key,
                       localm_server=True, **kw)


def make_openai_backend(model: str = "gpt-4o", **kw) -> HTTPBackend:
    key = os.environ.get("OPENAI_API_KEY", "")
    return HTTPBackend("https://api.openai.com/v1", model, api_key=key, native_tools=True, **kw)


def make_anthropic_backend(model: str = "claude-opus-4-5", **kw) -> HTTPBackend:
    """
    Anthropic Messages API backend.

    Posts to ``https://api.anthropic.com/v1/messages`` with an ``x-api-key``
    header and an ``anthropic-version`` header (NOT a Bearer ``Authorization``
    header against ``/chat/completions``). The OpenAI-style message list is
    translated to the Messages request/response shape in HTTPBackend.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return HTTPBackend(
        "https://api.anthropic.com/v1",
        model,
        api_key=key,
        anthropic=True,
        native_tools=True,
        **kw,
    )
