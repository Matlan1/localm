"""
HTTP backend — connects to any OpenAI-compatible inference endpoint.

Covers:
  - localm serve (offline, local model)
  - OpenAI API          (online, requires OPENAI_API_KEY)
  - Anthropic Messages  (online, requires ANTHROPIC_API_KEY) via openai-compat shim
  - Ollama              (offline, localhost:11434)
  - LM Studio, llama.cpp server, etc.
"""

from __future__ import annotations

import os
from typing import Iterator

import requests

from .base import BaseLLMBackend


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

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "localm",
        timeout: int = 300,
        native_tools: bool = False,
        **extra_params,
    ) -> None:
        self._base_url     = base_url.rstrip("/")
        self._model        = model
        self._api_key      = api_key
        self._timeout      = timeout
        self._extra        = extra_params
        self._last_usage: dict = {}
        self.native_tools  = native_tools
        self._tool_defs: list = []   # OpenAI-format tool definitions

        # GBNF grammar sampling is only supported by our own local server.
        # Passing grammar= to external APIs (OpenAI, Anthropic) causes errors.
        _external_prefixes = ("https://api.openai.com", "https://api.anthropic.com")
        self.supports_grammar = not any(
            base_url.startswith(p) for p in _external_prefixes
        )

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
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _body(self, messages: list[dict], stream: bool, **kwargs) -> dict:
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

    # ------------------------------------------------------------------ #

    def chat(self, messages: list[dict], **kwargs) -> str:
        self._last_usage = {}
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=self._body(messages, stream=False, **kwargs),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
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
        import json as _json
        self._last_usage = {}
        # Accumulate streaming native tool_calls: index → {name, arguments_buf}
        _tc_buf: dict[int, dict] = {}

        with requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=self._body(messages, stream=True, **kwargs),
            timeout=self._timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
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


# ------------------------------------------------------------------ #
#  Convenience constructors
# ------------------------------------------------------------------ #

def make_localm_backend(model: str, port: int = 8642, **kw) -> HTTPBackend:
    return HTTPBackend(f"http://127.0.0.1:{port}/v1", model, api_key="localm", **kw)


def make_openai_backend(model: str = "gpt-4o", **kw) -> HTTPBackend:
    key = os.environ.get("OPENAI_API_KEY", "")
    return HTTPBackend("https://api.openai.com/v1", model, api_key=key, native_tools=True, **kw)


def make_anthropic_backend(model: str = "claude-opus-4-5", **kw) -> HTTPBackend:
    """Uses the OpenAI-compat shim included in the ``anthropic`` SDK."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    # anthropic provides an openai-compat base URL
    return HTTPBackend(
        "https://api.anthropic.com/v1",
        model,
        api_key=key,
        **kw,
    )
