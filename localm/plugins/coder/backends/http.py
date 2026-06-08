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
        **extra_params,
    ) -> None:
        self._base_url   = base_url.rstrip("/")
        self._model      = model
        self._api_key    = api_key
        self._timeout    = timeout
        self._extra      = extra_params
        self._last_usage: dict = {}

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def last_usage(self) -> dict:
        """Usage dict from the most recent call: {prompt_tokens, completion_tokens, total_tokens}."""
        return dict(self._last_usage)

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
        return data["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        import json as _json
        self._last_usage = {}
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
                piece = delta.get("content") or ""
                if piece:
                    yield piece


# ------------------------------------------------------------------ #
#  Convenience constructors
# ------------------------------------------------------------------ #

def make_localm_backend(model: str, port: int = 8080, **kw) -> HTTPBackend:
    return HTTPBackend(f"http://127.0.0.1:{port}/v1", model, api_key="localm", **kw)


def make_openai_backend(model: str = "gpt-4o", **kw) -> HTTPBackend:
    key = os.environ.get("OPENAI_API_KEY", "")
    return HTTPBackend("https://api.openai.com/v1", model, api_key=key, **kw)


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
