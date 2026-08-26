# SPDX-License-Identifier: AGPL-3.0-or-later
"""An Engine-compatible client that talks to a running localm server's ``/v1`` API.

The thin-client piece: ``localm run`` ATTACHES to the per-directory server (chat
flows through its OpenAI-compatible ``/v1/chat/completions``) instead of loading
a SECOND copy of the model in-process - the same attach-or-spawn model the GUI
uses. It is duck-typed to :class:`localm.inference.engine.Engine`, implementing
ONLY the surface the chat REPL touches:

    chat_stream(messages, **gen_opts) -> Iterator[str]
    count_tokens(text) -> int
    display_name
    load() / unload()            (no-ops - the server owns the model + its VRAM)
    __enter__ / __exit__         (so ``with engine:`` works)
    loaded                       (always True for an attached server)

The CLI must NOT manage the remote server's model or VRAM (other clients share
it), so load/unload are no-ops.
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional

from localm.inference.backends.base import UnsupportedInputError


def _error_detail(resp) -> str:
    """A short human message from an error response (FastAPI's ``{"detail": ...}``
    when present, else the raw body)."""
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            d = body["detail"]
            return d if isinstance(d, str) else str(d)
    except Exception:
        pass
    text = getattr(resp, "text", "") or f"HTTP {getattr(resp, 'status_code', '?')}"
    return text[:400]


class HttpEngine:
    """Engine stand-in that routes chat through a remote localm server's ``/v1`` API."""

    def __init__(self, base_url: str, *, token: Optional[str] = None,
                 model: Optional[str] = None, display_name: Optional[str] = None):
        self._base = base_url.rstrip("/")
        self._token = token
        self._model = model
        self.display_name = display_name or model or "remote model"

    # --- lifecycle (no-ops: the server owns the model) ---------------------- #
    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def __enter__(self) -> "HttpEngine":
        return self

    def __exit__(self, *_exc) -> None:
        pass

    # --- token estimate ----------------------------------------------------- #
    def count_tokens(self, text: str) -> int:
        """Estimate token count. There is no remote tokenizer; the REPL uses this
        only for a tok/s readout and the compaction trigger, where a chars-over-4
        estimate is adequate (the in-process Engine uses the same fallback when its
        model is not loaded)."""
        return max(1, len(text or "") // 4)

    def context_capacity(self) -> Optional[int]:
        """The loaded model's RESOLVED context ceiling from the server's
        /v1/config (VRAM-derived under ctx_auto), cached after the first
        successful fetch. Best-effort: None on any error."""
        if getattr(self, "_ctx_capacity_cached", False):
            return self._ctx_capacity
        cap = None
        try:
            import requests
            resp = requests.get(f"{self._base}/config",
                                headers=self._headers(), timeout=15)
            if resp.status_code == 200:
                v = resp.json().get("effective_ctx_max")
                if isinstance(v, int) and v > 0:
                    cap = v
        except Exception:
            cap = None
        # Cache ONLY a successful resolution (matching the docstring). A
        # transient / early failure - the model not loaded yet, a non-200, a
        # network blip - must not permanently latch None; a later call retries
        # once /v1/config can answer.
        if cap is not None:
            self._ctx_capacity = cap
            self._ctx_capacity_cached = True
        return cap

    # --- chat --------------------------------------------------------------- #
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def chat_stream(self, messages: List[dict], *,
                    max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None,
                    top_p: Optional[float] = None,
                    top_k: Optional[int] = None,
                    repeat_penalty: Optional[float] = None,
                    grammar: Optional[str] = None,
                    seed: Optional[int] = None) -> Iterator[str]:
        """Stream assistant tokens from the server's ``/v1/chat/completions``.

        Raises :class:`UnsupportedInputError` when the server refuses image input on
        a text-only model (so the REPL shows the same vision guidance as in-process),
        and ``RuntimeError`` for an unreachable server or other error responses."""
        import requests

        body: dict = {
            "model": self._model or self.display_name,
            "messages": messages,
            "stream": True,
        }
        # Only send params the caller set, so server defaults apply otherwise. top_k /
        # repeat_penalty / seed / grammar are localm /v1 extensions; a stricter server
        # ignores unknown fields.
        for key, val in (("max_tokens", max_tokens), ("temperature", temperature),
                         ("top_p", top_p), ("top_k", top_k),
                         ("repeat_penalty", repeat_penalty), ("seed", seed),
                         ("grammar", grammar)):
            if val is not None:
                body[key] = val

        try:
            resp = requests.post(f"{self._base}/chat/completions",
                                 json=body, headers=self._headers(),
                                 stream=True, timeout=300)
        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not reach the localm server at {self._base}: {e}")

        if resp.status_code >= 400:
            detail = _error_detail(resp)
            # Mirror the in-process path: image-on-text-only is an UnsupportedInputError
            # the REPL catches to print vision guidance.
            if resp.status_code == 400 and "image" in detail.lower():
                raise UnsupportedInputError(detail)
            raise RuntimeError(f"server error (HTTP {resp.status_code}): {detail}")

        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if not line or line == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            choices = chunk.get("choices") or [{}]
            delta = (choices[0] or {}).get("delta") or {}
            # The server already splits <think> reasoning out of `content` into its
            # own `reasoning_content` field. Re-wrap it in the same inline
            # <think>...</think> markers the in-process Engine's raw stream carries,
            # so cli/chat.py's ThinkSplitter/_ThinkPrinter dims it identically in
            # attach mode instead of silently dropping it.
            # No escaping needed: the server derives reasoning_content by
            # scanning for the FIRST "</think>" (ThinkSplitter in textnorm.py),
            # so a reasoning_content value can never itself contain an intact
            # "</think>" - reinjecting one here cannot confuse the client-side
            # splitter that re-parses this wrapped stream.
            reasoning = delta.get("reasoning_content")
            if reasoning:
                yield f"<think>{reasoning}</think>"
            piece = delta.get("content")
            if piece:
                yield piece


def remote_model_status(base_url: str, token: Optional[str] = None,
                        *, timeout: float = 5.0) -> tuple[str, Optional[str]]:
    """Probe a server's loaded model via ``GET /v1/models``. Returns ``(state, id)``:

      - ``("loaded", "<id>")``  the server reports a model (its id).
      - ``("empty",  None)``    the server ANSWERED (200) but reports no model.
      - ``("unknown", None)``   we could not tell: unreachable, an auth/scope error,
        or a malformed reply. This is NOT proof that no model is loaded - ``/v1/models``
        needs the ``models`` scope, so an attach token scoped only for chat gets a 403
        here while chatting still works. Callers must not claim "no model loaded" on
        ``unknown``.

    Best-effort labelling only; never gates the attach."""
    import requests
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models",
                         headers=headers, timeout=timeout)
        if r.status_code >= 400:
            return "unknown", None
        data = (r.json() or {}).get("data") or []
        # /v1/models lists EVERY registered model (sorted), each with a `loaded`
        # flag and an `active` flag. Pick the active model (else the first loaded
        # one) - NOT data[0], which is the alphabetically-first REGISTERED model
        # and is almost never the loaded one, so reading it makes `localm run`
        # attach to, and force-load, the wrong model.
        dicts = [m for m in data if isinstance(m, dict) and m.get("id")]
        chosen = next((m for m in dicts if m.get("active")), None) \
            or next((m for m in dicts if m.get("loaded")), None)
        if chosen is None and dicts and not any(
                ("active" in m or "loaded" in m) for m in dicts):
            # An older / minimal server that sends no active/loaded markers at
            # all: fall back to the first entry (the pre-marker behaviour). When
            # markers ARE present we never fall back to data[0].
            chosen = dicts[0]
        if chosen is not None:
            return "loaded", chosen.get("id")
        return "empty", None
    except Exception:
        return "unknown", None


def remote_active_model(base_url: str, token: Optional[str] = None,
                        *, timeout: float = 5.0) -> Optional[str]:
    """The id of the server's active model, or None when unknown / none loaded.
    Thin wrapper over :func:`remote_model_status` for callers that only want the
    label. Prefer ``remote_model_status`` when you must distinguish "no model" from
    "could not read the model list" (see that function)."""
    return remote_model_status(base_url, token, timeout=timeout)[1]
