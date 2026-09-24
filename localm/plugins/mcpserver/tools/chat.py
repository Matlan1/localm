# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools that generate with a local model: ``chat`` and ``embed``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..server import EngineCache, _log, _quiet_stdout, _text_result
from ._common import MODEL_PARAM

# Largest image file the chat tool reads from disk.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_CAPABILITY_WORDS = {"vision": "reading images", "tool_use": "structured tool calls",
                     "reasoning": "reasoning", "context_length": "a longer conversation"}


def _image_part(ref: str) -> dict:
    """An OpenAI ``image_url`` content part for *ref*: a ``data:image/...`` URI
    as given, or a local image file read into one. Raises ValueError for a
    UNC or device path, a missing or oversized file, or a file that is not an
    image by name."""
    import base64
    import mimetypes
    from localm.pathsafe import is_unc_or_device_path
    if not isinstance(ref, str) or not ref:
        raise ValueError("each image must be a file path or a data: URI")
    if ref.startswith("data:"):
        if not ref.startswith("data:image/"):
            raise ValueError("a data: URI image must be data:image/...")
        return {"type": "image_url", "image_url": {"url": ref}}
    if is_unc_or_device_path(ref):
        raise ValueError(f"image {ref!r} must be a local file, not a UNC or device path")
    p = Path(ref).expanduser()
    if not p.is_file():
        raise ValueError(f"image file not found: {ref}")
    mime = mimetypes.guess_type(p.name)[0] or ""
    if not mime.startswith("image/"):
        raise ValueError(f"{ref} is not an image file")
    size = p.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"{ref} is {size} bytes, over the {MAX_IMAGE_BYTES}-byte limit")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def routing_note(decision) -> Optional[str]:
    """One line naming the model that answered and why, for a routed
    *decision*; None when it was not routed."""
    if decision is None or not decision.routed:
        return None
    needs = ", ".join(_CAPABILITY_WORDS.get(g, g) for g in sorted(decision.gaps))
    return (f"[answered by {decision.resolved}: {decision.current} lacks "
            f"{needs or 'what this request needed'}]")


def answer_with(engines: EngineCache, decision, run):
    """Call ``run(engine, name)`` with the engine *decision* resolves to,
    falling back through the other capable candidates and then the model the
    request would otherwise use when one cannot be loaded or answered by.
    Another instance's copy that stops answering is dropped for this server's
    own. Returns ``(result, decision)``, the decision rewritten to name the
    model that answered."""
    names: List[str] = []
    if decision.routed:
        names = list(decision.candidates or (decision.resolved,))
    names.append(decision.current)
    errors: List[str] = []
    for name in names:
        try:
            # A load prints native sizing diagnostics straight to stdout, the
            # stream the JSON-RPC frames travel on.
            with _quiet_stdout():
                engine = engines.get_chat(name)
            try:
                result = run(engine, name)
            except RuntimeError as e:
                if not engines.is_peer(engine):
                    raise
                engines.drop_peer(name)
                _log(f"the instance answering {name} failed ({e}); loading it here")
                with _quiet_stdout():
                    engine = engines.get(name)
                result = run(engine, name)
        except Exception as e:
            if name == decision.current:
                raise
            errors.append(f"{name}: {e}")
            _log(f"warning: could not answer with {name}: {e}")
            continue
        if name == decision.resolved:
            return result, decision
        if name == decision.current:
            return result, decision.without_route(errors)
        return result, decision.answered_by(name)
    raise RuntimeError("no model could answer: " + "; ".join(errors))


def build(engines: EngineCache) -> Dict[str, dict]:
    """``chat`` and ``embed``, both served from *engines*."""
    def chat(args: dict) -> dict:
        prompt = args.get("prompt", "")
        if not prompt:
            return _text_result("'prompt' is required", is_error=True)
        images = args.get("images") or []
        if isinstance(images, str):
            images = [images]
        try:
            parts = [_image_part(ref) for ref in images]
        except ValueError as e:
            return _text_result(str(e), is_error=True)
        messages = []
        if args.get("system"):
            messages.append({"role": "system", "content": args["system"]})
        if parts:
            messages.append({"role": "user",
                             "content": parts + [{"type": "text", "text": prompt}]})
        else:
            messages.append({"role": "user", "content": prompt})
        gen: dict = {}
        for key in ("max_tokens", "temperature", "seed"):
            if args.get(key) is not None:
                gen[key] = args[key]
        # A model the client named is always used; otherwise a request the
        # default model cannot serve is answered by an installed model that
        # can.
        decision = engines.route(args.get("model"), messages)

        def _run(engine, _name):
            with engines.serving(engine):
                return "".join(engine.chat_stream(messages, **gen))

        text, decision = answer_with(engines, decision, _run)
        result = _text_result(text)
        note = routing_note(decision)
        if note:
            result["content"].append({"type": "text", "text": note})
        return result

    def embed(args: dict) -> dict:
        texts = args.get("texts")
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return _text_result("'texts' is required (string or list)", is_error=True)
        # A fresh embedder load can print to stdout too, like chat() above.
        with _quiet_stdout():
            engine = engines.get(args.get("model"))
        try:
            with engines.serving(engine):
                vecs = engine.embed(texts)
        except NotImplementedError as e:
            return _text_result(str(e), is_error=True)
        return _text_result(json.dumps(vecs))

    return {
        "chat": {
            "description": (
                "Generate a response with a local LLM (fully offline). Without "
                "'model', a request the default model cannot serve (attached "
                "images it cannot read, a longer conversation than it was "
                "trained for) is answered by an installed model that can, and "
                "the reply ends with a note naming it; a named 'model' always "
                "answers."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt":      {"type": "string", "description": "User prompt"},
                    "system":      {"type": "string", "description": "Optional system prompt"},
                    "images":      {"type": "array", "items": {"type": "string"},
                                    "description": "Images for the model to read: local "
                                                   "image file paths or data:image/... "
                                                   "URIs"},
                    "model":       MODEL_PARAM,
                    "max_tokens":  {"type": "integer", "description": "Max tokens to generate"},
                    "temperature": {"type": "number", "description": "Sampling temperature"},
                    "seed":        {"type": "integer", "description": "Seed for reproducible output"},
                },
                "required": ["prompt"],
            },
            "handler": chat,
        },
        "embed": {
            "description": "Compute embedding vectors for one or more texts with a local model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "texts": {"type": "array", "description": "Texts to embed",
                              "items": {"type": "string"}},
                    "model": MODEL_PARAM,
                },
                "required": ["texts"],
            },
            "handler": embed,
        },
    }
