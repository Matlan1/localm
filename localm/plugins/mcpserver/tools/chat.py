# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools that generate with a local model: ``chat`` and ``embed``."""

from __future__ import annotations

import json
from typing import Dict

from ..server import EngineCache, _quiet_stdout, _text_result
from ._common import MODEL_PARAM


def build(engines: EngineCache) -> Dict[str, dict]:
    """``chat`` and ``embed``, both served from *engines*."""
    def chat(args: dict) -> dict:
        prompt = args.get("prompt", "")
        if not prompt:
            return _text_result("'prompt' is required", is_error=True)
        # engines.get() can trigger a fresh model load, and a GGUF load prints
        # native sizing/context diagnostics (e.g. the "ctx auto" note) straight
        # to stdout - the same stream the JSON-RPC frames travel on.
        with _quiet_stdout():
            engine = engines.get(args.get("model"))
        messages = []
        if args.get("system"):
            messages.append({"role": "system", "content": args["system"]})
        messages.append({"role": "user", "content": prompt})
        gen: dict = {}
        for key in ("max_tokens", "temperature", "seed"):
            if args.get(key) is not None:
                gen[key] = args[key]
        with engines.serving(engine):
            text = "".join(engine.chat_stream(messages, **gen))
        return _text_result(text)

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
            "description": "Generate a response with a local LLM (fully offline).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt":      {"type": "string", "description": "User prompt"},
                    "system":      {"type": "string", "description": "Optional system prompt"},
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
