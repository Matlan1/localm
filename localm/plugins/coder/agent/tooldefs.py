# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native tool-calling helper: convert TOOL_REGISTRY into the OpenAI
/v1/chat/completions ``tools`` schema for backends with native_tools=True."""

from __future__ import annotations

import localm.plugins.coder.agent as _agent


def _build_openai_tool_defs() -> list:
    """
    Convert TOOL_REGISTRY into the OpenAI /v1/chat/completions ``tools`` format.

    Used when the backend has ``native_tools=True`` (e.g. the OpenAI API),
    so the model receives a validated schema instead of relying on text parsing.
    """
    TOOL_REGISTRY = _agent.TOOL_REGISTRY  # live: honour a patched agent.TOOL_REGISTRY
    defs = []
    for tool in TOOL_REGISTRY.values():
        properties: dict = {}
        required:   list = []
        for param_name, meta in tool.params.items():
            prop: dict = {"description": meta.get("description", "")}
            raw_type = meta.get("type", "string")
            # Map our shorthand types to JSON Schema types
            prop["type"] = {
                "int":   "integer",
                "float": "number",
                "bool":  "boolean",
                "array": "array",
            }.get(raw_type, "string")
            if raw_type == "array":
                prop["items"] = {"type": "string"}
            properties[param_name] = prop
            if meta.get("required"):
                required.append(param_name)
        defs.append({
            "type": "function",
            "function": {
                "name":        tool.name,
                "description": tool.description,
                "parameters": {
                    "type":       "object",
                    "properties": properties,
                    "required":   required,
                },
            },
        })
    return defs
