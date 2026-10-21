# SPDX-License-Identifier: AGPL-3.0-or-later
"""Map the argument names models commonly use onto a tool's own parameter names,
and explain a call whose arguments still do not fit the tool."""

from __future__ import annotations

import inspect
import json
from typing import Optional

# Canonical parameter name -> names other coding harnesses use for it.
_ALIASES: dict[str, tuple[str, ...]] = {
    "path": ("file_path", "filepath", "file", "filename", "file_name", "target_file"),
    "old": ("old_str", "old_string", "old_text", "old_code", "search", "find", "original"),
    "new": ("new_str", "new_string", "new_text", "new_code", "replace", "replacement"),
    "content": ("contents", "file_content", "file_contents", "text", "code"),
}


def _params(tool_def) -> dict:
    params = getattr(tool_def, "params", None)
    return params if isinstance(params, dict) else {}


def normalize_args(tool_def, args: dict,
                   beside: Optional[dict] = None) -> tuple[dict, list[tuple[str, str]]]:
    """``(args, renamed)``: *args* with each alias renamed to the canonical name
    the tool declares, and the ``(alias, canonical)`` pairs that were renamed.

    An alias is renamed only when the tool declares the canonical name, does not
    declare the alias itself, and the call does not already carry the canonical
    name. *beside* holds keys the call object carried next to its args; each one
    the tool declares (directly or through an alias) and the args lack is added,
    and the rest are dropped. Neither input is modified."""
    params = _params(tool_def)
    out = dict(args)
    for key, value in (beside or {}).items():
        canonical = key if key in params else next(
            (c for c, aliases in _ALIASES.items()
             if key in aliases and c in params), None)
        if canonical is not None and canonical not in out:
            out[canonical] = value
    renamed: list[tuple[str, str]] = []
    for canonical, aliases in _ALIASES.items():
        if canonical not in params or canonical in out:
            continue
        for alias in aliases:
            if alias in out and alias not in params:
                out[canonical] = out.pop(alias)
                renamed.append((alias, canonical))
                break
    return out, renamed


def bad_args_message(name: str, tool_def, args: dict) -> Optional[str]:
    """The error for a call whose *args* cannot be bound to the tool's function,
    naming what was sent, what is missing and what is unknown, with an example
    call; ``None`` when the arguments bind.

    Only the model-visible parameters count: a parameter whose name starts with
    an underscore is injected by the agent, never sent by the model."""
    fn = getattr(tool_def, "fn", None)
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    names = [p for p in list(sig.parameters)[1:] if not p.startswith("_")]
    try:
        sig.bind(None, **args)
        return None
    except TypeError:
        pass
    required = [p for p in names
                if sig.parameters[p].default is inspect.Parameter.empty
                and sig.parameters[p].kind not in (inspect.Parameter.VAR_KEYWORD,
                                                   inspect.Parameter.VAR_POSITIONAL)]
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD
                      for p in sig.parameters.values())
    missing = [p for p in required if p not in args]
    unknown = [] if accepts_any else [k for k in args if k not in sig.parameters]
    sent = ", ".join(args) if args else "nothing"
    lines = [f"{name} was not run: its arguments do not fit the tool. You sent: {sent}."]
    if missing:
        lines.append(f"Missing: {', '.join(missing)}.")
    if unknown:
        lines.append(f"Unknown: {', '.join(unknown)}. {name} takes: {', '.join(names)}.")
    from .prompts import _example_args
    example = json.dumps({"name": name, "args": _example_args(name, tool_def)},
                         ensure_ascii=False)
    lines.append("Every argument goes inside \"args\", and text written outside the "
                 "call is not passed to the tool. For example:\n"
                 f"<tool_call>\n{example}\n</tool_call>")
    return "\n".join(lines)
