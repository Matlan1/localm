# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register tools exported by external plugins into the coder's TOOL_REGISTRY."""

from __future__ import annotations

from typing import List, Tuple

from .provenance import neutralise
from .tool_registration import register_foreign_tool
from .tools import ToolResult


def _neutralise_params(params: dict) -> dict:
    """Defang control tokens / frame markers in param names and descriptions, so a plugin's param metadata (which lands in the system prompt and the native tool schema) cannot forge a role boundary."""
    out: dict = {}
    for pname, meta in params.items():
        key = neutralise(str(pname))
        if isinstance(meta, dict):
            m = dict(meta)
            if "description" in m:
                m["description"] = neutralise(str(m["description"]))
            out[key] = m
        else:
            out[key] = meta
    return out


def _coerce_result(out) -> ToolResult:
    """Accept a ToolResult, a plain string, or anything stringable."""
    if isinstance(out, ToolResult):
        return out
    if isinstance(out, str):
        return ToolResult.success(out)
    return ToolResult.success(str(out))


def _make_plugin_tool_fn(raw_fn, reg_name: str):
    """Adapt a plugin callable to the (cwd, **args) -> ToolResult contract, coercing its return value and containing its errors."""
    def _fn(cwd, **args) -> ToolResult:
        try:
            return _coerce_result(raw_fn(cwd, **args))
        except TypeError as e:
            return ToolResult.error(f"Bad arguments for {reg_name}: {e}")
        except Exception as e:
            return ToolResult.error(f"Plugin tool '{reg_name}' failed: {e}")
    return _fn


def register_plugin_tools() -> Tuple[List[str], List[str]]:
    """Discover installed plugins and register their exported tools."""
    from localm.plugins.loader import (
        PluginError, discover_plugins, import_plugin_module,
    )

    registered: List[str] = []
    warnings: List[str] = []

    try:
        manifests = discover_plugins()
    except Exception as e:                       # discovery must never crash the agent
        return [], [f"Plugin discovery failed: {e}"]

    for manifest in manifests:
        if not manifest.tool_exports:
            continue
        try:
            module = import_plugin_module(manifest)
        except PluginError as e:
            warnings.append(str(e))
            continue
        except Exception as e:                   # defensive - plugin import bugs stay contained
            warnings.append(f"Plugin {manifest.name!r} failed to import: {e}")
            continue

        for export in manifest.tool_exports:
            reg_name = f"plugin_{manifest.name}_{export}".replace("-", "_")

            fn = getattr(module, export, None)
            if not callable(fn):
                warnings.append(
                    f"Plugin {manifest.name!r}: exported tool {export!r} is "
                    "missing or not callable"
                )
                continue

            description = (
                getattr(fn, "tool_description", None)
                or (fn.__doc__ or "").strip().split("\n")[0]
                or "external plugin tool"
            )
            params = getattr(fn, "tool_params", {})
            if not isinstance(params, dict):
                params = {}
            params = _neutralise_params(params)
            destructive = bool(getattr(fn, "tool_destructive", True))

            register_foreign_tool(
                reg_name,
                fn=_make_plugin_tool_fn(fn, reg_name),
                description=f"[plugin:{manifest.name}] {description}",
                params=params,
                destructive=destructive,
                source_label="Plugin",
                registered=registered,
                warnings=warnings,
                # Already registered by us in a prior agent init (e.g. a
                # sub-agent): reuse it, still surface it to the model.
                reuse_if_already_ours=lambda existing, _n=manifest.name:
                    existing.description.startswith(f"[plugin:{_n}]"),
            )

    return registered, warnings
