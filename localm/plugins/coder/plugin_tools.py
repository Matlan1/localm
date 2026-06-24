# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register tools exported by external plugins into the coder's TOOL_REGISTRY.

A plugin opts in through its manifest:

    [tools]
    exports = ["tool_search_issues"]

Each exported name must be a callable in the plugin's entry module that follows
the same contract as a built-in coder tool:

    def tool_search_issues(cwd, **args):
        ...
        return ToolResult.success("...")   # a plain str return is also accepted

The first positional argument is the working directory (a Path); the rest are
keyword arguments matching the declared parameter schema. Optional function
attributes describe the tool to the model and the safety layer:

    fn.tool_description : str   one-line description (else the docstring's first line)
    fn.tool_params      : dict  {param: {"type", "description", "required"}}
    fn.tool_destructive : bool  whether it needs confirmation
                                (default True - external code is treated as
                                 destructive, the same posture as MCP tools)

Registered names are namespaced ``plugin_<plugin>_<export>`` so they cannot
clash with built-in tools or with another plugin's exports. The model learns
the exact names from the system prompt (same as MCP tools).

This mirrors localm.plugins.coder.mcp.register_mcp_tools and is wired into the
agent the same way: discovered once at agent start, failures warn and continue.
"""

from __future__ import annotations

from typing import List, Tuple

from .provenance import neutralise
from .tools import TOOL_REGISTRY, ToolDef, ToolResult


def _neutralise_params(params: dict) -> dict:
    """Defang control tokens / frame markers in param names and descriptions, so a
    plugin's param metadata (which lands in the system prompt and the native tool
    schema) cannot forge a role boundary. Parity with mcp._schema_to_params; a
    no-op for ordinary param names. (Plugin code is local/author-controlled, so
    this is defense-in-depth, not a remote-attacker surface.)"""
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
    """Adapt a plugin callable to the (cwd, **args) -> ToolResult contract,
    coercing its return value and containing its errors."""
    def _fn(cwd, **args) -> ToolResult:
        try:
            return _coerce_result(raw_fn(cwd, **args))
        except TypeError as e:
            return ToolResult.error(f"Bad arguments for {reg_name}: {e}")
        except Exception as e:
            return ToolResult.error(f"Plugin tool '{reg_name}' failed: {e}")
    return _fn


def register_plugin_tools() -> Tuple[List[str], List[str]]:
    """Discover installed plugins and register their exported tools.

    Returns ``(registered_names, warnings)``. Never raises - a broken plugin
    must not break the agent.
    """
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

            if reg_name in TOOL_REGISTRY:
                existing = TOOL_REGISTRY[reg_name]
                if existing.description.startswith(f"[plugin:{manifest.name}]"):
                    # Already registered by us in a prior agent init (e.g. a
                    # sub-agent): reuse it, still surface it to the model.
                    registered.append(reg_name)
                else:
                    warnings.append(f"Plugin tool name clash, skipped: {reg_name}")
                continue

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

            TOOL_REGISTRY[reg_name] = ToolDef(
                name=reg_name,
                fn=_make_plugin_tool_fn(fn, reg_name),
                # A plugin tool's description lands in the system prompt; defang
                # control tokens / frame markers in it so an installed plugin (or
                # one whose description is sourced non-locally) cannot forge a role
                # boundary. A no-op for ordinary descriptions.
                description=neutralise(f"[plugin:{manifest.name}] {description}".strip()),
                params=params,
                destructive=destructive,
            )
            registered.append(reg_name)

    return registered, warnings
