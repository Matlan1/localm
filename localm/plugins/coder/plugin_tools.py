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


from .provenance import neutralise
from .tool_registration import register_foreign_tool
from .tools import ToolResult


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


def register_plugin_tools() -> tuple[list[str], list[str]]:
    """Register the exported tools of every installed AND enabled plugin, discovered
    through the plugin engine.

    Returns ``(registered_names, warnings)``. Never raises - a broken plugin
    must not break the agent.
    """
    from localm.plugins.engine import PluginManager, import_tool_module

    registered: list[str] = []
    warnings: list[str] = []

    try:
        manager = PluginManager(app=None)
        specs = manager.discover()
        enabled = manager._enabled_set()
    except Exception as e:                       # discovery must never crash the agent
        return [], [f"Plugin discovery failed: {e}"]

    # An enabled plugin that discovery refused (an unreadable manifest, a scope
    # collision) is not in specs; say so, since its tools are missing.
    for name, err in sorted(getattr(manager, "_discover_errors", {}).items()):
        if name in enabled and name not in specs:
            warnings.append(
                f"Plugin {name!r} was skipped ({err}), so none of its tools are "
                "available in this session.")

    for manifest in (specs[n] for n in sorted(specs)):
        if manifest.name not in enabled or not manifest.tool_exports:
            continue
        try:
            module = import_tool_module(manifest)
        except Exception as e:                   # plugin import bugs stay contained
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
