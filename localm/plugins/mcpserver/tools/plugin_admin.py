# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools that administer engine plugins: list, install, enable, disable,
uninstall."""

from __future__ import annotations

from typing import Dict

from ..server import _quiet_stdout, _text_result


def _run_mgr_action(mgr, fn, *, plugin: str):
    """Call fn(mgr) inside the stdout-quieting guard, mapping KeyError to a
    "no such plugin" result and ValueError to its own message. Returns the
    mapped error _text_result, or None on success."""
    with _quiet_stdout():
        try:
            fn(mgr)
        except KeyError:
            return _text_result(f"No such plugin: {plugin}", is_error=True)
        except ValueError as e:
            return _text_result(str(e), is_error=True)
    return None


def build() -> Dict[str, dict]:
    """``list_plugins``, ``install_plugin``, ``enable_plugin``,
    ``disable_plugin`` and ``uninstall_plugin``."""
    def list_plugins(args: dict) -> dict:
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        state = mgr.api_state()
        plugins = state.get("plugins", [])
        if not plugins:
            return _text_result("No engine plugins discovered.")
        lines = []
        for p in plugins:
            status = "enabled" if p.get("active") else ("disabled" if p.get("installed") else "available")
            desc = f" - {p['description']}" if p.get("description") else ""
            lines.append(f"{p['name']}  [{status}]{desc}")
        return _text_result("\n".join(lines))

    def install_plugin(args: dict) -> dict:
        plugin = args.get("plugin", "")
        if not plugin:
            return _text_result("'plugin' is required", is_error=True)
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        err = _run_mgr_action(
            mgr, lambda m: m.set_installed_state(plugin, True), plugin=plugin)
        if err is not None:
            return err
        dep_result = None
        with _quiet_stdout():
            with_deps = args.get("with_deps", True)
            if with_deps and mgr.plugin_missing_deps(plugin):
                dep_result = mgr.install_plugin_deps(plugin)
        if dep_result is not None and not dep_result.ok:
            # Left ENABLED rather than rolled back, matching the CLI's own
            # `plugin install` behaviour, which also leaves a plugin installed
            # on a dep failure and points the operator at retrying the extras
            # later. The failure is SURFACED rather than swallowed: folded into
            # the reply below, plus a warning here since this reply is the only
            # place it is seen.
            from localm.debuglog import logger
            logger.warning("install_plugin(%s): pip extras failed to install: %s",
                            plugin, dep_result.error)
            failed = ", ".join(dep_result.failed) or "see error"
            return _text_result(
                f"Plugin '{plugin}' installed and enabled, but its dependencies "
                f"failed to install ({failed}): {dep_result.error}", is_error=True)
        return _text_result(f"Plugin '{plugin}' successfully installed and enabled.")

    def enable_plugin(args: dict) -> dict:
        plugin = args.get("plugin", "")
        if not plugin:
            return _text_result("'plugin' is required", is_error=True)
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        err = _run_mgr_action(
            mgr, lambda m: m.set_enabled_state(plugin, True), plugin=plugin)
        if err is not None:
            return err
        return _text_result(f"Plugin '{plugin}' successfully enabled.")

    def disable_plugin(args: dict) -> dict:
        plugin = args.get("plugin", "")
        if not plugin:
            return _text_result("'plugin' is required", is_error=True)
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        err = _run_mgr_action(
            mgr, lambda m: m.set_enabled_state(plugin, False), plugin=plugin)
        if err is not None:
            return err
        return _text_result(f"Plugin '{plugin}' successfully disabled.")

    def uninstall_plugin(args: dict) -> dict:
        plugin = args.get("plugin", "")
        if not plugin:
            return _text_result("'plugin' is required", is_error=True)
        delete_data = args.get("delete_data", False)
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        # Bypasses _run_mgr_action (unlike install/enable/disable above):
        # uninstall()'s bool is the only signal that the installed directory
        # actually came off disk (a locked file, an AV hold, a permission
        # denial), so it is read here. is_installed_or_on_disk(), not
        # is_installed(): a manifest-less directory is something uninstall()
        # below can still act on.
        existed = mgr.is_installed_or_on_disk(plugin)
        with _quiet_stdout():
            try:
                removed = mgr.uninstall(plugin, delete_data=delete_data)
            except KeyError:
                return _text_result(f"No such plugin: {plugin}", is_error=True)
            except ValueError as e:
                return _text_result(str(e), is_error=True)
        if existed and not removed:
            detail = (
                f"Plugin '{plugin}' was disabled and unloaded, but it was not "
                f"fully uninstalled: something it owns could not be removed "
                f"from disk.")
            if delete_data:
                detail += (
                    " delete_data was requested and this uninstall did not "
                    "complete, so do not assume its stored data is gone.")
            return _text_result(detail, is_error=True)
        return _text_result(f"Plugin '{plugin}' successfully uninstalled.")

    return {
        "list_plugins": {
            "description": "List engine plugins, their descriptions, and activation status.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "title": "List plugins"},
            "handler": list_plugins,
        },
        "install_plugin": {
            "description": "Install and enable an engine plugin.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "plugin": {"type": "string", "description": "Plugin name to install"},
                    "with_deps": {"type": "boolean", "description": "Also install pip dependencies (default true)"}
                },
                "required": ["plugin"],
            },
            "handler": install_plugin,
        },
        "enable_plugin": {
            "description": "Enable an installed engine plugin.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "plugin": {"type": "string", "description": "Plugin name to enable"}
                },
                "required": ["plugin"],
            },
            "handler": enable_plugin,
        },
        "disable_plugin": {
            "description": "Disable an installed engine plugin.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "plugin": {"type": "string", "description": "Plugin name to disable"}
                },
                "required": ["plugin"],
            },
            "handler": disable_plugin,
        },
        "uninstall_plugin": {
            "description": "Uninstall (deselect) an engine plugin.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "plugin": {"type": "string", "description": "Plugin name to uninstall"},
                    "delete_data": {"type": "boolean", "description": "Also delete stored data (default false)"}
                },
                "required": ["plugin"],
            },
            # Removes the plugin (and, with delete_data, its stored data on disk) -
            # declare it so an MCP client can confirm before calling.
            "annotations": {"destructiveHint": True, "title": "Uninstall plugin"},
            "handler": uninstall_plugin,
        },
    }
