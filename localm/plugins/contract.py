# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Plugin contract (v1): the interface every localm plugin implements, plus the
host API the engine provides to it.

This is a SUPERSET of the legacy CLI manifest in localm/plugins/loader.py. A
plugin may still expose a Click command (``cli_entry = "module:attr"``), but it
can also contribute a *server surface* via ``register(host)``: routes, a GUI
tab, a settings section, and a capability scope. Built-in (in-tree) plugins and
third-party (~/.localm/plugins) plugins use the same contract. Chat is the
canonical built-in, *protected* plugin (cannot be disabled or uninstalled).

Phase 0 defines the interfaces only. The engine (PluginManager) and the concrete
Host land in Phase 2; loader.py is extended to populate PluginSpec in Phase 2;
the existing features migrate onto register() in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

#: Host<->plugin contract version. Bump on breaking changes; a plugin declares
#: the api_version it targets and the engine refuses incompatible plugins.
API_VERSION = 1

#: Every key a plugin.toml [plugin] table may carry, across BOTH manifest
#: formats: this engine contract (parsed by engine.parse_spec) and the legacy
#: CLI format (parsed by loader.parse_manifest; its own keys are name/version/
#: description/entry plus the separate [tools] table). Both formats share the
#: installed dir, so each parser tolerates the other format's keys and warns
#: only on a key known to NEITHER - typo protection without false alarms
#: (LM-DA-007). Shared here so the two parsers cannot drift.
KNOWN_PLUGIN_KEYS = frozenset({
    "name", "version", "api_version", "description", "scope",
    "requires_extras", "requires", "capabilities", "data_subdir",
    "protected", "default_enabled", "cli", "register",
    "entry",                                 # legacy CLI manifest key
})


@dataclass
class Surface:
    """A plugin's GUI contribution: a tab in the SPA and/or a settings section."""
    tab_id: str = ""            # "" = no tab (settings-only or headless plugin)
    label: str = ""
    icon: str = ""              # emoji or static-asset name
    assets_dir: str = ""        # static frontend assets, relative to the plugin;
                                # served at /plugins/<name>/ when set
    client_entry: str = ""      # ES module under assets_dir the SPA import()s on
                                # boot for an active plugin (e.g. "tts.js"); the
                                # module exports register(ctx) - a headless plugin
                                # can ship client-side behaviour with no tab
    settings_group: str = ""    # group label for this plugin's settings section
    group: str = ""             # nav category id (e.g. "studio"); the SPA collapses
                                # tabs sharing a group under one parent when 2+ are
                                # enabled, and shows a flat tab when exactly one is


#: The keys a plugin.toml [surface] table understands. Derived from the Surface
#: dataclass so it can never drift from what parse_spec actually reads; used to
#: warn (never fail) on an unknown/misspelled key (LM-DA-007).
KNOWN_SURFACE_KEYS = frozenset(Surface.__dataclass_fields__)


@dataclass
class PluginSpec:
    """Validated manifest of a plugin (superset of loader.PluginManifest)."""
    name: str
    version: str = "0.0.0"
    api_version: int = API_VERSION
    description: str = ""
    scope: str = ""                          # capability scope; defaults to name
    requires_extras: list = field(default_factory=list)   # pip extras to install
    requires: list = field(default_factory=list)          # other plugins this needs
    capabilities: list = field(default_factory=list)      # declared; shown at install
    data_subdir: str = ""                    # under the data dir; "" = none
    builtin: bool = False                    # ships in-tree
    protected: bool = False                  # cannot be disabled/uninstalled (chat)
    default_enabled: bool = False            # auto-enabled on first run (preinstalled #0)
    surface: Surface = field(default_factory=Surface)
    cli_entry: str = ""                      # legacy "module:attr" Click command
    register_entry: str = ""                 # "module:attr" -> register(host)
    path: Optional[str] = None               # plugin directory (third-party)

    def __post_init__(self) -> None:
        if not self.scope:
            self.scope = self.name

    def compatible(self) -> bool:
        """True when this plugin targets an api_version the engine supports."""
        return self.api_version == API_VERSION


@runtime_checkable
class Host(Protocol):
    """The API the engine hands a plugin at register() time. The plugin attaches
    itself through this object and never imports the app or global config
    directly - which is what lets a plugin be loaded/unloaded at runtime."""

    api_version: int

    def mount_router(self, router: Any) -> None: ...
    def mount_static(self, directory: str, *, url_prefix: str = "") -> str: ...
    def add_settings(self, fields: list) -> None: ...
    def register_tab(self, surface: Surface) -> None: ...
    # Config r/w is CONFINED to the plugin's own block; a different name is refused
    # (compartmentalisation). name is optional and defaults to the plugin's own.
    def plugin_config(self, name: Optional[str] = ...) -> dict: ...
    def save_plugin_config(self, name: Optional[str] = ..., cfg: Optional[dict] = ...) -> None: ...
    # Host-side scope checks are NOT implemented: the host has no request
    # context (scopes are enforced per-request on the routes mounted via
    # mount_router). Both raise NotImplementedError so a plugin that builds a
    # guard on them fails loudly at development time instead of shipping a
    # check that silently always passes (LM-DA-008).
    def has_scope(self, scope: str) -> bool: ...
    def require_scope(self, scope: str) -> None: ...
    def engine(self) -> Any: ...                          # inference engine handle
    def audit(self, event: str, data: dict) -> None: ...
    def browse_dirs(self, path: str) -> dict: ...         # server-side folder picker

    def register_chat_hook(self, phase: str, fn: Any, *,
                           priority: int = 0) -> None:
        """Register a transform run on every ``/v1/chat/completions`` turn.

        *phase* is one of:
          - ``"inlet"``  - ``fn(messages, ctx) -> messages`` (sync or async),
            run before token counting and inference.
          - ``"stream"`` - ``fn(token, ctx) -> token`` (SYNC; per streamed text
            piece on the hot path - an async fn is skipped).
          - ``"outlet"`` - ``fn(text, messages, ctx) -> text`` (sync or async),
            run on the final reply.

        Returning None from a hook keeps the prior value (mutate-in-place is
        fine). Lower *priority* runs first; ties keep registration order. A hook
        that raises is logged and skipped, never breaking the turn. The engine
        removes this plugin's hooks automatically on disable/uninstall.

        Hooks see scrubbed content text, not model control markers. In a
        streaming turn the outlet runs after all chunks have been sent (for
        record/side-effects only); use a stream hook to rewrite streamed output.
        """
        ...


@runtime_checkable
class Plugin(Protocol):
    """What a plugin module/object exposes. ``register`` / ``unregister`` are
    required. Optional lifecycle hooks: ``on_install`` (called once at install),
    ``on_uninstall(delete_data)`` (called at uninstall), and ``on_first_use``
    (called once the FIRST time the plugin is loaded/activated, persisted so it
    never re-fires on a later server start) are invoked if present."""

    def register(self, host: Host) -> None: ...
    def unregister(self) -> None: ...
