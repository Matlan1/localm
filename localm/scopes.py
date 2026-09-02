# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Capability / permission scopes for localm.

A *scope* is a single capability string that an API key (permission system) can
be granted, and that a route, plugin, or chat tool can require. Every plugin
owns a scope equal to its name (the coder plugin -> "coder"); cross-cutting
kernel capabilities have explicit scopes ("models:write", "config:write", ...).

This module is the single source of truth shared by the permission system, the
plugin engine, and the chat control surface.
"""

from __future__ import annotations

# --- Kernel capabilities (not owned by any one plugin) --------------------- #
MODELS_READ   = "models:read"     # list / inspect models
MODELS_WRITE  = "models:write"    # load / unload / pull / remove / alias models
CONFIG_READ   = "config:read"     # read settings
CONFIG_WRITE  = "config:write"    # change settings
PLUGINS_READ  = "plugins:read"    # list / inspect plugins
PLUGINS_ADMIN = "plugins:admin"   # enable/disable/install/uninstall (privileged)
KEYS_ADMIN    = "keys:admin"      # create / scope / revoke API keys (privileged)
ADMIN         = "admin"           # wildcard owner scope: implies every scope

# First-party plugin capabilities (each == the plugin's name). A third-party
# plugin registers its own scope at install time via is_valid_scope(extra=...).
CHAT  = "chat"     # the built-in, always-enabled chat plugin
CODER = "coder"            # restricted coder: read + confined edit, no shell
CODER_FULL = "coder:full"  # unrestricted coder (shell exec); PRIVILEGED, owner-only to mint
IMAGE = "image"
MUSIC = "music"
VIDEO = "video"
RAG   = "rag"
WEB   = "web"
VOICE = "voice"
MCP   = "mcp"
BROWSER = "browser"

KERNEL_SCOPES: dict[str, str] = {
    MODELS_READ:   "List and inspect installed models",
    MODELS_WRITE:  "Load, unload, download, remove, or alias models",
    CONFIG_READ:   "Read settings",
    CONFIG_WRITE:  "Change settings",
    PLUGINS_READ:  "List and inspect plugins",
    PLUGINS_ADMIN: "Enable, disable, install, or uninstall plugins (privileged)",
    KEYS_ADMIN:    "Create, scope, and revoke API keys (privileged)",
    ADMIN:         "Full access - implies every other capability",
}

# First-party plugin scopes shipped in-tree (third-party add their own at install).
BUILTIN_PLUGIN_SCOPES: dict[str, str] = {
    CHAT:  "Chat (built-in, always enabled)",
    CODER: "AI coding agent - restricted: read + edit files (no shell)",
    IMAGE: "Image generation",
    MUSIC: "Music generation",
    VIDEO: "Video generation",
    RAG:   "Knowledge base / retrieval",
    WEB:   "Web search and fetch",
    VOICE: "Speech-to-text",
    MCP:   "MCP server interface",
    BROWSER: "Automated web browser (navigate, read, click, fill)",
}

# Capability variants that are not a plugin name of their own.
EXTRA_SCOPES: dict[str, str] = {
    CODER_FULL: "AI coding agent - FULL: shell execution + edits (privileged)",
}

# Scopes that are never granted implicitly or to an untrusted key.
PRIVILEGED_SCOPES = frozenset(
    {PLUGINS_ADMIN, KEYS_ADMIN, CONFIG_WRITE, ADMIN, CODER_FULL})


def all_known_scopes() -> set[str]:
    """Every scope localm ships with (kernel + first-party plugins + variants)."""
    return set(KERNEL_SCOPES) | set(BUILTIN_PLUGIN_SCOPES) | set(EXTRA_SCOPES)


def is_valid_scope(scope: str, *, extra: set[str] | None = None) -> bool:
    """True if *scope* is a known kernel/first-party scope, a registered
    third-party scope (*extra*), or a well-formed plugin-name scope."""
    if not isinstance(scope, str) or not scope:
        return False
    if scope in all_known_scopes():
        return True
    if extra and scope in extra:
        return True
    # plugin-name scope: a bare identifier (a plugin owns the scope == its name)
    return scope.replace("-", "_").isidentifier()


def grants(held: set[str], required: str) -> bool:
    """Does a key holding *held* scopes satisfy *required*? ADMIN implies every
    scope; coder:full implies the base coder scope, so a coder:full key passes
    routes gated on plain coder."""
    if ADMIN in held or required in held:
        return True
    if required == CODER and CODER_FULL in held:
        return True
    return False


def normalize(scopes) -> list[str]:
    """De-dupe and sort a scope iterable into a stable list (drops blanks)."""
    return sorted({s.strip() for s in scopes if isinstance(s, str) and s.strip()})
