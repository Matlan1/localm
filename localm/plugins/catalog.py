# SPDX-License-Identifier: AGPL-3.0-or-later
"""The first-party plugin catalog: core's ONLY knowledge of plugins it does not have installed."""

from __future__ import annotations

from dataclasses import dataclass

# Base for first-party plugin repos. Left blank until the plugins are published
# to their own repos; the install fallback only tries GitHub when a repo is set
# (until then a first-party plugin is always sourced from the bundled store).
GITHUB_BASE = ""


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    description: str = ""
    commands: tuple = ()          # command verbs that imply this plugin
    extra: str = ""               # pip extra carrying its heavy dependencies
    github: str = ""              # explicit repo URL (else derived from GITHUB_BASE)
    preinstalled: bool = False    # installed out of the box (chat)
    protected: bool = False       # cannot be uninstalled or disabled (chat)

    def source_url(self) -> str:
        """Repo to fetch from when the plugin is missing from the bundled store."""
        if self.github:
            return self.github
        if GITHUB_BASE:
            return f"{GITHUB_BASE.rstrip('/')}/localm-plugin-{self.name}"
        return ""


# The bundled first-party catalog. Order is display order.
CATALOG: tuple = (
    CatalogEntry("chat", "Chat with your local model",
                 preinstalled=True, protected=True),
    CatalogEntry("coder", "Offline AI coding agent",
                 commands=("coder",), extra="coder"),
    CatalogEntry("image", "Image generation (ComfyUI by default)",
                 commands=("generate-image",)),
    CatalogEntry("music", "Music generation (ComfyUI ACE-Step by default)",
                 commands=("generate-music",)),
    CatalogEntry("video", "Short-video generation (ComfyUI Wan by default)",
                 commands=("generate-video",)),
    CatalogEntry("rag", "Document collections for retrieval-augmented chat",
                 commands=("knowledge",)),
    CatalogEntry("web", "Web search and page fetch for chat",
                 commands=("search-web",)),
    CatalogEntry("memory", "Remember durable facts about you across chats"),
    CatalogEntry("voice", "Speech-to-text for the chat mic button",
                 extra="voice"),
    CatalogEntry("tts", "Text-to-speech (neural voices) for reading replies aloud"),
    CatalogEntry("mcp", "Expose your local models to MCP clients",
                 commands=("mcp",)),
    CatalogEntry("jobs", "Scheduled recurring tasks",
                 commands=("job",)),
)

_BY_NAME = {e.name: e for e in CATALOG}
_BY_COMMAND = {cmd: e for e in CATALOG for cmd in e.commands}


def get(name: str) -> "CatalogEntry | None":
    return _BY_NAME.get(name)


def for_command(command: str) -> "CatalogEntry | None":
    """The first-party plugin that provides *command*, if any (for the 'that needs the X plugin - install it?' hint)."""
    return _BY_COMMAND.get(command)


def commands() -> dict:
    """Map of command verb -> providing plugin name, across the whole catalog."""
    return {cmd: e.name for cmd, e in _BY_COMMAND.items()}


def suggestion(command: str) -> "str | None":
    """A friendly 'that command needs the X plugin' hint for *command* when it belongs to a known first-party plugin that is not handling it (because the plugin is not installed or not enabled), else None for a truly unknown command."""
    entry = for_command(command)
    if entry is None or entry.preinstalled:
        return None
    return (f"/{command} needs the {entry.name} plugin ({entry.description}). "
            f"Install it from the Plugins page, or run: "
            f"localm plugin install {entry.name}")


def names() -> tuple:
    return tuple(_BY_NAME)


def preinstalled() -> tuple:
    """Plugins that ship installed out of the box (chat)."""
    return tuple(e.name for e in CATALOG if e.preinstalled)


def protected() -> frozenset:
    """Plugins that cannot be uninstalled or disabled (chat)."""
    return frozenset(e.name for e in CATALOG if e.protected)
