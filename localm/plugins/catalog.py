# SPDX-License-Identifier: AGPL-3.0-or-later
"""The first-party plugin catalog: core's ONLY knowledge of plugins it does not
have installed.

localm itself is just a model loader + chat (chat being the preinstalled,
protected plugin #0) + the plugin engine. It knows nothing about the other
features except this small static catalog, which lets it:

  1. Tell a surface which chat commands each first-party plugin provides, so
     the chat composer can say "/remember needs the memory plugin" for a plugin
     that is off. The hint can be turned off (config: suggest_plugins=False).
  2. Resolve WHERE to install a first-party plugin from: the bundled store on
     disk. A plugin missing from the store has no source today; install raises
     NotImplementedError when the entry carries a repo URL and KeyError otherwise.
  3. List what is available to install (the catalog minus what is installed).

This module holds NO plugin code and imports nothing heavy; it is just data.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache

# Base for first-party plugin repos; blank, so a first-party plugin is always
# sourced from the bundled store. A set repo URL is not fetched.
GITHUB_BASE = ""


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    description: str = ""
    commands: tuple = ()          # chat composer commands (and aliases) it provides
    extra: str = ""               # pip extra carrying its heavy dependencies
    github: str = ""              # explicit repo URL (else derived from GITHUB_BASE)
    preinstalled: bool = False    # installed out of the box (chat)
    protected: bool = False       # cannot be uninstalled or disabled (chat)
    offered: bool = True          # listed as available to install; False keeps
                                  # it out of every listing while an explicit
                                  # `localm plugin install <name>` still works
    opt_in: bool = False          # never part of "install everything"
    needs_nvidia_linux: bool = False  # runs only on Linux with an NVIDIA GPU

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
    CatalogEntry("coder", "Offline AI coding agent"),
    CatalogEntry("browser", "An automated web browser the model can drive",
                 extra="browser"),
    CatalogEntry("image", "Image generation (built in, or ComfyUI)",
                 commands=("generate-image", "imagine")),
    CatalogEntry("music", "Music generation (ACE-Step, native or through ComfyUI)",
                 commands=("generate-music", "music")),
    CatalogEntry("video", "Short-video generation (built in, or ComfyUI)",
                 commands=("generate-video", "video")),
    CatalogEntry("rag", "Document collections for retrieval-augmented chat"),
    CatalogEntry("web", "Web search and page fetch for chat",
                 commands=("web",)),
    CatalogEntry("memory", "Remember durable facts about you across chats",
                 commands=("remember", "memory")),
    CatalogEntry("voice", "Speech-to-text for the chat mic button",
                 extra="voice"),
    CatalogEntry("tts", "Text-to-speech (neural voices) for reading replies aloud"),
    CatalogEntry("mcp", "Expose your local models to MCP clients"),
    CatalogEntry("jobs", "Scheduled recurring tasks"),
    CatalogEntry("exllama", "Run EXL2 and EXL3 models with ExLlama (NVIDIA GPU, Linux). "
                 "Tested by the project without a GPU so far; bug reports welcome",
                 extra="exllama", opt_in=True, needs_nvidia_linux=True),
)

_BY_NAME = {e.name: e for e in CATALOG}
_BY_COMMAND = {cmd: e for e in CATALOG for cmd in e.commands}


def get(name: str) -> CatalogEntry | None:
    return _BY_NAME.get(name)


def for_command(command: str) -> CatalogEntry | None:
    """The first-party plugin that provides the chat command *command*, if any."""
    return _BY_COMMAND.get(command)


def commands() -> dict:
    """Map of chat command -> providing plugin name, across the whole catalog.
    Lets the chat composer recognise a command that belongs to a known but
    inactive plugin instead of treating it as unknown."""
    return {cmd: e.name for cmd, e in _BY_COMMAND.items()}


def names() -> tuple:
    return tuple(_BY_NAME)


@lru_cache(maxsize=1)
def _nvidia_linux() -> bool:
    """Whether this machine is Linux with an NVIDIA GPU detected (cached for the
    process; hwdetect never raises)."""
    from localm import hwdetect
    return hwdetect.platform_key() == "linux" and "nvidia" in hwdetect.detect().vendors


# Held across the hardware check, which is bounded by hwdetect's subprocess
# timeout. Taken only on worker threads, and no other lock is taken under it.
_hardware_check_lock = threading.Lock()
_hardware_checked = False


def check_hardware() -> None:
    """Run the hardware check :func:`offered` depends on and cache its answer.
    The first call of a process runs ``lspci`` on Linux, for up to 8 s, so call
    this off the event loop. Concurrent callers wait for one check. Never raises."""
    global _hardware_checked
    with _hardware_check_lock:
        _nvidia_linux()
        _hardware_checked = True


async def check_hardware_off_loop() -> None:
    """:func:`check_hardware` on a worker thread, unless it has already run.
    Afterwards :func:`offered` starts no process and can run on the event loop."""
    if not _hardware_checked:
        import asyncio
        await asyncio.to_thread(check_hardware)


def runnable_here(entry: CatalogEntry) -> bool:
    """False when *entry* needs hardware or a platform this machine lacks."""
    return not entry.needs_nvidia_linux or _nvidia_linux()


def not_runnable_reason(entry: CatalogEntry) -> str:
    """Why *entry* cannot run on this machine, or "" when it can."""
    if runnable_here(entry):
        return ""
    return "needs Linux with an NVIDIA GPU, which this machine does not have"


def offered_entry(entry: CatalogEntry) -> bool:
    """Whether *entry* is offered by setup on this machine: offered, and able
    to run here."""
    return entry.offered and runnable_here(entry)


def listed(name: str) -> bool:
    """Whether *name* is listed as available to install: False only for a
    catalog entry that is not offered. An entry that cannot run on this
    machine is listed, with :func:`not_runnable_reason` saying why."""
    entry = _BY_NAME.get(name)
    return entry is None or entry.offered


def offered(name: str) -> bool:
    """False for a catalog entry that is not offered or cannot run on this
    machine; True for every other name, including one the catalog does not
    list."""
    entry = _BY_NAME.get(name)
    return entry is None or offered_entry(entry)


def preinstalled() -> tuple:
    """Plugins that ship installed out of the box (chat)."""
    return tuple(e.name for e in CATALOG if e.preinstalled)


def protected() -> frozenset:
    """Plugins that cannot be uninstalled or disabled (chat)."""
    return frozenset(e.name for e in CATALOG if e.protected)
