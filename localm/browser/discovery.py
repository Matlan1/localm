# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find the Chromium-family browsers installed on this machine.

The automated browser's "system" engine drives a browser the user already has
instead of the build localm downloads. This module lists what is installed:
Google Chrome (stable, beta, dev), Chromium, Microsoft Edge (stable, beta, dev)
and Brave, on Linux, macOS and Windows.

Playwright launches a browser by ``channel`` when it knows the browser's install
location (Chrome and Edge, at the locations in the table below) and by
``executable_path`` otherwise. A browser found at its channel location is
reported with that channel; a browser found anywhere else, and every Chromium
and Brave, is reported by path only.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

#: The browser families looked for, in the words a user would use.
LOOKED_FOR = ("Google Chrome", "Chromium", "Microsoft Edge", "Brave")


@dataclass(frozen=True)
class SystemBrowser:
    """One installed browser: its display name, the executable to launch, and
    the playwright channel that resolves to that same executable (None when
    playwright has no channel for it, or it is installed elsewhere)."""

    name: str
    path: str
    channel: Optional[str] = None

    def launch_options(self) -> dict:
        """The keyword arguments for playwright's ``chromium.launch`` that
        select this browser."""
        if self.channel:
            return {"channel": self.channel}
        return {"executable_path": self.path}


@dataclass(frozen=True)
class _Spec:
    name: str
    #: The playwright channel for this browser, and where playwright itself
    #: looks for it, per platform. Windows entries are relative to a program
    #: files directory; the others are absolute.
    channel: Optional[str] = None
    channel_paths: Mapping[str, str] = None
    #: Other absolute locations, per platform.
    paths: Mapping[str, tuple] = None
    #: Program names looked up on PATH, per platform.
    commands: Mapping[str, tuple] = None
    #: Locations relative to a program files directory, per platform (Windows).
    relative: Mapping[str, tuple] = None


_LINUX_COMMANDS_CHROME = ("google-chrome-stable", "google-chrome")
_LINUX_COMMANDS_EDGE = ("microsoft-edge-stable", "microsoft-edge")

# Stable releases first, then the pre-release channels.
_SPECS = (
    _Spec("Google Chrome", "chrome", {
        "linux": "/opt/google/chrome/chrome",  # hygiene-ok: vendor install location
        "darwin": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "win32": "Google/Chrome/Application/chrome.exe",
    }, commands={"linux": _LINUX_COMMANDS_CHROME}),
    _Spec("Chromium", None, None,
          paths={"linux": ("/usr/bin/chromium", "/usr/bin/chromium-browser",
                           "/snap/bin/chromium"),
                 "darwin": ("/Applications/Chromium.app/Contents/MacOS/Chromium",)},
          commands={"linux": ("chromium", "chromium-browser")},
          relative={"win32": ("Chromium/Application/chrome.exe",)}),
    _Spec("Microsoft Edge", "msedge", {
        "linux": "/opt/microsoft/msedge/msedge",  # hygiene-ok: vendor install location
        "darwin": "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "win32": "Microsoft/Edge/Application/msedge.exe",
    }, commands={"linux": _LINUX_COMMANDS_EDGE}),
    _Spec("Brave", None, None,
          paths={"linux": ("/opt/brave.com/brave/brave", "/snap/bin/brave"),  # hygiene-ok: vendor install location
                 "darwin": ("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",)},
          commands={"linux": ("brave-browser", "brave-browser-stable", "brave")},
          relative={"win32": ("BraveSoftware/Brave-Browser/Application/brave.exe",)}),
    _Spec("Google Chrome Beta", "chrome-beta", {
        "linux": "/opt/google/chrome-beta/chrome",  # hygiene-ok: vendor install location
        "darwin": "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
        "win32": "Google/Chrome Beta/Application/chrome.exe",
    }),
    _Spec("Microsoft Edge Beta", "msedge-beta", {
        "linux": "/opt/microsoft/msedge-beta/msedge",  # hygiene-ok: vendor install location
        "darwin": "/Applications/Microsoft Edge Beta.app/Contents/MacOS/Microsoft Edge Beta",
        "win32": "Microsoft/Edge Beta/Application/msedge.exe",
    }),
    _Spec("Google Chrome Dev", "chrome-dev", {
        "linux": "/opt/google/chrome-unstable/chrome",  # hygiene-ok: vendor install location
        "darwin": "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev",
        "win32": "Google/Chrome Dev/Application/chrome.exe",
    }),
    _Spec("Microsoft Edge Dev", "msedge-dev", {
        "linux": "/opt/microsoft/msedge-dev/msedge",  # hygiene-ok: vendor install location
        "darwin": "/Applications/Microsoft Edge Dev.app/Contents/MacOS/Microsoft Edge Dev",
        "win32": "Microsoft/Edge Dev/Application/msedge.exe",
    }),
)


def _platform_key(platform: Optional[str]) -> str:
    plat = platform if platform is not None else sys.platform
    if plat.startswith("linux"):
        return "linux"
    if plat == "darwin":
        return "darwin"
    if plat in ("win32", "cygwin"):
        return "win32"
    return plat


def _is_runnable(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    return os.name == "nt" or os.access(path, os.X_OK)


def _program_dirs(env: Mapping[str, str]) -> list:
    """The directories playwright itself searches for a Windows browser."""
    drive = env.get("HOMEDRIVE")
    dirs = [env.get("LOCALAPPDATA"), env.get("PROGRAMFILES"),
            env.get("PROGRAMFILES(X86)")]
    if drive:
        dirs += [drive + "\\Program Files", drive + "\\Program Files (x86)"]
    return [d for d in dirs if d]


def _under(base: str, relative: str) -> str:
    return os.path.join(base, *relative.split("/"))


def find_system_browsers(
        *, platform: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        which: Optional[Callable[[str], Optional[str]]] = None,
        runnable: Optional[Callable[[str], bool]] = None) -> list:
    """The installed browsers, most preferred first; empty when none is found.

    *platform*, *env*, *which* and *runnable* default to this machine's
    ``sys.platform``, ``os.environ``, ``shutil.which`` over ``env``'s PATH, and
    a file-exists-and-executable check. A browser is found at most once, at the
    first of these it has: its playwright channel location, another known
    location, a program name on PATH."""
    plat = _platform_key(platform)
    environ = os.environ if env is None else env
    if which is None:
        search_path = environ.get("PATH")

        def which(command: str) -> Optional[str]:
            return shutil.which(command, path=search_path)
    check = runnable or _is_runnable

    def locations(spec: _Spec):
        channel_path = (spec.channel_paths or {}).get(plat)
        if channel_path:
            if plat == "win32":
                for base in _program_dirs(environ):
                    yield _under(base, channel_path), spec.channel
            else:
                yield channel_path, spec.channel
        for path in (spec.paths or {}).get(plat, ()):
            yield path, None
        for relative in (spec.relative or {}).get(plat, ()):
            for base in _program_dirs(environ):
                yield _under(base, relative), None
        for command in (spec.commands or {}).get(plat, ()):
            resolved = which(command)
            if resolved:
                yield resolved, None

    found = []
    for spec in _SPECS:
        for path, channel in locations(spec):
            if check(path):
                found.append(SystemBrowser(spec.name, path, channel))
                break
    return found
