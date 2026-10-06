# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turn a failed browser launch into words a localm user can act on.

Playwright's launch errors carry its own advice: a boxed banner telling the
reader to run ``playwright install``, ``playwright install-deps`` or
``npx playwright ...``, none of which a localm user has. This module reads such
an error, classifies it, drops that advice, and states the underlying reason
with localm's own remedy. The raw text is for the debug log, not the user.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Iterable, Optional

#: Where the browser settings live in the GUI.
SETTINGS_PLACE = "Settings > Server & network"

#: Where a user downloads the bundled browser, and the equivalent command.
DOWNLOAD_ADVICE = ("Download it with the Download browser button in "
                   f"{SETTINGS_PLACE} or on the Browser tab, or run:  "
                   "localm setup-browser")

BUNDLED_MISSING = "bundled_missing"
SYSTEM_MISSING = "system_missing"
MISSING_LIBRARIES = "missing_libraries"
OTHER = "other"

#: Box-drawing characters playwright draws its banner with.
_BOX = "╔╗╚╝║═"

_ADVICE = re.compile(
    r"playwright(-cli)?\s+install|npx\s+playwright|install-deps|"
    r"<3\s+Playwright|Looks like Playwright|"
    r"Please run the following command|"
    r"Please install them with the following command|"
    r"^Alternatively, use apt|^Run\s+\"playwright|"
    r"Docker image|^Either:|^- \(",
    re.IGNORECASE)

_PREFIX = re.compile(r"^(?:BrowserType|Browser)\.[A-Za-z_]+:\s*")
_SONAME = re.compile(r"\blib[A-Za-z0-9_.+-]*\.so(?:\.[0-9][0-9.]*)?")
_LOADER = re.compile(r"error while loading shared libraries:\s*(\S+?):")
_LOADER_LINE = re.compile(
    r"^(?:\[pid=\d+\]\[err\]\s*)?(.+?):\s+error while loading shared libraries")
_STDERR_LINE = re.compile(r"^\[pid=\d+\]\[err\]\s*(.*)")
_LDD_MISSING = re.compile(r"^\s*(\S+)\s+=>\s+not found", re.MULTILINE)
_PACKAGE = re.compile(r"[a-z0-9][a-z0-9.+-]*")
_NOT_FOUND = re.compile(r"distribution '[^']*' is not found", re.IGNORECASE)

#: Libraries or packages named in a message before it is cut to "and N more".
_LIST_LIMIT = 8

#: Characters of a reason kept in a message.
_REASON_LIMIT = 300


def plain_lines(raw: object) -> list:
    """*raw* as its non-empty lines with any box-drawing frame removed."""
    lines = []
    for line in str(raw).splitlines():
        text = line.strip().strip(_BOX).strip()
        if text:
            lines.append(text)
    return lines


def classify(raw: object, engine: str) -> str:
    """Which kind of launch failure *raw* describes under *engine*."""
    lower = str(raw).lower()
    if ("host system is missing dependencies" in lower
            or "error while loading shared libraries" in lower):
        return MISSING_LIBRARIES
    if "executable doesn't exist" in lower:
        return SYSTEM_MISSING if engine == "system" else BUNDLED_MISSING
    if _NOT_FOUND.search(lower):
        return SYSTEM_MISSING
    return OTHER


def _ldd_missing(path: str) -> list:
    """The shared libraries the dynamic linker cannot find for the executable
    at *path*, all of them; empty off Linux, when *path* is not a file, or when
    ``ldd`` cannot be run."""
    if not sys.platform.startswith("linux") or not os.path.isfile(path):
        return []
    try:
        done = subprocess.run(["ldd", path], capture_output=True, text=True,
                              timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return _LDD_MISSING.findall(done.stdout)


def missing_libraries(raw: object) -> tuple:
    """``(libraries, packages)`` named in a missing-dependencies error: the
    shared libraries missing, and the Debian package names playwright
    suggested. Either may be empty. The dynamic loader reports one missing
    library per start, so when it names the executable, ``ldd`` supplies the
    full list."""
    libraries = []
    packages = []
    mode = None
    checked = set()
    for line in plain_lines(raw):
        loader = _LOADER_LINE.match(line)
        if loader and loader.group(1) not in checked:
            checked.add(loader.group(1))
            libraries.extend(_ldd_missing(loader.group(1)))
        libraries.extend(_LOADER.findall(line))
        if line.lower().startswith("missing libraries"):
            mode = "libraries"
            continue
        if "apt-get install" in line:
            mode = "packages"
            line = line.split("apt-get install", 1)[1]
        if mode == "libraries":
            libraries.extend(_SONAME.findall(line))
        elif mode == "packages":
            if _ADVICE.search(line):
                mode = None
                continue
            packages.extend(_PACKAGE.findall(line.replace("\\", " ")))
    return _unique(libraries), _unique(packages)


def _unique(items: Iterable[str]) -> list:
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _listed(items: list) -> str:
    shown = ", ".join(items[:_LIST_LIMIT])
    extra = len(items) - _LIST_LIMIT
    return shown + (f" and {extra} more" if extra > 0 else "")


def reason(raw: object) -> str:
    """The reason *raw* states on one line, without playwright's advice, its
    launch log or its boxed banner, followed by what the browser process itself
    printed to stderr when the error carries that."""
    kept = []
    stderr = []
    lines = plain_lines(raw)
    for line in lines:
        printed = _STDERR_LINE.match(line)
        if printed and printed.group(1) not in stderr:
            stderr.append(printed.group(1))
    for line in lines:
        if line.lower().startswith(("call log", "browser logs")):
            break
        if _ADVICE.search(line):
            continue
        kept.append(_PREFIX.sub("", line))
    if stderr:
        kept.append("the browser reported: " + " / ".join(stderr[:3]))
    text = "; ".join(kept).strip()
    if len(text) > _REASON_LIMIT:
        text = text[:_REASON_LIMIT].rstrip() + "..."
    return text or "the browser did not start"


def _libraries_sentence(raw: object) -> str:
    libraries, packages = missing_libraries(raw)
    parts = []
    if libraries:
        parts.append("Missing: " + _listed(libraries) + ".")
    if packages:
        parts.append("On Debian and Ubuntu these come from the packages: "
                     + _listed(packages) + ".")
    parts.append("Install them with your system's package manager, then try again.")
    return " ".join(parts)


def launch_failure(raw: object, *, engine: str,
                   browser: Optional[str] = None) -> tuple:
    """``(kind, message)`` for a failed launch under *engine* of *browser*, or
    of the bundled build when *browser* is None. The message is one line and
    names no playwright command."""
    kind = classify(raw, engine)
    if kind == BUNDLED_MISSING:
        return kind, ("The bundled browser has not been downloaded yet. It is a "
                      "one-time download, separate from the localm install. "
                      + DOWNLOAD_ADVICE + ".")
    name = browser or "the bundled browser"
    if kind == MISSING_LIBRARIES:
        return kind, (f"{name[0].upper() + name[1:]} is installed but cannot "
                      "start, because this computer lacks system libraries it "
                      "needs. " + _libraries_sentence(raw))
    if kind == SYSTEM_MISSING:
        return kind, (f"{name} could not be found where it was expected. "
                      "Reinstall it, or set Browser to drive to 'bundled' in "
                      f"{SETTINGS_PLACE}.")
    tail = (f" You can set Browser to drive to 'bundled' in {SETTINGS_PLACE}."
            if engine == "system" else "")
    return kind, f"Could not start {name}: {reason(raw).rstrip('.')}.{tail}"


def _short_reason(raw: object) -> str:
    if classify(raw, "system") == MISSING_LIBRARIES:
        libraries, _ = missing_libraries(raw)
        return ("missing system libraries"
                + (": " + _listed(libraries) if libraries else ""))
    return reason(raw).rstrip(".")


def system_launch_failure(attempts: list) -> tuple:
    """``(kind, message)`` for installed browsers that all failed to start.
    *attempts* is a non-empty list of ``(browser name, raw error)``."""
    if len(attempts) == 1:
        name, raw = attempts[0]
        return launch_failure(raw, engine="system", browser=name)
    detail = "; ".join(f"{name}: {_short_reason(raw)}" for name, raw in attempts)
    return OTHER, (f"None of the installed browsers could be started ({detail}). "
                   f"You can set Browser to drive to 'bundled' in {SETTINGS_PLACE}.")


def no_system_browser(looked_for: Iterable[str]) -> str:
    """The message for a system engine with no browser to drive."""
    names = list(looked_for)
    listed = (", ".join(names[:-1]) + " and " + names[-1]
              if len(names) > 1 else "".join(names))
    return ("No supported browser was found on this computer. localm looked for "
            f"{listed}. Install one of them, or set Browser to drive to "
            f"'bundled' in {SETTINGS_PLACE} and download the bundled browser.")
