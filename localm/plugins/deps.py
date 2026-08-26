# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-side installer for a plugin's declared pip extras (``requires_extras``).

This module ONLY ever installs localm's own declared extra requirements,
resolved from the installed distribution's metadata - never an arbitrary
package name handed in by a caller. A caller passes an extra NAME (e.g.
``"voice"``); we look up what that extra maps to in localm's ``Requires-Dist``
and install exactly those specifiers.

It runs on the HOST only: the CLI, or a loopback GUI request. A remote client
must never reach the install path (enforced at the route). Failures surface the
real pip/uv output instead of being swallowed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from localm import config
from localm.debuglog import logger

#: Progress sink: called with one human-readable line at a time (a pip output
#: line, or a status note). Best-effort; a raising sink never breaks an install.
ProgressCb = Optional[Callable[[str], None]]

#: The installed distribution whose extras are resolved. A module constant, so
#: it can be repointed at another distribution.
DIST_NAME = "localm"


@dataclass
class InstallResult:
    """Outcome of an install attempt. ``ok`` is True only when every required
    package ended up satisfied."""
    ok: bool = True
    installed: list = field(default_factory=list)   # specifiers we installed
    skipped: list = field(default_factory=list)     # already satisfied, untouched
    failed: list = field(default_factory=list)      # specifiers still missing
    error: str = ""                                 # trimmed failure tail
    log: str = ""                                   # full combined pip output

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "installed": self.installed, "skipped": self.skipped,
            "failed": self.failed, "error": self.error,
        }


def _emit(cb: ProgressCb, line: str) -> None:
    if cb is None:
        return
    try:
        cb(line)
    except Exception:  # a broken progress sink must never abort an install
        logger.debug("progress sink raised (ignored)", exc_info=True)


def _marker_names_extra(marker: str, extra: str) -> bool:
    """True when an environment marker selects ``extra == '<extra>'``."""
    return bool(re.search(r"extra\s*==\s*['\"]%s['\"]" % re.escape(extra), marker))


def extra_requirements(extra: str) -> list:
    """Concrete requirement strings for one pip extra, read from the installed
    ``localm`` metadata. Falls back to the ``localm[extra]`` specifier when the
    metadata cannot be read, so the install is still attempted."""
    import importlib.metadata as md
    try:
        reqs = md.metadata(DIST_NAME).get_all("Requires-Dist") or []
    except Exception:
        logger.debug("could not read %s metadata for extra %r", DIST_NAME, extra,
                     exc_info=True)
        return [f"{DIST_NAME}[{extra}]"]
    out = []
    for r in reqs:
        # e.g. "faster-whisper>=1.0; extra == 'voice'"
        head, _, marker = r.partition(";")
        if marker and _marker_names_extra(marker, extra):
            out.append(head.strip())
    return out or [f"{DIST_NAME}[{extra}]"]


def plugin_requirements(extras: Iterable[str]) -> list:
    """Flattened, de-duplicated requirement strings for a set of extras."""
    seen, out = set(), []
    for e in extras or ():
        for r in extra_requirements(e):
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out


def _req_name(req: str) -> str:
    """The distribution name from a requirement string (drops version/marker)."""
    return re.split(r"[<>=!~ \[;@]", req.strip(), 1)[0].strip()


def is_satisfied(req: str) -> bool:
    """True when *req* is already installed. Version-aware when ``packaging`` is
    importable, otherwise name-presence only. An ``extra`` specifier form
    (``localm[voice]``) is never treated as satisfied - it means we could not
    resolve concrete packages, so the install should still run."""
    if "[" in req:                       # unresolved localm[extra] fallback form
        return False
    name = _req_name(req)
    if not name:
        return False
    import importlib.metadata as md
    try:
        installed = md.version(name)
    except Exception:
        return False
    try:
        from packaging.requirements import Requirement
        spec = Requirement(req).specifier
        return spec.contains(installed, prereleases=True) if str(spec) else True
    except Exception:
        # packaging missing or an odd specifier: fall back to present-by-name.
        return True


def missing_requirements(reqs: Iterable[str]) -> list:
    """The subset of *reqs* not already satisfied, order preserved."""
    return [r for r in reqs if not is_satisfied(r)]


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def _run_pip(reqs: list, *, on_progress: ProgressCb = None):
    """Install *reqs* into the CURRENT interpreter's environment. Tries ``uv pip
    install`` first, then ``pip``, streaming output to *on_progress*. Returns
    ``(ok, combined_output)``. ``--python sys.executable`` pins uv to this venv
    regardless of the ambient VIRTUAL_ENV.

    ``env`` pins uv's AND pip's caches inside the data dir. Both are set: the uv
    attempt runs first and pip second, and each caches to a per-user location
    OUTSIDE the data dir when left to its default. See
    ``config.contained_pip_env``."""
    env = config.contained_pip_env()
    attempts = (
        ["uv", "pip", "install", "--python", sys.executable, *reqs],
        [sys.executable, "-m", "pip", "install", *reqs],
    )
    last = ""
    for cmd in attempts:
        _emit(on_progress, "$ " + " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env)
        except FileNotFoundError:
            _emit(on_progress, f"{cmd[0]} not found, trying next installer...")
            continue
        lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            _emit(on_progress, line.rstrip())
        proc.wait()
        out = "".join(lines)
        if proc.returncode == 0:
            return True, out
        last = out
        # Keep the real failure for the caller instead of discarding it.
        logger.debug("%s install failed (rc=%s): %s", cmd[0], proc.returncode,
                     out[-2000:])
    return False, last


def install_requirements(reqs: Iterable[str], *,
                         on_progress: ProgressCb = None) -> InstallResult:
    """Install any of *reqs* that are not already satisfied. Re-checks after the
    install so a pip that exits 0 but did not actually provide a package is still
    reported as failed."""
    reqs = list(reqs)
    todo = missing_requirements(reqs)
    res = InstallResult(ok=True, skipped=[r for r in reqs if r not in todo])
    if not todo:
        _emit(on_progress, "All dependencies already satisfied.")
        return res
    _emit(on_progress, "Installing: " + ", ".join(todo))
    ok, out = _run_pip(todo, on_progress=on_progress)
    res.log = out
    if not ok:
        res.ok = False
        res.failed = todo
        res.error = _tail(out) or "installer not available (need uv or pip)"
        _emit(on_progress, "Install failed: " + res.error)
        return res
    still = missing_requirements(todo)
    res.installed = [r for r in todo if r not in still]
    if still:
        res.ok = False
        res.failed = still
        res.error = "installer reported success but still missing: " + ", ".join(still)
        _emit(on_progress, res.error)
    else:
        _emit(on_progress, "Done.")
    return res


def install_plugin_extras(extras: Iterable[str], *,
                          on_progress: ProgressCb = None) -> InstallResult:
    """Resolve *extras* to requirements and install the missing ones. A plugin
    with no extras is a no-op success."""
    reqs = plugin_requirements(extras)
    if not reqs:
        _emit(on_progress, "No pip extras declared for this plugin.")
        return InstallResult(ok=True)
    return install_requirements(reqs, on_progress=on_progress)
