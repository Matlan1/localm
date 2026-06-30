# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single source of truth for the RUNNING localm version.

Read live from the repo-root ``VERSION`` file, NOT ``importlib.metadata``: the
install is editable (``uv pip install -e``), so a code-only update that swaps files
and reboots changes the ``VERSION`` file but does NOT refresh the installed
dist-info. The updater compares this live value against the latest GitHub Release
tag, so it must reflect the files on disk right now - hence a file read.

Falls back to the installed distribution metadata (non-editable installs have no
repo-root ``VERSION`` next to the package), then to ``"unknown"``. Never raises.
"""

from __future__ import annotations

from pathlib import Path


def version_file() -> Path:
    """Path to the repo-root ``VERSION`` file (one level above the package dir)."""
    return Path(__file__).resolve().parents[1] / "VERSION"


def read_version() -> str:
    """The running version string. VERSION file (live) > installed metadata >
    ``"unknown"``. Never raises."""
    try:
        text = version_file().read_text(encoding="utf-8").strip()
        if text:
            return text.splitlines()[0].strip()
    except OSError:
        pass
    try:
        from importlib.metadata import version
        return version("localm")
    except Exception:
        return "unknown"


def normalize(tag: str) -> str:
    """Normalize a release tag / version for comparison: strip a leading ``v``
    (``v0.2.0`` -> ``0.2.0``). Leaves anything else untouched."""
    t = (tag or "").strip()
    if t[:1] in ("v", "V") and t[1:2].isdigit():
        return t[1:]
    return t


def _parse(v: str) -> tuple:
    """Best-effort version tuple for ordering. Splits on '.', takes the leading
    integer of each part (so ``1.2.0rc1`` -> ``(1, 2, 0)``); non-numeric leading
    parts become 0. Used only for an ordering hint, never for equality."""
    out = []
    for part in normalize(v).split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        out.append(int(num) if num else 0)
    return tuple(out) or (0,)


def is_newer(candidate: str, current: str) -> bool:
    """True if *candidate* is a strictly newer version than *current*.

    Compares normalized numeric tuples; if they tie numerically but the strings
    differ (e.g. a build suffix), treat *candidate* as NOT newer (avoid offering an
    update that is not actually ahead). ``"unknown"`` current => any real candidate
    is newer (so a fresh install with no signal still sees updates)."""
    cand = normalize(candidate)
    cur = normalize(current)
    if not cand:
        return False
    if cur in ("", "unknown"):
        return True
    ct, ut = _parse(cand), _parse(cur)
    # Pad to equal length for a fair tuple compare.
    n = max(len(ct), len(ut))
    ct = ct + (0,) * (n - len(ct))
    ut = ut + (0,) * (n - len(ut))
    return ct > ut
