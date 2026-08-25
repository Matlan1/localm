# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single source of truth for the RUNNING localm version."""

from __future__ import annotations

from pathlib import Path


def version_file() -> Path:
    """Path to the repo-root ``VERSION`` file (one level above the package dir)."""
    return Path(__file__).resolve().parents[1] / "VERSION"


def read_version() -> str:
    """The running version string."""
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
    """Normalize a release tag / version for comparison: strip a leading ``v`` (``v0.2.0`` -> ``0.2.0``)."""
    t = (tag or "").strip()
    if t[:1] in ("v", "V") and t[1:2].isdigit():
        return t[1:]
    return t


def _parse(v: str) -> tuple:
    """Best-effort version tuple for ordering."""
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


def _prerelease_suffix(v: str):
    """``(tag, number)`` from the trailing non-numeric remainder of *v*'s LAST dot-separated segment, or ``None`` when that segment is purely numeric (a final release, for ordering purposes)."""
    parts = normalize(v).split(".")
    if not parts:
        return None
    last = parts[-1]
    i = 0
    for ch in last:
        if ch.isdigit():
            i += 1
        else:
            break
    suffix = last[i:].lstrip("-_")
    if not suffix:
        return None
    tag = ""
    j = 0
    for ch in suffix:
        if ch.isalpha():
            tag += ch
            j += 1
        else:
            break
    tail_num = ""
    for ch in suffix[j:]:
        if ch.isdigit():
            tail_num += ch
        else:
            break
    return (tag.lower(), int(tail_num) if tail_num else 0)


def _leading_digit(v: str) -> bool:
    """Whether *v* (after ``normalize()``) starts with a digit - i.e. looks like an actual version number, as opposed to an arbitrary tag (``'stable'``, ``'nightly'``, ``'release-5'``)."""
    t = normalize(v)
    return bool(t) and t[0].isdigit()


def comparable(candidate: str, current: str) -> bool:
    """Whether ``is_newer(candidate, current)`` can meaningfully ORDER the two, as opposed to both silently degrading to the same ``(0,)`` tuple because one side is not a recognizable version number."""
    cand = normalize(candidate)
    cur = normalize(current)
    if not cand:
        return False
    if cur in ("", "unknown"):
        return True
    return _leading_digit(candidate) and _leading_digit(current)


def is_newer(candidate: str, current: str) -> bool:
    """True if *candidate* is a strictly newer version than *current*."""
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
    if ct != ut:
        return ct > ut
    cand_pre = _prerelease_suffix(cand)
    cur_pre = _prerelease_suffix(cur)
    if cand_pre == cur_pre:
        return False
    if cand_pre is None:
        return True     # candidate is the final release of a prerelease current is on
    if cur_pre is None:
        return False    # candidate is a prerelease; current is already the final release
    return cand_pre > cur_pre
