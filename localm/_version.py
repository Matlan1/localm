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


def _prerelease_suffix(v: str):
    """``(tag, number)`` from the trailing non-numeric remainder of *v*'s LAST
    dot-separated segment, or ``None`` when that segment is purely numeric (a
    final release, for ordering purposes). E.g. ``"0.1.4-rc1"`` -> ``("rc", 1)``;
    ``"0.1.4-rc"`` -> ``("rc", 0)``; ``"0.1.4"`` -> ``None``.

    Scoped to the LAST segment only (not a scan of the whole string) so this
    matches ``_parse``'s own per-segment leading-digit extraction exactly for
    every part before it - a non-numeric remainder in an EARLIER segment (not
    a shape this project's tags ever use) is silently dropped from the release
    tuple by ``_parse``, same as before this function existed; it is not also
    picked up here as a suffix. Not a full PEP 440/SemVer implementation
    (no epoch, build metadata, dev/post releases) - covers this project's
    actual scheme (``MAJOR.MINOR.PATCH`` with an optional ``-TAG[N]`` suffix)
    on purpose, per the "hand-rolled is fine for this version scheme, do not
    adopt `packaging`" call - `packaging` is only pulled in by this project's
    optional GPU/dev extras, not a safe dependency for a core-install module."""
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


def is_newer(candidate: str, current: str) -> bool:
    """True if *candidate* is a strictly newer version than *current*.

    Compares normalized numeric release tuples first; if THOSE tie, breaks the
    tie via each side's prerelease suffix (see ``_prerelease_suffix``): a bare
    release always outranks any prerelease of the SAME numeric version (so
    ``"0.1.4"`` is newer than ``"0.1.4-rc1"``, and updating FROM an rc TO the
    matching final release is offered), and between two prereleases of the
    same numeric version, a higher (tag, number) wins (so ``"0.1.4-rc2"`` is
    newer than ``"0.1.4-rc1"``). An exact tie (identical suffix, or both bare)
    is NOT newer, matching a plain re-offer of the same build.

    This function GATES anti-rollback (see ``updater._refuse_downgrade``) as
    well as whether an update is OFFERED - it must never become MORE
    permissive in the wrong direction. The prerelease tie-break above only
    ADDS resolution to cases that previously tied at ``False`` (never newer);
    it never turns an already-``True`` numeric comparison ``False``, and it
    never makes an actually-older or actually-equal candidate compare newer.

    ``"unknown"`` current => any real candidate is newer (so a fresh install
    with no signal still sees updates)."""
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
