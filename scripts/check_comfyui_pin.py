#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Warn in CI when the bundled ComfyUI pin falls behind upstream releases.

localm's managed-ComfyUI installer clones a single PINNED commit/tag
(``COMFYUI_PINNED_COMMIT`` / ``COMFYUI_PINNED_VERSION`` in
``localm/media/managed_comfy_fresh.py``) rather than tracking upstream HEAD.
This script reads the pinned version, asks the GitHub releases API for
upstream's current release list, and reports how many releases the pin is
behind.

This is a MAINTENANCE SIGNAL, not a build gate: it always exits 0, prints what
it found, and never fails the build. It runs in CI only - nothing under localm/
imports it, and it makes no network call from any user's install.

Tag ordering: ComfyUI tags look like "v0.31.1". Plain string comparison is
wrong across a digit-count change ("v0.9.2" > "v0.31.1" lexically, since '9'
> '3') so tags are parsed into an (int, int, ...) tuple and compared
numerically. Draft and prerelease entries are excluded from both "latest" and
the behind-count, matching what GitHub's own /releases/latest endpoint
excludes.

Fails soft on the API: unreachable, rate-limited, or a malformed/empty
response all print a clearly-labelled "could not check" and exit 0, never a
false "up to date".

Usage:
    python scripts/check_comfyui_pin.py               # compare against the real pin
    python scripts/check_comfyui_pin.py --pinned v0.9.2   # sanity-check a hypothetical pin

Stdlib only (urllib + re + json), so it runs anywhere without extra installs -
the CI job that runs this does not even need localm installed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO = "comfyanonymous/ComfyUI"
_CONSTANTS_PATH = (
    Path(__file__).resolve().parent.parent / "localm" / "media" / "managed_comfy_fresh.py"
)
_PIN_RE = re.compile(r'^COMFYUI_PINNED_VERSION\s*=\s*"([^"]+)"', re.M)
_VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)$")
_PER_PAGE = 100


# --------------------------------------------------------------------------- #
#  Reading the pin                                                            #
# --------------------------------------------------------------------------- #

def _pinned_version(path: Path = _CONSTANTS_PATH) -> str:
    """COMFYUI_PINNED_VERSION out of managed_comfy_fresh.py, by text, not import:
    importing that module would pull in hwdetect and the rest of localm's
    hardware stack."""
    text = path.read_text(encoding="utf-8")
    m = _PIN_RE.search(text)
    if not m:
        raise SystemExit(
            f"could not find COMFYUI_PINNED_VERSION in {path} - has the constant "
            "been renamed or reformatted?"
        )
    return m.group(1)


# --------------------------------------------------------------------------- #
#  Fetching upstream releases                                                 #
# --------------------------------------------------------------------------- #

def _fetch_releases_http(repo: str):
    """Real GitHub API call. Raises on any failure; callers must not let that
    propagate uncaught (see _fetch_releases)."""
    url = f"https://api.github.com/repos/{repo}/releases?per_page={_PER_PAGE}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "localm-comfyui-pin-check",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 - fixed https:// URL
        return json.loads(r.read().decode("utf-8"))


def _fetch_releases(repo: str = _REPO, *, opener=None) -> list | None:
    """Upstream's release list (newest-first-ish, GitHub does not guarantee an
    order), or None if it could not be obtained for ANY reason. None must be
    read as "unknown", never as "no releases" and never as "current" - see
    main()'s handling.

    *opener* is injectable so a caller can drive the unreachable-API and
    malformed-response paths with a plain function. It defaults to None and the
    module-level _fetch_releases_http is resolved INSIDE the function body, so
    every call re-reads the current module attribute and a monkeypatched
    replacement is honoured."""
    if opener is None:
        opener = _fetch_releases_http
    try:
        data = opener(repo)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        print(f"  (could not reach GitHub releases API: {e})", file=sys.stderr)
        return None
    if not isinstance(data, list):
        print(
            f"  (unexpected response shape from GitHub releases API: {type(data).__name__})",
            file=sys.stderr,
        )
        return None
    return data


# --------------------------------------------------------------------------- #
#  Comparison                                                                 #
# --------------------------------------------------------------------------- #

def _parse_version(tag) -> tuple[int, ...] | None:
    """'v0.31.1' -> (0, 31, 1); anything not a plain vX.Y[.Z...] tag -> None
    (a suffix like '-rc1' or a non-version tag is skipped, not guessed at).

    Tuple comparison, not string comparison: (0, 9, 2) < (0, 31, 1) is correct,
    while "v0.9.2" < "v0.31.1" as strings is FALSE ('9' > '3' lexically)."""
    if not isinstance(tag, str):
        return None
    m = _VERSION_RE.match(tag.strip())
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def _compare(pinned_tag: str, releases: list) -> dict:
    """Pure comparison: pinned tag + raw GitHub releases JSON -> a result dict.

    status is one of:
      "unparseable_pin"  pinned_tag itself is not a plain vX.Y[.Z] tag
      "no_data"           releases had nothing usable to compare against
      "current"           no eligible release is newer than the pin
      "stale"             at least one eligible release is newer; see "behind"
    """
    pinned_v = _parse_version(pinned_tag)
    if pinned_v is None:
        return {"status": "unparseable_pin"}

    eligible = []
    for rel in releases:
        if not isinstance(rel, dict):
            continue
        if rel.get("draft") or rel.get("prerelease"):
            continue
        v = _parse_version(rel.get("tag_name"))
        if v is None:
            continue
        eligible.append((v, rel.get("tag_name")))

    if not eligible:
        return {"status": "no_data"}

    latest_v, latest_tag = max(eligible, key=lambda pair: pair[0])
    behind = sum(1 for v, _ in eligible if v > pinned_v)

    if behind == 0:
        return {"status": "current", "latest": latest_tag}
    return {
        "status": "stale",
        "latest": latest_tag,
        "behind": behind,
        # True when the fetched page was full and more releases may exist.
        "capped": len(releases) >= _PER_PAGE,
    }


# --------------------------------------------------------------------------- #
#  Reporting                                                                  #
# --------------------------------------------------------------------------- #

def _report(pinned: str, result: dict) -> None:
    status = result["status"]
    if status == "unparseable_pin":
        print(
            f"COMFYUI PIN CHECK: could not parse pinned version {pinned!r} as "
            "vX.Y[.Z]; skipping the comparison."
        )
    elif status == "no_data":
        print(
            "COMFYUI PIN CHECK: could not check (the GitHub API returned no usable "
            f"release data). Bundled pin is {pinned}; upstream currency unknown this run."
        )
    elif status == "current":
        print(
            f"COMFYUI PIN CHECK: up to date. Bundled pin {pinned} is upstream's "
            "latest release."
        )
    elif status == "stale":
        or_more = " or more" if result["capped"] else ""
        print(
            f"COMFYUI PIN CHECK: bundled pin {pinned} is {result['behind']}{or_more} "
            f"release(s) behind upstream's latest, {result['latest']}."
        )
        print(
            "  Remedy: a maintainer tests the newer ComfyUI on real hardware and bumps "
            "COMFYUI_PINNED_COMMIT / COMFYUI_PINNED_VERSION in "
            "localm/media/managed_comfy_fresh.py. `localm comfy update` only ever moves "
            "an existing managed install to whatever that constant already says, so "
            "bumping the constant is what actually advances it."
        )
    else:  # pragma: no cover - _compare only ever returns the four statuses above
        raise AssertionError(f"unreachable status: {status!r}")


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pinned", default=None,
                    help="check this version instead of the real pin in "
                         "managed_comfy_fresh.py (for manually sanity-checking a "
                         "hypothetical pin; does not edit anything)")
    ap.add_argument("--repo", default=_REPO,
                    help=f"owner/repo to query (default: {_REPO})")
    args = ap.parse_args(argv)

    pinned = args.pinned or _pinned_version()

    releases = _fetch_releases(args.repo)
    if releases is None:
        print(
            "COMFYUI PIN CHECK: could not check (GitHub releases API unreachable). "
            f"Bundled pin is {pinned}; upstream currency unknown this run."
        )
        return 0

    _report(pinned, _compare(pinned, releases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
