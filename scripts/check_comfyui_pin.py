#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Warn in CI when the bundled ComfyUI pin falls behind upstream releases.

localm's managed-ComfyUI installer clones a single PINNED commit/tag
(``COMFYUI_PINNED_COMMIT`` / ``COMFYUI_PINNED_VERSION`` in
``localm/media/managed_comfy_fresh.py``) rather than tracking upstream HEAD.
This script reads the pinned version, asks the GitHub releases API for
upstream's current release list, and reports how many releases the pin is
behind.

TWO MODES, the same shape as scripts/check_llama_pin.py:

  * default: a MAINTENANCE SIGNAL. Always exits 0, prints what it found, never
    fails the build. It runs in CI only - nothing under localm/ imports it, and
    it makes no network call from any user's install.
  * ``--gate``: a CURRENCY GATE. Exit 0 while the pin is within ``--max-age-days``
    of upstream's latest release, 1 once it is older than that, 2 when nothing
    could be compared (API unreachable, rate-limited, malformed, or a release
    date unreadable). 2 is never reported as 0 and never as 1.
    .github/workflows/pin-currency.yml runs this mode on every push to master
    and on its own schedule.

Tag ordering: ComfyUI tags look like "v0.31.1". Plain string comparison is
wrong across a digit-count change ("v0.9.2" > "v0.31.1" lexically, since '9'
> '3') so tags are parsed into an (int, int, ...) tuple and compared
numerically. Draft and prerelease entries are excluded from both "latest" and
the behind-count, matching what GitHub's own /releases/latest endpoint
excludes.

Fails soft on the API: unreachable, rate-limited, or a malformed/empty
response all print a clearly-labelled "could not check" and exit 0 in default
mode (2 under --gate), never a false "up to date".

Usage:
    python scripts/check_comfyui_pin.py               # compare against the real pin
    python scripts/check_comfyui_pin.py --pinned v0.9.2   # sanity-check a hypothetical pin
    python scripts/check_comfyui_pin.py --gate
    python scripts/check_comfyui_pin.py --gate --max-age-days 14

Stdlib only (urllib + re + json), so it runs anywhere without extra installs -
the CI job that runs this does not even need localm installed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
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

# How many days upstream's latest release may postdate the pinned one before
# --gate exits 1. Matches scripts/check_llama_pin.py's DEFAULT_MAX_AGE_DAYS.
DEFAULT_MAX_AGE_DAYS = 21

# --gate exit codes.
EXIT_CURRENT, EXIT_STALE, EXIT_UNKNOWN = 0, 1, 2

# Result statuses produced by assess(). _compare()'s own "current"/"stale" are
# reused as-is (its "stale" just means "something is newer", with no age
# concept); assess() refines "stale" into BEHIND or STALE using the age.
CURRENT, BEHIND, STALE, UNKNOWN = "current", "behind", "stale", "unknown"


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


def _fetch_release_by_tag_http(repo: str, tag: str):
    """Real GitHub API call for one release by tag. Raises on any failure; see
    release_date(), which never lets that propagate."""
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "localm-comfyui-pin-check",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 - fixed https:// URL
        return json.loads(r.read().decode("utf-8"))


def release_date(tag: str, repo: str = _REPO, *, opener=None) -> "_dt.datetime | None":
    """*tag*'s own ``published_at``, looked up directly - for when the fetched
    listing page does not reach it (an old, stale pin is exactly the one that
    has fallen off page 1). Never raises: any failure reads as unknown, the same
    contract as scripts/check_llama_pin.py's release_date()."""
    if opener is None:
        opener = _fetch_release_by_tag_http
    try:
        body = opener(repo, tag)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    return _parse_date(body.get("published_at"))


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


def _parse_date(value) -> "_dt.datetime | None":
    """'2026-08-12T12:18:24Z' -> aware UTC datetime; anything else -> None."""
    if not isinstance(value, str):
        return None
    try:
        return _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc)
    except ValueError:
        return None


def _eligible_releases(releases: list) -> "list[tuple[tuple[int, ...], str, _dt.datetime | None]]":
    """[(parsed version tuple, tag name, published_at or None), ...] for every
    release eligible to be a "latest"/"behind" candidate: not a dict, or
    draft/prerelease, are excluded; the tag must parse as vX.Y[.Z...]. Shared by
    _compare() (ignores the date) and assess() (uses it)."""
    out = []
    for rel in releases:
        if not isinstance(rel, dict):
            continue
        if rel.get("draft") or rel.get("prerelease"):
            continue
        v = _parse_version(rel.get("tag_name"))
        if v is None:
            continue
        out.append((v, rel.get("tag_name"), _parse_date(rel.get("published_at"))))
    return out


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

    eligible = _eligible_releases(releases)
    if not eligible:
        return {"status": "no_data"}

    latest_v, latest_tag, _latest_date = max(eligible, key=lambda item: item[0])
    behind = sum(1 for v, _, _ in eligible if v > pinned_v)

    if behind == 0:
        return {"status": "current", "latest": latest_tag}
    return {
        "status": "stale",
        "latest": latest_tag,
        "behind": behind,
        # True when the fetched page was full and more releases may exist.
        "capped": len(releases) >= _PER_PAGE,
    }


def _pin_published_at(releases: list, tag: str) -> "_dt.datetime | None":
    """*tag*'s own published_at read out of the raw (unfiltered) *releases*
    page, or None when the page does not include it. Searches the RAW list,
    not _eligible_releases(): the pin's own publish date is a plain fact,
    independent of whether it currently reads as draft/prerelease."""
    for rel in releases:
        if isinstance(rel, dict) and rel.get("tag_name") == tag:
            return _parse_date(rel.get("published_at"))
    return None


def assess(pinned_tag: str, releases: list, pin_date, max_age_days: int) -> dict:
    """_compare()'s categorisation plus an AGE verdict, for --gate.

    *pin_date* is the pinned tag's own published_at, resolved by the CALLER
    (from the fetched page via _pin_published_at(), falling back to
    release_date() when the page does not reach it) - this function fetches
    nothing itself, mirroring check_llama_pin.py's assess().

    Adds to _compare()'s dict:
      days_behind   the latest eligible release's published_at minus pin_date,
                    or None when either date is missing
      pin_date      the datetime passed in
      newest_date   the latest eligible release's published_at, or None
    status becomes UNKNOWN (instead of _compare()'s "stale") when something is
    newer but the age could not be computed; BEHIND/STALE split _compare()'s
    "stale" at max_age_days. "unparseable_pin"/"no_data"/"current" pass through
    unchanged - none of them has an age to compute.
    """
    base = _compare(pinned_tag, releases)
    result = dict(base, days_behind=None, pin_date=pin_date, newest_date=None)
    if base["status"] in ("unparseable_pin", "no_data", "current"):
        return result

    newest_date = next(
        (d for v, t, d in _eligible_releases(releases) if t == base["latest"]), None)
    result["newest_date"] = newest_date
    if pin_date is None or newest_date is None:
        result["status"] = UNKNOWN
        return result
    days = max((newest_date - pin_date).days, 0)
    result["days_behind"] = days
    result["status"] = STALE if days > max_age_days else BEHIND
    return result


# --------------------------------------------------------------------------- #
#  Reporting (default mode)                                                   #
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
#  Reporting (--gate mode)                                                    #
# --------------------------------------------------------------------------- #

def _annotate(level: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{message}")


def _summarise(lines: "list[str]") -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        print(f"(could not write the step summary to {path}: {e})")


def _date_str(d) -> str:
    return d.strftime("%Y-%m-%d") if d else "date unknown"


def _report_unknown_gate(pinned: str, reason: str) -> int:
    print(f"COULD NOT CHECK: {reason}")
    print("This is NOT 'the pin is up to date' - nothing was compared.")
    _annotate("warning", f"ComfyUI pin {pinned}: could not be compared against "
                         f"upstream ({reason}); this is not 'current'")
    _summarise(["## ComfyUI pin currency: COULD NOT CHECK",
                f"pin `{pinned}`; {reason}. Nothing was compared."])
    return EXIT_UNKNOWN


def _report_gate(pinned: str, result: dict, pin_date, max_age_days: int) -> int:
    """Print + annotate + summarise a resolved assess() result, and return the
    --gate exit code. Split out of main() so main() stays a thin arg-parse +
    fetch shell."""
    if result["status"] == "no_data":
        return _report_unknown_gate(pinned, "no usable release data in the API response")

    print(f"ComfyUI bundled pin: {pinned} ({_date_str(pin_date)})")

    if result["status"] == CURRENT:
        print(f"OK: the pin is current (upstream's latest release is {pinned}).")
        _annotate("notice", f"ComfyUI pin {pinned} is current")
        _summarise(["## ComfyUI pin currency: OK",
                    f"pin `{pinned}` is upstream's latest release."])
        return EXIT_CURRENT

    latest = result["latest"]
    or_more = " or more" if result.get("capped") else ""
    print(f"upstream latest: {latest} (released {_date_str(result['newest_date'])})")
    print(f"BEHIND by {result['behind']}{or_more} release(s)")

    if result["status"] == UNKNOWN:
        print(f"  age: UNKNOWN - a release date could not be read, so the "
              f"{max_age_days}-day tolerance cannot be applied")
        return _report_unknown_gate(
            pinned, f"{result['behind']} release(s) behind, but the age could not "
                    "be computed (a release date is missing)")

    days = result["days_behind"]
    print(f"  age: {days} day(s) between the pinned and the latest release "
          f"(tolerance {max_age_days})")
    print("  Remedy: a maintainer tests the newer ComfyUI on real hardware and bumps "
          "COMFYUI_PINNED_COMMIT / COMFYUI_PINNED_VERSION in "
          "localm/media/managed_comfy_fresh.py.")
    summary = [f"## ComfyUI pin currency: {result['status'].upper()}",
               "", "| | |", "|---|---|",
               f"| pinned | `{pinned}` ({_date_str(pin_date)}) |",
               f"| upstream latest | `{latest}` ({_date_str(result['newest_date'])}) |",
               f"| behind | {days} day(s), {result['behind']} release(s) |",
               f"| tolerance | {max_age_days} day(s) |"]
    if result["status"] == STALE:
        print(f"STALE: older than the {max_age_days}-day tolerance.")
        _annotate("error", f"ComfyUI pin {pinned} is STALE: {days} days behind "
                            f"upstream's {latest} (tolerance {max_age_days})")
        _summarise(summary)
        return EXIT_STALE
    print(f"within the {max_age_days}-day tolerance.")
    _annotate("warning", f"ComfyUI pin {pinned} is {days} days behind upstream's "
                          f"{latest} (within the {max_age_days}-day tolerance)")
    _summarise(summary)
    return EXIT_CURRENT


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
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 when the pin is older than --max-age-days, 2 when "
                         "nothing could be compared (default: always exit 0)")
    ap.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                    help="days upstream's latest release may postdate the pin "
                         f"before --gate fails (default {DEFAULT_MAX_AGE_DAYS})")
    args = ap.parse_args(argv)

    pinned = args.pinned or _pinned_version()

    if not args.gate:
        releases = _fetch_releases(args.repo)
        if releases is None:
            print(
                "COMFYUI PIN CHECK: could not check (GitHub releases API unreachable). "
                f"Bundled pin is {pinned}; upstream currency unknown this run."
            )
            return 0
        _report(pinned, _compare(pinned, releases))
        return 0

    releases = _fetch_releases(args.repo)
    if releases is None:
        return _report_unknown_gate(pinned, "GitHub releases API unreachable")

    pin_v = _parse_version(pinned)
    if pin_v is None:
        return _report_unknown_gate(
            pinned, f"{pinned!r} is not a plain vX.Y[.Z] tag, so it cannot be compared")

    pin_date = _pin_published_at(releases, pinned)
    if pin_date is None:
        pin_date = release_date(pinned, args.repo)

    result = assess(pinned, releases, pin_date, args.max_age_days)
    return _report_gate(pinned, result, pin_date, args.max_age_days)


if __name__ == "__main__":
    raise SystemExit(main())
