#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Report how far localm's pinned lemonade-sdk llama.cpp ROCm build has fallen behind upstream.

``localm/setup_llama.py`` installs ``_ROCM_TAG``, one lemonade-sdk/llamacpp-rocm
release confirmed to work on gfx103X hardware, rather than whatever that repo
published most recently. This is a SEPARATE tag series from ``_PINNED_TAG``
(ggml-org/llama.cpp, checked by scripts/check_llama_pin.py) - see the comments
on ``_ROCM_TAG`` and ``_tag_for`` in setup_llama.py.

TWO MODES, the same shape as scripts/check_llama_pin.py:

  * default: a MAINTENANCE SIGNAL, the same shape as scripts/check_llama_pin.py
    and scripts/check_comfyui_pin.py for this tree's other pinned dependencies.
    Always exits 0.
  * ``--gate``: a CURRENCY GATE. Exit 0 while the pin is within ``--max-age-days``
    of upstream's newest release, 1 once it is older than that, 2 when nothing
    could be compared (API unreachable, rate-limited, malformed, or the pin's
    own release date unreadable). 2 is never reported as 0 and never as 1.
    .github/workflows/pin-currency.yml runs this mode on every push to master
    and on its own schedule.

Unlike ggml-org/llama.cpp, lemonade-sdk/llamacpp-rocm's releases carry a
meaningful draft/prerelease flag, so a candidate is filtered the way
scripts/check_comfyui_pin.py filters ComfyUI's: excluded on draft OR
prerelease. The tag SHAPE is "bNNNN" like _PINNED_TAG rather than ComfyUI's
"vX.Y.Z", so tags are compared as an integer build number, and --gate's age
math is the same days-between-published_at approach
scripts/check_llama_pin.py uses.

There is no confirm_llama_runtime.py-style automated confirmation for this
pin: that script's backend list excludes amd-rocm, because the amd-rocm build
never resolves from an upstream ggml-org tag at all. Advancing _ROCM_TAG means
a maintainer running the newer lemonade-sdk build through localm's real loader
on AMD ROCm hardware, confirming it loads and generates, and then updating
_ROCM_TAG, DEFAULT_URL, DEFAULT_URL_SHA256 and the affected
_PINNED_FALLBACK_SHA256 entries in setup_llama.py together - a person's
decision, not something this script does.

Fails soft on the API: unreachable, rate-limited, or a malformed response all
print a clearly-labelled "could not check" and exit 0 in default mode (2 under
--gate) - never a false "up to date".

Usage:
    python scripts/check_llama_rocm_pin.py
    python scripts/check_llama_rocm_pin.py --pinned b1300   # sanity-check a hypothetical
    python scripts/check_llama_rocm_pin.py --gate
    python scripts/check_llama_rocm_pin.py --gate --max-age-days 14

Stdlib only (urllib + re + json), so the CI job running it does not need
localm installed - same as check_llama_pin.py and check_comfyui_pin.py.
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

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ci_runner_files  # noqa: E402

_REPO = "lemonade-sdk/llamacpp-rocm"
_SETUP_PATH = Path(__file__).resolve().parent.parent / "localm" / "setup_llama.py"
_PIN_RE = re.compile(r'^_ROCM_TAG\s*=\s*"([^"]+)"', re.M)
# lemonade-sdk build tags are "b" plus a monotonically increasing build number,
# the same shape as ggml-org's own but a different, unrelated numbering.
_TAG_RE = re.compile(r"^b(\d+)$")
_PER_PAGE = 100

# How many days upstream's newest release may postdate the pinned one before
# --gate exits 1. Matches scripts/check_llama_pin.py's DEFAULT_MAX_AGE_DAYS.
DEFAULT_MAX_AGE_DAYS = 21

# --gate exit codes.
EXIT_CURRENT, EXIT_STALE, EXIT_UNKNOWN = 0, 1, 2

# Result statuses produced by assess().
CURRENT, BEHIND, STALE, UNKNOWN = "current", "behind", "stale", "unknown"


def pinned_tag(path: Path = _SETUP_PATH) -> str:
    """_ROCM_TAG out of setup_llama.py, BY TEXT rather than by importing it.

    Importing would drag in click, rich and the rest of localm, which this script
    is independent of so the CI job can run it with nothing installed. Reading by
    text also keeps the script usable when the tree it is reading does not import
    cleanly."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"could not read {path}: {e}")
    m = _PIN_RE.search(text)
    if not m:
        raise SystemExit(
            f"no _ROCM_TAG assignment found in {path}. If the constant was "
            "renamed, this script needs updating - it is not evidence the pin "
            "is fine.")
    return m.group(1)


def _build_number(tag: str):
    m = _TAG_RE.match((tag or "").strip())
    return int(m.group(1)) if m else None


def _parse_date(value) -> "_dt.datetime | None":
    """'2026-08-12T12:18:24Z' -> aware UTC datetime; anything else -> None."""
    if not isinstance(value, str):
        return None
    try:
        return _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc)
    except ValueError:
        return None


def upstream_releases() -> "tuple[list, str]":
    """(releases newest-first, error). Each release is a dict with ``tag`` and
    ``published_at`` (aware datetime or None). Never raises: a failed lookup is
    reported as an error string, never as an empty list that would read as
    "nothing newer". Same filter as upstream_tags() (which delegates here):
    excluded on draft OR prerelease - unlike check_llama_pin.py's ggml-org
    filter, lemonade-sdk's draft/prerelease flags are meaningful for this
    repo."""
    url = f"https://api.github.com/repos/{_REPO}/releases?per_page={_PER_PAGE}"
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json",
                      "User-Agent": "localm-check-llama-rocm-pin"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            releases = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return [], f"HTTP {e.code} from the GitHub releases API"
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"
    if not isinstance(releases, list):
        return [], "the releases API returned something that is not a list"
    out = []
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft") or rel.get("prerelease"):
            continue
        tag = rel.get("tag_name")
        # Skip releases whose assets have not finished uploading.
        if isinstance(tag, str) and _build_number(tag) is not None and rel.get("assets"):
            out.append({"tag": tag, "published_at": _parse_date(rel.get("published_at"))})
    if not out:
        return [], "no usable release with uploaded assets in the API response"
    out.sort(key=lambda r: _build_number(r["tag"]), reverse=True)
    return out, ""


def upstream_tags() -> "tuple[list, str]":
    """(tags newest-first, error): upstream_releases() reduced to tag names."""
    releases, err = upstream_releases()
    return [r["tag"] for r in releases], err


def release_date(tag: str, repo: str = _REPO, *, opener=None) -> "_dt.datetime | None":
    """*tag*'s own published_at, looked up directly - for when the fetched
    listing page does not reach it (a stale pin is exactly the one that has
    fallen off page 1). Never raises: any failure reads as unknown, mirroring
    check_llama_pin.py's release_date()."""
    if opener is None:
        opener = _fetch_release_by_tag_http
    try:
        body = opener(repo, tag)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    return _parse_date(body.get("published_at"))


def _fetch_release_by_tag_http(repo: str, tag: str):
    """Real GitHub API call for one release by tag. Raises on any failure; see
    release_date(), which never lets that propagate."""
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json",
                      "User-Agent": "localm-check-llama-rocm-pin"})
    with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 - fixed https:// URL
        return json.loads(r.read().decode("utf-8"))


def assess(pin: str, releases: list, pin_date, max_age_days: int) -> dict:
    """Pure comparison of *pin* against *releases* (newest-first, as returned
    by upstream_releases()). Identical shape to check_llama_pin.py's own
    assess() - the two tag series (ggml-org bNNNNN, lemonade-sdk bNNNN) are
    compared the same way, just against a different upstream.

    Returns a dict with:
      status          CURRENT (nothing newer), BEHIND (newer releases exist,
                      within max_age_days), STALE (older than max_age_days), or
                      UNKNOWN (newer releases exist but the age could not be
                      computed because a date is missing)
      newest          the newest tag
      newer           newer tags on the page, newest first
      capped          True when every release on the page is newer than the pin
      builds_behind   newest build number minus the pinned one
      days_behind     newest published_at minus pin_date in days, or None
      pin_date        the datetime passed in
      newest_date     the newest release's published_at, or None
    """
    pin_n = _build_number(pin)
    newest = releases[0]
    newest_n = _build_number(newest["tag"])
    newer = [r["tag"] for r in releases if (_build_number(r["tag"]) or 0) > pin_n]
    result = {
        "newest": newest["tag"],
        "newer": newer,
        "capped": bool(newer) and len(newer) == len(releases),
        "builds_behind": max(newest_n - pin_n, 0),
        "days_behind": None,
        "pin_date": pin_date,
        "newest_date": newest["published_at"],
    }
    if not newer:
        result["status"] = CURRENT
        return result
    if pin_date is not None and newest["published_at"] is not None:
        result["days_behind"] = max((newest["published_at"] - pin_date).days, 0)
        result["status"] = STALE if result["days_behind"] > max_age_days else BEHIND
    else:
        result["status"] = UNKNOWN
    return result


def _annotate(level: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{message}")


def _summarise(lines: "list[str]") -> None:
    try:
        ci_runner_files.append(ci_runner_files.STEP_SUMMARY, "\n".join(lines) + "\n")
    except OSError as e:
        print(f"(could not write the step summary: {e})")


def _date_str(d) -> str:
    return d.strftime("%Y-%m-%d") if d else "date unknown"


def _report_unknown_gate(pin: str, reason: str) -> int:
    print(f"COULD NOT CHECK: {reason}")
    print("This is NOT 'the pin is up to date' - nothing was compared.")
    _annotate("warning", f"ROCm build pin {pin}: could not be compared against "
                         f"upstream ({reason}); this is not 'current'")
    _summarise(["## llama.cpp ROCm pin currency: COULD NOT CHECK",
                f"pin `{pin}`; {reason}. Nothing was compared."])
    return EXIT_UNKNOWN


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pinned", default=None,
                    help="check this tag instead of the one in setup_llama.py")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 when the pin is older than --max-age-days, 2 when "
                         "nothing could be compared (default: always exit 0)")
    ap.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                    help="days upstream's newest release may postdate the pin "
                         f"before --gate fails (default {DEFAULT_MAX_AGE_DAYS})")
    args = ap.parse_args(argv)

    pin = args.pinned or pinned_tag()
    pin_n = _build_number(pin)
    print(f"localm pins the lemonade-sdk ROCm build at {pin}")

    if not args.gate:
        if pin_n is None:
            print(f"COULD NOT CHECK: {pin!r} is not upstream's bNNNNN tag shape, so "
                  "it cannot be compared. This is not a statement that the pin is "
                  "current.")
            return 0

        tags, err = upstream_tags()
        if err:
            print(f"COULD NOT CHECK upstream releases: {err}")
            print("This is NOT 'the pin is up to date' - nothing was compared.")
            return 0

        newest = tags[0]
        behind = [t for t in tags if (_build_number(t) or 0) > pin_n]
        print(f"upstream newest with assets: {newest}")
        if not behind:
            print(f"OK: the pin is current (nothing newer than {pin} in the last "
                  f"{len(tags)} releases).")
            return 0
        print(f"BEHIND by {len(behind)} release(s): {', '.join(behind[:10])}"
              f"{' ...' if len(behind) > 10 else ''}")
        if len(behind) >= _PER_PAGE:
            print(f"(that is the whole {_PER_PAGE}-release page, so the real gap may "
                  "be larger)")
        print("\nTo advance the pin: run the newer lemonade-sdk build through "
              "localm's real loader on AMD ROCm hardware and confirm it loads AND "
              "generates. There is no confirm_llama_runtime.py-style automated "
              "check for this pin - that script's backend list excludes amd-rocm, "
              "since this build never resolves from an upstream ggml-org tag. Once "
              "confirmed, update in setup_llama.py together:")
        print(f"    _ROCM_TAG = {newest!r}")
        print("    DEFAULT_URL, DEFAULT_URL_SHA256, and the affected "
              "_PINNED_FALLBACK_SHA256 entries")
        return 0

    if pin_n is None:
        return _report_unknown_gate(
            pin, f"{pin!r} is not upstream's bNNNNN tag shape, so it cannot be compared")

    releases, err = upstream_releases()
    if err:
        return _report_unknown_gate(pin, f"upstream releases: {err}")

    pin_date = next((r["published_at"] for r in releases if r["tag"] == pin), None)
    if pin_date is None:
        pin_date = release_date(pin)
    result = assess(pin, releases, pin_date, args.max_age_days)
    newest = result["newest"]
    print(f"  pinned release date: {_date_str(pin_date)}")
    print(f"upstream newest with assets: {newest} "
          f"(released {_date_str(result['newest_date'])})")

    if result["status"] == CURRENT:
        print(f"OK: the pin is current (nothing newer than {pin} in the last "
              f"{len(releases)} releases).")
        _annotate("notice", f"ROCm build pin {pin} is current")
        _summarise(["## llama.cpp ROCm pin currency: OK",
                    f"pin `{pin}` is upstream's newest release with assets."])
        return EXIT_CURRENT

    n = len(result["newer"])
    scale = (f"{result['builds_behind']} builds, {n}{'+' if result['capped'] else ''} "
             f"newer release(s) on the first API page")
    print(f"BEHIND: {scale}")
    print(f"  newer, newest first: {', '.join(result['newer'][:10])}"
          f"{' ...' if n > 10 else ''}")

    if result["status"] == UNKNOWN:
        print(f"  age: UNKNOWN - a release date could not be read, so the "
              f"{args.max_age_days}-day tolerance cannot be applied")
        return _report_unknown_gate(
            pin, f"{scale}, but the age could not be computed (a release date is "
                 "missing)")

    days = result["days_behind"]
    print(f"  age: {days} day(s) between the pinned and the newest release "
          f"(tolerance {args.max_age_days})")
    print("\nTo advance the pin: run the newer lemonade-sdk build through "
          "localm's real loader on AMD ROCm hardware and confirm it loads AND "
          "generates. Update _ROCM_TAG, DEFAULT_URL, DEFAULT_URL_SHA256 and the "
          "affected _PINNED_FALLBACK_SHA256 entries in setup_llama.py together.")
    summary = [f"## llama.cpp ROCm pin currency: {result['status'].upper()}",
               "", "| | |", "|---|---|",
               f"| pinned | `{pin}` ({_date_str(pin_date)}) |",
               f"| upstream newest with assets | `{newest}` ({_date_str(result['newest_date'])}) |",
               f"| behind | {days} day(s), {scale} |",
               f"| tolerance | {args.max_age_days} day(s) |"]
    if result["status"] == STALE:
        print(f"STALE: older than the {args.max_age_days}-day tolerance.")
        _annotate("error", f"ROCm build pin {pin} is STALE: {days} days behind "
                            f"upstream's {newest} (tolerance {args.max_age_days})")
        _summarise(summary)
        return EXIT_STALE
    print(f"within the {args.max_age_days}-day tolerance.")
    _annotate("warning", f"ROCm build pin {pin} is {days} days behind upstream's "
                         f"{newest} (within the {args.max_age_days}-day tolerance)")
    _summarise(summary)
    return EXIT_CURRENT


if __name__ == "__main__":
    sys.exit(main())
