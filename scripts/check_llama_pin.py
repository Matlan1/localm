#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Report how far localm's pinned llama.cpp build has fallen behind upstream, and
with ``--gate`` fail once it is further behind than the project tolerates.

``localm/setup_llama.py`` installs ``_PINNED_TAG``, one release confirmed to load
AND generate, rather than whatever upstream published most recently. This script
compares that constant against upstream's newest release with uploaded assets.

TWO MODES:

  * default: a MAINTENANCE SIGNAL, the same shape as scripts/check_comfyui_pin.py
    and scripts/check_llama_rocm_pin.py. Always exits 0.
  * ``--gate``: a CURRENCY GATE. Exit 0 while the pin is within ``--max-age-days``
    of upstream's newest release, 1 once it is older than that, 2 when nothing
    could be compared (API unreachable, rate-limited, malformed, or the pin's own
    release date unreadable). 2 is never reported as 0 and never as 1.
    .github/workflows/llama-pin-currency.yml runs this mode on every push to
    master and on its own schedule.

WHAT "BEHIND" MEASURES: the days between the pinned release's ``published_at`` and
the newest asset-bearing release's ``published_at``, both read from the API. The
build-counter gap (newest minus pinned build number) and the number of newer
releases on the first API page are printed alongside for scale.

Environment:
    GITHUB_TOKEN          sent as a bearer token so the API call uses the
                          authenticated quota
    GITHUB_STEP_SUMMARY   a markdown summary is appended to this file
    GITHUB_ACTIONS=true   ::error / ::warning / ::notice annotations are emitted

Usage:
    python scripts/check_llama_pin.py                       # report, exit 0
    python scripts/check_llama_pin.py --gate                # 0 / 1 / 2 as above
    python scripts/check_llama_pin.py --gate --max-age-days 14
    python scripts/check_llama_pin.py --pinned b10361       # sanity-check a hypothetical

Advancing the pin is a person's decision with its own procedure:
scripts/bump_llama_pin.py performs the mechanical half and prints the rest.

Stdlib only (urllib + re + json), so the CI job running it does not need localm
installed - same as check_comfyui_pin.py.
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

_REPO = "ggml-org/llama.cpp"
_SETUP_PATH = Path(__file__).resolve().parent.parent / "localm" / "setup_llama.py"
_PIN_RE = re.compile(r'^_PINNED_TAG\s*=\s*"([^"]+)"', re.M)
# Upstream build tags are "b" plus a monotonically increasing build number.
_TAG_RE = re.compile(r"^b(\d+)$")
_PER_PAGE = 100

# How many days upstream's newest release may postdate the pinned one before
# --gate exits 1.
DEFAULT_MAX_AGE_DAYS = 21

# --gate exit codes.
EXIT_CURRENT, EXIT_STALE, EXIT_UNKNOWN = 0, 1, 2

# Result statuses produced by assess().
CURRENT, BEHIND, STALE, UNKNOWN = "current", "behind", "stale", "unknown"


def pinned_tag(path: Path = _SETUP_PATH) -> str:
    """_PINNED_TAG out of setup_llama.py, BY TEXT rather than by importing it.

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
            f"no _PINNED_TAG assignment found in {path}. If the constant was "
            "renamed, this script needs updating - it is not evidence the pin "
            "is fine.")
    return m.group(1)


def _build_number(tag: str):
    m = _TAG_RE.match((tag or "").strip())
    return int(m.group(1)) if m else None


def _request(url: str) -> urllib.request.Request:
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "localm-check-llama-pin"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _get_json(url: str) -> "tuple[object, str]":
    """(parsed body, error). Never raises; a failed request is an error string."""
    try:
        with urllib.request.urlopen(_request(url), timeout=20) as r:
            return json.loads(r.read().decode("utf-8")), ""
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code} from the GitHub releases API"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


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
    ``published_at`` (aware datetime or None). Only non-draft releases with a
    bNNNNN tag and uploaded assets are included. Never raises: a failed lookup is
    reported as an error string, never as an empty list that would read as
    "nothing newer"."""
    url = f"https://api.github.com/repos/{_REPO}/releases?per_page={_PER_PAGE}"
    releases, err = _get_json(url)
    if err:
        return [], err
    if not isinstance(releases, list):
        return [], "the releases API returned something that is not a list"
    out = []
    for rel in releases:
        # Not excluded on prerelease. See test_currency_treats_a_prerelease_flagged_release_as_a_real_candidate.
        if not isinstance(rel, dict) or rel.get("draft"):
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


def release_date(tag: str) -> "_dt.datetime | None":
    """``published_at`` of one release looked up by tag, or None when it cannot
    be read."""
    body, err = _get_json(f"https://api.github.com/repos/{_REPO}/releases/tags/{tag}")
    if err or not isinstance(body, dict):
        return None
    return _parse_date(body.get("published_at"))


def assess(pin: str, releases: list, pin_date, max_age_days: int) -> dict:
    """Pure comparison of *pin* against *releases* (newest-first, as returned by
    upstream_releases()).

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


def _remedy(newest: str) -> str:
    return (
        "To advance the pin - one procedure, not a constant edit, because a bump "
        "without the confirm is exactly the untested-build problem the pin exists "
        "to remove:\n"
        f"    python scripts/check_llama_abi.py --ref {newest}\n"
        f"    python scripts/confirm_llama_runtime.py --tag {newest} --backend cpu "
        "--backend vulkan --receipt confirm.json\n"
        f"    python scripts/bump_llama_pin.py --tag {newest} --receipt confirm.json\n"
        "    # then the checklist bump_llama_pin.py prints (MTP allowlist, "
        "pre-tokenizer regex check, tests)")


def _report_unknown(pin: str, reason: str, gate: bool) -> int:
    print(f"COULD NOT CHECK: {reason}")
    print("This is NOT 'the pin is up to date' - nothing was compared.")
    _annotate("warning", f"llama.cpp pin {pin}: could not be compared against "
                         f"upstream ({reason}); this is not 'current'")
    _summarise(["## llama.cpp pin currency: COULD NOT CHECK",
                f"pin `{pin}`; {reason}. Nothing was compared."])
    return EXIT_UNKNOWN if gate else EXIT_CURRENT


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
    print(f"localm pins llama.cpp {pin}")
    if pin_n is None:
        return _report_unknown(
            pin, f"{pin!r} is not upstream's bNNNNN tag shape, so it cannot be compared",
            args.gate)

    releases, err = upstream_releases()
    if err:
        return _report_unknown(pin, f"upstream releases: {err}", args.gate)

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
        _annotate("notice", f"llama.cpp pin {pin} is current")
        _summarise(["## llama.cpp pin currency: OK", f"pin `{pin}` is upstream's newest "
                    "release with assets."])
        return EXIT_CURRENT

    n = len(result["newer"])
    scale = (f"{result['builds_behind']} builds, {n}{'+' if result['capped'] else ''} "
             f"newer release(s) on the first API page")
    if result["capped"]:
        scale += " (the page ended before reaching the pin, so the release count is a floor)"
    print(f"BEHIND: {scale}")
    print(f"  newer, newest first: {', '.join(result['newer'][:10])}"
          f"{' ...' if n > 10 else ''}")

    if result["status"] == UNKNOWN:
        print(f"  age: UNKNOWN - a release date could not be read, so the "
              f"{args.max_age_days}-day tolerance cannot be applied")
        print(_remedy(newest))
        return _report_unknown(
            pin, f"{scale}, but the age could not be computed (a release date is "
                 "missing)", args.gate)

    days = result["days_behind"]
    print(f"  age: {days} day(s) between the pinned and the newest release "
          f"(tolerance {args.max_age_days})")
    print(_remedy(newest))
    summary = [f"## llama.cpp pin currency: {result['status'].upper()}",
               "", "| | |", "|---|---|",
               f"| pinned | `{pin}` ({_date_str(pin_date)}) |",
               f"| upstream newest with assets | `{newest}` ({_date_str(result['newest_date'])}) |",
               f"| behind | {days} day(s), {scale} |",
               f"| tolerance | {args.max_age_days} day(s) |"]
    if result["status"] == STALE:
        print(f"STALE: older than the {args.max_age_days}-day tolerance.")
        _annotate("error" if args.gate else "warning",
                  f"llama.cpp pin {pin} is STALE: {days} days behind upstream's {newest} "
                  f"(tolerance {args.max_age_days}); see scripts/bump_llama_pin.py")
        summary.append("")
        summary.append(f"Older than the tolerance. Advance it with `scripts/bump_llama_pin.py"
                       f" --tag {newest}` after confirming the build.")
        _summarise(summary)
        return EXIT_STALE if args.gate else EXIT_CURRENT
    print(f"within the {args.max_age_days}-day tolerance.")
    _annotate("warning", f"llama.cpp pin {pin} is {days} days behind upstream's {newest} "
                         f"(within the {args.max_age_days}-day tolerance)")
    _summarise(summary)
    return EXIT_CURRENT


if __name__ == "__main__":
    sys.exit(main())
