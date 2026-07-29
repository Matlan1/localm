#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-module coverage floors for the trust boundary (see pyproject.toml
[tool.coverage.report] for the global ratchet this complements).

WHY A SEPARATE SCRIPT. coverage.py has no native per-file fail_under: its
`[report] fail_under` (see pyproject.toml) is a single number for the whole
run. A few modules sit on a trust boundary - a missed branch there is a
potential vulnerability, not a cosmetic gap - and deserve their own floor
that the global number could hide entirely (the whole repo could gain
coverage elsewhere while one of these modules quietly regresses). Two ways
to get that: a coverage.py plugin, or a small script over `coverage json`'s
already-structured per-file output. The script wins on readability - it is a
plain dict of floors and a loop anyone can read in ten seconds, versus a
plugin that has to hook coverage.py's reporting internals for the same
result - and it needs no dependency beyond stdlib `json`, since pytest-cov's
own dependency (`coverage`) is what produces the coverage.json this reads.

WHAT THIS IS AND IS NOT. Same caveat as the global ratchet: this is DRIFT
PROTECTION, not defect prevention. A module holding its floor proves its
EXECUTED fraction did not shrink; it says nothing about whether the branches
that do run assert the right thing. All nine real defects found in this repo
in the week this floor was written lived in code at 85-96% coverage - green
by exactly this kind of gate. Treat a pass here as "no coverage regressed",
never as "this module is safe".

WHY WINDOWS-ONLY IN CI. Several of these modules gate real behavior on the
live OS (`os.name`/`sys.platform`), not on test input: pathsafe.py's UNC/
device-path rejection only takes its `raise` arc when `os.name == "nt"`
(tests assert the OPPOSITE behavior on POSIX, they do not monkeypatch
os.name to force it), and config.py's `_secure_file_perms` /
`_is_transient_permission_error` similarly branch on the real OS. So a small
file's measured percentage is not one number, it is one number PER PLATFORM
- gating both matrix legs against a single baseline would misattribute that
structural platform split as a regression the first time CI's Linux leg ran
this check. The floors below were measured and verified (both pass and fail)
on Windows only; see .github/workflows/ci.yml for where this runs.

Floors are set at (or, for rounding safety, one point below) the measured
value on 2026-07-29 (merged master d62b244b), so today passes and a floor can
only be RAISED later, never silently lowered:
  - bindhost.py / scopes.py are pinned at the full 100: coverage.py's percent
    display only ever rounds TO "100" when the value is exactly 100.0 (see
    coverage/results.py Numbers.pc_covered_str), so there is no rounding risk
    in holding these at the ceiling.
  - Every other floor (pathsafe, netpolicy, auth, tls, config) is ONE POINT
    BELOW its measured integer, not AT it: that integer is coverage.py's
    rounded DISPLAY, not the exact float this script reads, so pinning the
    floor to the display value risks a same-day spurious failure if the real
    number rounds up (91.6 displays as "92" but is below a floor of 92). One
    point of headroom absorbs that without giving up real protection - the
    same reasoning pyproject.toml uses for the global floor.
  - portmux.py is the deliberate exception: it is 38% and under active repair
    in a separate unit, so its floor is pinned AT 38 rather than one below -
    the point is only to stop it sliding further while that work lands, not
    to hold it to a rounding buffer it does not need yet.

Run after a --cov pytest run has produced coverage.json (`pytest --cov=localm
--cov-report=json ...`, or `coverage json` against an existing .coverage
file):
    python scripts/check_coverage_floors.py [path/to/coverage.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# module (coverage.json's repo-relative, forward-slash key) -> floor (percent,
# combined statement+branch, matching coverage.py's `percent_covered`).
_MODULE_FLOORS: dict[str, int] = {
    "localm/bindhost.py": 100,
    "localm/scopes.py": 100,
    "localm/pathsafe.py": 92,
    "localm/netpolicy.py": 91,
    "localm/auth.py": 90,
    "localm/tls.py": 87,
    "localm/config.py": 81,
    "localm/portmux.py": 38,
}


def check_floors(coverage_json: dict, floors: dict[str, int] = _MODULE_FLOORS) -> list[str]:
    """Pure check over an already-parsed coverage.json ``dict``. Returns a list
    of human-readable problems (empty == every floor holds). Kept separate
    from all file/subprocess I/O so tests can feed a synthetic report instead
    of a real coverage run."""
    problems = []
    files = coverage_json.get("files", {})
    for module, floor in floors.items():
        entry = files.get(module)
        if entry is None:
            problems.append(
                f"{module}: not present in the coverage report at all (floor "
                f"is {floor}%) - the run did not import/exercise this module, "
                "which is a worse signal than a low score and needs its own "
                "investigation, not a floor change")
            continue
        actual = entry["summary"]["percent_covered"]
        if actual < floor:
            problems.append(
                f"{module}: coverage {actual:.2f}% is BELOW its floor of "
                f"{floor}% - this is a regression, not noise. Either restore "
                "the missing coverage, or if the drop is deliberate, lower "
                "the floor in scripts/check_coverage_floors.py in the SAME "
                "PR that explains why")
    return problems


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else REPO / "coverage.json"
    if not path.is_file():
        print(f"Coverage floor check could not run: {path} does not exist - "
              "run a --cov pytest with --cov-report=json (or `coverage json`) "
              "first.", file=sys.stderr)
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        print(f"Coverage floor check could not run: {path} is not valid "
              f"JSON: {e}", file=sys.stderr)
        return 1
    problems = check_floors(data)
    if problems:
        print("Per-module coverage floor check FAILED (see pyproject.toml "
              "[tool.coverage.report] and scripts/check_coverage_floors.py):\n",
              file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(f"\n{len(problems)} module(s) below floor.", file=sys.stderr)
        return 1
    print(f"Per-module coverage floor check passed ({len(_MODULE_FLOORS)} "
          "trust-boundary modules).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
