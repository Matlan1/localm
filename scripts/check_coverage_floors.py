#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-module coverage floors for the trust boundary (see pyproject.toml
[tool.coverage.report] for the global ratchet this complements).

coverage.py has no native per-file fail_under: its `[report] fail_under` (see
pyproject.toml) is a single number for the whole run. This script reads the
per-file output of `coverage json` and enforces one floor per module, so a
trust-boundary module cannot regress behind a rising repo-wide total.

WHAT THIS IS AND IS NOT. Same caveat as the global ratchet: this is DRIFT
PROTECTION, not defect prevention. A module holding its floor proves its
EXECUTED fraction did not shrink; it says nothing about whether the branches
that do run assert the right thing. Treat a pass here as "no coverage
regressed", never as "this module is safe".

WINDOWS ONLY. Several of these modules gate real behavior on the live OS
(`os.name`/`sys.platform`), not on test input: pathsafe.py's UNC/device-path
rejection only takes its `raise` arc when `os.name == "nt"`, and config.py's
`_secure_file_perms` / `_is_transient_permission_error` similarly branch on the
real OS. portmux.py has two `if sys.platform == "win32":` blocks (the Ctrl+C
wakeup task) that Linux can never execute, which is about a 4-point split. So a
module's percentage is one number PER PLATFORM, and every floor below holds on
WINDOWS ONLY; see .github/workflows/ci.yml for where this runs. Per-platform
floors would need a per-platform TABLE, not one table run in more places.

MEASURE FROM A WORKTREE OR A FRESH CLONE, NOT THE MAIN CHECKOUT. `config.py`
resolves the data directory by testing `repo_root / "home"` (see its
`_warn_unconfigured_home` fallback). `home/` is GITIGNORED local state: it
exists in a working main checkout and does NOT exist in a git worktree or in
CI's fresh clone, so importing config.py from the main checkout takes the
"configured" branch while a worktree and CI take the "no data directory
configured" branch and execute the warning path. That is worth about 2 points,
enough to put config.py below a floor it never actually breached. If you see
config.py alone below its floor, check where you measured BEFORE changing any
number.

Every floor is set ONE POINT BELOW the value measured on a real `--cov=localm`
run, not the rounded integer a report displays, so a floor can only be RAISED
later, never silently lowered. bindhost.py and scopes.py are pinned at the full
100: coverage.py's percent display only ever rounds TO "100" when the value is
exactly 100.0 (see coverage/results.py Numbers.pc_covered_str), so there is no
rounding risk at the ceiling.

Run after a --cov pytest run has produced coverage.json (`pytest --cov=localm
--cov-report=json ...`, or `coverage json` against an existing .coverage
file):
    python scripts/check_coverage_floors.py [path/to/coverage.json]

Add `--report` to print every module's REAL measured percent next to its floor,
plus the headroom between them:
    python scripts/check_coverage_floors.py --report

The report also names any floor that has drifted more than a few points below
reality. `--report` does not suppress the check - it prints and then enforces
as usual.
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
    "localm/netpolicy.py": 90,
    "localm/auth.py": 90,
    "localm/tls.py": 87,
    "localm/config.py": 83,
    "localm/portmux.py": 96,
}


def check_floors(coverage_json: dict, floors: dict[str, int] = _MODULE_FLOORS) -> list[str]:
    """Pure check over an already-parsed coverage.json ``dict``. Returns a list
    of human-readable problems (empty == every floor holds). Does no file or
    subprocess I/O of its own.

    coverage.json's ``files`` keys are the file_reporter's ``relative_filename()``,
    which uses the HOST's native separator - on Windows
    ``"localm\\\\pathsafe.py"``, backslash, not the forward slash this module's
    keys use. Both sides are normalized to forward slash before comparing, so the
    lookup works regardless of which OS produced the report."""
    problems = []
    files = {k.replace("\\", "/"): v for k, v in coverage_json.get("files", {}).items()}
    for module, floor in floors.items():
        entry = files.get(module.replace("\\", "/"))
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


def report_rows(coverage_json: dict,
                floors: dict[str, int] = _MODULE_FLOORS
                ) -> list[tuple[str, float | None, int]]:
    """``(module, measured_percent_or_None, floor)`` for every floored module,
    in floor-table order. ``None`` means the module was absent from the report,
    the same condition ``check_floors`` treats as its own problem.

    Pure, like ``check_floors``, and sharing its key-normalization: coverage.json
    keys use the HOST separator, so both sides are folded to forward slash."""
    files = {k.replace("\\", "/"): v for k, v in coverage_json.get("files", {}).items()}
    rows: list[tuple[str, float | None, int]] = []
    for module, floor in floors.items():
        entry = files.get(module.replace("\\", "/"))
        actual = entry["summary"]["percent_covered"] if entry is not None else None
        rows.append((module, actual, floor))
    return rows


# Headroom in points above a floor at which --report flags it as stale.
# Reported, never failed.
_STALE_HEADROOM = 5.0


def main(argv: list[str]) -> int:
    # --report prints the table and still runs the check, returning the same
    # exit code.
    args = [a for a in argv if a != "--report"]
    want_report = len(args) != len(argv)
    path = Path(args[0]) if args else REPO / "coverage.json"
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
    # Pass the table explicitly; the parameter default is bound at def time and
    # would not follow a replaced module attribute.
    if want_report:
        stale = []
        print(f"{'module':<24} {'measured':>9} {'floor':>6} {'headroom':>9}")
        for module, actual, floor in report_rows(data, _MODULE_FLOORS):
            if actual is None:
                print(f"{module:<24} {'ABSENT':>9} {floor:>5}% {'-':>9}")
                continue
            headroom = actual - floor
            print(f"{module:<24} {actual:>8.4f}% {floor:>5}% {headroom:>8.2f}")
            if headroom > _STALE_HEADROOM:
                stale.append((module, actual, floor))
        if stale:
            print(f"\n{len(stale)} floor(s) more than {_STALE_HEADROOM:.0f} points "
                  "below the measured value, so they are no longer protecting "
                  "much - raise them to int(measured) - 1:")
            for module, actual, floor in stale:
                print(f"  {module}: floor {floor}%, measured {actual:.4f}% "
                      f"-> {int(actual) - 1}")
        print()
        # stdout is block-buffered when piped, stderr is not: flush so the
        # failure list below lands after the table.
        sys.stdout.flush()

    problems = check_floors(data, _MODULE_FLOORS)
    if problems:
        print("Per-module coverage floor check FAILED (see pyproject.toml "
              "[tool.coverage.report] and scripts/check_coverage_floors.py):\n",
              file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(f"\n{len(problems)} module(s) below floor.", file=sys.stderr)
        # config.py branches on a `home/` directory, so name that confounder
        # when its floor fails.
        if (REPO / "home").is_dir() and any("config.py" in p for p in problems):
            print("\nNOTE: this checkout has a `home/` directory, which config.py "
                  "branches on. Its coverage is measured ~2 points LOWER here than "
                  "in a worktree or CI's fresh clone, where `home/` is absent. "
                  "Re-measure from a worktree before changing config.py's floor - "
                  "the drop may be the environment, not a regression.",
                  file=sys.stderr)
        return 1
    print(f"Per-module coverage floor check passed ({len(_MODULE_FLOORS)} "
          "trust-boundary modules).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
