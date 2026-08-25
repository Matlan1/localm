#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-module coverage floors for the trust boundary (see pyproject.toml [tool.coverage.report] for the global ratchet this complements)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# module (coverage.json's repo-relative, forward-slash key) -> floor (percent,
# combined statement+branch, matching coverage.py's `percent_covered`). Each
# is one point below its REAL measured value (see the module docstring for
# why "one point below the rounded display" is not enough headroom).
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
    """Pure check over an already-parsed coverage.json ``dict``."""
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
    """``(module, measured_percent_or_None, floor)`` for every floored module, in floor-table order. ``None`` means the module was absent from the report, the same condition ``check_floors`` treats as its own problem."""
    files = {k.replace("\\", "/"): v for k, v in coverage_json.get("files", {}).items()}
    rows: list[tuple[str, float | None, int]] = []
    for module, floor in floors.items():
        entry = files.get(module.replace("\\", "/"))
        actual = entry["summary"]["percent_covered"] if entry is not None else None
        rows.append((module, actual, floor))
    return rows


# A floor is meant to sit ONE POINT below its measured value, so anything much
# wider means the floor stopped tracking reality and is no longer protecting the
# module it names. Reported, never failed: a module legitimately gains coverage
# between the PR that adds tests and the PR that bumps the floor, and turning
# that normal interval into a red gate would make the two PRs have to be one.
_STALE_HEADROOM = 5.0


def main(argv: list[str]) -> int:
    # --report is ADDITIVE, never a bypass: it prints the table and still runs
    # the check, returning the same exit code. A flag that both reported and
    # skipped enforcement would be an always-green gate one CI edit away.
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
    # Pass the table EXPLICITLY rather than leaning on the parameter default.
    # The default is bound once at def time, so it would keep pointing at the
    # original dict even after the module attribute is replaced - which would
    # make a test that swaps the table silently assert against the shipped one
    # instead, and pass for the wrong reason.
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
        # stdout is block-buffered when piped (a CI log is), stderr is not, so
        # without this the failure list below lands ABOVE the table it refers to.
        sys.stdout.flush()

    problems = check_floors(data, _MODULE_FLOORS)
    if problems:
        print("Per-module coverage floor check FAILED (see pyproject.toml "
              "[tool.coverage.report] and scripts/check_coverage_floors.py):\n",
              file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(f"\n{len(problems)} module(s) below floor.", file=sys.stderr)
        # The message above tells the reader they may lower the floor. Before
        # anyone acts on that for config.py, name the one confounder known to
        # produce a false sub-floor reading: measuring from a checkout that has
        # a `home/` directory (see the module docstring). Keyed on config.py
        # specifically because that is the module measured to branch on it -
        # printing this for every failure would be noise on the real ones, and
        # noise is how a genuine warning stops being read.
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
