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

MEASURE FROM A WORKTREE OR A FRESH CLONE, NOT THE MAIN CHECKOUT. This is a
SECOND environmental axis, separate from the platform one above, and it is
easy to trip because nothing about it is visible in the diff.

`config.py` resolves the data directory by testing `repo_root / "home"` (see
its `_warn_unconfigured_home` fallback). `home/` is GITIGNORED local state: it
exists in a working main checkout and does NOT exist in a git worktree or in
CI's fresh clone. So importing config.py from the main checkout takes the
"configured" branch, while a worktree and CI take the "no data directory
configured" branch and execute the warning path.

Measured 2026-07-29 on ONE unchanged commit (config.py had zero commits and an
identical line count between the two runs): main checkout 82.4201%, which is
BELOW the floor of 83, against 84.7032% from a worktree, which is the value
this floor was set from and the value CI reproduces.

The trap that makes this worth a docstring rather than a comment: the failure
message this script prints tells the reader to "lower the floor in the SAME PR
that explains why". Someone who measured from the main checkout would follow
that advice and weaken a floor that was never breached. A gate quietly lowered
by a person doing exactly what it told them to is worse than no gate.

If you see config.py alone below its floor, check where you measured BEFORE
changing any number.

Floors are set ONE POINT BELOW the ACTUAL value measured on a real
`--cov=localm` run on merged master (d62b244b, 2026-07-29, Windows), not the
rounded integer a report displays, so today passes and a floor can only be
RAISED later, never silently lowered:
  - bindhost.py / scopes.py are pinned at the full 100: coverage.py's percent
    display only ever rounds TO "100" when the value is exactly 100.0 (see
    coverage/results.py Numbers.pc_covered_str), so there is no rounding risk
    in holding these at the ceiling. Measured: both exactly 100.0000%.
  - Every other floor is ONE POINT BELOW its measured value, not AT it, and
    this is not theoretical caution - it was PROVEN necessary by the numbers
    below. netpolicy.py measured 91.5865%: a floor of 92, the rounded
    whole-percent figure, would have FAILED on the very first CI run, on
    unchanged code, for no real regression. portmux.py measured 37.7289%: a
    floor of exactly 38, the same kind of rounded figure, fails RIGHT NOW for
    the same reason. Both were caught by actually running the suite once and
    reading coverage.json, not by trusting a rounded figure. Measured values
    (Windows, 2026-07-29, d62b244b): pathsafe.py 93.1034%, netpolicy.py
    91.5865%, auth.py 91.0112%, tls.py 88.3041%, config.py 84.7032% (an
    earlier estimate had put this one at 82%; the real figure is comfortably
    higher, which only widens this floor's headroom, so the floor below is
    still one point below the TRUE value, not the stale estimate), portmux.py
    37.7289%.
  - portmux.py's repair HAS since landed (#898 took it to 97.8022%), and this
    is the follow-up bump that entry anticipated: floor = int(97.8022) - 1 = 96.
    Re-measured 2026-07-29 on 1fa3af2c from a worktree. It sat at 36 for a
    while first, because #898 merged BEFORE the PR that measured the floors,
    leaving ~62 points that could erode without tripping anything and no owner
    for the bump - which is why `--report` now exists and why CI runs it.

Current measured values (Windows, 2026-07-29, 1fa3af2c, FROM A WORKTREE):
bindhost 100.0000, scopes 100.0000, pathsafe 93.1034, netpolicy 91.5865,
auth 91.0112, tls 88.3041, config 84.9315, portmux 97.8022. Every floor above
is int(measured) - 1 against these, so only portmux moved; the rest were
already correct. Repo total 79.53%, so pyproject.toml's fail_under=78 is also
still int(total) - 1 and needs no change.

Run after a --cov pytest run has produced coverage.json (`pytest --cov=localm
--cov-report=json ...`, or `coverage json` against an existing .coverage
file):
    python scripts/check_coverage_floors.py [path/to/coverage.json]

Add `--report` to print every module's REAL measured percent next to its floor,
plus the headroom between them:
    python scripts/check_coverage_floors.py --report

Use it before changing a floor. Without it, raising one means re-deriving every
value by hand out of coverage.json, and that friction is not hypothetical: it
is why portmux.py sat at a floor of 36 for a while after its coverage reached
the high nineties (the floor was measured one merge before the PR that raised
it, and the follow-up bump had no owner). The report also names any floor that
has drifted more than a few points below reality, which is exactly that state.
`--report` does not suppress the check - it prints and then enforces as usual.
"""

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
    """Pure check over an already-parsed coverage.json ``dict``. Returns a list
    of human-readable problems (empty == every floor holds). Kept separate
    from all file/subprocess I/O so tests can feed a synthetic report instead
    of a real coverage run.

    coverage.json's ``files`` keys are the file_reporter's ``relative_filename()``,
    which uses the HOST's native separator - measured on a real Windows run:
    ``"localm\\\\pathsafe.py"``, backslash, not the forward slash this module's
    keys use. A naive dict lookup would report every floored module "missing"
    on Windows (the exact platform this check runs on - see the module
    docstring), which is worse than silent: it reads as a real gate while
    actually checking nothing. Normalize both sides to forward slash before
    comparing so the lookup works regardless of which OS produced the report."""
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
