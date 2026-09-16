#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the test files scripts/affected_tests.py selects for a change, as the
per-PR Python gate.

The selector's exit status and output decide what runs:

  selected   exit 0 and one or more paths: pytest runs exactly those files.
  nothing    exit 0 and the NO_TEST_FILE_IS_AFFECTED sentinel alone: pytest is
             not run and this script exits 0.
  wide       exit 3 (the selection exceeds the selector's --max-share):
             --wide full runs the whole unit suite once on this host, without
             coverage; --wide fail exits 1 without running pytest.
  failed     any other exit status, no output, or a line that is not an
             existing tests/**/test_*.py file: exit 1 without running pytest.

pytest runs with `-m "not integration" -n auto` plus any arguments after `--`,
and its exit status is this script's, except that 5 (every selected test was
deselected by the marker expression) exits 0 after saying so.

The selection, its reasons and the mode are printed, and appended to the file
named by GITHUB_STEP_SUMMARY when that variable is set.

    python scripts/run_affected_tests.py [--depth N] [--base REF] [--files ...]
                                         [--wide full|fail] [--dry-run] [-- pytest args]

Stdlib only, apart from the pytest it launches.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SELECTOR = REPO / "scripts" / "affected_tests.py"
NOTHING_AFFECTED = "tests/NO_TEST_FILE_IS_AFFECTED"
WIDE_EXIT = 3
PYTEST_NO_TESTS_COLLECTED = 5
_TEST_PATH = re.compile(r"^tests(?:/[A-Za-z0-9_\-][A-Za-z0-9_.\-]*)*/test_[A-Za-z0-9_.\-]+\.py$")
_PYTEST_ARGS = ["-m", "not integration", "-n", "auto"]


@dataclass
class Selection:
    """The parsed selector result."""

    mode: str                                   # selected | nothing | wide | failed
    paths: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    detail: str = ""                            # the selector's stderr, or the reason it failed
    exit_status: int = 0


def parse_selection(exit_status: int, stdout: str, stderr: str, repo: Path | None = None) -> Selection:
    """Classify the selector's exit status and `--why` output. Every path must
    be an existing tests/**/test_*.py file under *repo*; the nothing sentinel
    counts only when it is the whole output."""
    repo = REPO if repo is None else repo
    lines = [ln.rstrip("\r") for ln in stdout.splitlines() if ln.strip()]
    if exit_status == WIDE_EXIT:
        return Selection("wide", detail=stderr.strip(), exit_status=exit_status)
    if exit_status != 0:
        return Selection("failed", detail=f"selector exited {exit_status}\n{stderr.strip()}".strip(),
                         exit_status=exit_status)
    if lines == [NOTHING_AFFECTED]:
        return Selection("nothing", detail=stderr.strip())
    if not lines:
        return Selection("failed", detail="selector exited 0 and printed no selection", exit_status=1)
    paths: list[str] = []
    reasons: dict[str, str] = {}
    for line in lines:
        path, _, why = line.partition("  # ")
        path = path.strip()
        if not _TEST_PATH.match(path) or not (repo / path).is_file():
            return Selection("failed", detail=f"selector printed a line that is not an existing "
                                              f"tests/**/test_*.py file: {line!r}", exit_status=1)
        paths.append(path)
        reasons[path] = why.strip()
    return Selection("selected", paths=paths, reasons=reasons, detail=stderr.strip())


def _run_selector(depth: int, base: str, files: list[str] | None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SELECTOR), "--why", "--depth", str(depth), "--base", base]
    if files is not None:
        cmd += ["--files", *files]
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def _run_pytest(args: list[str]) -> int:
    return subprocess.run([sys.executable, "-m", "pytest", *args], cwd=str(REPO)).returncode


def pytest_args(selection: Selection, wide: str, extra: list[str]) -> list[str] | None:
    """The pytest argument list for *selection*, or None when pytest is not run."""
    if selection.mode == "selected":
        return [*selection.paths, *_PYTEST_ARGS, *extra]
    if selection.mode == "wide" and wide == "full":
        return ["tests", *_PYTEST_ARGS, *extra]
    return None


def render_summary(selection: Selection, wide: str, depth: int, args: list[str] | None,
                   pytest_status: int | None = None) -> str:
    """Markdown for the step summary: the mode, the selection with reasons,
    and the selector's own count line."""
    lines = ["### Affected tests (python-pr-gate)", ""]
    if selection.mode == "selected":
        lines.append(f"**{len(selection.paths)} test file(s)** selected at `--depth {depth}`; "
                     f"pytest runs exactly those.")
    elif selection.mode == "nothing":
        lines.append("**No test file is affected** by this change; pytest was not run.")
    elif selection.mode == "wide":
        lines.append("**Selection wider than the selector's limit** (exit 3): " + (
            "the whole unit suite runs once on this host, without coverage."
            if wide == "full" else "pytest was not run and the job fails; this change needs "
                                   "the full suite."))
    else:
        lines.append("**The selector failed**; pytest was not run and the job fails.")
    if pytest_status == PYTEST_NO_TESTS_COLLECTED:
        lines.append("")
        lines.append("pytest collected no test: every test in the selection is deselected by "
                     "`-m \"not integration\"`.")
    if selection.detail:
        lines += ["", "```", selection.detail, "```"]
    if selection.paths:
        lines += ["", "<details><summary>Selection with reasons</summary>", "", "```"]
        lines += [f"{p}  # {selection.reasons.get(p, '')}".rstrip(" #") for p in selection.paths]
        lines += ["```", "", "</details>"]
    if args is not None:
        lines += ["", f"`pytest {shlex.join(args)}`"]
    return "\n".join(lines) + "\n"


def _publish(text: str) -> None:
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--depth", type=int, default=1,
                    help="passed to the selector: follow importers this many hops (default 1)")
    ap.add_argument("--base", default="origin/master",
                    help="passed to the selector: ref the committed changes are diffed from")
    ap.add_argument("--files", nargs="*", help="passed to the selector: use these changed paths")
    ap.add_argument("--wide", choices=("full", "fail"), default="full",
                    help="on a wide selection (selector exit 3): run the whole unit suite "
                         "without coverage, or fail (default full)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run without running pytest")
    ap.add_argument("pytest_args", nargs="*", help="arguments after -- are passed to pytest")
    args = ap.parse_args(argv)

    proc = _run_selector(args.depth, args.base, args.files)
    selection = parse_selection(proc.returncode, proc.stdout, proc.stderr)
    cmd = pytest_args(selection, args.wide, args.pytest_args)

    if cmd is None:
        _publish(render_summary(selection, args.wide, args.depth, cmd))
        return 0 if selection.mode == "nothing" else 1
    if args.dry_run:
        _publish(render_summary(selection, args.wide, args.depth, cmd))
        return 0
    status = _run_pytest(cmd)
    _publish(render_summary(selection, args.wide, args.depth, cmd, pytest_status=status))
    if status == PYTEST_NO_TESTS_COLLECTED:
        return 0
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
