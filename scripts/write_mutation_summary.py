#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render the mutmut-cicd-stats.json produced by `mutmut export-cicd-stats` as
a GitHub Actions step summary.

Read-only and never fails the build: this is a display step, not a gate. See
.github/workflows/ci.yml for the job this runs in. The separate `--check`
mode is a different, gating entry point - see its own docstring below.

Run:  python scripts/write_mutation_summary.py [path/to/mutmut-cicd-stats.json]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_OUTCOME_KEYS = (
    "killed", "survived", "no_tests", "skipped", "suspicious", "timeout",
    "check_was_interrupted_by_user", "segfault",
)


def accounted_for(stats: dict) -> bool:
    """True if every generated mutant reached a real outcome.

    mutmut's own pytest invocation stops at the first failing test
    (regardless of whether that test belongs to the mutated scope), and
    `mutmut run` still exits 0 in that case: mutants are generated (so
    `total` is nonzero) but never tested, leaving every outcome count at 0.
    `total` equals the sum of the outcome counts only when testing actually
    ran to completion."""
    return stats.get("total", 0) == sum(stats.get(k, 0) for k in _OUTCOME_KEYS)


def render_summary(stats: dict) -> str:
    """Markdown for the mutation-testing step summary."""
    total = stats.get("total", 0)
    killed = stats.get("killed", 0)
    survived = stats.get("survived", 0)
    lines = ["### Mutation testing (mutmut)", ""]
    if total <= 0:
        lines.append("No mutants were tested.")
        return "\n".join(lines) + "\n"

    if not accounted_for(stats):
        lines.append(
            f"**Incomplete run**: {total} mutant(s) were generated but none "
            "reached a real outcome. mutmut's own test run likely failed "
            "before mutation testing started - this is not a clean result.")
        return "\n".join(lines) + "\n"

    score = 100.0 * killed / total
    lines.append(f"**{score:.1f}%** mutation score ({killed}/{total} mutants killed).")
    if survived:
        lines.append(f"\n**{survived}** mutant(s) survived - a test gap, not necessarily a bug.")
    lines += [
        "",
        "<details><summary>Full outcome breakdown</summary>",
        "",
        "| outcome | count |",
        "|---|---:|",
    ]
    for key in _OUTCOME_KEYS:
        lines.append(f"| `{key}` | {stats.get(key, 0)} |")
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def _publish(text: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else REPO / "mutants" / "mutmut-cicd-stats.json"

    if not path.is_file():
        _publish("### Mutation testing (mutmut)\n\nNo stats report at "
                  f"`{path}` - the mutation-test job likely did not produce one.\n")
        return 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        _publish(f"### Mutation testing (mutmut)\n\nCould not read `{path.name}`: {e}\n")
        return 0

    if not isinstance(data, dict):
        _publish(f"### Mutation testing (mutmut)\n\n`{path.name}` did not contain a JSON object.\n")
        return 0

    _publish(render_summary(data))
    return 0


def check(argv: list[str]) -> int:
    """Gating entry point, separate from `main`'s always-0 display contract.

    Exits 1 if the stats report is missing, unreadable, or `accounted_for`
    is False - i.e. mutants were generated but mutmut's own test run did not
    reach a real outcome for them. Exits 0 otherwise, whatever the mutation
    score is: this checks that the run COMPLETED, not that it scored well.

    Run:  python scripts/write_mutation_summary.py --check [path]
    """
    path = Path(argv[0]) if argv else REPO / "mutants" / "mutmut-cicd-stats.json"
    if not path.is_file():
        print(f"mutation-test completeness check: no stats report at {path}", file=sys.stderr)
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        print(f"mutation-test completeness check: could not read {path.name}: {e}", file=sys.stderr)
        return 1
    if not isinstance(data, dict) or not accounted_for(data):
        print("mutation-test completeness check: the run did not complete - "
              "see the step summary above.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        raise SystemExit(check(sys.argv[2:]))
    raise SystemExit(main(sys.argv[1:]))
