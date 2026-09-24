#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reports whether a change touches the mutation-tested trust boundary, for
the `mutation-scope` notice job in .github/workflows/ci.yml.

The diff from the merge base with ``--base`` (default origin/master) to HEAD
is checked against:

* the ``[tool.mutmut] only_mutate`` modules in pyproject.toml;
* the mutation gate itself: scripts/mutation_baseline.json,
  scripts/check_mutation_floors.py, scripts/mutmut_run.py,
  scripts/write_mutation_summary.py, this script;
* the ``[tool.mutmut]`` section of pyproject.toml (compared parsed, so an
  unrelated pyproject edit does not count).

Prints one ``touched=true`` or ``touched=false`` line followed by the matching
paths. ``--github-output`` appends the ``touched=`` line to the file
``$GITHUB_OUTPUT`` names; ``--notice`` emits a ``::notice`` annotation on a hit
and requires ``--gate-runs``: with ``--gate-runs true`` (the pull request
carries the ``mutation-test`` label) the notice says the mutation gate runs on
this pull request, with ``--gate-runs false`` it says the gate did not run and
that the ``mutation-test`` label runs it. Exits 1 when the diff cannot be
computed (no merge base, git failure) or when ``[tool.mutmut] only_mutate`` is
empty or missing: an unknown answer is never reported as ``touched=false``.
Exits 2 on a usage error, including ``--notice`` without ``--gate-runs``.

Run:  python scripts/mutation_scope.py [--base REF] [--github-output]
                                       [--notice --gate-runs true|false]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ci_runner_files  # noqa: E402

GATE_FILES = (
    "scripts/mutation_baseline.json",
    "scripts/check_mutation_floors.py",
    "scripts/mutmut_run.py",
    "scripts/write_mutation_summary.py",
    "scripts/mutation_scope.py",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def mutmut_section(pyproject_text: str) -> dict:
    """The parsed ``[tool.mutmut]`` table, ``{}`` when absent or unparsable."""
    try:
        return tomllib.loads(pyproject_text).get("tool", {}).get("mutmut", {}) or {}
    except (tomllib.TOMLDecodeError, ValueError):
        return {}


def touched(changed: list[str], modules: list[str], mutmut_config_changed: bool) -> list[str]:
    """The changed paths that put a change in scope: a mutated module, a gate
    file, or ``pyproject.toml`` when its ``[tool.mutmut]`` table changed."""
    norm = {c.replace("\\", "/") for c in changed}
    watched = {m.replace("\\", "/") for m in modules} | set(GATE_FILES)
    hits = sorted(norm & watched)
    if mutmut_config_changed and "pyproject.toml" in norm:
        hits.append("pyproject.toml ([tool.mutmut] changed)")
    return hits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="origin/master",
                    help="ref the change is diffed from (via merge-base)")
    ap.add_argument("--github-output", action="store_true",
                    help="also append touched=... to $GITHUB_OUTPUT")
    ap.add_argument("--notice", action="store_true",
                    help="on a hit, print a ::notice annotation naming the touched "
                         "paths and whether the mutation gate runs on this pull request")
    ap.add_argument("--gate-runs", choices=("true", "false"),
                    help="with --notice: whether the pull request carries the "
                         "mutation-test label, which runs the mutation gate on it")
    ap.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.notice and args.gate_runs is None:
        ap.error("--notice requires --gate-runs true|false")

    try:
        merge_base = _git(args.repo, "merge-base", args.base, "HEAD").strip()
        changed = [c for c in _git(args.repo, "diff", "--name-only", merge_base, "HEAD").splitlines() if c]
        head_pyproject = _git(args.repo, "show", "HEAD:pyproject.toml")
        base_pyproject = _git(args.repo, "show", f"{merge_base}:pyproject.toml")
    except (subprocess.CalledProcessError, OSError) as e:
        detail = getattr(e, "stderr", "") or str(e)
        print(f"mutation scope: could not diff against {args.base}: {detail.strip()}",
              file=sys.stderr)
        return 1

    modules = [str(m) for m in mutmut_section(head_pyproject).get("only_mutate", [])]
    if not modules:
        print("mutation scope: could not resolve scope: [tool.mutmut] only_mutate "
              "is empty or missing", file=sys.stderr)
        return 1
    config_changed = mutmut_section(head_pyproject) != mutmut_section(base_pyproject)
    hits = touched(changed, modules, config_changed)
    verdict = "true" if hits else "false"
    print(f"touched={verdict}")
    for h in hits:
        print(f"  {h}")
    if args.github_output:
        ci_runner_files.append(ci_runner_files.OUTPUT, f"touched={verdict}\n")
    if args.notice and hits:
        if args.gate_runs == "true":
            print("::notice title=Mutation gate runs on this pull request::This change "
                  f"touches the mutation-tested trust boundary ({', '.join(hits)}) - the "
                  "mutation-test label runs the mutation shards and the mutation-test "
                  "gate on this pull request.")
        else:
            print("::notice title=Mutation gate did not run::This change touches the "
                  f"mutation-tested trust boundary ({', '.join(hits)}) - the mutation "
                  "shards only run on the weekly schedule, a dispatch, or a pull "
                  "request carrying the mutation-test label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
