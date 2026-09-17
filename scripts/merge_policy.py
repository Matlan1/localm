#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decide whether a pull request may merge, as the `merge-policy` job in
.github/workflows/ci.yml: the one check that sums up the others.

The job runs on every pull_request once python-pr-gate, lint, gui-tests and
test have finished, whatever their results, and passes only when:

  - lint and gui-tests succeeded, and no needed job failed or was cancelled;
  - without the `full-ci` label: python-pr-gate succeeded and the change is
    not a release (VERSION is unchanged);
  - with the `full-ci` label: the test matrix succeeded.

The two-platform matrix runs at release, not on an ordinary pull request: a
release PR (one that changes VERSION) without the label fails with the label
named, every other PR merges on python-pr-gate, lint and gui-tests alone. A
skipped, failed or missing result never reads as a pass.

The summary also lists the matrix categories the change touches (CATEGORIES
below: the trust boundary, the plugin engine, inference/workers/the native
binding, packaging and installers, the CI workflows and gates) for the
release run to know what it covers; none of them blocks a merge.

    python scripts/merge_policy.py --full-ci false \
        --result python-pr-gate=success --result lint=success \
        --result gui-tests=success --result test=skipped [--files ...]

The verdict, the results and the matched categories are printed, and appended
to the file named by GITHUB_STEP_SUMMARY when that variable is set. Exit
status 0 is a pass, 1 a fail.

Stdlib only; imports scripts/affected_tests.py and scripts/run_affected_tests.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import affected_tests  # noqa: E402
import run_affected_tests  # noqa: E402

LABEL = "full-ci"
JOBS = ("python-pr-gate", "lint", "gui-tests", "test")
ALWAYS_REQUIRED = ("lint", "gui-tests")
RELEASE = "release"

# Category -> patterns. `**` matches across directories, `*` and `?` within
# one path segment, a leading `!` excludes what it matches from the category.
# Only RELEASE blocks a merge; the others are reported.
CATEGORIES: dict[str, tuple[str, ...]] = {
    RELEASE: ("VERSION",),
    "trust boundary (auth, scopes, TLS, bind handling, network policy, path safety, config)": (
        "localm/auth.py",
        "localm/scopes.py",
        "localm/tls.py",
        "localm/bindhost.py",
        "localm/netlisten.py",
        "localm/portmux.py",
        "localm/netpolicy.py",
        "localm/netpin.py",
        "localm/pathsafe.py",
        "localm/config.py",
    ),
    "plugin engine and contract": ("localm/plugins/*.py",),
    "inference, workers and the native binding": (
        "localm/inference/**",
        "!localm/inference/routes/**",
        "localm/_mp_spawn.py",
        "localm/_torch_gpu_probe.py",
        "localm/setup_llama.py",
        "runtime/**",
    ),
    "packaging and installers": (
        "pyproject.toml",
        "uv.lock",
        "install.sh",
        "setup.sh",
        "setup.bat",
        "setup-gui.sh",
        "setup-gui.bat",
        "installer/**",
        "launcher.pyw",
        "localm-launcher.bat",
        "localm-launcher.sh",
        "localm.bat",
        "localm.sh",
        "localm.py",
        "rollback.bat",
        "rollback.sh",
        "localm/__main__.py",
        "localm/_venvguard.py",
        "localm/install_manifest.py",
        "localm/updater.py",
        "localm/_apply_update.py",
    ),
    "CI workflows and gates": (
        ".github/workflows/**",
        "scripts/affected_tests.py",
        "scripts/run_affected_tests.py",
        "scripts/merge_policy.py",
        "scripts/check_coverage_floors.py",
    ),
}


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif ch == "*":
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return re.compile("".join(out) + r"\Z")


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    """Whether *path* is in the category *patterns* describe: it matches an
    inclusion pattern and no exclusion (`!`) pattern."""
    path = path.replace("\\", "/")
    included = excluded = False
    for pattern in patterns:
        if pattern.startswith("!"):
            excluded = excluded or bool(_pattern_regex(pattern[1:]).match(path))
        else:
            included = included or bool(_pattern_regex(pattern).match(path))
    return included and not excluded


def classify(changed: list[str]) -> dict[str, list[str]]:
    """Category -> the changed paths in it, for the categories with a match,
    in CATEGORIES order."""
    found: dict[str, list[str]] = {}
    for name, patterns in CATEGORIES.items():
        hits = sorted({p.replace("\\", "/") for p in changed if matches(p, patterns)})
        if hits:
            found[name] = hits
    return found


@dataclass
class Verdict:
    """The decision and everything it was made from."""

    ok: bool
    reasons: list[str]                          # why it fails; empty on a pass
    full_ci: bool
    results: dict[str, str]                     # job -> result as reported
    categories: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_release(self) -> bool:
        return RELEASE in self.categories


def decide(full_ci: bool, results: dict[str, str],
           categories: dict[str, list[str]]) -> Verdict:
    """Apply the policy to the label state, the needed jobs' results and the
    matched categories."""
    reasons: list[str] = []
    state = {job: results.get(job, "missing") for job in JOBS}
    for job in JOBS:
        if state[job] not in ("success", "skipped"):
            reasons.append(f"{job}: {state[job]}")
    for job in (*ALWAYS_REQUIRED, "test" if full_ci else "python-pr-gate"):
        if state[job] == "skipped":
            reasons.append(f"{job}: skipped (must be success on "
                           f"{'a labelled' if full_ci else 'an unlabelled'} PR)")
    if not full_ci and RELEASE in categories:
        reasons.append(
            f"this is a release ({', '.join(categories[RELEASE])} changed) and a release "
            f"runs the two-platform test matrix, which runs only on a pull request "
            f"labelled `{LABEL}`: add the label, and merge-policy passes on the run the "
            "label starts once the matrix is green")
    return Verdict(ok=not reasons, reasons=reasons, full_ci=full_ci, results=dict(results),
                   categories=categories)


def evaluate(full_ci: bool, results: dict[str, str], base: str,
             files: list[str] | None) -> Verdict:
    """The verdict for the change git reports since the merge base with *base*
    (or *files*)."""
    changed = files if files is not None else affected_tests.changed_files(base)[0]
    return decide(full_ci, results, classify(changed))


def render_summary(verdict: Verdict) -> str:
    """Markdown for the step summary."""
    lines = ["### Merge policy", ""]
    lines.append("**PASS**: this pull request may merge." if verdict.ok
                 else "**FAIL**: this pull request may not merge yet.")
    for reason in verdict.reasons:
        lines.append(f"- {reason}")
    lines += ["", f"`{LABEL}` label: {'yes' if verdict.full_ci else 'no'}", "",
              "| job | result |", "| --- | --- |"]
    for job in JOBS:
        lines.append(f"| {job} | {verdict.results.get(job, 'missing')} |")
    lines.append("")
    if verdict.categories:
        lines.append("Matrix categories this change touches (the two-platform matrix runs at "
                     "release; only a release blocks here):")
        for name, files in verdict.categories.items():
            lines.append(f"- {name}: " + ", ".join(f"`{f}`" for f in files))
    else:
        lines.append("No changed file is in a matrix category.")
    return "\n".join(lines) + "\n"


def _parse_result(value: str) -> tuple[str, str]:
    job, sep, result = value.partition("=")
    if not sep or job not in JOBS:
        raise argparse.ArgumentTypeError(f"expected one of {', '.join(JOBS)} followed by "
                                         f"=<result>, got {value!r}")
    return job, result.strip().lower()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--full-ci", choices=("true", "false"), required=True,
                    help=f"whether the pull request carries the `{LABEL}` label")
    ap.add_argument("--result", action="append", default=[], type=_parse_result,
                    metavar="JOB=RESULT", help="a needed job's result, e.g. lint=success")
    ap.add_argument("--base", default="origin/master",
                    help="ref the committed changes are diffed from (via merge-base)")
    ap.add_argument("--files", nargs="*", help="use these changed paths instead of git")
    args = ap.parse_args(argv)
    verdict = evaluate(args.full_ci == "true", dict(args.result), args.base, args.files)
    run_affected_tests._publish(render_summary(verdict))
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
