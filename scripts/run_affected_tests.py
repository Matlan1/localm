#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the test files scripts/affected_tests.py selects for a change, as the
per-PR Python gate. It never runs the whole suite.

The selector's exit status and output decide what runs:

  selected   exit 0 and one or more paths: pytest runs exactly those files.
  nothing    exit 0 and the NO_TEST_FILE_IS_AFFECTED sentinel alone: pytest is
             not run and this script exits 0.
  wide       exit 3 (the selection exceeds the selector's --max-share): with
             --depth above 0 the selector runs again at depth 0 and its
             selection, if there is one, runs instead. A selection that is
             still wide at depth 0 (a hub module most tests import) runs in
             full: the selector runs once more at depth 0 with the limit
             lifted to WHOLE_SUITE_SHARE, and pytest runs every file it
             selects. Only a selection of every test file (a conftest.py or
             dependency-lock change) exits 1 without running pytest; so does
             a depth-0 retry that selects nothing or fails.
  failed     any other exit status, no output, or a line that is not an
             existing tests/**/test_*.py file: exit 1 without running pytest.

pytest runs with `-m "not integration" -n auto` plus any arguments after `--`,
and its exit status is this script's, except that 5 (every selected test was
deselected by the marker expression) exits 0 after saying so.

The selection, its reasons, the depth it was computed at and the mode are
printed, and appended to the file named by GITHUB_STEP_SUMMARY when that
variable is set.

CI runs the gate in three steps, so a wide selection runs as parallel shards
(tests/_shard.py splits it, the same split the full matrix uses):

  --plan-out FILE   select as above, never run pytest, and write the plan: the
                    mode, the depth, the selection and the shard count
                    (shard_count()). The exit status is the one the
                    selection would have without --plan-out: 1 when it is
                    wide or failed. The GITHUB_OUTPUT outputs are `run`
                    (true when there are tests to run), `count` and `shards`
                    (the JSON list 1..count; [1] when nothing runs).
  --plan FILE --shard I --shard-out DIR
                    run shard I of the plan's selection, writing
                    DIR/ids-I.json (--shard-ids-out of tests/_shard.py). A plan
                    that selects no test file fails.
  --collect DIR --plan FILE --plan-result RESULT
                    pass only when the plan job ended `success` and either
                    the plan selected nothing and DIR holds no shard result,
                    or `python -m tests._shard verify DIR --count N` passes
                    (every shard 1..N reported success and every selected test
                    ran in exactly one shard) and every test that ran is in a
                    file of the plan's selection.

    python scripts/run_affected_tests.py [--depth N] [--base REF] [--files ...]
                                         [--dry-run] [-- pytest args]
    python scripts/run_affected_tests.py --plan-out FILE [--depth N] [--base REF] [--files ...]
    python scripts/run_affected_tests.py --plan FILE --shard I --shard-out DIR [-- pytest args]
    python scripts/run_affected_tests.py --collect DIR --plan FILE --plan-result RESULT

Stdlib only, apart from the pytest it launches (and --collect, which runs
tests/_shard.py and so needs pytest installed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ci_runner_files  # noqa: E402

SELECTOR = REPO / "scripts" / "affected_tests.py"
NOTHING_AFFECTED = "tests/NO_TEST_FILE_IS_AFFECTED"
WIDE_EXIT = 3
# Passed to the selector as --max-share to lift its limit: only a selection of
# every test file is still wide (n of n files is 1.0, n-1 of n is below this
# for any suite under a million files).
WHOLE_SUITE_SHARE = 0.999999
PYTEST_NO_TESTS_COLLECTED = 5
_TEST_PATH = re.compile(r"^tests(?:/[A-Za-z0-9_\-][A-Za-z0-9_.\-]*)*/test_[A-Za-z0-9_.\-]+\.py$")
_PYTEST_ARGS = ["-m", "not integration", "-n", "auto"]
#: The shard count grows by one per this many selected test files.
FILES_PER_SHARD = 75
#: The most shards a selection is split into, per platform.
MAX_SHARDS = 5


@dataclass
class Selection:
    """The parsed selector result."""

    mode: str                                   # selected | nothing | wide | failed
    paths: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    detail: str = ""                            # the selector's stderr, or the reason it failed
    exit_status: int = 0
    depth: int = 0                              # the --depth the selector ran at
    wider: Selection | None = None              # the wide result this one replaced
    over_limit: bool = False                    # selected, though wider than the selector's limit
    retry: str = ""                             # how the narrower runs ended, on a kept wide result


def parse_selection(exit_status: int, stdout: str, stderr: str, repo: Path | None = None,
                    depth: int = 0) -> Selection:
    """Classify the selector's exit status and `--why` output. Every path must
    be an existing tests/**/test_*.py file under *repo*; the nothing sentinel
    counts only when it is the whole output."""
    repo = REPO if repo is None else repo
    lines = [ln.rstrip("\r") for ln in stdout.splitlines() if ln.strip()]
    if exit_status == WIDE_EXIT:
        return Selection("wide", detail=stderr.strip(), exit_status=exit_status, depth=depth)
    if exit_status != 0:
        return Selection("failed", detail=f"selector exited {exit_status}\n{stderr.strip()}".strip(),
                         exit_status=exit_status, depth=depth)
    if lines == [NOTHING_AFFECTED]:
        return Selection("nothing", detail=stderr.strip(), depth=depth)
    if not lines:
        return Selection("failed", detail="selector exited 0 and printed no selection",
                         exit_status=1, depth=depth)
    paths: list[str] = []
    reasons: dict[str, str] = {}
    for line in lines:
        path, _, why = line.partition("  # ")
        path = path.strip()
        if not _TEST_PATH.match(path) or not (repo / path).is_file():
            return Selection("failed", detail=f"selector printed a line that is not an existing "
                                              f"tests/**/test_*.py file: {line!r}",
                             exit_status=1, depth=depth)
        paths.append(path)
        reasons[path] = why.strip()
    return Selection("selected", paths=paths, reasons=reasons, detail=stderr.strip(), depth=depth)


def _run_selector(depth: int, base: str, files: list[str] | None,
                  max_share: float | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SELECTOR), "--why", "--depth", str(depth), "--base", base]
    if max_share is not None:
        cmd += ["--max-share", str(max_share)]
    if files is not None:
        cmd += ["--files", *files]
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


#: Longest pytest argument list passed on the command line; a longer one goes
#: through an ``@file`` (Windows refuses a command line over 32767 characters).
ARGS_INLINE_MAX = 20000


def _run_pytest(args: list[str], inline: list[str] | tuple[str, ...] = ()) -> int:
    """Run pytest with *inline* then *args*, reading *args* from a temporary
    ``@file`` when together they are longer than ``ARGS_INLINE_MAX``
    characters. *inline* always stays on the command line: pytest loads
    ``-p`` plugins from the command line before it expands an ``@file``.
    See test_the_shard_plugin_stays_on_the_command_line_when_the_paths_go_to_an_argfile."""
    base = [sys.executable, "-m", "pytest", *inline]
    if sum(len(a) + 1 for a in args) <= ARGS_INLINE_MAX:
        return subprocess.run([*base, *args], cwd=str(REPO)).returncode
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        argfile = Path(tmp) / "pytest-args.txt"
        argfile.write_text("\n".join(args) + "\n", encoding="utf-8")
        return subprocess.run([*base, f"@{argfile}"], cwd=str(REPO)).returncode


def _base_resolves(base: str) -> bool:
    """Whether *base* names a commit in this checkout."""
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet",
                           f"{base}^{{commit}}"], capture_output=True).returncode == 0


def select(depth: int, base: str, files: list[str] | None) -> Selection:
    """The selection to run: the selector's result at *depth*, or, when that
    is wide and *depth* is above 0, its result at depth 0 if that one selects
    test files. A depth-0 retry that is nothing or failed is returned as the
    original wide result, carrying the retry as its detail. A selection still
    wide at depth 0 is selected again at depth 0 with the limit lifted to
    WHOLE_SUITE_SHARE and returned with ``over_limit`` set; only a selection
    of every test file stays wide. Without *files*, a *base* that does not
    resolve is a failed selection: the selector would diff HEAD against itself
    and select nothing."""
    if files is None and not _base_resolves(base):
        return Selection("failed", detail=f"base ref {base!r} does not resolve in this checkout, "
                                          "so the change cannot be computed (a shallow checkout "
                                          "has no origin/master)", exit_status=1, depth=depth)
    proc = _run_selector(depth, base, files)
    selection = parse_selection(proc.returncode, proc.stdout, proc.stderr, depth=depth)
    if selection.mode != "wide":
        return selection
    if depth > 0:
        proc0 = _run_selector(0, base, files)
        narrower = parse_selection(proc0.returncode, proc0.stdout, proc0.stderr, depth=0)
        if narrower.mode == "selected":
            narrower.wider = selection
            return narrower
        selection.retry = narrower.mode
        selection.detail = (f"{selection.detail}\n\nat --depth 0: {narrower.mode}\n"
                            f"{narrower.detail}").strip()
        if narrower.mode != "wide":
            return selection
    procw = _run_selector(0, base, files, WHOLE_SUITE_SHARE)
    whole = parse_selection(procw.returncode, procw.stdout, procw.stderr, depth=0)
    if whole.mode == "selected":
        whole.wider = selection
        whole.over_limit = True
        return whole
    selection.retry = "whole"
    selection.detail = (f"{selection.detail}\n\nat --depth 0 up to every test file but the whole "
                        f"suite: {whole.mode}\n{whole.detail}").strip()
    return selection


def pytest_args(selection: Selection, extra: list[str]) -> list[str] | None:
    """The pytest argument list for *selection*, or None when pytest is not
    run. Never an argument list without test paths: that is a whole-suite run."""
    if selection.mode == "selected" and selection.paths:
        return [*selection.paths, *_PYTEST_ARGS, *extra]
    return None


def render_summary(selection: Selection, args: list[str] | None,
                   pytest_status: int | None = None) -> str:
    """Markdown for the step summary: the mode, the depth, the selection with
    reasons, and the selector's own count lines."""
    lines = ["### Affected tests (python-pr-gate)", ""]
    if selection.mode == "selected":
        lines.append(f"**{len(selection.paths)} test file(s)** selected at "
                     f"`--depth {selection.depth}`; pytest runs exactly those.")
        if selection.over_limit:
            lines.append("**This change is wide**: the selection is wider than the selector's "
                         "limit (a quarter of the test files) even at `--depth 0`, because it "
                         "touches a module most tests import. pytest runs all of it. Test files "
                         "that reach the change only through other modules are not in it; the "
                         "`full-ci` label adds the whole suite.")
        elif selection.wider is not None:
            lines.append(f"The selection at `--depth {selection.wider.depth}` was wider than the "
                         f"selector's limit (exit 3); the depth-0 selection runs instead.")
    elif selection.mode == "nothing":
        lines.append("**No test file is affected** by this change; pytest was not run.")
    elif selection.mode == "wide":
        retry = {"wide": " and at `--depth 0`",
                 "whole": " and at `--depth 0`, even with the limit lifted to every test file "
                          "but the whole suite",
                 "nothing": ", and nothing at `--depth 0`",
                 "failed": ", and the selector failed at `--depth 0`"}.get(selection.retry, "")
        lines.append(f"**Selection wider than the selector's limit** (exit 3) at "
                     f"`--depth {selection.depth}`{retry}: pytest was not run and the job "
                     "fails. This gate never runs the whole suite; this change needs the full "
                     "suite (the `full-ci` label).")
    else:
        lines.append("**The selection could not be computed**; pytest was not run and the job "
                     "fails.")
    if pytest_status == PYTEST_NO_TESTS_COLLECTED:
        lines.append("")
        lines.append("pytest collected no test: every test in the selection is deselected by "
                     "`-m \"not integration\"`.")
    details = [selection.detail]
    if selection.wider is not None:
        details.insert(0, f"at --depth {selection.wider.depth}: wide\n{selection.wider.detail}")
    for detail in details:
        if detail:
            lines += ["", "```", detail, "```"]
    if selection.paths:
        lines += ["", "<details><summary>Selection with reasons</summary>", "", "```"]
        lines += [f"{p}  # {selection.reasons.get(p, '')}".rstrip(" #") for p in selection.paths]
        lines += ["```", "", "</details>"]
    if args is not None:
        lines += ["", f"`pytest {shlex.join(args)}`"]
    return "\n".join(lines) + "\n"


def _publish(text: str) -> None:
    print(text)
    ci_runner_files.append(ci_runner_files.STEP_SUMMARY, text)


def shard_count(selection: Selection) -> int:
    """How many shards per platform the selection runs in: one per
    FILES_PER_SHARD selected test files, rounded up, at least 1 and at most
    MAX_SHARDS. 1 when the selection is not ``selected``."""
    if selection.mode != "selected" or not selection.paths:
        return 1
    return max(1, min(MAX_SHARDS, math.ceil(len(selection.paths) / FILES_PER_SHARD)))


def _shard_args(index: int, count: int, ids_out: Path) -> list[str]:
    """The pytest arguments that load tests/_shard.py and run shard *index*
    of *count*, writing its node ids to *ids_out*."""
    return ["-p", "tests._shard", "--shard", f"{index}/{count}", "--shard-ids-out", str(ids_out)]


def make_plan(selection: Selection, extra: list[str]) -> dict:
    """The plan file's content for *selection*: its mode, depth, paths, shard
    count and the rendered step summary."""
    count = shard_count(selection)
    summary = render_summary(selection, pytest_args(selection, extra))
    if selection.mode == "selected":
        summary += (f"\nSplit into **{count} shard(s)** per platform (one per {FILES_PER_SHARD} "
                    f"selected test files, at most {MAX_SHARDS}); shard I runs the selection with "
                    f"`-p tests._shard --shard I/{count}` added.\n")
    return {"mode": selection.mode, "depth": selection.depth, "paths": list(selection.paths),
            "count": count, "summary": summary}


def load_plan(path: Path) -> tuple[dict | None, str]:
    """``(plan, "")`` for a well-formed plan file, else ``(None, problem)``.
    Well-formed: a known mode; every path an existing tests/**/test_*.py file;
    a ``selected`` plan with paths and the count shard_count() gives them;
    any other plan without paths and a count of 1."""
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"the plan {path} cannot be read: {e}"
    if not isinstance(plan, dict):
        return None, f"the plan {path} is not a JSON object"
    mode, paths, count = plan.get("mode"), plan.get("paths"), plan.get("count")
    if mode not in ("selected", "nothing", "wide", "failed"):
        return None, f"the plan {path} has an unknown mode {mode!r}"
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        return None, f"the plan {path} has no list of paths"
    for p in paths:
        if not _TEST_PATH.match(p) or not (REPO / p).is_file():
            return None, f"the plan {path} names {p!r}, which is not an existing tests/**/test_*.py file"
    expected = shard_count(Selection(mode, paths=paths))
    if mode == "selected" and not paths:
        return None, f"the plan {path} is `selected` with no paths"
    if mode != "selected" and paths:
        return None, f"the plan {path} is `{mode}` yet names test files"
    if count != expected:
        return None, f"the plan {path} has shard count {count!r}, its selection needs {expected}"
    return plan, ""


def write_plan(selection: Selection, path: Path, extra: list[str]) -> int:
    """Write the plan for *selection* to *path*, publish its summary and the
    GITHUB_OUTPUT outputs, run nothing. 0 for a ``selected`` or ``nothing``
    plan, 1 otherwise."""
    plan = make_plan(selection, extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    _publish(plan["summary"])
    run = selection.mode == "selected"
    shards = list(range(1, plan["count"] + 1))
    ci_runner_files.append(ci_runner_files.OUTPUT, f"run={'true' if run else 'false'}\n"
                                                   f"count={plan['count']}\n"
                                                   f"shards={json.dumps(shards)}\n")
    return 0 if selection.mode in ("selected", "nothing") else 1


def run_shard(plan_path: Path, index: int, out_dir: Path, extra: list[str]) -> int:
    """Run shard *index* of the plan at *plan_path*, writing
    ``out_dir/ids-<index>.json``. pytest's exit status, except 5 (no test
    left in this shard) is 0; 1 without running pytest for a malformed plan,
    a plan that selects no test file, or an index outside 1..count."""
    plan, problem = load_plan(plan_path)
    if plan is not None and plan["mode"] != "selected":
        problem = f"the plan is `{plan['mode']}`: it selects no test file for a shard to run"
    if plan is not None and not problem and not 1 <= index <= plan["count"]:
        problem = f"shard {index} is outside the plan's 1..{plan['count']}"
    if plan is None or problem:
        _publish(f"### Affected tests, shard {index}\n\n**No test ran**: {problem}\n")
        return 1
    count = plan["count"]
    out_dir.mkdir(parents=True, exist_ok=True)
    inline = _shard_args(index, count, (out_dir / f"ids-{index}.json").resolve())
    args = [*plan["paths"], *_PYTEST_ARGS, *extra]
    status = _run_pytest(args, inline)
    lines = [f"### Affected tests, shard {index}/{count}", "",
             f"Shard {index} of {count} of the {len(plan['paths'])} selected test file(s), "
             f"selected at `--depth {plan['depth']}`; pytest exited {status}."]
    if status == PYTEST_NO_TESTS_COLLECTED:
        lines.append("No test is left in this shard after `-m \"not integration\"` and the split.")
    lines += ["", f"`pytest {shlex.join([*inline, *args])}`"]
    _publish("\n".join(lines) + "\n")
    return 0 if status == PYTEST_NO_TESTS_COLLECTED else status


def _verify_shards(directory: Path, count: int) -> tuple[int, str]:
    """Run ``python -m tests._shard verify`` over *directory*; (exit status, output)."""
    proc = subprocess.run([sys.executable, "-m", "tests._shard", "verify", str(directory),
                           "--count", str(count)], cwd=str(REPO), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def collect(directory: Path, plan_path: Path, plan_result: str) -> int:
    """The gate's verdict over the plan and the shard results in *directory*
    (see the module docstring). 0 passes, 1 fails; every reason it fails is
    printed as a ``::error::`` line and published with the plan's summary."""
    problems: list[str] = []
    plan, problem = load_plan(plan_path) if plan_path.is_file() else (None, f"no plan at {plan_path}")
    results = sorted(directory.glob("ids-*.json")) + sorted(directory.glob("status-*.txt"))
    verified = ""
    if plan_result != "success":
        problems.append(f"the plan job ended {plan_result!r}: its lock check, hygiene gate, ruff "
                        "or selection failed or did not run, so no shard ran")
    elif plan is None:
        problems.append(problem)
    elif plan["mode"] == "nothing":
        if results:
            problems.append(f"the plan selected no test file, yet {len(results)} shard result "
                            "file(s) were found")
    elif plan["mode"] != "selected":
        problems.append(f"the plan job succeeded with a `{plan['mode']}` plan")
    else:
        files = set(plan["paths"])
        for ids in sorted(directory.glob("ids-*.json")):
            try:
                selected = json.loads(ids.read_text(encoding="utf-8"))["selected"]
            except (OSError, ValueError, KeyError, TypeError) as e:
                problems.append(f"{ids.name} cannot be read: {e}")
                continue
            outside = sorted({n.split("::", 1)[0] for n in selected} - files)
            if outside:
                problems.append(f"{ids.name} ran tests outside the plan's selection: "
                                f"{', '.join(outside)}")
        status, verified = _verify_shards(directory, plan["count"])
        if status != 0:
            problems += [ln.removeprefix("::error::") for ln in verified.splitlines()
                         if ln.startswith("::error::")] or [f"tests._shard verify exited {status}"]
    lines = [plan["summary"] if plan is not None else "### Affected tests (python-pr-gate)\n", ""]
    if problems:
        lines.append("**The gate fails**:")
        lines += [f"- {p}" for p in problems]
    elif plan is not None and plan["mode"] == "nothing":
        lines.append("**The gate passes**: no test file is affected and no shard ran.")
    else:
        lines.append(f"**The gate passes**: {verified}")
    for p in problems:
        print(f"::error::{p}")
    _publish("\n".join(lines) + "\n")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--depth", type=int, default=1,
                    help="passed to the selector: follow importers this many hops (default 1); "
                         "a wide selection above 0 is retried at depth 0")
    ap.add_argument("--base", default="origin/master",
                    help="passed to the selector: ref the committed changes are diffed from")
    ap.add_argument("--files", nargs="*", help="passed to the selector: use these changed paths")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run without running pytest")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan-out", type=Path, metavar="FILE",
                      help="write the plan for the selection to FILE and run nothing")
    mode.add_argument("--shard", type=int, metavar="I",
                      help="run shard I of the selection in the --plan file")
    mode.add_argument("--shard-env", metavar="NAME",
                      help="--shard with I read from environment variable NAME, for a CI step "
                           "that runs under either shell")
    mode.add_argument("--collect", type=Path, metavar="DIR",
                      help="verify the shard results in DIR against the --plan file")
    ap.add_argument("--plan", type=Path, metavar="FILE", help="the plan file --plan-out wrote")
    ap.add_argument("--shard-out", type=Path, metavar="DIR",
                    help="with --shard: the directory the shard's ids-I.json goes to")
    ap.add_argument("--plan-result", metavar="RESULT",
                    help="with --collect: how the job that wrote the plan ended")
    ap.add_argument("pytest_args", nargs="*", help="arguments after -- are passed to pytest")
    args = ap.parse_args(argv)

    if args.shard_env is not None:
        value = os.environ.get(args.shard_env, "")
        if not value.strip().isdigit():
            ap.error(f"--shard-env {args.shard_env}: the variable holds {value!r}, not a shard number")
        args.shard = int(value)
    if args.shard is not None:
        if args.plan is None or args.shard_out is None:
            ap.error("--shard needs --plan and --shard-out")
        return run_shard(args.plan, args.shard, args.shard_out, args.pytest_args)
    if args.collect is not None:
        if args.plan is None or args.plan_result is None:
            ap.error("--collect needs --plan and --plan-result")
        return collect(args.collect, args.plan, args.plan_result)
    if args.plan is not None or args.shard_out is not None or args.plan_result is not None:
        ap.error("--plan, --shard-out and --plan-result go with --shard or --collect")

    selection = select(args.depth, args.base, args.files)
    if args.plan_out is not None:
        return write_plan(selection, args.plan_out, args.pytest_args)
    cmd = pytest_args(selection, args.pytest_args)

    if cmd is None:
        _publish(render_summary(selection, cmd))
        return 0 if selection.mode == "nothing" else 1
    if args.dry_run:
        _publish(render_summary(selection, cmd))
        return 0
    status = _run_pytest(cmd)
    _publish(render_summary(selection, cmd, pytest_status=status))
    if status == PYTEST_NO_TESTS_COLLECTED:
        return 0
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
