# SPDX-License-Identifier: AGPL-3.0-or-later
"""The nightly and weekly crons of .github/workflows/ci.yml.

The nightly cron runs exactly test, gui-tests and optional-stacks; the weekly
cron keeps every job it ran before. Each job's `if:` is evaluated here, from
the real workflow text, under each trigger, by a small evaluator for the
GitHub Actions expression subset the file uses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

NIGHTLY = "23 3 * * *"
WEEKLY = "37 6 * * 1"

NIGHTLY_JOBS = {"test", "gui-tests", "optional-stacks", "lint"}
WEEKLY_JOBS = NIGHTLY_JOBS | {
    "abi-check", "comfyui-pin-check", "llama-rocm-pin-check",
    "mutation-run", "mutation-test", "web-search-canary", "voice-stack",
}

_TOKEN = re.compile(
    r"\s*(?:(?P<str>'[^']*')|(?P<op>&&|\|\||==|!=|[()!,])|(?P<name>[A-Za-z_][\w.*-]*))"
)


def _load() -> dict:
    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    wf["on"] = wf.pop(True, wf.get("on"))
    return wf


def _tokens(expr: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    expr = expr.strip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        assert m, f"cannot tokenise {expr[pos:]!r}"
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


class _Eval:
    """Recursive-descent evaluator: ||, &&, ==/!=, !, calls, strings, context
    paths. An unknown context path is null; null compares equal to ''."""

    def __init__(self, expr: str, ctx: dict):
        m = re.fullmatch(r"\s*\$\{\{(.*)\}\}\s*", expr, re.S)
        self.toks = _tokens(m.group(1) if m else expr)
        self.i = 0
        self.ctx = ctx

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def _take(self):
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def run(self) -> bool:
        value = self._or()
        assert self.i == len(self.toks), f"trailing tokens {self.toks[self.i:]}"
        return bool(value)

    def _or(self):
        left = self._and()
        while self._peek() == ("op", "||"):
            self._take()
            right = self._and()
            left = left or right
        return left

    def _and(self):
        left = self._cmp()
        while self._peek() == ("op", "&&"):
            self._take()
            right = self._cmp()
            left = left and right
        return left

    def _cmp(self):
        left = self._unary()
        while self._peek() in (("op", "=="), ("op", "!=")):
            op = self._take()[1]
            right = self._unary()
            same = (left or "") == (right or "")
            left = same if op == "==" else not same
        return left

    def _unary(self):
        if self._peek() == ("op", "!"):
            self._take()
            return not self._unary()
        return self._atom()

    def _atom(self):
        kind, text = self._take()
        if kind == "str":
            return text[1:-1]
        if (kind, text) == ("op", "("):
            value = self._or()
            assert self._take() == ("op", ")")
            return value
        assert kind == "name", (kind, text)
        if self._peek() == ("op", "("):
            self._take()
            args = []
            while self._peek() != ("op", ")"):
                args.append(self._or())
                if self._peek() == ("op", ","):
                    self._take()
            self._take()
            return self._call(text, args)
        return self.ctx.get(text)

    def _call(self, name: str, args: list):
        if name == "contains":
            haystack, needle = args
            return needle in (haystack or [])
        if name == "cancelled":
            return False
        raise AssertionError(f"unsupported function {name}")


def _scenario(event: str, schedule: str = "", labels: tuple = (), run_abi: bool = False) -> dict:
    return {
        "github.event_name": event,
        "github.event.schedule": schedule,
        "github.event.pull_request.labels.*.name": list(labels),
        "inputs.run_abi": run_abi or None,
    }


def _jobs_that_run(scenario: dict) -> set[str]:
    jobs = _load()["jobs"]
    ran: set[str] = set()
    for _ in range(len(jobs)):
        for name, job in jobs.items():
            if name in ran:
                continue
            ctx = dict(scenario)
            for need in job.get("needs", []):
                ctx[f"needs.{need}.result"] = "success" if need in ran else "skipped"
            cond = job.get("if")
            if cond is None or _Eval(str(cond), ctx).run():
                ran.add(name)
    return ran


def test_the_workflow_has_exactly_a_nightly_and_a_weekly_cron():
    crons = [entry["cron"] for entry in _load()["on"]["schedule"]]
    assert sorted(crons) == sorted([NIGHTLY, WEEKLY])
    for cron in crons:
        minute = cron.split()[0]
        assert minute not in ("0", "30"), "keep scheduled runs off the top of the hour"


def test_no_other_workflow_shares_a_cron_slot():
    ours = {NIGHTLY, WEEKLY}
    checked = 0
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        if path == _CI:
            continue
        wf = yaml.safe_load(path.read_text(encoding="utf-8"))
        wf["on"] = wf.pop(True, wf.get("on"))
        if not isinstance(wf["on"], dict):
            continue
        for entry in wf["on"].get("schedule", []):
            checked += 1
            assert entry["cron"] not in ours, path.name
    assert checked >= 1


def test_the_nightly_run_runs_exactly_the_matrix_jobs_and_lint():
    assert _jobs_that_run(_scenario("schedule", NIGHTLY)) == NIGHTLY_JOBS


def test_the_weekly_run_runs_everything_it_ran_before_the_split():
    assert _jobs_that_run(_scenario("schedule", WEEKLY)) == WEEKLY_JOBS


def test_every_weekly_only_job_is_absent_from_the_nightly_run():
    nightly = _jobs_that_run(_scenario("schedule", NIGHTLY))
    for job in WEEKLY_JOBS - NIGHTLY_JOBS:
        assert job not in nightly, job


def test_a_dispatch_still_runs_the_weekly_only_jobs_except_the_opt_in_abi_check():
    ran = _jobs_that_run(_scenario("workflow_dispatch"))
    assert ran >= WEEKLY_JOBS - {"abi-check"}
    assert "abi-check" not in ran
    assert "abi-check" in _jobs_that_run(_scenario("workflow_dispatch", run_abi=True))


def test_the_push_trigger_runs_the_same_jobs_as_before_the_split():
    assert _jobs_that_run(_scenario("push")) == {
        "abi-check", "comfyui-pin-check", "llama-rocm-pin-check"}


@pytest.mark.parametrize("labels,expected_present,expected_absent", [
    ((), {"lint", "python-pr-gate", "voice-stack", "merge-policy"}, {"test", "optional-stacks"}),
    (("full-ci",), {"lint", "test", "gui-tests", "optional-stacks", "merge-policy"}, {"python-pr-gate"}),
])
def test_pull_request_behaviour_is_unchanged_by_the_schedule_split(labels, expected_present, expected_absent):
    ran = _jobs_that_run(_scenario("pull_request", labels=labels))
    assert expected_present <= ran
    assert not (expected_absent & ran)


def test_the_concurrency_group_separates_the_two_crons():
    group = _load()["concurrency"]["group"]
    assert "github.event.schedule" in group


def test_the_evaluator_fires_a_job_that_runs_on_every_schedule():
    """Control: an `if` without the weekly term is reported as running nightly."""
    ctx = _scenario("schedule", NIGHTLY)
    assert _Eval("github.event_name == 'schedule'", ctx).run()
    assert not _Eval(
        "github.event_name == 'schedule' && github.event.schedule == '37 6 * * 1'", ctx).run()
