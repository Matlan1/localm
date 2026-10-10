# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/pin_weekly.py.

The confirm and bump scripts it drives are REAL subprocesses here: tiny stand-ins
written into a temp repo that follow dev-notes/pin-automation/CONTRACT.md (exit
0/1/2, a receipt file, a bump that edits a file). Only the git and gh effects
(worktree, branch, commit, PR, CI wait, merge) are replaced, by recording spies,
because there is no remote to push to.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "pin_weekly.py"
_spec = importlib.util.spec_from_file_location("pin_weekly_under_test", _PATH)
pw = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pw
_spec.loader.exec_module(pw)
pp = pw.pp
cp = pw.cp

CONFIRM = """\
import json, os, sys, time
args = sys.argv[1:]
receipt = args[args.index("--receipt") + 1]
rc = int(os.environ.get("FAKE_CONFIRM_RC", "0"))
if os.environ.get("FAKE_CONFIRM_SLEEP"):
    time.sleep(float(os.environ["FAKE_CONFIRM_SLEEP"]))
open(os.environ.get("FAKE_CONFIRM_LOG", os.devnull), "a").write(" ".join(args) + "\\n")
verdict = {0: "PASS", 1: "FAIL"}.get(rc, "INCONCLUSIVE")
json.dump({"schema": 1, "verdict": verdict, "why": os.environ.get("FAKE_CONFIRM_WHY", ""),
           "checks": {"load": {"status": verdict}}}, open(receipt, "w"))
sys.exit(rc)
"""

BUMP = """\
import os, sys
args = sys.argv[1:]
if os.environ.get("FAKE_BUMP_RC") == "1":
    print("REFUSED: bad receipt")
    sys.exit(1)
assert "--write" in args and "--tag" in args
open("bumped.txt", "w").write(args[args.index("--tag") + 1])
"""

LEASE = """\
import subprocess, sys
sys.exit(subprocess.call(sys.argv[sys.argv.index("--") + 1:]))
"""

ISSUES = (
    "an entry lives\nin exactly one. Move it when its state changes.\n\n"
    "OLD-ENTRY [OPEN] something\n"
)


class Spies:
    def __init__(self):
        self.calls: list[tuple] = []
        self.ci = "GREEN"


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "confirm_fake_runtime.py").write_text(CONFIRM, encoding="utf-8")
    (repo / "scripts" / "bump_fake_pin.py").write_text(BUMP, encoding="utf-8")
    worktree = tmp_path / "wt"
    (worktree / "scripts").mkdir(parents=True)
    (worktree / "scripts" / "bump_fake_pin.py").write_text(BUMP, encoding="utf-8")
    (worktree / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Changed\n- old bullet\n\n## [0.1.0]\n- x\n",
        encoding="utf-8")
    issues = tmp_path / "issues.txt"
    issues.write_text(ISSUES, encoding="utf-8")
    state = tmp_path / "state"
    monkeypatch.setattr(pw, "REPO", repo)
    monkeypatch.setattr(pw, "WEEKLY_DIR", state / "weekly")
    monkeypatch.setattr(pp, "STATE_DIR", state)
    monkeypatch.setattr(pp, "ISSUES_PATH", issues)
    for name in ("FAKE_CONFIRM_RC", "FAKE_CONFIRM_SLEEP", "FAKE_BUMP_RC", "FAKE_CONFIRM_LOG",
                 "FAKE_CONFIRM_WHY"):
        monkeypatch.delenv(name, raising=False)

    spies = Spies()
    monkeypatch.setattr(pp, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pp, "prepare_bump_branch",
                        lambda wt, cand, pin: spies.calls.append(("branch", cand, pin)) or "br")
    monkeypatch.setattr(pp, "commit_and_push",
                        lambda wt, br, cand, old, message=None: spies.calls.append(("push", cand)))
    monkeypatch.setattr(pp, "open_pr",
                        lambda wt, br, cand, old, title=None, body=None:
                        spies.calls.append(("pr", title)) or 77)
    monkeypatch.setattr(pw, "close_superseded_prs",
                        lambda wt, key, keep: spies.calls.append(("supersede", key, keep)) or [])
    monkeypatch.setattr(pp, "wait_for_ci", lambda wt: spies.calls.append(("ci",)) or spies.ci)
    monkeypatch.setattr(pp, "merge_pr",
                        lambda n, wt, br, cand, old, title=None, body=None:
                        spies.calls.append(("merge", n)))
    spies.repo, spies.worktree, spies.issues, spies.state = repo, worktree, issues, state
    return spies


def _adv(**kw):
    base = dict(key="fake", title="Fake runtime", confirm_script="scripts/confirm_fake_runtime.py",
                bump_script="scripts/bump_fake_pin.py", candidate=lambda: ("v1", "v2"), gpu=False)
    base.update(kw)
    return pw.Advancer(**base)


# --------------------------------------------------------------------------- #
#  advance()                                                                  #
# --------------------------------------------------------------------------- #

def test_nothing_newer_runs_no_confirm(env, monkeypatch, tmp_path):
    log = tmp_path / "confirm.log"
    monkeypatch.setenv("FAKE_CONFIRM_LOG", str(log))
    out = pw.advance(_adv(candidate=lambda: None), dry_run=False)
    assert out.verdict == pw.NONE_NEWER and not log.exists()


def test_missing_scripts_are_reported_not_built_never_current(env):
    out = pw.advance(_adv(confirm_script="scripts/confirm_absent_runtime.py"), dry_run=False)
    assert out.verdict == pw.NOT_BUILT and "does not exist" in out.detail
    out = pw.advance(_adv(bump_script="scripts/bump_absent_pin.py"), dry_run=False)
    assert out.verdict == pw.NOT_BUILT


def test_candidate_lookup_failure_is_inconclusive(env):
    def boom():
        raise cp.FetchError("api down")
    out = pw.advance(_adv(candidate=boom), dry_run=False)
    assert out.verdict == pw.INCONCLUSIVE and "api down" in out.detail


def test_dry_run_confirms_but_never_bumps(env, monkeypatch, tmp_path):
    log = tmp_path / "confirm.log"
    monkeypatch.setenv("FAKE_CONFIRM_LOG", str(log))
    out = pw.advance(_adv(), dry_run=True)
    assert out.verdict == pw.DRY_PASS
    assert "--tag v2" in log.read_text(encoding="utf-8")
    assert env.calls == []


def test_fail_is_recorded_logged_and_never_retried(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONFIRM_RC", "1")
    out = pw.advance(_adv(), dry_run=False)
    assert out.verdict == pw.FAIL
    assert pp.load_state(pin="fake")["verdict"] == "FAIL"
    assert "NEW-PIN-PIPELINE-FAKE-V2-CONFIRM-FAILED" in env.issues.read_text(encoding="utf-8")
    again = pw.advance(_adv(), dry_run=False)
    assert again.verdict == pw.SKIPPED and "already recorded FAIL" in again.detail
    assert env.calls == []


def test_a_newer_candidate_clears_a_recorded_fail(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONFIRM_RC", "1")
    pw.advance(_adv(), dry_run=False)
    monkeypatch.setenv("FAKE_CONFIRM_RC", "0")
    out = pw.advance(_adv(candidate=lambda: ("v1", "v3")), dry_run=True)
    assert out.verdict == pw.DRY_PASS


def test_inconclusive_is_recorded_cooled_down_and_not_logged_as_a_fail(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONFIRM_RC", "2")
    out = pw.advance(_adv(), dry_run=False)
    assert out.verdict == pw.INCONCLUSIVE
    assert pp.load_state(pin="fake")["verdict"] == "INCONCLUSIVE"
    assert "FAKE" not in env.issues.read_text(encoding="utf-8")
    assert pw.advance(_adv(), dry_run=False).verdict == pw.SKIPPED


def test_pass_bumps_in_the_worktree_commits_opens_pr_waits_and_merges(env, tmp_path):
    out = pw.advance(_adv(changelog=lambda o, n: f"- moved {o} to {n}\n"), dry_run=False)
    assert out.verdict == pw.MERGED and out.pr == 77
    assert (env.worktree / "bumped.txt").read_text(encoding="utf-8") == "v2"
    assert "- moved v1 to v2" in (env.worktree / "CHANGELOG.md").read_text(encoding="utf-8")
    assert [c[0] for c in env.calls] == ["supersede", "branch", "push", "pr", "ci", "merge"]
    assert env.calls[0] == ("supersede", "fake", "claude/pin-pipeline-fake-v2")
    assert env.calls[1] == ("branch", "v2", "fake")
    state = pp.load_state(pin="fake")
    assert state["verdict"] == "PASS" and state["merged_pr"] == 77


def test_a_refused_bump_is_a_fail_and_nothing_is_pushed(env, monkeypatch):
    monkeypatch.setenv("FAKE_BUMP_RC", "1")
    out = pw.advance(_adv(), dry_run=False)
    assert out.verdict == pw.FAIL and "REFUSED: bad receipt" in out.detail
    assert [c[0] for c in env.calls] == ["supersede", "branch"]


def test_failing_targeted_tests_stop_before_any_push(env):
    out = pw.advance(_adv(tests=("tests/test_that_does_not_exist.py",)), dry_run=False)
    assert out.verdict == pw.FAIL and "targeted tests failed" in out.detail
    assert "push" not in [c[0] for c in env.calls]


def test_a_candidate_that_needs_a_code_change_is_reported_once_not_as_a_build_fail(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONFIRM_RC", "1")
    monkeypatch.setenv("FAKE_CONFIRM_WHY", "binding needs a code update, not an automatic bump")
    out = pw.advance(_adv(candidate=lambda: ("v1", "v2")), dry_run=False)
    assert out.verdict == pw.NEEDS_UPDATE and "needs a change to localm's own code" in out.detail
    text = env.issues.read_text(encoding="utf-8")
    assert "NEW-PIN-PIPELINE-FAKE-BINDING-NEEDS-CODE-UPDATE" in text
    assert "CONFIRM-FAILED" not in text
    again = pw.advance(_adv(candidate=lambda: ("v1", "v3")), dry_run=False)
    assert again.verdict == pw.NEEDS_UPDATE
    assert env.issues.read_text(encoding="utf-8").count("NEEDS-CODE-UPDATE [OPEN") == 1
    assert env.calls == []


def test_an_ordinary_fail_is_still_a_build_fail(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONFIRM_RC", "1")
    out = pw.advance(_adv(), dry_run=False)
    assert out.verdict == pw.FAIL


def test_red_ci_is_a_fail_and_never_merges(env):
    env.ci = "RED"
    out = pw.advance(_adv(), dry_run=False)
    assert out.verdict == pw.FAIL and "CI red on PR #77" in out.detail
    assert "merge" not in [c[0] for c in env.calls]


def test_pending_ci_is_inconclusive_and_never_merges(env):
    env.ci = "PENDING"
    out = pw.advance(_adv(), dry_run=False)
    assert out.verdict == pw.INCONCLUSIVE
    assert "merge" not in [c[0] for c in env.calls]


def test_a_bump_that_is_not_auto_merged_opens_the_pr_and_stops(env):
    out = pw.advance(_adv(auto_merge=lambda o, n: False), dry_run=False)
    assert out.verdict == pw.REVIEW and out.pr == 77
    assert [c[0] for c in env.calls] == ["supersede", "branch", "push", "pr"]
    assert pp.load_state(pin="fake")["verdict"] == "REVIEW"


# --------------------------------------------------------------------------- #
#  Review-only pins (no confirm script)                                       #
# --------------------------------------------------------------------------- #

def _review_adv(**kw):
    return _adv(confirm_script=None, **kw)


def test_a_pin_without_a_confirm_script_can_never_auto_merge():
    assert _review_adv().auto_merge("v1", "v2") is False
    assert _adv().auto_merge("v1", "v2") is True


def test_review_only_bump_opens_a_pr_without_a_receipt_and_never_merges(env, monkeypatch, tmp_path):
    seen = []
    real = pw.subprocess.run

    def spy(cmd, *a, **k):
        seen.append(cmd)
        return real(cmd, *a, **k)
    monkeypatch.setattr(pw.subprocess, "run", spy)
    out = pw.advance(_review_adv(bump_args=("--lib", "marked")), dry_run=False)
    assert out.verdict == pw.REVIEW and out.pr == 77
    assert [c[0] for c in env.calls] == ["supersede", "branch", "push", "pr"]
    bump = next(c for c in seen if "bump_fake_pin.py" in " ".join(map(str, c)))
    flat = [str(x) for x in bump]
    assert "--receipt" not in flat
    assert flat[flat.index("--lib"):flat.index("--lib") + 4] == ["--lib", "marked", "--tag", "v2"]
    assert "--write" in flat
    assert (env.worktree / "bumped.txt").read_text(encoding="utf-8") == "v2"
    assert pp.load_state(pin="fake")["verdict"] == "REVIEW"


def test_review_only_dry_run_changes_nothing(env):
    out = pw.advance(_review_adv(), dry_run=True)
    assert out.verdict == pw.DRY_PASS and env.calls == []


def test_a_review_pr_already_open_for_the_same_candidate_is_not_reopened(env):
    pw.advance(_review_adv(), dry_run=False)
    env.calls.clear()
    again = pw.advance(_review_adv(), dry_run=False)
    assert again.verdict == pw.SKIPPED and "#77" in again.detail and env.calls == []


def test_a_newer_candidate_after_a_review_pr_opens_a_fresh_one(env):
    pw.advance(_review_adv(), dry_run=False)
    env.calls.clear()
    out = pw.advance(_review_adv(candidate=lambda: ("v1", "v3")), dry_run=False)
    assert out.verdict == pw.REVIEW
    assert env.calls[0] == ("supersede", "fake", "claude/pin-pipeline-fake-v3")


def test_verify_current_of_a_review_only_pin_is_reported_as_not_measured(env):
    out = pw.verify_current(_review_adv())
    assert out.verdict == pw.NOT_MEASURED and "no runtime check" in out.detail


def test_close_superseded_prs_closes_only_this_pins_older_branches(monkeypatch, tmp_path):
    listing = json.dumps([
        {"number": 1, "headRefName": "claude/pin-pipeline-fake-v1"},
        {"number": 2, "headRefName": "claude/pin-pipeline-fake-v3"},
        {"number": 3, "headRefName": "claude/pin-pipeline-other-v1"},
        {"number": 4, "headRefName": "claude/unrelated"}])
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=listing if cmd[2] == "list" else "", stderr="")
    monkeypatch.setattr(pw.subprocess, "run", fake_run)
    closed = pw.close_superseded_prs(tmp_path, "fake", "claude/pin-pipeline-fake-v3")
    assert closed == [1]
    assert [c[2] for c in calls] == ["list", "close"] and calls[1][3] == "1"


def test_close_superseded_prs_raises_infra_error_when_the_listing_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(pw.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stdout="", stderr="gh: not logged in"))
    with pytest.raises(pp.InfraError):
        pw.close_superseded_prs(tmp_path, "fake", "x")


def test_an_infra_error_after_a_pass_is_inconclusive_not_a_build_fail(env, monkeypatch):
    def boom(wt, br, cand, old, message=None):
        raise pp.InfraError("push failed")
    monkeypatch.setattr(pp, "commit_and_push", boom)
    out = pw.advance(_adv(), dry_run=False)
    assert out.verdict == pw.INCONCLUSIVE and "push failed" in out.detail
    assert "FAKE" not in env.issues.read_text(encoding="utf-8")


def test_an_unexpected_exception_after_a_pass_is_logged(env, monkeypatch):
    def boom(wt, br, cand, old, message=None):
        raise FileNotFoundError("gh not on PATH")
    monkeypatch.setattr(pp, "commit_and_push", boom)
    out = pw.advance(_adv(), dry_run=False)
    assert out.verdict == pw.INCONCLUSIVE and "gh not on PATH" in out.detail
    assert "FAKE" in env.issues.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Process handling                                                           #
# --------------------------------------------------------------------------- #

def test_run_cmd_times_out_kills_the_tree_and_reports_inconclusive(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    code = "; ".join([
        "import subprocess, sys, time",
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))",
        "time.sleep(60)",
    ])
    rc, out = pw.run_cmd([sys.executable, "-c", code], cwd=tmp_path, timeout=3)
    assert rc == 2 and "timed out" in out
    grandchild = int(pidfile.read_text(encoding="utf-8"))
    from localm.instances import pid_alive
    deadline = time.monotonic() + 10
    while pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not pid_alive(grandchild)


def test_run_cmd_returns_the_real_exit_code_and_output(tmp_path):
    rc, out = pw.run_cmd([sys.executable, "-c", "print('hi'); raise SystemExit(3)"],
                         cwd=tmp_path, timeout=30)
    assert rc == 3 and "hi" in out


def test_gpu_advancer_runs_under_the_lease_wrapper(env, monkeypatch, tmp_path):
    lease = tmp_path / "gpu_lease.py"
    lease.write_text(LEASE, encoding="utf-8")
    monkeypatch.setenv("LOCALM_PIN_PIPELINE_GPU_LEASE", str(lease))
    seen = []
    real = pw.run_cmd

    def spy(cmd, **kw):
        seen.append(cmd)
        return real(cmd, **kw)
    monkeypatch.setattr(pw, "run_cmd", spy)
    out = pw.advance(_adv(gpu=True), dry_run=True)
    assert out.verdict == pw.DRY_PASS
    assert str(lease) in seen[0] and "run" in seen[0] and "--" in seen[0]


def test_gpu_advancer_without_a_lease_script_is_inconclusive(env, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_PIN_PIPELINE_GPU_LEASE", str(tmp_path / "missing.py"))
    out = pw.advance(_adv(gpu=True), dry_run=True)
    assert out.verdict == pw.INCONCLUSIVE and "GPU lease script not found" in out.detail
    assert pp.load_state(pin="fake") == {}


def test_lease_busy_exit_is_not_run_and_says_why(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONFIRM_RC", str(pw.LEASE_BUSY_EXIT))
    env.state.mkdir(parents=True, exist_ok=True)
    verdict, detail = pw.run_confirm(_adv(), ["--tag", "v2"], env.state / "r.json")
    assert verdict == pw.NOT_RUN and "busy" in detail


def test_a_busy_gpu_records_no_verdict_so_the_candidate_is_retried_next_run(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONFIRM_RC", str(pw.LEASE_BUSY_EXIT))
    out = pw.advance(_adv(), dry_run=False)
    assert out.verdict == pw.INCONCLUSIVE and "busy" in out.detail
    assert pp.load_state(pin="fake") == {}


# --------------------------------------------------------------------------- #
#  verify_current                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rc,verdict", [(0, pw.PASS), (1, pw.FAIL), (2, pw.INCONCLUSIVE)])
def test_verify_current_maps_exit_codes(env, monkeypatch, tmp_path, rc, verdict):
    log = tmp_path / "confirm.log"
    monkeypatch.setenv("FAKE_CONFIRM_LOG", str(log))
    monkeypatch.setenv("FAKE_CONFIRM_RC", str(rc))
    out = pw.verify_current(_adv())
    assert out.verdict == verdict and out.kind == "verify-current"
    assert "--current" in log.read_text(encoding="utf-8")
    assert "load=" in out.detail


def test_verify_current_of_a_missing_confirm_script_is_not_built(env):
    out = pw.verify_current(_adv(confirm_script="scripts/confirm_absent_runtime.py"))
    assert out.verdict == pw.NOT_BUILT


def test_verify_current_uses_a_custom_command_when_given(env, tmp_path):
    marker = tmp_path / "ran.txt"

    def cmd(work, receipt):
        return [sys.executable, "-c",
                f"import json; open({str(marker)!r}, 'w').write('x'); "
                f"json.dump({{'checks': {{'a': {{'status': 'PASS'}}}}}}, open({str(receipt)!r}, 'w'))"]
    out = pw.verify_current(_adv(verify_current_cmd=cmd))
    assert out.verdict == pw.PASS and marker.exists() and "a=PASS" in out.detail


def test_a_broken_current_pin_is_logged_to_issues(env):
    out = pw.Outcome("fake", "Fake runtime", "verify-current", pw.FAIL, "load=FAIL")
    pw._report_current_failure(out)
    text = env.issues.read_text(encoding="utf-8")
    assert "CURRENT-PIN-BROKEN" in text and "no longer works" in text
    before = text
    pw._report_current_failure(out)
    assert env.issues.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------- #
#  Delegated pipelines                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rc,state,verdict", [
    (1, {"reason": "bad"}, pw.FAIL),
    (2, {"reason": "busy"}, pw.INCONCLUSIVE),
    (0, {"verdict": "PASS", "merged_pr": 5, "last_tag_tried": "b2"}, pw.MERGED),
    (0, {}, pw.NONE_NEWER),
])
def test_delegated_outcome_comes_from_exit_code_and_state(env, rc, state, verdict):
    pp.save_state(state, pin="deleg")
    adv = _adv(key="deleg", delegate=lambda dry: rc)
    assert pw.advance_delegated(adv, dry_run=False).verdict == verdict


def test_delegated_dry_run_pass(env):
    pp.save_state({"verdict": "PASS", "last_tag_tried": "b2"}, pin="deleg")
    assert pw.advance_delegated(_adv(key="deleg", delegate=lambda dry: 0), dry_run=True).verdict == pw.DRY_PASS


def test_delegated_pipeline_error_is_a_fail(env):
    def boom(dry):
        raise pp.PipelineError("pin constant renamed")
    out = pw.advance_delegated(_adv(key="deleg", delegate=boom), dry_run=False)
    assert out.verdict == pw.FAIL and "renamed" in out.detail


# --------------------------------------------------------------------------- #
#  run_weekly, report, exit codes                                             #
# --------------------------------------------------------------------------- #

def _stub_currency(monkeypatch):
    monkeypatch.setattr(cp, "build_registry", lambda: [])


def test_severity_ordering():
    o = lambda v: pw.Outcome("k", "t", "advance", v)  # noqa: E731
    assert pw.severity([o(pw.MERGED), o(pw.NONE_NEWER), o(pw.PASS)]) == 0
    assert pw.severity([o(pw.NOT_BUILT)]) == 2
    assert pw.severity([o(pw.INCONCLUSIVE), o(pw.PASS)]) == 2
    assert pw.severity([o(pw.INCONCLUSIVE), o(pw.FAIL)]) == 1
    assert pw.severity([o(pw.REVIEW)]) == 2
    assert pw.severity([o(pw.NOT_MEASURED)]) == 0
    assert pw.severity([o(pw.NEEDS_UPDATE)]) == 2
    assert pw.severity([]) == 0


def test_run_weekly_writes_a_dated_report_and_a_merged_pin_is_not_reverified(env, monkeypatch, tmp_path):
    _stub_currency(monkeypatch)
    log = tmp_path / "confirm.log"
    monkeypatch.setenv("FAKE_CONFIRM_LOG", str(log))
    code, path = pw.run_weekly([_adv()], dry_run=False)
    assert code == 0 and path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "## Advancing" in text and "MERGED" in text and "## Pinned build still works" in text
    assert (pw.WEEKLY_DIR / "latest.md").is_file()
    assert any(p.suffix == ".json" for p in pw.WEEKLY_DIR.iterdir())
    assert "--current" not in log.read_text(encoding="utf-8")


def test_run_weekly_verifies_the_current_pin_when_nothing_is_newer(env, monkeypatch, tmp_path):
    _stub_currency(monkeypatch)
    log = tmp_path / "confirm.log"
    monkeypatch.setenv("FAKE_CONFIRM_LOG", str(log))
    code, _ = pw.run_weekly([_adv(candidate=lambda: None)], dry_run=False)
    assert code == 0 and "--current" in log.read_text(encoding="utf-8")


def test_run_weekly_exits_one_and_logs_when_the_current_pin_is_broken(env, monkeypatch):
    _stub_currency(monkeypatch)
    monkeypatch.setenv("FAKE_CONFIRM_RC", "1")
    code, path = pw.run_weekly([_adv(candidate=lambda: None)], dry_run=False)
    assert code == 1
    assert "CURRENT-PIN-BROKEN" in env.issues.read_text(encoding="utf-8")
    assert "FAIL" in path.read_text(encoding="utf-8")


def test_run_weekly_exits_two_for_a_runtime_that_is_not_built(env, monkeypatch):
    _stub_currency(monkeypatch)
    code, _ = pw.run_weekly([_adv(confirm_script="scripts/confirm_absent_runtime.py")],
                            dry_run=False)
    assert code == 2


def test_run_weekly_skip_verify_current(env, monkeypatch, tmp_path):
    _stub_currency(monkeypatch)
    log = tmp_path / "confirm.log"
    monkeypatch.setenv("FAKE_CONFIRM_LOG", str(log))
    pw.run_weekly([_adv(candidate=lambda: None)], dry_run=False, verify=False)
    assert not log.exists()


def test_ensure_github_token_keeps_an_existing_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")

    def must_not_run(*a, **k):
        raise AssertionError("gh must not be asked when GITHUB_TOKEN is set")
    monkeypatch.setattr(pw.subprocess, "run", must_not_run)
    assert pw.ensure_github_token() == "env"


def test_ensure_github_token_exports_the_gh_token_to_children(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(pw.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(
        cmd, 0, stdout="tok-123\n", stderr=""))
    assert pw.ensure_github_token() == "gh"
    assert pw.os.environ["GITHUB_TOKEN"] == "tok-123"
    monkeypatch.delenv("GITHUB_TOKEN")


@pytest.mark.parametrize("result", [
    subprocess.CompletedProcess([], 1, stdout="", stderr="not logged in"),
    subprocess.CompletedProcess([], 0, stdout="  \n", stderr="")])
def test_ensure_github_token_without_a_usable_gh_token_is_none(monkeypatch, result):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(pw.subprocess, "run", lambda cmd, **k: result)
    assert pw.ensure_github_token() == "none" and "GITHUB_TOKEN" not in pw.os.environ


def test_ensure_github_token_when_gh_is_missing_is_none(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def missing(*a, **k):
        raise FileNotFoundError("gh")
    monkeypatch.setattr(pw.subprocess, "run", missing)
    assert pw.ensure_github_token() == "none"


def test_main_rejects_an_unknown_only_name(capsys):
    assert pw.main(["--only", "nope"]) == 2
    assert "no runtime matches" in capsys.readouterr().err


def test_main_reports_inconclusive_when_another_run_holds_the_lock(env, monkeypatch, capsys):
    def busy():
        raise pp.PipelineLockBusy("another run holds it")
    monkeypatch.setattr(pp, "pipeline_lock", busy)
    assert pw.main(["--only", "koboldcpp"]) == 2
    assert "INCONCLUSIVE" in capsys.readouterr().out


def test_main_infra_error_before_any_runtime_is_inconclusive(env, monkeypatch, capsys):
    import contextlib

    @contextlib.contextmanager
    def lock():
        yield
    monkeypatch.setattr(pp, "pipeline_lock", lock)

    def boom():
        raise pp.InfraError("git fetch failed")
    monkeypatch.setattr(pp, "sync_main_checkout", boom)
    assert pw.main(["--only", "koboldcpp"]) == 2


# --------------------------------------------------------------------------- #
#  Candidate detection and the shipped advancer table                         #
# --------------------------------------------------------------------------- #

def _rel(tag, days_ago):
    when = (dt.datetime(2026, 10, 10, tzinfo=dt.UTC) - dt.timedelta(days=days_ago))
    return {"tag_name": tag, "published_at": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prerelease": False, "draft": False}


def test_github_candidate_picks_the_newest_release_not_the_next_one(monkeypatch):
    monkeypatch.setattr(cp, "_get_json", lambda url: [_rel("v1.122.1", 40), _rel("v1.123.0", 20),
                                                       _rel("v1.125.0", 2)])
    find = pw._github_candidate("o/r", "localm/media/koboldcpp/pins.py", r'^TAG = "([^"]+)"',
                                cp._semver_key)
    assert find() == (cp._read_const("localm/media/koboldcpp/pins.py", r'^TAG = "([^"]+)"'),
                      "v1.125.0")


def test_github_candidate_none_when_the_pin_is_newest(monkeypatch):
    pinned = cp._read_const("localm/media/koboldcpp/pins.py", r'^TAG = "([^"]+)"')
    monkeypatch.setattr(cp, "_get_json", lambda url: [_rel(pinned, 3)])
    find = pw._github_candidate("o/r", "localm/media/koboldcpp/pins.py", r'^TAG = "([^"]+)"',
                                cp._semver_key)
    assert find() is None


def test_github_candidate_unparseable_pin_raises(monkeypatch):
    monkeypatch.setattr(cp, "_get_json", lambda url: [_rel("v9", 1)])
    find = pw._github_candidate("o/r", "localm/media/koboldcpp/pins.py", r'^TAG = "([^"]+)"',
                                lambda t: None)
    with pytest.raises(cp.FetchError):
        find()


def test_shipped_advancers_cover_every_runtime_with_a_pipeline_or_a_named_gap():
    advs = pw.build_advancers()
    assert [a.key for a in advs] == [
        "llama", "comfyui", "rocm", "koboldcpp", "sdcpp", "uv", "vendored-marked",
        "vendored-dompurify", "vendored-highlightjs", "vendored-katex", "cuda-runtime",
        "gguf-node", "docker-base"]
    assert all(a.auto_merge("1.0.0", "1.0.1") is False for a in advs if a.confirm_script is None)
    assert len({a.key for a in advs}) == len(advs)
    for a in advs:
        assert a.confirm_script is None or a.confirm_script.startswith("scripts/confirm_")
        assert a.bump_script.startswith("scripts/bump_")
        assert a.delegate is not None or a.candidate is not None
        for t in a.tests:
            assert t.startswith("tests/test_")


def test_every_advancer_script_that_exists_follows_the_contract_cli():
    for a in pw.build_advancers():
        scripts = [a.bump_script] if a.confirm_script is None else [a.confirm_script, a.bump_script]
        for rel in scripts:
            path = Path(_PATH).parent.parent / rel
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            assert '"--tag"' in text or a.delegate, rel
            if a.confirm_script is not None:
                assert '"--receipt"' in text, rel
            if rel == a.confirm_script and not a.verify_current_cmd:
                assert '"--current"' in text, rel
