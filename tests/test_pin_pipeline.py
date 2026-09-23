# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/pin_pipeline.py.

Three layers, matching how the script itself is built:
  * pure functions (should_skip, changelog_bullet, insert_changelog_bullet,
    append_fail_issue) - tested directly against real/synthetic text, no I/O
    mocking needed.
  * subprocess-boundary functions (run_confirm, run_bump, run_under_gpu_lease,
    wait_for_ci) - subprocess.run is monkeypatched to a recording fake, so
    what is asserted is the ACTUAL command/cwd/env constructed, never a
    hand-waved "it was called".
  * git-worktree functions (ensure_pipeline_worktree, prepare_bump_branch) -
    exercised against a REAL throwaway git repo (never the real localm repo),
    because a worktree-isolation safety check is exactly the kind of thing
    that needs real git behaviour behind it, not a mock that assumes the
    answer.

No real network, no real GPU, no real localm import needed for any of this -
pin_pipeline.py's own subprocess calls are the seam.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "pin_pipeline.py"
_spec = importlib.util.spec_from_file_location("pin_pipeline", _PATH)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def _day(n: int) -> dt.datetime:
    return dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=n)


def _iso(d: dt.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
#  should_skip - the never-retry-a-known-bad-candidate logic                 #
# --------------------------------------------------------------------------- #

def test_should_skip_none_when_no_state_at_all():
    assert pipeline.should_skip({}, "b10999") is None


def test_should_skip_none_when_state_is_for_a_different_older_candidate():
    """A FAIL recorded against an OLDER candidate must never block a NEWER
    one - only the exact same candidate is ever skipped."""
    state = {"last_tag_tried": "b10900", "verdict": "FAIL", "timestamp": _iso(_day(0))}
    assert pipeline.should_skip(state, "b10999") is None


def test_should_skip_fail_is_never_retried_regardless_of_age():
    state = {"last_tag_tried": "b10999", "verdict": "FAIL", "timestamp": _iso(_day(0))}
    reason = pipeline.should_skip(state, "b10999", now=_day(9999))
    assert reason is not None and "FAIL" in reason


def test_should_skip_inconclusive_retried_after_cooldown_not_before():
    state = {"last_tag_tried": "b10999", "verdict": "INCONCLUSIVE", "timestamp": _iso(_day(0))}
    within = pipeline.should_skip(state, "b10999", now=_day(0) + dt.timedelta(hours=1))
    assert within is not None and "cooldown" in within
    after = pipeline.should_skip(
        state, "b10999",
        now=_day(0) + dt.timedelta(hours=pipeline.INCONCLUSIVE_COOLDOWN_HOURS + 1))
    assert after is None


def test_should_skip_pass_state_never_blocks_a_later_run():
    """A PASS is historical fact (this candidate already merged); it must
    never read as a reason to skip a check on the SAME tag run again later
    (which will simply find nothing newer via newest_candidate() anyway)."""
    state = {"last_tag_tried": "b10999", "verdict": "PASS", "timestamp": _iso(_day(0))}
    assert pipeline.should_skip(state, "b10999") is None


# --------------------------------------------------------------------------- #
#  newest_candidate - imports check_llama_pin.py directly, no text-scraping  #
# --------------------------------------------------------------------------- #

def test_newest_candidate_none_when_already_current(tmp_path, monkeypatch):
    check_llama_pin = pipeline._load_module(
        Path(pipeline.REPO) / "scripts" / "check_llama_pin.py", "check_llama_pin_for_test")
    pin = check_llama_pin.pinned_tag()
    monkeypatch.setattr(pipeline, "_load_module", lambda path, name: check_llama_pin)
    monkeypatch.setattr(check_llama_pin, "upstream_releases",
                        lambda: ([{"tag": pin, "published_at": None}], ""))
    assert pipeline.newest_candidate() is None


def test_newest_candidate_returns_pair_when_upstream_is_ahead(monkeypatch):
    check_llama_pin = pipeline._load_module(
        Path(pipeline.REPO) / "scripts" / "check_llama_pin.py", "check_llama_pin_for_test2")
    pin = check_llama_pin.pinned_tag()
    n = check_llama_pin._build_number(pin)
    newest = f"b{n + 5}"
    monkeypatch.setattr(pipeline, "_load_module", lambda path, name: check_llama_pin)
    monkeypatch.setattr(check_llama_pin, "upstream_releases",
                        lambda: ([{"tag": newest, "published_at": None},
                                  {"tag": pin, "published_at": None}], ""))
    assert pipeline.newest_candidate() == (pin, newest)


def test_newest_candidate_none_on_upstream_error(monkeypatch):
    check_llama_pin = pipeline._load_module(
        Path(pipeline.REPO) / "scripts" / "check_llama_pin.py", "check_llama_pin_for_test3")
    monkeypatch.setattr(pipeline, "_load_module", lambda path, name: check_llama_pin)
    monkeypatch.setattr(check_llama_pin, "upstream_releases", lambda: ([], "simulated failure"))
    assert pipeline.newest_candidate() is None


# --------------------------------------------------------------------------- #
#  changelog                                                                  #
# --------------------------------------------------------------------------- #

def test_changelog_bullet_names_both_tags_and_the_pickup_command():
    bullet = pipeline.changelog_bullet("b10905", "b10999")
    assert "b10905" in bullet and "b10999" in bullet
    assert "localm setup-llama --force" in bullet


def test_insert_changelog_bullet_creates_changed_section_when_absent():
    text = "## [Unreleased]\n\n### Added\n- something\n\n## [0.1.0] - 2026-01-01\nold\n"
    out = pipeline.insert_changelog_bullet(text, pipeline.changelog_bullet("a", "b"))
    assert "### Changed\n- **The bundled llama.cpp runtime moved from a to b.**" in out
    assert "## [0.1.0] - 2026-01-01\nold" in out, "the released section must be untouched"


def test_insert_changelog_bullet_prepends_under_existing_changed_section():
    text = ("## [Unreleased]\n\n### Changed\n- existing bullet\n\n"
            "## [0.1.0] - 2026-01-01\nold\n")
    out = pipeline.insert_changelog_bullet(text, pipeline.changelog_bullet("a", "b"))
    changed_idx = out.index("### Changed\n")
    new_bullet_idx = out.index("moved from a to b")
    existing_idx = out.index("existing bullet")
    assert changed_idx < new_bullet_idx < existing_idx, "new bullet goes first, existing stays"


def test_insert_changelog_bullet_never_touches_a_released_section():
    """FIRES: a naive non-anchored regex could match the FIRST '## [' heading
    of a RELEASED section instead of [Unreleased] if the anchor were wrong -
    prove the released section's own content is byte-identical after the
    insert."""
    text = ("## [Unreleased]\n\n### Added\n- x\n\n"
            "## [0.2.0] - 2026-02-01\n\n### Changed\n- a released bullet, never move this\n")
    out = pipeline.insert_changelog_bullet(text, pipeline.changelog_bullet("a", "b"))
    assert "## [0.2.0] - 2026-02-01\n\n### Changed\n- a released bullet, never move this\n" in out


def test_insert_changelog_bullet_refuses_when_no_unreleased_section():
    with pytest.raises(pipeline.PipelineError):
        pipeline.insert_changelog_bullet("# Changelog\nno unreleased section here\n",
                                         pipeline.changelog_bullet("a", "b"))


def test_insert_changelog_bullet_against_the_real_shipped_changelog():
    """Bound to the real file, not only a synthetic fixture - proves the
    anchor still matches the actual shipped CHANGELOG.md shape."""
    real = (Path(pipeline.REPO) / "CHANGELOG.md").read_text(encoding="utf-8")
    out = pipeline.insert_changelog_bullet(real, pipeline.changelog_bullet("bOLD", "bNEW"))
    assert "moved from bOLD to bNEW" in out
    assert out.count("## [Unreleased]") == 1


# --------------------------------------------------------------------------- #
#  issues.txt logging                                                        #
# --------------------------------------------------------------------------- #

_ISSUES_FIXTURE = (
    "LocaLM - issue backlog\n======================\n\n"
    "The OPEN work, grouped by STATE. Each section below is one state; an entry lives\n"
    "in exactly one. Resolved items are removed once merged.\n\n"
    "SOME-EXISTING-ENTRY [OPEN] category - unrelated\n"
)


def test_append_fail_issue_inserts_right_after_the_intro_anchor(tmp_path):
    path = tmp_path / "issues.txt"
    path.write_text(_ISSUES_FIXTURE, encoding="utf-8")
    wrote = pipeline.append_fail_issue("b10999", "simulated FAIL reason", Path("r.json"),
                                       issues_path=path)
    assert wrote is True
    out = path.read_text(encoding="utf-8")
    assert "NEW-PIN-PIPELINE-LLAMA-B10999-CONFIRM-FAILED" in out
    assert out.index("NEW-PIN-PIPELINE-LLAMA") < out.index("SOME-EXISTING-ENTRY"), (
        "the new entry goes at the top, the existing entry must survive untouched")
    assert "simulated FAIL reason" in out


def test_append_fail_issue_is_idempotent_for_the_same_candidate(tmp_path):
    path = tmp_path / "issues.txt"
    path.write_text(_ISSUES_FIXTURE, encoding="utf-8")
    first = pipeline.append_fail_issue("b10999", "first reason", None, issues_path=path)
    second = pipeline.append_fail_issue("b10999", "second reason", None, issues_path=path)
    assert first is True and second is True
    out = path.read_text(encoding="utf-8")
    assert out.count("NEW-PIN-PIPELINE-LLAMA-B10999-CONFIRM-FAILED") == 1
    assert "first reason" in out and "second reason" not in out


def test_append_fail_issue_no_op_when_file_missing(tmp_path):
    missing = tmp_path / "does-not-exist.txt"
    wrote = pipeline.append_fail_issue("b10999", "reason", None, issues_path=missing)
    assert wrote is False
    assert not missing.exists()


def test_append_fail_issue_warns_and_returns_false_when_anchor_no_longer_matches(tmp_path, capsys):
    path = tmp_path / "issues.txt"
    path.write_text("LocaLM - issue backlog\n======================\n\nsomething else entirely\n",
                    encoding="utf-8")
    wrote = pipeline.append_fail_issue("b10999", "reason", None, issues_path=path)
    assert wrote is False
    assert path.read_text(encoding="utf-8") == (
        "LocaLM - issue backlog\n======================\n\nsomething else entirely\n"), (
        "a mismatched anchor must never touch the file's content")
    captured = capsys.readouterr()
    assert "WARNING" in captured.out and "anchor" in captured.out


def test_issues_intro_anchor_matches_the_real_shipped_file(tmp_path):
    """Bound to the real issues/issues.txt, not only a synthetic fixture -
    proves _ISSUES_INTRO_RE still matches the real file's actual current
    shape, per diff-review-discipline item 19's remedy for fixture blindness."""
    real = pipeline.REPO / "issues" / "issues.txt"
    if not real.exists():
        pytest.skip("the real local issue backlog file does not exist on this machine")
    before = real.read_text(encoding="utf-8")
    m = pipeline._ISSUES_INTRO_RE.search(before)
    assert m is not None, "the anchor regex must match the real file's current intro paragraph"

    path = tmp_path / "issues.txt"
    path.write_text(before, encoding="utf-8")
    wrote = pipeline.append_fail_issue("b10999-real-anchor-probe", "simulated", None,
                                       issues_path=path)

    assert wrote is True
    after = path.read_text(encoding="utf-8")
    assert after.startswith(before[:m.end()]), "everything before the anchor must survive untouched"
    assert after.endswith(before[m.end():]), "everything after the anchor must survive untouched"
    assert "NEW-PIN-PIPELINE-LLAMA-B10999-REAL-ANCHOR-PROBE-CONFIRM-FAILED" in after


# --------------------------------------------------------------------------- #
#  gpu_lease_script - configurable, verified to exist, never silently absent #
# --------------------------------------------------------------------------- #

def test_gpu_lease_script_uses_env_override_when_set(tmp_path, monkeypatch):
    fake = tmp_path / "gpu_lease.py"
    fake.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setenv(pipeline._GPU_LEASE_ENV, str(fake))
    assert pipeline.gpu_lease_script() == fake


def test_gpu_lease_script_refuses_loudly_when_not_found(monkeypatch):
    monkeypatch.setenv(pipeline._GPU_LEASE_ENV, "Z:/definitely/not/a/real/path/gpu_lease.py")
    with pytest.raises(pipeline.PipelineError, match="GPU lease script not found"):
        pipeline.gpu_lease_script()


# --------------------------------------------------------------------------- #
#  subprocess-boundary functions - assert the ACTUAL command constructed     #
# --------------------------------------------------------------------------- #

class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_under_gpu_lease_wraps_the_command_and_passes_purpose(monkeypatch, tmp_path):
    fake_lease = tmp_path / "gpu_lease.py"
    fake_lease.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setenv(pipeline._GPU_LEASE_ENV, str(fake_lease))
    captured = {}

    def fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return _FakeCompleted(returncode=0)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc = pipeline.run_under_gpu_lease(["echo", "hi"], purpose="test purpose", cwd=tmp_path)
    assert rc == 0
    assert captured["cwd"] == tmp_path
    cmd = captured["cmd"]
    assert cmd[-3:] == ["--", "echo", "hi"]
    assert "--purpose" in cmd and "test purpose" in cmd
    assert str(fake_lease) in cmd


def test_run_confirm_requests_cpu_and_vulkan_and_the_given_receipt(monkeypatch, tmp_path):
    fake_lease = tmp_path / "gpu_lease.py"
    fake_lease.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setenv(pipeline._GPU_LEASE_ENV, str(fake_lease))
    captured = {}

    def fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompleted(returncode=0)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    receipt = tmp_path / "r.json"
    pipeline.run_confirm("b10999", receipt)
    cmd = captured["cmd"]
    assert "confirm_llama_runtime.py" in " ".join(cmd)
    assert cmd.count("--backend") == 2
    assert "cpu" in cmd and "vulkan" in cmd
    assert "--tag" in cmd and "b10999" in cmd
    assert str(receipt) in cmd


def test_run_bump_uses_the_worktrees_own_copy_not_the_main_checkout(monkeypatch, tmp_path):
    """FIRES: the whole point of this function is running the WORKTREE's
    bump_llama_pin.py, never the main checkout's - prove the constructed
    command path is rooted at the worktree argument, and that PYTHONPATH is
    set to the worktree (the documented worktree-import gotcha)."""
    worktree = tmp_path / "the-worktree"
    worktree.mkdir()
    captured = {}

    def fake_run(cmd, cwd=None, env=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return _FakeCompleted(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc, _ = pipeline.run_bump(worktree, "b10999", tmp_path / "r.json", write=True)
    assert rc == 0
    cmd = captured["cmd"]
    assert str(worktree / "scripts" / "bump_llama_pin.py") in cmd
    assert "--write" in cmd
    assert captured["cwd"] == worktree
    assert captured["env"]["PYTHONPATH"] == str(worktree)


def test_run_bump_without_write_omits_the_flag(monkeypatch, tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    captured = {}
    monkeypatch.setattr(pipeline.subprocess, "run",
                        lambda cmd, **k: (captured.update(cmd=cmd), _FakeCompleted())[-1])
    pipeline.run_bump(worktree, "b10999", tmp_path / "r.json", write=False)
    assert "--write" not in captured["cmd"]


def test_run_targeted_tests_runs_against_the_worktree_with_pythonpath_set(monkeypatch, tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    captured = {}

    def fake_run(cmd, cwd=None, env=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return _FakeCompleted(returncode=0)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    passed, _ = pipeline.run_targeted_tests(worktree)
    assert passed
    assert captured["cwd"] == worktree
    assert captured["env"]["PYTHONPATH"] == str(worktree)
    assert "test_llama_pin_constant_and_currency.py" in " ".join(captured["cmd"])
    assert "not integration" in " ".join(captured["cmd"])


# --------------------------------------------------------------------------- #
#  wait_for_ci - GREEN/RED/PENDING, never mergeable/mergeStateStatus         #
# --------------------------------------------------------------------------- #

def _checks_response(runs):
    return _FakeCompleted(returncode=0, stdout=json.dumps({"check_runs": runs}))


def test_wait_for_ci_green_when_everything_completed_and_succeeded(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "_run_git", lambda args, cwd, **k: _FakeCompleted(stdout="deadbeef\n"))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda cmd, **k: _checks_response(
        [{"status": "completed", "conclusion": "success"},
         {"status": "completed", "conclusion": "success"}]))
    assert pipeline.wait_for_ci(tmp_path) == "GREEN"


def test_wait_for_ci_red_on_any_failure_even_if_others_still_running(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "_run_git", lambda args, cwd, **k: _FakeCompleted(stdout="deadbeef\n"))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda cmd, **k: _checks_response(
        [{"status": "completed", "conclusion": "failure"},
         {"status": "in_progress", "conclusion": None}]))
    assert pipeline.wait_for_ci(tmp_path) == "RED"


def test_wait_for_ci_ignores_skipped_checks_for_the_green_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "_run_git", lambda args, cwd, **k: _FakeCompleted(stdout="deadbeef\n"))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda cmd, **k: _checks_response(
        [{"status": "completed", "conclusion": "success"},
         {"status": "completed", "conclusion": "skipped"}]))
    assert pipeline.wait_for_ci(tmp_path) == "GREEN"


def test_wait_for_ci_pending_on_timeout_never_green_never_red(monkeypatch, tmp_path):
    """An empty/still-running check list must never be misread as GREEN -
    same class of bug documented for wait_for_checks.py in this repo."""
    monkeypatch.setattr(pipeline, "_run_git", lambda args, cwd, **k: _FakeCompleted(stdout="deadbeef\n"))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda cmd, **k: _checks_response([]))
    calls = {"n": 0}
    real_monotonic = pipeline.time.monotonic
    start = real_monotonic()

    def fake_monotonic():
        calls["n"] += 1
        # advance past the deadline after a couple of polls
        return start + (pipeline.CI_WAIT_TIMEOUT_SECONDS + 1 if calls["n"] > 2 else 0)
    monkeypatch.setattr(pipeline.time, "monotonic", fake_monotonic)
    assert pipeline.wait_for_ci(tmp_path) == "PENDING"


# --------------------------------------------------------------------------- #
#  git-worktree functions - real throwaway git repo, never the real localm   #
# --------------------------------------------------------------------------- #

def _init_scratch_repo(root: Path) -> Path:
    """A tiny, real, throwaway git repo standing in for the localm main
    checkout - never touches the real one."""
    repo = root / "scratch-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "master"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("scratch\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def test_ensure_pipeline_worktree_refuses_when_not_on_master(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    subprocess.run(["git", "checkout", "-q", "-b", "not-master"], cwd=repo, check=True)
    with pytest.raises(pipeline.PipelineError, match="not master"):
        pipeline.ensure_pipeline_worktree(repo)


def test_ensure_pipeline_worktree_refuses_when_run_from_a_linked_worktree(tmp_path):
    """A linked worktree's OWN toplevel is itself, so the toplevel-consistency
    check alone cannot tell it apart from the main checkout - what actually
    catches it is that git refuses to let two worktrees hold the same branch
    at once, so a linked worktree can never itself be `master` while the main
    checkout holds it. Confirm the refusal fires anyway, via that mechanism."""
    repo = _init_scratch_repo(tmp_path)
    other = tmp_path / "some-other-worktree"
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(other), "master"],
                   cwd=repo, check=True)
    with pytest.raises(pipeline.PipelineError, match="not master"):
        pipeline.ensure_pipeline_worktree(other)


def test_ensure_pipeline_worktree_creates_and_reuses_the_same_dedicated_worktree(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    # No real "origin" remote exists for this scratch repo; ensure_pipeline_worktree
    # fetches origin (best-effort, check=False) then creates off origin/master,
    # which does not exist here - so point it at "master" directly by pre-creating
    # an origin/master-shaped ref is unnecessary: git worktree add falls back to
    # the local branch when origin/master is unresolvable only if we ask it to;
    # instead, add a same-named local remote pointing at itself so origin/master
    # resolves for real, matching what a real clone looks like.
    subprocess.run(["git", "remote", "add", "origin", str(repo)], cwd=repo, check=True)
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=repo, check=True)

    first = pipeline.ensure_pipeline_worktree(repo)
    assert first.exists()
    assert first == repo.parent / f"{repo.name}-pin-pipeline-worktree"

    second = pipeline.ensure_pipeline_worktree(repo)
    assert second == first, "a second call must reuse the same worktree, not create another"


def test_sync_main_checkout_refuses_when_not_on_master(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    subprocess.run(["git", "checkout", "-q", "-b", "not-master"], cwd=repo, check=True)
    with pytest.raises(pipeline.PipelineError, match="not master"):
        pipeline.sync_main_checkout(repo)


def test_sync_main_checkout_refuses_when_run_from_a_linked_worktree(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    other = tmp_path / "some-other-worktree"
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(other), "master"],
                   cwd=repo, check=True)
    with pytest.raises(pipeline.PipelineError, match="not master"):
        pipeline.sync_main_checkout(other)


def test_sync_main_checkout_fast_forwards_to_a_newer_origin_master(tmp_path):
    """The real bug this exists to fix: newest_candidate() reads the pin
    straight off this checkout's working tree, and nothing else in the
    pipeline ever refreshes it (a merge happens from the dedicated worktree
    instead) - so after another commit lands on origin/master (e.g. a prior
    successful merge by this same pipeline), the main checkout must catch up
    or it would re-read a stale pin forever. Needs a genuinely separate
    (bare) origin: pushing to a non-bare repo's own checked-out branch is
    refused by git outright."""
    repo = tmp_path / "clone"
    bare = tmp_path / "bare-origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "master", str(bare)], check=True)
    subprocess.run(["git", "clone", "-q", str(bare), str(repo)], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("scratch\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    subprocess.run(["git", "push", "-q", "-u", "origin", "master"], cwd=repo, check=True)

    # Simulate another clone of the same remote landing a new commit.
    other_clone = tmp_path / "other-clone"
    subprocess.run(["git", "clone", "-q", str(bare), str(other_clone)], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=other_clone, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=other_clone, check=True)
    (other_clone / "NEW-FILE.md").write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "add", "NEW-FILE.md"], cwd=other_clone, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "advance"], cwd=other_clone, check=True)
    subprocess.run(["git", "push", "-q", "origin", "master"], cwd=other_clone, check=True)

    before = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=repo, capture_output=True, text=True).stdout.strip()
    assert not (repo / "NEW-FILE.md").exists(), "sanity: repo has not seen the new commit yet"

    pipeline.sync_main_checkout(repo)

    after = subprocess.run(["git", "rev-parse", "HEAD"],
                           cwd=repo, capture_output=True, text=True).stdout.strip()
    assert after != before, "the main checkout must have advanced"
    assert (repo / "NEW-FILE.md").exists(), "the working tree content must reflect the new commit"


def test_prepare_bump_branch_creates_a_fresh_branch_off_origin_master(tmp_path, monkeypatch):
    repo = _init_scratch_repo(tmp_path)
    subprocess.run(["git", "remote", "add", "origin", str(repo)], cwd=repo, check=True)
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=repo, check=True)
    worktree = pipeline.ensure_pipeline_worktree(repo)
    monkeypatch.setattr(pipeline, "close_stale_pr", lambda branch, worktree: None)

    branch = pipeline.prepare_bump_branch(worktree, "b10999")
    assert branch == "claude/pin-pipeline-llama-b10999"
    current = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                             cwd=worktree, capture_output=True, text=True).stdout.strip()
    assert current == branch


def test_prepare_bump_branch_refuses_when_fetch_fails(tmp_path):
    """A failed fetch must raise InfraError rather than silently falling
    through to branch off whatever the worktree was already on."""
    repo = _init_scratch_repo(tmp_path)
    subprocess.run(["git", "remote", "add", "origin", str(repo)], cwd=repo, check=True)
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=repo, check=True)
    worktree = pipeline.ensure_pipeline_worktree(repo)
    head_before = subprocess.run(["git", "rev-parse", "HEAD"],
                                 cwd=worktree, capture_output=True, text=True).stdout.strip()

    subprocess.run(["git", "remote", "set-url", "origin", str(tmp_path / "no-such-remote")],
                   cwd=worktree, check=True)

    with pytest.raises(pipeline.InfraError, match="fetch"):
        pipeline.prepare_bump_branch(worktree, "b10999")

    head_after = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=worktree, capture_output=True, text=True).stdout.strip()
    assert head_after == head_before, "a failed fetch must leave the worktree untouched"


def _init_bare_origin_and_clone(tmp_path):
    """A genuinely separate bare origin plus a real clone, mirroring an
    actual GitHub remote - unlike _init_scratch_repo's `remote add origin
    <self>` pattern, a linked worktree does NOT share refs with this origin,
    so a push here exercises real non-fast-forward semantics rather than the
    degenerate same-object-database case."""
    bare = tmp_path / "bare-origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "master", str(bare)], check=True)
    repo = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(bare), str(repo)], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("scratch\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    subprocess.run(["git", "push", "-q", "-u", "origin", "master"], cwd=repo, check=True)
    return repo


def test_close_stale_pr_no_op_when_none_open(tmp_path, monkeypatch):
    """No open PR AND no remote branch (the common case: nothing to clean
    up) must never attempt a close or a delete. gh is stubbed; the git
    ls-remote existence check is real, against a branch that genuinely
    does not exist on this origin."""
    repo = _init_bare_origin_and_clone(tmp_path)
    worktree = pipeline.ensure_pipeline_worktree(repo)
    real_run = pipeline.subprocess.run
    gh_calls = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == "gh":
            gh_calls.append(cmd)
            return _FakeCompleted(returncode=0, stdout="[]", stderr="")
        return real_run(cmd, **kwargs)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    pipeline.close_stale_pr("claude/pin-pipeline-llama-b105", worktree)

    assert gh_calls == [["gh", "pr", "list", "--head", "claude/pin-pipeline-llama-b105",
                         "--state", "open", "--json", "number"]]
    assert "claude/pin-pipeline-llama-b105" not in subprocess.run(
        ["git", "ls-remote", "--heads", "origin"], cwd=worktree,
        capture_output=True, text=True).stdout, "sanity: no branch was created out of nothing"


def test_close_stale_pr_closes_the_pr_and_deletes_the_real_remote_branch(tmp_path, monkeypatch):
    """The real bug this exists to fix, reproduced for real: a prior run
    pushed claude/pin-pipeline-llama-b105 and then crashed before merging.
    close_stale_pr must delete that real remote branch so the NEXT push for
    the same candidate is a clean fast-forward, not a rejection."""
    repo = _init_bare_origin_and_clone(tmp_path)
    branch = "claude/pin-pipeline-llama-b105"
    worktree = pipeline.ensure_pipeline_worktree(repo)
    # Simulate the prior (crashed) run: push a real branch to the real bare origin.
    subprocess.run(["git", "checkout", "-q", "-b", branch], cwd=worktree, check=True)
    (worktree / "stale-attempt.md").write_text("stale\n", encoding="utf-8")
    subprocess.run(["git", "add", "stale-attempt.md"], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "stale attempt"], cwd=worktree, check=True)
    subprocess.run(["git", "push", "-q", "-u", "origin", branch], cwd=worktree, check=True)
    remote_before = subprocess.run(["git", "ls-remote", "--heads", "origin", branch],
                                   cwd=worktree, capture_output=True, text=True).stdout
    assert branch in remote_before, "sanity: the stale branch really is on the remote"

    real_run = pipeline.subprocess.run
    gh_close_calls = []

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["gh", "pr", "list"]:
            return _FakeCompleted(returncode=0, stdout='[{"number": 42}]', stderr="")
        if cmd[:3] == ["gh", "pr", "close"]:
            gh_close_calls.append(cmd)
            return _FakeCompleted(returncode=0, stdout="", stderr="")
        return real_run(cmd, **kwargs)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    pipeline.close_stale_pr(branch, worktree)

    assert gh_close_calls == [["gh", "pr", "close", "42"]]
    remote_after = subprocess.run(["git", "ls-remote", "--heads", "origin", branch],
                                  cwd=worktree, capture_output=True, text=True).stdout
    assert branch not in remote_after, "the stale remote branch must actually be gone"

    # And the payoff: prepare_bump_branch for the SAME candidate now pushes cleanly
    # (gh pr list is faked empty this time, since close_stale_pr already ran above;
    # every git call in fake_run above and here falls through to the real subprocess.run).
    def fake_run_second_attempt(cmd, **kwargs):
        if cmd[0] == "gh":
            return _FakeCompleted(returncode=0, stdout="[]", stderr="")
        return real_run(cmd, **kwargs)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run_second_attempt)
    new_branch = pipeline.prepare_bump_branch(worktree, "b105")
    assert new_branch == branch
    (worktree / "fresh-attempt.md").write_text("fresh\n", encoding="utf-8")
    subprocess.run(["git", "add", "fresh-attempt.md"], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fresh attempt"], cwd=worktree, check=True)
    push = subprocess.run(["git", "push", "-u", "origin", branch],
                          cwd=worktree, capture_output=True, text=True)
    assert push.returncode == 0, f"the retried push must succeed cleanly: {push.stderr}"


def test_close_stale_pr_raises_infra_error_when_gh_pr_list_fails(tmp_path, monkeypatch):
    repo = _init_bare_origin_and_clone(tmp_path)
    worktree = pipeline.ensure_pipeline_worktree(repo)
    monkeypatch.setattr(pipeline.subprocess, "run",
                        lambda cmd, **k: _FakeCompleted(returncode=1, stdout="", stderr="boom"))
    with pytest.raises(pipeline.InfraError, match="gh pr list"):
        pipeline.close_stale_pr("claude/pin-pipeline-llama-b105", worktree)


def test_commit_and_push_stages_every_tracked_modification_not_just_the_bump_files(tmp_path):
    """run_bump() always touches setup_llama.py/_api.py/CHANGELOG.md, but a
    bump can also require a manual follow-on fix elsewhere in the tree (a
    safety-relevant constant plus its own tests - see the b11118 cuda-13
    13.3->13.4 toolkit rename). commit_and_push must stage that too, or the
    fix silently never reaches the commit: the PR would pass locally (the
    working tree has it) and fail on CI (the commit does not)."""
    repo = _init_bare_origin_and_clone(tmp_path)
    worktree = pipeline.ensure_pipeline_worktree(repo)
    branch = "claude/pin-pipeline-llama-b999"
    subprocess.run(["git", "checkout", "-q", "-b", branch], cwd=worktree, check=True)
    # CHANGELOG.md is not part of the minimal _init_bare_origin_and_clone fixture;
    # in the real repo it always already exists and is tracked, so seed that here
    # too - run_bump() only ever MODIFIES it, never creates it fresh.
    (worktree / "CHANGELOG.md").write_text("## [Unreleased]\n", encoding="utf-8")
    subprocess.run(["git", "add", "CHANGELOG.md"], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed CHANGELOG.md"], cwd=worktree, check=True)

    # README.md is a real TRACKED file (committed by _init_bare_origin_and_clone)
    # that is nowhere on the old hardcoded list - the exact shape of "a follow-on
    # fix outside setup_llama.py/_api.py/CHANGELOG.md".
    (worktree / "README.md").write_text("scratch\nfollow-on fix\n", encoding="utf-8")
    (worktree / "CHANGELOG.md").write_text("## [Unreleased]\n- bumped\n", encoding="utf-8")

    pipeline.commit_and_push(worktree, branch, "b999", "b998")

    committed = subprocess.run(["git", "show", "--stat", "--format=", "HEAD"],
                               cwd=worktree, capture_output=True, text=True).stdout
    assert "README.md" in committed, (
        "a follow-on fix outside the bump's own 3 files must not be silently "
        f"dropped from the commit:\n{committed}")
    assert "CHANGELOG.md" in committed
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           cwd=worktree, capture_output=True, text=True).stdout
    assert dirty == "", f"every tracked modification must be committed, nothing left dirty: {dirty!r}"


def _patch_state_dir(monkeypatch, tmp_path):
    """Also no-ops sync_main_checkout: every run_llama_pipeline orchestration
    test calls this, and none of them are testing the main-checkout sync
    itself (that has its own dedicated real-git tests below)."""
    state_dir = tmp_path / "state"
    monkeypatch.setattr(pipeline, "STATE_DIR", state_dir)
    monkeypatch.setattr(pipeline, "sync_main_checkout", lambda repo=None: None)
    return state_dir


class _CallSpy:
    """A stand-in for a step that must not run. Raising from a stub used
    inside pipeline's try/except PipelineError block would be a risky
    assertion (see diff-review-discipline item 13: an AssertionError raised
    inside code under test can be swallowed by a broad except) - this
    instead records calls and is asserted on from OUTSIDE the call."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return None


def _patch_every_write_path_step_as_spy(monkeypatch) -> dict:
    """Spy EVERY function run_llama_pipeline can call past the confirm gate,
    not only the first one. A fires-control run that deliberately disables a
    gate proceeds past whichever functions are mocked - one real function
    left in that chain (verified live: prepare_bump_branch, unmocked, ran a
    real `git checkout -b` against this session's own worktree with cwd
    defaulting to the process cwd, moving it off its branch) executes for
    real. Every "must not proceed" test uses this instead of spying one
    function at a time."""
    names = ("ensure_pipeline_worktree", "prepare_bump_branch", "run_bump",
             "run_targeted_tests", "commit_and_push", "open_pr", "wait_for_ci", "merge_pr")
    spies = {}
    for name in names:
        spy = _CallSpy()
        monkeypatch.setattr(pipeline, name, spy)
        spies[name] = spy
    return spies


# --------------------------------------------------------------------------- #
#  run_llama_pipeline - the full orchestration, mocked at each named seam.   #
#  Fires-control for this section: see the manual break/restore recorded in  #
#  the PR description - reverting the `if rc == 1: return 1` gate made      #
#  test_run_llama_pipeline_fail_receipt_stops_before_any_write fail with     #
#  ensure_pipeline_worktree called once, confirming the assertion is live.  #
# --------------------------------------------------------------------------- #

def test_run_llama_pipeline_fail_receipt_stops_before_any_write(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 1)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)
    spies = _patch_every_write_path_step_as_spy(monkeypatch)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 1
    assert not any(s.calls for s in spies.values()), (
        "a FAIL receipt must never reach the worktree/bump/commit stage")
    assert issue_spy.calls, "a genuine FAIL must be logged to issues.txt"


def test_run_llama_pipeline_inconclusive_receipt_stops_before_any_write(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 2)
    spies = _patch_every_write_path_step_as_spy(monkeypatch)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 2
    assert not any(s.calls for s in spies.values()), (
        "an INCONCLUSIVE receipt must never reach the worktree/bump/commit stage")


def test_run_llama_pipeline_unexpected_confirm_exit_code_is_fail_not_silent_pass(monkeypatch, tmp_path):
    """run_confirm's contract is 0/1/2/LEASE_BUSY_EXIT; falling through to
    PASS on ANY other value (e.g. a native crash producing an OS exit code)
    would silently proceed to bump/commit/merge on an unconfirmed build."""
    _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 3)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)
    spies = _patch_every_write_path_step_as_spy(monkeypatch)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 1
    assert not any(s.calls for s in spies.values()), (
        "an unexpected exit code must never reach the worktree/bump/commit stage")
    assert issue_spy.calls


def test_run_llama_pipeline_sync_failure_stops_before_any_candidate_is_read(monkeypatch, tmp_path):
    """sync_main_checkout runs before newest_candidate() even looks at the
    pin - if it fails, nothing candidate-specific can be recorded, and
    newest_candidate itself must never run."""
    _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "sync_main_checkout",
                        lambda: (_ for _ in ()).throw(pipeline.PipelineError("not on master")))
    candidate_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "newest_candidate", candidate_spy)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 1
    assert not candidate_spy.calls


def test_run_llama_pipeline_sync_infra_failure_is_inconclusive(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "sync_main_checkout",
                        lambda: (_ for _ in ()).throw(pipeline.InfraError("fetch failed")))
    candidate_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "newest_candidate", candidate_spy)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 2
    assert not candidate_spy.calls


def test_run_llama_pipeline_lease_busy_records_no_verdict(monkeypatch, tmp_path):
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm",
                        lambda candidate, receipt_path: pipeline.LEASE_BUSY_EXIT)
    spies = _patch_every_write_path_step_as_spy(monkeypatch)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 2
    assert not any(s.calls for s in spies.values()), (
        "a lease-busy run must never reach the worktree/bump/commit stage either")
    assert not (state_dir / "llama-state.json").exists(), (
        "a lease-busy run never measured the candidate and must not record a verdict for it")


def test_run_llama_pipeline_dry_run_stops_before_worktree(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 0)
    spies = _patch_every_write_path_step_as_spy(monkeypatch)

    rc = pipeline.run_llama_pipeline(dry_run=True)

    assert rc == 0
    assert not any(s.calls for s in spies.values()), "--dry-run must never create the pipeline worktree"


def test_run_llama_pipeline_skips_a_recorded_fail_without_calling_confirm(monkeypatch, tmp_path):
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    state_dir.mkdir(parents=True)
    (state_dir / "llama-state.json").write_text(
        json.dumps({"last_tag_tried": "b105", "verdict": "FAIL", "timestamp": _iso(_day(0))}),
        encoding="utf-8")
    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    confirm_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "run_confirm", confirm_spy)
    spies = _patch_every_write_path_step_as_spy(monkeypatch)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 0
    assert not confirm_spy.calls, "a recorded FAIL for the same candidate must never re-run confirm"
    assert not any(s.calls for s in spies.values())


def _fake_worktree_with_changelog(tmp_path) -> Path:
    worktree = tmp_path / "fake-worktree"
    worktree.mkdir()
    (worktree / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n### Added\n- x\n\n## [0.1.0] - 2026-01-01\nold\n", encoding="utf-8")
    return worktree


def test_run_llama_pipeline_full_pass_path_merges_on_green_ci(monkeypatch, tmp_path):
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)

    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 0)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pipeline, "prepare_bump_branch",
                        lambda wt, candidate: "claude/pin-pipeline-llama-b105")
    monkeypatch.setattr(pipeline, "run_bump",
                        lambda wt, candidate, receipt_path, write: (0, "ok"))
    monkeypatch.setattr(pipeline, "run_targeted_tests", lambda wt: (True, "ok"))
    commit_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "commit_and_push", commit_spy)
    monkeypatch.setattr(pipeline, "open_pr", lambda *a: 4242)
    monkeypatch.setattr(pipeline, "wait_for_ci", lambda wt: "GREEN")
    merge_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "merge_pr", merge_spy)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 0
    assert commit_spy.calls, "a PASS must commit the bump"
    assert merge_spy.calls == [((4242, worktree, "claude/pin-pipeline-llama-b105", "b105", "b100"), {})]
    changed = (worktree / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "moved from b100 to b105" in changed
    state = json.loads((state_dir / "llama-state.json").read_text(encoding="utf-8"))
    assert state["verdict"] == "PASS"
    assert state["merged_pr"] == 4242


def test_run_llama_pipeline_red_ci_leaves_pr_open_never_merges(monkeypatch, tmp_path):
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)

    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 0)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pipeline, "prepare_bump_branch", lambda wt, candidate: "branch")
    monkeypatch.setattr(pipeline, "run_bump",
                        lambda wt, candidate, receipt_path, write: (0, "ok"))
    monkeypatch.setattr(pipeline, "run_targeted_tests", lambda wt: (True, "ok"))
    monkeypatch.setattr(pipeline, "commit_and_push", _CallSpy())
    monkeypatch.setattr(pipeline, "open_pr", lambda *a: 55)
    monkeypatch.setattr(pipeline, "wait_for_ci", lambda wt: "RED")
    merge_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "merge_pr", merge_spy)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 1
    assert not merge_spy.calls, "must never merge on a red CI result"
    assert issue_spy.calls
    state = json.loads((state_dir / "llama-state.json").read_text(encoding="utf-8"))
    assert state["verdict"] == "FAIL"
    assert state["open_pr"] == 55


def test_run_llama_pipeline_pending_ci_is_inconclusive_never_a_permanent_fail(monkeypatch, tmp_path):
    """A CI wait that times out with checks still running is not evidence
    the build is bad - it must retry after a cooldown like any other
    INCONCLUSIVE, and (unlike a genuine FAIL) never gets logged to
    issues.txt, since there is nothing wrong to report yet."""
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)

    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 0)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pipeline, "prepare_bump_branch", lambda wt, candidate: "branch")
    monkeypatch.setattr(pipeline, "run_bump",
                        lambda wt, candidate, receipt_path, write: (0, "ok"))
    monkeypatch.setattr(pipeline, "run_targeted_tests", lambda wt: (True, "ok"))
    monkeypatch.setattr(pipeline, "commit_and_push", _CallSpy())
    monkeypatch.setattr(pipeline, "open_pr", lambda *a: 77)
    monkeypatch.setattr(pipeline, "wait_for_ci", lambda wt: "PENDING")
    merge_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "merge_pr", merge_spy)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 2
    assert not merge_spy.calls, "must never merge while CI is still pending"
    assert not issue_spy.calls, "a pending CI wait is not a FAIL and must not be logged as one"
    state = json.loads((state_dir / "llama-state.json").read_text(encoding="utf-8"))
    assert state["verdict"] == "INCONCLUSIVE"
    assert state["open_pr"] == 77


def test_run_llama_pipeline_bump_refusal_stops_before_commit(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)

    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 0)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pipeline, "prepare_bump_branch", lambda wt, candidate: "branch")
    monkeypatch.setattr(pipeline, "run_bump",
                        lambda wt, candidate, receipt_path, write: (1, "refused: measured set mismatch"))
    commit_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "commit_and_push", commit_spy)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 1
    assert not commit_spy.calls, "a refused bump must never be committed"
    assert issue_spy.calls


def test_run_llama_pipeline_unexpected_exception_is_inconclusive_and_logged(monkeypatch, tmp_path):
    """Found by a real live run: a raw FileNotFoundError (gh not resolvable
    via this process's PATH) escaped prepare_bump_branch() uncaught, crashed
    the whole script, and recorded no state at all - so a retry would have
    blindly redone the entire download+confirm instead of recognizing this
    was a tooling problem, not evidence about the (already-confirmed-good)
    candidate. Unlike an ordinary InfraError-driven INCONCLUSIVE, this one
    must ALSO be logged: an uncaught exception is always worth a human's
    attention, whatever a retry eventually does."""
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)

    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 0)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)

    def _raise_unexpected(wt, candidate):
        raise FileNotFoundError("[WinError 2] The system cannot find the file specified")
    monkeypatch.setattr(pipeline, "prepare_bump_branch", _raise_unexpected)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 2
    assert issue_spy.calls, "an uncaught exception must still be logged, unlike an ordinary INCONCLUSIVE"
    state = json.loads((state_dir / "llama-state.json").read_text(encoding="utf-8"))
    assert state["verdict"] == "INCONCLUSIVE"
    assert "FileNotFoundError" in state["reason"]
    skip_reason = pipeline.should_skip(state, "b105")
    assert skip_reason is not None and "cooldown" in skip_reason, (
        "must retry after a cooldown, never be permanently blocked")


def test_run_llama_pipeline_infra_error_during_commit_is_inconclusive_not_fail(monkeypatch, tmp_path):
    """A push/gh-CLI hiccup is about THIS RUN's plumbing, not the confirmed
    candidate build - it must retry after cooldown (INCONCLUSIVE), never
    permanently block an otherwise-good candidate the way a FAIL would."""
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)

    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 0)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pipeline, "prepare_bump_branch", lambda wt, candidate: "branch")
    monkeypatch.setattr(pipeline, "run_bump",
                        lambda wt, candidate, receipt_path, write: (0, "ok"))
    monkeypatch.setattr(pipeline, "run_targeted_tests", lambda wt: (True, "ok"))

    def _raise_infra(*a, **k):
        raise pipeline.InfraError("push failed: simulated network blip")
    monkeypatch.setattr(pipeline, "commit_and_push", _raise_infra)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 2
    assert not issue_spy.calls, "an infra hiccup on a good build must not be logged as a FAIL"
    state = json.loads((state_dir / "llama-state.json").read_text(encoding="utf-8"))
    assert state["verdict"] == "INCONCLUSIVE"
    # should_skip must treat this as cooldown-retriable, never a permanent block
    skip_reason = pipeline.should_skip(state, "b105")
    assert skip_reason is not None and "cooldown" in skip_reason


def test_run_llama_pipeline_failing_tests_after_bump_stop_before_commit(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)

    monkeypatch.setattr(pipeline, "newest_candidate", lambda: ("b100", "b105"))
    monkeypatch.setattr(pipeline, "run_confirm", lambda candidate, receipt_path: 0)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pipeline, "prepare_bump_branch", lambda wt, candidate: "branch")
    monkeypatch.setattr(pipeline, "run_bump",
                        lambda wt, candidate, receipt_path, write: (0, "ok"))
    monkeypatch.setattr(pipeline, "run_targeted_tests", lambda wt: (False, "2 failed"))
    commit_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "commit_and_push", commit_spy)
    monkeypatch.setattr(pipeline, "append_fail_issue", _CallSpy())

    rc = pipeline.run_llama_pipeline(dry_run=False)

    assert rc == 1
    assert not commit_spy.calls, "a bump that fails its own targeted tests must never be committed"


def test_prepare_bump_branch_is_idempotent_across_a_retried_candidate(tmp_path, monkeypatch):
    """FIRES: a second call for the SAME candidate (e.g. a prior attempt
    crashed after branching but before pushing) must not fail trying to
    create a branch that already exists. Stale-PR cleanup has its own
    dedicated tests below; this one is about the local branch mechanics."""
    repo = _init_scratch_repo(tmp_path)
    subprocess.run(["git", "remote", "add", "origin", str(repo)], cwd=repo, check=True)
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=repo, check=True)
    worktree = pipeline.ensure_pipeline_worktree(repo)
    monkeypatch.setattr(pipeline, "close_stale_pr", lambda branch, worktree: None)

    pipeline.prepare_bump_branch(worktree, "b10999")
    branch_again = pipeline.prepare_bump_branch(worktree, "b10999")
    assert branch_again == "claude/pin-pipeline-llama-b10999"


def test_merge_pr_detaches_and_deletes_local_branch_without_delete_branch_flag(tmp_path, monkeypatch):
    """merge_pr must never pass --delete-branch to `gh pr merge` - that
    flag's local cleanup tries to switch to the default branch, which fails
    from a worktree since master lives in the main checkout - and must
    detach off and delete the local branch itself afterward. `gh` is
    stubbed (no real GitHub call); the git half is real."""
    repo = _init_scratch_repo(tmp_path)
    subprocess.run(["git", "remote", "add", "origin", str(repo)], cwd=repo, check=True)
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=repo, check=True)
    worktree = pipeline.ensure_pipeline_worktree(repo)
    monkeypatch.setattr(pipeline, "close_stale_pr", lambda branch, worktree: None)
    branch = pipeline.prepare_bump_branch(worktree, "b105")

    real_run = pipeline.subprocess.run
    gh_calls = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == "gh":
            gh_calls.append(cmd)
            return _FakeCompleted(returncode=0, stdout="", stderr="")
        return real_run(cmd, **kwargs)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    pipeline.merge_pr(99, worktree, branch, "b105", "b100")

    assert gh_calls and "--delete-branch" not in gh_calls[0]
    current = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                             cwd=worktree, capture_output=True, text=True).stdout.strip()
    assert current == "HEAD", "must have detached off the merged branch"
    branches = subprocess.run(["git", "branch", "--list", branch],
                              cwd=worktree, capture_output=True, text=True).stdout
    assert branch not in branches, "the local branch must be deleted after merge"


# =============================================================================
#  ComfyUI pin - the parallel pipeline. Mirrors the llama sections above     #
#  where the shape matches; new tests where it does not (the 3-part          #
#  candidate, the 3-phase confirm, the cross-pin lock).                      #
# =============================================================================

def test_newest_comfyui_candidate_none_when_already_current(monkeypatch):
    check_comfyui_pin = pipeline._load_module(
        Path(pipeline.REPO) / "scripts" / "check_comfyui_pin.py", "check_comfyui_pin_for_test")
    pin = check_comfyui_pin._pinned_version()
    monkeypatch.setattr(pipeline, "_load_module", lambda path, name: check_comfyui_pin)
    monkeypatch.setattr(check_comfyui_pin, "_fetch_releases",
                        lambda: [{"tag_name": pin, "prerelease": False, "draft": False}])
    assert pipeline.newest_comfyui_candidate() is None


def test_newest_comfyui_candidate_returns_triple_when_upstream_is_ahead_and_resolves(monkeypatch):
    check_comfyui_pin = pipeline._load_module(
        Path(pipeline.REPO) / "scripts" / "check_comfyui_pin.py", "check_comfyui_pin_for_test2")
    pin = check_comfyui_pin._pinned_version()
    newest = "v99.0.0"
    commit = "a" * 40
    monkeypatch.setattr(pipeline, "_load_module", lambda path, name: check_comfyui_pin)
    monkeypatch.setattr(check_comfyui_pin, "_fetch_releases",
                        lambda: [{"tag_name": newest, "prerelease": False, "draft": False},
                                 {"tag_name": pin, "prerelease": False, "draft": False}])
    monkeypatch.setattr(check_comfyui_pin, "resolve_tag_commit", lambda tag: commit)
    assert pipeline.newest_comfyui_candidate() == (pin, newest, commit)


def test_newest_comfyui_candidate_none_when_commit_does_not_resolve(monkeypatch):
    """The candidate exists upstream but its commit could not be resolved (a
    transient GitHub API failure, say) - this is nothing to do THIS run,
    never a FAIL, and never a candidate silently passed downstream with no
    commit to actually confirm against."""
    check_comfyui_pin = pipeline._load_module(
        Path(pipeline.REPO) / "scripts" / "check_comfyui_pin.py", "check_comfyui_pin_for_test3")
    pin = check_comfyui_pin._pinned_version()
    monkeypatch.setattr(pipeline, "_load_module", lambda path, name: check_comfyui_pin)
    monkeypatch.setattr(check_comfyui_pin, "_fetch_releases",
                        lambda: [{"tag_name": "v99.0.0", "prerelease": False, "draft": False},
                                 {"tag_name": pin, "prerelease": False, "draft": False}])
    monkeypatch.setattr(check_comfyui_pin, "resolve_tag_commit", lambda tag: None)
    assert pipeline.newest_comfyui_candidate() is None


def test_newest_comfyui_candidate_none_on_upstream_error(monkeypatch):
    check_comfyui_pin = pipeline._load_module(
        Path(pipeline.REPO) / "scripts" / "check_comfyui_pin.py", "check_comfyui_pin_for_test4")
    monkeypatch.setattr(pipeline, "_load_module", lambda path, name: check_comfyui_pin)
    monkeypatch.setattr(check_comfyui_pin, "_fetch_releases", lambda: None)
    assert pipeline.newest_comfyui_candidate() is None


def test_newest_comfyui_candidate_raises_when_the_constant_is_unreadable(monkeypatch):
    check_comfyui_pin = pipeline._load_module(
        Path(pipeline.REPO) / "scripts" / "check_comfyui_pin.py", "check_comfyui_pin_for_test5")

    def _boom():
        raise SystemExit("constant renamed")
    monkeypatch.setattr(pipeline, "_load_module", lambda path, name: check_comfyui_pin)
    monkeypatch.setattr(check_comfyui_pin, "_pinned_version", _boom)
    with pytest.raises(pipeline.PipelineError, match="could not read the ComfyUI pin"):
        pipeline.newest_comfyui_candidate()


# --------------------------------------------------------------------------- #
#  comfyui_changelog_bullet                                                   #
# --------------------------------------------------------------------------- #

def test_comfyui_changelog_bullet_names_both_tags():
    bullet = pipeline.comfyui_changelog_bullet("v0.31.1", "v0.32.0")
    assert "v0.31.1" in bullet and "v0.32.0" in bullet
    assert "localm comfy update" in bullet
    assert "--reinstall-requirements" not in bullet


def test_comfyui_changelog_bullet_names_reinstall_requirements_when_changed():
    bullet = pipeline.comfyui_changelog_bullet("v0.31.1", "v0.32.0", requirements_changed=True)
    assert "--reinstall-requirements" in bullet


# --------------------------------------------------------------------------- #
#  run_comfyui_bump / run_comfyui_targeted_tests - the worktree-scoped        #
#  analogs of run_bump/run_targeted_tests                                     #
# --------------------------------------------------------------------------- #

def test_run_comfyui_bump_uses_the_worktrees_own_copy(monkeypatch, tmp_path):
    worktree = tmp_path / "wt"
    (worktree / "scripts").mkdir(parents=True)
    calls = []

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: (calls.append((cmd, k)), _Proc())[1])

    rc, out = pipeline.run_comfyui_bump(worktree, "v0.32.0", "a" * 40,
                                        tmp_path / "r.json", write=True)
    assert rc == 0
    cmd, kwargs = calls[0]
    assert cmd[1] == str(worktree / "scripts" / "bump_comfyui_pin.py")
    assert "--tag" in cmd and "v0.32.0" in cmd
    assert "--commit" in cmd and "a" * 40 in cmd
    assert "--write" in cmd
    assert kwargs["cwd"] == worktree
    assert kwargs["env"]["PYTHONPATH"] == str(worktree)


def test_run_comfyui_bump_without_write_omits_the_flag(monkeypatch, tmp_path):
    worktree = tmp_path / "wt"
    calls = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: (calls.append(cmd), _Proc())[1])
    pipeline.run_comfyui_bump(worktree, "v0.32.0", "a" * 40, tmp_path / "r.json", write=False)
    assert "--write" not in calls[0]


def test_run_comfyui_targeted_tests_runs_against_the_worktree_with_pythonpath_set(
        monkeypatch, tmp_path):
    worktree = tmp_path / "wt"
    calls = []

    class _Proc:
        returncode = 0
        stdout = "9 passed"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: (calls.append((cmd, k)), _Proc())[1])
    passed, out = pipeline.run_comfyui_targeted_tests(worktree)
    assert passed is True
    cmd, kwargs = calls[0]
    assert "tests/test_check_comfyui_pin.py" in cmd
    assert "tests/test_bump_comfyui_pin.py" in cmd
    assert "tests/test_confirm_comfyui_runtime.py" in cmd
    assert kwargs["env"]["PYTHONPATH"] == str(worktree)


# --------------------------------------------------------------------------- #
#  run_comfyui_confirm - the 3-phase subprocess orchestration + GPU lease     #
#  scoping (provision/teardown unlocked, smoke locked)                       #
# --------------------------------------------------------------------------- #

def test_run_comfyui_confirm_provision_and_teardown_never_hold_the_lease(monkeypatch, tmp_path):
    """The load-bearing property from the module docstring of
    confirm_comfyui_runtime.py: an install can run well past the lease TTL,
    so only smoke (the phase that actually touches the GPU) may go through
    run_under_gpu_lease."""
    phases_run = []

    def fake_subprocess_run(cmd, **kwargs):
        phase = cmd[cmd.index("--phase") + 1]
        phases_run.append(("subprocess.run", phase))
        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Proc()

    def fake_lease(cmd, *, purpose, cwd):
        phase = cmd[cmd.index("--phase") + 1]
        phases_run.append(("run_under_gpu_lease", phase))
        return 0

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(pipeline, "run_under_gpu_lease", fake_lease)
    monkeypatch.setattr(pipeline, "_comfyui_scratch_workdir", lambda: tmp_path / "scratch")

    rc = pipeline.run_comfyui_confirm("v0.32.0", "a" * 40, tmp_path / "r.json")

    assert rc == 0
    assert phases_run[0] == ("subprocess.run", "teardown"), "pre-run cleanup, unlocked"
    assert ("subprocess.run", "provision") in phases_run, "provision must never be under the lease"
    assert ("run_under_gpu_lease", "smoke") in phases_run, "smoke MUST be under the lease"
    assert phases_run[-1] == ("subprocess.run", "teardown"), "post-run cleanup, unlocked"
    assert phases_run.count(("subprocess.run", "teardown")) == 2, "pre AND post, always"


def test_run_comfyui_confirm_still_tears_down_after_a_provision_failure(monkeypatch, tmp_path):
    calls = []

    def fake_subprocess_run(cmd, **kwargs):
        phase = cmd[cmd.index("--phase") + 1]
        calls.append(phase)
        class _Proc:
            returncode = 1 if phase == "provision" else 0
            stdout = "provision failed" if phase == "provision" else ""
            stderr = ""
        return _Proc()
    lease_spy = _CallSpy()
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(pipeline, "run_under_gpu_lease", lease_spy)
    monkeypatch.setattr(pipeline, "_comfyui_scratch_workdir", lambda: tmp_path / "scratch")

    rc = pipeline.run_comfyui_confirm("v0.32.0", "a" * 40, tmp_path / "r.json")

    assert rc == 1
    assert not lease_spy.calls, "smoke must never run after a provision failure"
    assert calls.count("teardown") == 2, "still torn down: pre-run AND after the failure"


def test_run_comfyui_confirm_still_tears_down_after_a_smoke_crash(monkeypatch, tmp_path):
    """Teardown is the finally-equivalent: even if the lease-wrapped smoke
    call itself raises, the scratch install must not be left behind."""
    calls = []

    def fake_subprocess_run(cmd, **kwargs):
        phase = cmd[cmd.index("--phase") + 1]
        calls.append(("subprocess.run", phase))
        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Proc()

    def fake_lease(cmd, *, purpose, cwd):
        raise RuntimeError("simulated: the lease subprocess itself blew up")
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(pipeline, "run_under_gpu_lease", fake_lease)
    monkeypatch.setattr(pipeline, "_comfyui_scratch_workdir", lambda: tmp_path / "scratch")

    with pytest.raises(RuntimeError):
        pipeline.run_comfyui_confirm("v0.32.0", "a" * 40, tmp_path / "r.json")

    teardowns = [p for kind, p in calls if p == "teardown"]
    assert len(teardowns) == 2, "pre-run AND post-crash teardown must both still have run"


def test_run_comfyui_confirm_returns_lease_busy_without_a_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: type(
        "P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(pipeline, "run_under_gpu_lease",
                        lambda cmd, *, purpose, cwd: pipeline.LEASE_BUSY_EXIT)
    monkeypatch.setattr(pipeline, "_comfyui_scratch_workdir", lambda: tmp_path / "scratch")
    rc = pipeline.run_comfyui_confirm("v0.32.0", "a" * 40, tmp_path / "r.json")
    assert rc == pipeline.LEASE_BUSY_EXIT


# --------------------------------------------------------------------------- #
#  pipeline_lock - the cross-pin serialization                                #
# --------------------------------------------------------------------------- #

def test_pipeline_lock_acquires_and_releases_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path)
    lock = pipeline._pipeline_lock_path()
    with pipeline.pipeline_lock():
        assert lock.exists()
    assert not lock.exists()


def test_pipeline_lock_refuses_when_held_by_a_live_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path)
    lock = pipeline._pipeline_lock_path()
    lock.mkdir(parents=True)
    import os
    (lock / pipeline._LOCK_OWNER_FILE).write_text(json.dumps({"pid": os.getpid()}))
    with pytest.raises(pipeline.PipelineLockBusy, match="already holds"):
        with pipeline.pipeline_lock():
            pass
    assert lock.exists(), "the lock of a genuinely live holder must not be touched"


def test_pipeline_lock_reclaims_a_stale_lock_from_a_dead_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path)
    lock = pipeline._pipeline_lock_path()
    lock.mkdir(parents=True)
    # A pid that (almost certainly) does not exist on any real machine.
    (lock / pipeline._LOCK_OWNER_FILE).write_text(json.dumps({"pid": 999999999}))
    with pipeline.pipeline_lock():
        assert lock.exists()
    assert not lock.exists()


def test_pipeline_lock_refuses_when_owner_file_is_unreadable(monkeypatch, tmp_path):
    """An unreadable owner must never be treated as reclaimable - that would
    let a lock with a corrupted owner file be silently stolen."""
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path)
    lock = pipeline._pipeline_lock_path()
    lock.mkdir(parents=True)
    with pytest.raises(pipeline.PipelineLockBusy, match="owner unreadable"):
        with pipeline.pipeline_lock():
            pass


# --------------------------------------------------------------------------- #
#  run_comfyui_pipeline - mirrors the run_llama_pipeline orchestration tests  #
# --------------------------------------------------------------------------- #

def _patch_comfyui_write_path_as_spy(monkeypatch) -> dict:
    names = ("ensure_pipeline_worktree", "prepare_bump_branch", "run_comfyui_bump",
             "run_comfyui_targeted_tests", "commit_and_push", "open_pr", "wait_for_ci",
             "merge_pr")
    spies = {}
    for name in names:
        spy = _CallSpy()
        monkeypatch.setattr(pipeline, name, spy)
        spies[name] = spy
    return spies


def test_run_comfyui_pipeline_fail_receipt_stops_before_any_write(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "newest_comfyui_candidate",
                        lambda: ("v0.31.1", "v0.32.0", "a" * 40))
    monkeypatch.setattr(pipeline, "run_comfyui_confirm", lambda tag, commit, receipt_path: 1)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)
    spies = _patch_comfyui_write_path_as_spy(monkeypatch)

    rc = pipeline.run_comfyui_pipeline(dry_run=False)

    assert rc == 1
    assert not any(s.calls for s in spies.values())
    assert issue_spy.calls
    assert issue_spy.calls[0][1].get("pin") == "comfyui"


def test_run_comfyui_pipeline_inconclusive_receipt_stops_before_any_write(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "newest_comfyui_candidate",
                        lambda: ("v0.31.1", "v0.32.0", "a" * 40))
    monkeypatch.setattr(pipeline, "run_comfyui_confirm", lambda tag, commit, receipt_path: 2)
    spies = _patch_comfyui_write_path_as_spy(monkeypatch)

    rc = pipeline.run_comfyui_pipeline(dry_run=False)

    assert rc == 2
    assert not any(s.calls for s in spies.values())


def test_run_comfyui_pipeline_lease_busy_records_no_verdict(monkeypatch, tmp_path):
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "newest_comfyui_candidate",
                        lambda: ("v0.31.1", "v0.32.0", "a" * 40))
    monkeypatch.setattr(pipeline, "run_comfyui_confirm",
                        lambda tag, commit, receipt_path: pipeline.LEASE_BUSY_EXIT)
    spies = _patch_comfyui_write_path_as_spy(monkeypatch)

    rc = pipeline.run_comfyui_pipeline(dry_run=False)

    assert rc == 2
    assert not any(s.calls for s in spies.values())
    assert not (state_dir / "comfyui-state.json").exists()


def test_run_comfyui_pipeline_dry_run_stops_before_worktree(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "newest_comfyui_candidate",
                        lambda: ("v0.31.1", "v0.32.0", "a" * 40))
    monkeypatch.setattr(pipeline, "run_comfyui_confirm", lambda tag, commit, receipt_path: 0)
    spies = _patch_comfyui_write_path_as_spy(monkeypatch)

    rc = pipeline.run_comfyui_pipeline(dry_run=True)

    assert rc == 0
    assert not any(s.calls for s in spies.values())


def test_run_comfyui_pipeline_skips_a_recorded_fail_without_calling_confirm(monkeypatch, tmp_path):
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    state_dir.mkdir(parents=True)
    (state_dir / "comfyui-state.json").write_text(
        json.dumps({"last_tag_tried": "v0.32.0", "verdict": "FAIL", "timestamp": _iso(_day(0))}),
        encoding="utf-8")
    monkeypatch.setattr(pipeline, "newest_comfyui_candidate",
                        lambda: ("v0.31.1", "v0.32.0", "a" * 40))
    confirm_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "run_comfyui_confirm", confirm_spy)
    spies = _patch_comfyui_write_path_as_spy(monkeypatch)

    rc = pipeline.run_comfyui_pipeline(dry_run=False)

    assert rc == 0
    assert not confirm_spy.calls
    assert not any(s.calls for s in spies.values())


def test_run_comfyui_pipeline_full_pass_path_merges_on_green_ci(monkeypatch, tmp_path):
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)
    receipt_path_holder = {}

    def fake_confirm(tag, commit, receipt_path):
        receipt_path_holder["path"] = receipt_path
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({
            "baseline": {"requirements_changed": True}}), encoding="utf-8")
        return 0

    monkeypatch.setattr(pipeline, "newest_comfyui_candidate",
                        lambda: ("v0.31.1", "v0.32.0", "a" * 40))
    monkeypatch.setattr(pipeline, "run_comfyui_confirm", fake_confirm)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pipeline, "prepare_bump_branch",
                        lambda wt, candidate, pin: "claude/pin-pipeline-comfyui-v0.32.0")
    monkeypatch.setattr(pipeline, "run_comfyui_bump",
                        lambda wt, tag, commit, receipt_path, write: (0, "ok"))
    monkeypatch.setattr(pipeline, "run_comfyui_targeted_tests", lambda wt: (True, "ok"))
    commit_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "commit_and_push", commit_spy)
    monkeypatch.setattr(pipeline, "open_pr", lambda *a, **k: 5150)
    monkeypatch.setattr(pipeline, "wait_for_ci", lambda wt: "GREEN")
    merge_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "merge_pr", merge_spy)

    rc = pipeline.run_comfyui_pipeline(dry_run=False)

    assert rc == 0
    assert commit_spy.calls, "a PASS must commit the bump"
    assert merge_spy.calls[0][0][:4] == (5150, worktree, "claude/pin-pipeline-comfyui-v0.32.0",
                                         "v0.32.0")
    changed = (worktree / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "v0.31.1" in changed and "v0.32.0" in changed
    assert "--reinstall-requirements" in changed, (
        "the receipt said requirements changed - the bullet must say so too")
    state = json.loads((state_dir / "comfyui-state.json").read_text(encoding="utf-8"))
    assert state["verdict"] == "PASS"
    assert state["merged_pr"] == 5150


def test_run_comfyui_pipeline_red_ci_leaves_pr_open_never_merges(monkeypatch, tmp_path):
    state_dir = _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)

    def fake_confirm(tag, commit, receipt_path):
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({"baseline": {}}), encoding="utf-8")
        return 0
    monkeypatch.setattr(pipeline, "newest_comfyui_candidate",
                        lambda: ("v0.31.1", "v0.32.0", "a" * 40))
    monkeypatch.setattr(pipeline, "run_comfyui_confirm", fake_confirm)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pipeline, "prepare_bump_branch", lambda wt, candidate, pin: "b")
    monkeypatch.setattr(pipeline, "run_comfyui_bump",
                        lambda wt, tag, commit, receipt_path, write: (0, "ok"))
    monkeypatch.setattr(pipeline, "run_comfyui_targeted_tests", lambda wt: (True, "ok"))
    monkeypatch.setattr(pipeline, "commit_and_push", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "open_pr", lambda *a, **k: 1)
    monkeypatch.setattr(pipeline, "wait_for_ci", lambda wt: "RED")
    merge_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "merge_pr", merge_spy)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)

    rc = pipeline.run_comfyui_pipeline(dry_run=False)

    assert rc == 1
    assert not merge_spy.calls
    assert issue_spy.calls
    state = json.loads((state_dir / "comfyui-state.json").read_text(encoding="utf-8"))
    assert state["verdict"] == "FAIL"


def test_run_comfyui_pipeline_bump_refusal_stops_before_commit(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    worktree = _fake_worktree_with_changelog(tmp_path)

    def fake_confirm(tag, commit, receipt_path):
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({"baseline": {}}), encoding="utf-8")
        return 0
    monkeypatch.setattr(pipeline, "newest_comfyui_candidate",
                        lambda: ("v0.31.1", "v0.32.0", "a" * 40))
    monkeypatch.setattr(pipeline, "run_comfyui_confirm", fake_confirm)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", lambda: worktree)
    monkeypatch.setattr(pipeline, "prepare_bump_branch", lambda wt, candidate, pin: "b")
    monkeypatch.setattr(pipeline, "run_comfyui_bump",
                        lambda wt, tag, commit, receipt_path, write: (1, "REFUSED: bad receipt"))
    commit_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "commit_and_push", commit_spy)

    rc = pipeline.run_comfyui_pipeline(dry_run=False)

    assert rc == 1
    assert not commit_spy.calls


def test_run_comfyui_pipeline_unexpected_exception_is_inconclusive_and_logged(monkeypatch, tmp_path):
    _patch_state_dir(monkeypatch, tmp_path)
    worktree_spy_calls = []

    def _boom():
        worktree_spy_calls.append(True)
        raise FileNotFoundError("simulated: gh not on PATH")

    def fake_confirm(tag, commit, receipt_path):
        # A real receipt must exist before ensure_pipeline_worktree() runs -
        # the PASS path reads it first (requirements_changed) - otherwise a
        # missing-file read would raise before the mocked worktree step is
        # ever reached, and this test would pass for the wrong reason.
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({"baseline": {}}), encoding="utf-8")
        return 0
    monkeypatch.setattr(pipeline, "newest_comfyui_candidate",
                        lambda: ("v0.31.1", "v0.32.0", "a" * 40))
    monkeypatch.setattr(pipeline, "run_comfyui_confirm", fake_confirm)
    monkeypatch.setattr(pipeline, "ensure_pipeline_worktree", _boom)
    issue_spy = _CallSpy()
    monkeypatch.setattr(pipeline, "append_fail_issue", issue_spy)

    rc = pipeline.run_comfyui_pipeline(dry_run=False)

    assert rc == 2
    assert worktree_spy_calls, "ensure_pipeline_worktree must actually have been reached"
    assert issue_spy.calls, "an uncaught exception is always logged, even though INCONCLUSIVE"


# --------------------------------------------------------------------------- #
#  main() dispatch                                                            #
# --------------------------------------------------------------------------- #

def test_main_dispatches_comfyui_under_the_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path)
    spy = _CallSpy()
    monkeypatch.setattr(pipeline, "run_comfyui_pipeline", lambda dry_run: (spy(dry_run), 0)[1])
    rc = pipeline.main(["--pin", "comfyui"])
    assert rc == 0
    assert spy.calls == [((False,), {})]


def test_main_rocm_still_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path)
    assert pipeline.main(["--pin", "rocm"]) == 1


def test_main_reports_inconclusive_when_the_lock_is_held(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path)
    lock = pipeline._pipeline_lock_path()
    lock.mkdir(parents=True)
    import os
    (lock / pipeline._LOCK_OWNER_FILE).write_text(json.dumps({"pid": os.getpid()}))
    rc = pipeline.main(["--pin", "llama"])
    assert rc == 2
    assert "INCONCLUSIVE" in capsys.readouterr().out
