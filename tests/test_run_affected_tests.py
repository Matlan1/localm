# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/run_affected_tests.py, the per-PR Python gate, and the ci.yml job
that runs it.

The wrapper's contract is that pytest's exit status becomes the job's, that
it never runs the whole suite, and that every way the selection can be wrong
is refused rather than read as green: a selector exit 3 is retried at depth 0;
one still wide there runs in full with the selector's limit lifted, and fails
only when the selection is every test file; a depth-0 retry that is empty or
failed fails, a selector crash fails, an empty selection fails, a path that is
not an existing tests/**/test_*.py fails, and only the nothing-affected
sentinel on its own skips pytest. The sharded gate's plan, shards and verdict
are pinned next: the shard count, the shard's pytest command, and a verdict
that fails on a missing or failed shard, a test run twice or outside the plan,
and a plan job that did not succeed. The last section binds the wrapper to the
real tree and pins the shape of the ci.yml jobs so the gate cannot be quietly
narrowed, label-gated or made non-blocking.
"""

import fnmatch
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "run_affected_tests.py"
_SELECTOR = REPO_ROOT / "scripts" / "affected_tests.py"
_CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load(path=_SCRIPT, name="run_affected_tests"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Registered before execution: a dataclass with string annotations looks
    # its own module up in sys.modules while the class is being built.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rat():
    return _load()


@pytest.fixture
def repo(tmp_path):
    """A throwaway tree with two real test files and a non-test file."""
    (tmp_path / "tests" / "sub").mkdir(parents=True)
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    (tmp_path / "tests" / "sub" / "test_b.py").write_text("def test_b():\n    assert True\n")
    (tmp_path / "tests" / "helper.py").write_text("X = 1\n")
    (tmp_path / "localm").mkdir()
    (tmp_path / "localm" / "test_not_a_test_dir.py").write_text("")
    return tmp_path


def _why(*lines):
    return "".join(f"{ln}\n" for ln in lines)


# --- parse_selection: the selector's exit status and output ------------------

def test_sentinels_match_the_selector(rat):
    """The strings the wrapper keys on are the selector's own."""
    sel = _load(_SELECTOR, "affected_tests_for_wrapper")
    assert rat.NOTHING_AFFECTED == sel._NOTHING_AFFECTED
    assert rat.WIDE_EXIT == sel._WIDE_EXIT


def test_a_selection_with_reasons_is_parsed_and_each_path_verified(rat, repo):
    out = _why("tests/test_a.py  # imports localm.a; names localm.a",
               "tests/sub/test_b.py  # changed")
    s = rat.parse_selection(0, out, "2 of 3 test files affected by 1 changed file(s)", repo)
    assert s.mode == "selected"
    assert s.paths == ["tests/test_a.py", "tests/sub/test_b.py"]
    assert s.reasons["tests/test_a.py"] == "imports localm.a; names localm.a"
    assert s.reasons["tests/sub/test_b.py"] == "changed"
    assert s.detail == "2 of 3 test files affected by 1 changed file(s)"


def test_crlf_output_is_parsed(rat, repo):
    s = rat.parse_selection(0, "tests/test_a.py  # changed\r\n", "", repo)
    assert s.mode == "selected" and s.paths == ["tests/test_a.py"]


def test_the_nothing_sentinel_alone_means_no_tests(rat, repo):
    s = rat.parse_selection(0, "tests/NO_TEST_FILE_IS_AFFECTED\n", "0 of 3 ...", repo)
    assert s.mode == "nothing" and s.paths == []


def test_the_nothing_sentinel_next_to_a_path_is_refused(rat, repo):
    s = rat.parse_selection(0, _why("tests/test_a.py  # changed", "tests/NO_TEST_FILE_IS_AFFECTED"),
                            "", repo)
    assert s.mode == "failed"
    assert "NO_TEST_FILE_IS_AFFECTED" in s.detail


def test_exit_zero_with_no_output_is_refused(rat, repo):
    s = rat.parse_selection(0, "", "", repo)
    assert s.mode == "failed" and "no selection" in s.detail
    s = rat.parse_selection(0, "\n  \n", "", repo)
    assert s.mode == "failed"


@pytest.mark.parametrize("line", [
    "tests/test_missing.py  # changed",             # does not exist
    "tests/helper.py  # changed",                   # exists, not a test file
    "localm/test_not_a_test_dir.py  # changed",     # exists, outside tests/
    "tests/../localm/test_not_a_test_dir.py  # x",  # traversal
    "tests/./test_a.py  # x",                       # dot segment
    "/tests/test_a.py  # x",                        # absolute
    "tests\\test_a.py  # x",                        # backslash spelling
    "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR",
    "tests/AFFECTED_TESTS_FAILED_SEE_STDERR",
    "-k something  # x",
])
def test_a_line_that_is_not_an_existing_test_file_is_refused(rat, repo, line):
    s = rat.parse_selection(0, _why("tests/test_a.py  # changed", line), "", repo)
    assert s.mode == "failed", line
    assert "not an existing tests/**/test_*.py" in s.detail
    assert s.paths == []


def test_exit_three_is_wide_whatever_was_printed(rat, repo):
    s = rat.parse_selection(3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
                            "800 of 800 ...\nWIDE: 100% of the suite exceeds --max-share 25%", repo)
    assert s.mode == "wide" and s.paths == [] and s.exit_status == 3
    assert "WIDE" in s.detail


@pytest.mark.parametrize("status", [1, 2, 4, -1, 127])
def test_any_other_exit_status_is_a_failure_even_with_a_plausible_list(rat, repo, status):
    s = rat.parse_selection(status, "tests/test_a.py  # changed\n", "Traceback ...", repo)
    assert s.mode == "failed" and s.paths == []
    assert f"selector exited {status}" in s.detail and "Traceback" in s.detail


# --- pytest_args: what runs per mode -----------------------------------------

def test_selected_runs_exactly_the_selection_with_the_unit_marker(rat):
    s = rat.Selection("selected", paths=["tests/test_a.py", "tests/sub/test_b.py"])
    assert rat.pytest_args(s, ["-x"]) == [
        "tests/test_a.py", "tests/sub/test_b.py", "-m", "not integration", "-n", "auto", "-x"]


def test_wide_nothing_and_failed_run_no_pytest_and_never_the_whole_suite(rat):
    for mode in ("wide", "nothing", "failed"):
        assert rat.pytest_args(rat.Selection(mode), []) is None
    assert "tests" not in rat.pytest_args(rat.Selection("selected", paths=["tests/test_a.py"]), [])


def test_a_selection_without_paths_runs_no_pytest(rat):
    """An argument list with no test path is a whole-suite run under
    testpaths; the argv builder refuses it whatever the mode says."""
    assert rat.pytest_args(rat.Selection("selected", paths=[]), []) is None
    assert rat.pytest_args(rat.Selection("selected", paths=[]), ["-x"]) is None


# --- select: the depth-0 retry of a wide selection ----------------------------

def _selector_by_depth(monkeypatch, rat, by_depth):
    """Replace the selector with canned answers keyed by depth, or by
    (depth, max_share) for a call that lifts the limit. Returns the list of
    (depth, max_share) calls made."""
    calls = []

    def fake(depth, base, files, max_share=None):
        calls.append((depth, max_share))
        return by_depth.get((depth, max_share), by_depth.get(depth))

    monkeypatch.setattr(rat, "_run_selector", fake)
    monkeypatch.setattr(rat, "_base_resolves", lambda base: True)
    return calls


def test_a_wide_depth_one_selection_is_retried_at_depth_zero_and_that_one_runs(
        rat, repo, monkeypatch):
    calls = _selector_by_depth(monkeypatch, rat, {
        1: CompletedProcess([], 3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
                            "700 of 800 test files affected by 1 changed file(s)\nWIDE: 88% ..."),
        0: CompletedProcess([], 0, "tests/test_a.py  # imports localm.a\n",
                            "1 of 800 test files affected by 1 changed file(s)")})
    monkeypatch.setattr(rat, "REPO", repo)
    s = rat.select(1, "origin/master", None)
    assert calls == [(1, None), (0, None)]
    assert s.mode == "selected" and s.depth == 0 and s.paths == ["tests/test_a.py"]
    assert s.over_limit is False
    assert s.wider is not None and s.wider.mode == "wide" and s.wider.depth == 1
    text = rat.render_summary(s, rat.pytest_args(s, []))
    assert "selected at `--depth 0`" in text
    assert "at `--depth 1` was wider" in text and "WIDE: 88%" in text
    assert "1 of 800 test files" in text


_WIDE = CompletedProcess([], 3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
                         "WIDE: 88%")


@pytest.mark.parametrize("retry", [
    CompletedProcess([], 0, "tests/NO_TEST_FILE_IS_AFFECTED\n", "0 of 800 ..."),
    CompletedProcess([], 1, "tests/AFFECTED_TESTS_FAILED_SEE_STDERR\n", "Traceback"),
    CompletedProcess([], 0, "", ""),
], ids=["nothing", "crash", "empty"])
def test_a_depth_zero_retry_that_selects_nothing_keeps_the_wide_result(
        rat, repo, monkeypatch, retry):
    """A hub-module change whose depth-0 selection is not a list of test files
    is wide, never nothing: the job must fail rather than run no test, and the
    selector is not asked again with its limit lifted."""
    calls = _selector_by_depth(monkeypatch, rat, {1: _WIDE, 0: retry})
    monkeypatch.setattr(rat, "REPO", repo)
    s = rat.select(1, "origin/master", None)
    assert calls == [(1, None), (0, None)]
    assert s.mode == "wide" and s.depth == 1
    assert "WIDE: 88%" in s.detail and "at --depth 0:" in s.detail
    assert rat.pytest_args(s, []) is None
    headline = rat.render_summary(s, None).splitlines()[2]
    assert headline.startswith("**Selection wider than the selector's limit** (exit 3) at `--depth 1`")
    expected = (", and nothing at `--depth 0`" if retry.stdout.strip() and retry.returncode == 0
                else ", and the selector failed at `--depth 0`")
    assert expected in headline, headline
    assert "job fails" in headline and "needs the full suite" in headline


_LIFTED_PATHS = ("tests/test_a.py  # imports localm.config\n"
                 "tests/sub/test_b.py  # names localm.config\n")


def test_a_selection_still_wide_at_depth_zero_runs_in_full_with_the_limit_lifted(
        rat, repo, monkeypatch):
    calls = _selector_by_depth(monkeypatch, rat, {
        1: _WIDE, 0: _WIDE,
        (0, rat.WHOLE_SUITE_SHARE): CompletedProcess(
            [], 0, _LIFTED_PATHS, "478 of 924 test files affected by 16 changed file(s)")})
    monkeypatch.setattr(rat, "REPO", repo)
    s = rat.select(1, "origin/master", None)
    assert calls == [(1, None), (0, None), (0, rat.WHOLE_SUITE_SHARE)]
    assert s.mode == "selected" and s.depth == 0 and s.over_limit is True
    assert s.paths == ["tests/test_a.py", "tests/sub/test_b.py"]
    assert s.wider is not None and s.wider.mode == "wide"
    args = rat.pytest_args(s, [])
    assert args[:2] == s.paths and "-m" in args
    text = rat.render_summary(s, args)
    assert "**2 test file(s)** selected at `--depth 0`" in text
    assert "This change is wide" in text and "pytest runs all of it" in text
    assert "478 of 924 test files" in text
    assert "at --depth 1: wide" in text and "WIDE: 88%" in text


def test_a_wide_selection_at_depth_zero_is_selected_again_with_the_limit_lifted(
        rat, repo, monkeypatch):
    calls = _selector_by_depth(monkeypatch, rat, {
        0: _WIDE,
        (0, rat.WHOLE_SUITE_SHARE): CompletedProcess([], 0, _LIFTED_PATHS, "2 of 3 ...")})
    monkeypatch.setattr(rat, "REPO", repo)
    s = rat.select(0, "origin/master", None)
    assert calls == [(0, None), (0, rat.WHOLE_SUITE_SHARE)]
    assert s.mode == "selected" and s.over_limit is True and s.depth == 0


@pytest.mark.parametrize("lifted", [
    CompletedProcess([], 3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
                     "WIDE: 100%"),
    CompletedProcess([], 1, "tests/AFFECTED_TESTS_FAILED_SEE_STDERR\n", "Traceback"),
    CompletedProcess([], 0, "", ""),
], ids=["every-test-file", "crash", "empty"])
def test_a_selection_that_is_every_test_file_or_fails_when_lifted_stays_wide(
        rat, repo, monkeypatch, lifted):
    _selector_by_depth(monkeypatch, rat, {1: _WIDE, 0: _WIDE,
                                          (0, rat.WHOLE_SUITE_SHARE): lifted})
    monkeypatch.setattr(rat, "REPO", repo)
    s = rat.select(1, "origin/master", None)
    assert s.mode == "wide" and s.retry == "whole" and s.over_limit is False
    assert rat.pytest_args(s, []) is None
    headline = rat.render_summary(s, None).splitlines()[2]
    assert "even with the limit lifted to every test file but the whole suite" in headline
    assert "job fails" in headline and "needs the full suite" in headline
    assert "up to every test file but the whole suite:" in s.detail


def test_the_lifted_limit_is_passed_to_the_selector_as_max_share(rat, monkeypatch):
    seen = []
    monkeypatch.setattr(rat.subprocess, "run", lambda cmd, **kw: seen.append(cmd)
                        or CompletedProcess(cmd, 0, "", ""))
    rat._run_selector(0, "origin/master", None)
    rat._run_selector(0, "origin/master", ["a.py"], rat.WHOLE_SUITE_SHARE)
    assert "--max-share" not in seen[0]
    assert seen[1][seen[1].index("--max-share") + 1] == str(rat.WHOLE_SUITE_SHARE)
    assert seen[1][-2:] == ["--files", "a.py"]


def test_only_a_selection_of_every_test_file_exceeds_the_lifted_limit(rat):
    """n of n files is share 1.0; n-1 of n stays under the lifted limit for any
    suite under a hundred thousand files."""
    assert 1.0 > rat.WHOLE_SUITE_SHARE
    assert all((n - 1) / n <= rat.WHOLE_SUITE_SHARE for n in (2, 3, 924, 20_000, 100_000))


def test_the_real_selector_honours_the_lifted_limit(rat):
    """A hub change exceeds a tiny limit but not the lifted one; a change to
    tests/conftest.py is every test file and exceeds both."""
    hub = ["localm/config.py"]
    assert rat._run_selector(0, "HEAD", hub, 0.001).returncode == rat.WIDE_EXIT
    lifted = rat._run_selector(0, "HEAD", hub, rat.WHOLE_SUITE_SHARE)
    assert lifted.returncode == 0, lifted.stderr
    assert rat.parse_selection(0, lifted.stdout, lifted.stderr, depth=0).mode == "selected"
    every = rat._run_selector(0, "HEAD", ["tests/conftest.py"], rat.WHOLE_SUITE_SHARE)
    assert every.returncode == rat.WIDE_EXIT, every.stderr


# --- the base ref: an unresolvable one must never read as nothing affected ----

def test_the_base_ref_check_answers_for_the_real_checkout(rat):
    assert rat._base_resolves("HEAD") is True
    assert rat._base_resolves("refs/nonexistent/branch") is False
    assert rat._base_resolves("") is False


def test_an_unresolvable_base_is_a_failed_selection_and_the_selector_never_runs(
        rat, repo, monkeypatch):
    """The selector diffs HEAD against the merge base with --base; with no such
    ref it diffs HEAD against HEAD, selects nothing and exits 0. The wrapper
    refuses before that can happen."""
    calls = []
    monkeypatch.setattr(rat, "_run_selector",
                        lambda depth, base, files, max_share=None: calls.append(depth))
    monkeypatch.setattr(rat, "_base_resolves", lambda base: False)
    monkeypatch.setattr(rat, "REPO", repo)
    s = rat.select(1, "origin/master", None)
    assert s.mode == "failed" and calls == []
    assert "'origin/master' does not resolve" in s.detail
    assert rat.pytest_args(s, []) is None


def test_an_explicit_file_list_needs_no_base(rat, repo, monkeypatch):
    calls = _selector_by_depth(monkeypatch, rat, {
        1: CompletedProcess([], 0, "tests/test_a.py  # names a.py\n", "1 of 3 ...")})
    monkeypatch.setattr(rat, "_base_resolves", lambda base: False)
    monkeypatch.setattr(rat, "REPO", repo)
    s = rat.select(1, "refs/nonexistent/branch", ["a.py"])
    assert calls == [(1, None)] and s.mode == "selected"


def test_main_fails_on_an_unresolvable_base_without_running_pytest(rat, repo, monkeypatch):
    """Even when the selector would answer nothing-affected (which is what it
    does answer for a base it cannot resolve)."""
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo, _cp(0, "tests/NO_TEST_FILE_IS_AFFECTED\n", "0 of 3 ..."),
        argv=["--base", "refs/nonexistent/branch"], base_resolves=False)
    assert status == 1
    run_pytest.assert_not_called()
    assert "could not be computed" in text and "does not resolve" in text
    assert "No test file is affected" not in text


# --- main: exit status and the summary ----------------------------------------

def _drive(rat, monkeypatch, tmp_path, selector, pytest_status=0, argv=(), base_resolves=True):
    """Run main() with the selector (a CompletedProcess for every call, or a
    function of the selector's arguments), the base-ref check and pytest
    replaced; returns (exit status, the pytest mock, the step summary text)."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(rat, "REPO", tmp_path)
    monkeypatch.setattr(rat, "_run_selector", selector if callable(selector)
                        else lambda depth, base, files, max_share=None: selector)
    monkeypatch.setattr(rat, "_base_resolves", lambda base: base_resolves)
    run_pytest = MagicMock(return_value=pytest_status)
    monkeypatch.setattr(rat, "_run_pytest", run_pytest)
    status = rat.main(list(argv))
    text = summary.read_text(encoding="utf-8") if summary.exists() else ""
    return status, run_pytest, text


def _cp(status, stdout, stderr=""):
    return CompletedProcess(args=[], returncode=status, stdout=stdout, stderr=stderr)


def test_a_failing_affected_test_fails_the_job(rat, repo, monkeypatch):
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo, _cp(0, "tests/test_a.py  # changed\n", "1 of 3 ..."),
        pytest_status=1)
    assert status == 1
    run_pytest.assert_called_once_with(["tests/test_a.py", "-m", "not integration", "-n", "auto"])
    assert "tests/test_a.py  # changed" in text


def test_a_passing_selection_passes_and_publishes_the_selection(rat, repo, monkeypatch):
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo,
        _cp(0, "tests/test_a.py  # imports localm.a\ntests/sub/test_b.py  # changed\n",
            "2 of 3 test files affected by 2 changed file(s)"))
    assert status == 0
    run_pytest.assert_called_once_with(
        ["tests/test_a.py", "tests/sub/test_b.py", "-m", "not integration", "-n", "auto"])
    assert "**2 test file(s)** selected at `--depth 1`" in text
    assert "tests/test_a.py  # imports localm.a" in text
    assert "tests/sub/test_b.py  # changed" in text
    assert "2 of 3 test files affected by 2 changed file(s)" in text
    assert "`pytest tests/test_a.py tests/sub/test_b.py -m 'not integration' -n auto`" in text


def test_nothing_affected_runs_no_pytest_and_passes(rat, repo, monkeypatch):
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo, _cp(0, "tests/NO_TEST_FILE_IS_AFFECTED\n", "0 of 3 ..."))
    assert status == 0
    run_pytest.assert_not_called()
    assert "No test file is affected" in text and "pytest was not run" in text


def test_a_selection_of_every_test_file_fails_without_running_pytest(rat, repo, monkeypatch):
    """The same wide result at depth 1, at the depth-0 retry and with the limit
    lifted (every test file): the job fails, nothing runs, and the summary says
    the change needs the full suite."""
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo,
        _cp(3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
            "800 of 800 test files affected by 3 changed file(s)\nWIDE: 100% ..."))
    assert status == 1
    run_pytest.assert_not_called()
    assert "wider than the selector's limit" in text and "at `--depth 0`" in text
    assert "even with the limit lifted" in text
    assert "job fails" in text and "needs the full suite" in text
    assert "never runs the whole suite" in text
    assert "WIDE: 100%" in text


def test_a_wide_change_runs_in_full_and_its_pytest_status_is_the_job_status(
        rat, repo, monkeypatch):
    """The hub-module case end to end through main(): wide at depths 1 and 0,
    selected once the limit is lifted, and pytest decides the job."""
    def selector(depth, base, files, max_share=None):
        if max_share is None:
            return _cp(3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
                       f"WIDE: 52% at depth {depth}")
        return _cp(0, _LIFTED_PATHS, "2 of 3 test files affected by 1 changed file(s)")

    for pytest_status in (0, 1):
        summary = repo / "summary.md"
        summary.unlink(missing_ok=True)
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        monkeypatch.setattr(rat, "REPO", repo)
        monkeypatch.setattr(rat, "_run_selector", selector)
        monkeypatch.setattr(rat, "_base_resolves", lambda base: True)
        run_pytest = MagicMock(return_value=pytest_status)
        monkeypatch.setattr(rat, "_run_pytest", run_pytest)
        assert rat.main([]) == pytest_status
        run_pytest.assert_called_once_with(
            ["tests/test_a.py", "tests/sub/test_b.py", "-m", "not integration", "-n", "auto"])
        text = summary.read_text(encoding="utf-8")
        assert "This change is wide" in text and "**2 test file(s)**" in text


def test_wide_is_never_treated_as_nothing_affected(rat, repo, monkeypatch):
    """The wide sentinel arrives with exit 3 AND a path pytest would refuse;
    neither half may read as a docs-only change, at either depth."""
    for stdout in ("tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
                   "tests/NO_TEST_FILE_IS_AFFECTED\n", ""):
        for depth in ("0", "1"):
            status, run_pytest, text = _drive(rat, monkeypatch, repo, _cp(3, stdout, "WIDE"),
                                              argv=["--depth", depth])
            assert status == 1, (stdout, depth)
            run_pytest.assert_not_called()
            assert "No test file is affected" not in text


def test_a_selector_crash_fails_without_running_pytest(rat, repo, monkeypatch):
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo,
        _cp(1, "tests/AFFECTED_TESTS_FAILED_SEE_STDERR\n", "Traceback (most recent call last)"))
    assert status == 1
    run_pytest.assert_not_called()
    assert "could not be computed" in text and "selector exited 1" in text and "Traceback" in text


def test_an_empty_selection_fails_without_running_pytest(rat, repo, monkeypatch):
    status, run_pytest, text = _drive(rat, monkeypatch, repo, _cp(0, "", ""))
    assert status == 1
    run_pytest.assert_not_called()
    assert "printed no selection" in text


def test_a_fabricated_path_fails_without_running_pytest(rat, repo, monkeypatch):
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo,
        _cp(0, "tests/test_a.py  # changed\ntests/test_ghost.py  # changed\n", "2 of 3 ..."))
    assert status == 1
    run_pytest.assert_not_called()
    assert "tests/test_ghost.py" in text and "not an existing" in text


def test_no_tests_collected_after_deselection_passes_and_says_so(rat, repo, monkeypatch):
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo, _cp(0, "tests/test_a.py  # changed\n", "1 of 3 ..."),
        pytest_status=5)
    assert status == 0
    run_pytest.assert_called_once()
    assert "collected no test" in text and "deselected" in text


@pytest.mark.parametrize("pytest_status", [2, 3, 4])
def test_every_other_pytest_status_is_the_job_status(rat, repo, monkeypatch, pytest_status):
    status, _, _ = _drive(rat, monkeypatch, repo, _cp(0, "tests/test_a.py  # changed\n"),
                          pytest_status=pytest_status)
    assert status == pytest_status


def test_dry_run_prints_the_command_and_runs_nothing(rat, repo, monkeypatch, capsys):
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo, _cp(0, "tests/test_a.py  # changed\n"), argv=["--dry-run"])
    assert status == 0
    run_pytest.assert_not_called()
    assert "`pytest tests/test_a.py -m 'not integration' -n auto`" in text
    assert "tests/test_a.py  # changed" in capsys.readouterr().out


def test_arguments_after_the_separator_reach_pytest(rat, repo, monkeypatch):
    _, run_pytest, _ = _drive(rat, monkeypatch, repo, _cp(0, "tests/test_a.py  # changed\n"),
                              argv=["--", "-x", "-k", "smoke"])
    run_pytest.assert_called_once_with(
        ["tests/test_a.py", "-m", "not integration", "-n", "auto", "-x", "-k", "smoke"])


def test_selector_flags_are_passed_through(rat, repo, monkeypatch):
    seen = {}

    def fake_selector(depth, base, files, max_share=None):
        seen.update(depth=depth, base=base, files=files)
        return _cp(0, "tests/NO_TEST_FILE_IS_AFFECTED\n")

    monkeypatch.setattr(rat, "REPO", repo)
    monkeypatch.setattr(rat, "_run_selector", fake_selector)
    monkeypatch.setattr(rat, "_base_resolves", lambda base: True)
    monkeypatch.setattr(rat, "_run_pytest", MagicMock(return_value=0))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert rat.main(["--depth", "2", "--base", "origin/dev", "--files", "a.py", "b.md"]) == 0
    assert seen == {"depth": 2, "base": "origin/dev", "files": ["a.py", "b.md"]}
    assert rat.main(["--depth", "0"]) == 0
    assert seen["depth"] == 0
    assert rat.main([]) == 0
    assert seen == {"depth": 1, "base": "origin/master", "files": None}


def test_without_a_summary_file_the_summary_still_prints(rat, repo, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(rat, "_run_selector",
                        lambda *a: _cp(0, "tests/NO_TEST_FILE_IS_AFFECTED\n", "0 of 3 ..."))
    monkeypatch.setattr(rat, "_run_pytest", MagicMock(return_value=0))
    assert rat.main([]) == 0
    assert "No test file is affected" in capsys.readouterr().out


# --- the sharded gate: the plan ------------------------------------------------

@pytest.mark.parametrize("files,count", [(1, 1), (75, 1), (76, 2), (150, 2), (151, 3),
                                         (300, 4), (301, 5), (376, 5), (2000, 5)])
def test_the_shard_count_grows_by_one_per_75_files_up_to_five(rat, files, count):
    selection = rat.Selection("selected", paths=[f"tests/test_{i}.py" for i in range(files)])
    assert rat.shard_count(selection) == count


@pytest.mark.parametrize("mode", ["nothing", "wide", "failed"])
def test_a_selection_that_runs_nothing_plans_one_shard(rat, mode):
    assert rat.shard_count(rat.Selection(mode)) == 1


def _many(repo, n):
    """*n* real test files in *repo*, as the selector's `--why` output."""
    lines = []
    for i in range(n):
        (repo / "tests" / f"test_m{i:04d}.py").write_text("def test_m():\n    pass\n")
        lines.append(f"tests/test_m{i:04d}.py  # imports localm.hub")
    return _why(*lines)


def _plan(rat, monkeypatch, repo, selector, argv=()):
    """main() with --plan-out and the selector replaced; returns (exit status,
    the pytest mock, the step summary, the plan file, the GITHUB_OUTPUT outputs)."""
    plan = repo / "plan" / "plan.json"
    out = repo / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    status, run_pytest, text = _drive(rat, monkeypatch, repo, selector,
                                      argv=["--plan-out", str(plan), *argv])
    outputs = dict(ln.split("=", 1) for ln in out.read_text(encoding="utf-8").splitlines())
    return status, run_pytest, text, json.loads(plan.read_text(encoding="utf-8")), outputs


def test_a_narrow_plan_runs_one_shard_and_no_pytest(rat, repo, monkeypatch):
    status, run_pytest, text, plan, outputs = _plan(
        rat, monkeypatch, repo,
        _cp(0, "tests/test_a.py  # imports localm.a\ntests/sub/test_b.py  # changed\n",
            "2 of 3 test files affected by 2 changed file(s)"))
    assert status == 0
    run_pytest.assert_not_called()
    assert outputs == {"run": "true", "count": "1", "shards": "[1]"}
    assert (plan["mode"], plan["depth"], plan["count"]) == ("selected", 1, 1)
    assert plan["paths"] == ["tests/test_a.py", "tests/sub/test_b.py"]
    assert "**2 test file(s)** selected at `--depth 1`" in text
    assert "tests/test_a.py  # imports localm.a" in text
    assert "Split into **1 shard(s)**" in text
    assert plan["summary"] == text


def test_a_wide_plan_runs_five_shards(rat, repo, monkeypatch):
    """The hub-module case: wide at depths 1 and 0, selected with the limit
    lifted; 400 test files plan five shards."""
    lifted = _many(repo, 400)

    def selector(depth, base, files, max_share=None):
        if max_share is None:
            return _cp(3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
                       f"WIDE: 52% at depth {depth}")
        return _cp(0, lifted, "400 of 800 test files affected by 1 changed file(s)")

    status, run_pytest, text, plan, outputs = _plan(rat, monkeypatch, repo, selector)
    assert status == 0
    run_pytest.assert_not_called()
    assert outputs == {"run": "true", "count": "5", "shards": "[1, 2, 3, 4, 5]"}
    assert (plan["mode"], plan["depth"], plan["count"], len(plan["paths"])) == ("selected", 0, 5, 400)
    assert "This change is wide" in text and "Split into **5 shard(s)**" in text


def test_a_selection_of_every_test_file_fails_the_plan_with_no_shard_to_run(rat, repo, monkeypatch):
    status, run_pytest, text, plan, outputs = _plan(
        rat, monkeypatch, repo,
        _cp(3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n", "WIDE: 100% ..."))
    assert status == 1
    run_pytest.assert_not_called()
    assert outputs == {"run": "false", "count": "1", "shards": "[1]"}
    assert (plan["mode"], plan["paths"]) == ("wide", [])
    assert "needs the full suite" in text and "Split into" not in text


def test_a_failed_selection_fails_the_plan_with_no_shard_to_run(rat, repo, monkeypatch):
    status, run_pytest, text, plan, outputs = _plan(
        rat, monkeypatch, repo, _cp(1, "tests/AFFECTED_TESTS_FAILED_SEE_STDERR\n", "Traceback"))
    assert status == 1
    run_pytest.assert_not_called()
    assert outputs["run"] == "false" and plan["mode"] == "failed"
    assert "could not be computed" in text


def test_nothing_affected_passes_the_plan_with_no_shard_to_run(rat, repo, monkeypatch):
    status, run_pytest, text, plan, outputs = _plan(
        rat, monkeypatch, repo, _cp(0, "tests/NO_TEST_FILE_IS_AFFECTED\n", "0 of 3 ..."))
    assert status == 0
    run_pytest.assert_not_called()
    assert outputs == {"run": "false", "count": "1", "shards": "[1]"}
    assert (plan["mode"], plan["paths"], plan["count"]) == ("nothing", [], 1)
    assert "No test file is affected" in text


# --- the sharded gate: one shard ---------------------------------------------

def _write_plan(path, mode, paths, count, depth=1):
    path.write_text(json.dumps({"mode": mode, "depth": depth, "paths": paths, "count": count,
                                "summary": "### Affected tests (python-pr-gate)\n\nTHE SELECTION\n"}),
                    encoding="utf-8")
    return path


def _shard(rat, monkeypatch, repo, plan, argv, pytest_status=0):
    """main() for one shard with pytest replaced; (exit status, the pytest mock, the summary)."""
    summary = repo / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(rat, "REPO", repo)
    run_pytest = MagicMock(return_value=pytest_status)
    monkeypatch.setattr(rat, "_run_pytest", run_pytest)
    status = rat.main(["--plan", str(plan), "--shard-out", str(repo / "shard-out"), *argv])
    return status, run_pytest, summary.read_text(encoding="utf-8") if summary.exists() else ""


def test_a_shard_runs_the_plan_selection_with_the_shard_plugin(rat, repo, monkeypatch):
    plan = _write_plan(repo / "plan.json", "selected", ["tests/test_a.py", "tests/sub/test_b.py"], 1)
    status, run_pytest, text = _shard(rat, monkeypatch, repo, plan, ["--shard", "1", "--", "-x"])
    assert status == 0
    run_pytest.assert_called_once_with(
        ["tests/test_a.py", "tests/sub/test_b.py", "-m", "not integration", "-n", "auto", "-x"],
        ["-p", "tests._shard", "--shard", "1/1", "--shard-ids-out",
         str((repo / "shard-out" / "ids-1.json").resolve())])
    assert "shard 1/1" in text and "-p tests._shard --shard 1/1" in text


def test_the_shard_number_can_come_from_the_environment(rat, repo, monkeypatch):
    plan = _write_plan(repo / "plan.json", "selected", ["tests/test_a.py"], 1)
    monkeypatch.setenv("PR_GATE_SHARD", "1")
    status, run_pytest, _ = _shard(rat, monkeypatch, repo, plan, ["--shard-env", "PR_GATE_SHARD"])
    assert status == 0
    assert run_pytest.call_args.args[1][:4] == ["-p", "tests._shard", "--shard", "1/1"]
    for bad in ("", "x", "-1"):
        monkeypatch.setenv("PR_GATE_SHARD", bad)
        with pytest.raises(SystemExit) as exc:
            _shard(rat, monkeypatch, repo, plan, ["--shard-env", "PR_GATE_SHARD"])
        assert exc.value.code == 2


@pytest.mark.parametrize("pytest_status,expected", [(0, 0), (5, 0), (1, 1), (2, 2)])
def test_a_shard_left_with_no_test_passes_and_any_other_status_is_the_shard_status(
        rat, repo, monkeypatch, pytest_status, expected):
    plan = _write_plan(repo / "plan.json", "selected", ["tests/test_a.py"], 1)
    status, run_pytest, text = _shard(rat, monkeypatch, repo, plan, ["--shard", "1"],
                                      pytest_status=pytest_status)
    assert status == expected
    run_pytest.assert_called_once()
    assert ("No test is left in this shard" in text) == (pytest_status == 5)


@pytest.mark.parametrize("mode,paths,count,index,why", [
    ("nothing", [], 1, 1, "selects no test file"),
    ("wide", [], 1, 1, "selects no test file"),
    ("selected", ["tests/test_a.py"], 1, 2, "outside the plan's 1..1"),
    ("selected", ["tests/test_a.py"], 1, 0, "outside the plan's 1..1"),
    ("selected", ["tests/test_a.py"], 5, 1, "its selection needs 1"),
    ("selected", ["tests/test_ghost.py"], 1, 1, "not an existing"),
    ("selected", [], 1, 1, "with no paths"),
    ("bogus", [], 1, 1, "unknown mode"),
])
def test_a_shard_of_a_plan_it_cannot_run_fails_without_pytest(
        rat, repo, monkeypatch, mode, paths, count, index, why):
    plan = _write_plan(repo / "plan.json", mode, paths, count)
    status, run_pytest, text = _shard(rat, monkeypatch, repo, plan, ["--shard", str(index)])
    assert status == 1
    run_pytest.assert_not_called()
    assert "No test ran" in text and why in text


def test_the_shard_plugin_stays_on_the_command_line_when_the_paths_go_to_an_argfile(
        rat, monkeypatch):
    """pytest loads -p plugins from the command line before it expands an
    @file, so the shard options never go into the file."""
    seen = []

    def fake_run(cmd, cwd=None):
        seen.append(list(cmd))
        if cmd[-1].startswith("@"):
            seen.append(Path(cmd[-1][1:]).read_text(encoding="utf-8").splitlines())
        return CompletedProcess(cmd, 0)

    monkeypatch.setattr(rat.subprocess, "run", fake_run)
    inline = ["-p", "tests._shard", "--shard", "2/5", "--shard-ids-out", "ids-2.json"]
    paths = [f"tests/test_{i:05d}.py" for i in range(rat.ARGS_INLINE_MAX // 10)]
    assert rat._run_pytest(paths, inline) == 0
    cmd, filed = seen
    assert cmd[1:3] == ["-m", "pytest"] and cmd[3:9] == inline
    assert len(cmd) == 10 and cmd[9].startswith("@")
    assert filed == paths
    seen.clear()
    assert rat._run_pytest(["tests/test_a.py"], inline) == 0
    assert seen == [[sys.executable, "-m", "pytest", *inline, "tests/test_a.py"]]


# --- the sharded gate: the verdict --------------------------------------------

_REAL_FILES = sorted(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "tests").glob("test_*.py"))


def _ids(index, count, everything, selected):
    from tests import _shard
    return json.dumps({"index": index, "count": count, "all_total": len(set(everything)),
                       "all_sha256": _shard.digest(everything), "selected": sorted(selected)})


def _collect(rat, monkeypatch, tmp_path, plan, files, plan_result="success"):
    """main() --collect over *files* (name: text) written to a shard-out
    directory, against *plan* (mode, paths, count); the real tests._shard
    verify runs. Returns (exit status, the step summary, stdout)."""
    plan_path = _write_plan(tmp_path / "plan.json", *plan)
    out = tmp_path / "shard-out"
    out.mkdir()
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    status = rat.main(["--collect", str(out), "--plan", str(plan_path), "--plan-result", plan_result])
    return status, summary.read_text(encoding="utf-8")


def _two_shards():
    """A plan of 76 real test files (two shards) and the node ids it collects."""
    paths = _REAL_FILES[:76]
    assert len(paths) == 76
    return ("selected", paths, 2), [f"{p}::test_x" for p in paths]


def test_the_verdict_passes_when_every_selected_test_ran_in_exactly_one_shard(
        rat, monkeypatch, tmp_path):
    plan, nodes = _two_shards()
    status, text = _collect(rat, monkeypatch, tmp_path, plan, {
        "ids-1.json": _ids(1, 2, nodes, nodes[::2]), "status-1.txt": "success\n",
        "ids-2.json": _ids(2, 2, nodes, nodes[1::2]), "status-2.txt": "success\n"})
    assert status == 0, text
    assert "THE SELECTION" in text, "the verdict repeats the plan's summary"
    assert "The gate passes" in text and "2 shards ran 76 tests, each exactly once" in text


def test_a_shard_that_selects_nothing_is_fine(rat, monkeypatch, tmp_path):
    plan, nodes = _two_shards()
    status, text = _collect(rat, monkeypatch, tmp_path, plan, {
        "ids-1.json": _ids(1, 2, nodes, nodes), "status-1.txt": "success\n",
        "ids-2.json": _ids(2, 2, nodes, []), "status-2.txt": "success\n"})
    assert status == 0, text


def test_the_verdict_fails_on_a_missing_shard(rat, monkeypatch, tmp_path):
    plan, nodes = _two_shards()
    status, text = _collect(rat, monkeypatch, tmp_path, plan, {
        "ids-1.json": _ids(1, 2, nodes, nodes[::2]), "status-1.txt": "success\n"})
    assert status == 1
    assert "The gate fails" in text and "shard 2 recorded no result" in text


def test_the_verdict_fails_on_a_failed_shard(rat, monkeypatch, tmp_path):
    plan, nodes = _two_shards()
    status, text = _collect(rat, monkeypatch, tmp_path, plan, {
        "ids-1.json": _ids(1, 2, nodes, nodes[::2]), "status-1.txt": "success\n",
        "ids-2.json": _ids(2, 2, nodes, nodes[1::2]), "status-2.txt": "failure\n"})
    assert status == 1
    assert "shard 2 ended 'failure'" in text


def test_the_verdict_fails_on_a_test_run_twice(rat, monkeypatch, tmp_path):
    plan, nodes = _two_shards()
    status, text = _collect(rat, monkeypatch, tmp_path, plan, {
        "ids-1.json": _ids(1, 2, nodes, nodes[::2]), "status-1.txt": "success\n",
        "ids-2.json": _ids(2, 2, nodes, [nodes[0], *nodes[1::2]]), "status-2.txt": "success\n"})
    assert status == 1
    assert f"{nodes[0]} ran in shards 1 and 2" in text


def test_the_verdict_fails_on_a_test_outside_the_plan(rat, monkeypatch, tmp_path):
    plan, nodes = _two_shards()
    stranger = f"{_REAL_FILES[100]}::test_y"
    everything = [*nodes, stranger]
    status, text = _collect(rat, monkeypatch, tmp_path, plan, {
        "ids-1.json": _ids(1, 2, everything, nodes[::2]), "status-1.txt": "success\n",
        "ids-2.json": _ids(2, 2, everything, [*nodes[1::2], stranger]), "status-2.txt": "success\n"})
    assert status == 1
    assert "outside the plan's selection" in text and _REAL_FILES[100] in text


@pytest.mark.parametrize("plan_result", ["failure", "skipped", "cancelled", ""])
def test_the_verdict_fails_when_the_plan_job_did_not_succeed(rat, monkeypatch, tmp_path, plan_result):
    verify = MagicMock(return_value=(0, "fine"))
    monkeypatch.setattr(rat, "_verify_shards", verify)
    status, text = _collect(rat, monkeypatch, tmp_path, ("wide", [], 1), {}, plan_result=plan_result)
    assert status == 1
    verify.assert_not_called()
    assert f"the plan job ended {plan_result!r}" in text


def test_the_verdict_fails_without_a_plan(rat, monkeypatch, tmp_path):
    out = tmp_path / "shard-out"
    out.mkdir()
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    assert rat.main(["--collect", str(out), "--plan", str(tmp_path / "absent.json"),
                     "--plan-result", "success"]) == 1
    assert "no plan at" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_nothing_affected_passes_the_verdict_only_with_no_shard_result(rat, monkeypatch, tmp_path):
    (tmp_path / "none").mkdir()
    (tmp_path / "stray").mkdir()
    status, text = _collect(rat, monkeypatch, tmp_path / "none", ("nothing", [], 1), {})
    assert status == 0, text
    assert "no test file is affected and no shard ran" in text
    status, text = _collect(rat, monkeypatch, tmp_path / "stray", ("nothing", [], 1),
                            {"status-1.txt": "success\n"})
    assert status == 1
    assert "selected no test file, yet 1 shard result file(s)" in text


def test_a_plan_job_that_succeeded_with_a_refused_plan_fails_the_verdict(rat, monkeypatch, tmp_path):
    status, text = _collect(rat, monkeypatch, tmp_path, ("wide", [], 1), {})
    assert status == 1
    assert "succeeded with a `wide` plan" in text


# --- the real tree and the real job -------------------------------------------

def test_a_change_to_the_wrapper_selects_this_file_in_the_real_tree():
    """The gate is bound to itself: editing the wrapper runs these tests."""
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--dry-run", "--files", "scripts/run_affected_tests.py"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "GITHUB_STEP_SUMMARY": ""})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tests/test_run_affected_tests.py  # names run_affected_tests.py" in proc.stdout
    cmd = [ln for ln in proc.stdout.splitlines() if ln.startswith("`pytest ")][-1]
    assert " tests/test_run_affected_tests.py " in cmd
    assert cmd.endswith("-m 'not integration' -n auto`")


def _load_workflow(path):
    wf = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML reads the bare key `on` as the boolean True.
    wf["on"] = wf.pop(True, wf.get("on"))
    return wf


def _norm(expr):
    return " ".join(str(expr).split())


_GATE_IF = _norm("""
    github.event_name == 'pull_request' &&
    !contains(github.event.pull_request.labels.*.name, 'full-ci')""")
_VERDICTS = (("python-pr-gate", "ubuntu-latest"), ("python-pr-gate-windows", "windows-latest"))


def _by_name(job, name):
    hits = [s for s in job["steps"] if s.get("name") == name]
    assert len(hits) == 1, f"{name!r} must appear exactly once in {job.get('name')}"
    return hits[0]


def test_the_plan_job_runs_the_checks_once_and_the_selection_on_every_unlabelled_pull_request():
    ci = _load_workflow(_CI)
    job = ci["jobs"]["python-pr-gate-plan"]
    assert job["runs-on"] == "ubuntu-latest"
    assert _norm(job["if"]) == _GATE_IF, (
        "every unlabelled PR, and never a labelled one, whose whole suite the matrix runs")
    assert isinstance(job.get("timeout-minutes"), int)
    assert "strategy" not in job, "the checks and the selection run once"

    checkout = job["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0, (
        "the selector diffs from the merge base with origin/master and the hygiene "
        "gate resolves the same baseline; a shallow checkout has no origin/master")
    assert checkout["with"]["persist-credentials"] is False

    runs = [s.get("run", "") for s in job["steps"]]
    wanted = ["uv lock --check", "python scripts/check_hygiene.py", "ruff check .",
              "python scripts/run_affected_tests.py --depth 1 --plan-out"]
    positions = []
    for cmd in wanted:
        hits = [i for i, r in enumerate(runs) if cmd in r]
        assert len(hits) == 1, f"{cmd!r} must appear exactly once, got {hits}"
        positions.append(hits[0])
    assert positions == sorted(positions), "lock, hygiene, ruff, then the selection"
    upload = _by_name(job, "Upload the plan")
    for step in job["steps"]:
        assert not step.get("continue-on-error"), f"{step.get('name')} must be able to fail the job"
        if step is not upload:
            assert step.get("if") is None, f"{step.get('name')} must run unconditionally"
    plan_step = job["steps"][positions[-1]]
    assert plan_step["id"] == "plan"
    assert job["outputs"] == {k: "${{ steps.plan.outputs.%s }}" % k for k in ("run", "count", "shards")}
    assert "--wide" not in plan_step["run"] and "pytest tests" not in plan_step["run"], (
        "the gate runs the selection only, never the whole suite")
    assert '"$RUNNER_TEMP/pr-gate-plan/plan.json"' in plan_step["run"]
    assert upload["if"] == "always()", "a refused plan still reaches the verdict jobs"
    assert upload["with"]["name"] == "pr-gate-plan"
    assert upload["with"]["path"] == "${{ runner.temp }}/pr-gate-plan/plan.json"


def test_every_shard_runs_its_slice_of_the_one_plan():
    ci = _load_workflow(_CI)
    job = ci["jobs"]["python-pr-gate-shard"]
    assert job["needs"] == ["python-pr-gate-plan"]
    assert _norm(job["if"]) == _norm(
        "${{ needs.python-pr-gate-plan.result == 'success' && "
        "needs.python-pr-gate-plan.outputs.run == 'true' }}")
    matrix = job["strategy"]["matrix"]
    assert matrix["shard"] == "${{ fromJSON(needs.python-pr-gate-plan.outputs.shards) }}"
    assert job["strategy"]["fail-fast"] is False
    assert job["timeout-minutes"] >= ci["jobs"]["test-shard"]["timeout-minutes"], (
        "a shard of a hub-module change runs as much as a matrix shard")
    checkout = job["steps"][0]
    assert checkout["with"]["fetch-depth"] == 0 and checkout["with"]["persist-credentials"] is False

    tests = _by_name(job, "Affected tests (this shard)")
    assert tests["env"] == {"PR_GATE_SHARD": "${{ matrix.shard }}"}
    assert _norm(tests["run"]) == (
        'python scripts/run_affected_tests.py --plan "${{ runner.temp }}/pr-gate-plan/plan.json" '
        "--shard-env PR_GATE_SHARD --shard-out shard-out")
    download = _by_name(job, "Download the plan")
    assert download["with"] == {"pattern": "pr-gate-plan", "merge-multiple": True,
                                "path": "${{ runner.temp }}/pr-gate-plan"}
    for step in job["steps"]:
        assert "--plan-out" not in step.get("run", "") and "--depth" not in step.get("run", ""), (
            "a shard never selects: it runs the plan's selection")
        assert not step.get("continue-on-error"), f"{step.get('name')} must be able to fail the job"

    record = _by_name(job, "Record this shard's result")
    assert record["if"] == "always()"
    assert record["env"] == {"JOB_STATUS": "${{ job.status }}", "SHARD": "${{ matrix.shard }}"}
    assert 'shard-out/status-$SHARD.txt' in record["run"]
    upload = _by_name(job, "Upload this shard's results")
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == "pr-gate-shard-${{ matrix.os }}-${{ matrix.shard }}"
    assert upload["with"]["path"].rstrip("/") == "shard-out"
    assert upload["with"]["overwrite"] is True, "re-running one failed shard must replace its artifact"


def test_the_gate_verdict_cannot_pass_over_a_missing_or_failed_shard():
    ci = _load_workflow(_CI)
    upload = _by_name(ci["jobs"]["python-pr-gate-shard"], "Upload this shard's results")
    oses = ci["jobs"]["python-pr-gate-shard"]["strategy"]["matrix"]["os"]
    assert sorted(oses) == sorted(os_ for _, os_ in _VERDICTS), "one verdict per shard OS"
    steps_without_pattern = []
    for name, os_ in _VERDICTS:
        job = ci["jobs"][name]
        assert sorted(job["needs"]) == ["python-pr-gate-plan", "python-pr-gate-shard"]
        assert _norm(job["if"]) == "!cancelled() && " + _GATE_IF, (
            "without a status function a skipped or failed shard skips the verdict, which "
            "merge-policy would read as a skipped gate instead of a red one")
        assert isinstance(job.get("timeout-minutes"), int)
        assert "strategy" not in job
        for step in job["steps"]:
            assert not step.get("continue-on-error"), f"{step.get('name')} must be able to fail the job"
            assert step.get("if") is None, f"{step.get('name')} must run unconditionally"
        download = _by_name(job, "Download this platform's shard results")
        pattern = download["with"]["pattern"]
        assert pattern == f"pr-gate-shard-{os_}-*"
        own = upload["with"]["name"].replace("${{ matrix.os }}", os_).replace("${{ matrix.shard }}", "3")
        other = upload["with"]["name"].replace("${{ matrix.os }}", [o for o in oses if o != os_][0])
        assert fnmatch.fnmatchcase(own, pattern) and not fnmatch.fnmatchcase(other, pattern)
        assert download["with"]["merge-multiple"] is True and download["with"]["path"] == "shard-out"
        plan = _by_name(job, "Download the plan")
        assert plan["with"] == _by_name(ci["jobs"]["python-pr-gate-shard"], "Download the plan")["with"]
        verify = _by_name(job, "Every selected test ran in exactly one shard")
        assert verify["env"] == {"PLAN_RESULT": "${{ needs.python-pr-gate-plan.result }}"}
        assert _norm(verify["run"]) == (
            'python scripts/run_affected_tests.py --collect shard-out '
            '--plan "$RUNNER_TEMP/pr-gate-plan/plan.json" --plan-result "$PLAN_RESULT"')
        steps_without_pattern.append([{**s, "with": {**s.get("with", {}), "pattern": None}}
                                      if s is download else s for s in job["steps"]])
    assert steps_without_pattern[0] == steps_without_pattern[1], (
        "the two verdicts differ only in the platform whose shards they read")


def test_the_windows_gate_runs_the_linux_gate_selection_on_windows():
    ci = _load_workflow(_CI)
    plan = ci["jobs"]["python-pr-gate-plan"]
    shard = ci["jobs"]["python-pr-gate-shard"]
    test_shard = ci["jobs"]["test-shard"]
    assert plan["runs-on"] == "ubuntu-latest", "the selection is computed once, on Linux"
    assert shard["strategy"]["matrix"]["os"] == ["ubuntu-latest", "windows-latest"]
    assert shard["runs-on"] == "${{ matrix.os }}"
    assert ci["jobs"]["python-pr-gate-windows"]["name"] == "Python PR gate, Windows (affected tests)"
    for name in ("Create the project venv", "Install", "Download the plan", "Affected tests (this shard)"):
        assert _by_name(shard, name).get("if") is None, f"{name!r} runs the same on both platforms"
    for name in ("Create the project venv", "Install"):
        assert _by_name(shard, name)["run"] == _by_name(plan, name)["run"], (
            f"{name!r} must stay identical to the plan job's step")
    for name in ("Put the venv on PATH (POSIX)", "Put the venv on PATH (Windows)", "Install"):
        mine, theirs = _by_name(shard, name), _by_name(test_shard, name)
        assert (mine.get("if"), mine.get("shell")) == (theirs.get("if"), theirs.get("shell")), (
            f"{name!r}: the venv goes on PATH and installs the way the test shard does it")
    assert _by_name(shard, "Put the venv on PATH (Windows)")["run"] == \
        _by_name(test_shard, "Put the venv on PATH (Windows)")["run"]
    assert "defaults" not in shard and "defaults" not in test_shard
    assert _by_name(shard, "Affected tests (this shard)").get("shell") is None, (
        "pytest runs under the runner's default shell, as in the Windows test shard")
    assert _by_name(test_shard, "Tests (this shard)").get("shell") is None


def test_the_full_ci_matrix_is_left_in_place_as_its_own_gate():
    ci = _load_workflow(_CI)
    assert "full-ci" in ci["jobs"]["test"]["if"]
    assert ci["jobs"]["test"]["strategy"]["matrix"]["os"] == ["windows-latest", "ubuntu-latest"]
    assert "labeled" in ci["on"]["pull_request"]["types"]
