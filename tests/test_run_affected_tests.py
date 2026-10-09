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
sentinel on its own skips pytest. The last section binds the wrapper to the
real tree and pins the shape of the ci.yml job so the gate cannot be quietly
narrowed, label-gated or made non-blocking.
"""

import importlib.util
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
    """Run main() with the selector, the base-ref check and pytest replaced;
    returns (exit status, the pytest mock, the step summary text)."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(rat, "REPO", tmp_path)
    monkeypatch.setattr(rat, "_run_selector", lambda depth, base, files, max_share=None: selector)
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


def test_the_ci_job_runs_on_every_pull_request_and_cannot_be_skipped_green():
    ci = _load_workflow(_CI)
    job = ci["jobs"]["python-pr-gate"]
    assert job["runs-on"] == "ubuntu-latest"
    assert _norm(job["if"]) == _norm("""
        github.event_name == 'pull_request' &&
        !contains(github.event.pull_request.labels.*.name, 'full-ci')"""), (
        "every unlabelled PR, and never a labelled one, whose whole suite the matrix runs")
    assert isinstance(job.get("timeout-minutes"), int)
    assert job["timeout-minutes"] >= ci["jobs"]["test"]["timeout-minutes"], (
        "a change to a hub module runs most of what the matrix job runs, on one host")
    assert "strategy" not in job, "one host: a subset cannot be measured per platform"

    checkout = job["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0, (
        "the selector diffs from the merge base with origin/master and the hygiene "
        "gate resolves the same baseline; a shallow checkout has no origin/master")
    assert checkout["with"]["persist-credentials"] is False

    runs = [s.get("run", "") for s in job["steps"]]
    wanted = ["uv lock --check", "python scripts/check_hygiene.py", "ruff check .",
              "python scripts/run_affected_tests.py --depth 1"]
    positions = []
    for cmd in wanted:
        hits = [i for i, r in enumerate(runs) if cmd in r]
        assert len(hits) == 1, f"{cmd!r} must appear exactly once, got {hits}"
        positions.append(hits[0])
    assert positions == sorted(positions), "lock, hygiene, ruff, then the tests"
    for step in job["steps"]:
        assert not step.get("continue-on-error"), f"{step.get('name')} must be able to fail the job"
        assert step.get("if") is None, f"{step.get('name')} must run unconditionally"
    tests_step = [r for r in runs if "run_affected_tests.py" in r][0]
    assert "--wide" not in tests_step and "pytest tests" not in tests_step, (
        "the gate runs the selection only, never the whole suite")


def test_the_full_ci_matrix_is_left_in_place_as_its_own_gate():
    ci = _load_workflow(_CI)
    assert "full-ci" in ci["jobs"]["test"]["if"]
    assert ci["jobs"]["test"]["strategy"]["matrix"]["os"] == ["windows-latest", "ubuntu-latest"]
    assert "labeled" in ci["on"]["pull_request"]["types"]
