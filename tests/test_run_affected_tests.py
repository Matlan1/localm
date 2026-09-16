# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/run_affected_tests.py, the per-PR Python gate, and the ci.yml job
that runs it.

The wrapper's contract is that pytest's exit status becomes the job's, and
that every way the selection can be wrong or too wide is refused rather than
read as green: a selector exit 3 runs the whole unit suite (or fails, by flag),
a selector crash fails, an empty selection fails, a path that is not an
existing tests/**/test_*.py fails, and only the nothing-affected sentinel on
its own skips pytest. The last section binds the wrapper to the real tree and
pins the shape of the ci.yml job so the gate cannot be quietly narrowed,
label-gated or made non-blocking.
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
    assert rat.pytest_args(s, "full", ["-x"]) == [
        "tests/test_a.py", "tests/sub/test_b.py", "-m", "not integration", "-n", "auto", "-x"]


def test_wide_full_runs_the_whole_unit_suite_without_coverage(rat):
    args = rat.pytest_args(rat.Selection("wide"), "full", [])
    assert args == ["tests", "-m", "not integration", "-n", "auto"]
    assert not any(a.startswith("--cov") for a in args)


def test_wide_fail_nothing_and_failed_run_no_pytest(rat):
    assert rat.pytest_args(rat.Selection("wide"), "fail", []) is None
    assert rat.pytest_args(rat.Selection("nothing"), "full", []) is None
    assert rat.pytest_args(rat.Selection("failed"), "full", []) is None


# --- main: exit status and the summary ----------------------------------------

def _drive(rat, monkeypatch, tmp_path, selector, pytest_status=0, argv=()):
    """Run main() with the selector and pytest replaced; returns
    (exit status, the pytest mock, the step summary text)."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(rat, "REPO", tmp_path)
    monkeypatch.setattr(rat, "_run_selector", lambda depth, base, files: selector)
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


def test_wide_runs_the_whole_unit_suite_and_propagates_its_status(rat, repo, monkeypatch):
    for pytest_status in (0, 1):
        status, run_pytest, text = _drive(
            rat, monkeypatch, repo,
            _cp(3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
                "800 of 800 test files affected by 3 changed file(s)\nWIDE: 100% ..."),
            pytest_status=pytest_status, argv=["--wide", "full"])
        assert status == pytest_status
        run_pytest.assert_called_once_with(["tests", "-m", "not integration", "-n", "auto"])
        assert "wider than the selector's limit" in text
        assert "whole unit suite runs once" in text
        assert "WIDE: 100%" in text


def test_wide_fail_flag_fails_without_running_pytest(rat, repo, monkeypatch):
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo,
        _cp(3, "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n", "WIDE: 100% ..."),
        argv=["--wide", "fail"])
    assert status == 1
    run_pytest.assert_not_called()
    assert "job fails" in text and "needs the full suite" in text


def test_wide_is_never_treated_as_nothing_affected(rat, repo, monkeypatch):
    """The wide sentinel arrives with exit 3 AND a path pytest would refuse;
    neither half may read as a docs-only change."""
    for stdout in ("tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n",
                   "tests/NO_TEST_FILE_IS_AFFECTED\n", ""):
        status, run_pytest, text = _drive(rat, monkeypatch, repo, _cp(3, stdout, "WIDE"),
                                          argv=["--wide", "fail"])
        assert status == 1, stdout
        run_pytest.assert_not_called()
        assert "No test file is affected" not in text


def test_a_selector_crash_fails_without_running_pytest(rat, repo, monkeypatch):
    status, run_pytest, text = _drive(
        rat, monkeypatch, repo,
        _cp(1, "tests/AFFECTED_TESTS_FAILED_SEE_STDERR\n", "Traceback (most recent call last)"))
    assert status == 1
    run_pytest.assert_not_called()
    assert "selector failed" in text and "selector exited 1" in text and "Traceback" in text


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

    def fake_selector(depth, base, files):
        seen.update(depth=depth, base=base, files=files)
        return _cp(0, "tests/NO_TEST_FILE_IS_AFFECTED\n")

    monkeypatch.setattr(rat, "REPO", repo)
    monkeypatch.setattr(rat, "_run_selector", fake_selector)
    monkeypatch.setattr(rat, "_run_pytest", MagicMock(return_value=0))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert rat.main(["--depth", "2", "--base", "origin/dev", "--files", "a.py", "b.md"]) == 0
    assert seen == {"depth": 2, "base": "origin/dev", "files": ["a.py", "b.md"]}
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
    assert "`pytest tests/test_run_affected_tests.py -m 'not integration' -n auto`" in proc.stdout


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
    assert _norm(job["if"]) == "github.event_name == 'pull_request'"
    assert "full-ci" not in job["if"], "the gate must not be label-gated"
    assert isinstance(job.get("timeout-minutes"), int)
    assert "strategy" not in job, "one host: a subset cannot be measured per platform"

    checkout = job["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0, (
        "the selector diffs from the merge base with origin/master and the hygiene "
        "gate resolves the same baseline; a shallow checkout has no origin/master")
    assert checkout["with"]["persist-credentials"] is False

    runs = [s.get("run", "") for s in job["steps"]]
    wanted = ["uv lock --check", "python scripts/check_hygiene.py", "ruff check .",
              "python scripts/run_affected_tests.py --depth 1 --wide full"]
    positions = []
    for cmd in wanted:
        hits = [i for i, r in enumerate(runs) if cmd in r]
        assert len(hits) == 1, f"{cmd!r} must appear exactly once, got {hits}"
        positions.append(hits[0])
    assert positions == sorted(positions), "lock, hygiene, ruff, then the tests"
    for step in job["steps"]:
        assert not step.get("continue-on-error"), f"{step.get('name')} must be able to fail the job"
        assert step.get("if") is None, f"{step.get('name')} must run unconditionally"


def test_the_full_ci_matrix_is_left_in_place_as_its_own_gate():
    ci = _load_workflow(_CI)
    assert "full-ci" in ci["jobs"]["test"]["if"]
    assert ci["jobs"]["test"]["strategy"]["matrix"]["os"] == ["windows-latest", "ubuntu-latest"]
    assert "labeled" in ci["on"]["pull_request"]["types"]
