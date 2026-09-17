# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/merge_policy.py, the decision behind the `merge-policy` job in
ci.yml, and the shape of that job.

The policy's contract: lint and gui-tests must succeed on every pull request;
without the `full-ci` label python-pr-gate must succeed and the change must
not be a release; with the label the matrix must succeed; a release PR
without the label fails with the label named, every other PR merges without
the matrix; a skipped, failed, cancelled or missing needed job never passes.
The last section binds the script to the real tree and pins the ci.yml job so
it cannot be quietly skipped, narrowed or made non-blocking.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "merge_policy.py"
_CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

GREEN_UNLABELLED = {"python-pr-gate": "success", "lint": "success",
                    "gui-tests": "success", "test": "skipped"}
GREEN_LABELLED = {"python-pr-gate": "skipped", "lint": "success",
                  "gui-tests": "success", "test": "success"}


def _load(path=_SCRIPT, name="merge_policy"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mp():
    return _load()


# --- classifying the change -------------------------------------------------

@pytest.mark.parametrize("path, category", [
    ("VERSION", "release"),
    ("localm/auth.py", "trust boundary"),
    ("localm/scopes.py", "trust boundary"),
    ("localm/tls.py", "trust boundary"),
    ("localm/bindhost.py", "trust boundary"),
    ("localm/netlisten.py", "trust boundary"),
    ("localm/portmux.py", "trust boundary"),
    ("localm/netpolicy.py", "trust boundary"),
    ("localm/netpin.py", "trust boundary"),
    ("localm/pathsafe.py", "trust boundary"),
    ("localm/config.py", "trust boundary"),
    ("localm/plugins/engine.py", "plugin engine"),
    ("localm/plugins/contract.py", "plugin engine"),
    ("localm/plugins/loader.py", "plugin engine"),
    ("localm/inference/engine.py", "inference"),
    ("localm/inference/http_server.py", "inference"),
    ("localm/inference/backends/llamacpp/_worker.py", "inference"),
    ("localm/inference/_embedder_runner.py", "inference"),
    ("localm/_mp_spawn.py", "inference"),
    ("localm/setup_llama.py", "inference"),
    ("runtime/localm_llama_runtime/__init__.py", "inference"),
    ("pyproject.toml", "packaging"),
    ("uv.lock", "packaging"),
    ("setup.sh", "packaging"),
    ("setup.bat", "packaging"),
    ("install.sh", "packaging"),
    ("installer/gui.py", "packaging"),
    ("launcher.pyw", "packaging"),
    ("localm/updater.py", "packaging"),
    ("localm/_apply_update.py", "packaging"),
    (".github/workflows/ci.yml", "CI workflows"),
    (".github/workflows/publish-pypi.yml", "CI workflows"),
    ("scripts/affected_tests.py", "CI workflows"),
    ("scripts/run_affected_tests.py", "CI workflows"),
    ("scripts/merge_policy.py", "CI workflows"),
    ("scripts/check_coverage_floors.py", "CI workflows"),
])
def test_each_matrix_category_matches_its_files(mp, path, category):
    found = mp.classify([path])
    assert len(found) == 1, found
    (name, files), = found.items()
    assert name.startswith(category)
    assert files == [path]


@pytest.mark.parametrize("path", [
    "docs/architecture.md",
    "README.md",
    "CHANGELOG.md",
    "tests/test_auth.py",
    "tests/conftest.py",
    "localm/inference/routes/admin.py",
    "localm/inference/routes/__init__.py",
    "localm/plugins/gui/static/app/init.js",
    "localm/plugins/coder/plug.py",
    "localm/plugins/builtin/chat/plug.py",
    "localm/plugins/mcpserver/server.py",
    "localm/bugreport.py",
    "localm/discover.py",
    "scripts/check_hygiene.py",
    "scripts/write_coverage_summary.py",
    ".github/dependabot.yml",
    ".github/workflows-parked/build-llama-cuda-linux.yml",
    "package.json",
    "localm/authx.py",
    "VERSIONS",
])
def test_files_outside_every_category_need_no_matrix(mp, path):
    assert mp.classify([path]) == {}


def test_a_route_change_is_excluded_from_the_inference_category_by_the_bang_pattern(mp):
    patterns = mp.CATEGORIES["inference, workers and the native binding"]
    assert "!localm/inference/routes/**" in patterns
    assert mp.matches("localm/inference/routes/admin.py", patterns) is False
    assert mp.matches("localm/inference/engine.py", patterns) is True


def test_star_stays_within_one_segment_and_double_star_crosses(mp):
    assert mp.matches("localm/plugins/engine.py", ("localm/plugins/*.py",))
    assert not mp.matches("localm/plugins/gui/plug.py", ("localm/plugins/*.py",))
    assert mp.matches("localm/plugins/gui/plug.py", ("localm/plugins/**",))
    assert not mp.matches("localm/plugins_engine.py", ("localm/plugins/*.py",))
    assert mp.matches("a/b.py", ("a/?.py",))
    assert not mp.matches("a/bc.py", ("a/?.py",))


def test_backslash_paths_are_matched_and_reported_with_forward_slashes(mp):
    found = mp.classify(["localm\\auth.py", ".github\\workflows\\ci.yml"])
    assert [f for files in found.values() for f in files] == [
        "localm/auth.py", ".github/workflows/ci.yml"]


def test_classify_keeps_category_order_and_sorts_files_within_one(mp):
    found = mp.classify(["setup.sh", "VERSION", "localm/tls.py", "install.sh", "localm/auth.py"])
    assert list(found) == [
        "release",
        "trust boundary (auth, scopes, TLS, bind handling, network policy, path safety, config)",
        "packaging and installers"]
    assert found["packaging and installers"] == ["install.sh", "setup.sh"]


def test_every_pattern_names_something_that_exists_in_the_tree(mp):
    """A category pattern that matches no tracked path is a typo the policy
    would never notice."""
    tracked = []
    for extra in ((), ("--others", "--exclude-standard")):
        tracked += subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", *extra],
                                  capture_output=True, text=True, encoding="utf-8",
                                  check=True).stdout.splitlines()
    for name, patterns in mp.CATEGORIES.items():
        for pattern in patterns:
            bare = pattern.lstrip("!")
            hits = [p for p in tracked if mp.matches(p, (bare,))]
            assert hits, f"{name}: pattern {pattern!r} matches no tracked file"


# --- the decision -------------------------------------------------------------

def test_a_docs_only_unlabelled_pr_passes_on_the_three_cheap_jobs(mp):
    v = mp.decide(False, GREEN_UNLABELLED, {})
    assert v.ok and v.reasons == []
    assert v.is_release is False


@pytest.mark.parametrize("path", [
    "localm/auth.py", "localm/inference/engine.py", "pyproject.toml", "setup.sh",
    ".github/workflows/ci.yml", "localm/plugins/engine.py", "scripts/merge_policy.py"])
def test_a_matrix_category_alone_never_blocks_an_unlabelled_pr(mp, path):
    """The two-platform matrix runs at release, not on an ordinary PR."""
    categories = mp.classify([path])
    assert categories and mp.RELEASE not in categories
    v = mp.decide(False, GREEN_UNLABELLED, categories)
    assert v.ok, v.reasons
    assert v.categories == categories


def test_a_release_without_the_label_fails_naming_version_and_the_label(mp):
    v = mp.decide(False, GREEN_UNLABELLED, mp.classify(["VERSION", "CHANGELOG.md"]))
    assert not v.ok
    assert len(v.reasons) == 1
    assert "release" in v.reasons[0] and "VERSION" in v.reasons[0]
    assert "full-ci" in v.reasons[0]
    assert v.is_release is True


def test_a_release_with_the_label_and_a_green_matrix_passes(mp):
    v = mp.decide(True, GREEN_LABELLED, mp.classify(["VERSION", "CHANGELOG.md"]))
    assert v.ok, v.reasons


def test_a_trust_boundary_change_with_the_label_and_a_green_matrix_passes(mp):
    v = mp.decide(True, GREEN_LABELLED, mp.classify(["localm/auth.py"]))
    assert v.ok, v.reasons


def test_the_labelled_arm_fails_when_the_matrix_was_skipped_rather_than_passing_silently(mp):
    v = mp.decide(True, {**GREEN_LABELLED, "test": "skipped"}, {})
    assert not v.ok
    assert v.reasons == ["test: skipped (must be success on a labelled PR)"]


def test_the_labelled_arm_fails_on_a_red_matrix(mp):
    v = mp.decide(True, {**GREEN_LABELLED, "test": "failure"}, {})
    assert not v.ok
    assert v.reasons == ["test: failure"]


def test_the_unlabelled_arm_fails_when_the_gate_was_skipped_rather_than_passing_silently(mp):
    v = mp.decide(False, {**GREEN_UNLABELLED, "python-pr-gate": "skipped"}, {})
    assert not v.ok
    assert v.reasons == ["python-pr-gate: skipped (must be success on an unlabelled PR)"]


def test_the_unlabelled_arm_fails_on_a_red_gate(mp):
    v = mp.decide(False, {**GREEN_UNLABELLED, "python-pr-gate": "failure"}, {})
    assert not v.ok
    assert v.reasons == ["python-pr-gate: failure"]


@pytest.mark.parametrize("job", ["lint", "gui-tests"])
@pytest.mark.parametrize("full_ci, results", [(False, GREEN_UNLABELLED), (True, GREEN_LABELLED)])
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_lint_and_gui_tests_must_succeed_on_both_arms(mp, job, full_ci, results, result):
    v = mp.decide(full_ci, {**results, job: result}, {})
    assert not v.ok
    assert len(v.reasons) == 1 and v.reasons[0].startswith(f"{job}: {result}")


@pytest.mark.parametrize("job", ["python-pr-gate", "test"])
@pytest.mark.parametrize("result", ["failure", "cancelled"])
def test_a_job_skipped_by_design_on_one_arm_still_fails_that_arm_when_it_ran_red(mp, job, result):
    v = mp.decide(job == "python-pr-gate", {**GREEN_LABELLED, **GREEN_UNLABELLED,
                                           "test": "success", job: result}, {})
    assert not v.ok
    assert f"{job}: {result}" in v.reasons


@pytest.mark.parametrize("job", ["python-pr-gate", "lint", "gui-tests", "test"])
def test_a_missing_result_never_passes(mp, job):
    for full_ci, results in ((False, GREEN_UNLABELLED), (True, GREEN_LABELLED)):
        results = {k: v for k, v in results.items() if k != job}
        v = mp.decide(full_ci, results, {})
        assert not v.ok, (full_ci, job)
        assert f"{job}: missing" in v.reasons


def test_an_unknown_result_string_never_passes(mp):
    v = mp.decide(False, {**GREEN_UNLABELLED, "lint": "SUCCESS "}, {})
    assert not v.ok
    assert v.reasons[0].startswith("lint: SUCCESS ")


# --- the summary and the command line ---------------------------------------

def test_the_summary_carries_the_verdict_the_results_and_the_reasons(mp):
    v = mp.decide(False, GREEN_UNLABELLED, mp.classify(["VERSION", "localm/auth.py"]))
    text = mp.render_summary(v)
    assert "**FAIL**" in text and "**PASS**" not in text
    assert "- this is a release (VERSION changed)" in text
    assert "`full-ci` label: no" in text
    assert "| python-pr-gate | success |" in text and "| test | skipped |" in text
    assert "- release: `VERSION`" in text
    assert "`localm/auth.py`" in text
    ok = mp.render_summary(mp.decide(True, GREEN_LABELLED, {}))
    assert "**PASS**" in ok and "No changed file is in a matrix category." in ok


def _run(args, env_extra=None):
    return subprocess.run([sys.executable, str(_SCRIPT), *args], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "GITHUB_STEP_SUMMARY": "", **(env_extra or {})})


_GREEN_UNLABELLED_ARGS = ["--full-ci", "false", "--result", "python-pr-gate=success",
                          "--result", "lint=success", "--result", "gui-tests=success",
                          "--result", "test=skipped"]
_GREEN_LABELLED_ARGS = ["--full-ci", "true", "--result", "python-pr-gate=skipped",
                        "--result", "lint=success", "--result", "gui-tests=success",
                        "--result", "test=success"]


def test_main_passes_a_docs_only_change():
    proc = _run([*_GREEN_UNLABELLED_ARGS, "--files", "docs/architecture.md"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "**PASS**" in proc.stdout
    assert "No changed file is in a matrix category." in proc.stdout


def test_main_passes_an_unlabelled_trust_boundary_change_and_lists_the_category():
    proc = _run([*_GREEN_UNLABELLED_ARGS, "--files", "localm/auth.py"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "**PASS**" in proc.stdout
    assert "- trust boundary" in proc.stdout and "`localm/auth.py`" in proc.stdout


def test_main_fails_an_unlabelled_release():
    proc = _run([*_GREEN_UNLABELLED_ARGS, "--files", "VERSION", "CHANGELOG.md"])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "**FAIL**" in proc.stdout
    assert "this is a release (VERSION changed)" in proc.stdout
    assert "labelled `full-ci`" in proc.stdout


def test_main_passes_a_labelled_release_with_a_green_matrix():
    proc = _run([*_GREEN_LABELLED_ARGS, "--files", "VERSION", "CHANGELOG.md"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "**PASS**" in proc.stdout


def test_main_fails_a_labelled_pr_whose_matrix_did_not_run():
    args = [a if a != "test=success" else "test=skipped" for a in _GREEN_LABELLED_ARGS]
    proc = _run([*args, "--files", "docs/architecture.md"])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "test: skipped (must be success on a labelled PR)" in proc.stdout


def test_main_writes_the_summary_to_the_step_summary_file(tmp_path):
    summary = tmp_path / "summary.md"
    proc = _run([*_GREEN_UNLABELLED_ARGS, "--files", "docs/architecture.md"],
                {"GITHUB_STEP_SUMMARY": str(summary)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert summary.read_text(encoding="utf-8").startswith("### Merge policy")


def test_main_refuses_a_result_for_a_job_the_policy_does_not_know():
    proc = _run(["--full-ci", "false", "--result", "mutation-test=success"])
    assert proc.returncode == 2
    assert "python-pr-gate, lint, gui-tests, test" in proc.stderr


def test_evaluate_reads_the_change_from_git_when_no_files_are_given(mp, monkeypatch):
    seen = {}
    monkeypatch.setattr(mp.affected_tests, "changed_files",
                        lambda base: (seen.setdefault("base", base) and ["VERSION"],
                                      "deadbeef"))
    v = mp.evaluate(False, GREEN_UNLABELLED, "origin/master", None)
    assert seen["base"] == "origin/master"
    assert not v.ok and "VERSION" in v.reasons[0]


def test_a_change_to_the_policy_itself_selects_this_file(mp):
    assert "CI workflows and gates" in mp.classify(["scripts/merge_policy.py"])
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "affected_tests.py"), "--why",
         "--files", "scripts/merge_policy.py"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tests/test_merge_policy.py" in proc.stdout


# --- the ci.yml job ----------------------------------------------------------

def _load_workflow(path):
    wf = yaml.safe_load(path.read_text(encoding="utf-8"))
    wf["on"] = wf.pop(True, wf.get("on"))
    return wf


def _norm(expr):
    return " ".join(str(expr).split())


def test_the_merge_policy_job_cannot_be_skipped_green():
    ci = _load_workflow(_CI)
    job = ci["jobs"]["merge-policy"]
    assert job["name"] == "merge-policy", "the one check name to read"
    assert sorted(job["needs"]) == ["gui-tests", "lint", "python-pr-gate", "test"]
    cond = _norm(job["if"])
    assert "!cancelled()" in cond or "always()" in cond, (
        "without a status function success() is implied and a failed or skipped needed job "
        "skips this one, which reads as a pass")
    assert "github.event_name == 'pull_request'" in cond
    assert isinstance(job.get("timeout-minutes"), int)
    assert "strategy" not in job
    for step in job["steps"]:
        assert not step.get("continue-on-error"), f"{step.get('name')} must be able to fail the job"
        assert step.get("if") is None, f"{step.get('name')} must run unconditionally"


def test_the_merge_policy_job_feeds_the_script_every_needed_result_and_the_label():
    ci = _load_workflow(_CI)
    job = ci["jobs"]["merge-policy"]
    checkout = job["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False
    run_steps = [s for s in job["steps"] if "merge_policy.py" in s.get("run", "")]
    assert len(run_steps) == 1
    step = run_steps[0]
    cmd = _norm(step["run"])
    env = step["env"]
    assert env["FULL_CI"] == "${{ contains(github.event.pull_request.labels.*.name, 'full-ci') }}"
    for needed in ("python-pr-gate", "lint", "gui-tests", "test"):
        var = "RESULT_" + needed.upper().replace("-", "_")
        assert env[var] == "${{ needs.%s.result }}" % needed
        assert f'--result "{needed}=${var}"' in cmd
    assert '--full-ci "$FULL_CI"' in cmd
    assert "--files" not in cmd, "CI reads the change from git"
    assert "${{" not in step["run"], "expressions reach the shell through env, not inline"


def test_the_jobs_the_policy_needs_still_exist_with_their_gating():
    ci = _load_workflow(_CI)
    assert "full-ci" in ci["jobs"]["test"]["if"]
    assert "full-ci" in ci["jobs"]["python-pr-gate"]["if"]
    assert ci["jobs"]["lint"]["if"] == "github.event_name != 'push'"
    assert ci["jobs"]["gui-tests"]["if"] == "github.event_name != 'push'"
    assert "labeled" in ci["on"]["pull_request"]["types"]
    assert "merge-policy" not in ci["jobs"]["mutation-test"].get("needs", []), (
        "mutation testing stays label/dispatch-only and informational")
