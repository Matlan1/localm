"""The documentation workflow builds on docs changes and deploys only on a release."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "docs.yml"


@pytest.fixture(scope="module")
def wf():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers(wf):
    return wf["on"] if "on" in wf else wf[True]


def test_triggers_are_pull_request_release_and_dispatch_only(wf):
    assert set(_triggers(wf)) == {"pull_request", "release", "workflow_dispatch"}


def test_release_trigger_is_published_only(wf):
    assert _triggers(wf)["release"] == {"types": ["published"]}


def test_pull_request_is_path_filtered_to_the_docs_inputs(wf):
    paths = set(_triggers(wf)["pull_request"]["paths"])
    assert {"docs/**", "mkdocs.yml", "scripts/export_openapi.py"} <= paths


def test_dispatch_defaults_to_a_dry_run(wf):
    dry_run = _triggers(wf)["workflow_dispatch"]["inputs"]["dry_run"]
    assert dry_run["type"] == "boolean"
    assert dry_run["default"] is True


def test_the_build_runs_mkdocs_in_strict_mode(wf):
    runs = [s.get("run", "") for s in wf["jobs"]["build"]["steps"]]
    assert "mkdocs build --strict" in runs


def test_nothing_is_uploaded_on_a_pull_request(wf):
    upload = next(s for s in wf["jobs"]["build"]["steps"]
                  if str(s.get("uses", "")).startswith("actions/upload-pages-artifact@"))
    assert upload["if"] == "github.event_name != 'pull_request'"


def test_only_the_deploy_job_can_write_pages(wf):
    assert wf["permissions"] == {"contents": "read"}
    assert wf["jobs"]["build"]["permissions"] == {"contents": "read", "pages": "read"}
    assert wf["jobs"]["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}


def test_deploy_waits_on_the_build_decision(wf):
    deploy = wf["jobs"]["deploy"]
    assert deploy["needs"] == "build"
    assert deploy["if"] == "needs.build.outputs.deploy == 'true'"


def test_the_decision_step_deploys_only_on_release_or_a_wet_dispatch(wf):
    step = next(s for s in wf["jobs"]["build"]["steps"] if s.get("id") == "decide")
    script = step["run"]
    assert '[ "$EVENT" = "release" ]' in script
    assert '[ "$EVENT" = "workflow_dispatch" ] && [ "$DRY_RUN" = "false" ]' in script
    assert "pull_request" not in script


def _bash():
    found = shutil.which("bash")
    if found is None or "system32" in found.lower().replace("\\", "/"):
        pytest.skip("no POSIX bash on PATH")
    return found


def _run_decision(wf, tmp_path, *, event, dry_run, gh_mode):
    script = next(s for s in wf["jobs"]["build"]["steps"] if s.get("id") == "decide")["run"]
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    calls = tmp_path / "gh_calls"
    shim = shim_dir / "gh"
    shim.write_text(
        '#!/bin/sh\n'
        f'echo "$@" >> "{calls.as_posix()}"\n'
        'case "$GH_MODE" in\n'
        '  enabled) echo "{}"; exit 0;;\n'
        '  missing) echo "gh: Not Found (HTTP 404)" >&2; exit 1;;\n'
        '  *) echo "gh: Server Error (HTTP 500)" >&2; exit 1;;\n'
        'esac\n', encoding="utf-8", newline="\n")
    shim.chmod(0o755)
    out, summary = tmp_path / "out", tmp_path / "summary"
    env = dict(os.environ, EVENT=event, DRY_RUN=dry_run, GH_TOKEN="x", REPO="o/r",
               GH_MODE=gh_mode, GITHUB_OUTPUT=out.as_posix(),
               GITHUB_STEP_SUMMARY=summary.as_posix(),
               PATH=shim_dir.as_posix() + os.pathsep + os.environ["PATH"])
    proc = subprocess.run([_bash(), "-c", script], capture_output=True, text=True, env=env,
                          timeout=60)
    deploy = out.read_text(encoding="utf-8").strip() if out.exists() else None
    return proc, deploy, calls.exists()


@pytest.mark.parametrize("event,dry_run", [
    ("pull_request", ""), ("workflow_dispatch", "true"), ("push", "")])
def test_a_build_only_event_never_asks_about_pages_and_never_deploys(wf, tmp_path, event, dry_run):
    proc, deploy, asked = _run_decision(wf, tmp_path, event=event, dry_run=dry_run,
                                        gh_mode="enabled")
    assert proc.returncode == 0, proc.stderr
    assert deploy == "deploy=false"
    assert not asked


@pytest.mark.parametrize("event,dry_run", [
    ("release", ""), ("workflow_dispatch", "false")])
def test_a_deploying_event_deploys_when_pages_is_enabled(wf, tmp_path, event, dry_run):
    proc, deploy, asked = _run_decision(wf, tmp_path, event=event, dry_run=dry_run,
                                        gh_mode="enabled")
    assert proc.returncode == 0, proc.stderr
    assert deploy == "deploy=true"
    assert asked


def test_a_release_without_pages_warns_and_does_not_deploy(wf, tmp_path):
    proc, deploy, _ = _run_decision(wf, tmp_path, event="release", dry_run="",
                                    gh_mode="missing")
    assert proc.returncode == 0, proc.stderr
    assert deploy == "deploy=false"
    assert "::warning title=GitHub Pages is not enabled::" in proc.stdout


def test_an_unexpected_pages_api_error_fails_the_step(wf, tmp_path):
    proc, deploy, _ = _run_decision(wf, tmp_path, event="release", dry_run="",
                                    gh_mode="error")
    assert proc.returncode != 0
    assert deploy is None
    assert "HTTP 500" in proc.stderr
