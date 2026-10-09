"""The documentation workflow builds on docs changes and deploys only on a release."""

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
