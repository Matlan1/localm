# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shape of the supply-chain workflows: SHA pins, release-job gating, Scorecard.

A release-triggered workflow that goes red blocks a release, and an attestation
is a permanent public record, so the properties pinned here are the ones a
well-meant edit could silently break:

  - every third-party action in every workflow is pinned to a full commit SHA;
  - dependabot keeps updating those pins, including github/codeql-action;
  - release-attest.yml writes nothing (no attestation, no release upload)
    unless it runs for a published release or a dispatch with dry_run off, and
    only its attest job holds id-token, attestations or contents write;
  - scorecard.yml keeps the shape the Scorecard webapp accepts for publishing.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = REPO_ROOT / ".github" / "workflows"
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _load(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(doc: dict) -> dict:
    # PyYAML reads the bare key `on` as the boolean True.
    return doc.get("on", doc.get(True))


def _steps(doc: dict, job: str) -> list[dict]:
    return doc["jobs"][job]["steps"]


def _uses(doc: dict) -> list[str]:
    return [
        s["uses"]
        for job in doc["jobs"].values()
        for s in job.get("steps", [])
        if "uses" in s
    ]


@pytest.mark.parametrize("path", sorted(_WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_every_third_party_action_is_pinned_to_a_commit_sha(path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    unpinned = []
    for ref in _uses(doc):
        if ref.startswith("./"):
            continue
        _, _, version = ref.partition("@")
        if not _SHA.match(version):
            unpinned.append(ref)
    assert unpinned == []


def test_codeql_actions_are_sha_pinned_and_dependabot_does_not_ignore_them():
    refs = [
        r for r in _uses(_load("codeql.yml")) if r.startswith("github/codeql-action/")
    ]
    assert {r.split("/")[2].split("@")[0] for r in refs} == {"init", "analyze"}
    config = yaml.safe_load(
        (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )
    actions = next(
        u for u in config["updates"] if u["package-ecosystem"] == "github-actions"
    )
    ignored = [i["dependency-name"] for i in actions.get("ignore", [])]
    assert not any(name.startswith("github/codeql-action") for name in ignored)


def test_release_attest_triggers_on_published_release_and_dispatch():
    on = _triggers(_load("release-attest.yml"))
    assert on["release"]["types"] == ["published"]
    assert on["workflow_dispatch"]["inputs"]["dry_run"]["default"] is True


def test_release_attest_attest_job_runs_only_for_a_release_or_a_real_dispatch():
    condition = _load("release-attest.yml")["jobs"]["attest"]["if"]
    assert condition == "github.event_name == 'release' || inputs.dry_run == false"


def test_release_attest_only_the_attest_job_holds_write_scopes():
    doc = _load("release-attest.yml")
    assert doc["permissions"] == {"contents": "read"}
    build = doc["jobs"]["build"]["permissions"]
    assert build == {"contents": "read"}
    attest = doc["jobs"]["attest"]["permissions"]
    assert attest == {"contents": "write", "id-token": "write", "attestations": "write"}


def test_release_attest_build_job_never_attests_or_uploads_to_the_release():
    for step in _steps(_load("release-attest.yml"), "build"):
        assert "attest" not in step.get("uses", "")
        assert "gh release upload" not in step.get("run", "")


def test_release_attest_attests_every_file_it_attaches():
    steps = _steps(_load("release-attest.yml"), "attest")
    attest = next(s for s in steps if "attest-build-provenance" in s.get("uses", ""))
    subjects = attest["with"]["subject-path"].split()
    upload = next(s for s in steps if "gh release upload" in s.get("run", ""))["run"]
    for pattern in ("out/*.tar.gz", "out/*.whl", "out/*.cdx.json"):
        assert pattern in subjects
        assert pattern in upload
    assert "out/*.zip" in subjects


def test_scorecard_shape_accepted_for_publishing():
    doc = _load("scorecard.yml")
    assert "env" not in doc and "defaults" not in doc
    assert doc["permissions"] == {"contents": "read"}
    job = doc["jobs"]["analysis"]
    assert job["permissions"] == {"security-events": "write", "id-token": "write"}
    allowed = {
        "actions/checkout",
        "actions/upload-artifact",
        "github/codeql-action/upload-sarif",
        "ossf/scorecard-action",
    }
    assert {s["uses"].split("@")[0] for s in job["steps"]} == allowed
    assert all("run" not in s for s in job["steps"])
    scorecard = next(
        s for s in job["steps"] if s["uses"].startswith("ossf/scorecard-action@")
    )
    assert scorecard["with"]["publish_results"] is True
    triggers = _triggers(doc)
    assert triggers["push"]["branches"] == ["master"]
    assert "schedule" in triggers
