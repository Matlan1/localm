# SPDX-License-Identifier: AGPL-3.0-or-later
"""The sharded test matrix in .github/workflows/ci.yml: the shard jobs and the job that
gathers them agree on the shard count, the file names and the gates."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _jobs() -> dict:
    return yaml.safe_load(_CI.read_text(encoding="utf-8"))["jobs"]


def _norm(text) -> str:
    return " ".join(str(text).split())


def _step(job: dict, name_part: str) -> dict:
    hits = [s for s in job["steps"] if name_part in str(s.get("name", ""))]
    assert len(hits) == 1, f"{name_part!r}: {len(hits)} steps"
    return hits[0]


def _shard_count() -> int:
    shards = _jobs()["test-shard"]["strategy"]["matrix"]["shard"]
    assert shards == list(range(1, len(shards) + 1))
    return len(shards)


def test_the_shard_count_is_written_the_same_everywhere():
    jobs = _jobs()
    count = _shard_count()
    shard, gather = jobs["test-shard"], jobs["test"]
    assert shard["name"].endswith(f"shard ${{{{ matrix.shard }}}}/{count})")
    assert f"--shard ${{{{ matrix.shard }}}}/{count}" in _norm(_step(shard, "Tests")["run"])
    assert f"--count {count} --coverage-files" in _step(gather, "exactly one shard")["run"]


def test_every_shard_runs_xdist_auto():
    run = _norm(_step(_jobs()["test-shard"], "Tests")["run"])
    assert re.search(r"(^| )-n auto( |$)", run)
    assert "-n logical" not in run


def test_every_shard_runs_the_same_pytest_selection_the_split_is_defined_over():
    run = _norm(_step(_jobs()["test-shard"], "Tests")["run"])
    assert '-m "not integration"' in run
    assert "-p tests._shard" in run
    assert "--cov=localm" in run and "--cov-fail-under=0" in run


def test_the_gathering_job_runs_after_the_shards_and_under_the_same_label_gate():
    jobs = _jobs()
    assert jobs["test"]["needs"] == ["test-shard"]
    assert _norm(jobs["test"]["if"]) == "!cancelled() && " + _norm(jobs["test-shard"]["if"])
    assert "full-ci" in jobs["test-shard"]["if"]
    assert jobs["test"]["strategy"]["matrix"]["os"] == jobs["test-shard"]["strategy"]["matrix"]["os"]
    assert jobs["test-shard"]["strategy"]["fail-fast"] is False


def test_the_test_job_cannot_pass_over_a_failed_shard():
    gather = _jobs()["test"]
    names = [s.get("name", "") for s in gather["steps"]]
    verify = _step(gather, "exactly one shard")
    assert "tests._shard verify" in verify["run"]
    for name in ("exactly one shard", "Combine coverage", "Coverage ratchet"):
        step = _step(gather, name)
        assert not step.get("continue-on-error"), f"{name} must be able to fail the job"
        assert step.get("if") is None, f"{name} must run unconditionally"
    assert names.index(verify["name"]) < names.index(_step(gather, "Combine coverage")["name"])
    record = _step(_jobs()["test-shard"], "Record this shard's result")
    assert record["if"] == "always()"
    assert record["env"]["JOB_STATUS"] == "${{ job.status }}"
    assert "${{" not in record["run"], "expressions reach the shell through env, not inline"


def test_shard_results_are_uploaded_and_downloaded_under_matching_names():
    jobs = _jobs()
    upload = _step(jobs["test-shard"], "Upload this shard's results")
    download = _step(jobs["test"], "Download this platform")
    assert upload["with"]["name"] == "shard-${{ matrix.os }}-${{ matrix.shard }}"
    assert download["with"]["pattern"] == "shard-${{ matrix.os }}-*"
    assert download["with"]["merge-multiple"] is True
    assert upload["with"]["path"].rstrip("/") == download["with"]["path"]
    assert upload["if"] == "always()"
    assert upload["with"]["overwrite"] is True, "re-running one failed shard must replace its artifact"


def test_each_shard_writes_the_coverage_file_the_combine_step_reads():
    jobs = _jobs()
    written = jobs["test-shard"]["env"]["COVERAGE_FILE"]
    assert written == "shard-out/coverage-${{ matrix.shard }}.dat"
    combine = _step(jobs["test"], "Combine coverage")["run"]
    assert 'coverage combine shard-out/coverage-*.dat' in combine
    assert "coverage json -o coverage.json" in combine


def test_the_ids_and_durations_each_shard_writes_are_the_names_the_gathering_job_reads():
    shard_run = _norm(_step(_jobs()["test-shard"], "Tests")["run"])
    assert "--shard-ids-out shard-out/ids-${{ matrix.shard }}.json" in shard_run
    assert "--shard-durations-out shard-out/durations-${{ matrix.shard }}.json" in shard_run
    record = _step(_jobs()["test-shard"], "Record this shard's result")
    assert record["env"]["SHARD"] == "${{ matrix.shard }}"
    assert 'shard-out/status-$SHARD.txt' in record["run"]


def test_the_coverage_gates_are_the_ones_the_unsharded_job_ran():
    gather = _jobs()["test"]
    assert _step(gather, "Coverage ratchet")["run"].strip() == "python -m coverage report"
    floors = _step(gather, "Per-module coverage floors")
    assert floors["if"] == "matrix.os == 'windows-latest'"
    assert floors["run"].strip() == "python scripts/check_coverage_floors.py --report"
    assert _step(gather, "Hygiene gate")["run"].strip() == "python scripts/check_hygiene.py"
    assert _step(gather, "uv.lock")["run"].strip() == "uv lock --check"
    assert _step(gather, "Publish coverage summary")["if"] == "always()"
