# SPDX-License-Identifier: AGPL-3.0-or-later
"""tests/_shard.py: the shards partition a real pytest selection, balanced by duration."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _shard

REPO_ROOT = Path(__file__).resolve().parent.parent

SUITE = {
    "test_alpha.py": "import pytest\n\n@pytest.mark.parametrize('n', range(9))\ndef test_a(n):\n    pass\n",
    "test_beta.py": "def test_b1():\n    pass\n\ndef test_b2():\n    pass\n\ndef test_b3():\n    pass\n",
    "test_gamma.py": "import pytest\n\n@pytest.mark.integration\ndef test_g_integration():\n    pass\n\ndef test_g():\n    pass\n",
}


@pytest.fixture
def suite(tmp_path):
    root = tmp_path / "suite"
    root.mkdir()
    for name, body in SUITE.items():
        (root / name).write_text(body, encoding="utf-8")
    (root / "pytest.ini").write_text("[pytest]\nmarkers =\n    integration: x\n", encoding="utf-8")
    return root


def _pytest(suite_dir, *args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests._shard", "-p", "no:cacheprovider",
         "-p", "no:randomly", "-m", "not integration", "-q", *args],
        cwd=str(suite_dir), env=env, capture_output=True, text=True, timeout=120)


def _collected(suite_dir, *args):
    run = _pytest(suite_dir, "--collect-only", *args)
    assert run.returncode == 0, run.stdout + run.stderr
    return sorted(line for line in run.stdout.splitlines() if "::" in line)


def _shard_ids(suite_dir, index, count, *extra):
    out = suite_dir / f"ids-{index}.json"
    measured = suite_dir / f"measured-{index}.json"
    run = _pytest(suite_dir, "--shard", f"{index}/{count}", "--shard-ids-out", str(out),
                  "--shard-durations-out", str(measured), *extra)
    assert run.returncode == 0, run.stdout + run.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["executed"] = sorted(json.loads(measured.read_text(encoding="utf-8")))
    return payload


def test_parse_shard_accepts_i_of_n_and_rejects_the_rest():
    assert _shard.parse_shard("2/4") == (2, 4)
    assert _shard.parse_shard("1/1") == (1, 1)
    for bad in ("4", "0/4", "5/4", "1/0", "a/b", "-1/4", ""):
        with pytest.raises(ValueError):
            _shard.parse_shard(bad)


def test_assign_balances_by_duration_and_ignores_input_order():
    durations = {"a": 1.0, "b": 5.0, "c": 5.0, "d": 9.0, "e": 10.0}
    ids = list(durations)
    owner = _shard.assign(ids, 1.0, durations, 2)
    loads = [sum(durations[n] for n in ids if owner[n] == s) for s in (0, 1)]
    assert sorted(loads) == [15.0, 15.0]
    shuffled = ids[:]
    random.Random(7).shuffle(shuffled)
    assert _shard.assign(shuffled, 1.0, durations, 2) == owner


def test_assign_weighs_a_test_the_file_does_not_list_by_the_default():
    owner = _shard.assign(["heavy", "u1", "u2", "u3", "u4"], 1.0, {"heavy": 4.0}, 2)
    assert {owner["heavy"]} != {owner["u1"], owner["u2"], owner["u3"], owner["u4"]}
    assert sorted(list(owner.values()).count(s) for s in (0, 1)) == [1, 4]


def test_every_selected_test_runs_in_exactly_one_shard(suite):
    full = _collected(suite)
    assert len(full) == 13
    parts = [_shard_ids(suite, i, 3) for i in (1, 2, 3)]
    assert _shard.verify_partition(parts, 3) == []
    assert [p["executed"] for p in parts] == [p["selected"] for p in parts]
    ran = sorted(n for p in parts for n in p["executed"])
    assert ran == full
    assert all(len(p["selected"]) >= 4 for p in parts)


def test_the_integration_deselection_is_applied_before_the_split(suite):
    parts = [_shard_ids(suite, i, 2) for i in (1, 2)]
    assert not any("integration" in n for p in parts for n in p["selected"])
    assert parts[0]["all_total"] == 13


def test_a_shard_run_under_xdist_selects_the_same_tests(suite):
    pytest.importorskip("xdist")
    serial = _shard_ids(suite, 2, 3)
    parallel = _shard_ids(suite, 2, 3, "-n", "2")
    assert parallel["selected"] == serial["selected"]
    assert parallel["executed"] == parallel["selected"]


def test_durations_file_decides_which_tests_share_a_shard(suite):
    full = _collected(suite)
    heavy = full[0]
    durations = suite / "durations.json"
    durations.write_text(json.dumps({"default": 0.01, "tests": {heavy: 50.0}}), encoding="utf-8")
    parts = [_shard_ids(suite, i, 2, "--shard-durations", str(durations)) for i in (1, 2)]
    alone = next(p for p in parts if heavy in p["selected"])
    assert alone["selected"] == [heavy]
    assert _shard.verify_partition(parts, 2) == []


def test_verify_partition_names_a_shard_that_never_reported(suite):
    parts = [_shard_ids(suite, i, 3) for i in (1, 2, 3)]
    problems = _shard.verify_partition(parts[:2], 3)
    assert any("expected shards 1..3" in p for p in problems)
    assert any("distinct tests" in p for p in problems)


def test_verify_partition_names_a_test_two_shards_ran(suite):
    parts = [_shard_ids(suite, i, 2) for i in (1, 2)]
    duplicate = parts[0]["selected"][0]
    parts[1]["selected"].append(duplicate)
    problems = _shard.verify_partition(parts, 2)
    assert any(duplicate in p and "ran in shards" in p for p in problems)


def test_verify_partition_names_a_test_no_shard_ran(suite):
    parts = [_shard_ids(suite, i, 2) for i in (1, 2)]
    parts[0]["selected"].pop()
    problems = _shard.verify_partition(parts, 2)
    assert any("distinct tests" in p for p in problems)


def test_verify_partition_rejects_shards_that_collected_different_selections(suite):
    parts = [_shard_ids(suite, i, 2) for i in (1, 2)]
    parts[1]["all_sha256"] = "0" * 64
    assert any("different selections" in p for p in _shard.verify_partition(parts, 2))


def test_verify_cli_exit_status_follows_the_partition(suite, tmp_path):
    for i in (1, 2):
        _shard_ids(suite, i, 2)
    ids = tmp_path / "ids"
    ids.mkdir()
    for i in (1, 2):
        (suite / f"ids-{i}.json").replace(ids / f"ids-{i}.json")
        (ids / f"status-{i}.txt").write_text("success\n", encoding="utf-8")
    assert _shard.main(["verify", str(ids), "--count", "2"]) == 0
    (ids / "ids-2.json").unlink()
    assert _shard.main(["verify", str(ids), "--count", "2"]) == 1


def test_verify_cli_fails_on_a_shard_whose_job_did_not_succeed(suite, tmp_path):
    ids = tmp_path / "ids"
    ids.mkdir()
    for i in (1, 2):
        _shard_ids(suite, i, 2)
        (suite / f"ids-{i}.json").replace(ids / f"ids-{i}.json")
        (ids / f"status-{i}.txt").write_text("success\n", encoding="utf-8")
    assert _shard.main(["verify", str(ids), "--count", "2"]) == 0
    (ids / "status-2.txt").write_text("failure\n", encoding="utf-8")
    assert _shard.main(["verify", str(ids), "--count", "2"]) == 1
    (ids / "status-2.txt").unlink()
    assert _shard.main(["verify", str(ids), "--count", "2"]) == 1


def test_a_run_records_measured_durations_and_merge_folds_the_small_ones(suite, tmp_path):
    out = tmp_path / "measured.json"
    run = _pytest(suite, "--shard-durations-out", str(out))
    assert run.returncode == 0, run.stdout + run.stderr
    measured = json.loads(out.read_text(encoding="utf-8"))
    assert len(measured) == 13 and all(v >= 0 for v in measured.values())
    folded = _shard.merge_durations({"slow": 2.0, "fast1": 0.02, "fast2": 0.04})
    assert folded == {"default": 0.03, "tests": {"slow": 2.0}}
    assert _shard.merge_durations({"only": 0.5})["default"] == _shard.UNKNOWN_WEIGHT


def test_merge_cli_writes_a_file_load_durations_reads_back(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    (directory / "durations-a.json").write_text(json.dumps({"t1": 3.0, "t2": 0.01}), encoding="utf-8")
    (directory / "durations-b.json").write_text(json.dumps({"t3": 1.5}), encoding="utf-8")
    out = tmp_path / "durations.json"
    assert _shard.main(["merge", str(directory), "--out", str(out)]) == 0
    default, tests = _shard.load_durations(out)
    assert tests == {"t1": 3.0, "t3": 1.5} and default == 0.01


def test_a_missing_durations_file_weighs_every_test_the_same(tmp_path):
    assert _shard.load_durations(tmp_path / "absent.json") == (_shard.UNKNOWN_WEIGHT, {})
    assert _shard.load_durations(None) == (_shard.UNKNOWN_WEIGHT, {})


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_the_committed_durations_for_each_ci_platform_are_well_formed(platform):
    path = _shard.TESTS_DIR / f"shard_durations_{platform}.json"
    default, tests = _shard.load_durations(path)
    assert 0 < default < _shard.KEEP_AT_LEAST
    assert len(tests) > 500
    assert all(node_id.startswith("tests/") and "::" in node_id for node_id in tests)
    assert all(seconds >= _shard.KEEP_AT_LEAST for seconds in tests.values())


def test_the_default_durations_file_is_named_for_the_running_platform():
    assert _shard.default_durations_path().name == f"shard_durations_{sys.platform}.json"


def test_the_digest_ignores_repeats_and_order():
    assert _shard.digest(["b", "a", "a"]) == _shard.digest(["a", "b"])
    assert _shard.digest(["a"]) != _shard.digest(["a", "b"])


def test_verify_cli_can_require_each_shards_coverage_file(suite, tmp_path):
    ids = tmp_path / "ids"
    ids.mkdir()
    for i in (1, 2):
        _shard_ids(suite, i, 2)
        (suite / f"ids-{i}.json").replace(ids / f"ids-{i}.json")
        (ids / f"status-{i}.txt").write_text("success\n", encoding="utf-8")
    argv = ["verify", str(ids), "--count", "2", "--coverage-files"]
    assert _shard.main(argv) == 1
    (ids / "coverage-1.dat").write_bytes(b"x")
    assert _shard.main(argv) == 1
    (ids / "coverage-2.dat").write_bytes(b"x")
    assert _shard.main(argv) == 0
