# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/mutation_parts.py - splitting a module's mutants over
parallel mutation-run jobs and merging the parts' results for the gate.

The split is checked with mutmut's own matcher (fnmatch over mutant ids);
the merge on synthetic part directories shaped like the downloaded
artifacts, and its output is read back by scripts/check_mutation_floors.py.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mp = _load("mutation_parts")
cmf = _load("check_mutation_floors")

IDS = [f"localm.auth.x_verify__mutmut_{i}" for i in range(1, 131)] + [
    f"localm.auth.xǁKeyStoreǁload__mutmut_{i}" for i in range(1, 25)]


class TestSplit:
    @pytest.mark.parametrize("n", range(1, 11))
    def test_digit_groups_cover_every_digit_once(self, n):
        groups = mp.digit_groups(n)
        assert len(groups) == n
        digits = [d for g in groups for d in "0123456789" if fnmatch.fnmatch(d, f"[{g}]")]
        assert digits == list("0123456789")

    @pytest.mark.parametrize("n", [0, 11])
    def test_digit_groups_refuse_an_impossible_count(self, n):
        with pytest.raises(ValueError):
            mp.digit_groups(n)

    def test_every_mutant_id_matches_exactly_one_part(self):
        jobs = [j for j in mp.shards(["auth"]) if j["module"] == "auth"]
        assert len(jobs) == mp.PARTS["auth"] > 1
        for mutant in IDS:
            hits = [j["part"] for j in jobs if fnmatch.fnmatch(mutant, j["pattern"])]
            assert len(hits) == 1, (mutant, hits)
        assert not any(fnmatch.fnmatch("localm.config.x_f__mutmut_3", j["pattern"]) for j in jobs)
        sizes = [sum(fnmatch.fnmatch(m, j["pattern"]) for m in IDS) for j in jobs]
        assert sum(sizes) == len(IDS) and min(sizes) > 0

    def test_an_unsplit_module_is_one_job_over_all_its_mutants(self):
        assert mp.shards(["scopes"]) == [
            {"module": "scopes", "part": 1, "parts": 1, "pattern": "localm.scopes.*"}]

    def test_shards_keep_the_module_order(self):
        jobs = mp.shards(["scopes", "auth"])
        assert [j["module"] for j in jobs] == ["scopes"] + ["auth"] * mp.PARTS["auth"]
        assert [j["part"] for j in jobs[1:]] == list(range(1, mp.PARTS["auth"] + 1))

    def test_the_slow_modules_are_split(self):
        assert mp.PARTS["auth"] > 1 and mp.PARTS["config"] > 1


HASHES = {"x_verify": "aaaaaaaaaaaa"}


def _part(root: Path, module: str, part: int, codes: dict, hashes=None, stats=True,
          flat: bool = False) -> Path:
    """One part's artifact as download-artifact lays it out: in its own
    directory, or (*flat*, the single-artifact download) in *root* itself."""
    d = root if flat else root / f"mutation-meta-{module}-{part}"
    d.mkdir(parents=True, exist_ok=flat)
    mp.mark(module, part, d)
    (d / f"{module}.py.meta").write_text(json.dumps({
        "exit_code_by_key": codes, "hash_by_function_name": dict(hashes or HASHES),
        "type_check_error_by_key": {}, "durations_by_key": {k: 1.0 for k, v in codes.items() if v is not None},
        "estimated_durations_by_key": {k: 2.0 for k in codes}}), encoding="utf-8")
    (d / f"{module}.py").write_text("def x_verify():\n    pass\n", encoding="utf-8")
    if stats:
        (d / f"{module}.stats.json").write_text("{}", encoding="utf-8")
    return d


A1, A2, A3 = (f"localm.auth.x_verify__mutmut_{i}" for i in (1, 5, 8))


N = mp.PARTS["auth"]
OWN = {1: {A1: 1}, 2: {A2: 0}, N: {A3: -24}}


class TestMerge:
    def _all(self, tmp_path, skip=(), hashes_of=None):
        """Every auth part: A1 in part 1, A2 in part 2, A3 in the last part."""
        parts = tmp_path / "parts"
        for k in range(1, N + 1):
            if k in skip:
                continue
            codes = {m: None for m in (A1, A2, A3)}
            codes.update(OWN.get(k, {}))
            _part(parts, "auth", k, codes, hashes=(hashes_of or {}).get(k))
        return parts

    def test_merge_takes_each_exit_code_from_its_part(self, tmp_path):
        out = tmp_path / "localm"
        assert mp.merge(self._all(tmp_path), out) == []
        merged = json.loads((out / "auth.py.meta").read_text(encoding="utf-8"))
        assert merged["exit_code_by_key"] == {A1: 1, A2: 0, A3: -24}
        assert merged["durations_by_key"] == {A1: 1.0, A2: 1.0, A3: 1.0}
        assert (out / "auth.py").is_file()
        assert sorted(p.name for p in out.glob("*.stats.json")) == sorted(
            f"auth.{k}.stats.json" for k in range(1, N + 1))

    def test_the_gate_reads_the_merged_result(self, tmp_path):
        out = tmp_path / "mutants" / "localm"
        assert mp.merge(self._all(tmp_path), out) == []
        results, problems = cmf.load_results(tmp_path / "mutants", ["localm/auth.py"])
        assert problems == []
        assert results["localm/auth.py"]["mutants"] == {A1: "killed", A2: "survived", A3: "timeout"}

    def test_a_missing_part_is_a_problem_and_leaves_its_mutants_unchecked(self, tmp_path):
        """NEGATIVE: a part that never uploaded must not read as complete."""
        parts = self._all(tmp_path, skip=(2,))
        out = tmp_path / "mutants" / "localm"
        assert mp.merge(parts, out) == [f"auth: part(s) 2 of {N} have no results"]
        results, _ = cmf.load_results(tmp_path / "mutants", ["localm/auth.py"])
        assert results["localm/auth.py"]["mutants"][A2] == "not checked"

    def test_one_exit_code_in_two_parts_is_a_problem(self, tmp_path):
        parts = self._all(tmp_path)
        meta = parts / "mutation-meta-auth-2" / "auth.py.meta"
        data = json.loads(meta.read_text(encoding="utf-8"))
        data["exit_code_by_key"][A1] = 0
        meta.write_text(json.dumps(data), encoding="utf-8")
        problems = mp.merge(parts, tmp_path / "out")
        assert problems == [f"auth: {A1}: an exit code in part 1 and part 2"]

    def test_parts_disagreeing_on_a_function_hash_is_a_problem(self, tmp_path):
        parts = self._all(tmp_path, hashes_of={2: {"x_verify": "bbbbbbbbbbbb"}})
        assert "auth: x_verify: the parts disagree on the function hash" in mp.merge(parts, tmp_path / "o")

    def test_an_unreadable_part_is_a_problem(self, tmp_path):
        parts = self._all(tmp_path)
        (parts / "mutation-meta-auth-2" / "auth.py.meta").write_text("{", encoding="utf-8")
        problems = mp.merge(parts, tmp_path / "o")
        assert len(problems) == 1 and problems[0].startswith("auth part 2: could not read auth.py.meta")

    def test_no_part_directory_at_all_is_a_problem(self, tmp_path):
        (tmp_path / "parts").mkdir()
        assert mp.merge(tmp_path / "parts", tmp_path / "o")
        assert mp.merge(tmp_path / "nowhere", tmp_path / "o")

    def test_a_single_artifact_downloaded_flat_merges(self, tmp_path):
        """download-artifact puts a lone matching artifact straight into the
        target directory, not into a directory named after it."""
        parts = tmp_path / "parts"
        _part(parts, "portmux", 1, {"localm.portmux.x_f__mutmut_1": 1}, hashes={"x_f": "d" * 12},
              flat=True)
        out = tmp_path / "mutants" / "localm"
        assert mp.merge(parts, out) == []
        results, problems = cmf.load_results(tmp_path / "mutants", ["localm/portmux.py"])
        assert problems == [] and results["localm/portmux.py"]["mutants"] == {
            "localm.portmux.x_f__mutmut_1": "killed"}

    def test_a_directory_without_a_marker_is_not_a_part(self, tmp_path):
        """NEGATIVE: results whose module and part nobody recorded are never merged."""
        parts = tmp_path / "parts"
        d = _part(parts, "scopes", 1, {"localm.scopes.x_grants__mutmut_1": 1}, hashes={"x_grants": "c" * 12})
        (d / "scopes.part.json").unlink()
        problems = mp.merge(parts, tmp_path / "o")
        assert problems == [f"no mutation part marker (<module>.part.json) under {parts}"]

    def test_two_results_claiming_one_part_is_a_problem(self, tmp_path):
        parts = tmp_path / "parts"
        _part(parts, "scopes", 1, {"localm.scopes.x_grants__mutmut_1": 1}, hashes={"x_grants": "c" * 12})
        twin = parts / "copy"
        twin.mkdir()
        mp.mark("scopes", 1, twin)
        assert any("two results claim part 1" in p for p in mp.merge(parts, tmp_path / "o"))

    @pytest.mark.parametrize("module, part", [("auth", 0), ("auth", mp.PARTS["auth"] + 1),
                                              ("scopes", 2), ("../x", 1)])
    def test_mark_refuses_a_part_that_does_not_exist(self, tmp_path, module, part):
        with pytest.raises(ValueError):
            mp.mark(module, part, tmp_path)

    def test_an_unreadable_marker_is_a_problem(self, tmp_path):
        parts = tmp_path / "parts"
        d = _part(parts, "scopes", 1, {"localm.scopes.x_grants__mutmut_1": 1}, hashes={"x_grants": "c" * 12})
        (d / "scopes.part.json").write_text('{"module": "scopes"}', encoding="utf-8")
        assert any("unreadable part marker" in p for p in mp.merge(parts, tmp_path / "o"))

    def test_an_unsplit_module_merges_from_its_one_part(self, tmp_path):
        parts = tmp_path / "parts"
        _part(parts, "scopes", 1, {"localm.scopes.x_grants__mutmut_1": 1}, hashes={"x_grants": "c" * 12})
        assert mp.merge(parts, tmp_path / "o") == []

    def test_main_exits_one_and_names_the_problem(self, tmp_path, capsys):
        parts = tmp_path / "parts"
        _part(parts, "auth", 1, {A1: 1})
        assert mp.main(["merge", "--parts", str(parts), "--out", str(tmp_path / "o")]) == 1
        assert "have no results" in capsys.readouterr().err
        assert mp.main(["merge", "--parts", str(self._all(tmp_path / "b")), "--out",
                        str(tmp_path / "o2")]) == 0
