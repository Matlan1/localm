# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/check_mutation_floors.py - the per-module mutation-score
ratchet with per-mutant dispositions that gates the mutation-test CI job.

Pure and mutmut-free: every test feeds synthetic ``mutants/<module>.meta``
files (the per-file result format mutmut 3.x writes) and a synthetic
baseline. The real committed baseline is checked for internal consistency
against the real pyproject.toml at the end.
"""

from __future__ import annotations

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


cmf = _load("check_mutation_floors")

MOD = "localm/scopes.py"
M1 = "localm.scopes.x_grants__mutmut_1"
M2 = "localm.scopes.x_grants__mutmut_2"
M3 = "localm.scopes.x_normalize__mutmut_1"
HASHES = {"x_grants": "aaaaaaaaaaaa", "x_normalize": "bbbbbbbbbbbb"}


def _results(mutants: dict[str, str], hashes: dict[str, str] | None = None,
             module: str = MOD) -> dict:
    return {module: {"mutants": dict(mutants), "function_hashes": dict(hashes or HASHES)}}


def _baseline(mutants: dict, floor: float = 100.0, hashes: dict | None = None,
              controls: dict | None = None, module: str = MOD) -> dict:
    return {
        "schema": 1,
        "mutmut": "3.7.0",
        "modules": {module: {
            "score_floor": floor,
            "function_hashes": dict(hashes or HASHES),
            "mutants": dict(mutants),
        }},
        "controls": dict(controls or {}),
    }


def _write_meta(root: Path, module: str, exit_codes: dict, hashes: dict | None = None) -> Path:
    meta = root / (module + ".meta")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({
        "exit_code_by_key": exit_codes,
        "hash_by_function_name": dict(hashes or HASHES),
        "type_check_error_by_key": {},
        "durations_by_key": {},
        "estimated_durations_by_key": {},
    }), encoding="utf-8")
    return meta


PYPROJECT = '[tool.mutmut]\nonly_mutate = ["localm/scopes.py"]\n'


# --------------------------------------------------------------------------- #
#  Parsing helpers                                                             #
# --------------------------------------------------------------------------- #

class TestParsing:
    @pytest.mark.parametrize("code,status", [
        (None, "not checked"), (0, "survived"), (1, "killed"), (3, "killed"),
        (33, "no tests"), (34, "skipped"), (35, "suspicious"), (36, "timeout"),
        (37, "caught by type check"), (-24, "timeout"), (-9, "segfault"),
        (99, "suspicious"),
    ])
    def test_status_of_follows_mutmut_exit_code_table(self, code, status):
        assert cmf.status_of(code) == status

    def test_function_of_plain_and_class_mangled_ids(self):
        assert cmf.function_of("localm.scopes.x_grants__mutmut_12") == "x_grants"
        assert cmf.function_of("localm.auth.xǁStoreǁverify__mutmut_3") == "xǁStoreǁverify"

    def test_function_of_rejects_a_non_mutant_id(self):
        with pytest.raises(ValueError):
            cmf.function_of("localm.scopes.grants")

    def test_only_mutate_modules_reads_the_pyproject_list(self):
        text = '[tool.mutmut]\nonly_mutate = ["localm/a.py", "localm\\\\b.py"]\n'
        assert cmf.only_mutate_modules(text) == ["localm/a.py", "localm/b.py"]

    def test_load_results_maps_exit_codes_to_statuses(self, tmp_path):
        _write_meta(tmp_path, MOD, {M1: 1, M2: 0, M3: None})
        results, problems = cmf.load_results(tmp_path, [MOD])
        assert problems == []
        assert results[MOD]["mutants"] == {M1: "killed", M2: "survived", M3: "not checked"}
        assert results[MOD]["function_hashes"] == HASHES

    def test_load_results_reports_a_module_with_no_meta_file(self, tmp_path):
        results, problems = cmf.load_results(tmp_path, [MOD])
        assert results == {}
        assert len(problems) == 1 and MOD in problems[0] and "not a passing" in problems[0]

    def test_load_results_reports_a_malformed_meta_file(self, tmp_path):
        meta = tmp_path / (MOD + ".meta")
        meta.parent.mkdir(parents=True)
        meta.write_text("{not json", encoding="utf-8")
        results, problems = cmf.load_results(tmp_path, [MOD])
        assert results == {} and len(problems) == 1 and "could not read" in problems[0]


# --------------------------------------------------------------------------- #
#  check(): the pure gate                                                      #
# --------------------------------------------------------------------------- #

class TestCheck:
    def test_passes_when_results_match_the_baseline(self):
        res = _results({M1: "killed", M2: "killed", M3: "survived"})
        base = _baseline({M1: "killed", M2: "killed", M3: "survived"}, floor=66.66)
        problems, warnings, rows = cmf.check(res, base, [MOD])
        assert problems == []
        assert warnings == []
        assert rows[0]["score"] == pytest.approx(200 / 3)

    def test_killed_mutant_that_now_survives_is_a_named_regression(self):
        """NEGATIVE: a weakened test must be caught by name, whatever the score does."""
        res = _results({M1: "killed", M2: "survived", M3: "killed"})
        base = _baseline({M1: "killed", M2: "killed", M3: "survived"}, floor=66.66)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert any("recorded as killed are no longer detected" in p and M2 in p for p in problems)

    def test_regression_message_names_a_nondeterministic_outcome_as_a_cause(self):
        """A killed mutant that survives one run is not proof of a weakened
        test: the message offers both causes and the re-run that tells them
        apart."""
        res = _results({M1: "killed", M2: "survived", M3: "survived"})
        base = _baseline({M1: "killed", M2: "killed", M3: "survived"}, floor=33.33)
        problems, _, _ = cmf.check(res, base, [MOD])
        (message,) = [p for p in problems if "recorded as killed are no longer detected" in p]
        assert "weakened or removed" in message
        assert "nondeterministic" in message and "hash seed" in message
        assert "PYTHONHASHSEED=<n> python scripts/mutmut_run.py run <mutant id>" in message
        assert '{"unstable": "<reason>"}' in message
        assert "restore it" not in message

    def test_score_below_floor_fails_even_with_dispositions_present(self):
        res = _results({M1: "killed", M2: "survived", M3: "survived"})
        base = _baseline({M1: "killed", M2: "survived", M3: "survived"}, floor=50.0)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1
        assert "BELOW its floor of 50.00%" in problems[0] and "33.33%" in problems[0]

    def test_score_exactly_at_floor_passes(self):
        res = _results({M1: "killed", M2: "survived"})
        base = _baseline({M1: "killed", M2: "survived"}, floor=50.0)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert problems == []

    def test_new_mutant_without_disposition_fails(self):
        res = _results({M1: "killed", M2: "killed"})
        base = _baseline({M1: "killed"}, floor=100.0)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1
        assert "no disposition" in problems[0] and M2 in problems[0]

    def test_changed_function_makes_all_its_dispositions_stale(self):
        """A function whose source hash moved gets fresh mutant numbering, so
        its recorded dispositions no longer describe the same mutations."""
        res = _results({M1: "killed", M2: "killed", M3: "killed"},
                       hashes={"x_grants": "changed000000", "x_normalize": "bbbbbbbbbbbb"})
        base = _baseline({M1: "killed", M2: "killed", M3: "killed"}, floor=100.0)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1
        assert "x_grants" in problems[0] and "2 mutant(s)" in problems[0]
        assert "x_normalize" not in problems[0]

    def test_equivalent_with_reason_is_excluded_from_the_score(self):
        res = _results({M1: "killed", M2: "survived"})
        base = _baseline({M1: "killed", M2: {"equivalent": "message text only"}}, floor=100.0)
        problems, warnings, rows = cmf.check(res, base, [MOD])
        assert problems == [] and warnings == []
        assert rows[0]["score"] == 100.0 and rows[0]["equivalent"] == 1 and rows[0]["scored"] == 1

    @pytest.mark.parametrize("value", [
        {"equivalent": ""}, {"equivalent": "   "}, {"equivalent": None},
        "equivalent", "EQUIVALENT", {"equivalent": "x", "extra": 1}, 1, None,
    ])
    def test_equivalent_without_a_reason_or_any_other_shape_is_invalid(self, value):
        res = _results({M1: "killed", M2: "survived"})
        base = _baseline({M1: "killed", M2: value}, floor=0.0)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert any("invalid disposition" in p or "no disposition" in p for p in problems), problems

    def test_killed_equivalent_is_a_warning_not_a_failure(self):
        res = _results({M1: "killed", M2: "killed"})
        base = _baseline({M1: "killed", M2: {"equivalent": "believed equivalent"}}, floor=100.0)
        problems, warnings, _ = cmf.check(res, base, [MOD])
        assert problems == []
        assert len(warnings) == 1 and "classified equivalent but was killed" in warnings[0]

    def test_survived_mutant_now_killed_is_a_promotable_warning(self):
        res = _results({M1: "killed", M2: "killed"})
        base = _baseline({M1: "killed", M2: "survived"}, floor=50.0)
        problems, warnings, _ = cmf.check(res, base, [MOD])
        assert problems == []
        assert len(warnings) == 1 and "now killed" in warnings[0] and M2 in warnings[0]

    def test_timeout_and_type_check_count_as_detected(self):
        res = _results({M1: "timeout", M2: "caught by type check"})
        base = _baseline({M1: "killed", M2: "killed"}, floor=100.0)
        problems, _, rows = cmf.check(res, base, [MOD])
        assert problems == [] and rows[0]["score"] == 100.0

    @pytest.mark.parametrize("status", ["no tests", "suspicious", "segfault", "survived", "skipped"])
    def test_every_undetected_outcome_regresses_a_killed_mutant(self, status):
        res = _results({M1: "killed", M2: status})
        base = _baseline({M1: "killed", M2: "killed"}, floor=100.0)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert any(M2 in p and status in p for p in problems), problems

    def test_only_an_equivalent_classification_excludes_a_mutant(self):
        """A `skipped` outcome is scored as undetected: the reason string on an
        equivalent entry is the one way out of the denominator."""
        res = _results({M1: "killed", M2: "skipped"})
        base = _baseline({M1: "killed", M2: "survived"}, floor=50.0)
        problems, _, rows = cmf.check(res, base, [MOD])
        assert problems == [] and rows[0]["scored"] == 2 and rows[0]["score"] == 50.0

    def test_incomplete_run_fails(self):
        res = _results({M1: "killed", M2: "not checked"})
        base = _baseline({M1: "killed", M2: "killed"}, floor=100.0)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1 and "did not complete" in problems[0]

    def test_mutant_vanishing_from_an_unchanged_function_fails(self):
        """NEGATIVE: a `pragma: no mutate` removes a mutant without touching the
        function's AST hash, so the baseline entry outlives the mutant. That
        is an exclusion with no reason attached, and it fails until the entry
        is classified equivalent or deleted by hand."""
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed", M2: "survived"}, floor=50.0)
        problems, warnings, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1 and "not generated although their function is unchanged" in problems[0]
        assert M2 in problems[0] and "pragma" in problems[0]

    def test_mutant_vanishing_with_its_changed_or_removed_function_is_a_warning(self):
        res = _results({M1: "killed"}, hashes={"x_grants": HASHES["x_grants"]})
        base = _baseline({M1: "killed", M3: "survived"}, floor=50.0)
        problems, warnings, _ = cmf.check(res, base, [MOD])
        assert problems == []
        assert len(warnings) == 1 and "changed or were removed" in warnings[0] and M3 in warnings[0]

    def test_module_without_a_baseline_entry_fails(self):
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed"}, module="localm/other.py")
        problems, _, _ = cmf.check(res, base, [MOD, "localm/other.py"])
        assert any(MOD in p and "no baseline entry" in p for p in problems)

    def test_baseline_module_outside_only_mutate_is_a_warning(self):
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed"})
        base["modules"]["localm/gone.py"] = {"score_floor": 1, "function_hashes": {}, "mutants": {}}
        problems, warnings, _ = cmf.check(res, base, [MOD])
        assert problems == []
        assert any("localm/gone.py" in w and "stale" in w for w in warnings)

    def test_wrong_schema_fails(self):
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed"})
        base["schema"] = 2
        problems, _, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1 and "schema" in problems[0]

    def test_non_numeric_floor_fails(self):
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed"}, floor="100")
        problems, _, _ = cmf.check(res, base, [MOD])
        assert any("score_floor is not a number" in p for p in problems)


class TestControls:
    CTL = {"scope-check-weakened": {"module": MOD, "mutant": M1,
                                    "describes": "grants(): required-in-held flipped"}}

    def test_killed_control_passes(self):
        res = _results({M1: "killed", M2: "killed"})
        base = _baseline({M1: "killed", M2: "killed"}, controls=self.CTL)
        assert cmf.check(res, base, [MOD])[0] == []

    def test_surviving_control_fails_by_class_name(self):
        """NEGATIVE: the six SEC-01 classes are pinned individually, so a
        survivor among them is named as such, not folded into the score."""
        res = _results({M1: "survived", M2: "killed"})
        base = _baseline({M1: "survived", M2: "killed"}, floor=50.0, controls=self.CTL)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1
        assert "scope-check-weakened" in problems[0] and "must be killed" in problems[0]
        assert "required-in-held flipped" in problems[0]

    def test_control_whose_function_changed_must_be_repinned(self):
        res = _results({M1: "killed", M2: "killed"},
                       hashes={"x_grants": "changed000000", "x_normalize": "bbbbbbbbbbbb"})
        base = _baseline({M1: "killed", M2: "killed"}, controls=self.CTL)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert any("scope-check-weakened" in p and "re-pin" in p for p in problems)

    def test_control_mutant_that_was_not_generated_fails(self):
        ctl = {"c": {"module": MOD, "mutant": "localm.scopes.x_grants__mutmut_99"}}
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed"}, controls=ctl)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert any("was not generated" in p for p in problems)

    def test_control_pointing_at_an_unmutated_module_fails(self):
        ctl = {"c": {"module": "localm/nope.py", "mutant": "localm.nope.x_f__mutmut_1"}}
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed"}, controls=ctl)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert any("is not mutated" in p for p in problems)

    def test_malformed_control_fails(self):
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed"}, controls={"c": {"mutant": M1}})
        problems, _, _ = cmf.check(res, base, [MOD])
        assert any("needs 'module' and 'mutant'" in p for p in problems)

    def test_control_recorded_as_unstable_fails_even_when_killed(self):
        res = _results({M1: "killed", M2: "killed"})
        base = _baseline({M1: {"unstable": TestUnstable.REASON}, M2: "killed"},
                         controls=self.CTL)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1
        assert "scope-check-weakened" in problems[0] and "recorded as unstable" in problems[0]


class TestUnstable:
    """``{"unstable": "<reason>"}``: a mutant whose outcome differs between
    runs of the same source and tests. It never regresses, is never
    promotable, and is excluded from the score."""

    REASON = "killed or survived with the hash seed: socket order comes from a set"

    def test_unstable_mutant_that_survives_passes_while_a_stable_one_still_regresses(self):
        """NEGATIVE and POSITIVE in one run, function hashes unchanged: the
        unstable mutant's survival is not a failure, the stable one's is."""
        res = _results({M1: "survived", M2: "survived", M3: "killed"})
        base = _baseline({M1: "killed", M2: {"unstable": self.REASON}, M3: "killed"},
                         floor=50.0)
        problems, warnings, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1, problems
        assert "recorded as killed are no longer detected" in problems[0]
        assert M1 in problems[0] and M2 not in problems[0]
        assert warnings == []

    def test_unstable_mutant_is_excluded_from_the_score(self):
        res = _results({M1: "killed", M2: "survived"})
        base = _baseline({M1: "killed", M2: {"unstable": self.REASON}}, floor=100.0)
        problems, warnings, rows = cmf.check(res, base, [MOD])
        assert problems == [] and warnings == []
        assert rows[0]["score"] == 100.0 and rows[0]["scored"] == 1
        assert rows[0]["unstable"] == 1 and rows[0]["equivalent"] == 0

    def test_killed_unstable_mutant_is_not_promotable(self):
        res = _results({M1: "killed", M2: "killed"})
        base = _baseline({M1: "killed", M2: {"unstable": self.REASON}}, floor=100.0)
        problems, warnings, _ = cmf.check(res, base, [MOD])
        assert problems == [] and warnings == []

    @pytest.mark.parametrize("value", [
        {"unstable": ""}, {"unstable": "   "}, {"unstable": None}, {"unstable": 3},
        {"unstable": "x", "equivalent": "y"}, "unstable", "UNSTABLE",
    ])
    def test_unstable_without_a_reason_or_in_any_other_shape_is_invalid(self, value):
        res = _results({M1: "killed", M2: "survived"})
        base = _baseline({M1: "killed", M2: value}, floor=0.0)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert any("invalid disposition" in p and M2 in p for p in problems), problems

    def test_unstable_entry_of_a_changed_function_is_stale(self):
        res = _results({M1: "killed", M2: "survived", M3: "killed"},
                       hashes={"x_grants": "changed000000", "x_normalize": "bbbbbbbbbbbb"})
        base = _baseline({M1: "killed", M2: {"unstable": self.REASON}, M3: "killed"},
                         floor=66.66)
        problems, _, _ = cmf.check(res, base, [MOD])
        assert len(problems) == 1 and "no disposition" in problems[0] and "x_grants" in problems[0]

    def test_update_keeps_an_unstable_entry_and_leaves_it_out_of_the_floor(self):
        res = _results({M1: "killed", M2: "survived", M3: "survived"})
        base = _baseline({M1: "killed", M2: {"unstable": self.REASON}, M3: "survived"},
                         floor=10.0)
        prop = cmf.propose_baseline(res, base, [MOD])
        entry = prop["modules"][MOD]
        assert entry["mutants"][M2] == {"unstable": self.REASON}
        assert entry["score_floor"] == 50.0
        assert cmf.check(res, prop, [MOD])[0] == []

    def test_update_re_evaluates_an_unstable_entry_of_a_changed_function(self):
        res = _results({M1: "killed", M2: "killed"},
                       hashes={"x_grants": "changed000000", "x_normalize": "bbbbbbbbbbbb"})
        base = _baseline({M1: "killed", M2: {"unstable": self.REASON}})
        prop = cmf.propose_baseline(res, base, [MOD])
        assert prop["modules"][MOD]["mutants"][M2] == "killed"


# --------------------------------------------------------------------------- #
#  propose_baseline(): --update                                               #
# --------------------------------------------------------------------------- #

class TestProposeBaseline:
    def test_dispositions_follow_outcomes_and_floor_is_rounded_down(self):
        res = _results({M1: "killed", M2: "survived", M3: "timeout"})
        prop = cmf.propose_baseline(res, {"schema": 1, "modules": {}}, [MOD])
        entry = prop["modules"][MOD]
        assert entry["mutants"] == {M1: "killed", M2: "survived", M3: "killed"}
        assert entry["score_floor"] == 66.66
        assert entry["function_hashes"] == HASHES

    def test_floor_ratchets_up_but_never_down(self):
        res = _results({M1: "killed", M2: "survived"})
        higher = _baseline({M1: "killed", M2: "killed"}, floor=100.0)
        assert cmf.propose_baseline(res, higher, [MOD])["modules"][MOD]["score_floor"] == 100.0
        lower = _baseline({M1: "killed", M2: "survived"}, floor=10.0)
        assert cmf.propose_baseline(res, lower, [MOD])["modules"][MOD]["score_floor"] == 50.0

    def test_equivalent_classification_and_reason_survive_an_update(self):
        res = _results({M1: "killed", M2: "survived"})
        base = _baseline({M1: "killed", M2: {"equivalent": "message text only"}}, floor=100.0)
        prop = cmf.propose_baseline(res, base, [MOD])
        assert prop["modules"][MOD]["mutants"][M2] == {"equivalent": "message text only"}
        assert prop["modules"][MOD]["score_floor"] == 100.0

    def test_equivalent_on_a_changed_function_is_re_evaluated(self):
        res = _results({M1: "killed", M2: "survived"},
                       hashes={"x_grants": "changed000000", "x_normalize": "bbbbbbbbbbbb"})
        base = _baseline({M1: "killed", M2: {"equivalent": "was true before the edit"}})
        prop = cmf.propose_baseline(res, base, [MOD])
        assert prop["modules"][MOD]["mutants"][M2] == "survived"

    def test_mutants_of_a_removed_function_and_incomplete_ones_are_pruned(self):
        res = _results({M1: "killed", M2: "skipped", M3: "not checked"},
                       hashes={"x_grants": HASHES["x_grants"]})
        base = _baseline({M1: "killed", "localm.scopes.x_normalize__mutmut_7": "killed"})
        prop = cmf.propose_baseline(res, base, [MOD])
        assert prop["modules"][MOD]["mutants"] == {M1: "killed", M2: "survived"}

    def test_a_vanished_mutant_of_an_unchanged_function_keeps_its_entry(self):
        """--update never silently drops the entry a `pragma: no mutate` orphaned;
        the check keeps failing on it until a human classifies or deletes it."""
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed", M2: "survived"})
        prop = cmf.propose_baseline(res, base, [MOD])
        assert prop["modules"][MOD]["mutants"] == {M1: "killed", M2: "survived"}
        assert cmf.check(res, prop, [MOD])[0]

    def test_controls_and_mutmut_version_are_carried_over(self):
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed"}, controls=TestControls.CTL)
        prop = cmf.propose_baseline(res, base, [MOD])
        assert prop["controls"] == TestControls.CTL and prop["mutmut"] == "3.7.0"

    def test_module_without_results_keeps_its_old_entry(self):
        res = _results({M1: "killed"})
        base = _baseline({M1: "killed"})
        base["modules"]["localm/other.py"] = {"score_floor": 42.0, "function_hashes": {}, "mutants": {}}
        prop = cmf.propose_baseline(res, base, [MOD, "localm/other.py"])
        assert prop["modules"]["localm/other.py"]["score_floor"] == 42.0

    def test_proposal_passes_its_own_check(self):
        res = _results({M1: "killed", M2: "survived", M3: "no tests"})
        prop = cmf.propose_baseline(res, {"schema": 1, "modules": {}}, [MOD])
        problems, warnings, _ = cmf.check(res, prop, [MOD])
        assert problems == [] and warnings == []


# --------------------------------------------------------------------------- #
#  main(): the CLI                                                             #
# --------------------------------------------------------------------------- #

class TestMain:
    def _setup(self, tmp_path, exit_codes, baseline):
        mutants = tmp_path / "mutants"
        _write_meta(mutants, MOD, exit_codes)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(PYPROJECT, encoding="utf-8")
        base = tmp_path / "baseline.json"
        if baseline is not None:
            base.write_text(json.dumps(baseline), encoding="utf-8")
        return ["--mutants", str(mutants), "--baseline", str(base), "--pyproject", str(pyproject)]

    def test_passing_run_exits_zero_and_prints_the_table(self, tmp_path, capsys):
        argv = self._setup(tmp_path, {M1: 1, M2: 0}, _baseline({M1: "killed", M2: "survived"}, floor=50.0))
        assert cmf.main(argv) == 0
        out = capsys.readouterr().out
        assert "localm/scopes.py" in out and "50.00%" in out and "passed" in out

    def test_regression_exits_one_and_names_the_mutant(self, tmp_path, capsys):
        argv = self._setup(tmp_path, {M1: 0, M2: 0}, _baseline({M1: "killed", M2: "survived"}, floor=50.0))
        assert cmf.main(argv) == 1
        err = capsys.readouterr().err
        assert "FAILED" in err and M1 in err and "BELOW its floor" in err

    def test_surviving_unstable_mutant_exits_zero_and_is_counted_in_the_table(self, tmp_path, capsys):
        base = _baseline({M1: "killed", M2: {"unstable": TestUnstable.REASON}}, floor=100.0)
        argv = self._setup(tmp_path, {M1: 1, M2: 0}, base)
        assert cmf.main(argv) == 0
        out = capsys.readouterr().out
        header, row = out.splitlines()[:2]
        assert row.split()[header.split().index("unst")] == "1"

    def test_missing_baseline_exits_one(self, tmp_path, capsys):
        argv = self._setup(tmp_path, {M1: 1}, None)
        assert cmf.main(argv) == 1
        assert "Generate one with --update" in capsys.readouterr().err

    def test_missing_results_exit_one_and_never_read_as_clean(self, tmp_path, capsys):
        argv = self._setup(tmp_path, {M1: 1}, _baseline({M1: "killed"}))
        argv[1] = str(tmp_path / "nowhere")
        assert cmf.main(argv) == 1
        assert "no mutation results" in capsys.readouterr().err

    def test_malformed_baseline_exits_one(self, tmp_path, capsys):
        argv = self._setup(tmp_path, {M1: 1}, None)
        Path(argv[3]).write_text("[1, 2", encoding="utf-8")
        assert cmf.main(argv) == 1
        assert "not valid JSON" in capsys.readouterr().err

    def test_update_writes_a_baseline_that_then_passes(self, tmp_path, capsys):
        argv = self._setup(tmp_path, {M1: 1, M2: 0}, None)
        assert cmf.main(argv + ["--update"]) == 0
        written = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
        assert written["modules"][MOD]["mutants"] == {M1: "killed", M2: "survived"}
        assert written["modules"][MOD]["score_floor"] == 50.0
        assert cmf.main(argv) == 0

    def test_update_out_leaves_the_baseline_untouched_and_still_reports_the_check(self, tmp_path, capsys):
        base = _baseline({M1: "killed", M2: "killed"}, floor=100.0)
        argv = self._setup(tmp_path, {M1: 1, M2: 0}, base)
        out = tmp_path / "proposed.json"
        assert cmf.main(argv + ["--update", "--out", str(out)]) == 1
        assert json.loads(Path(argv[3]).read_text(encoding="utf-8")) == base
        proposed = json.loads(out.read_text(encoding="utf-8"))
        assert proposed["modules"][MOD]["mutants"][M2] == "survived"
        assert proposed["modules"][MOD]["score_floor"] == 100.0
        err = capsys.readouterr().err
        assert "BELOW its floor" in err and "recorded as killed are no longer detected" in err

    def test_update_out_checks_the_committed_baseline_not_the_proposal(self, tmp_path, capsys):
        """NEGATIVE, the CI invocation: a compensated regression (one kill lost,
        one gained, score unchanged) and an edited function are invisible to a
        check run against the proposal, which by construction matches every
        current outcome. The gate must read the committed file."""
        base = _baseline({M1: "killed", M2: "survived", M3: "killed"}, floor=66.66)
        argv = self._setup(tmp_path, {M1: 0, M2: 1, M3: 1}, base)
        out = tmp_path / "proposed.json"
        assert cmf.main(argv + ["--update", "--out", str(out)]) == 1
        err = capsys.readouterr().err
        assert M1 in err and "recorded as killed are no longer detected" in err
        assert "BELOW its floor" not in err
        # An edited function: its mutants carry no valid disposition until the
        # committed baseline is updated by hand.
        _write_meta(tmp_path / "mutants", MOD, {M1: 1, M2: 0, M3: 1},
                    hashes={"x_grants": "edited000000", "x_normalize": HASHES["x_normalize"]})
        assert cmf.main(argv + ["--update", "--out", str(out)]) == 1
        assert "no disposition" in capsys.readouterr().err

    def test_step_summary_is_appended_when_github_env_is_set(self, tmp_path, capsys, monkeypatch):
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        argv = self._setup(tmp_path, {M1: 0, M2: 0}, _baseline({M1: "killed", M2: "survived"}, floor=50.0))
        assert cmf.main(argv) == 1
        text = summary.read_text(encoding="utf-8")
        assert "### Mutation floors" in text and "**FAILED**" in text and M1 in text

    def test_empty_only_mutate_exits_one(self, tmp_path, capsys):
        argv = self._setup(tmp_path, {M1: 1}, _baseline({M1: "killed"}))
        Path(argv[5]).write_text("[tool.mutmut]\nonly_mutate = []\n", encoding="utf-8")
        assert cmf.main(argv) == 1
        assert "only_mutate is empty" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
#  The committed baseline                                                      #
# --------------------------------------------------------------------------- #

class TestCommittedBaseline:
    """Consistency of scripts/mutation_baseline.json with the real
    pyproject.toml: the check in CI reads both, so a stale or malformed
    baseline would fail there for the wrong reason."""

    @pytest.fixture(scope="class")
    @classmethod
    def baseline(cls):
        return json.loads((SCRIPTS / "mutation_baseline.json").read_text(encoding="utf-8"))

    @pytest.fixture(scope="class")
    @classmethod
    def modules(cls):
        return cmf.only_mutate_modules((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_covers_exactly_the_only_mutate_modules(self, baseline, modules):
        assert baseline["schema"] == cmf.SCHEMA_VERSION
        assert set(baseline["modules"]) == set(modules)
        assert len(modules) == 8

    def test_every_disposition_is_valid_and_every_floor_is_earned(self, baseline):
        for module, entry in baseline["modules"].items():
            statuses = {}
            excluded = set()
            for mutant, value in entry["mutants"].items():
                kind, reason = cmf._disposition(value)
                assert kind != "invalid", f"{module}: {mutant} -> {value!r}"
                assert cmf.function_of(mutant) in entry["function_hashes"], (module, mutant)
                if kind in cmf.EXCLUDED_KINDS:
                    excluded.add(mutant)
                    assert len(reason) >= 20, f"{module}: {mutant} reason too thin: {reason!r}"
                statuses[mutant] = kind
            score, _, _ = cmf.module_score(statuses, excluded)
            floor = entry["score_floor"]
            assert 0.0 <= floor <= 100.0
            assert floor <= score + 1e-9, (
                f"{module}: floor {floor} exceeds the score its own dispositions "
                f"imply ({score:.2f}); the baseline cannot pass its own check")

    def test_ci_shards_are_exactly_the_only_mutate_modules(self, modules):
        """The mutation-run matrix in ci.yml is the only_mutate list by
        basename: a module added to one and not the other is either never
        mutation-tested or a shard that mutates nothing."""
        import yaml
        wf = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
        shards = wf["jobs"]["mutation-run"]["strategy"]["matrix"]["module"]
        assert sorted(shards) == sorted(Path(m).stem for m in modules)
        run_step = next(st for st in wf["jobs"]["mutation-run"]["steps"]
                        if st.get("name", "").startswith("Mutation test"))
        assert run_step["run"].endswith(' "localm.${MUTATION_MODULE}.*"')
        assert "--max-children" in run_step["run"]
        gate = wf["jobs"]["mutation-test"]
        assert gate["needs"] == ["mutation-run"]
        ratchet = [st for st in gate["steps"] if "check_mutation_floors.py" in (st.get("run") or "")]
        assert len(ratchet) == 1
        # --out keeps the proposal out of the check: the gate reads the
        # committed baseline, never a file synthesized from the same run.
        assert "--out mutants/mutation_baseline.proposed.json" in ratchet[0]["run"]
        assert "--baseline" not in ratchet[0]["run"]
        # The shards never key on mutation-scope: a PR is never made to wait.
        assert "mutation-scope" not in wf["jobs"]["mutation-run"]["if"]
        scope_step = next(st for st in wf["jobs"]["mutation-scope"]["steps"]
                          if "mutation_scope.py" in (st.get("run") or ""))
        assert "--notice" in scope_step["run"] and "--github-output" not in scope_step["run"]
        # The notice says the gate runs exactly when mutation-run's label clause holds.
        label_clause = "contains(github.event.pull_request.labels.*.name, 'mutation-test')"
        assert label_clause in " ".join(wf["jobs"]["mutation-run"]["if"].split())
        assert '--gate-runs "$MUTATION_TEST_LABEL"' in scope_step["run"]
        assert scope_step["env"]["MUTATION_TEST_LABEL"] == "${{ %s }}" % label_clause
        assert "${{" not in scope_step["run"]

    def test_every_sec01_control_class_is_pinned_to_a_killed_mutant(self, baseline):
        """The baseline's controls are exactly the controls of the six control
        mutant classes that live inside the only_mutate modules, each pinned to
        a killed mutant: a weakened scope check, a deny-to-allow fallback, a
        skipped SSRF redirect re-validation, a widened net_mode=off exemption, a
        path-confinement bypass, and a loopback classifier that accepts an
        unparseable host. The two classes that live in
        localm/inference/http_server.py (an unsafe route exempted from the origin
        gate, bind_host replaced by the peer address) are not mutmut mutants and
        are not pinned here."""
        expected = {
            "scope-check-weakened",
            "state-changing-fallback-deny-to-allow",
            "ssrf-redirect-revalidation-skipped", "ssrf-redirect-revalidation-skipped-2",
            "off-floor-exemption-default-widened", "off-floor-exemption-or",
            "off-floor-downloads-default-true",
            "path-confinement-bypassed", "path-traversal-check-inverted",
            "bind-host-loopback-classifier-fallback",
        }
        controls = baseline["controls"]
        assert set(controls) == expected, (
            f"missing: {sorted(expected - set(controls))}, "
            f"not listed here: {sorted(set(controls) - expected)}")
        for name, ctl in controls.items():
            entry = baseline["modules"][ctl["module"]]
            assert entry["mutants"].get(ctl["mutant"]) == "killed", (name, ctl)
            assert cmf.function_of(ctl["mutant"]) in entry["function_hashes"], name
            assert ctl.get("describes"), name
