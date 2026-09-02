# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/write_mutation_summary.py - the GitHub Actions step
summary writer for the mutmut-cicd-stats.json the mutation-test job produces.

Pure, mutmut-free: every test feeds a synthetic mutmut-cicd-stats.json rather
than a real mutation run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wms = _load("write_mutation_summary")


# --------------------------------------------------------------------------- #
#  accounted_for: distinguishes a completed run from an aborted one          #
# --------------------------------------------------------------------------- #

class TestAccountedFor:
    def test_true_when_total_equals_the_sum_of_every_outcome(self):
        assert wms.accounted_for({"killed": 33, "survived": 5, "total": 38})

    def test_false_when_mutants_were_generated_but_none_were_tested(self):
        """mutmut's own pytest run stopped at a failing test before mutation
        testing began: mutants exist (total > 0) but every outcome is 0."""
        assert not wms.accounted_for({"killed": 0, "survived": 0, "total": 2513})

    def test_true_when_total_is_zero_and_nothing_was_generated(self):
        assert wms.accounted_for({"killed": 0, "survived": 0, "total": 0})

    def test_missing_keys_default_to_zero_on_both_sides(self):
        assert wms.accounted_for({})


# --------------------------------------------------------------------------- #
#  render_summary: the pure markdown-building core                           #
# --------------------------------------------------------------------------- #

class TestRenderSummary:
    def test_incomplete_run_is_flagged_instead_of_shown_as_clean(self):
        """0 survived out of a run that never actually tested anything must
        never read as a clean result."""
        text = wms.render_summary({"killed": 0, "survived": 0, "total": 2513})
        assert "Incomplete run" in text
        assert "2513" in text
        assert "0.0%" not in text
        assert "mutation score" not in text

    def test_shows_score_and_killed_over_total(self):
        text = wms.render_summary({"killed": 33, "survived": 5, "total": 38})
        assert "### Mutation testing (mutmut)" in text
        assert "86.8%" in text
        assert "33/38" in text

    def test_survivors_get_their_own_callout_line(self):
        text = wms.render_summary({"killed": 33, "survived": 5, "total": 38})
        assert "**5** mutant(s) survived" in text

    def test_no_survivor_callout_when_zero_survived(self):
        text = wms.render_summary({"killed": 38, "survived": 0, "total": 38})
        assert "mutant(s) survived" not in text

    def test_zero_total_reports_no_mutants_tested_and_skips_the_table(self):
        text = wms.render_summary({"killed": 0, "survived": 0, "total": 0})
        assert "No mutants were tested." in text
        assert "|" not in text

    def test_outcome_table_includes_every_key_even_when_absent_from_input(self):
        text = wms.render_summary({"killed": 1, "survived": 0, "total": 1})
        for key in ("killed", "survived", "no_tests", "skipped", "suspicious",
                    "timeout", "check_was_interrupted_by_user", "segfault"):
            assert f"`{key}`" in text

    def test_output_is_a_single_trailing_newline_terminated_block(self):
        """GITHUB_STEP_SUMMARY is appended to across steps - each publish must
        not run into the next without a separating newline."""
        text = wms.render_summary({"killed": 33, "survived": 5, "total": 38})
        assert text.endswith("\n")
        assert not text.endswith("\n\n")


# --------------------------------------------------------------------------- #
#  main(): the CLI/file-I/O wrapper                                          #
# --------------------------------------------------------------------------- #

class TestMainPublishing:
    def _summary_file(self, tmp_path, monkeypatch) -> Path:
        target = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
        return target

    def test_missing_stats_json_publishes_a_note_and_exits_zero(self, tmp_path, monkeypatch):
        summary = self._summary_file(tmp_path, monkeypatch)
        missing = tmp_path / "mutmut-cicd-stats.json"
        rc = wms.main([str(missing)])
        assert rc == 0
        assert "No stats report" in summary.read_text(encoding="utf-8")

    def test_malformed_json_publishes_a_note_and_exits_zero(self, tmp_path, monkeypatch):
        summary = self._summary_file(tmp_path, monkeypatch)
        bad = tmp_path / "mutmut-cicd-stats.json"
        bad.write_text("{not json", encoding="utf-8")
        rc = wms.main([str(bad)])
        assert rc == 0
        assert "Could not read" in summary.read_text(encoding="utf-8")

    def test_non_object_json_publishes_a_note_and_exits_zero(self, tmp_path, monkeypatch):
        """A real mutmut-cicd-stats.json is always a JSON object. A truncated
        write or an incompatible mutmut version publishes a note and exits
        zero rather than crashing the job."""
        summary = self._summary_file(tmp_path, monkeypatch)
        p = tmp_path / "mutmut-cicd-stats.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        rc = wms.main([str(p)])
        assert rc == 0
        assert "did not contain a JSON object" in summary.read_text(encoding="utf-8")

    def test_no_github_step_summary_env_falls_back_to_stdout(
            self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        missing = tmp_path / "mutmut-cicd-stats.json"
        rc = wms.main([str(missing)])
        assert rc == 0
        assert "No stats report" in capsys.readouterr().out

    def test_real_stats_are_published(self, tmp_path, monkeypatch):
        summary = self._summary_file(tmp_path, monkeypatch)
        p = tmp_path / "mutmut-cicd-stats.json"
        p.write_text(json.dumps({
            "killed": 33, "survived": 5, "total": 38, "no_tests": 0,
            "skipped": 0, "suspicious": 0, "timeout": 0,
            "check_was_interrupted_by_user": 0, "segfault": 0,
        }), encoding="utf-8")
        rc = wms.main([str(p)])
        out = summary.read_text(encoding="utf-8")
        assert rc == 0
        assert "86.8%" in out
        assert "33/38" in out

    def test_appends_rather_than_truncates_an_existing_summary_file(
            self, tmp_path, monkeypatch):
        """GITHUB_STEP_SUMMARY accumulates across every step in a job - this
        step must not erase whatever an earlier step already wrote."""
        summary = self._summary_file(tmp_path, monkeypatch)
        summary.write_text("### Earlier step\n\nsomething\n", encoding="utf-8")
        p = tmp_path / "mutmut-cicd-stats.json"
        p.write_text(json.dumps({"killed": 1, "survived": 0, "total": 1}),
                     encoding="utf-8")
        rc = wms.main([str(p)])
        out = summary.read_text(encoding="utf-8")
        assert rc == 0
        assert "### Earlier step" in out
        assert "### Mutation testing" in out

    def test_incomplete_run_still_exits_zero(self, tmp_path, monkeypatch):
        """main() is a display step and must never fail the build, even when
        the run it is describing did not complete."""
        summary = self._summary_file(tmp_path, monkeypatch)
        p = tmp_path / "mutmut-cicd-stats.json"
        p.write_text(json.dumps({"killed": 0, "survived": 0, "total": 2513}),
                     encoding="utf-8")
        rc = wms.main([str(p)])
        assert rc == 0
        assert "Incomplete run" in summary.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  check(): the separate gating entry point                                  #
# --------------------------------------------------------------------------- #

class TestCheck:
    def test_exits_1_when_stats_file_is_missing(self, tmp_path):
        missing = tmp_path / "mutmut-cicd-stats.json"
        assert wms.check([str(missing)]) == 1

    def test_exits_1_on_malformed_json(self, tmp_path):
        p = tmp_path / "mutmut-cicd-stats.json"
        p.write_text("{not json", encoding="utf-8")
        assert wms.check([str(p)]) == 1

    def test_exits_1_on_a_non_object_json(self, tmp_path):
        p = tmp_path / "mutmut-cicd-stats.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert wms.check([str(p)]) == 1

    def test_exits_1_when_the_run_did_not_complete(self, tmp_path):
        p = tmp_path / "mutmut-cicd-stats.json"
        p.write_text(json.dumps({"killed": 0, "survived": 0, "total": 2513}),
                     encoding="utf-8")
        assert wms.check([str(p)]) == 1

    def test_exits_0_when_the_run_completed_regardless_of_score(self, tmp_path):
        """A completed run with a LOW score (many survivors, few killed) is
        still a real, accounted-for result - check() gates completeness, not
        the mutation score itself."""
        p = tmp_path / "mutmut-cicd-stats.json"
        p.write_text(json.dumps({"killed": 1, "survived": 37, "total": 38}),
                     encoding="utf-8")
        assert wms.check([str(p)]) == 0

    def test_exits_0_when_nothing_was_generated_at_all(self, tmp_path):
        p = tmp_path / "mutmut-cicd-stats.json"
        p.write_text(json.dumps({"killed": 0, "survived": 0, "total": 0}),
                     encoding="utf-8")
        assert wms.check([str(p)]) == 0
