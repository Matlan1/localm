# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/check_coverage_floors.py - the per-module trust-boundary
coverage ratchet that complements pyproject.toml's global fail_under.

Pure, git- and pytest-cov-free: every test feeds a synthetic coverage.json
dict (or a file holding one) rather than a real coverage run, so the
regression-detection logic itself is proven without needing --cov here.
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


ccf = _load("check_coverage_floors")


def _entry(percent: float) -> dict:
    """A minimal coverage.json 'files' entry carrying just what check_floors reads."""
    return {"summary": {"percent_covered": percent}}


# --------------------------------------------------------------------------- #
#  check_floors: the pure regression-detection core                          #
# --------------------------------------------------------------------------- #

class TestCheckFloors:
    def test_passes_when_every_module_meets_its_floor(self):
        floors = {"localm/a.py": 90, "localm/b.py": 100}
        report = {"files": {"localm/a.py": _entry(90.0), "localm/b.py": _entry(100.0)}}
        assert ccf.check_floors(report, floors) == []

    def test_fails_when_a_module_is_below_its_floor(self):
        """NEGATIVE: a real drop must be caught."""
        floors = {"localm/a.py": 90}
        report = {"files": {"localm/a.py": _entry(85.5)}}
        problems = ccf.check_floors(report, floors)
        assert len(problems) == 1
        assert "localm/a.py" in problems[0]
        assert "85.50" in problems[0]
        assert "90" in problems[0]

    def test_passes_at_exactly_the_floor(self):
        """The comparison is '<', not '<=': a module sitting exactly on its
        floor must not fail - only a genuine drop BELOW it should."""
        floors = {"localm/a.py": 90}
        report = {"files": {"localm/a.py": _entry(90.0)}}
        assert ccf.check_floors(report, floors) == []

    def test_fails_when_a_floored_module_is_missing_from_the_report(self):
        """A module absent from coverage.json entirely (never imported/exercised
        by the run) is a worse signal than a low score and must not pass silently."""
        floors = {"localm/a.py": 90}
        report = {"files": {}}
        problems = ccf.check_floors(report, floors)
        assert len(problems) == 1
        assert "not present" in problems[0]

    def test_windows_backslash_keys_are_matched_not_reported_missing(self):
        """coverage.json's file keys come from the file_reporter's
        relative_filename(), which uses the HOST's native path separator. A real
        Windows --cov run produces backslash-separated keys, not the forward
        slash _MODULE_FLOORS uses, so an unnormalised comparison reads every
        floored module as "not present" on the one platform this check actually
        runs on - a gate that LOOKS like it is checking coverage while checking
        nothing."""
        floors = {"localm/pathsafe.py": 90}
        report = {"files": {"localm\\pathsafe.py": _entry(95.0)}}
        assert ccf.check_floors(report, floors) == []

    def test_reports_one_problem_per_offending_module_not_just_the_first(self):
        floors = {"localm/a.py": 90, "localm/b.py": 90}
        report = {"files": {"localm/a.py": _entry(50.0), "localm/b.py": _entry(60.0)}}
        problems = ccf.check_floors(report, floors)
        assert len(problems) == 2

    def test_real_module_floors_pass_against_their_own_measured_baseline(self):
        """Each shipped floor is one point below the ACTUAL value from a real
        --cov run on merged master, taken from a worktree: a rounded display
        integer is not safe to floor against directly, and config.py branches on
        the gitignored `home/` and reads lower wherever it exists. Feeding that
        exact baseline back in must pass, which proves the shipped dict is
        internally consistent with its own convention rather than with a rounded
        approximation of it."""
        baseline = {
            "localm/bindhost.py": 100.0,
            "localm/scopes.py": 100.0,
            "localm/pathsafe.py": 93.1034,
            "localm/netpolicy.py": 91.5865,
            "localm/auth.py": 91.0112,
            "localm/tls.py": 88.3041,
            "localm/config.py": 84.9315,
            "localm/portmux.py": 97.8022,
        }
        report = {"files": {m: _entry(p) for m, p in baseline.items()}}
        assert ccf.check_floors(report) == []

    def test_every_shipped_floor_is_one_point_below_its_measured_baseline(self):
        """The CONVENTION itself, asserted as arithmetic rather than trusted to
        stay true by hand: every floor is one point below its measured
        baseline. Without this, a floor can be bumped to a rounded display
        integer, or left far below its module's real value, and every other
        test here still passes.

        bindhost/scopes are the documented exception: pinned AT 100 because
        coverage.py only ever displays 100 for exactly 100.0."""
        baseline = {
            "localm/bindhost.py": 100.0,
            "localm/scopes.py": 100.0,
            "localm/pathsafe.py": 93.1034,
            "localm/netpolicy.py": 91.5865,
            "localm/auth.py": 91.0112,
            "localm/tls.py": 88.3041,
            "localm/config.py": 84.9315,
            "localm/portmux.py": 97.8022,
        }
        assert set(baseline) == set(ccf._MODULE_FLOORS), (
            "the baseline and the shipped floor table have drifted apart - a "
            "module was added or removed without re-measuring")
        for module, measured in baseline.items():
            floor = ccf._MODULE_FLOORS[module]
            expected = 100 if measured == 100.0 else int(measured) - 1
            assert floor == expected, (
                f"{module}: floor is {floor}, but its measured {measured}% "
                f"means the convention gives {expected}")


# --------------------------------------------------------------------------- #
#  main(): the CLI/file-I/O wrapper                                           #
# --------------------------------------------------------------------------- #

class TestMain:
    def test_missing_coverage_json_fails_with_a_clear_message(self, tmp_path, capsys):
        missing = tmp_path / "coverage.json"
        rc = ccf.main([str(missing)])
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_malformed_json_fails_rather_than_crashing(self, tmp_path, capsys):
        bad = tmp_path / "coverage.json"
        bad.write_text("{not json", encoding="utf-8")
        rc = ccf.main([str(bad)])
        assert rc == 1
        assert "not valid JSON" in capsys.readouterr().err

    def test_passing_report_exits_zero_and_names_the_module_count(self, tmp_path, capsys):
        report = {"files": {m: _entry(100.0) for m in ccf._MODULE_FLOORS}}
        p = tmp_path / "coverage.json"
        p.write_text(json.dumps(report), encoding="utf-8")
        rc = ccf.main([str(p)])
        out = capsys.readouterr().out
        assert rc == 0
        assert str(len(ccf._MODULE_FLOORS)) in out

    def test_failing_report_exits_one_and_names_module_floor_and_actual(self, tmp_path, capsys):
        """The failure message contract this script promises: a gate must name
        what failed, not just that something did."""
        report = {"files": {m: _entry(0.0) for m in ccf._MODULE_FLOORS}}
        p = tmp_path / "coverage.json"
        p.write_text(json.dumps(report), encoding="utf-8")
        rc = ccf.main([str(p)])
        err = capsys.readouterr().err
        assert rc == 1
        for module, floor in ccf._MODULE_FLOORS.items():
            assert module in err
            assert str(floor) in err


# --------------------------------------------------------------------------- #
#  report_rows / --report: the floor-maintenance view                         #
# --------------------------------------------------------------------------- #

class TestReportRows:
    def test_returns_measured_value_and_floor_per_module_in_table_order(self):
        floors = {"localm/a.py": 90, "localm/b.py": 50}
        report = {"files": {"localm/a.py": _entry(95.5), "localm/b.py": _entry(60.25)}}
        assert ccf.report_rows(report, floors) == [
            ("localm/a.py", 95.5, 90),
            ("localm/b.py", 60.25, 50),
        ]

    def test_absent_module_reports_none_rather_than_a_fabricated_zero(self):
        """Absent and zero are different facts: zero means the module was
        measured and nothing ran, None means the run never saw it at all.
        Collapsing them would hide the second behind a plausible-looking 0%."""
        floors = {"localm/a.py": 90}
        assert ccf.report_rows({"files": {}}, floors) == [("localm/a.py", None, 90)]

    def test_windows_backslash_keys_are_matched_here_too(self):
        """Same host-separator hazard check_floors carries. Reported separately
        because a report that silently showed every module ABSENT on Windows
        would send someone hunting a coverage collapse that never happened."""
        floors = {"localm/pathsafe.py": 90}
        report = {"files": {"localm\\pathsafe.py": _entry(93.5)}}
        assert ccf.report_rows(report, floors) == [("localm/pathsafe.py", 93.5, 90)]


class TestReportFlag:
    def _write(self, tmp_path, percents: dict) -> Path:
        p = tmp_path / "coverage.json"
        p.write_text(json.dumps({"files": {m: _entry(v) for m, v in percents.items()}}),
                     encoding="utf-8")
        return p

    def test_report_prints_measured_values_alongside_floors(self, tmp_path, capsys):
        p = self._write(tmp_path, {m: 100.0 for m in ccf._MODULE_FLOORS})
        rc = ccf.main(["--report", str(p)])
        out = capsys.readouterr().out
        assert rc == 0
        for module in ccf._MODULE_FLOORS:
            assert module in out
        assert "measured" in out and "headroom" in out

    def test_report_does_not_suppress_a_real_failure(self, tmp_path, capsys):
        """--report is additive: it prints and then enforces. A reporting flag
        that also skipped the check would be an always-green gate one CI edit
        away, and the green would look identical to a real pass."""
        p = self._write(tmp_path, {m: 0.0 for m in ccf._MODULE_FLOORS})
        rc = ccf.main(["--report", str(p)])
        captured = capsys.readouterr()
        assert rc == 1
        assert "measured" in captured.out          # the report still printed
        assert "BELOW its floor" in captured.err   # and the check still failed

    def test_report_names_a_floor_that_has_drifted_far_below_reality(
            self, tmp_path, capsys, monkeypatch):
        """The stranding this flag exists to prevent: a floor left far under its
        module's real value protects nothing, and nothing else surfaces it.

        Uses a synthetic one-module floor table rather than the shipped one, so
        the test states the PROPERTY (a wide gap is named, with the bump to
        make) instead of pinning it to whichever module happens to be stale
        today - the shipped table is meant to have no stale entry at rest."""
        monkeypatch.setattr(ccf, "_MODULE_FLOORS", {"localm/example.py": 36})
        p = self._write(tmp_path, {"localm/example.py": 98.0})
        rc = ccf.main(["--report", str(p)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "no longer protecting" in out
        assert "-> 97" in out                      # int(98.0) - 1, the bump to make

    def test_a_floor_at_its_intended_one_point_gap_is_not_called_stale(self, tmp_path, capsys):
        """The shipped convention is floor = int(measured) - 1, so an ordinary
        well-maintained floor sits just under two points below its module's
        value and must NOT be flagged as stale. A notice that fired on every
        module would read exactly like one that fired on none."""
        p = self._write(tmp_path, {m: float(f) + 1.5 for m, f in ccf._MODULE_FLOORS.items()})
        rc = ccf.main(["--report", str(p)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "no longer protecting" not in out

    def test_report_flag_works_without_an_explicit_path_argument(self, tmp_path, capsys, monkeypatch):
        """--report must not be mistaken for the positional coverage.json path;
        stripping the flag has to leave the default-path branch reachable."""
        monkeypatch.setattr(ccf, "REPO", tmp_path)
        self._write(tmp_path, {m: 100.0 for m in ccf._MODULE_FLOORS})
        rc = ccf.main(["--report"])
        assert rc == 0
        assert "headroom" in capsys.readouterr().out


class TestHomeDirConfounderNote:
    """The failure message invites lowering a floor. For config.py there is one
    known way to read BELOW its floor without any regression - measuring from a
    checkout that has the gitignored `home/` directory, which config.py's
    data-dir resolution branches on. The note exists so nobody follows that
    invitation and weakens a floor that was never actually breached."""

    def _cov(self, tmp_path, percents: dict) -> Path:
        p = tmp_path / "coverage.json"
        p.write_text(json.dumps({"files": {m: _entry(v) for m, v in percents.items()}}),
                     encoding="utf-8")
        return p

    def test_note_appears_when_config_is_low_and_home_exists(
            self, tmp_path, capsys, monkeypatch):
        repo = tmp_path / "repo"
        (repo / "home").mkdir(parents=True)
        monkeypatch.setattr(ccf, "REPO", repo)
        monkeypatch.setattr(ccf, "_MODULE_FLOORS", {"localm/config.py": 83})
        p = self._cov(tmp_path, {"localm/config.py": 82.42})
        rc = ccf.main([str(p)])
        err = capsys.readouterr().err
        assert rc == 1                      # still a failure, never suppressed
        assert "`home/` directory" in err
        assert "Re-measure from a worktree" in err

    def test_no_note_without_a_home_dir(self, tmp_path, capsys, monkeypatch):
        """In a worktree or CI clone there is no `home/`, so a config.py failure
        there is a REAL regression and must not be softened by a hint that
        points at the wrong cause."""
        repo = tmp_path / "repo"
        repo.mkdir(parents=True)            # no home/ in this tree
        monkeypatch.setattr(ccf, "REPO", repo)
        monkeypatch.setattr(ccf, "_MODULE_FLOORS", {"localm/config.py": 83})
        p = self._cov(tmp_path, {"localm/config.py": 82.42})
        rc = ccf.main([str(p)])
        err = capsys.readouterr().err
        assert rc == 1
        assert "`home/` directory" not in err

    def test_no_note_for_an_unrelated_module_even_with_home_present(
            self, tmp_path, capsys, monkeypatch):
        """The confounder is measured for config.py only. Printing it on every
        failure would put an excuse in front of unrelated real regressions."""
        repo = tmp_path / "repo"
        (repo / "home").mkdir(parents=True)
        monkeypatch.setattr(ccf, "REPO", repo)
        monkeypatch.setattr(ccf, "_MODULE_FLOORS", {"localm/auth.py": 90})
        p = self._cov(tmp_path, {"localm/auth.py": 10.0})
        rc = ccf.main([str(p)])
        err = capsys.readouterr().err
        assert rc == 1
        assert "`home/` directory" not in err
