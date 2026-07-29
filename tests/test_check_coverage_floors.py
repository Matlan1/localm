# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/check_coverage_floors.py - the per-module trust-boundary
coverage ratchet that complements pyproject.toml's global fail_under.

Pure, git- and pytest-cov-free: every test feeds a synthetic coverage.json
dict (or a file holding one) rather than a real coverage run, so the
regression-detection logic itself is proven without needing --cov here.
"""

from __future__ import annotations

import importlib.util
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
        """NEGATIVE: this is the fires-control case - a real drop must be caught."""
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

    def test_reports_one_problem_per_offending_module_not_just_the_first(self):
        floors = {"localm/a.py": 90, "localm/b.py": 90}
        report = {"files": {"localm/a.py": _entry(50.0), "localm/b.py": _entry(60.0)}}
        problems = ccf.check_floors(report, floors)
        assert len(problems) == 2

    def test_real_module_floors_pass_against_their_own_measured_baseline(self):
        """Each shipped floor is defined as at-or-below its 2026-07-29 measurement
        (see the module docstring); feeding that exact baseline back in must pass -
        proves the shipped dict is internally consistent with its own rationale."""
        baseline = {
            "localm/bindhost.py": 100.0,
            "localm/scopes.py": 100.0,
            "localm/pathsafe.py": 93.0,
            "localm/netpolicy.py": 92.0,
            "localm/auth.py": 91.0,
            "localm/tls.py": 88.0,
            "localm/config.py": 82.0,
            "localm/portmux.py": 38.0,
        }
        report = {"files": {m: _entry(p) for m, p in baseline.items()}}
        assert ccf.check_floors(report) == []


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
        p.write_text(__import__("json").dumps(report), encoding="utf-8")
        rc = ccf.main([str(p)])
        out = capsys.readouterr().out
        assert rc == 0
        assert str(len(ccf._MODULE_FLOORS)) in out

    def test_failing_report_exits_one_and_names_module_floor_and_actual(self, tmp_path, capsys):
        """The failure message contract this script promises (AGENTS.md rule 5:
        a gate must name what failed, not just that something did)."""
        report = {"files": {m: _entry(0.0) for m in ccf._MODULE_FLOORS}}
        p = tmp_path / "coverage.json"
        p.write_text(__import__("json").dumps(report), encoding="utf-8")
        rc = ccf.main([str(p)])
        err = capsys.readouterr().err
        assert rc == 1
        for module, floor in ccf._MODULE_FLOORS.items():
            assert module in err
            assert str(floor) in err
