# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/write_coverage_summary.py - the GitHub Actions step
summary writer for the coverage.json the Tests step already produces.

Pure, git- and pytest-cov-free: every test feeds a synthetic coverage.json /
pyproject.toml rather than a real coverage run.
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


wcs = _load("write_coverage_summary")


# --------------------------------------------------------------------------- #
#  render_summary: the pure markdown-building core                           #
# --------------------------------------------------------------------------- #

class TestRenderSummary:
    def test_shows_measured_percent_and_positive_headroom(self):
        text = wcs.render_summary("Windows", 79.53, 78, None)
        assert "### Coverage (Windows)" in text
        assert "79.53%" in text
        assert "78%" in text
        assert "+1.53" in text

    def test_shows_negative_headroom_when_below_floor(self):
        """A below-floor run is not hidden or clamped - the actual gate
        (pytest-cov's own fail_under) already failed the step; this only
        has to report the number honestly."""
        text = wcs.render_summary("Linux", 70.0, 78, None)
        assert "-8.00" in text

    def test_reports_floor_unavailable_when_fail_under_is_none(self):
        text = wcs.render_summary("Windows", 79.53, None, None)
        assert "floor unavailable" in text
        assert "79.53%" in text

    def test_no_module_table_when_rows_are_none(self):
        text = wcs.render_summary("Linux", 75.0, 78, None)
        assert "trust-boundary" not in text
        assert "|" not in text

    def test_no_module_table_when_rows_are_empty(self):
        text = wcs.render_summary("Linux", 75.0, 78, [])
        assert "trust-boundary" not in text

    def test_module_table_renders_measured_and_absent_rows(self):
        rows = [
            ("localm/bindhost.py", 100.0, 100),
            ("localm/portmux.py", 97.8022, 96),
            ("localm/config.py", None, 83),
        ]
        text = wcs.render_summary("Windows", 82.0, 78, rows)
        assert "| `localm/bindhost.py` | 100.00% | 100% | +0.00 |" in text
        assert "| `localm/portmux.py` | 97.80% | 96% | +1.80 |" in text
        assert "| `localm/config.py` | absent | 83% | - |" in text

    def test_output_is_a_single_trailing_newline_terminated_block(self):
        """GITHUB_STEP_SUMMARY is appended to across steps - each publish must
        not run into the next without a separating newline."""
        text = wcs.render_summary("Windows", 79.53, 78, None)
        assert text.endswith("\n")
        assert not text.endswith("\n\n")


# --------------------------------------------------------------------------- #
#  main(): the CLI/file-I/O wrapper                                           #
# --------------------------------------------------------------------------- #

class TestMainPublishing:
    def _summary_file(self, tmp_path, monkeypatch) -> Path:
        target = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
        return target

    def test_missing_coverage_json_publishes_a_note_and_exits_zero(
            self, tmp_path, monkeypatch):
        summary = self._summary_file(tmp_path, monkeypatch)
        monkeypatch.delenv("RUNNER_OS", raising=False)
        missing = tmp_path / "coverage.json"
        rc = wcs.main([str(missing)])
        assert rc == 0
        assert "No coverage report" in summary.read_text(encoding="utf-8")

    def test_malformed_json_publishes_a_note_and_exits_zero(self, tmp_path, monkeypatch):
        summary = self._summary_file(tmp_path, monkeypatch)
        bad = tmp_path / "coverage.json"
        bad.write_text("{not json", encoding="utf-8")
        rc = wcs.main([str(bad)])
        assert rc == 0
        assert "Could not read" in summary.read_text(encoding="utf-8")

    def test_missing_totals_key_publishes_a_note_and_exits_zero(self, tmp_path, monkeypatch):
        """A real coverage.json always has 'totals'; this guards against a
        truncated write or an incompatible coverage.py version rather than
        crashing an otherwise-green job over a display step."""
        summary = self._summary_file(tmp_path, monkeypatch)
        p = tmp_path / "coverage.json"
        p.write_text(json.dumps({"files": {}}), encoding="utf-8")
        rc = wcs.main([str(p)])
        assert rc == 0
        assert "no totals.percent_covered" in summary.read_text(encoding="utf-8")

    def test_no_github_step_summary_env_falls_back_to_stdout(
            self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        missing = tmp_path / "coverage.json"
        rc = wcs.main([str(missing)])
        assert rc == 0
        assert "No coverage report" in capsys.readouterr().out

    def test_real_totals_are_published_with_platform_from_runner_os(
            self, tmp_path, monkeypatch):
        summary = self._summary_file(tmp_path, monkeypatch)
        monkeypatch.setenv("RUNNER_OS", "Linux")
        p = tmp_path / "coverage.json"
        p.write_text(json.dumps({"totals": {"percent_covered": 79.53}, "files": {}}),
                     encoding="utf-8")
        rc = wcs.main([str(p)])
        out = summary.read_text(encoding="utf-8")
        assert rc == 0
        assert "### Coverage (Linux)" in out
        assert "79.53%" in out

    def test_linux_never_shows_the_module_table_even_if_files_present(
            self, tmp_path, monkeypatch):
        """Per-module floors are measured on Windows only (see
        scripts/check_coverage_floors.py) - showing them against a Linux
        measurement would compare a real number to a floor that was never
        derived for this platform, which reads as a near-miss that is not one."""
        summary = self._summary_file(tmp_path, monkeypatch)
        monkeypatch.setenv("RUNNER_OS", "Linux")
        p = tmp_path / "coverage.json"
        p.write_text(json.dumps({
            "totals": {"percent_covered": 75.0},
            "files": {"localm/bindhost.py": {"summary": {"percent_covered": 100.0}}},
        }), encoding="utf-8")
        rc = wcs.main([str(p)])
        out = summary.read_text(encoding="utf-8")
        assert rc == 0
        assert "trust-boundary" not in out

    def test_windows_shows_the_real_module_table_via_check_coverage_floors(
            self, tmp_path, monkeypatch):
        """Integration with the real, shipped scripts/check_coverage_floors.py
        (not a fake): proves the sys.path wiring actually finds it and its
        real module names/floors reach the rendered table."""
        summary = self._summary_file(tmp_path, monkeypatch)
        monkeypatch.setenv("RUNNER_OS", "Windows")
        p = tmp_path / "coverage.json"
        p.write_text(json.dumps({
            "totals": {"percent_covered": 82.0},
            "files": {"localm/bindhost.py": {"summary": {"percent_covered": 100.0}}},
        }), encoding="utf-8")
        rc = wcs.main([str(p)])
        out = summary.read_text(encoding="utf-8")
        assert rc == 0
        assert "trust-boundary" in out
        assert "`localm/bindhost.py`" in out
        # A real floored module absent from this synthetic report renders as
        # absent rather than being silently dropped from the table.
        assert "`localm/scopes.py`" in out
        assert "absent" in out

    def test_fail_under_is_read_from_pyproject_toml(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text(
            "[tool.coverage.report]\nfail_under = 65\n", encoding="utf-8")
        monkeypatch.setattr(wcs, "REPO", repo)
        summary = self._summary_file(tmp_path, monkeypatch)
        monkeypatch.setenv("RUNNER_OS", "Linux")
        p = repo / "coverage.json"
        p.write_text(json.dumps({"totals": {"percent_covered": 70.0}, "files": {}}),
                     encoding="utf-8")
        rc = wcs.main([])
        out = summary.read_text(encoding="utf-8")
        assert rc == 0
        assert "65%" in out
        assert "+5.00" in out

    def test_missing_pyproject_toml_degrades_to_floor_unavailable_not_a_crash(
            self, tmp_path, monkeypatch):
        """FIRES-CONTROL for the best-effort fail_under read: without a
        pyproject.toml at all, main() must still publish the measured number
        rather than raising."""
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setattr(wcs, "REPO", repo)
        summary = self._summary_file(tmp_path, monkeypatch)
        monkeypatch.setenv("RUNNER_OS", "Linux")
        p = repo / "coverage.json"
        p.write_text(json.dumps({"totals": {"percent_covered": 70.0}, "files": {}}),
                     encoding="utf-8")
        rc = wcs.main([])
        out = summary.read_text(encoding="utf-8")
        assert rc == 0
        assert "70.00%" in out
        assert "floor unavailable" in out

    def test_malformed_pyproject_toml_degrades_to_floor_unavailable_not_a_crash(
            self, tmp_path, monkeypatch):
        """Same fires-control as the missing-file case, for a pyproject.toml
        that exists but fails to parse (tomllib.TOMLDecodeError)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("not valid toml {{{", encoding="utf-8")
        monkeypatch.setattr(wcs, "REPO", repo)
        summary = self._summary_file(tmp_path, monkeypatch)
        monkeypatch.setenv("RUNNER_OS", "Linux")
        p = repo / "coverage.json"
        p.write_text(json.dumps({"totals": {"percent_covered": 70.0}, "files": {}}),
                     encoding="utf-8")
        rc = wcs.main([])
        out = summary.read_text(encoding="utf-8")
        assert rc == 0
        assert "70.00%" in out
        assert "floor unavailable" in out

    def test_appends_rather_than_truncates_an_existing_summary_file(
            self, tmp_path, monkeypatch):
        """GITHUB_STEP_SUMMARY accumulates across every step in a job - this
        step must not erase whatever an earlier step already wrote."""
        summary = self._summary_file(tmp_path, monkeypatch)
        summary.write_text("### Earlier step\n\nsomething\n", encoding="utf-8")
        p = tmp_path / "coverage.json"
        p.write_text(json.dumps({"totals": {"percent_covered": 79.53}, "files": {}}),
                     encoding="utf-8")
        rc = wcs.main([str(p)])
        out = summary.read_text(encoding="utf-8")
        assert rc == 0
        assert "### Earlier step" in out
        assert "### Coverage" in out
