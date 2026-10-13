# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm info`` reports whether the install is portable and where temp files go."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import localm.cli.models as modelscli
from localm import config
from localm.cli import main

_REPO_ROOT = Path(config.__file__).resolve().parents[1]


@pytest.fixture
def info_output(monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.delenv(config.TEMP_DIR_ENV, raising=False)

    def run(home):
        monkeypatch.setattr(modelscli, "HOME_DIR", home)
        res = CliRunner().invoke(main, ["info"])
        assert res.exit_code == 0, res.output
        return res.output

    return run


def _squash(text: str) -> str:
    return "".join(text.split())


def _row(output: str, label: str) -> str:
    """The ``localm info`` row labelled *label*, with the console's wrapping and
    padding removed: the label line and the lines after it up to the next row."""
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == label or line.strip().startswith(label + " "):
            rest = []
            for follow in lines[i + 1:]:
                if not follow.strip() or follow.startswith("  "):
                    break
                rest.append(follow)
            return _squash(line + "".join(rest))
    raise AssertionError(f"no {label!r} row in: {output!r}")


class TestInfoPortableReport:
    def test_standard_install_reports_standard_and_system_temp(self, info_output, tmp_path):
        out = info_output(tmp_path / "data")
        assert _squash("standard") in _row(out, "mode")
        assert _squash("system temp folder") in _row(out, "temp dir")

    def test_portable_install_reports_portable_and_data_tmp(self, info_output):
        home = _REPO_ROOT / "portable-info-probe"
        out = info_output(home)
        assert _squash("portable") in _row(out, "mode")
        assert _squash(str(home / "tmp")) in _row(out, "temp dir")

    def test_temp_dir_override_is_reported(self, info_output, monkeypatch, tmp_path):
        override = tmp_path / "scratch"
        monkeypatch.setenv(config.TEMP_DIR_ENV, str(override))
        out = info_output(tmp_path / "data")
        assert _squash(str(override)) in _row(out, "temp dir")
