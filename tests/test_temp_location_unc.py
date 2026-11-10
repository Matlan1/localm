# SPDX-License-Identifier: AGPL-3.0-or-later
"""A UNC or device path stored as temp_location is not used as the temp folder."""
from __future__ import annotations

import json

import pytest

from localm import config


@pytest.mark.parametrize("value", ["\\\\attacker.example\\share\\tmp",
                                   "\\\\?\\C:\\tmp", "\\\\.\\PhysicalDrive0"])
def test_a_stored_unc_or_device_temp_location_falls_back_to_auto(
        tmp_path, monkeypatch, value):
    monkeypatch.delenv(config.TEMP_DIR_ENV, raising=False)
    (tmp_path / "config.json").write_text(
        json.dumps({config.TEMP_LOCATION_KEY: value}), encoding="utf-8")
    assert config.temp_dir(tmp_path, warn=False) in (None, tmp_path / "tmp")
    assert config.temp_dir(tmp_path, warn=False) != config._absolute_folder(value)


def test_a_stored_local_absolute_temp_location_is_used(tmp_path, monkeypatch):
    monkeypatch.delenv(config.TEMP_DIR_ENV, raising=False)
    target = tmp_path / "mytmp"
    (tmp_path / "config.json").write_text(
        json.dumps({config.TEMP_LOCATION_KEY: str(target)}), encoding="utf-8")
    assert config.temp_dir(tmp_path, warn=False) == target
