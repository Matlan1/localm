# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm rag add` with a path that does not exist says so, exits non-zero,
and leaves no collection behind."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    home = tmp_path / "userhome"
    localm = home / ".localm"
    localm.mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
    return home


def _add(*args):
    from localm.cli import rag as ragcli
    return CliRunner().invoke(ragcli.rag_add, list(args))


def test_a_missing_path_is_reported_and_exits_non_zero(env, tmp_path):
    from localm.rag import collection_names
    missing = tmp_path / "no" / "such" / "file.txt"

    result = _add("fresh", str(missing))

    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert str(missing) in " ".join(result.output.split())
    assert "0 added" not in result.output
    assert "fresh" not in collection_names()


def test_one_missing_path_among_real_ones_adds_nothing(env, tmp_path):
    from localm.rag import Collection, collection_names
    real = tmp_path / "real.txt"
    real.write_text("zebrafinch", encoding="utf-8")
    missing = tmp_path / "gone.txt"

    result = _add("fresh", str(real), str(missing))

    assert result.exit_code == 1
    assert "gone.txt" in result.output
    assert "fresh" not in collection_names()
    assert Collection("fresh").documents() == []


def test_existing_paths_still_add(env, tmp_path):
    from localm.rag import Collection
    real = tmp_path / "real.txt"
    real.write_text("zebrafinch lives here", encoding="utf-8")

    result = _add("fresh", str(real))

    assert result.exit_code == 0, result.output
    assert "1 added" in result.output
    assert len(Collection("fresh").documents()) == 1
