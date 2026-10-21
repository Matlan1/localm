# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm rag add` into a NEW collection leaves nothing behind when the path is
refused, and never removes a collection that already existed or holds documents."""

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


def test_a_refused_credential_folder_leaves_no_collection(env):
    from localm.rag import collection_names
    ssh = env / ".ssh"
    ssh.mkdir()
    (ssh / "id").write_text("key", encoding="utf-8")

    result = _add("fresh", str(ssh))

    assert result.exit_code == 1
    assert "credential" in result.output.lower()
    assert "fresh" not in collection_names()


def test_a_refused_unc_path_leaves_no_collection(env):
    from localm.rag import collection_names

    unc = chr(92) * 2 + "localhost" + chr(92) + "c$" + chr(92) + "x"

    result = _add("fresh", unc)

    assert result.exit_code == 1
    assert "fresh" not in collection_names()


def test_a_refused_add_keeps_a_collection_that_already_has_documents(env, tmp_path):
    from localm.rag import Collection, collection_names
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha turbines", encoding="utf-8")
    assert _add("kept", str(docs)).exit_code == 0
    ssh = env / ".ssh"
    ssh.mkdir()
    (ssh / "id").write_text("key", encoding="utf-8")

    result = _add("kept", str(ssh))

    assert result.exit_code == 1
    assert "kept" in collection_names()
    assert len(Collection("kept").documents()) == 1


def test_a_refused_add_keeps_a_collection_that_existed_empty(env):
    from localm.rag import Collection, collection_names
    Collection("preexisting").create()
    ssh = env / ".ssh"
    ssh.mkdir()
    (ssh / "id").write_text("key", encoding="utf-8")

    assert _add("preexisting", str(ssh)).exit_code == 1
    assert "preexisting" in collection_names()
