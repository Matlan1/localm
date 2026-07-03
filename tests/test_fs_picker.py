# SPDX-License-Identifier: AGPL-3.0-or-later
"""Functional tests for the GUI file/folder picker backend:

- GET /api/fs/dirs  - directory listing, the meta=true entries[] with size/mtime,
  include_files gating, hidden-file exclusion, 404 on a non-directory, and the
  back-compat shape (no entries[] unless meta=true).
- GET /api/fs/places - the Places rail: home + only the standard subfolders that
  actually exist, plus at least one drive/root.

Both are gated on CONFIG_READ (scope enforcement itself is covered in
test_key_scope_gui.py; here we exercise the responses with a CONFIG_READ key).
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.gui.web import attach_gui


@pytest.fixture
def fs_app(tmp_path, monkeypatch):
    """GUI stack on a throwaway home whose Path.home() is tmp_path, so
    /api/fs/places resolves against a directory tree we control."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "model-a")
    return app


def _reader():
    from localm import auth
    return {"Authorization": f"Bearer {auth.create_key('r', [S.CONFIG_READ])['key']}"}


@pytest.fixture
def tree(tmp_path):
    """A small directory to list: one subdir, two indexable files, a hidden one."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "sub").mkdir()
    (d / "a.txt").write_text("hello", encoding="utf-8")
    (d / "b.md").write_text("# hi", encoding="utf-8")
    (d / ".hidden").write_text("x", encoding="utf-8")
    return d


def test_dirs_meta_returns_entries_with_size_and_kind(fs_app, tree):
    with TestClient(fs_app) as c:
        r = c.get("/api/fs/dirs", params={"path": str(tree),
                                          "include_files": "true", "meta": "true"},
                  headers=_reader())
        assert r.status_code == 200, r.text
        body = r.json()
        # Flat arrays stay for older callers; folders and files are separate.
        assert body["dirs"] == ["sub"]
        assert body["files"] == ["a.txt", "b.md"]
        # meta=true adds the richer entries[].
        entries = {e["name"]: e for e in body["entries"]}
        assert set(entries) == {"sub", "a.txt", "b.md"}   # hidden ".hidden" excluded
        assert entries["sub"]["is_dir"] is True
        assert entries["a.txt"]["is_dir"] is False
        assert entries["a.txt"]["size"] == 5              # len("hello")
        assert entries["a.txt"]["mtime"] is not None
        # Directories carry no size (None), not a faked 0.
        assert entries["sub"]["size"] is None


def test_dirs_without_meta_is_backcompat(fs_app, tree):
    with TestClient(fs_app) as c:
        r = c.get("/api/fs/dirs", params={"path": str(tree), "include_files": "true"},
                  headers=_reader())
        assert r.status_code == 200, r.text
        body = r.json()
        assert "entries" not in body, "entries[] only appears with meta=true"
        assert body["files"] == ["a.txt", "b.md"]


def test_dirs_without_include_files_lists_only_folders(fs_app, tree):
    with TestClient(fs_app) as c:
        r = c.get("/api/fs/dirs", params={"path": str(tree), "meta": "true"},
                  headers=_reader())
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["files"] == []
        assert all(e["is_dir"] for e in body["entries"]), "only folders when files are off"


def test_dirs_404_on_missing_path(fs_app, tmp_path):
    with TestClient(fs_app) as c:
        r = c.get("/api/fs/dirs", params={"path": str(tmp_path / "nope")},
                  headers=_reader())
        assert r.status_code == 404


def test_places_lists_home_and_existing_subfolders_only(fs_app, tmp_path):
    (tmp_path / "Documents").mkdir()      # exists -> should appear
    # Desktop / Downloads deliberately absent -> must NOT be guessed.
    with TestClient(fs_app) as c:
        r = c.get("/api/fs/places", headers=_reader())
        assert r.status_code == 200, r.text
        body = r.json()
        by_label = {p["label"]: p for p in body["places"]}
        assert by_label["Home"]["path"] == str(tmp_path)
        assert by_label["Documents"]["path"] == str(tmp_path / "Documents")
        assert "Desktop" not in by_label and "Downloads" not in by_label
        # At least one drive/root, each well-formed.
        assert body["drives"], "at least one drive/root"
        for d in body["drives"]:
            assert d["path"] and d["label"] and d["icon"] == "drive"


def test_places_requires_config_read_scope(fs_app):
    from localm import auth
    narrow = {"Authorization": f"Bearer {auth.create_key('n', [S.MCP])['key']}"}
    with TestClient(fs_app) as c:
        assert c.get("/api/fs/places", headers=narrow).status_code == 403
