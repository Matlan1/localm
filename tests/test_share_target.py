# SPDX-License-Identifier: AGPL-3.0-or-later
"""PWA Web Share Target: a phone shares an image into localm via the OS share
sheet; it lands server-side and the app ingests it. Parsed without
python-multipart (localm stays self-contained), so this also guards the
hand-rolled multipart parser."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import attach_gui


@pytest.fixture
def share_client(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    app = FastAPI()

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return TestClient(app)


_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40


def test_share_image_roundtrip(share_client):
    r = share_client.post(
        "/share-target",
        files={"files": ("photo.png", _PNG, "image/png")},
        follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/?shared=1"

    items = share_client.get("/api/share/pending").json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "photo.png"
    assert items[0]["type"] == "image/png"
    assert items[0]["data_uri"].startswith("data:image/png;base64,")

    fid = items[0]["id"]
    r = share_client.post("/api/share/clear", json={"ids": [fid]})
    assert r.json()["removed"] == 1
    assert share_client.get("/api/share/pending").json()["items"] == []


def test_share_ignores_non_image_files(share_client):
    r = share_client.post(
        "/share-target",
        files={"files": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")},
        follow_redirects=False)
    assert r.headers["location"] == "/?shared=0"      # non-image skipped
    assert share_client.get("/api/share/pending").json()["items"] == []


def test_share_text_lands_as_a_note(share_client):
    # A shared link/text comes as form fields (no file part).
    r = share_client.post(
        "/share-target",
        data={"text": "look at this https://example.com"},
        files={"_": ("", b"", "application/octet-stream")},  # force multipart
        follow_redirects=False)
    assert r.headers["location"] == "/?shared=1"
    items = share_client.get("/api/share/pending").json()["items"]
    assert len(items) == 1 and items[0]["name"] == "shared.txt"


def test_share_clear_all_without_ids(share_client):
    for nm in ("a.png", "b.png"):
        share_client.post("/share-target",
                          files={"files": (nm, _PNG, "image/png")})
    assert len(share_client.get("/api/share/pending").json()["items"]) == 2
    r = share_client.post("/api/share/clear", json={})   # no ids -> clear all
    assert r.json()["removed"] == 2
    assert share_client.get("/api/share/pending").json()["items"] == []


def test_clear_id_cannot_escape_inbox(share_client, tmp_path):
    """A malicious clear id is matched as a filename prefix only - it cannot be
    turned into a path that deletes outside the inbox."""
    share_client.post("/share-target",
                      files={"files": ("keep.png", _PNG, "image/png")})
    victim = tmp_path / "victim.txt"
    victim.write_text("do not delete", encoding="utf-8")
    r = share_client.post("/api/share/clear",
                          json={"ids": ["../../victim", "..", "/etc/passwd"]})
    assert r.json()["removed"] == 0          # nothing matched, nothing escaped
    assert victim.is_file()                  # the outside file is untouched
    assert len(share_client.get("/api/share/pending").json()["items"]) == 1


# ------------------------------------------------------------------ #
#  Cross-principal ownership (HIGH-8 / #5 in CONSOLIDATED-FINDINGS)     #
# ------------------------------------------------------------------ #
#
# The open-mode share_client fixture above has no key at all, so ownership never
# comes into play there (owner=None is unrestricted, same as jobs). These tests
# mint two distinct scoped keys to prove one key's shared content is invisible to
# and undeletable by another.

def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _mk_keys(*scope_lists):
    from localm import auth
    return [auth.create_key(f"k{i}", s)["key"] for i, s in enumerate(scope_lists)]


@pytest.fixture
def share_app(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    app = FastAPI()

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return TestClient(app)


class TestShareInboxOwnership:
    def test_a_key_cannot_read_or_clear_b_keys_share(self, share_app):
        a, b = _mk_keys(["chat"], ["chat"])
        r = share_app.post("/share-target", headers=_h(a),
                           files={"files": ("photo.png", _PNG, "image/png")},
                           follow_redirects=False)
        assert r.status_code == 303

        # B (a different key) does not see A's share at all.
        assert share_app.get("/api/share/pending", headers=_h(b)).json()["items"] == []
        # B's "clear all" (no ids) removes nothing of A's.
        assert share_app.post("/api/share/clear", headers=_h(b), json={}).json()["removed"] == 0
        # A still sees it, untouched by B's clear-all.
        a_items = share_app.get("/api/share/pending", headers=_h(a)).json()["items"]
        assert len(a_items) == 1 and a_items[0]["name"] == "photo.png"

        # B cannot clear it even by guessing A's exact id.
        fid = a_items[0]["id"]
        assert share_app.post("/api/share/clear", headers=_h(b),
                              json={"ids": [fid]}).json()["removed"] == 0
        assert len(share_app.get("/api/share/pending", headers=_h(a)).json()["items"]) == 1

        # A can clear its own.
        assert share_app.post("/api/share/clear", headers=_h(a),
                              json={"ids": [fid]}).json()["removed"] == 1
        assert share_app.get("/api/share/pending", headers=_h(a)).json()["items"] == []

    def test_owner_admin_sees_every_share(self, share_app, monkeypatch):
        (a,) = _mk_keys(["chat"])
        share_app.post("/share-target", headers=_h(a),
                       files={"files": ("photo.png", _PNG, "image/png")})
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")   # owner = admin
        items = share_app.get("/api/share/pending", headers=_h("ownersecret")).json()["items"]
        assert any(it["name"] == "photo.png" for it in items)
