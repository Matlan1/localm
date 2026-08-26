# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the GUI file/folder picker backend and its host-access gate.

- GET /api/fs/dirs / /api/fs/places require HOST filesystem access
  (effective_fs_access == "host"): owner / open mode / a key minted with
  fs_access=host. A merely config:read key cannot enumerate the disk.
- The listing itself: meta=true entries[] with size/mtime, include_files gating,
  hidden-file/dir exclusion by default plus the include_hidden opt-in,
  404, the large-listing cap (`truncated`), and no symlink-follow for metadata.
- /api/fs/places: home + only the standard subfolders that exist, plus a drive.
- The fs_access attribute round-trips on a key and surfaces via /api/capabilities.
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
    /api/fs/places resolves against a directory tree we control. No owner key by
    default, so a minted key's own fs_access governs its reach."""
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


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _host_key():
    """A non-owner key that DOES have host filesystem access."""
    from localm import auth
    return auth.create_key("h", [S.CONFIG_READ], fs_access="host")["key"]


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


# --------------------------------------------------------------------------- #
#  Host-access gate                                                            #
# --------------------------------------------------------------------------- #

class TestHostAccessGate:
    def test_config_read_key_without_host_access_is_403(self, fs_app):
        from localm import auth
        # Default fs_access is "none" -> no host browse, even with config:read.
        none_key = auth.create_key("n", [S.CONFIG_READ])["key"]
        with TestClient(fs_app) as c:
            assert c.get("/api/fs/dirs", headers=_hdr(none_key)).status_code == 403
            assert c.get("/api/fs/places", headers=_hdr(none_key)).status_code == 403

    def test_host_access_key_reaches_the_browser(self, fs_app, tmp_path):
        with TestClient(fs_app) as c:
            key = _host_key()
            assert c.get("/api/fs/dirs", params={"path": str(tmp_path)},
                         headers=_hdr(key)).status_code == 200
            assert c.get("/api/fs/places", headers=_hdr(key)).status_code == 200

    def test_owner_key_reaches_the_browser(self, fs_app, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")   # owner = admin = host
        with TestClient(fs_app) as c:
            assert c.get("/api/fs/dirs", params={"path": str(tmp_path)},
                         headers=_hdr("ownersecret")).status_code == 200

    def test_open_mode_reaches_the_browser(self, fs_app, tmp_path):
        # No key configured at all -> loopback owner -> host access.
        with TestClient(fs_app) as c:
            assert c.get("/api/fs/dirs", params={"path": str(tmp_path)}).status_code == 200


# --------------------------------------------------------------------------- #
#  Listing behaviour (with a host-access key)                                 #
# --------------------------------------------------------------------------- #

class TestListing:
    def test_meta_returns_entries_with_size_and_kind(self, fs_app, tree):
        with TestClient(fs_app) as c:
            key = _host_key()
            r = c.get("/api/fs/dirs",
                      params={"path": str(tree), "include_files": "true", "meta": "true"},
                      headers=_hdr(key))
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["dirs"] == ["sub"]
            assert body["files"] == ["a.txt", "b.md"]
            assert body["truncated"] is False
            entries = {e["name"]: e for e in body["entries"]}
            assert set(entries) == {"sub", "a.txt", "b.md"}   # ".hidden" excluded
            assert entries["sub"]["is_dir"] is True
            assert entries["a.txt"]["size"] == 5              # len("hello")
            assert entries["a.txt"]["mtime"] is not None
            assert entries["sub"]["size"] is None             # dirs carry no faked size

    def test_include_hidden_reveals_dotdirs_and_dotfiles(self, fs_app, tmp_path):
        """Dot-directories (and dot-files) are invisible by default, matching
        plain `ls`. The GUI picker always requests include_hidden=true and
        shows its own toggle client-side, so the server has to actually have
        the entries to give it when asked - this covers a hidden DIRECTORY,
        which the `tree` fixture above never exercised (it only had a hidden
        file)."""
        d = tmp_path / "withdots"
        d.mkdir()
        (d / "visible_dir").mkdir()
        (d / ".hidden_dir").mkdir()
        (d / "visible.txt").write_text("x", encoding="utf-8")
        (d / ".hidden.txt").write_text("x", encoding="utf-8")
        with TestClient(fs_app) as c:
            key = _host_key()
            params = {"path": str(d), "include_files": "true", "meta": "true"}
            r = c.get("/api/fs/dirs", params=params, headers=_hdr(key))
            assert r.status_code == 200, r.text
            names = {e["name"] for e in r.json()["entries"]}
            assert names == {"visible_dir", "visible.txt"}, \
                "default must still exclude dot-entries (back-compat)"

            r2 = c.get("/api/fs/dirs", params={**params, "include_hidden": "true"},
                        headers=_hdr(key))
            assert r2.status_code == 200, r2.text
            body2 = r2.json()
            names2 = {e["name"] for e in body2["entries"]}
            assert names2 == {"visible_dir", "visible.txt", ".hidden_dir", ".hidden.txt"}
            by_name = {e["name"]: e for e in body2["entries"]}
            assert by_name[".hidden_dir"]["is_dir"] is True
            assert ".hidden_dir" in body2["dirs"]
            assert ".hidden.txt" in body2["files"]

    def test_without_meta_is_backcompat(self, fs_app, tree):
        with TestClient(fs_app) as c:
            r = c.get("/api/fs/dirs",
                      params={"path": str(tree), "include_files": "true"},
                      headers=_hdr(_host_key()))
            assert r.status_code == 200, r.text
            assert "entries" not in r.json()

    def test_without_include_files_lists_only_folders(self, fs_app, tree):
        with TestClient(fs_app) as c:
            r = c.get("/api/fs/dirs", params={"path": str(tree), "meta": "true"},
                      headers=_hdr(_host_key()))
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["files"] == []
            assert all(e["is_dir"] for e in body["entries"])

    def test_404_on_missing_path(self, fs_app, tmp_path):
        with TestClient(fs_app) as c:
            r = c.get("/api/fs/dirs", params={"path": str(tmp_path / "nope")},
                      headers=_hdr(_host_key()))
            assert r.status_code == 404

    def test_large_listing_is_capped_and_flagged(self, fs_app, tmp_path, monkeypatch):
        import localm.plugins.gui.routes.admin as admin
        monkeypatch.setattr(admin, "_FS_LIST_CAP", 3)
        d = tmp_path / "many"
        d.mkdir()
        for i in range(6):
            (d / f"f{i}.txt").write_text("x", encoding="utf-8")
        with TestClient(fs_app) as c:
            r = c.get("/api/fs/dirs",
                      params={"path": str(d), "include_files": "true", "meta": "true"},
                      headers=_hdr(_host_key()))
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["truncated"] is True
            assert len(body["files"]) == 3            # examined-cap, not silent
            assert len(body["entries"]) == 3

    def test_meta_does_not_follow_symlinks(self, fs_app, tmp_path):
        target = tmp_path / "big.bin"
        target.write_bytes(b"x" * 4096)
        d = tmp_path / "linkdir"
        d.mkdir()
        link = d / "lnk"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted on this platform/privilege level")
        with TestClient(fs_app) as c:
            r = c.get("/api/fs/dirs",
                      params={"path": str(d), "include_files": "true", "meta": "true"},
                      headers=_hdr(_host_key()))
            assert r.status_code == 200, r.text
            entries = {e["name"]: e for e in r.json()["entries"]}
            # The link's OWN metadata, never the 4096-byte target's.
            assert entries["lnk"]["size"] != 4096


# --------------------------------------------------------------------------- #
#  Places                                                                     #
# --------------------------------------------------------------------------- #

def test_places_lists_home_and_existing_subfolders_only(fs_app, tmp_path):
    (tmp_path / "Documents").mkdir()      # exists -> should appear
    # Desktop / Downloads absent, so they must not be guessed.
    with TestClient(fs_app) as c:
        r = c.get("/api/fs/places", headers=_hdr(_host_key()))
        assert r.status_code == 200, r.text
        body = r.json()
        by_label = {p["label"]: p for p in body["places"]}
        assert by_label["Home"]["path"] == str(tmp_path)
        assert by_label["Documents"]["path"] == str(tmp_path / "Documents")
        assert "Desktop" not in by_label and "Downloads" not in by_label
        assert body["drives"], "at least one drive/root"
        for dr in body["drives"]:
            assert dr["path"] and dr["label"] and dr["icon"] == "drive"


# --------------------------------------------------------------------------- #
#  fs_access attribute + capabilities                                         #
# --------------------------------------------------------------------------- #

class TestFsAccessAttribute:
    def test_create_key_defaults_to_none_and_round_trips(self, fs_app):
        from localm import auth
        auth.create_key("n", [S.CONFIG_READ])
        auth.create_key("h", [S.CONFIG_READ], fs_access="host")
        by_name = {k["name"]: k for k in auth.list_keys()}
        assert by_name["n"]["fs_access"] == "none"     # safe default
        assert by_name["h"]["fs_access"] == "host"

    def test_unknown_fs_access_is_coerced_to_none(self):
        from localm import auth
        assert auth.norm_fs_access("bogus") == "none"
        assert auth.norm_fs_access(None) == "none"
        assert auth.norm_fs_access("HOST") == "host"

    def test_capabilities_reports_fs_access(self, fs_app):
        from localm import auth
        none_key = auth.create_key("n", [S.CONFIG_READ])["key"]
        host_key = auth.create_key("h", [S.CONFIG_READ], fs_access="host")["key"]
        with TestClient(fs_app) as c:
            assert c.get("/api/capabilities",
                         headers=_hdr(none_key)).json()["fs_access"] == "none"
            assert c.get("/api/capabilities",
                         headers=_hdr(host_key)).json()["fs_access"] == "host"

    def test_capabilities_open_mode_is_host(self, fs_app):
        with TestClient(fs_app) as c:      # no key configured -> open/loopback owner
            assert c.get("/api/capabilities").json()["fs_access"] == "host"
