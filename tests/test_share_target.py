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


def _make_client(tmp_path, monkeypatch, mode):
    """A GUI app in a hermetic home, in session mode *mode*. Every disk-inbox
    test pins "log": the default test mode is privacy, which stages in memory."""
    monkeypatch.setenv("LOCALM_MODE", mode)
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


@pytest.fixture
def share_client(tmp_path, monkeypatch):
    return _make_client(tmp_path, monkeypatch, "log")


@pytest.fixture
def privacy_client(tmp_path, monkeypatch):
    return _make_client(tmp_path, monkeypatch, "privacy")


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


def _raw_share_body(*names, boundary=b"BOUND"):
    """A hand-built multipart body carrying *names* verbatim.

    httpx percent-encodes a NUL in the Content-Disposition it generates
    ("photo\\x00.png" goes out as "photo%00.png", an ordinary safe name), so
    `files=` cannot deliver one. A real client writes the header itself, and the
    route parses the body itself, so this is the reachable shape.
    """
    out = b""
    for nm in names:
        out += (b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="files"; filename="'
                + nm.encode() + b'"\r\n'
                b"Content-Type: image/png\r\n\r\n" + _PNG + b"\r\n")
    return out + b"--" + boundary + b"--\r\n"


def _post_raw(client, *names):
    return client.post(
        "/share-target", content=_raw_share_body(*names),
        headers={"Content-Type": "multipart/form-data; boundary=BOUND"},
        follow_redirects=False)


def _inbox():
    from localm.plugins.gui.web import _share_inbox
    return _share_inbox()


class TestShareFilenameGuard:
    """The shared name must clear the same lexical guard /api/upload applies.

    Path().name alone leaves "photo:stream.png" untouched, so the write lands in
    an NTFS alternate data stream: the listing shows a 0-byte "photo" and the
    payload is invisible to /api/share/pending. Not traversal (a uuid4 prefix
    bounds the path) - content smuggling.
    """

    def test_rejects_alternate_data_stream_name(self, share_client):
        r = share_client.post(
            "/share-target",
            files={"files": ("photo:stream.png", _PNG, "image/png")},
            follow_redirects=False)
        assert r.status_code == 400
        # Nothing on disk: an ADS write would still create a base directory
        # entry, so an empty inbox proves no stream was opened either.
        assert list(_inbox().iterdir()) == []
        assert share_client.get("/api/share/pending").json()["items"] == []

    def test_rejects_embedded_nul_name(self, share_client):
        r = _post_raw(share_client, "photo\x00.png")
        assert r.status_code == 400          # not a bare 500 out of write_bytes
        assert list(_inbox().iterdir()) == []

    def test_still_accepts_a_legitimate_name(self, share_client):
        """A guard that refused everything would look exactly as green as a
        correct one. This also shows _post_raw builds a request the route accepts,
        so the 400s above are the name, not the hand-built body."""
        r = _post_raw(share_client, "ok.png")
        assert r.status_code == 303
        assert r.headers["location"] == "/?shared=1"
        items = share_client.get("/api/share/pending").json()["items"]
        assert [i["name"] for i in items] == ["ok.png"]

    def test_one_bad_name_writes_none_of_the_batch(self, share_client):
        """Names are checked before any write, so a refused share cannot leave
        the good half of a multi-file share sitting in the inbox."""
        r = _post_raw(share_client, "good.png", "photo:stream.png")
        assert r.status_code == 400
        assert list(_inbox().iterdir()) == []


# ------------------------------------------------------------------ #
#  Cross-principal ownership                                          #
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
    return _make_client(tmp_path, monkeypatch, "log")


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


# --- a delete that FAILED must not read as a clean sweep ---------------- #
# share_clear reports a `failed` field, which is distinct from an entry the
# caller never asked about. chat.js reads that field, logs it and toasts the
# user.

def _inject_unlink_failure(monkeypatch, fail_on_name_containing: str):
    """Make Path.unlink raise OSError for matching entries only.

    The raising side_effect is the FAULT being injected, not an assertion.
    share_clear catches OSError by design, so an AssertionError raised from
    inside would be swallowed as an input and the test would pass in both
    directions.
    """
    real_unlink = Path.unlink

    def fake_unlink(self, *a, **kw):
        if fail_on_name_containing in self.name:
            raise OSError(13, "Permission denied")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", fake_unlink)


def test_share_clear_reports_a_delete_that_failed(share_client, monkeypatch):
    share_client.post("/share-target", files={"files": ("locked.png", _PNG, "image/png")})
    assert len(share_client.get("/api/share/pending").json()["items"]) == 1

    _inject_unlink_failure(monkeypatch, "locked")
    body = share_client.post("/api/share/clear", json={}).json()

    # Assert on the data first: if the injection failed to match, the entry is
    # gone and this fails on the deletion itself, not on a status code.
    still_there = share_client.get("/api/share/pending").json()["items"]
    assert len(still_there) == 1, "the entry was deleted, so no fault was injected"

    assert body["failed"] == 1, f"a failed delete was not reported: {body}"
    assert body["removed"] == 0


def test_share_clear_reports_zero_failed_on_a_clean_sweep(share_client):
    """The field is ALWAYS present, so a client can trust its absence-of-failure."""
    share_client.post("/share-target", files={"files": ("ok.png", _PNG, "image/png")})
    body = share_client.post("/api/share/clear", json={}).json()
    assert body["removed"] == 1
    assert body["failed"] == 0
    assert share_client.get("/api/share/pending").json()["items"] == []


def test_share_clear_partial_failure_reports_both_counts(share_client, monkeypatch):
    """The case the old code collapsed: some deleted, some not."""
    for nm in ("good.png", "bad.png"):
        share_client.post("/share-target", files={"files": (nm, _PNG, "image/png")})
    assert len(share_client.get("/api/share/pending").json()["items"]) == 2

    _inject_unlink_failure(monkeypatch, "bad")
    body = share_client.post("/api/share/clear", json={}).json()

    left = share_client.get("/api/share/pending").json()["items"]
    assert len(left) == 1 and "bad" in left[0]["name"], f"wrong entry survived: {left}"
    assert body["removed"] == 1
    assert body["failed"] == 1


def test_share_clear_logs_the_path_of_a_failed_delete(share_client, monkeypatch, caplog):
    """A count tells the user; the log tells whoever has to diagnose it.

    Asserted from OUTSIDE via caplog, not by raising inside the handler, which
    catches broadly and would absorb an assertion as an ordinary input.
    """
    import logging
    share_client.post("/share-target", files={"files": ("noisy.png", _PNG, "image/png")})
    _inject_unlink_failure(monkeypatch, "noisy")
    with caplog.at_level(logging.WARNING):
        share_client.post("/api/share/clear", json={})
    assert any("noisy" in r.getMessage() for r in caplog.records), \
        f"the failed path was not logged: {[r.getMessage() for r in caplog.records]}"


# --------------------------------------------------------------------------- #
#  Privacy mode: the share is staged in memory, never on disk.                 #
# --------------------------------------------------------------------------- #

def _disk_entries(tmp_path):
    inbox = tmp_path / ".localm" / "share_inbox"
    return sorted(p.name for p in inbox.glob("*__*")) if inbox.exists() else []


def test_privacy_share_never_touches_the_disk_inbox(privacy_client, tmp_path):
    r = privacy_client.post(
        "/share-target",
        files={"files": ("photo.png", _PNG, "image/png")},
        data={"text": "a note from the phone"},
        follow_redirects=False)
    # Assert on the data first: nothing staged on disk.
    assert _disk_entries(tmp_path) == [], "privacy mode wrote the share to disk"
    assert not (tmp_path / ".localm" / "share_inbox").exists(), \
        "privacy mode must not even create the inbox directory"
    assert r.status_code == 303
    assert r.headers["location"] == "/?shared=2"

    items = privacy_client.get("/api/share/pending").json()["items"]
    assert sorted(i["name"] for i in items) == ["photo.png", "shared.txt"]
    png = next(i for i in items if i["name"] == "photo.png")
    assert png["data_uri"].startswith("data:image/png;base64,")
    assert _disk_entries(tmp_path) == []

    body = privacy_client.post("/api/share/clear", json={"ids": [i["id"] for i in items]}).json()
    assert body == {"removed": 2, "failed": 0}
    assert privacy_client.get("/api/share/pending").json()["items"] == []


def test_privacy_share_clear_all_without_ids(privacy_client, tmp_path):
    for nm in ("a.png", "b.png"):
        privacy_client.post("/share-target", files={"files": (nm, _PNG, "image/png")})
    assert len(privacy_client.get("/api/share/pending").json()["items"]) == 2
    assert privacy_client.post("/api/share/clear", json={}).json()["removed"] == 2
    assert privacy_client.get("/api/share/pending").json()["items"] == []
    assert _disk_entries(tmp_path) == []


def test_privacy_share_refuses_a_bad_name_before_staging_anything(privacy_client, tmp_path):
    r = _post_raw(privacy_client, "ok.png", "photo:stream.png")
    assert r.status_code == 400
    assert privacy_client.get("/api/share/pending").json()["items"] == []
    assert _disk_entries(tmp_path) == []


def test_log_mode_share_still_stages_on_disk(share_client, tmp_path):
    """The control for the mode split: outside privacy mode the inbox is the
    disk directory, exactly as before."""
    share_client.post("/share-target", files={"files": ("photo.png", _PNG, "image/png")})
    entries = _disk_entries(tmp_path)
    assert len(entries) == 1 and entries[0].endswith("__photo.png")


class TestPrivacyShareInboxOwnership:
    def test_a_key_cannot_read_or_clear_b_keys_share_in_memory(self, privacy_client, tmp_path):
        a, b = _mk_keys(["chat"], ["chat"])
        r = privacy_client.post("/share-target", headers=_h(a),
                                files={"files": ("photo.png", _PNG, "image/png")},
                                follow_redirects=False)
        assert r.status_code == 303
        assert _disk_entries(tmp_path) == []

        assert privacy_client.get("/api/share/pending", headers=_h(b)).json()["items"] == []
        assert privacy_client.post("/api/share/clear", headers=_h(b),
                                   json={}).json()["removed"] == 0
        a_items = privacy_client.get("/api/share/pending", headers=_h(a)).json()["items"]
        assert len(a_items) == 1 and a_items[0]["name"] == "photo.png"

        fid = a_items[0]["id"]
        assert privacy_client.post("/api/share/clear", headers=_h(b),
                                   json={"ids": [fid]}).json()["removed"] == 0
        assert len(privacy_client.get("/api/share/pending", headers=_h(a)).json()["items"]) == 1

        assert privacy_client.post("/api/share/clear", headers=_h(a),
                                   json={"ids": [fid]}).json()["removed"] == 1
        assert privacy_client.get("/api/share/pending", headers=_h(a)).json()["items"] == []


# --------------------------------------------------------------------------- #
#  Expiry: an entry the app never ingested does not live forever.              #
# --------------------------------------------------------------------------- #

def _age(path, minutes):
    import os, time
    t = time.time() - minutes * 60
    os.utime(path, (t, t))


def test_disk_entry_older_than_the_ttl_is_swept_on_access(share_client, tmp_path):
    for nm in ("old.png", "fresh.png"):
        share_client.post("/share-target", files={"files": (nm, _PNG, "image/png")})
    inbox = tmp_path / ".localm" / "share_inbox"
    old = next(p for p in inbox.glob("*__old.png"))
    fresh = next(p for p in inbox.glob("*__fresh.png"))
    _age(old, 16)
    _age(fresh, 14)

    items = share_client.get("/api/share/pending").json()["items"]

    assert not old.exists(), "a 16 minute old entry must be removed on access"
    assert fresh.exists(), "a 14 minute old entry is still within the TTL"
    assert [i["name"] for i in items] == ["fresh.png"]


def test_stale_disk_entries_are_swept_at_startup(tmp_path, monkeypatch, caplog):
    import logging
    inbox = tmp_path / ".localm" / "share_inbox"
    inbox.mkdir(parents=True)
    stale = inbox / "deadbeef__-__old.png"
    stale.write_bytes(_PNG)
    _age(stale, 16)
    recent = inbox / "cafef00d__-__recent.png"
    recent.write_bytes(_PNG)

    with caplog.at_level(logging.INFO, logger="localm"):
        client = _make_client(tmp_path, monkeypatch, "log")   # register() runs the sweep

    assert not stale.exists(), "a stale entry from an earlier run must be swept at startup"
    assert recent.exists()
    assert any("deadbeef" in r.getMessage() for r in caplog.records)
    assert not any("old.png" in r.getMessage() for r in caplog.records), \
        "the sweep log names the entry id, never the shared file name"
    assert [i["name"] for i in client.get("/api/share/pending").json()["items"]] == ["recent.png"]


def test_memory_entry_older_than_the_ttl_is_dropped_on_access(privacy_client):
    import time
    privacy_client.post("/share-target", files={"files": ("old.png", _PNG, "image/png")})
    privacy_client.post("/share-target", files={"files": ("fresh.png", _PNG, "image/png")})
    store = privacy_client.app.state.share_memory_inbox
    assert len(store) == 2
    old = next(k for k in store if k.endswith("__old.png"))
    data, _exp = store[old]
    store[old] = (data, time.time() - 1)

    items = privacy_client.get("/api/share/pending").json()["items"]

    assert [i["name"] for i in items] == ["fresh.png"]
    assert old not in store
