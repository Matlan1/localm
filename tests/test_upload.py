# SPDX-License-Identifier: AGPL-3.0-or-later
"""A phone (or browser) uploads files INTO <home>/uploads/ so models and tools
can read them, beyond transient chat attachments. The endpoints are scope-gated
(CONFIG_WRITE to write/delete, CONFIG_READ to list) because writing host files is
privileged - a restricted shared key must not be able to drop files on the host.
File names are basename-confined so they cannot traverse out of the uploads dir.
"""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.gui import web
from localm.plugins.gui.web import attach_gui


@pytest.fixture
def scoped_app(tmp_path, monkeypatch):
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
    app.state._home = home
    return app


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _writer_key():
    # config:write is PRIVILEGED (owner-only), so allow_privileged mints it here -
    # exactly the posture the upload routes rely on: only an owner/companion-admin
    # key may write host files.
    from localm import auth
    return auth.create_key("writer", [S.CONFIG_WRITE, S.CONFIG_READ],
                           allow_privileged=True)["key"]


def test_upload_writes_file_to_home_uploads(scoped_app):
    key = _writer_key()
    with TestClient(scoped_app) as c:
        r = c.post("/api/upload", headers=_hdr(key),
                   files={"file": ("hello.txt", b"hello world")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["uploaded"] == [{"name": "hello.txt", "bytes": 11}]
    saved = scoped_app.state._home / "uploads" / "hello.txt"
    assert saved.is_file() and saved.read_bytes() == b"hello world"


def test_upload_collision_does_not_clobber(scoped_app):
    key = _writer_key()
    with TestClient(scoped_app) as c:
        c.post("/api/upload", headers=_hdr(key), files={"file": ("a.txt", b"first")})
        r = c.post("/api/upload", headers=_hdr(key), files={"file": ("a.txt", b"second")})
        assert r.json()["uploaded"][0]["name"] == "a (1).txt"
    up = scoped_app.state._home / "uploads"
    assert (up / "a.txt").read_bytes() == b"first"
    assert (up / "a (1).txt").read_bytes() == b"second"


def test_upload_path_traversal_is_confined(scoped_app):
    key = _writer_key()
    with TestClient(scoped_app) as c:
        r = c.post("/api/upload", headers=_hdr(key),
                   files={"file": ("../../evil.txt", b"pwned")})
        assert r.status_code == 200, r.text
    up = scoped_app.state._home / "uploads"
    # Saved as a basename INSIDE uploads, never written outside it.
    assert (up / "evil.txt").is_file()
    assert not (scoped_app.state._home.parent / "evil.txt").exists()
    assert not (scoped_app.state._home / "evil.txt").exists()


def test_upload_too_large_is_rejected(scoped_app, monkeypatch):
    monkeypatch.setattr(web, "_MAX_UPLOAD_BYTES", 8)   # 8-byte cap for the test
    key = _writer_key()
    with TestClient(scoped_app) as c:
        r = c.post("/api/upload", headers=_hdr(key),
                   files={"file": ("big.bin", b"way more than eight bytes")})
        assert r.status_code == 413, r.text
    assert not (scoped_app.state._home / "uploads" / "big.bin").exists()


def test_list_and_delete_uploads(scoped_app):
    key = _writer_key()
    with TestClient(scoped_app) as c:
        c.post("/api/upload", headers=_hdr(key), files={"file": ("doc.md", b"# hi")})
        r = c.get("/api/uploads", headers=_hdr(key))
        assert r.status_code == 200
        names = [it["name"] for it in r.json()["items"]]
        assert "doc.md" in names
        d = c.delete("/api/uploads/doc.md", headers=_hdr(key))
        assert d.status_code == 200 and d.json()["removed"] == "doc.md"
        assert not (scoped_app.state._home / "uploads" / "doc.md").exists()
        # Deleting again -> 404.
        assert c.delete("/api/uploads/doc.md", headers=_hdr(key)).status_code == 404


def test_delete_traversal_is_confined(scoped_app):
    key = _writer_key()
    # Plant a file OUTSIDE the uploads dir; a traversal delete must not reach it.
    victim = scoped_app.state._home / "config.json"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("{}", encoding="utf-8")
    with TestClient(scoped_app) as c:
        # %2e%2e%2f = "../": the router normalizes it away (-> /api/config.json,
        # which has no DELETE), so it never reaches a delete at all. The security
        # property that matters: the file outside uploads/ is NOT removed.
        r = c.delete("/api/uploads/%2e%2e%2fconfig.json", headers=_hdr(key))
        assert r.status_code != 200, r.text
    assert victim.exists(), "a traversal delete must not reach a file outside uploads/"


def test_confined_upload_path_strips_traversal(scoped_app):
    # Defense in depth: the confinement helper itself reduces any name to a
    # basename INSIDE the uploads dir (or rejects it), regardless of routing.
    up = (scoped_app.state._home / "uploads").resolve()
    # Traversal / absolute names collapse to a basename inside uploads.
    for bad, expect in [("../config.json", "config.json"),
                        ("/etc/passwd", "passwd"),
                        ("a/b/c.txt", "c.txt")]:
        p = web._confined_upload_path(bad)
        assert p.parent.resolve() == up, f"{bad!r} escaped to {p}"
        assert p.name == expect
    # Empty / dot names are rejected outright.
    for bad in ("", ".", ".."):
        with pytest.raises(Exception):
            web._confined_upload_path(bad)


def test_upload_routes_are_scope_gated(scoped_app):
    from localm import auth
    narrow = auth.create_key("narrow", [S.MCP])["key"]          # no config:* at all
    reader = auth.create_key("reader", [S.CONFIG_READ])["key"]   # read-only
    with TestClient(scoped_app) as c:
        # An under-scoped key is 403 on every uploads route.
        assert c.post("/api/upload", headers=_hdr(narrow),
                      files={"file": ("x.txt", b"x")}).status_code == 403
        assert c.get("/api/uploads", headers=_hdr(narrow)).status_code == 403
        assert c.delete("/api/uploads/x.txt", headers=_hdr(narrow)).status_code == 403
        # config:read may LIST but not write or delete.
        assert c.get("/api/uploads", headers=_hdr(reader)).status_code != 403
        assert c.post("/api/upload", headers=_hdr(reader),
                      files={"file": ("x.txt", b"x")}).status_code == 403
        assert c.delete("/api/uploads/x.txt", headers=_hdr(reader)).status_code == 403


def _raw_multipart(filename: bytes, data: bytes, boundary: bytes = b"BoUnDaRy"):
    """Build a multipart body with an arbitrary (possibly hostile) filename - httpx
    would sanitize it, so we hand-craft to exercise the server's own guard."""
    body = (b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="' + filename + b'"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n" + data + b"\r\n"
            b"--" + boundary + b"--\r\n")
    return body, "multipart/form-data; boundary=" + boundary.decode()


def test_upload_rejects_illegal_filename_chars(scoped_app):
    key = _writer_key()
    with TestClient(scoped_app) as c:
        # A Windows ADS name ("x.txt:stream") and a literal NUL must be rejected
        # (skipped), never written and never a bare 500.
        for fname in [b"real.txt:stream", b"ev\x00il.txt", b"n<o>.txt"]:
            body, ct = _raw_multipart(fname, b"payload")
            r = c.post("/api/upload", headers={**_hdr(key), "Content-Type": ct},
                       content=body)
            assert r.status_code == 400, f"{fname!r} -> {r.status_code} {r.text}"
    up = scoped_app.state._home / "uploads"
    assert not up.exists() or list(up.iterdir()) == [], "no illegal-named file written"


def test_confined_upload_path_rejects_illegal_chars(scoped_app):
    # A NUL / ADS / reserved char in a DELETE name is a clean 400, not an uncaught
    # ValueError surfacing as a 500.
    # Names whose BASENAME still carries an illegal char (a single-letter "a:" is a
    # Windows drive, so Path("a:b.txt").name == "b.txt" and is legitimately safe -
    # use a multi-char NTFS ADS name to keep the ':' in the basename).
    from fastapi import HTTPException
    for bad in ("real.txt:stream", "x\x00y", "n<o>.txt", "p|q"):
        with pytest.raises(HTTPException) as ei:
            web._confined_upload_path(bad)
        assert ei.value.status_code == 400


def test_name_is_safe_docstring_documents_device_names():
    # The fix IS a docstring, so its wording is pinned directly.
    doc = " ".join((web._name_is_safe.__doc__ or "").lower().split())
    assert "character/reserved-name check" not in doc
    assert "device names" in doc
    assert "not rejected" in doc


@pytest.mark.skipif(os.name != "nt", reason="Windows device-name behaviour")
def test_upload_bare_device_name_does_not_lose_data(scoped_app):
    # A bare, extensionless Windows device name ("NUL") is not rejected by
    # _name_is_safe (see its docstring), but _unique_upload_target's
    # exists() check redirects it to "NUL (1)" instead of writing through to the
    # actual device, which would silently discard the data.
    key = _writer_key()
    body = b"12345678901234567890"  # 20 bytes; content must survive a real write
    with TestClient(scoped_app) as c:
        r = c.post("/api/upload", headers=_hdr(key), files={"file": ("NUL", body)})
    up = scoped_app.state._home / "uploads"
    entries = list(up.iterdir()) if up.exists() else []
    # Assert on the data before the status code.
    assert len(entries) == 1, f"expected exactly one saved file, got {entries}"
    assert entries[0].read_bytes() == body
    assert r.status_code == 200, r.text
    assert r.json()["uploaded"][0]["name"] == "NUL (1)"


class TestConfinedUploadPathAliasSubstitution:
    """DELETE /api/uploads/{name} is the one route in this file that resolves a
    raw, caller-supplied name directly against an EXISTING file (every other
    route either writes through _unique_upload_target's non-clobbering
    retry-with-counter, or folds the name into an already-unique synthesized one
    - see share.py). An OS-level alias, such as an NTFS 8.3 short name, can pass
    basename/character checks while resolving to a pre-existing,
    DIFFERENTLY-NAMED real file: 'LONGMO~1.GGU' resolving to
    'LongModelNameThatIsVeryLong.gguf'. A containment-only check
    (target.is_relative_to(base)) is satisfied by the alias, since it stays
    inside uploads/, so a delete request naming an alias the caller never
    uploaded would unlink someone else's real file. pathsafe.confined_name
    rejects it.

    8dot3 short-name generation is volume/config-dependent, so this simulates the
    alias condition directly via Path.resolve rather than depending on the OS
    having generated one - deterministic regardless of which volume runs it."""

    def test_alias_name_is_rejected_by_the_validator(self, scoped_app, monkeypatch):
        up = scoped_app.state._home / "uploads"
        up.mkdir(parents=True, exist_ok=True)
        victim = up / "LongModelNameThatIsVeryLong.gguf"
        victim.write_bytes(b"VICTIM-CONTENT")

        real_resolve = Path.resolve

        def fake_resolve(self, *a, **k):
            if self.name == "LONGMO~1.GGU":
                return victim.resolve()
            return real_resolve(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", fake_resolve)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            web._confined_upload_path("LONGMO~1.GGU")
        assert ei.value.status_code == 400
        assert victim.exists(), "the alias check must reject BEFORE any delete is attempted"

    def test_alias_delete_request_does_not_remove_the_victim_end_to_end(
            self, scoped_app, monkeypatch):
        """Same scenario, through the real HTTP DELETE route - proves the
        route-level behavior, not just the validator in isolation."""
        up = scoped_app.state._home / "uploads"
        up.mkdir(parents=True, exist_ok=True)
        victim = up / "LongModelNameThatIsVeryLong.gguf"
        victim.write_bytes(b"VICTIM-CONTENT")

        real_resolve = Path.resolve

        def fake_resolve(self, *a, **k):
            if self.name == "LONGMO~1.GGU":
                return victim.resolve()
            return real_resolve(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", fake_resolve)
        key = _writer_key()
        with TestClient(scoped_app) as c:
            r = c.delete("/api/uploads/LONGMO~1.GGU", headers=_hdr(key))
            assert r.status_code == 400, r.text
        assert victim.exists(), "the victim file must survive an alias-named delete request"
        assert victim.read_bytes() == b"VICTIM-CONTENT"

    def test_same_name_delete_of_the_real_file_still_works(self, scoped_app):
        """The alias check must not reject the ordinary, legitimate case of
        deleting a file by its own exact, on-disk name."""
        up = scoped_app.state._home / "uploads"
        up.mkdir(parents=True, exist_ok=True)
        (up / "LongModelNameThatIsVeryLong.gguf").write_bytes(b"content")
        key = _writer_key()
        with TestClient(scoped_app) as c:
            r = c.delete("/api/uploads/LongModelNameThatIsVeryLong.gguf", headers=_hdr(key))
            assert r.status_code == 200, r.text
        assert not (up / "LongModelNameThatIsVeryLong.gguf").exists()
