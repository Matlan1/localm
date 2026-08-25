# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-device upload: a caller that cannot browse the server disk (a phone, a
scoped key with no host filesystem access) uploads document BYTES from its own
device, which are extracted in memory and persisted into a collection.

Unlike /add, /upload reads NO server path, so there is NO whitelist/blacklist
confinement and NO host-fs requirement - the bytes are the caller's own content.
Size caps + the extractor's zip-bomb guard are the protections that apply.
"""

import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.rag.store import Collection


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# --------------------------------------------------------------------------- #
#  Collection.add_uploads (unit)                                              #
# --------------------------------------------------------------------------- #

class TestAddUploads:
    def test_indexes_and_is_queryable(self, tmp_path):
        c = Collection("kb", base=tmp_path / "rag").create()
        res = c.add_uploads([{"filename": "notes.md", "data": b"rocm gfx1030 runtime"}])
        assert res["added"] == 1
        hits = c.query("gfx1030")
        assert hits and "gfx1030" in hits[0]["text"].lower()
        # Sourced under an upload: key, not a server path.
        assert any(d.startswith("upload:") for d in c.documents())

    def test_dedups_by_content_hash(self, tmp_path):
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_uploads([{"filename": "a.txt", "data": b"hello world"}])
        again = c.add_uploads([{"filename": "a.txt", "data": b"hello world"}])
        assert again["skipped"] == 1 and again["added"] == 0
        changed = c.add_uploads([{"filename": "a.txt", "data": b"hello there"}])
        assert changed["updated"] == 1

    def test_force_reindexes_unchanged(self, tmp_path):
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_uploads([{"filename": "a.txt", "data": b"hello"}])
        forced = c.add_uploads([{"filename": "a.txt", "data": b"hello"}], force=True)
        assert forced["updated"] == 1

    def test_unsupported_type_goes_to_failed_not_crash(self, tmp_path):
        c = Collection("kb", base=tmp_path / "rag").create()
        res = c.add_uploads([{"filename": "a.bin", "data": b"\x00\x01\x02"}])
        assert res["added"] == 0
        assert len(res["failed"]) == 1

    def test_embeds_when_embed_fn_given(self, tmp_path):
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_uploads([{"filename": "a.txt", "data": b"alpha beta gamma delta"}],
                      embed_fn=lambda ts: [[1.0, 0.0, 0.0] for _ in ts])
        assert c.stats()["has_vectors"]


# --------------------------------------------------------------------------- #
#  HTTP route /api/rag/collections/{name}/upload                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def upload_app(tmp_path, monkeypatch):
    """A GUI app with the rag plugin, open mode (owner). Home is isolated."""
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui
    localm = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")

    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

    async def switch_model(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


class TestUploadRoute:
    def test_upload_starts_a_job(self, upload_app):
        with TestClient(upload_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [{"filename": "a.md", "content_b64": _b64(b"gfx1030 rocm")}],
                "embed": False})
            assert r.status_code == 200, r.text
            assert "job_id" in r.json()

    def test_no_files_400(self, upload_app):
        with TestClient(upload_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/upload",
                       json={"files": [], "embed": False})
            assert r.status_code == 400

    def test_bad_base64_400(self, upload_app):
        with TestClient(upload_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [{"filename": "a.md", "content_b64": "!!!not-base64"}],
                "embed": False})
            assert r.status_code == 400

    def test_too_large_413(self, upload_app):
        with TestClient(upload_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            big = _b64(b"x" * 30_000_001)   # one byte over the per-file cap
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [{"filename": "a.txt", "content_b64": big}],
                "embed": False})
            assert r.status_code == 413

    def test_oversized_rejected_before_decode(self, upload_app):
        # The per-file cap is checked on the base64 STRING length BEFORE
        # decoding, so an oversized payload is never materialized: a valid
        # base64-alphabet string over the char cap is refused with 413.
        with TestClient(upload_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            over = "A" * 40_000_001         # valid alphabet, one char over the cap
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [{"filename": "a.txt", "content_b64": over}],
                "embed": False})
            assert r.status_code == 413, r.text

    def test_non_host_scoped_key_can_upload(self, tmp_path, monkeypatch):
        # A scoped rag key with NO host filesystem access can still upload from
        # its own device (no fs gate, no path confinement).
        from localm.plugins.engine import PluginManager
        from localm.plugins.gui.web import attach_gui
        localm = tmp_path / ".localm"
        localm.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(localm))
        monkeypatch.setenv("LOCALM_API_KEY", "owner-key-xyz")   # protected mode
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", localm)
        monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
        from localm import auth
        scoped = auth.create_key("phone", ["rag"])["key"]       # rag scope, fs none

        app = FastAPI()
        PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

        async def switch_model(name):
            pass
        attach_gui(app, self_url="http://127.0.0.1:9/v1",
                   switch_model=switch_model, active_model=lambda: "m")

        with TestClient(app) as c:
            sc = {"Authorization": f"Bearer {scoped}"}
            c.post("/api/rag/collections", json={"name": "kb"}, headers=sc)
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [{"filename": "a.md", "content_b64": _b64(b"hi there")}],
                "embed": False}, headers=sc)
            assert r.status_code == 200, r.text


class TestBodySizeGuard:
    """The global request-body cap (Content-Length rejected before the body is
    buffered) that backstops the upload/extract decode-time OOM DoS."""

    def test_oversized_body_rejected_before_parsing(self, tmp_path, monkeypatch):
        import localm.inference.http_server as hs
        import localm.config as cfg
        localm = tmp_path / ".localm"
        localm.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(localm))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.setattr(cfg, "HOME_DIR", localm)
        monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
        # Tiny cap so the test needs no giant payload.
        monkeypatch.setattr(hs, "MAX_REQUEST_BODY_BYTES", 1000)
        app = hs.create_app(None)
        with TestClient(
                app,
                headers={"Authorization": f"Bearer {app.state.shell_token}"}) as c:
            big = b"x" * 5000                       # Content-Length 5000 > 1000
            r = c.patch("/v1/config", content=big,
                        headers={"Content-Type": "application/json"})
            assert r.status_code == 413, r.text
            assert "too large" in r.text.lower()
            # A normal small request is NOT blocked by the size guard.
            ok = c.patch("/v1/config", json={"n_ctx": 8192})
            assert ok.status_code != 413, ok.text

    def test_chunked_body_without_content_length_still_capped(
            self, tmp_path, monkeypatch):
        # Transfer-Encoding: chunked never sends a Content-Length, so the check
        # above does not apply to it. A generator body drives TestClient's ASGI
        # transport to omit Content-Length and use chunked transfer, the same as
        # a real chunked POST, so this exercises the streaming-cap code path
        # rather than the Content-Length fast path covered above.
        import localm.inference.http_server as hs
        import localm.config as cfg
        localm = tmp_path / ".localm"
        localm.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(localm))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.setattr(cfg, "HOME_DIR", localm)
        monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
        monkeypatch.setattr(hs, "MAX_REQUEST_BODY_BYTES", 1000)
        app = hs.create_app(None)

        def oversized_chunks():
            for _ in range(20):            # 20 * 200 = 4000 bytes > 1000 cap
                yield b"a" * 200

        with TestClient(app) as c:
            r = c.post("/v1/chat/completions", content=oversized_chunks(),
                       headers={"Content-Type": "application/json"})
            assert r.status_code == 413, r.text
            assert "too large" in r.text.lower()
            # The server must stay healthy for the next request - the cap
            # must reject cleanly, not leave the app/connection wedged.
            ok = c.get("/health")
            assert ok.status_code in (200, 503)

    def test_chunked_body_within_cap_not_blocked(self, tmp_path, monkeypatch):
        # Negative case: a legitimately small chunked (no Content-Length)
        # request must not be rejected by the streaming cap.
        import localm.inference.http_server as hs
        import localm.config as cfg
        localm = tmp_path / ".localm"
        localm.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(localm))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.setattr(cfg, "HOME_DIR", localm)
        monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
        app = hs.create_app(None)

        def small_chunks():
            yield b'{"model":"x","messages":[{"role":"user",'
            yield b'"content":"hi"}]}'

        with TestClient(app) as c:
            r = c.post("/v1/chat/completions", content=small_chunks(),
                       headers={"Content-Type": "application/json"})
            assert r.status_code != 413, r.text
