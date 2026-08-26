# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-principal ownership on the generated-media gallery routes (image/music/
video): a key scoped to ONLY that plugin must not be able to enumerate, read,
delete, move, or rename another principal's generated media, and the history
listing must not leak another principal's items.

Mirrors the jobs family's own gate (job_owner_ok / owned_job) in
test_jobs_owner_binding.py. The media half lives in localm.media.gallery,
shared by all three plugins' plug.py.
"""

import json
import time as _time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import attach_gui


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _mk_keys(*scope_lists):
    from localm import auth
    return [auth.create_key(f"k{i}", s)["key"] for i, s in enumerate(scope_lists)]


def _gallery_app(tmp_path, monkeypatch, plugin):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install(plugin)

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


def _wait_job(client, job_id, headers, timeout=30):
    end = None
    deadline = _time.monotonic() + timeout
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=headers) as r:
        for raw in r.iter_lines():
            if _time.monotonic() > deadline:
                break
            if not raw.startswith("data: "):
                continue
            ev = json.loads(raw[6:])
            if ev["type"] == "end":
                end = ev
                break
    return end


class TestImageGalleryOwnership:
    @staticmethod
    def _generate(client, monkeypatch, key):
        import localm.image_gen.comfy as comfy
        monkeypatch.setattr(comfy, "ensure_comfy", lambda *a, **k: (True, "up"))
        monkeypatch.setattr(comfy, "free_comfy_vram", lambda *a, **k: False)
        monkeypatch.setattr("localm.vram.decide_media_swap", lambda *a, **k: False)

        def fake_gen(prompt, out_path, **kw):
            Path(out_path).write_bytes(b"\x89PNG fake")
            return True, f"Image saved to {out_path}"
        monkeypatch.setattr(comfy, "generate_image", fake_gen)

        r = client.post("/api/imagine", headers=_h(key), json={"prompt": "a fox"})
        assert r.status_code == 200
        end = _wait_job(client, r.json()["job_id"], _h(key))
        assert end and end["status"] == "done"
        return end["result"]

    def test_a_image_key_cannot_touch_b_image_key_media(self, tmp_path, monkeypatch):
        app = _gallery_app(tmp_path, monkeypatch, "image")
        a, b = _mk_keys(["image"], ["image"])
        with TestClient(app) as c:
            name = self._generate(c, monkeypatch, a)

            # B (a different key, same scope) is locked out of A's image - 404
            # everywhere, the SAME code a missing file would get.
            assert c.get(f"/api/imagine/file/{name}", headers=_h(b)).status_code == 404
            assert c.delete(f"/api/imagine/file/{name}", headers=_h(b)).status_code == 404
            assert c.post(f"/api/imagine/file/{name}/move", headers=_h(b),
                          json={"dest": str(tmp_path / "out")}).status_code == 404
            assert c.post(f"/api/imagine/file/{name}/rename", headers=_h(b),
                          json={"new_name": "stolen.png"}).status_code == 404
            # B's history does not reveal A's image; A's own history does.
            assert all(it["name"] != name for it in
                      c.get("/api/imagine/history", headers=_h(b)).json()["images"])
            assert any(it["name"] == name for it in
                      c.get("/api/imagine/history", headers=_h(a)).json()["images"])
            # A reaches its own image fine, and it is untouched (still on disk).
            assert c.get(f"/api/imagine/file/{name}", headers=_h(a)).status_code == 200

    def test_owner_admin_sees_every_image(self, tmp_path, monkeypatch):
        app = _gallery_app(tmp_path, monkeypatch, "image")
        (a,) = _mk_keys(["image"])
        with TestClient(app) as c:
            name = self._generate(c, monkeypatch, a)
            monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")   # owner = admin
            assert c.get(f"/api/imagine/file/{name}",
                        headers=_h("ownersecret")).status_code == 200
            assert any(it["name"] == name for it in
                      c.get("/api/imagine/history",
                           headers=_h("ownersecret")).json()["images"])

    def test_legacy_untracked_file_stays_visible_to_any_scoped_key(
            self, tmp_path, monkeypatch):
        """A file that never went through stamp_owner (placed directly on disk)
        has no recorded owner and stays reachable by any scoped key, matching
        jobs' "no recorded owner is unrestricted" rule."""
        app = _gallery_app(tmp_path, monkeypatch, "image")
        (a,) = _mk_keys(["image"])
        images_dir = Path.home() / ".localm" / "gui_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "legacy.png").write_bytes(b"\x89PNG fake")
        with TestClient(app) as c:
            assert c.get("/api/imagine/file/legacy.png", headers=_h(a)).status_code == 200
            assert any(it["name"] == "legacy.png" for it in
                      c.get("/api/imagine/history", headers=_h(a)).json()["images"])


class TestCorruptIndexFailsClosed:
    """An owner index that EXISTS but is unreadable or corrupt FAILS CLOSED and
    denies a non-owner, rather than collapsing to an empty index that reports
    every artifact as unowned (job_owner_ok treats owner=None as unrestricted).
    A genuinely-ABSENT index stays open/untracked, so the two are distinct."""

    def _corrupt_index(self, kind: str) -> Path:
        from localm.media import gallery
        p = gallery._index_path(kind)
        assert p.is_file()          # generation stamped an owner -> index exists
        p.write_text("{ this is not valid json", encoding="utf-8")
        return p

    def test_corrupt_index_denies_scoped_keys_but_admin_sees_all(
            self, tmp_path, monkeypatch):
        app = _gallery_app(tmp_path, monkeypatch, "image")
        a, b = _mk_keys(["image"], ["image"])
        with TestClient(app) as c:
            name = TestImageGalleryOwnership._generate(c, monkeypatch, a)
            self._corrupt_index("image")

            # A DIFFERENT scoped key is DENIED (fail closed), not handed every
            # artifact as if unowned.
            assert c.get(f"/api/imagine/file/{name}", headers=_h(b)).status_code == 404
            assert c.get("/api/imagine/history", headers=_h(b)).json()["images"] == []
            # Even the real owner's SCOPED key is denied until the index is
            # repaired: ownership is indeterminate, so every non-admin principal
            # fails closed.
            assert c.get(f"/api/imagine/file/{name}", headers=_h(a)).status_code == 404
            assert c.get("/api/imagine/history", headers=_h(a)).json()["images"] == []

            # The owner/admin key still reaches everything, so the gallery stays
            # usable and the index remains repairable.
            monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
            assert c.get(f"/api/imagine/file/{name}",
                         headers=_h("ownersecret")).status_code == 200
            assert any(it["name"] == name for it in
                       c.get("/api/imagine/history",
                            headers=_h("ownersecret")).json()["images"])

    def test_absent_index_stays_open(self, tmp_path, monkeypatch):
        """An ABSENT index (never created, or deleted) is the open/untracked
        case, so any scoped key still reaches the media, unlike the corrupt case
        above."""
        app = _gallery_app(tmp_path, monkeypatch, "image")
        a, b = _mk_keys(["image"], ["image"])
        with TestClient(app) as c:
            name = TestImageGalleryOwnership._generate(c, monkeypatch, a)
            from localm.media import gallery
            gallery._index_path("image").unlink()       # genuinely absent
            assert c.get(f"/api/imagine/file/{name}", headers=_h(b)).status_code == 200
            assert any(it["name"] == name for it in
                       c.get("/api/imagine/history", headers=_h(b)).json()["images"])


class TestGalleryIndexReadContract:
    """Unit-level contract for the read/write primitives: absent vs corrupt is a
    hard distinction, and writes are atomic (no temp-file residue, round-trips)."""

    def _home(self, tmp_path, monkeypatch):
        # _index_path -> home_dir() -> _detect_home() reads LOCALM_HOME at call
        # time, so set the env var (not just the HOME_DIR constant).
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path / ".localm"))

    def test_absent_returns_empty(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        from localm.media import gallery
        assert gallery._read_index("image") == {}
        assert gallery.owner_of("image", "whatever.png") is None

    def test_corrupt_raises_not_empty(self, tmp_path, monkeypatch):
        import pytest
        self._home(tmp_path, monkeypatch)
        from localm.media import gallery
        p = gallery._index_path("image")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{ not: valid, json", encoding="utf-8")
        with pytest.raises(gallery.GalleryIndexUnreadable):
            gallery._read_index("image")
        with pytest.raises(gallery.GalleryIndexUnreadable):
            gallery.owner_of("image", "x.png")

    @pytest.mark.parametrize("body", ["[1, 2, 3]", "null", '"x"', "42"])
    def test_non_dict_json_fails_closed(self, tmp_path, monkeypatch, body):
        # An index that PARSES but is not an object exists without being a
        # usable owner map, so it fails closed like a truncated one.
        self._home(tmp_path, monkeypatch)
        from localm.media import gallery
        p = gallery._index_path("image")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
        with pytest.raises(gallery.GalleryIndexUnreadable):
            gallery._read_index("image")
        with pytest.raises(gallery.GalleryIndexUnreadable):
            gallery.owner_of("image", "x.png")

    def test_atomic_write_roundtrip_leaves_no_temp(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        from localm.media import gallery
        gallery._write_index("image", {"a.png": "owner1"})
        assert gallery._read_index("image") == {"a.png": "owner1"}
        # os.replace leaves no half-written temp behind.
        leftovers = list(gallery._index_path("image").parent.glob("*.tmp.*"))
        assert leftovers == []

    def test_writers_do_not_clobber_corrupt_index(self, tmp_path, monkeypatch):
        # stamp/rename/forget must not overwrite a corrupt index with a fresh
        # map.
        self._home(tmp_path, monkeypatch)
        from localm.media import gallery
        p = gallery._index_path("image")
        p.parent.mkdir(parents=True, exist_ok=True)
        corrupt = "{ not valid json"
        p.write_text(corrupt, encoding="utf-8")
        gallery.stamp_owner("image", "new.png", "ownerX")
        gallery.rename_owner("image", "a.png", "b.png")
        gallery.forget_owner("image", "a.png")
        assert p.read_text(encoding="utf-8") == corrupt     # untouched


class TestMusicGalleryOwnership:
    @staticmethod
    def _generate(client, monkeypatch, key):
        import localm.music_gen as music_gen
        import localm.image_gen.comfy as comfy
        monkeypatch.setattr(comfy, "ensure_comfy", lambda *a, **k: (True, "up"))
        monkeypatch.setattr(comfy, "free_comfy_vram", lambda *a, **k: False)
        monkeypatch.setattr("localm.vram.decide_media_swap", lambda *a, **k: False)

        def fake_gen(tags, out_path, **kw):
            Path(out_path).write_bytes(b"FLACfake")
            return True, f"Track saved to {out_path}"
        monkeypatch.setattr(music_gen, "generate_music", fake_gen)

        r = client.post("/api/music", headers=_h(key),
                        json={"tags": "lofi", "duration_seconds": 10})
        assert r.status_code == 200
        end = _wait_job(client, r.json()["job_id"], _h(key))
        assert end and end["status"] == "done"
        return end["result"]

    def test_a_music_key_cannot_touch_b_music_key_media(self, tmp_path, monkeypatch):
        app = _gallery_app(tmp_path, monkeypatch, "music")
        a, b = _mk_keys(["music"], ["music"])
        with TestClient(app) as c:
            name = self._generate(c, monkeypatch, a)

            assert c.get(f"/api/music/file/{name}", headers=_h(b)).status_code == 404
            assert c.delete(f"/api/music/file/{name}", headers=_h(b)).status_code == 404
            assert c.post(f"/api/music/file/{name}/move", headers=_h(b),
                          json={"dest": str(tmp_path / "out")}).status_code == 404
            assert all(it["name"] != name for it in
                      c.get("/api/music/history", headers=_h(b)).json()["tracks"])
            assert any(it["name"] == name for it in
                      c.get("/api/music/history", headers=_h(a)).json()["tracks"])
            assert c.get(f"/api/music/file/{name}", headers=_h(a)).status_code == 200


class TestVideoGalleryOwnership:
    @staticmethod
    def _generate(client, monkeypatch, key):
        import localm.video_gen as video_gen
        import localm.image_gen.comfy as comfy
        monkeypatch.setattr(comfy, "ensure_comfy", lambda *a, **k: (True, "up"))
        monkeypatch.setattr(comfy, "free_comfy_vram", lambda *a, **k: False)
        monkeypatch.setattr("localm.vram.decide_media_swap", lambda *a, **k: False)

        def fake_gen(prompt, out_path, **kw):
            Path(out_path).write_bytes(b"fake mp4")
            return True, f"Video saved to {out_path}"
        monkeypatch.setattr(video_gen, "generate_video", fake_gen)

        r = client.post("/api/video", headers=_h(key), json={"prompt": "a fox"})
        assert r.status_code == 200
        end = _wait_job(client, r.json()["job_id"], _h(key))
        assert end and end["status"] == "done"
        return end["result"]

    def test_a_video_key_cannot_touch_b_video_key_media(self, tmp_path, monkeypatch):
        app = _gallery_app(tmp_path, monkeypatch, "video")
        a, b = _mk_keys(["video"], ["video"])
        with TestClient(app) as c:
            name = self._generate(c, monkeypatch, a)

            assert c.get(f"/api/video/file/{name}", headers=_h(b)).status_code == 404
            assert c.delete(f"/api/video/file/{name}", headers=_h(b)).status_code == 404
            assert c.post(f"/api/video/file/{name}/move", headers=_h(b),
                          json={"dest": str(tmp_path / "out")}).status_code == 404
            assert all(it["name"] != name for it in
                      c.get("/api/video/history", headers=_h(b)).json()["videos"])
            assert any(it["name"] == name for it in
                      c.get("/api/video/history", headers=_h(a)).json()["videos"])
            assert c.get(f"/api/video/file/{name}", headers=_h(a)).status_code == 200
