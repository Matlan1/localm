# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rename routes for the music and video galleries.

``/api/music/file/{name}/rename`` and ``/api/video/file/{name}/rename`` mirror
the image gallery's rename route, including its two guards, both asserted here:

  * ``gallery.require_owner`` on the SOURCE - another principal gets 404, the
    same code a missing file returns (no existence oracle).
  * ``confined_name`` on the CALLER-SUPPLIED DESTINATION - a traversing
    destination is rejected even when the caller owns the source.

Every assertion reads the FILESYSTEM before the status code.
"""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.media import paths as media_paths
from localm.plugins.gui.web import attach_gui


MEDIA = {
    "music": {"dir": media_paths.MUSIC_DIR_NAME, "api": "/api/music", "ext": ".flac"},
    "video": {"dir": media_paths.VIDEO_DIR_NAME, "api": "/api/video", "ext": ".mp4"},
}


def _app(tmp_path, monkeypatch, plugin):
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


def _seed(kind, stem, sidecar=True):
    """Put a file (and optionally its sidecar) in the gallery dir, unowned so
    every principal may reach it. Returns (path, sidecar_path)."""
    d = media_paths.gallery_dir(MEDIA[kind]["dir"])
    d.mkdir(parents=True, exist_ok=True)
    p = d / (stem + MEDIA[kind]["ext"])
    p.write_bytes(b"fake media bytes")
    s = p.with_suffix(p.suffix + ".json")
    if sidecar:
        s.write_text(json.dumps({"marker": "SEEDED-" + stem}), encoding="utf-8")
    return p, s


@pytest.mark.parametrize("kind", ["music", "video"])
class TestMediaRename:

    def test_renames_file_and_sidecar_on_disk(self, kind, tmp_path, monkeypatch):
        app = _app(tmp_path, monkeypatch, kind)
        with TestClient(app) as c:
            old, old_side = _seed(kind, "original")
            api, ext = MEDIA[kind]["api"], MEDIA[kind]["ext"]

            r = c.post(api + "/file/original" + ext + "/rename",
                       json={"new_name": "renamed" + ext})

            # The filesystem state.
            new = old.with_name("renamed" + ext)
            assert new.is_file(), "renamed file missing (HTTP %s)" % r.status_code
            assert not old.exists(), "the original was left behind"
            new_side = new.with_suffix(new.suffix + ".json")
            assert new_side.is_file(), "sidecar did not follow the file"
            assert not old_side.exists(), "old sidecar left behind"
            assert json.loads(new_side.read_text())["marker"] == "SEEDED-original"
            assert r.status_code == 200
            assert r.json()["name"] == "renamed" + ext

    def test_extension_is_kept_when_omitted(self, kind, tmp_path, monkeypatch):
        app = _app(tmp_path, monkeypatch, kind)
        with TestClient(app) as c:
            old, _ = _seed(kind, "keepext")
            api, ext = MEDIA[kind]["api"], MEDIA[kind]["ext"]

            r = c.post(api + "/file/keepext" + ext + "/rename",
                       json={"new_name": "bare"})

            assert old.with_name("bare" + ext).is_file(), \
                "extension not preserved (HTTP %s)" % r.status_code
            assert r.status_code == 200

    def test_traversal_destination_refused_and_nothing_escapes(
            self, kind, tmp_path, monkeypatch):
        app = _app(tmp_path, monkeypatch, kind)
        with TestClient(app) as c:
            old, _ = _seed(kind, "victim")
            api, ext = MEDIA[kind]["api"], MEDIA[kind]["ext"]
            gallery_dir = old.parent

            for attempt in ("../escaped" + ext,
                            "../../escaped" + ext,
                            "sub/escaped" + ext):
                r = c.post(api + "/file/victim" + ext + "/rename",
                           json={"new_name": attempt})

                # The source must still be where it was, and nothing may appear
                # outside the gallery dir.
                assert old.is_file(), \
                    "%s moved the source (HTTP %s)" % (attempt, r.status_code)
                escaped = gallery_dir / attempt
                assert not escaped.exists(), \
                    "%s escaped to %s" % (attempt, escaped)
                assert r.status_code >= 400, \
                    "%s was accepted with HTTP %s" % (attempt, r.status_code)

    def test_collision_refused_and_target_untouched(self, kind, tmp_path,
                                                    monkeypatch):
        app = _app(tmp_path, monkeypatch, kind)
        with TestClient(app) as c:
            src, _ = _seed(kind, "src")
            dst, _ = _seed(kind, "dst")
            dst_bytes = dst.read_bytes()
            api, ext = MEDIA[kind]["api"], MEDIA[kind]["ext"]

            r = c.post(api + "/file/src" + ext + "/rename",
                       json={"new_name": "dst" + ext})

            assert src.is_file(), "source lost on a refused collision"
            assert dst.read_bytes() == dst_bytes, "EXISTING FILE WAS OVERWRITTEN"
            assert r.status_code == 409

    def test_empty_name_refused(self, kind, tmp_path, monkeypatch):
        app = _app(tmp_path, monkeypatch, kind)
        with TestClient(app) as c:
            old, _ = _seed(kind, "keepme")
            api, ext = MEDIA[kind]["api"], MEDIA[kind]["ext"]

            r = c.post(api + "/file/keepme" + ext + "/rename",
                       json={"new_name": "   "})

            assert old.is_file(), "source lost on an empty rename"
            assert r.status_code == 400

    def test_missing_source_is_404(self, kind, tmp_path, monkeypatch):
        app = _app(tmp_path, monkeypatch, kind)
        with TestClient(app) as c:
            media_paths.gallery_dir(MEDIA[kind]["dir"]).mkdir(parents=True,
                                                              exist_ok=True)
            api, ext = MEDIA[kind]["api"], MEDIA[kind]["ext"]
            r = c.post(api + "/file/nope" + ext + "/rename",
                       json={"new_name": "x" + ext})
            assert r.status_code == 404


@pytest.mark.parametrize("kind", ["music", "video"])
def test_another_principal_cannot_rename(kind, tmp_path, monkeypatch):
    """Holding the scope is not owning the artifact: B renaming A's file gets
    the same 404 a missing file returns, and A's file keeps its name."""
    app = _app(tmp_path, monkeypatch, kind)
    from localm import auth
    from localm.media import gallery
    with TestClient(app) as c:
        old, _ = _seed(kind, "owned")
        api, ext = MEDIA[kind]["api"], MEDIA[kind]["ext"]
        auth.create_key("ka", [kind])
        b = auth.create_key("kb", [kind])["key"]
        gallery.stamp_owner(kind, "owned" + ext, "key:ka")

        r = c.post(api + "/file/owned" + ext + "/rename",
                   headers={"Authorization": "Bearer " + b},
                   json={"new_name": "stolen" + ext})

        assert old.is_file(), "B renamed A's file (HTTP %s)" % r.status_code
        assert not old.with_name("stolen" + ext).exists()
        assert r.status_code == 404
