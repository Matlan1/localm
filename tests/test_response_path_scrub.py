# SPDX-License-Identifier: AGPL-3.0-or-later
"""Absolute paths in HTTP responses: the local operator (open mode or an
ADMIN key) gets them as they are; any other caller gets them through
pathscrub.scrub_paths, so the data dir, home and account name never reach a
scoped key."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.pathscrub import DATA_DIR_LABEL
from localm.plugins.gui.web import attach_gui


@pytest.fixture(autouse=True)
def _comfyui_is_set_up(monkeypatch):
    monkeypatch.setattr("localm.media.backend_choice.comfy_is_set_up",
                        lambda *a, **k: True)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", h)
    monkeypatch.setattr(_cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", h / "registry.json")
    return h


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _key(scopes):
    from localm import auth
    return auth.create_key("k", scopes, allow_privileged=True)["key"]


def _gui_app(tmp_path, *plugins):
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    pm = PluginManager(app, external_root=tmp_path / "noplugins")
    for p in plugins:
        pm.install(p)

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


def _assert_scrubbed(value, home, tmp_path):
    assert str(home) not in value
    assert str(tmp_path) not in value
    assert value.startswith(DATA_DIR_LABEL)


_GALLERIES = [
    ("image", "/api/imagine/history", "images", "_images_dir", "a.png"),
    ("music", "/api/music/history", "tracks", "_music_dir", "a.flac"),
    ("video", "/api/video/history", "videos", "_video_dir", "a.mp4"),
]


@pytest.mark.parametrize("plugin,route,field,dir_fn,fname", _GALLERIES)
def test_gallery_history_path(home, tmp_path, plugin, route, field, dir_fn, fname):
    import importlib
    app = _gui_app(tmp_path, plugin)
    mod = importlib.import_module(f"localm.plugins.builtin.{plugin}.plug")
    real = getattr(mod, dir_fn)() / fname
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_bytes(b"x")
    source = home / "uploads" / "src.png"
    real.with_suffix(real.suffix + ".json").write_text(json.dumps(
        {"prompt": "a fox", "input_image": str(source), "steps": 4,
         "extra": {"paths": [str(source)]}}), encoding="utf-8")
    scoped, admin = _key([plugin]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.get(route, headers=_h(scoped)).json()[field]
        full = c.get(route, headers=_h(admin)).json()[field]
    assert [it["name"] for it in mine] == [fname]
    _assert_scrubbed(mine[0]["path"], home, tmp_path)
    assert mine[0]["path"].endswith(fname)
    assert full[0]["path"] == str(real)
    meta = mine[0]["meta"]
    _assert_scrubbed(meta["input_image"], home, tmp_path)
    _assert_scrubbed(meta["extra"]["paths"][0], home, tmp_path)
    assert meta["prompt"] == "a fox" and meta["steps"] == 4
    assert full[0]["meta"]["input_image"] == str(source)


def test_gallery_move_result_path(home, tmp_path, monkeypatch):
    from localm.plugins.builtin.image import plug as image_plug
    app = _gui_app(tmp_path, "image")
    gallery = image_plug._images_dir()
    gallery.mkdir(parents=True, exist_ok=True)
    (gallery / "m.png").write_bytes(b"x")
    dest = home / "uploads" / "moved"
    scoped = _key(["image"])
    with TestClient(app) as c:
        r = c.post("/api/imagine/file/m.png/move", headers=_h(scoped),
                   json={"dest": str(dest)})
    assert r.status_code == 200, r.text
    assert (dest / "m.png").is_file()
    _assert_scrubbed(r.json()["path"], home, tmp_path)


def test_memory_get_path(home, tmp_path, monkeypatch):
    from localm.plugins.builtin.memory import plug
    monkeypatch.setattr(plug, "_home", lambda: home)
    app = FastAPI()
    app.include_router(plug._router)
    scoped, admin = _key(["memory"]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.get("/api/memory", headers=_h(scoped)).json()["path"]
        full = c.get("/api/memory", headers=_h(admin)).json()["path"]
    _assert_scrubbed(mine, home, tmp_path)
    assert full.startswith(str(home))


def test_bug_report_path(home, tmp_path):
    from localm.inference.http_server import create_app
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    writer, admin = _key(["config:write"]), _key(["admin"])
    app = create_app(engine)
    with TestClient(app) as c:
        mine = c.post("/api/bug-report", headers=_h(writer),
                      json={"message": "the scoped one"}).json()
        full = c.post("/api/bug-report", headers=_h(admin),
                      json={"message": "the owner one"}).json()
    _assert_scrubbed(mine["path"], home, tmp_path)
    assert mine["path"].endswith(mine["filename"])
    assert Path(full["path"]).is_file()


def test_doctor_report_strings(home, tmp_path):
    from localm import diagnostics as d
    leaked = str(tmp_path / "AppData" / "Local" / "Temp" / "probe-venv")
    checks = [d.CheckResult(key=k, label=d.CHECK_LABELS[k], status=d.FAIL,
                            summary=f"{k} at {home}",
                            findings=(d.Finding(d.FAIL, f"failed: {leaked}"),))
              for k in d.CHECK_KEYS]
    app = _gui_app(tmp_path)
    app.state.diagnostics_run = {"job_id": None, "started_at": 1.0,
                                 "finished_at": 2.0, "progress": None,
                                 "report": d.build_report(checks).as_dict()}
    reader, admin = _key(["config:read"]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.get("/api/doctor", headers=_h(reader)).text
        full = c.get("/api/doctor", headers=_h(admin)).json()
    assert str(tmp_path).replace("\\", "\\\\") not in mine
    assert str(tmp_path) not in mine
    assert "probe-venv" in mine and "failed:" in mine
    assert full["report"]["checks"][0]["findings"][0]["text"] == f"failed: {leaked}"


def test_uploads_and_comfy_status_paths(home, tmp_path, monkeypatch):
    import localm.media.managed_comfy as mc
    root = home / "comfyui"
    (root / "models").mkdir(parents=True)
    monkeypatch.setattr(mc, "is_managed_comfy_installed", lambda: True)
    monkeypatch.setattr(mc, "managed_comfy_paths",
                        lambda: MagicMock(root=root, models_dir=root / "models"))
    app = _gui_app(tmp_path)
    reader, admin = _key(["config:read"]), _key(["admin"])
    with TestClient(app) as c:
        up_mine = c.get("/api/uploads", headers=_h(reader)).json()["dir"]
        up_full = c.get("/api/uploads", headers=_h(admin)).json()["dir"]
        cs_mine = c.get("/api/comfy/managed-status", headers=_h(reader)).json()
        cs_full = c.get("/api/comfy/managed-status", headers=_h(admin)).json()
    _assert_scrubbed(up_mine, home, tmp_path)
    assert up_full == str(home / "uploads")
    assert cs_mine["state"] == "installed"
    _assert_scrubbed(cs_mine["path"], home, tmp_path)
    _assert_scrubbed(cs_mine["models_dir"], home, tmp_path)
    assert cs_full["path"] == str(root)
    assert cs_full["models_dir"] == str(root / "models")


def test_open_mode_keeps_full_paths(home, tmp_path):
    from localm.plugins.builtin.image import plug as image_plug
    app = _gui_app(tmp_path, "image")
    gallery = image_plug._images_dir()
    gallery.mkdir(parents=True, exist_ok=True)
    (gallery / "o.png").write_bytes(b"x")
    with TestClient(app) as c:
        items = c.get("/api/imagine/history").json()["images"]
    assert items[0]["path"] == str(gallery / "o.png")


class TestInputImageAcceptsTheScrubbedForm:
    def _gallery_file(self, home):
        from localm.media import paths
        d = paths.gallery_dir(paths.IMAGE_DIR_NAME)
        d.mkdir(parents=True, exist_ok=True)
        f = d / "src.png"
        f.write_bytes(b"x")
        return f

    @pytest.mark.parametrize("sep", ["/", "\\"])
    def test_scrubbed_gallery_path_resolves(self, home, sep):
        from localm.media.paths import check_input_image
        f = self._gallery_file(home)
        rel = f.relative_to(home)
        raw = DATA_DIR_LABEL + sep + sep.join(rel.parts)
        if sep == "\\" and "\\" not in str(Path("a") / "b"):
            pytest.skip("backslash is not a separator on this platform")
        assert check_input_image(raw) == f.resolve()

    def test_round_trip_from_a_scoped_history_meta(self, home, tmp_path):
        from localm.media.paths import check_input_image
        from localm.plugins.builtin.image import plug as image_plug
        src = self._gallery_file(home)
        out = image_plug._images_dir() / "made.png"
        out.write_bytes(b"x")
        out.with_suffix(".png.json").write_text(
            json.dumps({"input_image": str(src)}), encoding="utf-8")
        app = _gui_app(tmp_path, "image")
        with TestClient(app) as c:
            items = c.get("/api/imagine/history",
                          headers=_h(_key(["image"]))).json()["images"]
        shown = next(it for it in items if it["name"] == "made.png")["meta"]["input_image"]
        _assert_scrubbed(shown, home, tmp_path)
        assert check_input_image(shown) == src.resolve()

    def test_round_trip_from_the_scrubber(self, home):
        from localm.media.paths import check_input_image
        from localm.pathscrub import scrub_paths
        f = self._gallery_file(home)
        assert check_input_image(scrub_paths(str(f))) == f.resolve()

    @pytest.mark.parametrize("tail", ["/config.json", "/../x.png", "/images/../config.json"])
    def test_scrubbed_form_outside_the_allowed_roots_is_refused(self, home, tail):
        from localm.media.paths import InputImageRefused, check_input_image
        (home / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(InputImageRefused) as e:
            check_input_image(DATA_DIR_LABEL + tail)
        assert str(home) not in str(e.value)

    def test_missing_file_message_echoes_the_callers_text(self, home):
        from localm.media import paths
        from localm.media.paths import InputImageRefused, check_input_image
        paths.gallery_dir(paths.IMAGE_DIR_NAME).mkdir(parents=True, exist_ok=True)
        raw = f"{DATA_DIR_LABEL}/{paths.IMAGE_DIR_NAME}/nope.png"
        with pytest.raises(InputImageRefused) as e:
            check_input_image(raw)
        assert raw in str(e.value)
        assert str(home) not in str(e.value)
