# SPDX-License-Identifier: AGPL-3.0-or-later
"""A key without the owner scope gets no host folder layout: host paths outside
the data dir, install root, env prefix and home (a model registered in place,
any other absolute path) reach it with their leading directories replaced by
``<path>``, in a bug report's log tail, in ``/api/models``'s ``last_path`` and
in the other routes such a key reaches. The owner keeps the full paths."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import pathscrub
from localm.pathscrub import PATH_LABEL, scrub_host_paths
from localm.plugins.gui.web import attach_gui

_WIN = sys.platform == "win32"


@pytest.fixture(autouse=True)
def _comfyui_is_set_up(monkeypatch):
    monkeypatch.setattr("localm.media.backend_choice.comfy_is_set_up",
                        lambda *a, **k: True)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "user" / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "user"))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", h)
    monkeypatch.setattr(_cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", h / "registry.json")
    return h


@pytest.fixture
def outside(tmp_path):
    """A folder outside the data dir, install root, env prefix and home."""
    d = tmp_path / "elsewhere" / "bin"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def spaced(tmp_path):
    """A model folder outside the data dir, install root, env prefix and home,
    with a space in one of its names."""
    d = tmp_path / "Model Store" / "gguf"
    d.mkdir(parents=True)
    return d


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _key(scopes):
    from localm import auth
    return auth.create_key("k", scopes, allow_privileged=True)["key"]


def _register(home, entries):
    (home / "registry.json").write_text(json.dumps(entries), encoding="utf-8")


def _gui_app(tmp_path, *plugins):
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    pm = PluginManager(app, external_root=tmp_path / "noplugins")
    for p in plugins:
        pm.install(p)

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "")
    return app


def _forms(folder: Path) -> list:
    raw = str(folder)
    return [raw, raw.replace("\\", "/"), raw.replace("\\", "\\\\")]


def _assert_no_folder(text: str, folder: Path) -> None:
    for form in _forms(folder):
        assert form not in text, f"host folder {form!r} reached a scoped key"


# --------------------------------------------------------------------------- #
#  The scrubber                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,want", [
    (r"model ready: D:\My Models\gguf\bge.gguf", r"model ready: <path>\bge.gguf"),
    (json.dumps({"p": r"D:\a b\c\x.gguf"}), json.dumps({"p": r"<path>\x.gguf"})),
    (r"copied D:\a\b.gguf to E:\c\d.gguf", r"copied <path>\b.gguf to <path>\d.gguf"),
    ("D:/x/y/z.gguf failed", "<path>/z.gguf failed"),
    (r"C:\Users\<redacted>\Downloads\m.gguf", r"<path>\m.gguf"),
    (r"see \\nas\models\q\m.gguf ok", r"see <path>\m.gguf ok"),
    (r"\\?\D:\deep\x.gguf", r"\\?\<path>\x.gguf"),
    (r"D:\x.gguf", r"<path>\x.gguf"),
    ("/mnt/models/x.gguf", "<path>/x.gguf"),
    ("PATH=/usr/bin:/usr/local/bin", "PATH=<path>/bin:<path>/bin"),
])
def test_host_paths_lose_their_folders(text, want):
    out = scrub_host_paths(text, {"mnt", "usr"})
    assert out == want
    assert scrub_host_paths(out, {"mnt", "usr"}) == out


@pytest.mark.parametrize("text", [
    r"<data>\models\x.gguf", r"<data>\\models\\x.gguf", r'File "<install>\localm\x.py"',
    "GET /api/models HTTP/1.1", "http://127.0.0.1:8080/v1/models", "~/foo/bar",
    "localm/inference/x.py", "/x.gguf", "a/b/c",
])
def test_text_that_is_not_a_host_path_is_kept(text):
    assert scrub_host_paths(text, {"mnt", "usr"}) == text


def test_an_unlistable_root_treats_every_posix_path_as_a_host_path(monkeypatch):
    def _boom(_p):
        raise OSError("no root")

    monkeypatch.setattr(pathscrub.os, "listdir", _boom)
    assert scrub_host_paths("GET /api/models/x") == "GET <path>/x"


def test_scrub_paths_drops_a_registry_folder_outside_the_known_roots(home, spaced):
    model = spaced / "bge-small-en-v1.5-q4_k_m.gguf"
    _register(home, {"bge": {"path": str(model), "source": "local"},
                     "inside": {"path": str(home / "models" / "i.gguf")}})
    text = f"ready: {model} and {home / 'models' / 'i.gguf'}"
    out = pathscrub.scrub_paths(text)
    _assert_no_folder(out, spaced)
    assert f"{PATH_LABEL}{os.sep}{model.name}" in out
    assert f"<data>{os.sep}models{os.sep}i.gguf" in out


# --------------------------------------------------------------------------- #
#  POST /api/bug-report with include_log                                       #
# --------------------------------------------------------------------------- #

def test_bug_report_log_tail_holds_no_host_path_for_a_scoped_key(home, tmp_path, spaced):
    from localm.inference.http_server import create_app
    model = spaced / "bge-small-en-v1.5-q4_k_m.gguf"
    _register(home, {"bge": {"path": str(model), "source": "local",
                             "model_type": "embedding"}})
    other = (Path(r"E:\Other Stuff\cache") if _WIN else Path("/usr/lib/other-cache"))
    logs = home / "logs"
    logs.mkdir()
    (logs / f"localm_2026-10-29_120000_{os.getpid()}.log").write_text(
        f"2026-10-29 12:00:00,001 INFO     localm: embedding model ready: {model}\n"
        f"2026-10-29 12:00:01,002 WARNING  localm.media: no cache at {other / 'thing.bin'}\n",
        encoding="utf-8")
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    scoped, admin = _key(["config:write"]), _key(["admin"])
    app = create_app(engine)
    with TestClient(app) as c:
        mine = c.post("/api/bug-report", headers=_h(scoped),
                      json={"message": "embedder", "include_log": True}).json()
        saved = (home / "bug-reports" / mine["filename"]).read_text(encoding="utf-8")
        full = c.post("/api/bug-report", headers=_h(admin),
                      json={"message": "embedder", "include_log": True}).json()
    text = mine["report_markdown"]
    assert saved == text
    assert "embedding model ready:" in text, "the log tail was not attached"
    _assert_no_folder(text, spaced)
    _assert_no_folder(text, other)
    assert f"embedding model ready: {PATH_LABEL}{os.sep}{model.name}" in text
    assert f"{PATH_LABEL}{os.sep}thing.bin" in text
    assert str(model) in full["report_markdown"], "the owner's report keeps the path"


# --------------------------------------------------------------------------- #
#  GET /api/models: last_path of a missing model                               #
# --------------------------------------------------------------------------- #

def test_models_last_path_holds_no_host_path_for_a_scoped_key(home, tmp_path, spaced):
    gone = spaced / "gone.gguf"
    _register(home, {"gone": {"path": str(gone), "source": "local"}})
    app = _gui_app(tmp_path)
    scoped, admin = _key(["models:read"]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.get("/api/models", headers=_h(scoped))
        full = c.get("/api/models", headers=_h(admin))
    assert mine.status_code == 200, mine.text
    row = next(m for m in mine.json()["models"] if m["name"] == "gone")
    assert row["missing"] is True
    assert row["last_path"] == f"{PATH_LABEL}{os.sep}gone.gguf"
    _assert_no_folder(mine.text, spaced)
    owner_row = next(m for m in full.json()["models"] if m["name"] == "gone")
    assert owner_row["last_path"] == str(gone)


# --------------------------------------------------------------------------- #
#  Siblings                                                                    #
# --------------------------------------------------------------------------- #

def test_rebuild_launcher_holds_no_host_path_for_a_scoped_key(home, tmp_path, outside,
                                                              monkeypatch):
    from localm import applaunch
    exe = outside / "LocaLM.exe"
    shortcut = outside / "Desktop" / "LocaLM.lnk"
    monkeypatch.setattr(applaunch, "make_launcher", lambda force=False: applaunch.LauncherResult(
        ok=True, path=exe, desktop_file=shortcut, notes=[f"wrote {exe}"]))
    app = _gui_app(tmp_path)
    scoped, admin = _key(["config:write"]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.post("/api/app/rebuild-launcher", headers=_h(scoped))
        full = c.post("/api/app/rebuild-launcher", headers=_h(admin)).json()
    assert mine.status_code == 200, mine.text
    _assert_no_folder(mine.text, outside)
    body = mine.json()
    assert body["path"] == f"{PATH_LABEL}{os.sep}LocaLM.exe"
    assert body["desktop_file"] == f"{PATH_LABEL}{os.sep}LocaLM.lnk"
    assert body["notes"] == [f"wrote {PATH_LABEL}{os.sep}LocaLM.exe"]
    assert full["path"] == str(exe) and full["desktop_file"] == str(shortcut)


def test_native_models_preview_holds_no_host_path_for_a_scoped_key(home, tmp_path, outside,
                                                                    monkeypatch):
    from localm.media.sdcpp import pairing
    hint = outside / "hint.bin"
    _register(home, {"m": {"path": str(outside / "m.gguf"), "source": "local",
                           "model_type": "diffusion"}})
    monkeypatch.setattr(pairing, "preview", lambda *a, **k: {
        "family": None, "variant": None, "defaults": None,
        "error": f"a helper file could not be written: {hint}"})
    app = _gui_app(tmp_path, "image")
    scoped, admin = _key(["image"]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.get("/api/imagine/native-models?model=m", headers=_h(scoped))
        full = c.get("/api/imagine/native-models?model=m", headers=_h(admin)).json()
    assert mine.status_code == 200, mine.text
    _assert_no_folder(mine.text, outside)
    assert mine.json()["preview"]["error"].endswith(f"{PATH_LABEL}{os.sep}hint.bin")
    assert full["preview"]["error"].endswith(str(hint))


def test_scrub_paths_without_a_readable_registry_still_scrubs_drive_paths(home, monkeypatch):
    import localm.config as _cfg

    def _boom():
        raise OSError("registry unreadable")

    monkeypatch.setattr(_cfg, "peek_registry", _boom)
    out = pathscrub.scrub_paths(r"load D:\Somewhere Else\m.gguf")
    assert out == r"load <path>\m.gguf"
