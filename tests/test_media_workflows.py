# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-plugin media workflow management (localm.media_workflows).

Users upload ComfyUI API-format workflows and select which one a media plugin
uses. These tests pin: upload validation (reject non-JSON / non-workflow),
the select/clear config marker, active-file resolution + the generator fallback
chain, delete refusing the active file, and path-traversal safety.
"""

from __future__ import annotations

import json

import pytest

from localm import media_workflows as mw


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


_WF = json.dumps({"3": {"class_type": "KSampler", "inputs": {}},
                  "4": {"class_type": "SaveImage", "inputs": {}}}).encode()


def test_is_workflow_json():
    assert mw.is_workflow_json({"1": {"class_type": "X"}})
    assert not mw.is_workflow_json({})
    assert not mw.is_workflow_json([1, 2])
    assert not mw.is_workflow_json({"a": {"no": "class_type"}})


def test_save_validates_and_normalizes_name():
    name = mw.save_workflow("image", "my-flux", _WF)   # no .json suffix
    assert name == "my-flux.json"
    assert (mw.workflows_dir("image") / "my-flux.json").is_file()


def test_save_rejects_non_json_and_non_workflow():
    with pytest.raises(ValueError):
        mw.save_workflow("image", "bad", b"not json{{")
    with pytest.raises(ValueError):
        mw.save_workflow("image", "plain", json.dumps({"hello": "world"}).encode())


def test_select_sets_and_clears_marker_and_resolves_active():
    mw.save_workflow("image", "a.json", _WF)
    assert mw.active_workflow_path("image") is None      # nothing selected yet
    mw.select_workflow("image", "a.json")
    assert mw.selected_name("image") == "a.json"
    p = mw.active_workflow_path("image")
    assert p is not None and p.name == "a.json"
    # clearing falls back (None) so the generator uses its legacy/example template
    mw.select_workflow("image", None)
    assert mw.selected_name("image") is None
    assert mw.active_workflow_path("image") is None


def test_select_unknown_rejected():
    with pytest.raises(ValueError):
        mw.select_workflow("image", "ghost.json")


def test_list_flags_active():
    mw.save_workflow("video", "one.json", _WF)
    mw.save_workflow("video", "two.json", _WF)
    mw.select_workflow("video", "two.json")
    by_name = {w["name"]: w for w in mw.list_workflows("video")}
    assert by_name["two.json"]["is_active"] is True
    assert by_name["one.json"]["is_active"] is False


def test_delete_refuses_active_and_removes_inactive():
    mw.save_workflow("music", "keep.json", _WF)
    mw.save_workflow("music", "drop.json", _WF)
    mw.select_workflow("music", "keep.json")
    with pytest.raises(ValueError):
        mw.delete_workflow("music", "keep.json")          # active -> refused
    mw.delete_workflow("music", "drop.json")              # inactive -> ok
    assert not (mw.workflows_dir("music") / "drop.json").is_file()


def test_delete_unknown_rejected():
    with pytest.raises(ValueError):
        mw.delete_workflow("image", "nope.json")


@pytest.mark.parametrize("bad", ["../escape.json", "a/b.json", "..", ""])
def test_path_traversal_rejected(bad):
    from fastapi import HTTPException
    with pytest.raises((HTTPException, ValueError)):
        mw.save_workflow("image", bad, _WF)


def test_router_endpoints_list_upload_select_delete():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(mw.make_workflow_router("image"))
    c = TestClient(app)

    assert c.get("/api/image/workflows").json() == {"workflows": [], "selected": None}

    wf = {"3": {"class_type": "KSampler", "inputs": {}}}
    r = c.post("/api/image/workflows",
               json={"name": "x", "workflow": wf, "activate": True})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "x.json" and r.json()["selected"] == "x.json"

    # bad uploads -> 400 with a reason
    assert c.post("/api/image/workflows",
                  json={"name": "y", "workflow": {"no": "nodes"}}).status_code == 400
    assert c.post("/api/image/workflows", json={"name": "y"}).status_code == 400

    # cannot delete the active one; clear then delete
    assert c.delete("/api/image/workflows/x.json").status_code == 400
    assert c.post("/api/image/workflows/select", json={"name": None}).json()["selected"] is None
    assert c.delete("/api/image/workflows/x.json").status_code == 200
    assert c.get("/api/image/workflows").json()["workflows"] == []


def test_generator_uses_selected_workflow():
    # The image generator's _workflow_path() resolves the selected file first.
    from localm.image_gen import comfy
    custom = mw.save_workflow("image", "custom.json", _WF)
    mw.select_workflow("image", custom)
    assert comfy._workflow_path() == mw.active_workflow_path("image")
    # cleared -> falls back to the committed example (no personal override here)
    mw.select_workflow("image", None)
    assert comfy._workflow_path() == comfy._WORKFLOW_EXAMPLE_PATH
