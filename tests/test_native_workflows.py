# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native workflows: validation, storage, built-ins, model resolution with the
downloads offered for missing models, the routes and the preflight."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.media import native_workflows as nw

SHA = "a" * 64


def _wf(**over):
    data = {"format": nw.FORMAT, "version": nw.VERSION, "media": "image", "name": "Mine",
            "description": "d", "models": {"model": {"name": "m1"}}, "loras": [],
            "params": {"steps": 8, "cfg": 2.5}}
    data.update(over)
    return data


def _src(**over):
    s = {"repo": "owner/repo", "file": "sub/model.gguf", "sha256": SHA, "size_bytes": 10,
         "model_type": "diffusion-unet"}
    s.update(over)
    return s


def _file(tmp_path, name, data=b"x"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


class TestParse:
    def test_valid_workflow_is_normalised(self):
        wf = nw.parse_workflow(_wf(models={"model": {"name": "m1", "source": _src()},
                                           "vae": "v1", "clip_l": ""},
                                   loras=[{"name": "l1", "strength": 0.5}],
                                   params={"steps": 8, "cfg": 2, "negative_prompt": "",
                                           "sample_method": "euler_a", "width": 512}),
                               "image")
        assert wf["models"] == {"model": {"name": "m1", "file": "model.gguf",
                                          "source": _src()},
                                "vae": {"name": "v1"}}
        assert wf["loras"] == [{"name": "l1", "strength": 0.5}]
        assert wf["params"] == {"steps": 8, "cfg": 2.0, "sample_method": "euler_a",
                                "width": 512}
        assert isinstance(wf["params"]["cfg"], float)

    @pytest.mark.parametrize("over,msg", [
        ({"format": "x"}, "format"),
        ({"version": 2}, "version"),
        ({"media": "video"}, "video workflow"),
        ({"name": "  "}, "empty"),
        ({"name": "x" * 101}, "longer"),
        ({"name": "a\x00b"}, "control"),
        ({"extra": 1}, "Unknown workflow field"),
        ({"models": {"bogus": "x"}}, "Unknown image model slot"),
        ({"models": {"model": {"name": "a/b"}}}, "usable model name"),
        ({"models": {"model": {"name": "m", "file": "dir/x.gguf"}}}, "plain file name"),
        ({"models": {"model": {"name": "m", "colour": 1}}}, "unknown field"),
        ({"params": {"bogus": 1}}, "Unknown image workflow parameter"),
        ({"params": {"steps": 151}}, "between"),
        ({"params": {"steps": 2.5}}, "whole number"),
        ({"params": {"steps": True}}, "number"),
        ({"params": {"cfg": float("nan")}}, "number"),
        ({"params": {"sample_method": "dpm_fast"}}, "Unknown sampler"),
        ({"params": {"flow_shift": 3}}, "Unknown image workflow parameter"),
        ({"loras": [{"name": f"l{i}"} for i in range(9)]}, "At most"),
        ({"loras": [{"name": "l", "strength": 5}]}, "strength"),
    ])
    def test_invalid_workflow_is_refused_with_a_reason(self, over, msg):
        with pytest.raises(ValueError) as e:
            nw.parse_workflow(_wf(**over), "image")
        assert msg in str(e.value)

    def test_music_workflow_takes_no_loras(self):
        with pytest.raises(ValueError, match="no LoRAs"):
            nw.parse_workflow(_wf(media="music", models={}, params={},
                                  loras=[{"name": "l"}]), "music")

    @pytest.mark.parametrize("source", [
        _src(repo="https://evil.example/x"),
        _src(repo="owner"),
        _src(repo="../x"),
        _src(file="../model.gguf"),
        _src(file="C:/models/model.gguf"),
        _src(file="/etc/model.gguf"),
        _src(file="sub/model.bin"),
        _src(file="model.ckpt"),
        _src(file="model.pt"),
        _src(file="a\\b.gguf"),
        _src(sha256="xyz"),
        _src(sha256="a" * 63),
        _src(size_bytes=-1),
        _src(size_bytes="10"),
        _src(model_type="llm-runner"),
        dict(_src(), url="http://x"),
        "owner/repo:model.gguf",
    ])
    def test_unsafe_download_source_is_refused(self, source):
        with pytest.raises(ValueError):
            nw.parse_workflow(_wf(models={"model": {"name": "m", "source": source}}), "image")

    def test_sha256_is_lowercased_and_optional(self):
        wf = nw.parse_workflow(_wf(models={"model": {"name": "m", "source": _src(
            sha256="A" * 64)}, "vae": {"name": "v", "source": _src(sha256=None)}}), "image")
        assert wf["models"]["model"]["source"]["sha256"] == "a" * 64
        assert wf["models"]["vae"]["source"]["sha256"] is None

    def test_video_and_music_params(self):
        v = nw.parse_workflow(_wf(media="video", models={}, params={
            "flow_shift": 3, "seconds": 2.5, "fps": 16, "scheduler": "simple"}), "video")
        assert v["params"] == {"flow_shift": 3.0, "seconds": 2.5, "fps": 16,
                               "scheduler": "simple"}
        m = nw.parse_workflow(_wf(media="music", models={"dit": "d"}, params={
            "duration_seconds": 60, "shift": 3, "tags": "lofi", "lyrics": "la\nla"}), "music")
        assert m["params"]["lyrics"] == "la\nla" and m["models"] == {"dit": {"name": "d"}}
        with pytest.raises(ValueError, match="Unknown music model slot"):
            nw.parse_workflow(_wf(media="music", models={"model": "x"}, params={}), "music")
        with pytest.raises(ValueError, match="between"):
            nw.parse_workflow(_wf(media="video", models={}, params={"seconds": 21}), "video")

    def test_load_json_text_limits(self):
        with pytest.raises(ValueError, match="larger"):
            nw.load_json_text(" " * (nw.MAX_FILE_BYTES + 1))
        with pytest.raises(ValueError, match="JSON"):
            nw.load_json_text("{")
        with pytest.raises(ValueError, match="nested too deeply"):
            nw.load_json_text("[" * 100000 + "]" * 100000)


class TestBuiltins:
    def test_each_media_has_starters(self):
        ids = {m: [w["id"] for w in nw.builtin_workflows(m)] for m in nw.MEDIA_TYPES}
        assert ids == {"image": ["builtin:sd-turbo", "builtin:flux1-schnell-q4",
                                 "builtin:flux1-dev-q4"],
                       "video": ["builtin:wan2.1-t2v-1.3b"],
                       "music": ["builtin:ace-step-1.5-turbo"]}

    def test_every_builtin_model_has_a_pinned_source(self):
        for media in nw.MEDIA_TYPES:
            for wf in nw.builtin_workflows(media):
                assert wf["builtin"] is True and wf["models"]
                for ref in wf["models"].values():
                    src = ref["source"]
                    assert len(src["sha256"]) == 64 and src["size_bytes"] > 0
                    assert ref["file"] == src["file"].rsplit("/", 1)[-1]

    def test_flux_workflows_pair_the_flux_components(self):
        from localm.media.sdcpp import pairing
        wf = nw.get_workflow("image", "builtin:flux1-dev-q4")
        assert wf["models"]["model"]["source"] == nw.FLUX_DEV_Q4
        for slot in ("clip_l", "t5xxl", "vae"):
            d = pairing.suggested_download(slot, "flux")
            assert wf["models"][slot]["source"]["sha256"] == d.sha256
        assert wf["params"]["guidance"] == 3.5
        assert nw.get_workflow("image", "builtin:flux1-schnell-q4")["params"]["steps"] == 4

    def test_builtin_sources_agree_with_every_other_pin_of_the_same_file(self):
        from localm.model_manager.registry import COMFY_MODEL_SOURCES
        from localm.media.sdcpp import pairing
        pins = {}
        for d in pairing.all_downloads():
            pins.setdefault(d.filename, set()).add(d.sha256)
        for name, s in COMFY_MODEL_SOURCES.items():
            pins.setdefault(name, set()).add(s.sha256)
        for media in nw.MEDIA_TYPES:
            for wf in nw.builtin_workflows(media):
                for ref in wf["models"].values():
                    assert pins.get(ref["file"], {ref["source"]["sha256"]}) == {
                        ref["source"]["sha256"]}, ref["file"]

    def test_builtins_cannot_be_changed(self):
        with pytest.raises(ValueError, match="cannot be renamed"):
            nw.rename_workflow("image", "builtin:sd-turbo", "x")
        with pytest.raises(ValueError, match="cannot be deleted"):
            nw.delete_workflow("image", "builtin:sd-turbo")
        with pytest.raises(ValueError, match="cannot be changed"):
            nw.save_workflow("image", _wf(), wid="builtin:sd-turbo")
        with pytest.raises(nw.WorkflowNotFound):
            nw.get_workflow("image", "builtin:nope")


class TestStorage:
    def test_save_list_get_rename_delete(self):
        wid = nw.save_workflow("image", _wf(name="My FLUX: v2"))
        assert wid == "My-FLUX-v2"
        path = nw.native_dir("image") / f"{wid}.json"
        assert path.is_file() and json.loads(path.read_text())["name"] == "My FLUX: v2"
        user = [w for w in nw.list_workflows("image") if not w["builtin"]]
        assert user == [{"id": wid, "name": "My FLUX: v2", "description": "d",
                         "builtin": False}]
        nw.rename_workflow("image", wid, "  Renamed ")
        assert nw.get_workflow("image", wid)["name"] == "Renamed"
        nw.save_workflow("image", _wf(name="Renamed", params={"steps": 3}), wid=wid)
        assert nw.get_workflow("image", wid)["params"] == {"steps": 3}
        nw.delete_workflow("image", wid)
        assert not path.exists()
        with pytest.raises(nw.WorkflowNotFound):
            nw.delete_workflow("image", wid)

    def test_same_name_gets_a_new_id(self):
        assert nw.save_workflow("video", _wf(media="video", models={}, name="a b")) == "a-b"
        assert nw.save_workflow("video", _wf(media="video", models={}, name="a b")) == "a-b-1"
        assert nw.save_workflow("video", _wf(media="video", models={}, name="..")) == "workflow"

    def test_export_is_importable(self):
        wid = nw.save_workflow("image", _wf())
        data = nw.export_data(nw.get_workflow("image", wid))
        assert "id" not in data and "builtin" not in data
        wid2, report = nw.import_workflow("image", data)
        assert wid2 != wid and report["unsupported"] == []
        assert nw.get_workflow("image", wid2)["models"] == nw.get_workflow("image", wid)["models"]

    @pytest.mark.parametrize("wid", ["../x", "a/b", "a\\b", "", "con", ".hidden", "x" * 81,
                                     "C:evil", None])
    def test_bad_ids_are_not_found(self, wid):
        with pytest.raises(nw.WorkflowNotFound):
            nw.get_workflow("image", wid)

    def test_unreadable_file_is_listed_with_its_error(self):
        d = nw.native_dir("image")
        d.mkdir(parents=True)
        (d / "broken.json").write_text("{", encoding="utf-8")
        (d / "other.json").write_text(json.dumps({"format": "x"}), encoding="utf-8")
        user = {w["id"]: w for w in nw.list_workflows("image") if not w["builtin"]}
        assert "JSON" in user["broken"]["error"] and "format" in user["other"]["error"]
        with pytest.raises(ValueError):
            nw.get_workflow("image", "broken")

    def test_comfyui_workflow_listing_does_not_see_native_files(self):
        from localm import media_workflows
        nw.save_workflow("image", _wf())
        media_workflows.save_workflow("image", "comfy.json",
                                      json.dumps({"1": {"class_type": "X"}}).encode())
        assert [w["name"] for w in media_workflows.list_workflows("image")] == ["comfy.json"]
        assert [w["id"] for w in nw.list_workflows("image") if not w["builtin"]] == ["Mine"]

    def test_workflow_limit(self, monkeypatch):
        monkeypatch.setattr(nw, "MAX_WORKFLOWS", 2)
        nw.save_workflow("image", _wf())
        nw.save_workflow("image", _wf())
        with pytest.raises(ValueError, match="already 2"):
            nw.save_workflow("image", _wf())


class TestResolve:
    def test_registered_names_resolve_and_missing_models_get_their_download(self, tmp_path):
        m1 = _file(tmp_path, "m1.gguf")
        lora = _file(tmp_path, "style.safetensors")
        other = _file(tmp_path, "ae.safetensors")
        reg = {"m1": {"path": str(m1), "model_type": "diffusion-unet"},
               "style": {"path": str(lora), "model_type": "lora"},
               "my-ae": {"path": str(other), "model_type": "vae"},
               "gone": {"path": str(tmp_path / "gone.gguf"), "model_type": "vae"}}
        wf = nw.parse_workflow(_wf(models={
            "model": {"name": "m1"},
            "vae": {"name": "ae", "source": nw.get_workflow(
                "image", "builtin:flux1-dev-q4")["models"]["vae"]["source"]},
            "clip_l": {"name": "gone", "source": _src(file="clip_l_custom.safetensors",
                                                     model_type="text-encoder")},
            "t5xxl": {"name": "t5-unknown"},
        }, loras=[{"name": "style", "strength": 0.7},
                  {"name": "m1", "strength": 1}]), "image")
        res = nw.resolve(wf, "image", reg)
        assert res["native_models"] == {"model": "m1", "vae": "my-ae"}
        assert res["loras"] == [{"name": "style", "strength": 0.7}]
        assert any("my-ae" in n for n in res["notes"])
        assert any("not as a LoRA" in n for n in res["notes"])
        by_slot = {m["slot"]: m for m in res["missing"]}
        assert set(by_slot) == {"clip_l", "t5xxl", "lora 2"}
        clip = by_slot["clip_l"]
        assert clip["native"] is True and clip["filename"] == "clip_l_custom.safetensors"
        assert clip["source"] == {"repo": "owner/repo", "file": "sub/clip_l_custom.safetensors",
                                  "spec": "owner/repo:sub/clip_l_custom.safetensors",
                                  "sha256": SHA, "size_bytes": 10,
                                  "model_type": "text-encoder", "name": None,
                                  "origin": "workflow"}
        assert by_slot["t5xxl"]["source"] is None and by_slot["t5xxl"]["searchable"] is False

    def test_builtin_sources_are_curated_and_register_under_the_workflow_name(self):
        res = nw.resolve(nw.get_workflow("image", "builtin:sd-turbo"), "image", {})
        (m,) = res["missing"]
        assert m["source"]["origin"] == "curated" and m["source"]["name"] == "sd-turbo"
        assert m["source"]["spec"] == "Green-Sky/SD-Turbo-GGUF:sd_turbo-f16-q8_0.gguf"
        assert res["native_models"] == {} and res["params"]["steps"] == 2

    def test_a_pinned_file_from_another_repo_is_not_curated(self):
        wf = nw.parse_workflow(_wf(models={"vae": {"name": "ae", "source": _src(
            file="ae.safetensors", model_type="vae")}}), "image")
        (m,) = nw.resolve(wf, "image", {})["missing"]
        assert m["source"]["origin"] == "workflow"

    def test_imported_model_without_source_is_searchable(self):
        wf = nw.parse_workflow(_wf(models={"model": {
            "name": "dreamy-v3", "file": "dreamy-v3.safetensors",
            "comfy": {"class_type": "CheckpointLoaderSimple", "input_name": "ckpt_name"}}}),
            "image")
        (m,) = nw.resolve(wf, "image", {})["missing"]
        assert m["searchable"] is True and m["class_type"] == "CheckpointLoaderSimple"
        assert m["input_name"] == "ckpt_name" and m["source"] is None
        wf["models"]["model"]["file"] = "model.safetensors"
        (m,) = nw.resolve(wf, "image", {})["missing"]
        assert m["searchable"] is False and m["reason"] == "name"

    def test_pull_name_is_dropped_when_taken_or_unsafe(self, tmp_path):
        wf = nw.parse_workflow(_wf(models={"model": {"name": "taken", "source": _src()},
                                           "vae": {"name": "has space", "source": _src(
                                               file="v.safetensors", model_type="vae")}}),
                               "image")
        reg = {"taken": {"path": str(tmp_path / "nothing.gguf"), "model_type": "vae"}}
        names = {m["slot"]: m["source"]["name"] for m in nw.resolve(wf, "image", reg)["missing"]}
        assert names == {"model": None, "vae": None}


def _client():
    app = FastAPI()
    for media in nw.MEDIA_TYPES:
        app.include_router(nw.make_router(media))
    return TestClient(app)


class TestRoutes:
    def test_crud_and_resolve(self):
        with _client() as c:
            r = c.get("/api/image/native-workflows")
            assert r.status_code == 200
            assert [w["id"] for w in r.json()["workflows"]][:1] == ["builtin:sd-turbo"]
            r = c.post("/api/image/native-workflows", json={"workflow": _wf()})
            assert r.status_code == 200, r.text
            wid = r.json()["id"]
            assert c.get(f"/api/image/native-workflows/{wid}").json()["workflow"]["name"] == "Mine"
            r = c.post(f"/api/image/native-workflows/{wid}/rename", json={"name": "New"})
            assert r.status_code == 200 and any(w["name"] == "New" for w in r.json()["workflows"])
            r = c.post(f"/api/image/native-workflows/{wid}/resolve")
            assert r.status_code == 200 and r.json()["missing"][0]["name"] == "m1"
            r = c.get("/api/image/native-workflows/builtin%3Aflux1-dev-q4")
            assert r.status_code == 200 and r.json()["builtin"] is True
            assert c.delete(f"/api/image/native-workflows/{wid}").status_code == 200
            assert c.get(f"/api/image/native-workflows/{wid}").status_code == 404

    def test_errors(self):
        with _client() as c:
            assert c.post("/api/image/native-workflows", json={}).status_code == 400
            r = c.post("/api/image/native-workflows", json={"workflow": _wf(params={"x": 1})})
            assert r.status_code == 400 and "Unknown" in r.json()["detail"]
            assert c.post("/api/image/native-workflows",
                          json={"workflow": _wf(), "id": "nope"}).status_code == 404
            assert c.delete("/api/image/native-workflows/builtin:sd-turbo").status_code == 400
            assert c.get("/api/video/native-workflows/..%2Fx").status_code == 404
            r = c.post("/api/image/native-workflows/import", json={"data": {"a": 1}})
            assert r.status_code == 400
            r = c.post("/api/music/native-workflows/import", json={"data": _wf()})
            assert r.status_code == 400 and "image workflow" in r.json()["detail"]

    def test_import_native_and_comfy(self):
        with _client() as c:
            r = c.post("/api/image/native-workflows/import", json={"data": _wf(), "name": "Got"})
            assert r.status_code == 200 and r.json()["report"]["kind"] == "native"
            assert c.get(f"/api/image/native-workflows/{r.json()['id']}").json()[
                "workflow"]["name"] == "Got"
            comfy = {"1": {"class_type": "KSampler", "inputs": {"steps": 7, "cfg": 3.0}},
                     "2": {"class_type": "Wobble", "inputs": {}}}
            r = c.post("/api/image/native-workflows/import", json={"data": comfy})
            assert r.status_code == 200, r.text
            assert r.json()["report"]["unsupported"] == [
                {"node": "2", "class_type": "Wobble", "title": ""}]


@pytest.fixture
def gui_app(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)

    async def switch_model(name):
        return None

    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=switch_model,
               active_model=lambda: "model-a")
    return app


class TestPreflight:
    def test_workflow_missing_models_are_offered(self, gui_app):
        with TestClient(gui_app) as c:
            r = c.post("/api/media/video/preflight",
                       json={"native_workflow": "builtin:wan2.1-t2v-1.3b"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "verified"
        assert {m["slot"] for m in body["missing"]} == {"model", "vae", "t5xxl"}
        assert all(m["native"] and m["source"]["sha256"] and m["source"]["origin"] == "curated"
                   for m in body["missing"])

    def test_workflow_with_everything_registered_misses_nothing(self, gui_app, tmp_path):
        from localm.config import save_registry
        reg = {}
        for ref in nw.get_workflow("image", "builtin:sd-turbo")["models"].values():
            reg[ref["name"]] = {"path": str(_file(tmp_path, ref["file"])),
                                "model_type": "diffusion-unet"}
        save_registry(reg)
        with TestClient(gui_app) as c:
            r = c.post("/api/media/image/preflight", json={"native_workflow": "builtin:sd-turbo"})
        assert r.json() == {"status": "verified", "missing": [], "warning": None}

    def test_unknown_workflow_is_unavailable_not_verified(self, gui_app):
        with TestClient(gui_app) as c:
            r = c.post("/api/media/image/preflight", json={"native_workflow": "nope"})
        assert r.json()["status"] == "unavailable" and "nope" in r.json()["warning"]

    def test_bad_workflow_file_is_unavailable(self, gui_app):
        d = nw.native_dir("music")
        d.mkdir(parents=True)
        (d / "bad.json").write_text("{", encoding="utf-8")
        with TestClient(gui_app) as c:
            r = c.post("/api/media/music/preflight", json={"native_workflow": "bad"})
        assert r.json()["status"] == "unavailable"


def test_known_sources_cover_the_curated_comfy_files():
    from localm.model_manager.registry import COMFY_MODEL_SOURCES
    known = nw.known_sources()
    for name in COMFY_MODEL_SOURCES:
        assert known[name]["sha256"]
    assert known["flux1-schnell-Q4_K_S.gguf"] == nw.FLUX_SCHNELL_Q4
    assert Path(nw.FLUX_DEV_Q4["file"]).suffix == ".gguf"
