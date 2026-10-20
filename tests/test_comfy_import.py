# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI workflow import into native workflows: API and UI format, every
mapped node, and the report of what could not be kept."""

import copy

import pytest

from localm.media import comfy_import as ci
from localm.media import native_workflows as nw
from localm.media.sdcpp import sampling


def _flux_api():
    return {
        "10": {"class_type": "UNETLoader",
               "inputs": {"unet_name": "flux1-dev-Q8_0.gguf", "weight_dtype": "default"}},
        "11": {"class_type": "DualCLIPLoader",
               "inputs": {"clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                          "clip_name2": "clip_l.safetensors", "type": "flux"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "13": {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["10", 0], "lora_name": "styles/inkwash-style.safetensors",
                          "strength_model": 0.8}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "a red fox", "clip": ["11", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["11", 0]}},
        "26": {"class_type": "FluxGuidance",
               "inputs": {"conditioning": ["6", 0], "guidance": 3.0}},
        "27": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": 1024, "height": 768, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"model": ["13", 0], "seed": 42, "steps": 24, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                         "positive": ["26", 0], "negative": ["7", 0],
                         "latent_image": ["27", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["12", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0],
                                                    "filename_prefix": "x"}},
        "40": {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": "4x.pth"}, "_meta": {"title": "Upscaler"}},
    }


def _ui(api: dict, *, control: bool = True, extra_nodes=()) -> dict:
    """The UI-format graph of API-format workflow *api*: widget values in the
    order ComfyUI keeps them, inputs wired through ``links``."""
    nodes, links = [], []
    link_id = 0
    for nid, node in api.items():
        widgets = []
        names = list(ci.WIDGETS.get(node["class_type"], ()))
        for name in names:
            if name == "control_after_generate":
                if control:
                    widgets.append("randomize")
                continue
            v = node["inputs"].get(name)
            if isinstance(v, list):
                v = None
            widgets.append(v)
        inputs = []
        for name, v in node["inputs"].items():
            if isinstance(v, list):
                link_id += 1
                links.append([link_id, int(v[0]), v[1], int(nid), len(inputs), "X"])
                inputs.append({"name": name, "type": "X", "link": link_id})
        nodes.append({"id": int(nid), "type": node["class_type"], "inputs": inputs,
                      "widgets_values": widgets, "mode": 0,
                      "title": (node.get("_meta") or {}).get("title")})
    nodes.extend(extra_nodes)
    return {"last_node_id": 99, "nodes": nodes, "links": links, "version": 0.4,
            "extra": {"title": "Fox"}}


def _check_flux(wf, report):
    assert wf["models"]["model"]["file"] == "flux1-dev-Q8_0.gguf"
    assert wf["models"]["model"]["source"]["sha256"] == (
        "129032f32224bf7138f16e18673d8008ba5f84c1ec74063bf4511a8bb4cf553d")
    assert wf["models"]["clip_l"]["file"] == "clip_l.safetensors"
    assert wf["models"]["t5xxl"]["file"] == "t5xxl_fp8_e4m3fn.safetensors"
    assert wf["models"]["vae"]["source"]["sha256"].startswith("afc8e282")
    (lora,) = wf["loras"]
    assert lora["file"] == "inkwash-style.safetensors" and lora["strength"] == 0.8
    assert lora["comfy"] == {"class_type": "LoraLoaderModelOnly", "input_name": "lora_name"}
    assert wf["params"] == {"steps": 24, "cfg": 1.0, "sample_method": "euler",
                            "scheduler": "simple", "prompt": "a red fox",
                            "negative_prompt": "blurry", "guidance": 3.0,
                            "width": 1024, "height": 768}
    assert {i["class_type"] for i in report["implicit"]} == {"VAEDecode", "SaveImage"}
    assert report["unsupported"] == [{"node": "40", "class_type": "UpscaleModelLoader",
                                      "title": "Upscaler"}]
    assert "UpscaleModelLoader" in wf["description"]
    assert any("seed" in n for n in report["notes"])


def test_api_format_flux_graph():
    wf, report = ci.convert(_flux_api(), "image", reg={})
    assert report["kind"] == "comfy-api"
    _check_flux(wf, report)
    assert wf["name"] == "Imported ComfyUI workflow"


@pytest.mark.parametrize("control", [True, False])
def test_ui_format_flux_graph_maps_the_same(control):
    wf, report = ci.convert(_ui(_flux_api(), control=control), "image", reg={})
    assert report["kind"] == "comfy-ui"
    _check_flux(wf, report)
    assert wf["name"] == "Fox"


def test_ui_widget_names_follow_the_seed_control():
    assert ci._widget_names("KSampler", [1, "fixed", 20])[:3] == ["seed",
                                                                   "control_after_generate",
                                                                   "steps"]
    assert ci._widget_names("KSampler", [1, 20])[:2] == ["seed", "steps"]


def test_every_mapped_sampler_and_scheduler_is_a_native_name():
    assert set(ci.SAMPLER_NAMES.values()) <= set(sampling.SAMPLE_METHODS)
    assert set(ci.SCHEDULER_NAMES.values()) <= set(sampling.SCHEDULERS)


def test_registered_file_keeps_its_registry_name(tmp_path):
    f = tmp_path / "flux1-dev-Q8_0.gguf"
    f.write_bytes(b"x")
    wf, _ = ci.convert(_flux_api(), "image",
                       reg={"my-flux": {"path": str(f), "model_type": "diffusion-unet"}})
    assert wf["models"]["model"]["name"] == "my-flux"


def test_wan_video_graph():
    api = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "wan2.1_t2v_1.3B_fp16.safetensors",
                         "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "static", "clip": ["2", 0]}},
        "7": {"class_type": "EmptyHunyuanLatentVideo",
              "inputs": {"width": 832, "height": 480, "length": 33, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "seed": 1, "steps": 30, "cfg": 6.0,
                         "sampler_name": "uni_pc", "scheduler": "ddim_uniform",
                         "denoise": 1.0, "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": 16.0}},
        "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0]}},
    }
    for data in (api, _ui(api)):
        wf, report = ci.convert(data, "video", reg={})
        assert set(wf["models"]) == {"model", "t5xxl", "vae"}
        assert wf["models"]["model"]["source"]["sha256"].startswith("be531024")
        assert wf["models"]["t5xxl"]["source"]["repo"] == "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
        assert wf["params"] == {"steps": 30, "cfg": 6.0, "prompt": "a cat",
                                "negative_prompt": "static", "flow_shift": 8.0,
                                "width": 832, "height": 480, "fps": 16, "seconds": 2.06}
        notes = " ".join(report["notes"])
        assert "uni_pc" in notes and "ddim_uniform" in notes and "fp8_e4m3fn" in notes
        assert report["unsupported"] == []


def test_seconds_round_trip_to_the_same_frame_count():
    from localm.plugins.builtin.video.backends.native import frame_count
    for length, fps in ((33, 16), (81, 16), (49, 24), (121, 24)):
        assert frame_count(round(length / fps, 2), fps) == length


def test_music_graph_keeps_params_and_explains_the_checkpoint():
    api = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ace_step_v1_3.5b.safetensors"}},
        "2": {"class_type": "TextEncodeAceStepAudio",
              "inputs": {"clip": ["1", 1], "tags": "lofi, piano", "lyrics": "[verse]\nla",
                         "lyrics_strength": 0.9}},
        "3": {"class_type": "EmptyAceStepLatentAudio", "inputs": {"seconds": 95.0,
                                                                  "batch_size": 1}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 5.0}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "seed": 3, "steps": 50, "cfg": 5.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                         "positive": ["2", 0], "negative": ["2", 0],
                         "latent_image": ["3", 0]}},
        "6": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveAudio", "inputs": {"audio": ["6", 0]}},
    }
    wf, report = ci.convert(api, "music", reg={})
    assert wf["models"] == {}
    assert wf["params"] == {"steps": 50, "cfg": 5.0, "shift": 5.0, "duration_seconds": 95.0,
                            "tags": "lofi, piano", "lyrics": "[verse]\nla"}
    notes = " ".join(report["notes"])
    assert "ACE-Step 1.5" in notes and "lyrics strength 0.9" in notes
    assert report["unsupported"] == []


def test_dropped_values_are_reported():
    api = _flux_api()
    api["3"]["inputs"].update(denoise=0.6, sampler_name="dpm_fast", steps=500)
    api["50"] = copy.deepcopy(api["3"])
    api["13"]["inputs"]["strength_model"] = 9.0
    api["60"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "mystery.safetensors", "type": "sd3"}}
    api["61"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "orphan"}}
    wf, report = ci.convert(api, "image", reg={})
    notes = " ".join(report["notes"])
    for needle in ("denoise 0.6", "dpm_fast", "between", "2 samplers", "clamped",
                   "mystery.safetensors", "not connected"):
        assert needle in notes, needle
    assert "steps" not in wf["params"] and wf["loras"][0]["strength"] == 4.0


def test_slot_without_a_place_in_the_media_is_reported():
    api = {"1": {"class_type": "DualCLIPLoader",
                 "inputs": {"clip_name1": "clip_l.safetensors",
                            "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}},
           "2": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}}}
    wf, report = ci.convert(api, "video", reg={})
    assert set(wf["models"]) == {"t5xxl", "vae"}
    assert any("no video slot" in n for n in report["notes"])


def test_primitive_linked_values_and_muted_nodes():
    api = _flux_api()
    ui = _ui(api)
    prim = {"id": 70, "type": "PrimitiveNode", "inputs": [], "widgets_values": [12],
            "mode": 0}
    muted = {"id": 71, "type": "LoraLoader", "inputs": [],
             "widgets_values": ["x.safetensors", 1, 1], "mode": 4}
    ui["nodes"] += [prim, muted]
    for node in ui["nodes"]:
        if node["type"] == "KSampler":
            ui["links"].append([500, 70, 0, node["id"], 9, "INT"])
            node["inputs"].append({"name": "steps", "type": "INT", "link": 500})
    wf, report = ci.convert(ui, "image", reg={})
    assert wf["params"]["steps"] == 12 and len(wf["loras"]) == 1
    assert any("bypassed" in n and "LoraLoader #71" in n for n in report["notes"])


def test_value_from_a_non_primitive_link_is_reported():
    api = _flux_api()
    api["6"]["inputs"]["text"] = ["80", 0]
    api["80"] = {"class_type": "StringConcat", "inputs": {}}
    wf, report = ci.convert(api, "image", reg={})
    assert "prompt" not in wf["params"]
    assert any("'text' comes from StringConcat #80" in n for n in report["notes"])
    assert {"node": "80", "class_type": "StringConcat", "title": ""} in report["unsupported"]


def test_cycles_and_dangling_links_terminate():
    api = {"1": {"class_type": "KSampler",
                 "inputs": {"model": ["2", 0], "positive": ["3", 0], "steps": 5,
                            "latent_image": ["9", 0]}},
           "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 1}},
           "3": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["3", 0],
                                                          "guidance": 2}}}
    wf, report = ci.convert(api, "image", reg={})
    assert wf["params"]["steps"] == 5 and wf["params"]["guidance"] == 2.0


def test_not_a_workflow_or_nothing_mapped():
    for data in ({}, [], {"nodes": "x"}, {"a": 1}, "text"):
        assert not ci.is_comfy_workflow(data)
        with pytest.raises(ValueError):
            ci.convert(data, "image", reg={})
    with pytest.raises(ValueError, match="maps to"):
        ci.convert({"1": {"class_type": "Wobble", "inputs": {}}}, "image", reg={})
    big = {str(i): {"class_type": "Note", "inputs": {}} for i in range(ci.MAX_NODES + 1)}
    with pytest.raises(ValueError, match="more than"):
        ci.convert(big, "image", reg={})


def test_import_stores_the_converted_workflow():
    wid, report = nw.import_workflow("image", _flux_api())
    assert report["kind"] == "comfy-api"
    wf = nw.get_workflow("image", wid)
    res = nw.resolve(wf, "image", {})
    lora = next(m for m in res["missing"] if m["slot"] == "lora 1")
    assert lora["searchable"] is True and lora["class_type"] == "LoraLoaderModelOnly"
    main = next(m for m in res["missing"] if m["slot"] == "model")
    assert main["source"]["origin"] == "curated"
    assert main["source"]["spec"] == "city96/FLUX.1-dev-gguf:flux1-dev-Q8_0.gguf"


def test_title_with_a_control_character_still_imports():
    ui = _ui(_flux_api())
    ui["extra"]["title"] = "Fox\x07 graph"
    wf, _ = ci.convert(ui, "image", reg={})
    assert wf["name"] == "Fox graph"


def test_more_dropped_values_are_reported():
    api = _flux_api()
    api["3"] = {"class_type": "KSamplerAdvanced",
                "inputs": {"model": ["13", 0], "add_noise": "disable", "noise_seed": 1,
                           "steps": 20, "cfg": 1.0, "sampler_name": "euler",
                           "scheduler": "simple", "positive": ["26", 0], "negative": ["7", 0],
                           "latent_image": ["27", 0], "start_at_step": 0, "end_at_step": 10000,
                           "return_with_leftover_noise": "enable"}}
    api["90"] = {"class_type": "ModelSamplingSD3", "inputs": {"shift": 2.0}}
    api["91"] = {"class_type": "ModelSamplingSD3", "inputs": {"shift": 3.0}}
    wf, report = ci.convert(api, "video", reg={})
    notes = " ".join(report["notes"])
    assert "add_noise 'disable'" in notes and "return_with_leftover_noise 'enable'" in notes
    assert "ModelSamplingSD3 #91: a second sampling shift" in notes
    assert wf["params"]["flow_shift"] == 2.0


def test_music_duration_conflict_and_second_encoder_are_reported():
    api = {
        "1": {"class_type": "EmptyAceStepLatentAudio", "inputs": {"seconds": 60.0}},
        "2": {"class_type": "TextEncodeAceStepAudio15",
              "inputs": {"tags": "a", "lyrics": "", "duration": 90.0}},
        "3": {"class_type": "TextEncodeAceStepAudio", "inputs": {"tags": "b", "lyrics": ""}},
    }
    wf, report = ci.convert(api, "music", reg={})
    notes = " ".join(report["notes"])
    assert wf["params"]["duration_seconds"] == 60.0 and wf["params"]["tags"] == "a"
    assert "duration 90 differs" in notes and "#3: a second one was not used" in notes


def test_huge_and_non_dict_values_are_notes_not_crashes():
    api = _flux_api()
    api["27"]["inputs"]["width"] = 10 ** 400
    api["13"]["inputs"]["strength_model"] = 10 ** 400
    api["50"] = {"class_type": "VAELoader", "inputs": ["ae.safetensors"]}
    wf, report = ci.convert(api, "image", reg={})
    assert "width" not in wf["params"] and wf["loras"][0]["strength"] == 1.0
    assert any("VAELoader #50: its inputs are not readable" in n for n in report["notes"])
