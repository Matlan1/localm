# SPDX-License-Identifier: AGPL-3.0-or-later
"""The native (stable-diffusion.cpp) image backend and how the image plugin
chooses it: ``auto`` resolution, explicit choices, model resolution, request
validation, the PNG/sidecar it writes, and the settings fields that configure
it. The worker is replaced by an in-process fake; the real worker is covered by
tests/test_sdcpp_runner.py."""

from __future__ import annotations

import io
import json
import sys
import struct
from pathlib import Path

import pytest

import localm.media.comfy_client as comfy_client
from localm.media import backend_choice
from localm.media.sdcpp import runtime as sd_runtime
from localm.media.sdcpp import shared
from localm.media.sdcpp.runner import SdCancelled, SdWorkerError
from localm.plugins.builtin.image import backend as image_backend
from localm.plugins.builtin.image.backends import native


# --------------------------------------------------------------------------- #
#  auto / explicit backend choice                                             #
# --------------------------------------------------------------------------- #

@pytest.fixture
def no_comfy(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    monkeypatch.setattr(backend_choice, "managed_comfy_active", lambda cfg=None: False)
    import localm.media.comfy_client as cc
    monkeypatch.setattr(cc, "is_comfy_confirmed", lambda api_url=None: False)


def test_auto_without_comfyui_is_native(no_comfy):
    s = image_backend.settings({})
    assert s["backend_choice"] == "auto"
    assert s["backend"] == "native"
    assert "no ComfyUI" in s["backend_note"]
    assert s["warning"] is None


@pytest.mark.parametrize("cfg", [
    {"comfy_target": "user"},
    {"comfy_launch_cmd": "run_comfy.bat"},
    {"comfy_api_url": "http://127.0.0.1:8188"},
    {"plugins": {"image": {"comfy": {"workdir": "C:/ComfyUI"}}}},
])
def test_auto_with_comfyui_configured_is_comfy(no_comfy, cfg):
    s = image_backend.settings(cfg)
    assert s["backend"] == "comfy"
    assert "ComfyUI is set up" in s["backend_note"]


def test_auto_with_a_recently_answering_comfyui_is_comfy(no_comfy, monkeypatch):
    import localm.media.comfy_client as cc
    monkeypatch.setattr(cc, "is_comfy_confirmed", lambda api_url=None: True)
    assert image_backend.settings({})["backend"] == "comfy"


def test_auto_with_managed_comfyui_is_comfy(no_comfy, monkeypatch):
    monkeypatch.setattr(backend_choice, "managed_comfy_active", lambda cfg=None: True)
    assert image_backend.settings({})["backend"] == "comfy"


def test_explicit_choices_are_never_overridden(no_comfy, monkeypatch):
    monkeypatch.setattr(backend_choice, "managed_comfy_active", lambda cfg=None: True)
    s = image_backend.settings({"plugins": {"image": {"backend": "native"}}})
    assert (s["backend"], s["backend_note"]) == ("native", None)
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: True)
    image_backend.prepare_for_job(s, {"plugins": {"image": {"backend": "native"}}})
    assert s["backend"] == "native"
    monkeypatch.setattr(backend_choice, "managed_comfy_active", lambda cfg=None: False)
    s = image_backend.settings({"plugins": {"image": {"backend": "comfy"}}})
    assert (s["backend"], s["backend_note"]) == ("comfy", None)


def test_auto_native_switches_to_a_comfyui_that_answers_at_job_time(no_comfy, monkeypatch):
    seen = []

    def alive(url, timeout=3.0):
        seen.append((url, timeout))
        return True

    monkeypatch.setattr(comfy_client, "_comfy_alive", alive)
    s = image_backend.prepare_for_job(image_backend.settings({}), {})
    assert s["backend"] == "comfy"
    assert "answered" in s["backend_note"]
    assert seen and seen[0][1] == backend_choice.AUTO_PROBE_TIMEOUT


def test_auto_native_stays_native_when_no_comfyui_answers(no_comfy, monkeypatch):
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: False)
    s = image_backend.prepare_for_job(image_backend.settings({}), {})
    assert s["backend"] == "native"


def test_the_facade_dispatches_native_to_the_native_module(no_comfy):
    s = image_backend.settings({})
    assert image_backend._impl(s).__name__.endswith("backends.native")


# --------------------------------------------------------------------------- #
#  model resolution                                                           #
# --------------------------------------------------------------------------- #

def _register(name: str, path: Path, model_type: str = "diffusion-unet") -> None:
    from localm.config import REGISTRY_FILE
    reg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8")) if REGISTRY_FILE.exists() else {}
    reg[name] = {"path": str(path), "model_type": model_type, "source": "test"}
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(reg), encoding="utf-8")


def test_no_model_names_the_recommended_download():
    with pytest.raises(native._ModelError) as ei:
        native.resolve_models({"native": {}})
    rec = native.RECOMMENDED_MODELS[0]
    assert f"localm pull {rec.spec} --type diffusion-unet" in str(ei.value)


def test_blank_model_picks_the_downloaded_recommended_model(tmp_path):
    rec = native.RECOMMENDED_MODELS[0]
    f = tmp_path / rec.file
    f.write_bytes(b"x")
    _register("sd-turbo", f)
    m = native.resolve_models({"native": {}})
    assert m["ctx"] == {"model_path": str(f)}
    assert m["label"] == "sd-turbo" and m["recommended"] is rec


def test_an_explicit_checkpoint_path_is_the_model_path(tmp_path):
    f = tmp_path / "my-model.safetensors"
    f.write_bytes(b"x")
    m = native.resolve_models({"native": {"model": str(f)}})
    assert m["ctx"] == {"model_path": str(f)}
    assert m["recommended"] is None


def test_text_encoders_make_the_model_a_diffusion_model(tmp_path):
    files = {k: tmp_path / f"{k}.gguf" for k in ("dit", "clip_l", "t5xxl", "vae")}
    for f in files.values():
        f.write_bytes(b"x")
    m = native.resolve_models({"native": {"model": str(files["dit"]),
                                          "clip_l": str(files["clip_l"]),
                                          "t5xxl": str(files["t5xxl"]),
                                          "vae": str(files["vae"])}})
    assert m["ctx"] == {"diffusion_model_path": str(files["dit"]),
                        "clip_l_path": str(files["clip_l"]),
                        "t5xxl_path": str(files["t5xxl"]), "vae_path": str(files["vae"])}


def test_a_vae_alone_keeps_the_checkpoint_path(tmp_path):
    ck, vae = tmp_path / "ck.gguf", tmp_path / "vae.gguf"
    ck.write_bytes(b"x")
    vae.write_bytes(b"x")
    m = native.resolve_models({"native": {"model": str(ck), "vae": str(vae)}})
    assert m["ctx"] == {"model_path": str(ck), "vae_path": str(vae)}


def test_a_missing_model_is_a_clear_error(tmp_path):
    with pytest.raises(native._ModelError, match="neither a registered model nor a file"):
        native.resolve_models({"native": {"model": str(tmp_path / "nope.gguf")}})


def test_a_directory_model_is_refused(tmp_path):
    d = tmp_path / "hf-dir"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(native._ModelError):
        native.resolve_models({"native": {"model": str(d)}})


# --------------------------------------------------------------------------- #
#  generate                                                                   #
# --------------------------------------------------------------------------- #

class _FakeRunner:
    def __init__(self, *, version="SD 2.x", image=True, raise_on_generate=None):
        self.version = version
        self.image = image
        self.raise_on_generate = raise_on_generate
        self.loaded_key = None
        self.loads = []
        self.generates = []
        self.alive = False
        self.pid = 4242

    def is_alive(self):
        return self.alive

    def ensure_loaded(self, key, runtime_dir, ctx, *, extra_dirs=None, on_event=None,
                      cancel_check=None, timeout=None):
        self.loads.append((key, ctx))
        self.loaded_key = key
        self.alive = True
        if on_event:
            on_event(("progress", {"phase": "loading", "step": 1, "steps": 2, "secs": 0.0}))
        return {"version": self.version, "image": self.image, "video": False}

    def generate_image(self, params, *, on_event=None, cancel_check=None, timeout=None):
        self.generates.append(params)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        if on_event:
            on_event(("progress", {"phase": "sampling", "step": 1, "steps": 2, "secs": 0.5}))
            on_event(("log", 3, "a warning from the runtime"))
        w, h = params["width"], params["height"]
        return {"width": w, "height": h, "channel": 3, "data": bytes([200, 30, 30]) * (w * h),
                "seed": params["seed"]}

    def shutdown(self, grace=5.0):
        self.alive = False


@pytest.fixture
def fake_native(tmp_path, monkeypatch):
    runner = _FakeRunner()
    monkeypatch.setattr(shared, "runner", runner)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    rt = sd_runtime.Runtime(backend="vulkan", path=tmp_path / "rt")
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": rt)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    return runner, {"backend": "native", "native": {"model": str(model)}}


def _png_chunks(path: Path) -> list[bytes]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    out, pos = [], 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        out.append(data[pos + 4:pos + 8])
        pos += 12 + length
    return out


def test_generate_writes_a_png_of_the_requested_size(fake_native, tmp_path):
    runner, s = fake_native
    out = tmp_path / "out" / "a.png"
    lines = []
    ok, msg = native.generate(s, "a red apple", out, write_sidecar=True, width=768,
                              height=512, seed=11, on_progress=lines.append)
    assert ok, msg
    from PIL import Image
    with Image.open(out) as im:
        assert im.size == (768, 512)
        assert im.getpixel((0, 0)) == (200, 30, 30)
    assert set(_png_chunks(out)) <= {b"IHDR", b"IDAT", b"IEND"}
    side = json.loads(out.with_suffix(".png.json").read_text(encoding="utf-8"))
    assert side["prompt"] == "a red apple" and side["seed"] == 11
    assert side["backend"] == "native" and side["width"] == 768
    assert runner.generates[0]["width"] == 768 and runner.generates[0]["height"] == 512
    assert any(line.startswith("Step 1/2") for line in lines)
    assert any("a warning from the runtime" in line for line in lines)
    assert "768x512" in msg and "seed 11" in msg
    assert not list(out.parent.glob("*.tmp"))


def test_no_sidecar_when_not_requested(fake_native, tmp_path):
    _runner, s = fake_native
    out = tmp_path / "b.png"
    ok, _ = native.generate(s, "p", out, write_sidecar=False)
    assert ok
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["b.png", "model.gguf"]


@pytest.mark.parametrize("version,expected", [("SD 1.x", (512, 512)), ("SD 2.x", (512, 512)),
                                              ("SDXL", (1024, 1024)), ("Flux", (1024, 1024))])
def test_default_size_follows_the_model_family(fake_native, tmp_path, version, expected):
    runner, s = fake_native
    runner.version = version
    ok, _ = native.generate(s, "p", tmp_path / "c.png", write_sidecar=False)
    assert ok
    assert (runner.generates[0]["width"], runner.generates[0]["height"]) == expected


def test_a_recommended_model_brings_its_steps_cfg_and_size(fake_native, tmp_path):
    runner, s = fake_native
    rec = native.RECOMMENDED_MODELS[0]
    f = tmp_path / rec.file
    f.write_bytes(b"x")
    s = {"backend": "native", "native": {"model": str(f)}}
    ok, _ = native.generate(s, "p", tmp_path / "d.png", write_sidecar=False)
    assert ok
    g = runner.generates[0]
    assert (g["steps"], g["cfg_scale"], g["width"], g["height"]) == (
        rec.steps, rec.cfg_scale, rec.width, rec.height)
    ok, _ = native.generate({"backend": "native", "native": {"model": str(f), "steps": 9,
                                                              "cfg_scale": 2.5}},
                            "p", tmp_path / "e.png", write_sidecar=False, cfg=4.0)
    assert (runner.generates[1]["steps"], runner.generates[1]["cfg_scale"]) == (9, 4.0)


@pytest.mark.parametrize("kwargs,needle", [
    ({"model_overrides": {"1": {"unet_name": "x"}}}, "ComfyUI"),
    ({"lora_name": "style.safetensors"}, "LoRA"),
    ({"placement": {"unet": 1}}, "placement"),
    ({"width": 512}, "both width and height"),
    ({"width": 513, "height": 512}, "multiple of 8"),
    ({"width": 4096, "height": 512}, "outside"),
])
def test_requests_the_native_backend_cannot_honour_are_refused(fake_native, tmp_path,
                                                                kwargs, needle):
    runner, s = fake_native
    out = tmp_path / "f.png"
    ok, msg = native.generate(s, "p", out, write_sidecar=False, **kwargs)
    assert not ok and needle in msg
    assert runner.loads == [] and not out.exists()


def test_a_cancel_is_reported_and_writes_nothing(fake_native, tmp_path):
    runner, s = fake_native
    runner.raise_on_generate = SdCancelled("Generation cancelled.")
    out = tmp_path / "g.png"
    assert native.generate(s, "p", out, write_sidecar=True) == (False, "Cancelled.")
    assert not out.exists() and not out.with_suffix(".png.json").exists()


def test_a_worker_failure_is_reported(fake_native, tmp_path):
    runner, s = fake_native
    runner.raise_on_generate = SdWorkerError("The image worker process crashed")
    ok, msg = native.generate(s, "p", tmp_path / "h.png", write_sidecar=False)
    assert not ok and "crashed" in msg


def test_a_model_that_cannot_make_images_is_refused(fake_native, tmp_path):
    runner, s = fake_native
    runner.image = False
    runner.version = "Wan 2.x"
    ok, msg = native.generate(s, "p", tmp_path / "i.png", write_sidecar=False)
    assert not ok and "not an image generation model" in msg
    assert runner.generates == []


def test_img2img_resizes_the_input_to_the_requested_size(fake_native, tmp_path):
    from PIL import Image
    runner, s = fake_native
    src = tmp_path / "src.png"
    Image.new("RGB", (300, 200), (0, 0, 255)).save(src)
    ok, _ = native.generate(s, "p", tmp_path / "j.png", write_sidecar=False,
                            input_image=src, width=512, height=512, denoise=0.4)
    assert ok
    init = runner.generates[0]["init_image"]
    assert (init["width"], init["height"], init["channel"]) == (512, 512, 3)
    assert len(init["data"]) == 512 * 512 * 3
    assert runner.generates[0]["strength"] == 0.4


def test_img2img_without_a_size_uses_the_input_size_rounded_down_to_8(fake_native, tmp_path):
    from PIL import Image
    runner, s = fake_native
    src = tmp_path / "src.png"
    Image.new("RGB", (301, 205)).save(src)
    ok, _ = native.generate(s, "p", tmp_path / "k.png", write_sidecar=False, input_image=src)
    assert ok
    g = runner.generates[0]
    assert (g["init_image"]["width"], g["init_image"]["height"]) == (296, 200)
    assert (g["width"], g["height"]) == (296, 200)


def test_ensure_available_reports_a_runtime_that_cannot_install(monkeypatch, tmp_path):
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": None)

    def fail(choice="auto", **k):
        raise sd_runtime.ProvisionError("the download was refused by the network policy")

    monkeypatch.setattr(sd_runtime, "provision", fail)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    ok, msg = native.ensure_available({"native": {"model": str(model)}})
    assert not ok and "network policy" in msg


def test_a_missing_model_is_reported_before_any_runtime_download(monkeypatch):
    calls = []
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": None)
    monkeypatch.setattr(sd_runtime, "provision", lambda *a, **k: calls.append(a))
    ok, msg = native.ensure_available({"native": {}})
    assert not ok and "No native image model" in msg
    assert calls == []


def test_ensure_available_reports_a_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(sd_runtime, "resolve",
                        lambda choice="auto": sd_runtime.Runtime("cpu", tmp_path))
    ok, msg = native.ensure_available({"native": {}})
    assert not ok and "No native image model" in msg


def test_free_vram_stops_the_worker(fake_native):
    runner, _s = fake_native
    runner.alive = True
    assert native.free_vram({}) is True
    assert not runner.alive


def test_vram_estimate_counts_the_model_files(fake_native, tmp_path):
    _runner, s = fake_native
    est = native.vram_estimate_bytes(s)
    assert est == int(len(b"weights") * 1.2) + native.COMPUTE_ALLOWANCE_BYTES
    assert native.vram_estimate_bytes({"native": {"model": str(tmp_path / "x")}}) is None


# --------------------------------------------------------------------------- #
#  settings fields                                                            #
# --------------------------------------------------------------------------- #

def test_backend_and_native_fields_validate_and_nest():
    from localm.settings_schema import validate_media_block
    merged = validate_media_block("image", {"backend": "native", "native_steps": "4",
                                            "native_cfg_scale": "1.5",
                                            "native_runtime": "vulkan",
                                            "native_model": " sd-turbo "})
    assert merged == {"backend": "native",
                      "native": {"steps": 4, "cfg_scale": 1.5, "runtime": "vulkan",
                                 "model": "sd-turbo"}}
    assert validate_media_block("image", {"native_steps": ""}) == {"native": {"steps": None}}


@pytest.mark.parametrize("updates,needle", [
    ({"backend": "a1111"}, "not one of"),
    ({"native_steps": "0"}, "outside"),
    ({"native_steps": "2.5"}, "whole number"),
    ({"native_steps": "many"}, "not a number"),
    ({"native_cfg_scale": "nan"}, "outside"),
    ({"native_cfg_scale": "inf"}, "outside"),
    ({"native_steps": "1e400"}, "outside"),
    ({"native_steps": "-inf"}, "outside"),
    ({"native_steps": float("inf")}, "outside"),
    ({"native_runtime": "opencl"}, "not one of"),
])
def test_bad_native_values_are_refused(updates, needle):
    from localm.settings_schema import validate_media_block
    with pytest.raises(ValueError, match=needle):
        validate_media_block("image", updates)


def test_native_fields_belong_to_the_image_plugin_only():
    from localm.settings_schema import validate_media_block
    with pytest.raises(ValueError, match="unknown media field"):
        validate_media_block("video", {"backend": "native"})
    with pytest.raises(ValueError, match="unknown media field"):
        validate_media_block("music", {"native_model": "sd-turbo"})


def test_the_schema_shows_auto_by_default_and_hides_paths_from_non_owners():
    from localm.settings_schema import media_admin_only_fields, media_schema_json
    fields = {f["key"]: f for f in media_schema_json("image", {}, {})}
    assert fields["backend"]["value"] == "auto" and fields["backend"]["is_override"] is False
    assert fields["native_runtime"]["value"] == "auto"
    assert {"native_model", "native_vae", "native_clip_l", "native_t5xxl",
            "native_llm"} <= media_admin_only_fields()
    public = {f["key"] for f in media_schema_json("image", {}, {}, is_owner=False)}
    assert "native_model" not in public and "backend" in public


# --------------------------------------------------------------------------- #
#  GGUF detection of stable-diffusion.cpp checkpoints                         #
# --------------------------------------------------------------------------- #

def _gguf_string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(path: Path, tensor_names, arch=None) -> Path:
    kvs = b""
    n_kv = 0
    if arch:
        kvs += _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string(arch)
        n_kv = 1
    infos = b""
    for i, name in enumerate(tensor_names):
        infos += _gguf_string(name) + struct.pack("<I", 1) + struct.pack("<Q", 4)
        infos += struct.pack("<I", 0) + struct.pack("<Q", i * 32)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensor_names), n_kv)
    body = head + kvs + infos
    body += b"\0" * ((32 - len(body) % 32) % 32)
    body += b"\0" * (32 * max(1, len(tensor_names)))
    path.write_bytes(body)
    return path


def test_sd_checkpoint_detection(tmp_path):
    from localm.model_manager import gguf_sd_checkpoint
    sd = _write_gguf(tmp_path / "sd.gguf", ["cond_stage_model.x.weight",
                                            "model.diffusion_model.input_blocks.0.weight"])
    vae = _write_gguf(tmp_path / "vae.gguf", ["first_stage_model.decoder.conv_in.weight"])
    llm = _write_gguf(tmp_path / "llm.gguf", ["blk.0.attn_q.weight"])
    declared = _write_gguf(tmp_path / "decl.gguf", ["model.diffusion_model.x"], arch="llama")
    not_gguf = tmp_path / "x.gguf"
    not_gguf.write_bytes(b"nope" * 100)
    assert gguf_sd_checkpoint(sd) is True
    assert gguf_sd_checkpoint(vae) is True
    assert gguf_sd_checkpoint(llm) is False
    assert gguf_sd_checkpoint(declared) is False
    assert gguf_sd_checkpoint(not_gguf) is False


def test_a_local_sd_checkpoint_registers_as_an_image_model(tmp_path):
    from localm.model_manager.registry import _detect_local_model_type
    sd = _write_gguf(tmp_path / "sd.gguf", ["model.diffusion_model.out.2.weight"])
    llm = _write_gguf(tmp_path / "llm.gguf", ["blk.0.attn_q.weight"])
    assert _detect_local_model_type(sd, is_gguf=True, is_hf=False)[0] == "diffusion-unet"
    assert _detect_local_model_type(llm, is_gguf=True, is_hf=False)[0] == "llm"


# --------------------------------------------------------------------------- #
#  the real image plugin routes and CLI, on the native backend                #
# --------------------------------------------------------------------------- #

@pytest.fixture
def native_app(fake_native, tmp_path, monkeypatch, no_comfy):
    import localm.media.comfy_client as cc
    monkeypatch.setattr(cc, "_comfy_alive", lambda url, timeout=3.0: False)
    import localm.image_gen.comfy as ic
    monkeypatch.setattr(ic, "_comfy_alive", lambda url, timeout=3.0: False)
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    from fastapi import FastAPI

    from localm.config import update_config
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.jobs import JobManager
    runner, s = fake_native
    model = s["native"]["model"]
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "image", {}).update({"native": {"model": model}}))
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("image")
    app.state.jobs = JobManager()
    app.state.self_url = "http://127.0.0.1:8642/v1"
    return app, runner


def _wait_job(app, job_id, timeout=30.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = app.state.jobs.get(job_id)
        if job is not None and job.status != "running" and getattr(job, "finished_at", None):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_imagine_route_generates_natively_at_the_requested_size(native_app):
    from fastapi.testclient import TestClient
    app, runner = native_app
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "a lighthouse", "size": "768x512"})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "done"
    from localm.media import paths as media_paths
    from PIL import Image
    out = media_paths.gallery_dir(media_paths.IMAGE_DIR_NAME) / job.result
    with Image.open(out) as im:
        assert im.size == (768, 512)
    lines = [e.get("text", "") for e in job._history if e.get("type") == "line"]
    assert any("native stable-diffusion.cpp" in ln for ln in lines)
    assert runner.generates[0]["prompt"] == "a lighthouse"


def test_imagine_route_rejects_a_bad_size(native_app):
    from fastapi.testclient import TestClient
    app, runner = native_app
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "size": "100x100"})
    assert r.status_code == 400 and "multiples of 8" in r.json()["detail"]
    assert runner.generates == []


def test_openai_images_route_returns_the_native_image(native_app):
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image
    app, _runner = native_app
    with TestClient(app) as c:
        r = c.post("/v1/images/generations",
                   json={"prompt": "p", "size": "512x256", "response_format": "b64_json"})
    assert r.status_code == 200, r.text
    png = base64.b64decode(r.json()["data"][0]["b64_json"])
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (512, 256)


def test_backend_route_reports_native_and_its_model(native_app):
    from fastapi.testclient import TestClient
    app, _runner = native_app
    with TestClient(app) as c:
        data = c.get("/api/imagine/backend").json()
    assert data["choice"] == "auto" and data["active"] == "native"
    assert data["native"]["runtime"] == "vulkan"
    assert data["native"]["missing"] is None and data["native"]["model"]
    assert data["native"]["recommended"]["spec"] == native.RECOMMENDED_MODELS[0].spec


def test_explicit_comfy_without_comfyui_fails_and_never_runs_native(native_app, monkeypatch):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    app, runner = native_app
    update_config(lambda cfg: cfg["plugins"]["image"].update({"backend": "comfy"}))
    mod = sys.modules["_localm_plugin_image.backend"]
    monkeypatch.setattr(mod._COMFY_REF, "ensure_available",
                        lambda s, on_progress=None: (False, "ComfyUI is not running"))
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "p"})
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "failed"
    assert runner.loads == [] and runner.generates == []


def test_cli_image_uses_the_native_backend(fake_native, cli_runner, tmp_path, monkeypatch,
                                           no_comfy):
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: False)
    runner, s = fake_native
    from localm.config import update_config
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "image", {}).update({"native": {"model": s["native"]["model"]}}))
    from localm.cli import main
    out = tmp_path / "cli.png"
    result = cli_runner.invoke(main, ["image", "a fox", "--size", "640x384", "-o", str(out)])
    assert result.exit_code == 0, result.output
    from PIL import Image
    with Image.open(out) as im:
        assert im.size == (640, 384)
    assert "native stable-diffusion.cpp" in result.output
    assert not runner.alive


def test_cli_image_rejects_a_bad_size(cli_runner):
    from localm.cli import main
    result = cli_runner.invoke(main, ["image", "a fox", "--size", "7x7"])
    assert result.exit_code == 2
    assert "between" in result.output


# --------------------------------------------------------------------------- #
#  GUI preflight on the native backend                                        #
# --------------------------------------------------------------------------- #

@pytest.fixture
def gui_app(tmp_path, monkeypatch, no_comfy):
    from fastapi import FastAPI

    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None, active_model=lambda: "model-a")
    return app


def test_preflight_offers_the_recommended_native_model_when_none_is_set_up(gui_app):
    from fastapi.testclient import TestClient
    with TestClient(gui_app) as c:
        r = c.post("/api/media/image/preflight", json={})
    assert r.status_code == 200, r.text
    data = r.json()
    rec = native.RECOMMENDED_MODELS[0]
    assert data["status"] == "verified"
    [entry] = data["missing"]
    assert entry["native"] is True and entry["filename"] == rec.file
    assert entry["source"]["spec"] == rec.spec
    assert entry["source"]["sha256"] == rec.sha256
    assert entry["source"]["model_type"] == "diffusion-unet"


def test_preflight_reports_nothing_missing_once_the_native_model_resolves(gui_app, tmp_path):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    model = tmp_path / "ck.gguf"
    model.write_bytes(b"x")
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "image", {}).update({"native": {"model": str(model)}}))
    with TestClient(gui_app) as c:
        data = c.post("/api/media/image/preflight", json={}).json()
    assert data == {"status": "verified", "missing": [], "warning": None}


def test_preflight_never_offers_a_download_over_an_explicit_model(gui_app, tmp_path):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "image", {}).update({"native": {"model": str(tmp_path / "gone.gguf")}}))
    with TestClient(gui_app) as c:
        data = c.post("/api/media/image/preflight", json={}).json()
    assert data["missing"] == []
    assert "neither a registered model nor a file" in data["warning"]


# --------------------------------------------------------------------------- #
#  surfaces that have their own ComfyUI path                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def native_configured(fake_native, no_comfy, monkeypatch):
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: False)
    runner, s = fake_native
    from localm.config import update_config
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "image", {}).update({"native": {"model": s["native"]["model"]}}))
    return runner


def test_comfy_launch_route_always_starts_comfyui(native_app, monkeypatch):
    from fastapi.testclient import TestClient

    import localm.image_gen.comfy as ic
    app, runner = native_app
    seen = []
    monkeypatch.setattr(ic, "ensure_comfy", lambda *a, **k: (seen.append(a), (True, "up"))[1])
    with TestClient(app) as c:
        r = c.post("/api/imagine/comfy-launch")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert seen and runner.loads == []


def test_coder_tool_generates_with_the_native_backend(native_configured, tmp_path):
    from localm.plugins.coder.tools.media import tool_generate_image
    from PIL import Image
    result = tool_generate_image(tmp_path, "a fox", output_path="art/fox.png")
    assert result.ok, result.output
    with Image.open(tmp_path / "art" / "fox.png") as im:
        assert im.size == (512, 512)
    assert native_configured.generates[0]["prompt"] == "a fox"
    assert not native_configured.alive


def test_coder_tool_reports_a_native_refusal(native_configured, tmp_path):
    from localm.plugins.coder.tools.media import tool_generate_image
    result = tool_generate_image(tmp_path, "a fox", output_path="fox.png",
                                 lora_name="style.safetensors")
    assert not result.ok and "LoRA" in result.output
    assert not (tmp_path / "fox.png").exists()


def test_chat_repl_generates_with_the_native_backend(native_configured, tmp_path):
    from rich.console import Console

    from localm.cli.chat import _cmd_generate_media
    from localm.media import paths as media_paths

    class _Engine:
        unloaded = 0

        def unload(self):
            _Engine.unloaded += 1

    out = io.StringIO()
    _cmd_generate_media("generate-image", "a fox", _Engine(), Console(file=out, width=200),
                        tmp_path)
    files = list((tmp_path / media_paths.IMAGE_DIR_NAME).glob("*.png"))
    assert len(files) == 1, out.getvalue()
    assert _Engine.unloaded == 1
    assert "native stable-diffusion.cpp" in out.getvalue()
    assert not native_configured.alive


def test_mcp_generate_image_tool_uses_the_native_backend(native_configured, monkeypatch):
    import localm.image_gen.comfy as ic
    import localm.plugins.mcpserver.server as srv
    from localm.config import home_dir
    from PIL import Image
    monkeypatch.setenv("LOCALM_MODE", "full")
    comfy_calls = []
    monkeypatch.setattr(ic, "generate_image", lambda *a, **k: comfy_calls.append(a))
    engines = srv.EngineCache(default_model="stub", engine_factory=lambda name: None)
    tools = srv.build_tools(engines, enable_images=True, enable_coder=False,
                            enable_memory=False)
    server = srv.MCPStdioServer({"generate_image": tools["generate_image"]})
    call = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                       "params": {"name": "generate_image", "arguments": {"prompt": "a fox"},
                                  "_meta": {"progressToken": "img"}}}) + "\n"
    out = io.StringIO()
    server.run_stdio(stdin=io.StringIO(call), stdout=out)
    msgs = [json.loads(line) for line in out.getvalue().splitlines()]
    assert msgs[-1]["id"] == 9 and msgs[-1]["result"].get("isError") is not True, msgs[-1]
    [png] = list((home_dir() / "mcp-images").glob("*.png"))
    with Image.open(png) as im:
        assert im.size == (512, 512)
    assert comfy_calls == []
    assert native_configured.generates[0]["prompt"] == "a fox"
    notes = [m["params"]["message"] for m in msgs if m.get("method") == "notifications/progress"]
    assert any(n.startswith("Step 1/2") for n in notes), notes
    assert not native_configured.alive


# --------------------------------------------------------------------------- #
#  what users see: file names only, refused paths, clear errors               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value,shown", [
    ("sd-turbo", "sd-turbo"),
    ("C:\\models\\private\\ck.gguf", "ck.gguf"),
    ("/srv/models/private/ck.gguf", "ck.gguf"),
    ("\\\\192.0.2.1\\share\\ck.gguf", "ck.gguf"),
])
def test_display_name_never_shows_a_directory(value, shown):
    assert native.display_name(value) == shown


def test_labels_and_errors_name_only_the_file(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    model = private / "ck.gguf"
    model.write_bytes(b"x")
    s = {"native": {"model": str(model)}}
    assert native.resolve_models(s)["label"] == "ck.gguf"
    assert native.status(s)["model"] == "ck.gguf"
    hf_dir = private / "hf-dir"
    hf_dir.mkdir()
    for bad in (private / "gone.gguf", hf_dir):
        with pytest.raises(native._ModelError) as ei:
            native.resolve_models({"native": {"model": str(bad)}})
        assert bad.name in str(ei.value)
        assert "private" not in str(ei.value)


def test_a_network_path_is_refused_without_touching_the_filesystem(monkeypatch):
    seen = []

    def spy(real):
        def wrapped(self, *a, **k):
            if str(self).replace("/", "\\").startswith("\\\\"):
                seen.append(str(self))
                return False
            return real(self, *a, **k)
        return wrapped

    monkeypatch.setattr(Path, "exists", spy(Path.exists))
    monkeypatch.setattr(Path, "is_file", spy(Path.is_file))
    with pytest.raises(native._ModelError, match="network or device path") as ei:
        native.resolve_models({"native": {"model": "\\\\192.0.2.1\\share\\ck.gguf"}})
    assert "192.0.2.1" not in str(ei.value)
    assert seen == []


def test_an_unreadable_model_path_is_a_clear_error(tmp_path, monkeypatch):
    model = tmp_path / "ck.gguf"
    real = Path.exists

    def exists(self, *a, **k):
        if self == model:
            raise PermissionError(13, "Permission denied")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "exists", exists)
    with pytest.raises(native._ModelError, match=r"cannot be read \(PermissionError\)"):
        native.resolve_models({"native": {"model": str(model)}})
    ok, msg = native.ensure_available({"native": {"model": str(model)}})
    assert not ok and "PermissionError" in msg
    assert native.status({"native": {"model": str(model)}})["missing"]
    assert native.vram_estimate_bytes({"native": {"model": str(model)}}) is None


def test_ensure_available_reports_an_unexpected_model_check_failure(monkeypatch):
    def broken(s):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(native, "resolve_models", broken)
    ok, msg = native.ensure_available({"native": {}})
    assert not ok and "registry unreadable" in msg


@pytest.mark.parametrize("size,expected", [
    ((301, 205), (296, 200)),
    ((4000, 3000), (2048, 1536)),
    ((3000, 1000), (2048, 680)),
    ((30, 5000), (64, 2048)),
    ((10, 10), (64, 64)),
])
def test_fit_size_keeps_the_aspect_ratio_within_the_limits(size, expected):
    assert native.fit_size(*size) == expected


def test_img2img_without_a_size_scales_a_large_input_keeping_its_aspect(fake_native,
                                                                         tmp_path):
    from PIL import Image
    runner, s = fake_native
    src = tmp_path / "wide.png"
    Image.new("RGB", (3000, 1000)).save(src)
    ok, msg = native.generate(s, "p", tmp_path / "w.png", write_sidecar=False,
                              input_image=src)
    assert ok, msg
    g = runner.generates[0]
    assert (g["width"], g["height"]) == (2048, 680)


def test_generate_never_raises_and_releases_the_worker(fake_native, tmp_path, monkeypatch):
    from PIL import Image
    runner, s = fake_native
    runner.raise_on_generate = RuntimeError("boom")
    out = tmp_path / "x.png"
    ok, msg = native.generate(s, "p", out, write_sidecar=True)
    assert not ok and "RuntimeError: boom" in msg
    assert not out.exists() and not out.with_suffix(".png.json").exists()
    runner.raise_on_generate = None
    src = tmp_path / "huge.png"
    Image.new("RGB", (64, 64)).save(src)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    ok, msg = native.generate(s, "p", out, write_sidecar=False, input_image=src)
    assert not ok and "DecompressionBombError" in msg
    assert not out.exists()
    assert shared.lock.acquire(blocking=False)
    shared.lock.release()


def test_a_native_refusal_fails_the_job_before_any_vram_handover(native_app, monkeypatch):
    from fastapi.testclient import TestClient
    app, runner = native_app
    unloads, provisions = [], []
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: True)
    monkeypatch.setattr("localm.vram.unload_chat_for_media",
                        lambda *a, **k: unloads.append(a) or True)
    monkeypatch.setattr(sd_runtime, "provision", lambda *a, **k: provisions.append(a))
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "lora_name": "style.safetensors"})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "failed"
    lines = [e.get("text", "") for e in job._history if e.get("type") == "line"]
    assert any("LoRA" in ln for ln in lines)
    assert unloads == [] and provisions == [] and runner.loads == []


def test_backend_route_shows_only_the_model_file_name(native_app):
    from fastapi.testclient import TestClient
    app, _runner = native_app
    with TestClient(app) as c:
        data = c.get("/api/imagine/backend").json()
    assert data["native"]["model"] == "model.gguf"


def test_backend_route_reports_comfy_when_one_answers_under_auto(native_app, monkeypatch):
    from fastapi.testclient import TestClient
    app, _runner = native_app
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: True)
    with TestClient(app) as c:
        data = c.get("/api/imagine/backend").json()
    assert data["choice"] == "auto" and data["active"] == "comfy"
    assert "native" not in data


def test_preflight_checks_comfyui_when_one_answers_under_auto(gui_app, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: True)
    asked = []
    monkeypatch.setattr(comfy_client, "describe_missing_models",
                        lambda workflow, api_url: asked.append(api_url) or [])
    with TestClient(gui_app) as c:
        data = c.post("/api/media/image/preflight", json={}).json()
    assert asked, data
    assert not any(m.get("native") for m in data.get("missing", []))


# --------------------------------------------------------------------------- #
#  the process-wide worker: idle countdown and serialisation                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def live_worker(monkeypatch):
    runner = _FakeRunner()
    runner.alive = True
    monkeypatch.setattr(shared, "runner", runner)
    yield runner
    shared.cancel_idle_timer()


def _wait_fired(timer):
    timer.join(5.0)
    assert not timer.is_alive()


def test_the_idle_countdown_stops_an_unused_worker(live_worker):
    assert shared.worker_pid() == live_worker.pid
    shared.arm_idle_timer(0.01)
    _wait_fired(shared._idle_timer)
    assert not live_worker.alive
    assert shared.worker_pid() is None


def test_the_idle_countdown_leaves_a_worker_that_is_in_use(live_worker):
    with shared.lock:
        shared.arm_idle_timer(0.01)
        _wait_fired(shared._idle_timer)
    assert live_worker.alive


def test_rearming_cancels_the_pending_countdown(live_worker):
    shared.arm_idle_timer(60)
    first = shared._idle_timer
    shared.arm_idle_timer(60)
    assert shared._idle_timer is not first and first.finished.is_set()


def test_free_cancels_the_countdown_and_stops_the_worker(live_worker):
    shared.arm_idle_timer(60)
    pending = shared._idle_timer
    assert shared.free() is True
    assert pending.finished.is_set() and shared._idle_timer is None
    assert not live_worker.alive


@pytest.fixture
def timed_native(tmp_path, monkeypatch):
    runner = _FakeRunner()
    monkeypatch.setattr(shared, "runner", runner)
    rt = sd_runtime.Runtime(backend="vulkan", path=tmp_path / "rt")
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": rt)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    yield runner, {"backend": "native", "native": {"model": str(model)}}
    shared.cancel_idle_timer()


def test_a_generation_restarts_the_idle_countdown(timed_native, tmp_path):
    runner, s = timed_native
    shared.arm_idle_timer(60)
    before = shared._idle_timer
    ok, msg = native.generate(s, "p", tmp_path / "o.png", write_sidecar=False)
    assert ok, msg
    after = shared._idle_timer
    assert before.finished.is_set()
    assert after is not None and after is not before
    assert after.interval == shared.IDLE_UNLOAD_SECONDS
    assert runner.alive


def test_generations_wait_for_the_worker_in_turn(timed_native, tmp_path):
    import threading
    runner, s = timed_native
    results = []
    with shared.lock:
        t = threading.Thread(target=lambda: results.append(
            native.generate(s, "p", tmp_path / "q.png", write_sidecar=False)))
        t.start()
        t.join(0.3)
        assert t.is_alive() and runner.loads == []
    t.join(10.0)
    assert not t.is_alive() and results[0][0] is True
