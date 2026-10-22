# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native image generation controls: batch count, sampler and scheduler,
hires fix, and upscaling a gallery image with a registered upscaler model.
The sd.cpp worker is an in-process fake (the real one is exercised by
tests/test_sdcpp_runner.py and the live checks)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image

from localm.media import native_models, paths as media_paths
from localm.media.sdcpp import _binding, sampling
from localm.media.sdcpp.runner import SdCancelled
from localm.plugins.builtin.image.backends import native
from tests.test_image_native_backend import (  # noqa: F401  (fixtures)
    _FakeRunner, _register, _wait_job, fake_native, native_app, no_comfy,
)


HIRES_LATENT = {"upscaler": "latent", "model": None, "scale": 1.5, "steps": 0, "denoise": 0.5}


class _QolRunner(_FakeRunner):
    """The fake worker plus the upscale command and a failure injected on the
    N-th generate call."""

    def __init__(self, *, cancel_on_call=None, **kw):
        super().__init__(**kw)
        self.cancel_on_call = cancel_on_call
        self.upscales = []

    def generate_image(self, params, **kw):
        if self.cancel_on_call is not None and len(self.generates) + 1 == self.cancel_on_call:
            self.generates.append(params)
            raise SdCancelled("Generation cancelled.")
        return super().generate_image(params, **kw)

    def upscale_image(self, params, runtime_dir, extra_dirs=None, **kw):
        self.upscales.append(params)
        img = params["image"]
        w, h = img["width"] * 4, img["height"] * 4
        return {"width": w, "height": h, "channel": img["channel"],
                "data": bytes([10, 200, 10]) * (w * h) if img["channel"] == 3
                else bytes([10, 200, 10, 255]) * (w * h), "scale": 4}


@pytest.fixture
def qol(fake_native, monkeypatch):  # noqa: F811
    from localm.media.sdcpp import shared
    runner, s = fake_native
    new = _QolRunner()
    monkeypatch.setattr(shared, "runner", new)
    return new, s


def _sidecar(path: Path) -> dict:
    return json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
#  batch                                                                      #
# --------------------------------------------------------------------------- #

def test_a_batch_saves_every_image_with_its_own_seed_and_sidecar(qol, tmp_path):
    runner, s = qol
    out = tmp_path / "out" / "a.png"
    saved = []
    ok, msg = native.generate(s, "a fox", out, seed=7, batch=3, width=64, height=64,
                              on_saved=saved.append)
    assert ok, msg
    assert [g["seed"] for g in runner.generates] == [7, 8, 9]
    names = [p.name for p in saved]
    assert names == ["a.png", "a_2.png", "a_3.png"]
    assert all(p.is_file() for p in saved)
    sides = [_sidecar(p) for p in saved]
    assert [x["seed"] for x in sides] == [7, 8, 9]
    assert [x["batch"] for x in sides] == [{"index": 1, "count": 3}, {"index": 2, "count": 3},
                                           {"index": 3, "count": 3}]
    assert "3 images" in msg and "seeds 7-9" in msg


def test_a_single_image_has_no_batch_record_and_the_old_message(qol, tmp_path):
    runner, s = qol
    out = tmp_path / "one.png"
    ok, msg = native.generate(s, "p", out, seed=5, width=64, height=64)
    assert ok
    assert "batch" not in _sidecar(out)
    assert msg.startswith("Saved one.png (64x64, seed 5,")


def test_a_cancel_mid_batch_keeps_the_images_already_made(qol, tmp_path):
    runner = qol[0]
    runner.cancel_on_call = 2
    _runner, s = qol
    out = tmp_path / "c.png"
    saved = []
    ok, msg = native.generate(s, "p", out, seed=1, batch=3, width=64, height=64,
                              on_saved=saved.append)
    assert not ok
    assert msg == "Cancelled. 1 of 3 images were saved."
    assert [p.name for p in saved] == ["c.png"] and out.is_file()
    assert not (tmp_path / "c_2.png").exists()


def test_the_batch_count_is_bounded(qol, tmp_path):
    runner, s = qol
    ok, msg = native.generate(s, "p", tmp_path / "x.png", batch=native.MAX_BATCH + 1)
    assert not ok and "between 1 and" in msg
    assert runner.loads == []


# --------------------------------------------------------------------------- #
#  sampler and scheduler                                                      #
# --------------------------------------------------------------------------- #

def test_the_requested_sampler_and_scheduler_reach_the_worker_and_the_sidecar(qol, tmp_path):
    runner, s = qol
    s["native"]["sample_method"] = "heun"
    out = tmp_path / "s.png"
    ok, msg = native.generate(s, "p", out, sample_method="euler_a", scheduler="karras",
                              width=64, height=64)
    assert ok, msg
    g = runner.generates[0]
    assert (g["sample_method"], g["scheduler"]) == ("euler_a", "karras")
    side = _sidecar(out)
    assert (side["sample_method"], side["scheduler"]) == ("euler_a", "karras")


def test_without_a_request_the_settings_and_then_the_model_default_apply(qol, tmp_path):
    runner, s = qol
    s["native"]["sample_method"] = "heun"
    native.generate(s, "p", tmp_path / "a.png", width=64, height=64)
    assert runner.generates[0]["sample_method"] == "heun"
    assert runner.generates[0]["scheduler"] is None
    del s["native"]["sample_method"]
    native.generate(s, "p", tmp_path / "b.png", width=64, height=64)
    assert runner.generates[1]["sample_method"] is None


@pytest.mark.parametrize("kwargs,needle", [
    ({"sample_method": "nope"}, "Unknown sampler 'nope'"),
    ({"scheduler": "nope"}, "Unknown scheduler 'nope'"),
])
def test_an_unknown_sampler_or_scheduler_is_refused_before_loading(qol, tmp_path, kwargs, needle):
    runner, s = qol
    ok, msg = native.generate(s, "p", tmp_path / "x.png", **kwargs)
    assert not ok and needle in msg
    assert runner.loads == []


# --------------------------------------------------------------------------- #
#  hires fix                                                                  #
# --------------------------------------------------------------------------- #

def test_a_latent_hires_pass_reaches_the_worker_and_the_sidecar(qol, tmp_path):
    runner, s = qol
    out = tmp_path / "h.png"
    ok, msg = native.generate(s, "p", out, width=64, height=64, hires=HIRES_LATENT)
    assert ok, msg
    assert runner.generates[0]["hires"] == {
        "upscaler": "Latent", "model_path": None, "scale": 1.5, "steps": 0,
        "denoising_strength": 0.5}
    assert _sidecar(out)["hires"] == HIRES_LATENT
    assert _sidecar(out)["size"] == "64x64"


def test_a_model_hires_pass_hands_the_worker_the_registered_file(qol, tmp_path):
    runner, s = qol
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    hires = dict(HIRES_LATENT, upscaler="model", model="up4x", steps=4)
    ok, msg = native.generate(s, "p", tmp_path / "m.png", width=64, height=64, hires=hires)
    assert ok, msg
    sent = runner.generates[0]["hires"]
    assert sent["upscaler"] == "Model" and sent["model_path"] == str(up) and sent["steps"] == 4


def test_hires_with_a_missing_upscaler_or_too_large_a_result_is_refused(qol, tmp_path):
    runner, s = qol
    ok, msg = native.generate(s, "p", tmp_path / "a.png",
                              hires=dict(HIRES_LATENT, upscaler="model", model="ghost"))
    assert not ok and "ghost" in msg
    ok, msg = native.generate(s, "p", tmp_path / "b.png", width=2048, height=2048,
                              hires=dict(HIRES_LATENT, scale=4.0))
    assert not ok and "limited to 4096" in msg
    assert runner.loads == []


def test_the_hires_limit_applies_to_the_size_taken_from_an_input_image(qol, tmp_path):
    runner, s = qol
    big = tmp_path / "big.png"
    Image.new("RGB", (1536, 1536), (1, 2, 3)).save(big)
    ok, msg = native.generate(s, "p", tmp_path / "e.png", input_image=big,
                              hires=dict(HIRES_LATENT, scale=3.0))
    assert not ok and "limited to 4096" in msg
    assert runner.generates == []
    ok, msg = native.generate(s, "p", tmp_path / "f.png", input_image=big,
                              hires=dict(HIRES_LATENT, scale=2.0))
    assert ok, msg


def test_a_hires_pass_with_every_field_left_out_refines_with_lanczos(qol, tmp_path):
    runner, s = qol
    out = tmp_path / "d.png"
    hires = native_models.validate_hires(native_models.HiresChoice())
    ok, msg = native.generate(s, "p", out, width=64, height=64, hires=hires)
    assert ok, msg
    assert runner.generates[0]["hires"] == {
        "upscaler": "Lanczos", "model_path": None, "scale": 2.0, "steps": 0,
        "denoising_strength": 0.35}
    assert _sidecar(out)["hires"] == {"upscaler": "lanczos", "model": None, "scale": 2.0,
                                      "steps": 0, "denoise": 0.35}


def test_a_latent_upscaler_without_a_denoise_gets_the_latent_denoise(qol, tmp_path):
    runner, s = qol
    hires = native_models.validate_hires(native_models.HiresChoice(upscaler="latent-bicubic"))
    ok, msg = native.generate(s, "p", tmp_path / "l.png", width=64, height=64, hires=hires)
    assert ok, msg
    sent = runner.generates[0]["hires"]
    assert sent["upscaler"] == "Latent (bicubic)"
    assert sent["denoising_strength"] == sampling.SUGGESTED_LATENT_HIRES_DENOISE
    hires = native_models.validate_hires(native_models.HiresChoice(upscaler="latent",
                                                                   denoise=0.8))
    native.generate(s, "p", tmp_path / "m.png", width=64, height=64, hires=hires)
    assert runner.generates[1]["hires"]["denoising_strength"] == 0.8


def test_the_hires_suggestion_follows_the_curated_model_then_the_family(qol, tmp_path,
                                                                        monkeypatch):
    from localm.media.sdcpp import families
    runner, s = qol
    turbo = tmp_path / native.RECOMMENDED_MODELS[0].file
    turbo.write_bytes(b"weights")
    s["native"]["model"] = str(turbo)
    monkeypatch.setattr(families, "defaults_for",
                        lambda info: families.Defaults(25, 6.0, 1024, 1024))
    assert native.resolve_models(s)["hires"]["scale"] == 2.0
    big = tmp_path / "big-model.gguf"
    big.write_bytes(b"weights")
    s["native"]["model"] = str(big)
    assert native.resolve_models(s)["hires"]["scale"] == 1.5
    ok, msg = native.generate(s, "p", tmp_path / "x.png", width=1024, height=1024,
                              hires=native_models.validate_hires(native_models.HiresChoice()))
    assert ok, msg
    assert runner.generates[0]["hires"]["scale"] == 1.5
    monkeypatch.setattr(families, "defaults_for", lambda info: None)
    assert native.resolve_models(s)["hires"] == sampling.suggested_hires(None)


def test_fill_hires_keeps_requested_values_and_fits_a_suggested_scale():
    sug = sampling.suggested_hires(512)
    asked = {"upscaler": "nearest", "model": None, "scale": 3.0, "steps": 5, "denoise": 0.6}
    assert sampling.fill_hires(asked, sug, 512, 512) == asked
    empty = {"upscaler": None, "model": None, "scale": None, "steps": None, "denoise": None}
    filled = sampling.fill_hires(empty, sug, 3000, 1000)
    assert filled["upscaler"] == "lanczos" and filled["denoise"] == 0.35
    assert sampling.check_hires_target(3000, 1000, filled["scale"]) is None
    assert 1.3 <= filled["scale"] < 1.37
    assert sampling.fill_hires(empty, sug, 4096, 4096)["scale"] == sampling.MIN_HIRES_SCALE
    assert sampling.fill_hires(empty, None, 512, 512)["scale"] == 2.0
    assert sampling.fill_hires(dict(empty, steps=0), sug, 512, 512)["steps"] == 0


def test_no_suggestion_ever_uses_the_regenerating_latent_default():
    for side in (None, 512, 640, 768, 1024, 1280):
        sug = sampling.suggested_hires(side)
        assert sug["upscaler"] == "lanczos" and not sampling.is_latent_upscaler(sug["upscaler"])
        assert sug["denoise"] <= 0.4 and 1.5 <= sug["scale"] <= 2.0
        assert sampling.hires_side(side or 512, sug["scale"]) <= sampling.MAX_HIRES_SIDE
    assert sampling.suggested_hires(512)["scale"] == 2.0
    assert sampling.suggested_hires(1024)["scale"] == 1.5


# --------------------------------------------------------------------------- #
#  upscale a gallery image                                                    #
# --------------------------------------------------------------------------- #

def _png(path: Path, size=(32, 16), mode="RGB") -> Path:
    Image.new(mode, size, (200, 30, 30) if mode == "RGB" else (200, 30, 30, 255)).save(path)
    return path


def test_upscale_writes_the_larger_image_and_a_sidecar_from_the_source(qol, tmp_path):
    runner, s = qol
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    src = _png(tmp_path / "src.png")
    out = tmp_path / "out.png"
    saved = []
    ok, msg = native.upscale(s, src, "up4x", out, extra_meta={"prompt": "a fox", "seed": 3},
                             on_saved=saved.append)
    assert ok, msg
    with Image.open(out) as im:
        assert im.size == (128, 64)
    assert runner.upscales[0]["model_path"] == str(up)
    assert (runner.upscales[0]["image"]["width"], runner.upscales[0]["image"]["height"]) == (32, 16)
    side = _sidecar(out)
    assert side["prompt"] == "a fox" and side["seed"] == 3
    assert (side["width"], side["height"]) == (128, 64)
    assert side["upscaled"] == {"from": "src.png", "model": "up4x", "scale": 4}
    assert saved == [out] and "128x64" in msg and "4x" in msg


def test_an_upscale_queued_behind_another_job_cancels_at_once(qol, tmp_path):
    from tests.test_image_native_backend import _cancel_while_queued
    runner, s = qol
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    src = _png(tmp_path / "src.png")
    _cancel_while_queued(
        lambda cancel, say: native.upscale(s, src, "up4x", tmp_path / "o.png",
                                           cancel_check=cancel, on_progress=say), runner)
    assert runner.upscales == [] and not (tmp_path / "o.png").exists()


def test_upscale_keeps_the_alpha_channel(qol, tmp_path):
    runner, s = qol
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    src = _png(tmp_path / "alpha.png", mode="RGBA")
    ok, msg = native.upscale(s, src, "up4x", tmp_path / "o.png")
    assert ok, msg
    assert runner.upscales[0]["image"]["channel"] == 4
    with Image.open(tmp_path / "o.png") as im:
        assert im.mode == "RGBA"


def test_upscale_refuses_an_unregistered_upscaler_and_an_oversized_source(qol, tmp_path):
    runner, s = qol
    src = _png(tmp_path / "src.png")
    ok, msg = native.upscale(s, src, "ghost", tmp_path / "o.png")
    assert not ok and "ghost" in msg and runner.upscales == []
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    big = _png(tmp_path / "big.png", size=(native.MAX_UPSCALE_INPUT_SIDE + 1, 8))
    ok, msg = native.upscale(s, big, "up4x", tmp_path / "o2.png")
    assert not ok and "at most" in msg and runner.upscales == []
    assert not (tmp_path / "o2.png").exists()


def test_upscale_reports_a_worker_cancel_without_writing(qol, tmp_path):
    runner, s = qol
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")

    def boom(*a, **k):
        raise SdCancelled("cancelled")
    runner.upscale_image = boom
    ok, msg = native.upscale(s, _png(tmp_path / "src.png"), "up4x", tmp_path / "o.png")
    assert not ok and msg == "Cancelled." and not (tmp_path / "o.png").exists()


# --------------------------------------------------------------------------- #
#  request validation (native_models)                                         #
# --------------------------------------------------------------------------- #

def test_hires_validation_requires_a_registered_upscaler_for_the_model_mode(tmp_path):
    up = tmp_path / "u.safetensors"
    up.write_bytes(b"x")
    reg = {"up": {"path": str(up), "model_type": "upscaler"},
           "lora": {"path": str(up), "model_type": "lora"}}
    ok = native_models.validate_hires(
        native_models.HiresChoice(upscaler="model", model="up"), reg)
    assert ok == {"upscaler": "model", "model": "up", "scale": None, "steps": None,
                  "denoise": None}
    for model in ("lora", "missing", None):
        with pytest.raises(ValueError):
            native_models.validate_hires(
                native_models.HiresChoice(upscaler="model", model=model), reg)
    with pytest.raises(ValueError, match="Unknown hires upscaler"):
        native_models.validate_hires(native_models.HiresChoice(upscaler="bogus"), reg)
    assert native_models.validate_hires(None) is None


def test_check_generation_extras_turns_a_bad_value_into_a_400():
    for kwargs in ({"sample_method": "nope"}, {"scheduler": "nope"}, {"batch": 0},
                   {"batch": native_models.MAX_BATCH + 1}):
        with pytest.raises(HTTPException) as ei:
            native_models.check_generation_extras(
                kwargs.get("sample_method"), kwargs.get("scheduler"), None,
                kwargs.get("batch", 1))
        assert ei.value.status_code == 400
    out = native_models.check_generation_extras("euler", "", None, 4)
    assert out == {"sample_method": "euler", "scheduler": None, "hires": None, "batch": 4}


def test_hires_model_bounds_are_enforced_by_the_request_model():
    from pydantic import ValidationError
    for bad in ({"scale": 0.5}, {"scale": 9}, {"steps": -1}, {"denoise": 0},
                {"scale": float("nan")}):
        with pytest.raises(ValidationError):
            native_models.HiresChoice(**bad)


def test_upscaler_options_lists_only_registered_existing_upscaler_files(tmp_path):
    f = tmp_path / "RealESRGAN_x4plus.safetensors"
    f.write_bytes(b"x")
    reg = {"real": {"path": str(f), "model_type": "upscaler"},
           "gone": {"path": str(tmp_path / "gone.pth"), "model_type": "upscaler"},
           "other": {"path": str(f), "model_type": "vae"}}
    assert native_models.upscaler_options(reg) == [
        {"name": "real", "file": "RealESRGAN_x4plus.safetensors"}]


# --------------------------------------------------------------------------- #
#  the name tables                                                            #
# --------------------------------------------------------------------------- #

def test_the_name_tables_have_the_sizes_the_binding_counts():
    assert len(sampling.SAMPLE_METHODS) == _binding._SAMPLE_METHOD_COUNT
    assert len(sampling.SCHEDULERS) == _binding._SCHEDULER_COUNT
    assert len(sampling.HIRES_UPSCALER_LIBRARY_NAMES) == _binding._SD_HIRES_UPSCALER_COUNT - 1
    assert len(set(sampling.SAMPLE_METHODS)) == len(sampling.SAMPLE_METHODS)


class _NamesLib:
    def __init__(self, methods, schedulers, hires):
        self.sd_sample_method_name = lambda i: methods[i].encode()
        self.sd_scheduler_name = lambda i: schedulers[i].encode()
        self.sd_hires_upscaler_name = lambda i: hires[i].encode()


def _names_lib(**override):
    parts = {"methods": list(sampling.SAMPLE_METHODS), "schedulers": list(sampling.SCHEDULERS),
             "hires": ["None", *sampling.HIRES_UPSCALER_LIBRARY_NAMES.values()]}
    parts.update(override)
    return _NamesLib(**parts)


def test_check_names_accepts_the_pinned_names_and_flags_a_drift():
    assert _binding.check_names(_names_lib()) == []
    drifted = list(sampling.SAMPLE_METHODS)
    drifted[3] = "renamed"
    assert _binding.check_names(_names_lib(methods=drifted)) == ["sample method names"]
    sched = list(sampling.SCHEDULERS)
    sched.reverse()
    assert _binding.check_names(_names_lib(schedulers=sched)) == ["scheduler names"]
    hires = ["None", *sampling.HIRES_UPSCALER_LIBRARY_NAMES.values()]
    hires[9] = "Other"
    assert _binding.check_names(_names_lib(hires=hires)) == ["hires upscaler names"]


# --------------------------------------------------------------------------- #
#  routes                                                                     #
# --------------------------------------------------------------------------- #

def _job_lines(job) -> list[str]:
    return [e.get("text", "") for e in job._history if e.get("type") == "line"]


def _gallery_file(name: str) -> Path:
    return media_paths.gallery_dir(media_paths.IMAGE_DIR_NAME) / name


@pytest.fixture
def qol_app(native_app, monkeypatch):  # noqa: F811
    from localm.media.sdcpp import shared
    app, _old = native_app
    runner = _QolRunner()
    monkeypatch.setattr(shared, "runner", runner)
    return app, runner


def test_the_imagine_route_runs_a_batch_and_announces_each_saved_image(qol_app):
    from fastapi.testclient import TestClient
    app, runner = qol_app
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "a lighthouse", "size": "64x64",
                                         "batch": 3, "seed": 20})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "done", _job_lines(job)
    announced = [e["name"] for e in job._history if e.get("type") == "image"]
    assert len(announced) == 3 and len(set(announced)) == 3
    assert job.result == announced[0]
    assert all(_gallery_file(n).is_file() for n in announced)
    assert [g["seed"] for g in runner.generates] == [20, 21, 22]


def test_the_imagine_route_forwards_sampler_scheduler_and_hires(qol_app):
    from fastapi.testclient import TestClient
    app, runner = qol_app
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={
            "prompt": "p", "size": "64x64", "sample_method": "dpm++2m", "scheduler": "karras",
            "hires": {"upscaler": "latent-bicubic", "scale": 1.5, "steps": 3, "denoise": 0.4}})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "done", _job_lines(job)
    g = runner.generates[0]
    assert (g["sample_method"], g["scheduler"]) == ("dpm++2m", "karras")
    assert g["hires"]["upscaler"] == "Latent (bicubic)" and g["hires"]["steps"] == 3


@pytest.mark.parametrize("hires", [{}, {"upscaler": ""}, {"upscaler": None, "scale": None}])
def test_the_imagine_route_fills_a_hires_pass_left_empty_from_the_model(qol_app, hires):
    from fastapi.testclient import TestClient
    app, runner = qol_app
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "size": "64x64", "hires": hires})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "done", _job_lines(job)
    assert runner.generates[0]["hires"] == {
        "upscaler": "Lanczos", "model_path": None, "scale": 2.0, "steps": 0,
        "denoising_strength": 0.35}


@pytest.mark.parametrize("body,status", [
    ({"sample_method": "nope"}, 400),
    ({"scheduler": "nope"}, 400),
    ({"hires": {"upscaler": "model"}}, 400),
    ({"hires": {"upscaler": "model", "model": "ghost"}}, 400),
    ({"hires": {"upscaler": "bogus"}}, 400),
    ({"batch": native_models.MAX_BATCH + 1}, 422),
    ({"hires": {"scale": 9}}, 422),
])
def test_the_imagine_route_rejects_bad_controls_before_any_job(qol_app, body, status):
    from fastapi.testclient import TestClient
    app, runner = qol_app
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "size": "64x64", **body})
    assert r.status_code == status, r.text
    assert runner.generates == [] and not list(app.state.jobs.snapshot())


def test_a_comfyui_job_is_told_these_controls_are_native_only(qol_app):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    app, runner = qol_app
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "image", {}).update({"backend": "comfy"}))
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "batch": 2})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "failed"
    assert any("apply to the native backend" in ln for ln in _job_lines(job))
    assert runner.generates == []


def test_native_models_route_serves_the_sampling_choices_and_upscalers(qol_app, tmp_path):
    from fastapi.testclient import TestClient
    app, _runner = qol_app
    up = tmp_path / "RealESRGAN_x4plus.safetensors"
    up.write_bytes(b"x")
    _register("real-esrgan", up, "upscaler")
    with TestClient(app) as c:
        data = c.get("/api/imagine/native-models").json()
    assert data["sampling"]["sample_methods"] == list(sampling.SAMPLE_METHODS)
    assert data["sampling"]["schedulers"] == list(sampling.SCHEDULERS)
    ids = [u["id"] for u in data["sampling"]["hires_upscalers"]]
    assert ids[0] == "latent" and "model" in ids
    assert data["upscalers"] == [{"name": "real-esrgan", "file": "RealESRGAN_x4plus.safetensors"}]


def _generate_one(c, app) -> str:
    r = c.post("/api/imagine", json={"prompt": "a lighthouse", "size": "64x64", "seed": 9})
    assert r.status_code == 200, r.text
    job = _wait_job(app, r.json()["job_id"])
    assert job.status == "done", _job_lines(job)
    return job.result


def test_the_upscale_route_adds_a_new_gallery_image_keeping_the_source_settings(
        qol_app, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("LOCALM_MODE", "log")
    app, runner = qol_app
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    with TestClient(app) as c:
        name = _generate_one(c, app)
        r = c.post(f"/api/imagine/file/{name}/upscale", json={"upscaler": "up4x"})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
        history = c.get("/api/imagine/history").json()["images"]
    assert job.status == "done", _job_lines(job)
    assert job.result != name
    with Image.open(_gallery_file(job.result)) as im:
        assert im.size == (256, 256)
    item = next(i for i in history if i["name"] == job.result)
    assert item["meta"]["prompt"] == "a lighthouse" and item["meta"]["seed"] == 9
    assert item["meta"]["upscaled"]["from"] == name
    assert (item["meta"]["width"], item["meta"]["height"]) == (256, 256)
    assert "created" in item["meta"] and item["meta"].get("batch") is None
    assert len(runner.upscales) == 1


def test_the_upscale_route_ignores_a_deeply_nested_source_sidecar(qol_app, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("LOCALM_MODE", "log")
    app, runner = qol_app
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    with TestClient(app) as c:
        name = _generate_one(c, app)
        Path(str(_gallery_file(name)) + ".json").write_text(
            "[" * 200_000 + "]" * 200_000, encoding="utf-8")
        r = c.post(f"/api/imagine/file/{name}/upscale", json={"upscaler": "up4x"})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "done", _job_lines(job)
    assert len(runner.upscales) == 1


@pytest.mark.parametrize("upscaler", ["ghost", "", "lora-file"])
def test_the_upscale_route_only_takes_a_registered_upscaler(qol_app, tmp_path, upscaler):
    from fastapi.testclient import TestClient
    app, runner = qol_app
    f = tmp_path / "x.safetensors"
    f.write_bytes(b"x")
    _register("lora-file", f, "lora")
    with TestClient(app) as c:
        name = _generate_one(c, app)
        r = c.post(f"/api/imagine/file/{name}/upscale", json={"upscaler": upscaler})
    assert r.status_code == 400, r.text
    assert runner.upscales == []


def test_the_upscale_route_404s_for_a_missing_source(qol_app, tmp_path):
    from fastapi.testclient import TestClient
    app, runner = qol_app
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    with TestClient(app) as c:
        r = c.post("/api/imagine/file/nope.png/upscale", json={"upscaler": "up4x"})
    assert r.status_code == 404
    assert runner.upscales == []


def test_privacy_mode_writes_no_sidecar_for_a_batch_or_an_upscale(qol_app, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    app, runner = qol_app
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "size": "64x64", "batch": 2})
        job = _wait_job(app, r.json()["job_id"])
        assert job.status == "done", _job_lines(job)
        names = [e["name"] for e in job._history if e.get("type") == "image"]
        r = c.post(f"/api/imagine/file/{names[0]}/upscale", json={"upscaler": "up4x"})
        up_job = _wait_job(app, r.json()["job_id"])
    assert up_job.status == "done", _job_lines(up_job)
    for name in [*names, up_job.result]:
        assert _gallery_file(name).is_file()
        assert not Path(str(_gallery_file(name)) + ".json").exists()


def test_an_upscaled_batch_image_does_not_inherit_the_batch_record(qol_app, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("LOCALM_MODE", "log")
    app, _runner = qol_app
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "size": "64x64", "batch": 2})
        job = _wait_job(app, r.json()["job_id"])
        second = [e["name"] for e in job._history if e.get("type") == "image"][1]
        assert _sidecar(_gallery_file(second))["batch"] == {"index": 2, "count": 2}
        r = c.post(f"/api/imagine/file/{second}/upscale", json={"upscaler": "up4x"})
        up_job = _wait_job(app, r.json()["job_id"])
    assert up_job.status == "done", _job_lines(up_job)
    side = _sidecar(_gallery_file(up_job.result))
    assert "batch" not in side and side["upscaled"]["from"] == second


def test_batch_images_and_upscales_belong_to_the_key_that_made_them(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    from localm.media.sdcpp import runtime as sd_runtime
    from localm.media.sdcpp import shared
    from tests.test_media_gallery_ownership import _gallery_app, _h, _mk_keys
    from tests.test_media_gallery_ownership import _wait_job as wait_stream

    app = _gallery_app(tmp_path, monkeypatch, "image")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"w")
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault("image", {}).update(
        {"backend": "native", "native": {"model": str(model)}}))
    _register("up4x", up, "upscaler")
    runner = _QolRunner()
    monkeypatch.setattr(shared, "runner", runner)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    rt = sd_runtime.Runtime(backend="vulkan", path=tmp_path / "rt")
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": rt)
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    a, b = _mk_keys(["image"], ["image"])
    with TestClient(app) as c:
        r = c.post("/api/imagine", headers=_h(a),
                   json={"prompt": "p", "size": "64x64", "batch": 2})
        assert r.status_code == 200, r.text
        end = wait_stream(c, r.json()["job_id"], _h(a))
        assert end and end["status"] == "done"
        names = [i["name"] for i in c.get("/api/imagine/history", headers=_h(a)).json()["images"]]
        assert len(names) == 2
        for name in names:
            assert c.get(f"/api/imagine/file/{name}", headers=_h(b)).status_code == 404
        r = c.post(f"/api/imagine/file/{names[0]}/upscale", headers=_h(b),
                   json={"upscaler": "up4x"})
        assert r.status_code == 404 and runner.upscales == []
        r = c.post(f"/api/imagine/file/{names[0]}/upscale", headers=_h(a),
                   json={"upscaler": "up4x"})
        assert r.status_code == 200, r.text
        end = wait_stream(c, r.json()["job_id"], _h(a))
        assert end and end["status"] == "done"
        assert c.get(f"/api/imagine/file/{end['result']}", headers=_h(a)).status_code == 200
        assert c.get(f"/api/imagine/file/{end['result']}", headers=_h(b)).status_code == 404
        assert all(i["name"] not in {*names, end["result"]} for i in
                   c.get("/api/imagine/history", headers=_h(b)).json()["images"])


def test_an_owner_that_cannot_be_written_is_reported_and_still_enforced(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    from localm.media import gallery
    from localm.media.sdcpp import runtime as sd_runtime
    from localm.media.sdcpp import shared
    from tests.test_media_gallery_ownership import _gallery_app, _h, _job_events, _mk_keys

    app = _gallery_app(tmp_path, monkeypatch, "image")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"w")
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault("image", {}).update(
        {"backend": "native", "native": {"model": str(model)}}))
    _register("up4x", up, "upscaler")
    monkeypatch.setattr(shared, "runner", _QolRunner())
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    rt = sd_runtime.Runtime(backend="vulkan", path=tmp_path / "rt")
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": rt)
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)

    def locked_write(media_kind, data):
        raise PermissionError(32, "being used by another process")
    monkeypatch.setattr(gallery, "_write_index", locked_write)
    a, b = _mk_keys(["image"], ["image"])

    def warned(events, name):
        return any(e.get("type") == "line" and "WARNING" in e.get("text", "")
                   and name in e["text"] for e in events)

    with TestClient(app) as c:
        r = c.post("/api/imagine", headers=_h(a), json={"prompt": "p", "size": "64x64"})
        assert r.status_code == 200, r.text
        events = _job_events(c, r.json()["job_id"], _h(a))
        assert events[-1]["status"] == "done"
        name = events[-1]["result"]
        assert warned(events, name), events
        assert c.get(f"/api/imagine/file/{name}", headers=_h(b)).status_code == 404

        r = c.post(f"/api/imagine/file/{name}/upscale", headers=_h(a),
                   json={"upscaler": "up4x"})
        assert r.status_code == 200, r.text
        events = _job_events(c, r.json()["job_id"], _h(a))
        assert events[-1]["status"] == "done"
        bigger = events[-1]["result"]
        assert warned(events, bigger), events
        assert c.get(f"/api/imagine/file/{bigger}", headers=_h(b)).status_code == 404

        r = c.post(f"/api/imagine/file/{name}/rename", headers=_h(a),
                   json={"new_name": "renamed.png"})
        assert r.status_code == 200, r.text
        assert "renamed.png" in r.json()["warning"]
        assert c.get("/api/imagine/file/renamed.png", headers=_h(b)).status_code == 404
        assert c.get("/api/imagine/file/renamed.png", headers=_h(a)).status_code == 200


def test_a_stop_between_batch_images_ends_the_batch_before_the_next_image(qol, tmp_path):
    runner, s = qol
    saved = []
    stop_after_first = lambda: len(saved) >= 1  # noqa: E731
    ok, msg = native.generate(s, "p", tmp_path / "w.png", seed=1, batch=3, width=64, height=64,
                              on_saved=saved.append, cancel_check=stop_after_first)
    assert not ok and msg == "Cancelled. 1 of 3 images were saved."
    assert len(runner.generates) == 1
    assert not (tmp_path / "w_2.png").exists()


def test_upscaler_options_never_touches_a_network_path(tmp_path, monkeypatch):
    seen = []
    real = Path.is_file

    def spy(self, *a, **k):
        seen.append(str(self))
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "is_file", spy)
    unc = r"\\fileserver\models\up.safetensors"
    local = tmp_path / "local.safetensors"
    local.write_bytes(b"x")
    reg = {"net": {"path": unc, "model_type": "upscaler"},
           "dev": {"path": r"\\.\pipe\x", "model_type": "upscaler"},
           "ok": {"path": str(local), "model_type": "upscaler"}}
    assert [o["name"] for o in native_models.upscaler_options(reg)] == ["ok"]
    assert not any(p.startswith("\\\\") for p in seen)


class _UpscaleLib:
    def __init__(self, scale=4):
        self.scale = scale
        self.contexts = 0

    def get_upscaler_model_scale(self, path):
        return self.scale

    def new_upscaler_ctx(self, *args):
        self.contexts += 1
        return None


@pytest.fixture
def child_state(monkeypatch):
    from localm.media.sdcpp import _child
    monkeypatch.setattr(_child._State, "lib", _UpscaleLib())
    monkeypatch.setattr(_child._State, "paths", ("D:/models/main.gguf",))
    monkeypatch.setattr(_child._State, "last_errors", [])
    return _child


def _upscale_payload(side, **extra):
    return {"model_path": "D:/models/up.safetensors", "runtime_dir": "x",
            "image": {"width": side, "height": side, "channel": 3, "data": bytes(side * side * 3)},
            **extra}


def test_an_upscale_keeps_the_loaded_models_paths_in_the_redaction_list(child_state):
    with pytest.raises(RuntimeError, match="could not load the upscaler"):
        child_state._do_upscale_image(_upscale_payload(8))
    assert child_state._State.paths == ("D:/models/main.gguf", "D:/models/up.safetensors")


def test_an_upscale_past_the_output_limit_is_refused_before_loading_the_model(child_state):
    lib = child_state._State.lib
    with pytest.raises(ValueError, match="limited to 8192"):
        child_state._do_upscale_image(_upscale_payload(2100, max_output_side=8192))
    assert lib.contexts == 0
    with pytest.raises(RuntimeError):
        child_state._do_upscale_image(_upscale_payload(2048, max_output_side=8192))
    assert lib.contexts == 1


def test_a_file_that_is_not_an_esrgan_model_is_refused(child_state):
    child_state._State.lib.scale = 0
    with pytest.raises(ValueError, match="is not an RGB ESRGAN"):
        child_state._do_upscale_image(_upscale_payload(8))
    assert child_state._State.lib.contexts == 0


def test_the_backend_hands_the_worker_the_output_limit(qol, tmp_path):
    runner, s = qol
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    ok, msg = native.upscale(s, _png(tmp_path / "s.png"), "up4x", tmp_path / "o.png")
    assert ok, msg
    assert runner.upscales[0]["max_output_side"] == native.MAX_UPSCALE_OUTPUT_SIDE


def test_progress_during_an_upscale_is_labelled_upscaling_not_the_last_job_phase(child_state):
    import queue
    q = queue.Queue()
    _log_cb, progress_cb, *_rest = child_state._make_callbacks(q)
    child_state._State.decoding = True
    child_state._State.sample_steps = 4
    child_state._State.upscaling = True
    progress_cb(4, 4, 0.2, None)
    assert q.get_nowait() == ("progress", {"phase": "upscaling", "step": 4, "steps": 4,
                                           "secs": pytest.approx(0.2, abs=1e-3)})
    child_state._State.upscaling = False
    progress_cb(4, 4, 0.2, None)
    assert q.get_nowait()[1]["phase"] == "decoding"


def test_an_upscale_resets_the_phase_state_of_the_previous_job(child_state):
    child_state._State.decoding = True
    child_state._State.sample_steps = child_state._State.hires_steps = 9
    with pytest.raises(RuntimeError):
        child_state._do_upscale_image(_upscale_payload(8))
    assert child_state._State.decoding is False
    assert (child_state._State.sample_steps, child_state._State.hires_steps) == (0, 0)
    assert child_state._State.upscaling is False


def test_the_relay_words_upscale_progress_as_upscaling():
    lines = []
    relay = native._event_relay(lines.append)
    relay(("progress", {"phase": "upscaling", "step": 2, "steps": 6, "secs": 0.1}))
    relay(("progress", {"phase": "upscaling", "step": 6, "steps": 6, "secs": 0.1}))
    assert lines == ["Upscaling 2/6", "Upscaling 6/6"]


@pytest.fixture(autouse=True)
def _no_live_vram_budget(monkeypatch):
    from localm.media.sdcpp import budget as sd_budget
    monkeypatch.setattr(sd_budget, "vram_budget", lambda backend, **kw: None)
