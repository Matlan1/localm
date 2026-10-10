# SPDX-License-Identifier: AGPL-3.0-or-later
"""The native (stable-diffusion.cpp) video backend and how the video plugin
chooses it. The worker is replaced by an in-process fake that returns frames;
the MP4 is written by the real PyAV encoder and read back."""

from __future__ import annotations

import io
import json
import struct
import sys
import time
from pathlib import Path

import pytest

import localm.media.comfy_client as comfy_client
from localm.media import backend_choice
from localm.media.sdcpp import runtime as sd_runtime
from localm.media.sdcpp import shared
from localm.media.sdcpp.runner import SdCancelled, SdWorkerError
from localm.plugins.builtin.video import backend as video_backend
from localm.plugins.builtin.video.backends import native

av = pytest.importorskip("av")


@pytest.fixture
def no_comfy(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    monkeypatch.setattr(backend_choice, "managed_comfy_active", lambda cfg=None: False)
    monkeypatch.setattr(comfy_client, "is_comfy_confirmed", lambda api_url=None: False)
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: False)


def test_auto_without_comfyui_is_native(no_comfy):
    s = video_backend.settings({})
    assert (s["backend_choice"], s["backend"]) == ("auto", "native")
    assert s["backend_note"].startswith("Video backend: native")
    assert s["warning"] is None


def test_auto_with_comfyui_configured_is_comfy(no_comfy):
    s = video_backend.settings({"comfy_target": "user"})
    assert s["backend"] == "comfy"


def test_explicit_choices_are_never_overridden(no_comfy, monkeypatch):
    s = video_backend.settings({"plugins": {"video": {"backend": "comfy"}}})
    assert (s["backend"], s["backend_note"]) == ("comfy", None)
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: True)
    s = video_backend.settings({"plugins": {"video": {"backend": "native"}}})
    video_backend.prepare_for_job(s, {"plugins": {"video": {"backend": "native"}}})
    assert s["backend"] == "native"


def test_auto_switches_to_a_comfyui_that_answers_at_job_time(no_comfy, monkeypatch):
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: True)
    s = video_backend.prepare_for_job(video_backend.settings({}), {})
    assert s["backend"] == "comfy"


@pytest.mark.parametrize("seconds,fps,expected", [
    (1, 16, 17), (2, 16, 33), (5, 16, 81), (0.1, 8, 5), (20, 60, native.MAX_FRAMES)])
def test_frame_count_is_4k_plus_1(seconds, fps, expected):
    n = native.frame_count(seconds, fps)
    assert n == expected and (n - 1) % 4 == 0


# --------------------------------------------------------------------------- #
#  model resolution                                                           #
# --------------------------------------------------------------------------- #

def _register(name: str, path: Path, model_type: str = "diffusion-unet") -> None:
    from localm.config import REGISTRY_FILE
    reg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8")) if REGISTRY_FILE.exists() else {}
    reg[name] = {"path": str(path), "model_type": model_type, "source": "test"}
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(reg), encoding="utf-8")


def test_no_model_lists_every_recommended_download():
    with pytest.raises(native._ModelError) as ei:
        native.resolve_models({"native": {}})
    for part in native.RECOMMENDED_VIDEO_MODELS[0].parts:
        assert f"localm pull {part.spec}" in str(ei.value)


def test_the_recommended_set_resolves_once_every_part_is_registered(tmp_path):
    rec = native.RECOMMENDED_VIDEO_MODELS[0]
    for part in rec.parts[:-1]:
        f = tmp_path / part.filename
        f.write_bytes(b"x")
        _register(part.role, f, part.model_type)
    with pytest.raises(native._ModelError) as ei:
        native.resolve_models({"native": {}})
    assert f"localm pull {rec.parts[-1].spec}" in str(ei.value)
    assert f"localm pull {rec.parts[0].spec}" not in str(ei.value)
    last = rec.parts[-1]
    (tmp_path / last.filename).write_bytes(b"x")
    _register(last.role, tmp_path / last.filename, last.model_type)
    m = native.resolve_models({"native": {}})
    assert set(m["ctx"]) == {"diffusion_model_path", "vae_path", "t5xxl_path",
                             "diffusion_flash_attn"}
    assert m["ctx"]["diffusion_flash_attn"] is True
    assert m["recommended"] is rec and m["label"] == rec.name


def test_explicit_parts_name_only_their_files(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    for name in ("dit.safetensors", "t5.gguf", "vae.safetensors", "cv.safetensors"):
        (private / name).write_bytes(b"w")
    s = {"native": {"model": str(private / "dit.safetensors"),
                    "t5xxl": str(private / "t5.gguf"), "vae": str(private / "vae.safetensors"),
                    "clip_vision": str(private / "cv.safetensors")}}
    m = native.resolve_models(s)
    assert m["ctx"]["clip_vision_path"] == str(private / "cv.safetensors")
    assert m["label"] == "dit.safetensors"
    assert native.status(s)["model"] == "dit.safetensors"
    with pytest.raises(native._ModelError) as ei:
        native.resolve_models({"native": {"model": str(private / "gone.safetensors")}})
    assert "gone.safetensors" in str(ei.value) and "private" not in str(ei.value)


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
        native.resolve_models({"native": {"model": "\\\\192.0.2.1\\share\\wan.safetensors"}})
    assert "192.0.2.1" not in str(ei.value)
    _register("remote", Path("\\\\192.0.2.1\\share\\" + native.RECOMMENDED_VIDEO_MODELS[0]
                                .parts[0].filename))
    assert native.missing_parts(native.RECOMMENDED_VIDEO_MODELS[0])[0].role == "model"
    assert seen == []


def test_status_lists_the_parts_still_to_download_with_their_digests(tmp_path):
    rec = native.RECOMMENDED_VIDEO_MODELS[0]
    vae = rec.parts[1]
    (tmp_path / vae.filename).write_bytes(b"x")
    _register("vae", tmp_path / vae.filename, vae.model_type)
    st = native.status({"native": {}})
    assert st["model"] is None and "No native video model" in st["missing"]
    parts = st["recommended"]["parts"]
    assert [p["role"] for p in parts] == ["model", "t5xxl"]
    assert {p["model_type"] for p in parts} == {"diffusion-unet", "text-encoder"}
    assert all(len(p["sha256"]) == 64 for p in parts)
    assert st["recommended"]["size_bytes"] == sum(p.size_bytes for p in rec.parts)


# --------------------------------------------------------------------------- #
#  generate                                                                   #
# --------------------------------------------------------------------------- #

class _FakeRunner:
    def __init__(self, *, video=True, raise_on_generate=None, audio=None, desc=""):
        self.video = video
        self.desc = desc
        self.raise_on_generate = raise_on_generate
        self.audio = audio
        self.loaded_key = None
        self.loads, self.generates = [], []
        self.alive = False
        self.pid = 4343

    def is_alive(self):
        return self.alive

    def ensure_loaded(self, key, runtime_dir, ctx, **kw):
        self.loads.append(ctx)
        self.loaded_key = key
        self.alive = True
        return {"version": "Wan 2.x", "desc": self.desc, "image": False, "video": self.video}

    def generate_video(self, params, *, on_event=None, cancel_check=None, timeout=None):
        self.generates.append(params)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        if on_event:
            on_event(("progress", {"phase": "sampling", "step": 1, "steps": 2, "secs": 1.5}))
        w, h, n = params["width"], params["height"], params["video_frames"]
        frames = [bytes([(i * 20) % 256, 80, 160]) * (w * h) for i in range(n)]
        return {"width": w, "height": h, "channel": 3, "frames": frames,
                "fps": params["fps"], "audio": self.audio, "seed": params["seed"]}

    def shutdown(self, grace=5.0):
        self.alive = False


@pytest.fixture
def fake_video(tmp_path, monkeypatch, no_comfy):
    runner = _FakeRunner()
    monkeypatch.setattr(shared, "runner", runner)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    rt = sd_runtime.Runtime(backend="vulkan", path=tmp_path / "rt")
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": rt)
    files = {}
    for k in ("dit", "t5", "vae"):
        files[k] = tmp_path / f"{k}.gguf"
        files[k].write_bytes(b"w")
    s = {"backend": "native", "native": {"model": str(files["dit"]), "t5xxl": str(files["t5"]),
                                         "vae": str(files["vae"])}}
    return runner, s


def _probe(path: Path):
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        frames = sum(1 for _ in c.decode(video=0))
        return {"w": vs.codec_context.width, "h": vs.codec_context.height, "frames": frames,
                "fps": float(vs.average_rate or 0), "audio": len(c.streams.audio)}


def test_generate_writes_an_mp4_with_every_frame(fake_video, tmp_path):
    runner, s = fake_video
    out = tmp_path / "clip.mp4"
    lines = []
    ok, msg = native.generate(s, "a fox runs", out, seconds=1, fps=16, width=128, height=96,
                              seed=3, write_sidecar=True, on_progress=lines.append)
    assert ok, msg
    info = _probe(out)
    assert (info["w"], info["h"], info["frames"]) == (128, 96, 17)
    assert round(info["fps"]) == 16 and info["audio"] == 0
    assert runner.loads[0] == {"diffusion_flash_attn": True,
                               "diffusion_model_path": s["native"]["model"],
                               "t5xxl_path": s["native"]["t5xxl"], "vae_path": s["native"]["vae"]}
    side = json.loads(out.with_suffix(".mp4.json").read_text(encoding="utf-8"))
    assert side["frames"] == 17 and side["backend"] == "native" and side["model"] == "dit.gguf"
    assert any(line.startswith("Step 1/2") for line in lines)
    assert not list(tmp_path.glob(".*.tmp.mp4"))


def test_the_default_size_is_the_recommended_one(fake_video, tmp_path):
    runner, s = fake_video
    ok, msg = native.generate(s, "p", tmp_path / "d.mp4", seconds=0.1, fps=8,
                              write_sidecar=False)
    assert ok, msg
    g = runner.generates[0]
    assert (g["width"], g["height"]) == (native.DEFAULT_WIDTH, native.DEFAULT_HEIGHT)


def test_no_sidecar_when_not_requested(fake_video, tmp_path):
    _runner, s = fake_video
    out = tmp_path / "out" / "n.mp4"
    ok, _ = native.generate(s, "p", out, seconds=0.1, fps=8, width=64, height=64,
                            write_sidecar=False)
    assert ok
    assert [p.name for p in out.parent.iterdir()] == ["n.mp4"]


def test_audio_returned_by_the_model_is_muxed(fake_video, tmp_path):
    runner, s = fake_video
    n = 16000
    runner.audio = {"sample_rate": 16000, "channels": 1, "sample_count": n,
                    "data": struct.pack(f"<{n}f", *([0.1] * n))}
    out = tmp_path / "talk.mp4"
    ok, msg = native.generate(s, "p", out, seconds=1, fps=16, width=64, height=64,
                              write_sidecar=False)
    assert ok, msg
    assert _probe(out)["audio"] == 1


@pytest.mark.parametrize("kwargs,needle", [
    ({"model_overrides": {"1": {"unet_name": "x"}}}, "ComfyUI"),
    ({"placement": {"unet": 1}}, "placement"),
    ({"width": 100, "height": 96}, "multiple of 16"),
    ({"width": 2048, "height": 96}, "multiple of 16"),
    ({"width": 96}, "both width and height"),
    ({"float_type": "fp16"}, "does not support: float_type"),
])
def test_requests_the_native_backend_cannot_honour_are_refused(fake_video, tmp_path,
                                                                kwargs, needle):
    runner, s = fake_video
    ok, msg = native.generate(s, "p", tmp_path / "x.mp4", write_sidecar=False, **kwargs)
    assert not ok and needle in msg
    assert runner.loads == [] and runner.generates == []


def test_a_model_that_cannot_make_video_is_refused(fake_video, tmp_path):
    runner, s = fake_video
    runner.video = False
    ok, msg = native.generate(s, "p", tmp_path / "x.mp4", write_sidecar=False)
    assert not ok and "not a video generation model" in msg
    assert runner.generates == [] and not runner.alive


def test_cancel_and_worker_failure_are_reported(fake_video, tmp_path):
    runner, s = fake_video
    runner.raise_on_generate = SdCancelled("Generation cancelled.")
    assert native.generate(s, "p", tmp_path / "a.mp4", write_sidecar=False) == (False, "Cancelled.")
    runner.raise_on_generate = SdWorkerError("worker crashed")
    ok, msg = native.generate(s, "p", tmp_path / "b.mp4", write_sidecar=False)
    assert not ok and "crashed" in msg
    assert not (tmp_path / "a.mp4").exists() and not (tmp_path / "b.mp4").exists()


def test_generate_never_raises_and_releases_the_worker(fake_video, tmp_path):
    runner, s = fake_video
    runner.raise_on_generate = RuntimeError("boom")
    out = tmp_path / "c.mp4"
    ok, msg = native.generate(s, "p", out, write_sidecar=True)
    assert not ok and "RuntimeError: boom" in msg
    assert not out.exists() and not out.with_suffix(".mp4.json").exists()
    assert shared.lock.acquire(blocking=False)
    shared.lock.release()


def test_missing_pyav_is_a_clear_error(fake_video, monkeypatch):
    _runner, s = fake_video
    monkeypatch.setitem(sys.modules, "av", None)
    monkeypatch.setitem(sys.modules, "av.codec", None)
    ok, msg = native.ensure_available(s)
    assert not ok and "PyAV" in msg


def test_ensure_available_reports_an_unexpected_model_check_failure(monkeypatch):
    def broken(s):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(native, "resolve_models", broken)
    ok, msg = native.ensure_available({"native": {}})
    assert not ok and "registry unreadable" in msg


# --------------------------------------------------------------------------- #
#  the real video plugin routes, on the native backend                        #
# --------------------------------------------------------------------------- #

@pytest.fixture
def video_app(fake_video, tmp_path, monkeypatch):
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    from fastapi import FastAPI

    from localm.config import update_config
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.jobs import JobManager
    runner, s = fake_video
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "video", {}).update({"native": dict(s["native"])}))
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("video")
    app.state.jobs = JobManager()
    app.state.self_url = "http://127.0.0.1:8642/v1"
    return app, runner


def _wait_job(app, job_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = app.state.jobs.get(job_id)
        if job is not None and job.status != "running" and getattr(job, "finished_at", None):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _lines(job):
    return [e.get("text", "") for e in job._history if e.get("type") == "line"]


def test_video_route_generates_natively(video_app):
    from fastapi.testclient import TestClient

    from localm.media import paths as media_paths
    app, runner = video_app
    with TestClient(app) as c:
        r = c.post("/api/video", json={"prompt": "a lighthouse", "seconds": 1, "fps": 8,
                                       "width": 96, "height": 64})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "done", _lines(job)
    out = media_paths.gallery_dir(media_paths.VIDEO_DIR_NAME) / job.result
    info = _probe(out)
    assert (info["w"], info["h"], info["frames"]) == (96, 64, 9)
    assert any("native stable-diffusion.cpp" in ln for ln in _lines(job))
    assert not any("Submitting Wan workflow" in ln for ln in _lines(job))
    assert runner.generates[0]["prompt"] == "a lighthouse"


def test_a_native_refusal_fails_the_job_before_any_vram_handover(video_app, monkeypatch):
    from fastapi.testclient import TestClient
    app, runner = video_app
    unloads, provisions = [], []
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: True)
    monkeypatch.setattr("localm.vram.unload_chat_for_media",
                        lambda *a, **k: unloads.append(a) or True)
    monkeypatch.setattr(sd_runtime, "provision", lambda *a, **k: provisions.append(a))
    with TestClient(app) as c:
        r = c.post("/api/video", json={"prompt": "p", "width": 100, "height": 96})
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "failed"
    assert any("multiple of 16" in ln for ln in _lines(job))
    assert unloads == [] and provisions == [] and runner.loads == []


def test_backend_route_reports_native_and_only_file_names(video_app):
    from fastapi.testclient import TestClient
    app, _runner = video_app
    with TestClient(app) as c:
        data = c.get("/api/video/backend").json()
    assert data["choice"] == "auto" and data["active"] == "native"
    assert data["native"]["model"] == "dit.gguf" and data["native"]["missing"] is None
    assert data["native"]["recommended"]["width"] == native.DEFAULT_WIDTH


def test_backend_route_reports_comfy_when_one_answers_under_auto(video_app, monkeypatch):
    from fastapi.testclient import TestClient
    app, _runner = video_app
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: True)
    with TestClient(app) as c:
        data = c.get("/api/video/backend").json()
    assert data["active"] == "comfy" and "native" not in data


def test_explicit_comfy_without_comfyui_fails_and_never_runs_native(video_app, monkeypatch):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    app, runner = video_app
    update_config(lambda cfg: cfg["plugins"]["video"].update({"backend": "comfy"}))
    mod = sys.modules["_localm_plugin_video.backend"]
    monkeypatch.setattr(mod._COMFY_REF, "ensure_available",
                        lambda s, on_progress=None: (False, "ComfyUI is not running"))
    with TestClient(app) as c:
        r = c.post("/api/video", json={"prompt": "p"})
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "failed"
    assert runner.loads == [] and runner.generates == []


def test_comfy_launch_route_always_starts_comfyui(video_app, monkeypatch):
    from fastapi.testclient import TestClient

    import localm.image_gen.comfy as ic
    app, runner = video_app
    seen = []
    monkeypatch.setattr(ic, "ensure_comfy", lambda *a, **k: (seen.append(a), (True, "up"))[1])
    with TestClient(app) as c:
        r = c.post("/api/video/comfy-launch")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert seen and runner.loads == []


# --------------------------------------------------------------------------- #
#  GUI preflight                                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def gui_app(no_comfy):
    from fastapi import FastAPI

    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None, active_model=lambda: "model-a")
    return app


def test_preflight_offers_every_missing_recommended_part(gui_app):
    from fastapi.testclient import TestClient
    with TestClient(gui_app) as c:
        r = c.post("/api/media/video/preflight", json={})
    assert r.status_code == 200, r.text
    data = r.json()
    rec = native.RECOMMENDED_VIDEO_MODELS[0]
    assert data["status"] == "verified"
    assert [m["filename"] for m in data["missing"]] == [p.filename for p in rec.parts]
    for entry, part in zip(data["missing"], rec.parts, strict=True):
        assert entry["native"] is True
        assert entry["source"]["spec"] == part.spec
        assert entry["source"]["sha256"] == part.sha256
        assert entry["source"]["model_type"] == part.model_type


def test_preflight_reports_nothing_missing_once_the_model_resolves(gui_app, tmp_path):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    model = tmp_path / "dit.gguf"
    model.write_bytes(b"x")
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "video", {}).update({"native": {"model": str(model)}}))
    with TestClient(gui_app) as c:
        data = c.post("/api/media/video/preflight", json={}).json()
    assert data == {"status": "verified", "missing": [], "warning": None}


def test_preflight_checks_comfyui_when_one_answers_under_auto(gui_app, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: True)
    asked = []
    monkeypatch.setattr(comfy_client, "describe_missing_models",
                        lambda workflow, api_url: asked.append(api_url) or [])
    with TestClient(gui_app) as c:
        data = c.post("/api/media/video/preflight", json={}).json()
    assert asked, data
    assert not any(m.get("native") for m in data.get("missing", []))


# --------------------------------------------------------------------------- #
#  the CLI and the chat REPL                                                  #
# --------------------------------------------------------------------------- #

def test_cli_video_uses_the_native_backend(fake_video, cli_runner, tmp_path):
    runner, s = fake_video
    from localm.config import update_config
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "video", {}).update({"native": s["native"]}))
    from localm.cli import main
    out = tmp_path / "cli.mp4"
    result = cli_runner.invoke(main, ["video", "a fox", "-d", "1", "--fps", "8",
                                      "--width", "64", "--height", "64", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert _probe(out)["frames"] == 9
    assert "native stable-diffusion.cpp" in result.output
    assert not runner.alive


def test_cli_video_reports_a_native_refusal(fake_video, cli_runner, tmp_path):
    runner, s = fake_video
    from localm.config import update_config
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "video", {}).update({"native": s["native"]}))
    from localm.cli import main
    result = cli_runner.invoke(main, ["video", "a fox", "--width", "100", "--height", "96",
                                      "-o", str(tmp_path / "x.mp4")])
    assert result.exit_code == 1
    assert "multiple of 16" in result.output and runner.loads == []


def test_chat_repl_generates_video_natively(fake_video, tmp_path):
    from rich.console import Console

    from localm.cli.chat import _cmd_generate_media
    from localm.config import update_config
    from localm.media import paths as media_paths
    runner, s = fake_video
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "video", {}).update({"native": s["native"]}))

    class _Engine:
        unloaded = 0

        def unload(self):
            _Engine.unloaded += 1

    out = io.StringIO()
    _cmd_generate_media("generate-video", "a fox", _Engine(), Console(file=out, width=200),
                        tmp_path)
    clips = list((tmp_path / media_paths.VIDEO_DIR_NAME).glob("*.mp4"))
    assert len(clips) == 1, out.getvalue()
    assert _Engine.unloaded == 1
    assert not runner.alive


# --------------------------------------------------------------------------- #
#  settings                                                                   #
# --------------------------------------------------------------------------- #

def test_video_native_fields_validate_and_nest():
    from localm.settings_schema import validate_media_block
    merged = validate_media_block("video", {"backend": "native", "native_flow_shift": "3",
                                            "native_clip_vision": " cv.safetensors ",
                                            "native_steps": "20"})
    assert merged == {"backend": "native",
                      "native": {"flow_shift": 3.0, "clip_vision": "cv.safetensors",
                                 "steps": 20}}


@pytest.mark.parametrize("plugin,field", [("video", "native_clip_g"),
                                          ("image", "native_flow_shift"),
                                          ("image", "native_clip_vision"),
                                          ("music", "backend")])
def test_native_fields_belong_to_their_plugins(plugin, field):
    from localm.settings_schema import validate_media_block
    with pytest.raises(ValueError, match="unknown media field"):
        validate_media_block(plugin, {field: "x"})


def test_video_model_paths_are_owner_only():
    from localm.settings_schema import media_admin_only_fields, media_schema_json
    assert {"native_clip_vision", "native_model", "native_t5xxl"} <= media_admin_only_fields()
    public = {f["key"] for f in media_schema_json("video", {}, {}, is_owner=False)}
    assert "native_model" not in public and "native_clip_vision" not in public
    assert {"backend", "native_flow_shift", "native_runtime"} <= public


# --------------------------------------------------------------------------- #
#  requests the native backend refuses up front                              #
# --------------------------------------------------------------------------- #

def _register_recommended(tmp_path, roles=("model", "vae", "t5xxl")):
    rec = native.RECOMMENDED_VIDEO_MODELS[0]
    for part in rec.parts:
        if part.role in roles:
            f = tmp_path / part.filename
            f.write_bytes(b"x")
            _register(part.role, f, part.model_type)
    return rec


def test_a_start_image_is_refused_for_the_recommended_text_to_video_model(
        fake_video, tmp_path):
    from PIL import Image
    runner, _s = fake_video
    _register_recommended(tmp_path)
    src = tmp_path / "start.png"
    Image.new("RGB", (64, 64)).save(src)
    ok, msg = native.generate({"backend": "native", "native": {}}, "p", tmp_path / "x.mp4",
                              write_sidecar=False, input_image=src)
    assert not ok and "text-to-video" in msg and "wan2.1-t2v-1.3b" in msg
    assert runner.loads == [] and runner.generates == []


def test_a_start_image_is_refused_once_a_text_to_video_model_is_loaded(fake_video, tmp_path):
    from PIL import Image
    runner, s = fake_video
    runner.desc = "Wan2.1-T2V-14B"
    src = tmp_path / "start.png"
    Image.new("RGB", (64, 64)).save(src)
    out = tmp_path / "y.mp4"
    ok, msg = native.generate(s, "p", out, write_sidecar=False, input_image=src,
                              width=64, height=64)
    assert not ok and "text-to-video" in msg and "dit.gguf" in msg
    assert runner.generates == [] and not out.exists()


def test_a_start_image_reaches_an_image_to_video_model(fake_video, tmp_path):
    from PIL import Image
    runner, s = fake_video
    runner.desc = "Wan2.1-I2V-14B"
    src = tmp_path / "start.png"
    Image.new("RGB", (100, 50)).save(src)
    ok, msg = native.generate(s, "p", tmp_path / "z.mp4", write_sidecar=False,
                              input_image=src, width=64, height=64, seconds=0.1, fps=8)
    assert ok, msg
    init = runner.generates[0]["init_image"]
    assert (init["width"], init["height"]) == (64, 64)


@pytest.mark.parametrize("seconds,fps", [(20, 24), (5, 60), (11, 24)])
def test_a_clip_longer_than_the_frame_limit_is_refused(fake_video, tmp_path, seconds, fps):
    runner, s = fake_video
    ok, msg = native.generate(s, "p", tmp_path / "l.mp4", write_sidecar=False,
                              seconds=seconds, fps=fps)
    assert not ok and f"at most {native.MAX_FRAMES} frames" in msg
    assert runner.loads == []


def test_the_longest_allowed_clip_is_generated(fake_video, tmp_path):
    runner, s = fake_video
    ok, msg = native.generate(s, "p", tmp_path / "m.mp4", write_sidecar=False,
                              seconds=10, fps=24, width=64, height=64)
    assert ok, msg
    assert runner.generates[0]["video_frames"] == native.MAX_FRAMES


def test_part_settings_replace_the_recommended_parts(tmp_path):
    rec = _register_recommended(tmp_path, roles=("model", "vae"))
    own_t5 = tmp_path / "own" / "my-umt5.gguf"
    own_t5.parent.mkdir()
    own_t5.write_bytes(b"t")
    own_vae = tmp_path / "own" / "my-vae.safetensors"
    own_vae.write_bytes(b"v")
    blk = {"t5xxl": str(own_t5), "vae": str(own_vae)}
    assert native.missing_parts(rec, blk) == []
    m = native.resolve_models({"native": blk})
    assert m["recommended"] is rec
    assert m["ctx"]["t5xxl_path"] == str(own_t5)
    assert m["ctx"]["vae_path"] == str(own_vae)
    with pytest.raises(native._ModelError) as ei:
        native.resolve_models({"native": {"vae": str(own_vae)}})
    assert rec.parts[2].spec in str(ei.value) and rec.parts[1].spec not in str(ei.value)


def test_a_registered_name_on_a_network_path_is_refused_without_touching_it(monkeypatch):
    seen = []

    def spy(real):
        def wrapped(self, *a, **k):
            if str(self).replace("/", "\\").startswith("\\\\"):
                seen.append(str(self))
                return False
            return real(self, *a, **k)
        return wrapped

    _register("remote-wan", Path("\\\\192.0.2.1\\share\\wan.safetensors"))
    monkeypatch.setattr(Path, "exists", spy(Path.exists))
    monkeypatch.setattr(Path, "is_file", spy(Path.is_file))
    with pytest.raises(native._ModelError, match="network or device path") as ei:
        native.resolve_models({"native": {"model": "remote-wan"}})
    assert "192.0.2.1" not in str(ei.value)
    assert seen == []


def test_a_registered_model_whose_file_is_gone_says_so(tmp_path):
    _register("gone-wan", tmp_path / "private" / "wan.safetensors")
    with pytest.raises(native._ModelError) as ei:
        native.resolve_models({"native": {"model": "gone-wan"}})
    assert "registered, but its file 'wan.safetensors' is missing" in str(ei.value)
    assert "private" not in str(ei.value)


def test_the_cli_refuses_before_installing_the_runtime(fake_video, cli_runner, tmp_path,
                                                       monkeypatch):
    runner, s = fake_video
    provisions = []
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": None)
    monkeypatch.setattr(sd_runtime, "provision", lambda *a, **k: provisions.append(a))
    from localm.config import update_config
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault(
        "video", {}).update({"native": s["native"]}))
    from localm.cli import main
    result = cli_runner.invoke(main, ["video", "a fox", "-d", "20", "--fps", "24",
                                      "-o", str(tmp_path / "x.mp4")])
    assert result.exit_code == 1, result.output
    assert "at most" in result.output
    assert provisions == [] and runner.loads == []


def test_the_video_route_refuses_a_start_image_before_any_vram_handover(video_app, tmp_path,
                                                                         monkeypatch):
    from fastapi.testclient import TestClient
    from PIL import Image

    from localm.config import update_config
    from localm.media import paths as media_paths
    app, runner = video_app
    _register_recommended(tmp_path)
    update_config(lambda cfg: cfg["plugins"]["video"].update({"native": {}}))
    uploads = media_paths.gallery_dir(media_paths.IMAGE_DIR_NAME)
    uploads.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64)).save(uploads / "start.png")
    unloads = []
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: True)
    monkeypatch.setattr("localm.vram.unload_chat_for_media",
                        lambda *a, **k: unloads.append(a) or True)
    with TestClient(app) as c:
        r = c.post("/api/video", json={"prompt": "p", "input_image": str(uploads / "start.png")})
        assert r.status_code == 200, r.text
        job = _wait_job(app, r.json()["job_id"])
    assert job.status == "failed"
    assert any("text-to-video" in ln for ln in _lines(job))
    assert unloads == [] and runner.loads == []


# --------------------------------------------------------------------------- #
#  the worker's handling of native log lines                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def child_state(monkeypatch):
    from localm.media.sdcpp import _child
    monkeypatch.setattr(_child._State, "paths", ())
    monkeypatch.setattr(_child._State, "redact", ())
    monkeypatch.setattr(_child._State, "model_desc", "")
    monkeypatch.setattr(_child._State, "last_errors", [])
    return _child


def test_relayed_native_lines_name_model_files_only(child_state):
    import queue
    _child = child_state
    _child._State.paths = ("D:\\models\\private\\wan.safetensors", "/srv/private/umt5.gguf")
    q = queue.Queue()
    log_cb, _progress = _child._make_callbacks(q)
    log_cb(3, b"loading diffusion model from 'D:\\models\\private\\wan.safetensors' failed"
              b" --- diffusion_engine.cpp:736", None)
    log_cb(4, b"failed to open 'D:/models/private/wan.safetensors' --- safetensors_io.cpp:158",
           None)
    log_cb(4, b"file /srv/private/umt5.gguf not found --- model_loader.cpp:233", None)
    lines = [q.get_nowait()[2] for _ in range(3)]
    assert lines[0].startswith("loading diffusion model from 'wan.safetensors' failed")
    assert lines[1].startswith("failed to open 'wan.safetensors'")
    assert lines[2].startswith("file umt5.gguf not found")
    assert not any("private" in ln for ln in lines + _child._State.last_errors)


def test_the_worker_records_the_wan_model_it_loaded(child_state):
    import queue
    _child = child_state
    q = queue.Queue()
    log_cb, _progress = _child._make_callbacks(q)
    log_cb(2, b"model manager memory on ROCm0: reported free 10018.88 MB", None)
    log_cb(2, b"Wan2.1-T2V-1.3B --- wan.hpp:921", None)
    log_cb(2, b"Wan2.1-I2V-14B --- wan.hpp:917", None)
    assert _child._State.model_desc == "Wan2.1-T2V-1.3B"
    assert q.empty()
