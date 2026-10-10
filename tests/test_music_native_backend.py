# SPDX-License-Identifier: AGPL-3.0-or-later
"""The music plugin's native backend and backend selection: ``auto`` keeps an
existing ComfyUI setup and picks native otherwise, an explicit choice is never
swapped, inputs only ComfyUI can honour are refused, the native backend writes a
metadata-free WAV plus a sidecar only outside privacy mode, and every entry point
(the route, ``/api/music/backend``, the GUI pre-check, ``localm music`` and the
chat REPL) follows the selected backend. The KoboldCpp process itself is covered
by test_koboldcpp_server; here the boundary is ``music.generate_wav`` /
``music.prepare``."""

from __future__ import annotations

import io
import json
import struct
import threading
import time as _time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localm.media.koboldcpp import models, music
from localm.plugins.builtin.music import backend as music_backend
from localm.plugins.builtin.music.backends import native


def _wav(seconds: float = 0.5, with_list: bool = True) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x01\x00\x02\x00" * int(48000 * seconds))
    raw = buf.getvalue()
    if not with_list:
        return raw
    info = b"INFOISFT" + struct.pack("<I", 4) + b"kcpp"
    lst = b"LIST" + struct.pack("<I", len(info)) + info
    body = raw[12:36] + lst + raw[36:]
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def _chunks(data: bytes) -> list:
    out, i = [], 12
    while i + 8 <= len(data):
        cid, size = data[i:i + 4], struct.unpack("<I", data[i + 4:i + 8])[0]
        out.append(cid)
        i += 8 + size + (size & 1)
    return out


class _Comfy(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = b'{"system": {}}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def comfy_answering():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Comfy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def no_comfy(monkeypatch):
    """Nothing about ComfyUI is set up and the ComfyUI path must never run."""
    import localm.image_gen.comfy as comfy
    import localm.music_gen as music_gen
    from localm.media import comfy_client

    def never(*a, **k):
        raise AssertionError("the ComfyUI path ran for a native job")

    monkeypatch.setattr(comfy, "ensure_comfy", never)
    monkeypatch.setattr(music_gen, "generate_music", never)
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda *a, **k: False)
    monkeypatch.delenv("FLUX_API_URL", raising=False)


# --------------------------------------------------------------------------- #
#  Backend selection                                                           #
# --------------------------------------------------------------------------- #

def test_auto_on_a_fresh_home_is_native_with_a_note(no_comfy):
    s = music_backend.settings({})
    assert s["backend"] == "native" and s["backend_choice"] == "auto"
    assert s["backend_note"] == "Music backend: native ACE-Step (auto: no ComfyUI is set up)."
    assert s["native_runtime"] == "auto" and s["plan"] is True and s["lowvram"] is False
    assert music_backend._impl(s) is native


@pytest.mark.parametrize("cfg", [
    {"comfy_target": "user"},
    {"comfy_api_url": "http://127.0.0.1:8188"},
    {"plugins": {"music": {"comfy": {"workdir": "D:/path/to/ComfyUI"}}}},
])
def test_auto_keeps_an_existing_comfyui_setup(no_comfy, cfg):
    s = music_backend.settings(cfg)
    assert s["backend"] == "comfy" and "ComfyUI is set up" in s["backend_note"]


def test_auto_switches_to_a_comfyui_that_answers_in_the_job(monkeypatch, comfy_answering):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    cfg = {"plugins": {"music": {}}}
    s = music_backend.settings(cfg)
    s["api_url"] = comfy_answering
    assert s["backend"] == "native"
    music_backend.prepare_for_job(s, cfg)
    assert s["backend"] == "comfy" and comfy_answering in s["backend_note"]


@pytest.mark.parametrize("choice", ["native", "comfy"])
def test_an_explicit_choice_is_never_swapped(no_comfy, choice):
    cfg = {"comfy_target": "user", "plugins": {"music": {"backend": choice}}}
    s = music_backend.prepare_for_job(music_backend.settings(cfg), cfg)
    assert s["backend"] == choice and s["backend_note"] is None


def test_the_native_block_is_read(no_comfy):
    cfg = {"plugins": {"music": {"backend": "native",
                                 "native": {"runtime": "CPU", "plan": False, "lowvram": True}}}}
    s = music_backend.settings(cfg)
    assert s["native_runtime"] == "cpu" and s["plan"] is False and s["lowvram"] is True


def test_a_native_job_estimates_vram_from_its_model_files(no_comfy, tmp_path):
    from tests.test_gguf_architecture_roles import _gguf
    files = {}
    for comp, arch in models.ARCHITECTURES.items():
        if comp != "lm":
            files[comp] = str(_gguf(tmp_path / f"{comp}.gguf", arch))
    sizes = sum((tmp_path / f"{c}.gguf").stat().st_size for c in files)
    cfg = {"plugins": {"music": {"backend": "native", "native": dict(files, plan=False)}}}
    s = music_backend.prepare_for_job(music_backend.settings(cfg), cfg)
    assert s["vram_estimate_bytes"] == sizes + models._OVERHEAD_WITHOUT_LM
    cfg["plugins"]["music"]["vram_estimate_gb"] = 3
    s = music_backend.prepare_for_job(music_backend.settings(cfg), cfg)
    assert s["vram_estimate_bytes"] == 3 * 1024 ** 3


def test_without_downloaded_models_the_estimate_is_the_default_set(no_comfy):
    s = music_backend.prepare_for_job(music_backend.settings({}), {})
    assert s["vram_estimate_bytes"] == (sum(models.DEFAULT_SIZES.values())
                                        + models._OVERHEAD_WITH_LM)


# --------------------------------------------------------------------------- #
#  The native backend module                                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_generate(monkeypatch):
    calls = []

    def gen(native_cfg, choice, request, **kw):
        calls.append({"cfg": native_cfg, "choice": choice, "request": request, **kw})
        cut = 1.0 if kw.get("plan") and request["duration"] > 1.5 else 0.0
        return _wav(float(request["duration"]) - cut), "vulkan"

    monkeypatch.setattr(music, "generate_wav", gen)
    monkeypatch.setattr(music, "prepare", lambda *a, **k: ("vulkan", None))
    return calls


def _s(**over):
    s = {"backend": "native", "native": {}, "native_runtime": "auto", "plan": True,
         "lowvram": False}
    s.update(over)
    return s


def test_generate_writes_a_clean_wav_and_a_sidecar(tmp_path, fake_generate):
    out = tmp_path / "t.wav"
    ok, msg = native.generate(_s(native_runtime="cpu"), "calm piano", out, lyrics="la la",
                              duration_seconds=2.0, seed=42, steps=8, cfg=1.0, shift=3.0)
    assert ok, msg
    assert _chunks(out.read_bytes()) == [b"fmt ", b"data"]
    call = fake_generate[0]
    assert call["request"] == {"caption": "calm piano", "lyrics": "la la",
                               "instrumental": False, "duration": 2.0, "seed": 42,
                               "stereo": True, "inference_steps": 8,
                               "guidance_scale": 1.0, "shift": 3.0}
    assert call["plan"] is True and call["choice"] == "cpu"
    side = json.loads(out.with_suffix(".wav.json").read_text(encoding="utf-8"))
    assert side["tags"] == "calm piano" and side["lyrics"] == "la la"
    assert side["seed"] == 42 and side["backend"] == "native (vulkan)"
    assert side["length_seconds"] == pytest.approx(1.0)
    assert "requested 2 s" in msg and "seed 42" in msg


def test_privacy_mode_writes_no_sidecar(tmp_path, fake_generate):
    out = tmp_path / "t.wav"
    ok, _msg = native.generate(_s(), "x", out, write_sidecar=False, duration_seconds=1.0)
    assert ok and out.is_file()
    assert not out.with_suffix(".wav.json").exists()


def test_instrumental_and_a_random_seed_when_none_is_given(tmp_path, fake_generate):
    native.generate(_s(plan=False), "x", tmp_path / "t.wav", duration_seconds=1.0)
    req = fake_generate[0]["request"]
    assert req["lyrics"] == "[Instrumental]" and req["instrumental"] is True
    assert isinstance(req["seed"], int) and req["seed"] > 0
    assert "inference_steps" not in req and "guidance_scale" not in req


@pytest.mark.parametrize("kwargs,match", [
    ({"model_overrides": {"1": {"ckpt_name": "a"}}}, "Workflow model choices apply to ComfyUI"),
    ({"sampler_name": "euler"}, "sampler setting applies to the ComfyUI workflow"),
    ({"scheduler": "karras", "lyrics_strength": 1.2}, "scheduler, lyrics strength"),
    ({"placement": {"unet": 1}}, "GPU placement applies to ComfyUI"),
])
def test_comfyui_only_inputs_are_refused_before_any_work(tmp_path, fake_generate, kwargs,
                                                        match):
    assert match in (native.refusal(**kwargs) or "")
    ok, msg = native.generate(_s(), "x", tmp_path / "t.wav", duration_seconds=1.0, **kwargs)
    assert ok is False and match in msg
    assert fake_generate == []


@pytest.mark.parametrize("exc,expected", [
    (music.NativeMusicError("cuda: no driver"), "Native music generation failed: cuda: no driver"),
    (music.ModelError("no dit"), "Native music generation failed: no dit"),
    (music.Cancelled(), "Cancelled."),
])
def test_failures_return_the_reason_and_write_nothing(tmp_path, monkeypatch, exc, expected):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(music, "generate_wav", boom)
    out = tmp_path / "t.wav"
    ok, msg = native.generate(_s(), "x", out, duration_seconds=1.0)
    assert ok is False and msg == expected
    assert not out.exists()


def test_ensure_available_reports_why_it_cannot_run(monkeypatch):
    def boom(*a, **k):
        raise music.ProvisionError("the download was refused by the network policy")
    monkeypatch.setattr(music, "prepare", boom)
    ok, msg = native.ensure_available(_s())
    assert ok is False and "refused by the network policy" in msg
    monkeypatch.setattr(music, "prepare", lambda *a, **k: ("vulkan", None))
    ok, msg = native.ensure_available(_s())
    assert ok is True and "vulkan" in msg


def test_free_vram_stops_the_music_server(monkeypatch):
    from localm.media.koboldcpp import server
    stopped = []
    monkeypatch.setattr(server, "stop", lambda: stopped.append(1) or True)
    assert native.free_vram(_s()) is True and stopped == [1]


def test_status_lists_the_defaults_to_download(no_comfy):
    st = native.status(_s())
    assert [m["component"] for m in st["missing"]] == ["text_encoder", "dit", "vae", "lm"]
    for m in st["missing"]:
        assert m["spec"] == f"{models.DEFAULT_REPO}:{models.DEFAULT_FILES[m['component']]}"
        assert m["size_bytes"] == models.DEFAULT_SIZES[m["component"]]
        assert m["model_type"] == models.REGISTRY_TYPES[m["component"]]
    assert set(st["models"].values()) == {None}
    no_plan = native.status(_s(plan=False))
    assert "lm" not in no_plan["models"] and len(no_plan["missing"]) == 3


# --------------------------------------------------------------------------- #
#  Entry points                                                                #
# --------------------------------------------------------------------------- #

def _music_app(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("music")

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=switch_model,
               active_model=lambda: "model-a")
    return app


def _wait_job(client, job_id, headers, timeout=30):
    lines, end = [], None
    deadline = _time.monotonic() + timeout
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=headers) as r:
        for raw in r.iter_lines():
            if _time.monotonic() > deadline:
                break
            if not raw.startswith("data: "):
                continue
            ev = json.loads(raw[6:])
            if ev["type"] == "line":
                lines.append(ev["text"])
            if ev["type"] == "end":
                end = ev
                break
    return end, lines


def _key(scopes):
    from localm import auth
    return {"Authorization": "Bearer " + auth.create_key("k", scopes)["key"]}


def test_route_generates_natively_on_a_fresh_home(tmp_path, monkeypatch, no_comfy,
                                                  fake_generate):
    from fastapi.testclient import TestClient
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda *a, **k: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda *a, **k: None)
    app = _music_app(tmp_path, monkeypatch)
    h = _key(["music"])
    with TestClient(app) as c:
        r = c.post("/api/music", headers=h, json={"tags": "lofi", "duration_seconds": 2})
        assert r.status_code == 200
        end, lines = _wait_job(c, r.json()["job_id"], h)
        assert end and end["status"] == "done", lines
        assert end["result"].endswith(".wav")
        assert "Music backend: native ACE-Step (auto: no ComfyUI is set up)." in lines
        f = c.get(f"/api/music/file/{end['result']}", headers=h)
        assert f.status_code == 200 and f.headers["content-type"] == "audio/wav"
        assert _chunks(f.content) == [b"fmt ", b"data"]
        names = [t["name"] for t in c.get("/api/music/history", headers=h).json()["tracks"]]
        assert end["result"] in names


def test_route_refuses_workflow_picks_for_native(tmp_path, monkeypatch, no_comfy,
                                                 fake_generate):
    from fastapi.testclient import TestClient
    prepared = []
    monkeypatch.setattr(music, "prepare",
                        lambda *a, **k: prepared.append(1) or ("vulkan", None))
    app = _music_app(tmp_path, monkeypatch)
    h = _key(["music"])
    with TestClient(app) as c:
        r = c.post("/api/music", headers=h, json={
            "tags": "lofi", "duration_seconds": 2,
            "model_overrides": {"1": {"ckpt_name": "ace.safetensors"}}})
        end, lines = _wait_job(c, r.json()["job_id"], h)
        assert end and end["status"] == "failed"
        assert any("Workflow model choices apply to ComfyUI" in ln for ln in lines)
        assert fake_generate == []
        assert prepared == []


def test_route_explicit_native_failure_fails_the_job_without_comfyui(tmp_path, monkeypatch,
                                                                    no_comfy):
    from fastapi.testclient import TestClient

    def refuse(*a, **k):
        raise music.ProvisionError("KoboldCpp publishes no build for this platform")
    monkeypatch.setattr(music, "prepare", refuse)
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("plugins", {}).setdefault("music", {})["backend"] = "native"
    save_config(cfg)
    app = _music_app(tmp_path, monkeypatch)
    h = _key(["music"])
    with TestClient(app) as c:
        r = c.post("/api/music", headers=h, json={"tags": "lofi", "duration_seconds": 2})
        end, lines = _wait_job(c, r.json()["job_id"], h)
        assert end and end["status"] == "failed"
        assert any("no build for this platform" in ln for ln in lines)


def test_backend_route_reports_native_and_what_it_uses(tmp_path, monkeypatch, no_comfy):
    from fastapi.testclient import TestClient
    app = _music_app(tmp_path, monkeypatch)
    with TestClient(app) as c:
        r = c.get("/api/music/backend", headers=_key(["music"]))
        assert r.status_code == 200
        data = r.json()
        assert data["choice"] == "auto" and data["active"] == "native"
        assert data["note"].startswith("Music backend: native ACE-Step")
        assert len(data["native"]["missing"]) == 4


def test_preflight_offers_the_default_models_for_native(tmp_path, monkeypatch, no_comfy):
    from fastapi.testclient import TestClient
    import localm.media.comfy_client as cc
    comfy_checked = []
    monkeypatch.setattr(cc, "describe_missing_models",
                        lambda *a, **k: comfy_checked.append(1) or [])
    app = _music_app(tmp_path, monkeypatch)
    with TestClient(app) as c:
        r = c.post("/api/media/music/preflight", headers=_key(["models:write", "music"]),
                   json={})
        assert r.status_code == 200
        data = r.json()
        files = [m["filename"] for m in data["missing"]]
        assert files == [models.DEFAULT_FILES[c] for c in ("text_encoder", "dit", "vae", "lm")]
        assert all(m["native"] and m["source"]["spec"].startswith(models.DEFAULT_REPO)
                   for m in data["missing"])
        assert data["status"] == "verified" and data["warning"] is None
    assert comfy_checked == []


def test_cli_music_runs_through_the_backend_and_stops_the_runtime(tmp_path, monkeypatch,
                                                                  cli_runner, no_comfy,
                                                                  fake_generate):
    from localm.media.koboldcpp import server
    stopped = []
    monkeypatch.setattr(server, "stop", lambda: stopped.append(1) or True)
    from localm.cli import main
    out = tmp_path / "song.wav"
    result = cli_runner.invoke(main, ["music", "lofi", "-d", "1", "-o", str(out), "--seed", "5"])
    assert result.exit_code == 0, result.output
    assert out.is_file() and stopped == [1]
    assert fake_generate[0]["request"]["seed"] == 5
    assert "native ACE-Step" in result.output
    bad = cli_runner.invoke(main, ["music", "lofi", "-o", str(tmp_path / "song.flac")])
    assert bad.exit_code == 2 and "ending in .wav" in bad.output


def test_repl_generate_music_runs_through_the_backend(tmp_path, monkeypatch, no_comfy,
                                                      fake_generate):
    from rich.console import Console
    from localm.cli import chat
    from localm.media.koboldcpp import server
    monkeypatch.setattr(server, "stop", lambda: True)

    class Engine:
        unloaded = 0

        def unload(self):
            Engine.unloaded += 1

    console = Console(file=io.StringIO(), width=200)
    chat._cmd_generate_media("generate-music", "happy lo-fi", Engine(), console, tmp_path)
    files = list((tmp_path / "gui_music").glob("*_cli.wav"))
    assert len(files) == 1 and Engine.unloaded == 1
    assert "Track saved to" in console.file.getvalue()


def test_a_job_never_downloads_models_and_says_how_to_get_them(monkeypatch, no_comfy):
    from localm.media.koboldcpp import runtime

    def never(*a, **k):
        raise AssertionError("a job started a download")

    monkeypatch.setattr(models, "_pull", never)
    monkeypatch.setattr(runtime, "ensure_for_backend", never)
    ok, msg = native.ensure_available(_s())
    assert ok is False
    assert "not downloaded" in msg and "localm setup-music" in msg and "Music page" in msg
    assert "ACE-Step text encoder, diffusion, VAE and planner models are not" in msg
    size = sum(models.DEFAULT_SIZES.values()) / 1024 ** 3
    assert f"({size:.1f} GB)" in msg
