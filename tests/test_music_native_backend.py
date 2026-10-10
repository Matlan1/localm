# SPDX-License-Identifier: AGPL-3.0-or-later
"""The music plugin's native backend and backend selection: ``auto`` keeps an
existing ComfyUI setup and picks native otherwise, an explicit choice is never
swapped, the native backend writes a metadata-free WAV plus a sidecar only
outside privacy mode, and every entry point (the route, ``localm music``, the
chat REPL, the GUI pre-check) follows the selected backend. The KoboldCpp
process itself is covered by test_koboldcpp_server; here the boundary is
``music.generate_wav`` / ``music.prepare``."""

from __future__ import annotations

import io
import json
import socket
import struct
import threading
import time as _time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localm.media.koboldcpp import music
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


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Comfy(BaseHTTPRequestHandler):
    def log_message(self, *a):
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
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


# --------------------------------------------------------------------------- #
#  Backend selection                                                           #
# --------------------------------------------------------------------------- #

def _choose(choice="auto", cfg=None, comfy=None, api_url=None, own=False):
    url = api_url or f"http://127.0.0.1:{_closed_port()}"
    return music_backend.resolve_backend_choice(choice, cfg or {}, comfy or {}, url, own)


def test_auto_without_any_comfyui_is_native(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    assert _choose() == ("native", "auto: no ComfyUI is set up")


@pytest.mark.parametrize("kwargs,reason", [
    ({"own": True}, "managed ComfyUI is installed"),
    ({"cfg": {"comfy_target": "user"}}, "your own ComfyUI"),
    ({"comfy": {"workdir": "D:/path/to/ComfyUI"}}, "a ComfyUI is configured"),
    ({"comfy": {"launch_cmd": "run.bat"}}, "a ComfyUI is configured"),
    ({"cfg": {"comfy_api_url": "http://127.0.0.1:8188"}}, "a ComfyUI is configured"),
])
def test_auto_keeps_an_existing_comfyui_setup(monkeypatch, kwargs, reason):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    name, why = _choose(**kwargs)
    assert name == "comfy" and reason in why


def test_auto_counts_the_flux_api_url_env_as_configured(monkeypatch):
    monkeypatch.setenv("FLUX_API_URL", "http://127.0.0.1:8188")
    assert _choose()[0] == "comfy"


def test_auto_uses_a_comfyui_that_answers(monkeypatch, comfy_answering):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    name, why = _choose(api_url=comfy_answering)
    assert name == "comfy" and comfy_answering in why


@pytest.mark.parametrize("choice", ["native", "comfy", "Native ", "ghost"])
def test_an_explicit_choice_is_returned_as_given(monkeypatch, comfy_answering, choice):
    name, why = _choose(choice, own=True, api_url=comfy_answering)
    assert name == choice.strip().lower() and why == "selected in settings"


def test_settings_resolve_native_on_a_fresh_home(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    monkeypatch.setattr(music_backend._comfy, "_comfy_alive", lambda *a, **k: False)
    s = music_backend.settings({})
    assert s["backend"] == "native"
    assert s["native_backend"] == "auto" and s["plan"] is True and s["lowvram"] is False
    assert s["vram_estimate_bytes"] == int(7.6 * 1024 ** 3)
    assert music_backend._impl(s) is native


def test_settings_honour_an_explicit_comfy_and_the_native_block(monkeypatch):
    monkeypatch.setattr(music_backend._comfy, "_comfy_alive", lambda *a, **k: False)
    cfg = {"plugins": {"music": {"backend": "comfy",
                                 "native": {"backend": "CPU", "plan": False,
                                            "lowvram": True}}}}
    s = music_backend.settings(cfg)
    assert s["backend"] == "comfy" and music_backend._impl(s) is music_backend._COMFY_REF
    assert s["native_backend"] == "cpu" and s["plan"] is False and s["lowvram"] is True


def test_the_native_vram_estimate_comes_from_the_model_files(monkeypatch, tmp_path):
    monkeypatch.setattr(music_backend._comfy, "_comfy_alive", lambda *a, **k: False)
    files = {}
    for comp, size in (("text_encoder", 3000), ("dit", 5000), ("vae", 700)):
        p = tmp_path / f"{comp}.gguf"
        p.write_bytes(b"\0" * size)
        files[comp] = str(p)
    cfg = {"plugins": {"music": {"backend": "native", "native": dict(files, plan=False)}}}
    s = music_backend.settings(cfg)
    assert s["vram_estimate_bytes"] == 8700 + int(1.75 * 1024 ** 3)


# --------------------------------------------------------------------------- #
#  The native backend module                                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_generate(monkeypatch):
    calls = []

    def gen(native_cfg, choice, request, **kw):
        calls.append({"cfg": native_cfg, "choice": choice, "request": request, **kw})
        return _wav(float(request["duration"]) - (0.2 if kw.get("plan") else 0.0)), "vulkan"

    monkeypatch.setattr(music, "generate_wav", gen)
    return calls


def _s(**over):
    s = {"backend": "native", "native": {}, "native_backend": "auto", "plan": True,
         "lowvram": False}
    s.update(over)
    return s


def test_generate_writes_a_clean_wav_and_a_sidecar(tmp_path, fake_generate):
    out = tmp_path / "t.wav"
    ok, msg = native.generate(_s(), "calm piano", out, lyrics="la la", duration_seconds=1.0,
                              seed=42, steps=8, cfg=1.0, shift=3.0)
    assert ok, msg
    assert _chunks(out.read_bytes()) == [b"fmt ", b"data"]
    req = fake_generate[0]["request"]
    assert req == {"caption": "calm piano", "lyrics": "la la", "instrumental": False,
                   "duration": 1.0, "seed": 42, "stereo": True, "inference_steps": 8,
                   "guidance_scale": 1.0, "shift": 3.0}
    assert fake_generate[0]["plan"] is True and fake_generate[0]["choice"] == "auto"
    side = json.loads(out.with_suffix(".wav.json").read_text(encoding="utf-8"))
    assert side["tags"] == "calm piano" and side["lyrics"] == "la la"
    assert side["seed"] == 42 and side["backend"] == "native (vulkan)"
    assert side["length_seconds"] == pytest.approx(0.8)
    assert "requested 1 s" in msg and "seed 42" in msg


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


def test_comfyui_only_options_are_reported_not_silently_dropped(tmp_path, fake_generate):
    lines = []
    native.generate(_s(), "x", tmp_path / "t.wav", duration_seconds=1.0,
                    on_progress=lines.append, sampler_name="euler",
                    model_overrides={"1": {"ckpt_name": "a"}})
    assert any("Not used by the native backend: sampler_name, model_overrides" in ln
               for ln in lines)


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


# --------------------------------------------------------------------------- #
#  Entry points                                                                #
# --------------------------------------------------------------------------- #

def _no_comfy(monkeypatch):
    import localm.image_gen.comfy as comfy
    import localm.music_gen as music_gen

    def never(*a, **k):
        raise AssertionError("the ComfyUI path ran for a native job")
    monkeypatch.setattr(comfy, "ensure_comfy", never)
    monkeypatch.setattr(music_gen, "generate_music", never)
    monkeypatch.setattr(music_backend._comfy, "_comfy_alive", lambda *a, **k: False)
    monkeypatch.delenv("FLUX_API_URL", raising=False)


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


def test_route_generates_natively_on_a_fresh_home(tmp_path, monkeypatch, fake_generate):
    from fastapi.testclient import TestClient
    _no_comfy(monkeypatch)
    monkeypatch.setattr(music, "prepare", lambda *a, **k: ("vulkan", None))
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
        assert "Music backend: native (auto: no ComfyUI is set up)." in lines
        f = c.get(f"/api/music/file/{end['result']}", headers=h)
        assert f.status_code == 200 and f.headers["content-type"] == "audio/wav"
        assert _chunks(f.content) == [b"fmt ", b"data"]
        names = [t["name"] for t in c.get("/api/music/history", headers=h).json()["tracks"]]
        assert end["result"] in names


def test_route_explicit_native_failure_fails_the_job_without_comfyui(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    _no_comfy(monkeypatch)

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
        assert "Music backend: native (selected in settings)." in lines
        assert any("no build for this platform" in ln for ln in lines)


def test_preflight_has_nothing_to_check_for_native(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    _no_comfy(monkeypatch)
    import localm.media.comfy_client as cc
    monkeypatch.setattr(cc, "describe_missing_models", lambda *a, **k: pytest.fail(
        "the ComfyUI workflow was checked for a native job"))
    app = _music_app(tmp_path, monkeypatch)
    with TestClient(app) as c:
        r = c.post("/api/media/music/preflight", headers=_key(["models:write", "music"]),
                   json={})
        assert r.status_code == 200
        assert r.json() == {"status": "verified", "missing": [], "warning": ""}


def test_cli_music_runs_natively_and_stops_the_runtime(tmp_path, monkeypatch, cli_runner,
                                                     fake_generate):
    _no_comfy(monkeypatch)
    from localm.media.koboldcpp import server
    stopped = []
    monkeypatch.setattr(server, "stop", lambda: stopped.append(1) or True)
    monkeypatch.setattr(music, "prepare", lambda *a, **k: ("cpu", None))
    from localm.cli import main
    out = tmp_path / "song.wav"
    result = cli_runner.invoke(main, ["music", "lofi", "-d", "1", "-o", str(out), "--seed", "5"])
    assert result.exit_code == 0, result.output
    assert out.is_file() and stopped == [1]
    assert fake_generate[0]["request"]["seed"] == 5
    bad = cli_runner.invoke(main, ["music", "lofi", "-o", str(tmp_path / "song.flac")])
    assert bad.exit_code == 2 and "ending in .wav" in bad.output


def test_repl_generate_music_runs_natively(tmp_path, monkeypatch, fake_generate):
    _no_comfy(monkeypatch)
    from rich.console import Console
    from localm.cli import chat
    from localm.media.koboldcpp import server
    monkeypatch.setattr(server, "stop", lambda: True)
    monkeypatch.setattr(music, "prepare", lambda *a, **k: ("vulkan", None))

    class Engine:
        unloaded = 0

        def unload(self):
            Engine.unloaded += 1

    console = Console(file=io.StringIO(), width=200)
    chat._cmd_generate_media("generate-music", "happy lo-fi", Engine(), console, tmp_path)
    files = list((tmp_path / "gui_music").glob("*_cli.wav"))
    assert len(files) == 1 and Engine.unloaded == 1
    assert "Track saved to" in console.file.getvalue()
