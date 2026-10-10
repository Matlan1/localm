# SPDX-License-Identifier: AGPL-3.0-or-later
"""The managed KoboldCpp music server, driven against a real subprocess that
speaks KoboldCpp's music API over real HTTP on 127.0.0.1: start and readiness,
the API key, the planner merge, reuse between jobs, restart on a different model
set, cancel and stop killing exactly the started process, and every start failure
reported with the process output. Also the WAV rewrite and the backend order."""

from __future__ import annotations

import io
import json
import struct
import sys
import textwrap
import time
import wave
from pathlib import Path

import pytest

from localm.media.koboldcpp import music, runtime, server
from localm.media.koboldcpp.runtime import Runtime

FAKE = textwrap.dedent(r'''
    import io, json, os, sys, time, wave
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    args = sys.argv[1:]
    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    port = int(opt("--port"))
    password = opt("--password")
    record = os.environ["FAKE_RECORD"]
    mode = os.environ.get("FAKE_MODE", "ok")
    with open(record, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "argv": args, "pid": os.getpid(),
                            "host": opt("--host")}) + "\n")
    print("Loading Music Gen LLM Model", flush=True)
    if os.environ.get("FAKE_GRANDCHILD") == "1":
        import subprocess
        gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        with open(record, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "grandchild", "pid": gc.pid}) + "\n")
    if mode == "crash":
        print("FATAL: failed to load DiT model", flush=True)
        sys.exit(3)
    if mode == "slowload":
        time.sleep(float(os.environ.get("FAKE_SLEEP", "30")))

    def wav(seconds):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
            w.writeframes(b"\x01\x00\x02\x00" * int(48000 * seconds))
        return buf.getvalue()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path == "/api/extra/version":
                self._send(200, json.dumps({"result": "KoboldCpp", "version": "fake",
                                            "music": mode != "nomusic"}).encode())
            else:
                self._send(404, b"{}")
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            auth = self.headers.get("Authorization")
            with open(record, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "post", "path": self.path, "body": body,
                                    "auth_ok": auth == "Bearer " + password}) + "\n")
            if auth != "Bearer " + password:
                self._send(401, b'{"detail": "Unauthorized"}')
                return
            if self.path == "/api/extra/music/prepare":
                planned = dict(body, caption="rewritten by the planner", bpm=136,
                               keyscale="G major", audio_codes="1,2,3")
                self._send(200, json.dumps(planned).encode())
            elif self.path == "/api/extra/music/generate":
                if mode == "slowgen":
                    time.sleep(float(os.environ.get("FAKE_SLEEP", "30")))
                if mode == "dieongen":
                    os._exit(9)
                self._send(200, wav(float(body.get("duration", 1.0))), "audio/wav")
            else:
                self._send(404, b"{}")

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')


@pytest.fixture
def fake_kcpp(tmp_path, monkeypatch):
    script = tmp_path / "fake_koboldcpp.py"
    script.write_text(FAKE, encoding="utf-8")
    record = tmp_path / "record.jsonl"
    record.write_text("", encoding="utf-8")
    monkeypatch.setenv("FAKE_RECORD", str(record))
    real_build = server.build_argv

    def build(key, port, password):
        argv = real_build(key, port, password)
        return [getattr(sys, "_base_executable", None) or sys.executable, str(script),
                *argv[1:]]

    monkeypatch.setattr(server, "build_argv", build)
    monkeypatch.setattr(server, "HEARTBEAT_SECONDS", 0.5)
    yield record
    server.stop()


def _events(record: Path) -> list:
    return [json.loads(ln) for ln in record.read_text(encoding="utf-8").splitlines() if ln]


def _rt(tmp_path) -> Runtime:
    return Runtime(build="nocuda", path=tmp_path, launcher=tmp_path / "koboldcpp-launcher")


MODELS = server.ModelSet(text_encoder="te.gguf", dit="dit.gguf", vae="vae.gguf", lm="lm.gguf")
REQUEST = {"caption": "calm piano", "lyrics": "[Instrumental]", "instrumental": True,
           "duration": 2.0, "seed": 7, "stereo": True}


def _pid_alive(pid: int) -> bool:
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _tree(pid: int) -> set:
    """*pid* and its descendants (a venv python.exe on Windows is a launcher
    whose child runs the script)."""
    import psutil
    try:
        proc = psutil.Process(pid)
        return {pid} | {c.pid for c in proc.children(recursive=True)}
    except psutil.NoSuchProcess:
        return set()


def test_build_argv_is_loopback_music_only_with_the_key():
    key = server.ServerKey("launcher", "vulkan", MODELS, lowvram=True)
    argv = server.build_argv(key, 5123, "secret")
    assert argv[0] == "launcher"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "5123"
    assert argv[argv.index("--password") + 1] == "secret"
    for flag, value in (("--musicembeddings", "te.gguf"), ("--musicdiffusion", "dit.gguf"),
                        ("--musicvae", "vae.gguf"), ("--musicllm", "lm.gguf")):
        assert argv[argv.index(flag) + 1] == value
    for flag in ("--skiplauncher", "--quiet", "--usevulkan", "--musiclowvram"):
        assert flag in argv
    for forbidden in ("--launch", "--model", "--sdmodel", "--remotetunnel", "--admin"):
        assert forbidden not in argv
    no_lm = server.ServerKey("launcher", "cpu", server.ModelSet("t", "d", "v"))
    argv = server.build_argv(no_lm, 1, "k")
    assert "--musicllm" not in argv and "--usecpu" in argv and "--musiclowvram" not in argv


def test_a_job_starts_the_server_plans_and_generates(fake_kcpp, tmp_path):
    lines = []
    data = server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "work", prepare=True,
                      request=REQUEST, timeout=60, on_progress=lines.append)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnframes() == 2 * 48000
    ev = _events(fake_kcpp)
    start = ev[0]
    assert start["host"] == "127.0.0.1"
    posts = [e for e in ev if e["event"] == "post"]
    assert [p["path"] for p in posts] == ["/api/extra/music/prepare",
                                          "/api/extra/music/generate"]
    assert all(p["auth_ok"] for p in posts)
    assert posts[0]["body"]["gen_codes"] is True
    assert posts[0]["body"]["rewrite_caption"] is False
    gen = posts[1]["body"]
    assert gen["caption"] == "calm piano"
    assert gen["audio_codes"] == "1,2,3" and gen["bpm"] == 136
    assert gen["seed"] == 7 and gen["duration"] == 2.0
    assert any("Music models loaded" in ln for ln in lines)
    assert any(ln.startswith("Planning the track") for ln in lines)
    assert any(ln.startswith("Audio generated in") for ln in lines)
    assert start["pid"] in _tree(server.running_pid())


def test_the_server_is_reused_and_restarted_for_another_model_set(fake_kcpp, tmp_path):
    rt = _rt(tmp_path)
    server.run(rt, "vulkan", MODELS, tmp_path / "w", prepare=False, request=REQUEST, timeout=60)
    first = server.running_pid()
    server.run(rt, "vulkan", MODELS, tmp_path / "w", prepare=False, request=REQUEST, timeout=60)
    assert server.running_pid() == first
    other = server.ModelSet("te2.gguf", "dit.gguf", "vae.gguf")
    server.run(rt, "vulkan", other, tmp_path / "w", prepare=False, request=REQUEST, timeout=60)
    second = server.running_pid()
    assert second != first
    assert not _pid_alive(first)
    starts = [e for e in _events(fake_kcpp) if e["event"] == "start"]
    assert len(starts) == 2
    assert not _pid_alive(starts[0]["pid"])
    assert len({s["argv"][s["argv"].index("--password") + 1] for s in starts}) == 2


def test_without_planning_only_generate_is_called(fake_kcpp, tmp_path):
    server.run(_rt(tmp_path), "cpu", MODELS, tmp_path / "w", prepare=False,
               request=REQUEST, timeout=60)
    paths = [e["path"] for e in _events(fake_kcpp) if e["event"] == "post"]
    assert paths == ["/api/extra/music/generate"]


def test_stop_kills_exactly_the_started_process(fake_kcpp, tmp_path):
    server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "w", prepare=False,
               request=REQUEST, timeout=60)
    pid = server.running_pid()
    script_pid = _events(fake_kcpp)[0]["pid"]
    assert script_pid in _tree(pid)
    assert server.stop() is True
    assert not _pid_alive(pid)
    assert not _pid_alive(script_pid)
    assert server.running_pid() is None
    assert server.stop() is False


def test_stop_kills_what_the_server_started(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_GRANDCHILD", "1")
    server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "w", prepare=False,
               request=REQUEST, timeout=60)
    gc = [e["pid"] for e in _events(fake_kcpp) if e["event"] == "grandchild"]
    assert len(gc) == 1 and _pid_alive(gc[0])
    server.stop()
    deadline = time.monotonic() + 10
    while _pid_alive(gc[0]) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _pid_alive(gc[0])


def test_cancel_during_generation_kills_the_server(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slowgen")
    started = time.monotonic()
    flag = {"cancel": False}

    def cancel():
        if time.monotonic() - started > 3:
            flag["cancel"] = True
        return flag["cancel"]

    lines = []
    with pytest.raises(server.Cancelled):
        server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "w", prepare=False,
                   request=REQUEST, timeout=120, on_progress=lines.append,
                   cancel_check=cancel)
    pid = _events(fake_kcpp)[0]["pid"]
    assert not _pid_alive(pid)
    assert server.running_pid() is None
    assert any(ln.startswith("Still generating") for ln in lines)


def test_cancel_while_loading_kills_the_server(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slowload")
    t0 = time.monotonic()
    with pytest.raises(server.Cancelled):
        server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "w", prepare=False,
                   request=REQUEST, timeout=60,
                   cancel_check=lambda: time.monotonic() - t0 > 2)
    deadline = time.monotonic() + 10
    while not _events(fake_kcpp) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _pid_alive(_events(fake_kcpp)[0]["pid"])


def test_a_crash_while_loading_is_a_start_error_with_the_output(fake_kcpp, tmp_path,
                                                               monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    with pytest.raises(server.StartError) as caught:
        server.run(_rt(tmp_path), "cuda", MODELS, tmp_path / "w", prepare=False,
                   request=REQUEST, timeout=60)
    assert "exit code 3" in str(caught.value)
    assert "failed to load DiT model" in str(caught.value)
    assert server.running_pid() is None


def test_models_not_loaded_is_a_start_error(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "nomusic")
    with pytest.raises(server.StartError, match="did not load the music models"):
        server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "w", prepare=False,
                   request=REQUEST, timeout=60)
    pid = _events(fake_kcpp)[0]["pid"]
    assert not _pid_alive(pid)


def test_a_slow_load_times_out_as_a_start_error(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slowload")
    monkeypatch.setattr(server, "START_TIMEOUT", 2.0)
    with pytest.raises(server.StartError, match="did not finish loading in 2 s"):
        server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "w", prepare=False,
                   request=REQUEST, timeout=60)
    assert not _pid_alive(_events(fake_kcpp)[0]["pid"])


def test_the_server_dying_mid_request_is_reported_and_forgotten(fake_kcpp, tmp_path,
                                                               monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "dieongen")
    with pytest.raises(server.ServerError, match="stopped during the request") as caught:
        server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "w", prepare=False,
                   request=REQUEST, timeout=60)
    assert not isinstance(caught.value, server.StartError)
    assert server.running_pid() is None


def test_an_idle_server_is_stopped(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "IDLE_SECONDS", 1.5)
    server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "w", prepare=False,
               request=REQUEST, timeout=60)
    pid = server.running_pid()
    deadline = time.monotonic() + 15
    while server.running_pid() is not None and time.monotonic() < deadline:
        time.sleep(0.2)
    assert server.running_pid() is None
    assert not _pid_alive(pid)


def test_proxy_settings_do_not_divert_loopback_requests(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    data = server.run(_rt(tmp_path), "vulkan", MODELS, tmp_path / "w", prepare=False,
                      request=REQUEST, timeout=60)
    assert data.startswith(b"RIFF")


def _wav_with_metadata() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x01" * 2 * 4800)
    raw = buf.getvalue()
    info = b"INFOISFT" + struct.pack("<I", 6) + b"Lavf\x00\x00"
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


def test_write_wav_keeps_only_format_and_data(tmp_path):
    src = _wav_with_metadata()
    assert b"LIST" in _chunks(src)
    out = tmp_path / "track.wav"
    seconds = music.write_wav(src, out)
    assert seconds == pytest.approx(0.1)
    assert _chunks(out.read_bytes()) == [b"fmt ", b"data"]
    assert not (tmp_path / "track.wav.part").exists()


@pytest.mark.parametrize("data", [b"", b"not a wav at all", b"RIFF\x00\x00\x00\x00WAVE"])
def test_write_wav_refuses_unreadable_audio(tmp_path, data):
    out = tmp_path / "track.wav"
    with pytest.raises(server.ServerError):
        music.write_wav(data, out)
    assert not out.exists()


def test_backend_order_auto_skips_a_failed_backend(monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    monkeypatch.setattr(runtime, "recommended_backend", lambda det=None: "cuda")
    assert music.backend_order("auto") == ["cuda", "vulkan", "cpu"]
    runtime.record_backend("cuda", False, "no driver")
    assert music.backend_order("auto") == ["vulkan", "cpu"]
    assert music.backend_order("cuda") == ["cuda"]
    with pytest.raises(music.NativeMusicError, match="unknown native music backend"):
        music.backend_order("rocm")
    with pytest.raises(music.NativeMusicError, match="no KoboldCpp build"):
        music.backend_order("metal")


def test_auto_falls_back_when_a_backend_does_not_start(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    monkeypatch.setattr(runtime, "recommended_backend", lambda det=None: "cuda")
    monkeypatch.setattr(runtime, "ensure_for_backend",
                        lambda backend, **k: _rt(tmp_path))
    monkeypatch.setattr(music, "resolve_models", lambda *a, **k: MODELS)
    monkeypatch.setattr(music, "work_dir", lambda: tmp_path / "w")
    lines = []
    monkeypatch.setenv("FAKE_MODE", "ok")
    import os as _os
    orig_start = server._Server.start

    def start(self, on_progress, cancel_check):
        if self.key.backend == "cuda":
            _os.environ["FAKE_MODE"] = "crash"
        try:
            return orig_start(self, on_progress, cancel_check)
        finally:
            _os.environ["FAKE_MODE"] = "ok"

    monkeypatch.setattr(server._Server, "start", start)
    data, used = music.generate_wav({}, "auto", REQUEST, plan=False, on_progress=lines.append)
    assert used == "vulkan" and data.startswith(b"RIFF")
    assert runtime.backend_failed("cuda") is not None
    assert runtime.backend_worked("vulkan")
    assert any("cuda backend did not start" in ln and "trying vulkan" in ln for ln in lines)


def test_an_explicit_backend_that_does_not_start_is_not_swapped(fake_kcpp, tmp_path,
                                                               monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    monkeypatch.setattr(runtime, "ensure_for_backend",
                        lambda backend, **k: _rt(tmp_path))
    monkeypatch.setattr(music, "resolve_models", lambda *a, **k: MODELS)
    monkeypatch.setattr(music, "work_dir", lambda: tmp_path / "w")
    with pytest.raises(music.NativeMusicError, match="vulkan: the music runtime exited"):
        music.generate_wav({}, "vulkan", REQUEST, plan=False)
    starts = [e for e in _events(fake_kcpp) if e["event"] == "start"]
    assert len(starts) == 1 and "--usevulkan" in starts[0]["argv"]
    assert runtime.backend_failed("vulkan") is None
