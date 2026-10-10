# SPDX-License-Identifier: AGPL-3.0-or-later
"""The managed KoboldCpp music server, driven against a real subprocess that
speaks KoboldCpp's music API over real HTTP on 127.0.0.1: start and readiness,
the API key (passed in the environment, never on the command line), the planner
merge, reuse between jobs, restart on a different model set, cancel and stop
killing exactly the started process tree, a port taken by another program, and
every start failure reported with the process output. Also the WAV rewrite, the
backend order, and which start failures ``auto`` remembers."""

from __future__ import annotations

import array
import io
import json
import os
import struct
import subprocess
import sys
import textwrap
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from localm.media.koboldcpp import _proc, music, runtime, server
from localm.media.koboldcpp.runtime import Runtime

FAKE = textwrap.dedent(r'''
    import array, io, json, os, sys, time, wave
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    args = sys.argv[1:]
    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    port = int(opt("--port"))
    password = opt("--password") or os.environ.get("KCPP_PASSWORD")
    record = os.environ["FAKE_RECORD"]
    mode = os.environ.get("FAKE_MODE", "ok")
    with open(record, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "argv": args, "pid": os.getpid(),
                            "host": opt("--host"), "key": password}) + "\n")
    print("Loading Music Gen LLM Model", flush=True)
    if os.environ.get("FAKE_GRANDCHILD") == "1":
        import subprocess
        gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        with open(record, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "grandchild", "pid": gc.pid}) + "\n")
    if mode == "crash":
        print("FATAL: failed to load DiT model", flush=True)
        sys.exit(3)
    if mode == "oomcrash":
        print("ggml_vulkan: Device memory allocation failed: out of memory", flush=True)
        sys.exit(1)
    if mode in ("slowload", "nobind"):
        time.sleep(float(os.environ.get("FAKE_SLEEP", "30")))

    def wav(seconds):
        n = 2 * int(48000 * seconds)
        if mode == "flat":
            samples = array.array("h", [32767]) * n
        else:
            samples = array.array("h", [((i // 2) % 200) - 100 for i in range(n)])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
            w.writeframes(samples.tobytes())
        return buf.getvalue()

    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
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

BASE_PYTHON = getattr(sys, "_base_executable", None) or sys.executable


@pytest.fixture
def started(monkeypatch):
    """Every ManagedProcess the server starts, in order."""
    seen = []
    real_start = _proc.start

    def start(argv, **kw):
        mp = real_start(argv, **kw)
        seen.append(mp)
        return mp

    monkeypatch.setattr(_proc, "start", start)
    return seen


@pytest.fixture
def fake_kcpp(tmp_path, monkeypatch, started):
    script = tmp_path / "fake_koboldcpp.py"
    script.write_text(FAKE, encoding="utf-8")
    record = tmp_path / "record.jsonl"
    record.write_text("", encoding="utf-8")
    monkeypatch.setenv("FAKE_RECORD", str(record))
    real_build = server.build_argv

    def build(key, port):
        argv = real_build(key, port)
        return [BASE_PYTHON, str(script), *argv[1:]]

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


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not _pid_alive(pid)


def _run(tmp_path, backend="vulkan", models=MODELS, **kw):
    kw.setdefault("prepare", False)
    kw.setdefault("timeout", 60)
    return server.run(_rt(tmp_path), backend, models, tmp_path / "w", request=REQUEST, **kw)


def test_build_argv_is_loopback_music_only_without_the_key():
    key = server.ServerKey("launcher", "vulkan", MODELS, lowvram=True)
    argv = server.build_argv(key, 5123)
    assert argv[0] == "launcher"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "5123"
    assert "--password" not in argv
    for flag, value in (("--musicembeddings", "te.gguf"), ("--musicdiffusion", "dit.gguf"),
                        ("--musicvae", "vae.gguf"), ("--musicllm", "lm.gguf")):
        assert argv[argv.index(flag) + 1] == value
    for flag in ("--skiplauncher", "--quiet", "--usevulkan", "--musiclowvram"):
        assert flag in argv
    for forbidden in ("--launch", "--model", "--sdmodel", "--remotetunnel", "--admin"):
        assert forbidden not in argv
    no_lm = server.ServerKey("launcher", "cpu", server.ModelSet("t", "d", "v"))
    argv = server.build_argv(no_lm, 1)
    assert "--musicllm" not in argv and "--usecpu" in argv and "--musiclowvram" not in argv


def test_a_job_starts_the_server_plans_and_generates(fake_kcpp, tmp_path, started):
    lines = []
    data = _run(tmp_path, prepare=True, on_progress=lines.append)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnframes() == 2 * 48000
    ev = _events(fake_kcpp)
    start = ev[0]
    assert start["host"] == "127.0.0.1"
    assert start["key"] and start["key"] not in " ".join(start["argv"])
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
    assert server.running_pid() == started[0].pid == start["pid"]


def test_the_server_is_reused_and_restarted_for_another_model_set(fake_kcpp, tmp_path, started):
    _run(tmp_path)
    first = server.running_pid()
    assert first is not None
    _run(tmp_path)
    assert server.running_pid() == first and len(started) == 1
    _run(tmp_path, models=server.ModelSet("te2.gguf", "dit.gguf", "vae.gguf"))
    assert len(started) == 2 and server.running_pid() == started[1].pid != first
    assert _wait_dead(first)
    keys = [e["key"] for e in _events(fake_kcpp) if e["event"] == "start"]
    assert len(set(keys)) == 2


def test_without_planning_only_generate_is_called(fake_kcpp, tmp_path):
    _run(tmp_path, backend="cpu")
    paths = [e["path"] for e in _events(fake_kcpp) if e["event"] == "post"]
    assert paths == ["/api/extra/music/generate"]


def test_stop_kills_exactly_the_started_process(fake_kcpp, tmp_path, started):
    _run(tmp_path)
    pid = started[0].pid
    assert server.stop() is True
    assert _wait_dead(pid)
    assert server.running_pid() is None
    assert server.stop() is False


def test_stop_kills_what_the_server_started(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_GRANDCHILD", "1")
    _run(tmp_path)
    gc = [e["pid"] for e in _events(fake_kcpp) if e["event"] == "grandchild"]
    assert len(gc) == 1 and _pid_alive(gc[0])
    try:
        t0 = time.monotonic()
        server.stop()
        assert time.monotonic() - t0 < 30
        assert _wait_dead(gc[0])
    finally:
        import psutil
        try:
            psutil.Process(gc[0]).kill()
        except psutil.NoSuchProcess:
            pass


def test_cancel_during_generation_kills_the_server(fake_kcpp, tmp_path, monkeypatch, started):
    monkeypatch.setenv("FAKE_MODE", "slowgen")
    lines = []
    seen = {}

    def cancel():
        if "Generating the audio..." in lines:
            seen.setdefault("t", time.monotonic())
            return time.monotonic() - seen["t"] > 1.5
        return False

    with pytest.raises(server.Cancelled):
        _run(tmp_path, timeout=120, on_progress=lines.append, cancel_check=cancel)
    assert _wait_dead(started[0].pid)
    assert server.running_pid() is None
    assert any(ln.startswith("Still generating") for ln in lines)


def test_cancel_while_loading_kills_the_server(fake_kcpp, tmp_path, monkeypatch, started):
    monkeypatch.setenv("FAKE_MODE", "slowload")
    t0 = time.monotonic()
    with pytest.raises(server.Cancelled):
        _run(tmp_path, cancel_check=lambda: time.monotonic() - t0 > 2)
    assert len(started) == 1 and _wait_dead(started[0].pid)
    assert server.running_pid() is None


def test_a_crash_while_loading_is_a_start_error_with_the_output(fake_kcpp, tmp_path,
                                                               monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    with pytest.raises(server.StartError) as caught:
        _run(tmp_path, backend="cuda")
    assert "exit code 3" in str(caught.value)
    assert "failed to load DiT model" in str(caught.value)
    assert caught.value.crashed is True
    assert server.running_pid() is None


def test_running_out_of_memory_while_loading_is_not_a_crash(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "oomcrash")
    with pytest.raises(server.StartError) as caught:
        _run(tmp_path, backend="cuda")
    assert "out of memory" in str(caught.value)
    assert caught.value.crashed is False


def test_models_not_loaded_is_a_start_error(fake_kcpp, tmp_path, monkeypatch, started):
    monkeypatch.setenv("FAKE_MODE", "nomusic")
    with pytest.raises(server.StartError, match="did not load the music models") as caught:
        _run(tmp_path)
    assert caught.value.crashed is False
    assert _wait_dead(started[0].pid)


def test_a_slow_load_times_out_as_a_start_error(fake_kcpp, tmp_path, monkeypatch, started):
    monkeypatch.setenv("FAKE_MODE", "slowload")
    monkeypatch.setattr(server, "START_TIMEOUT", 2.0)
    with pytest.raises(server.StartError, match="did not finish loading in 2 s"):
        _run(tmp_path)
    assert _wait_dead(started[0].pid)


class _Squatter(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = b'{"result": "KoboldCpp", "music": true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_a_port_answered_by_another_program_is_refused(fake_kcpp, tmp_path, monkeypatch,
                                                      started):
    pytest.importorskip("psutil")
    squatter = ThreadingHTTPServer(("127.0.0.1", 0), _Squatter)
    threading.Thread(target=squatter.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(server, "_free_port", lambda: squatter.server_address[1])
        monkeypatch.setenv("FAKE_MODE", "nobind")
        with pytest.raises(server.StartError, match="another program is listening"):
            _run(tmp_path, prepare=True)
        assert not [e for e in _events(fake_kcpp) if e["event"] == "post"]
        assert _wait_dead(started[0].pid)
    finally:
        squatter.shutdown()


def test_the_server_dying_mid_request_is_reported_reaped_and_forgotten(
        fake_kcpp, tmp_path, monkeypatch, started):
    monkeypatch.setenv("FAKE_MODE", "dieongen")
    kills = []
    real_kill = _proc.kill
    monkeypatch.setattr(_proc, "kill", lambda mp, **k: kills.append(mp.pid) or real_kill(mp, **k))
    with pytest.raises(server.ServerError, match="stopped during the request") as caught:
        _run(tmp_path)
    assert not isinstance(caught.value, server.StartError)
    assert kills == [started[0].pid]
    assert server.running_pid() is None


def test_an_idle_server_is_stopped(fake_kcpp, tmp_path, monkeypatch, started):
    monkeypatch.setattr(server, "IDLE_SECONDS", 1.5)
    _run(tmp_path)
    deadline = time.monotonic() + 15
    while server.running_pid() is not None and time.monotonic() < deadline:
        time.sleep(0.2)
    assert server.running_pid() is None
    assert _wait_dead(started[0].pid)


def test_proxy_settings_do_not_divert_loopback_requests(fake_kcpp, tmp_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert _run(tmp_path).startswith(b"RIFF")


# --------------------------------------------------------------------------- #
#  The process outlives nothing: localm going away takes it down               #
# --------------------------------------------------------------------------- #

PARENT = textwrap.dedent(r'''
    import json, sys, time
    sys.path.insert(0, sys.argv[1])
    from localm.media.koboldcpp import _proc
    mp = _proc.start([sys.argv[2], "-c", "import time; time.sleep(120)"], cwd=".",
                     env=None)
    print(json.dumps({"child": mp.pid,
                      "watcher": getattr(mp._watcher, "pid", None)}), flush=True)
    time.sleep(120)
''')


def _start_parent(tmp_path, **popen_kw):
    script = tmp_path / "parent.py"
    script.write_text(PARENT, encoding="utf-8")
    root = str(Path(__file__).resolve().parent.parent)
    parent = subprocess.Popen([BASE_PYTHON, str(script), root, BASE_PYTHON],
                              stdout=subprocess.PIPE, text=True, **popen_kw)
    assert parent.stdout is not None
    info = json.loads(parent.stdout.readline())
    return parent, info


@pytest.mark.skipif(sys.platform != "win32", reason="kill-on-close job objects are Windows")
def test_the_job_object_kills_the_server_when_localm_is_killed(tmp_path):
    parent, info = _start_parent(tmp_path)
    try:
        assert _pid_alive(info["child"])
        parent.kill()
        parent.wait(10)
        assert _wait_dead(info["child"])
    finally:
        import psutil
        try:
            psutil.Process(info["child"]).kill()
        except psutil.NoSuchProcess:
            pass


@pytest.mark.skipif(sys.platform == "win32", reason="the exit watcher is POSIX")
def test_the_watcher_outlives_localm_and_kills_the_group(tmp_path):
    import signal
    parent, info = _start_parent(tmp_path, start_new_session=True)
    try:
        assert _pid_alive(info["child"]) and info["watcher"]
        os.killpg(parent.pid, signal.SIGINT)
        os.killpg(parent.pid, signal.SIGHUP)
        parent.wait(10)
        assert _wait_dead(info["child"])
        assert _wait_dead(info["watcher"])
    finally:
        for pid in (info["child"], info["watcher"]):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
#  WAV output                                                                  #
# --------------------------------------------------------------------------- #

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


def test_write_wav_removes_the_partial_file_when_saving_fails(tmp_path, monkeypatch):
    def refuse(self, target):
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "replace", refuse)
    out = tmp_path / "track.wav"
    with pytest.raises(PermissionError):
        music.write_wav(_signal(), out)
    assert list(tmp_path.iterdir()) == []


def _signal(seconds: float = 1.0, *, rate: int = 48000, width: int = 2, edit=None) -> bytes:
    n = int(rate * seconds)
    samples = array.array("h", [((i * 37) % 2001) - 1000 for i in range(2 * n)])
    if edit is not None:
        edit(samples, rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes() if width == 2 else bytes(n * 2))
    return buf.getvalue()


def _pin(level: int, seconds: float, channel: int = 0):
    def edit(samples, rate):
        for i in range(int(rate * seconds)):
            samples[2 * (rate // 10 + i) + channel] = level
    return edit


def _constant(level: int):
    def edit(samples, rate):
        for i in range(len(samples)):
            samples[i] = level
    return edit


@pytest.mark.parametrize("data,reason", [
    (_signal(), None),
    (_signal(edit=_pin(32767, 0.1)), None),
    (_signal(edit=_pin(32767, 0.3)), "stuck at full scale"),
    (_signal(edit=_pin(-32768, 0.3, channel=1)), "stuck at full scale"),
    (_signal(edit=_constant(32767)), "stuck at full scale"),
    (_signal(edit=_constant(1200)), "constant level"),
    (_signal(edit=_constant(0)), "constant level"),
    (_signal(width=1), None),
    (b"not a wav", None),
], ids=["music", "short-clip", "pinned-left", "pinned-negative-right", "flat-full-scale",
        "flat-level", "silence", "8-bit", "unreadable"])
def test_broken_reason(data, reason):
    got = music.broken_reason(data)
    if reason is None:
        assert got is None
    else:
        assert got is not None and reason in got


@pytest.mark.parametrize("data", [b"", b"not a wav at all", b"RIFF\x00\x00\x00\x00WAVE"])
def test_write_wav_refuses_unreadable_audio(tmp_path, data):
    out = tmp_path / "track.wav"
    with pytest.raises(server.ServerError):
        music.write_wav(data, out)
    assert not out.exists()


# --------------------------------------------------------------------------- #
#  Backend order and what auto remembers                                       #
# --------------------------------------------------------------------------- #

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


@pytest.fixture
def two_backends(fake_kcpp, tmp_path, monkeypatch):
    """auto = cuda then vulkan; the mode the fake runs in is chosen per backend."""
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    monkeypatch.setattr(runtime, "recommended_backend", lambda det=None: "cuda")
    monkeypatch.setattr(runtime, "ensure_for_backend", lambda backend, **k: _rt(tmp_path))
    monkeypatch.setattr(music, "resolve_models", lambda *a, **k: MODELS)
    monkeypatch.setattr(music, "work_dir", lambda: tmp_path / "w")
    modes = {}
    orig_start = server._Server.start

    def start(self, on_progress, cancel_check):
        os.environ["FAKE_MODE"] = modes.get(self.key.backend, "ok")
        try:
            return orig_start(self, on_progress, cancel_check)
        finally:
            os.environ["FAKE_MODE"] = "ok"

    monkeypatch.setattr(server._Server, "start", start)
    monkeypatch.setenv("FAKE_MODE", "ok")
    return modes


def test_auto_falls_back_and_remembers_a_crash(two_backends):
    two_backends["cuda"] = "crash"
    lines = []
    data, used = music.generate_wav({}, "auto", REQUEST, plan=False, on_progress=lines.append)
    assert used == "vulkan" and data.startswith(b"RIFF")
    assert runtime.backend_failed("cuda") is not None
    assert runtime.backend_worked("vulkan")
    assert any("cuda backend did not start" in ln and "trying vulkan" in ln for ln in lines)


def test_auto_generates_a_broken_track_again_on_the_next_backend(two_backends):
    two_backends["cuda"] = "flat"
    lines = []
    data, used = music.generate_wav({}, "auto", REQUEST, plan=False, on_progress=lines.append)
    assert used == "vulkan" and music.broken_reason(data) is None
    assert any("cuda backend returned a broken track (the audio is stuck at full scale)" in ln
               and "generating it again on vulkan" in ln for ln in lines)
    assert runtime.backend_failed("cuda") is None and not runtime.backend_worked("cuda")


def test_an_explicit_backend_returning_a_broken_track_fails_with_the_cpu_hint(two_backends):
    two_backends["vulkan"] = "flat"
    with pytest.raises(music.NativeMusicError) as exc:
        music.generate_wav({}, "vulkan", REQUEST, plan=False)
    assert "vulkan: the track came back broken (the audio is stuck at full scale)" in str(exc.value)
    assert "set the native runtime to cpu" in str(exc.value)
    assert runtime.backend_failed("vulkan") is None


def test_auto_does_not_remember_running_out_of_memory(two_backends):
    two_backends["cuda"] = "oomcrash"
    _data, used = music.generate_wav({}, "auto", REQUEST, plan=False)
    assert used == "vulkan"
    assert runtime.backend_failed("cuda") is None


def test_a_later_success_clears_a_remembered_crash(two_backends):
    runtime.record_backend("cuda", False, "exit code 3")
    server.stop()
    _data, used = music.generate_wav({}, "cuda", REQUEST, plan=False)
    assert used == "cuda"
    assert runtime.backend_failed("cuda") is None and runtime.backend_worked("cuda")


def test_an_explicit_backend_that_does_not_start_is_not_swapped(two_backends, fake_kcpp,
                                                               started):
    two_backends["vulkan"] = "crash"
    with pytest.raises(music.NativeMusicError, match="vulkan: the music runtime exited"):
        music.generate_wav({}, "vulkan", REQUEST, plan=False)
    starts = [e for e in _events(fake_kcpp) if e["event"] == "start"]
    assert len(starts) == 1 and "--usevulkan" in starts[0]["argv"]
    assert runtime.backend_failed("vulkan") is None
