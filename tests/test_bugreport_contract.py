# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract for the bug report as a shareable artefact.

Pins, byte for byte against the golden files in
``tests/fixtures/bugreport_contract/``, every artefact the reporter produces:
the rendered markdown, the saved file, the upload request, the mailto link,
the console transcript of the consent flow, and the reports filed for a prior
crash and for a native fault during a clean stop. The fixtures carry fake
credentials (URL user:pass, query, header, JSON and .env forms, bearer and API
keys), absolute user paths in Windows and POSIX shapes, email addresses and
host names.

Also pinned: nothing is uploaded or opened without an explicit choice on the
consent path, automatic reports are only saved, and a scrubber that raises
yields no report and no upload.

Environment probes, the clock, the network and faulthandler are replaced on
the modules that own them. The only names replaced on ``localm.bugreport``
are ``_scrub_secrets`` and ``_handlers_installed``.

A golden that is missing or differs fails the test and writes the actual
output next to the test's ``tmp_path``.
"""

from __future__ import annotations

import email.message
import importlib.metadata
import io
import json
import os
import platform
import socket
import ssl
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from localm import bugreport

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "bugreport_contract"

PW = "Arch08UrlPass"
QV = "Arch08QueryValue"
HV = "Arch08HeaderValue"
JV = "Arch08JsonValue"
EV = "Arch08EnvValue"
BEARER = "Arch08BearerToken.abc-123"
OPENAI_KEY = "sk-" + "Arch08OpenAiKey0123456789"
LOCALM_KEY = "localm-sk-" + "Arch08LocalmKey0123"
CFG_HF = "hf_Arch08ConfigOnlyToken"
CFG_CIVITAI = "Arch08CivitaiKey"
UPLOAD_TOKEN = "Arch08UploadSharedSecret"
CHAT = "Arch08PrivateChatContent"

SECRETS = (PW, QV, HV, JV, EV, BEARER, OPENAI_KEY, LOCALM_KEY, CFG_HF,
           CFG_CIVITAI, UPLOAD_TOKEN, CHAT)

WIN_PATH = r"C:\Users\alice\Documents\localm\notes.txt"
WIN_JSON_PATH = r"C:\\Users\\alice\\AppData\\Roaming\\localm\\state.json"
POSIX_PATH = "/home/bob/.cache/localm/model.gguf"
MAC_PATH = "/Users/carol/Library/Logs/localm.log"
FOREIGN_PATH_HEADS = (r"C:\Users\alice", r"C:\\Users\\alice", "/home/bob/",
                      "/Users/carol/")
OTHER_EMAIL = "bob.builder@example.com"
HOST = "alice-desktop.local"
UNC = r"\\FILESERVER01\share\report.txt"

PROXY_URL = "https://bugreport-proxy.example.invalid/report"
CRASH_PID = 999_001
STOP_PID = 999_002
LIVE_PID = 999_003


def _home() -> str:
    return str(Path.home())


def _own_path() -> str:
    return f"{_home()}/models/private-model.gguf"


def _config(with_upload: bool = True) -> dict:
    cfg = {
        "plugins_enabled": ["chat", "gui", "coder"],
        "n_ctx": 8192,
        "n_gpu_layers": 99,
        "spec_source": "ngram",
        "port": 8765,
        "require_auth": True,
        "mode": "log",
        "binary_dir": r"C:\Users\alice\localm\bin",
        "comfy_workdir": f"{_home()}/ComfyUI",
        "comfy_api_url": f"http://admin:{PW}@{HOST}:8188/?api_key={QV}",
        "net_search_url": f"https://search.example.org/search?q=x&token={QV}",
        "coder_reviewer": f"http://reviewer.example.net/v1?key={QV}&model=m",
        "hf_token": CFG_HF,
        "civitai_api_key": CFG_CIVITAI,
    }
    if with_upload:
        cfg["bugreport_upload_url"] = PROXY_URL
        cfg["bugreport_upload_token"] = UPLOAD_TOKEN
    return cfg


class _Resp:
    def __init__(self, status: int, body: str):
        self.status = status
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    """Stands in for ``localm.http_ssl.verified_urlopen``: records each request
    and answers from ``responses`` (a status/body pair or an exception), else
    201 with an issue URL."""

    def __init__(self):
        self.requests: list = []
        self.responses: list = []

    def __call__(self, req, *, timeout=None, **kw):
        self.requests.append({
            "url": req.full_url,
            "method": req.get_method(),
            "headers": dict(req.header_items()),
            "data": req.data,
            "timeout": timeout,
        })
        nxt = self.responses.pop(0) if self.responses else (
            201, '{"url": "https://example.invalid/issues/42"}')
        if isinstance(nxt, BaseException):
            raise nxt
        return _Resp(*nxt)


def _http_error(code: int, body: str = "", retry_after=None):
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(PROXY_URL, code, "error", hdrs,
                                  io.BytesIO(body.encode("utf-8")))


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A fully deterministic machine: fixed platform, GPU, runtime, engine,
    config, activity ring, dependency versions, clock and transport."""
    from localm import audit, config, debuglog, discover, hwdetect, http_ssl
    from localm import _version as version_mod
    from localm import instances
    from localm import setup_llama
    from localm.inference import http_server
    from localm.inference.backends.llamacpp import _loader

    home = config.home_dir()
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    state = SimpleNamespace(
        home=home, logs=logs, cfg=_config(), transport=_Transport(),
        net_calls=[], sleeps=[], live_pids=set(), diagnostics=True,
        faulthandler=[], activity=[
            f"12:00:00 INFO localm: loaded {_own_path()}",
            f"12:00:01 WARNING localm: GET http://u:{PW}@{HOST}/?sig={QV} failed",
            f"12:00:02 INFO localm: reviewer {OTHER_EMAIL} at {UNC}",
            f"12:00:03 ERROR localm: header X-Api-Key: {HV} key {OPENAI_KEY}",
        ])

    monkeypatch.setattr(platform, "platform", lambda *a, **k: "TestOS-1.0-contract")
    monkeypatch.setattr(platform, "machine", lambda: "testarch")
    monkeypatch.setattr(version_mod, "read_version", lambda *a, **k: "0.0.0+contract")

    monkeypatch.setattr(hwdetect, "detect", lambda *a, **k: SimpleNamespace(
        vendors=["nvidia"], source="contract-probe", recommended="vulkan"))
    monkeypatch.setattr(hwdetect, "recommended_install_backend",
                        lambda det=None: "cuda")
    monkeypatch.setattr(setup_llama, "nvidia_preflight", lambda: SimpleNamespace(
        present=True, gpu_name="Contract GPU 9000", driver_version="999.99",
        cuda_capability="12.8", compute_capability="8.9", cuda_line="cu12",
        gpus=[{"index": 0, "name": "Contract GPU 9000",
               "free_mib": 8192, "total_mib": 16384}]))
    monkeypatch.setattr(setup_llama, "installed_build", lambda: "b1234")
    monkeypatch.setattr(setup_llama, "installed_backend", lambda: "cuda")
    monkeypatch.setattr(setup_llama, "pinned_tag", lambda: "b1234")
    gb = 1024 ** 3
    monkeypatch.setattr(discover, "last_gpu_reading", lambda: [
        {"index": 0, "name": "Contract GPU 9000", "free": 8 * gb,
         "total": 16 * gb, "source": "nvidia-smi"},
        {"index": 1, "name": None, "free": None, "total": 4 * gb}])

    rt = tmp_path / "runtime"
    rt.mkdir()
    for name in ("ggml-cuda.dll", "llama.dll", "libmtmd.so.1", "llama-server.exe",
                 "README.txt"):
        (rt / name).write_text("x", encoding="utf-8")
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: rt)
    monkeypatch.setattr(_loader, "lib_filename", lambda: "llama.dll")

    class ContractBackend:
        pass

    engine = SimpleNamespace(display_name="contract-model-Q4_K_M", loaded=True,
                             _backend=ContractBackend(), effective_ctx_max=8192)
    monkeypatch.setattr(http_server, "_engine", engine, raising=False)

    monkeypatch.setattr(audit, "effective_mode",
                        lambda surface, cwd=None: SimpleNamespace(value="log"))
    monkeypatch.setattr(audit, "diagnostics_allowed", lambda: state.diagnostics)
    monkeypatch.setattr(debuglog, "debug_enabled", lambda: False)
    monkeypatch.setattr(debuglog, "recent_activity", lambda: list(state.activity))
    monkeypatch.setattr(debuglog, "logs_dir", lambda: logs)
    monkeypatch.setattr(config, "load_config", lambda *a, **k: dict(state.cfg))
    monkeypatch.setattr(config, "load_config_checked",
                        lambda *a, **k: (dict(state.cfg), True))
    monkeypatch.setattr(instances, "pid_alive", lambda pid: pid in state.live_pids)

    deps = {"localm": "0.0.0+contract", "fastapi": "0.0.1", "uvicorn": "0.0.2"}

    def _version(dist):
        if dist in deps:
            return deps[dist]
        raise importlib.metadata.PackageNotFoundError(dist)

    monkeypatch.setattr(importlib.metadata, "version", _version)

    real_strftime = time.strftime
    monkeypatch.setattr(time, "strftime", lambda fmt, *a: (
        "20261009-120000" if fmt == "%Y%m%d-%H%M%S" else real_strftime(fmt, *a)))
    monkeypatch.setattr(time, "sleep", lambda s: state.sleeps.append(s))

    monkeypatch.setattr(http_ssl, "verified_urlopen", state.transport)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: state.net_calls.append(("urlopen", a)))
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: state.net_calls.append(("connect", a)))

    import faulthandler
    monkeypatch.setattr(faulthandler, "enable",
                        lambda **k: state.faulthandler.append(("enable", k)))
    monkeypatch.setattr(faulthandler, "disable",
                        lambda: state.faulthandler.append(("disable",)))
    monkeypatch.setattr(faulthandler, "is_enabled", lambda: True)

    monkeypatch.setattr(bugreport.console, "_width", 10_000)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    yield state
    assert state.net_calls == [], "a report path reached the real network"


def _expected(name: str) -> str:
    path = GOLDEN_DIR / name
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    return (text.replace("{{MAINTAINER_EMAIL}}", bugreport.MAINTAINER_EMAIL)
                .replace("{{PYTHON}}", sys.version.split()[0]))


def _template(actual: str) -> str:
    return (actual.replace(bugreport.MAINTAINER_EMAIL, "{{MAINTAINER_EMAIL}}")
                  .replace(sys.version.split()[0], "{{PYTHON}}"))


def _assert_golden(name: str, actual: str, tmp_path: Path) -> None:
    path = GOLDEN_DIR / name
    if path.exists() and _expected(name) == actual:
        return
    dump = tmp_path / f"ACTUAL-{name}"
    dump.write_bytes(_template(actual).encode("utf-8"))
    assert path.exists(), f"golden {name} is missing; actual output: {dump}"
    assert actual == _expected(name), f"golden {name} differs; actual output: {dump}"


def _assert_file_bytes(path: Path, golden: str, tmp_path: Path) -> str:
    text = path.read_bytes().decode("utf-8")
    _assert_golden(golden, text.replace(os.linesep, "\n"), tmp_path)
    assert path.read_bytes() == _expected(golden).replace("\n", os.linesep).encode("utf-8")
    return text.replace(os.linesep, "\n")


def _assert_scrubbed(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text, secret
    for head in FOREIGN_PATH_HEADS:
        assert head not in text, head
    assert _home() not in text


def _automatic_context() -> dict:
    return {
        "operation": "chat",
        "backend": "cuda",
        "requested_backend": "auto",
        "with_cudart": True,
        "native_trace": (
            "Windows fatal exception: access violation\n\n"
            "Current thread 0x00001a2b (most recent call first):\n"
            f'  File "{WIN_PATH}", line 12 in load\n'
            f'  File "{MAC_PATH}", line 3 in <module>\n'
            f"env OPENAI_API_KEY={EV}\n"),
        "prior_exit": {"exit_code": 3221225477, "watched_for_s": 12.5},
        "recent_log_tail": (
            "2026-10-09 12:00:00,000 WARNING  localm: upstream "
            f"{UNC} rejected X-Api-Key: {HV}\n"
            f'payload {{"api_key": "{JV}", "has_token": false}}\n'
            f'state {{"path": "{WIN_JSON_PATH}"}}\n'
            f"SECRET_KEY={EV} key=visible monkey=visible"),
        "hang_traces": (
            "Thread 0x0001 (most recent call first):\n"
            f'  File "{_own_path()}", line 5 in wait\n'
            f"  token={QV}\n"),
        "client": {
            "userAgent": "Mozilla/5.0 (contract)",
            "page": f"http://{HOST}:8765/#/chat?api_key={QV}",
            "viewport": "1280x720",
            "appVersion": "contract-build",
            "console": [
                f'fetch failed {{"api_key":"{JV}","has_token":false}}',
                f"Error: {LOCALM_KEY} at {MAC_PATH}",
                f"user {OTHER_EMAIL} SECRET_KEY={EV} Authorization: Basic {HV}",
            ],
        },
    }


def _automatic_error() -> RuntimeError:
    return RuntimeError(
        f"POST http://admin:{PW}@{HOST}:8188/prompt?token={QV}&mode=fast "
        f"failed: Authorization: Bearer {BEARER} key {OPENAI_KEY} at {POSIX_PATH}")


def _write_run_log(logs: Path, pid: int, *, truncated: bool = False) -> Path:
    lines = [
        f"2026-10-09 11:59:00,001 INFO     localm: server start on {HOST}:8765",
        "2026-10-09 11:59:01,002 DEBUG    localm.llama: raw model output:",
        f"this is {CHAT}",
        "2026-10-09 11:59:01,100 INFO     localm.http: GET /api/stats 200 1ms",
        "2026-10-09 11:59:01,200 INFO     localm.http: GET /api/stats 200 2ms",
        "2026-10-09 11:59:01,300 INFO     localm.http: GET /api/stats 200 3ms",
        f"2026-10-09 11:59:02,003 WARNING  localm.net: GET http://admin:{PW}@{HOST}/"
        f"?api_key={QV} failed for {_own_path()}",
        "2026-10-09 11:59:03,004 ERROR    localm.engine: load failed",
        "Traceback (most recent call last):",
        f'  File "{WIN_PATH}", line 7, in load',
        f"RuntimeError: Authorization: Bearer {BEARER} for {OTHER_EMAIL}",
    ]
    text = "\n".join(lines) + "\n"
    if truncated:
        text += "llama_context: constructing llama_co"
    path = logs / f"localm_2026-10-09_115900_{pid}.log"
    path.write_bytes(text.encode("utf-8"))
    return path


def _write_hang(logs: Path, pid: int, label: str = "0x0002") -> Path:
    path = logs / f"hang_2026-10-09_115930_{pid}.log"
    path.write_bytes((
        f"Thread {label} (most recent call first):\n"
        f'  File "{WIN_PATH}", line 40 in _run_once\n'
        f"  Authorization: Bearer {BEARER}\n").encode())
    old = time.time() - 60
    os.utime(path, (old, old))
    return path


def _saved_reports(world) -> list:
    d = world.home / "bug-reports"
    return sorted(d.glob("*.md")) if d.is_dir() else []


def _transcript(capsys, path=None) -> str:
    out = capsys.readouterr().out
    if path is not None:
        out = out.replace(str(path), "<REPORT>")
    return out


# --------------------------------------------------------------------------- #
#  Rendered and saved reports                                                 #
# --------------------------------------------------------------------------- #

def test_automatic_report_is_byte_identical(world, tmp_path):
    (world.logs / "pre_restart.log").write_bytes(
        f"11:59:59 INFO localm: before restart token={QV}\n".encode())
    text = bugreport.build_report(
        f"model load failed for {_own_path()} on {HOST} token={QV}",
        reason=f"backend said api_key={QV} reading {WIN_PATH}; ask {OTHER_EMAIL}",
        error=_automatic_error(), context=_automatic_context())
    _assert_golden("automatic_report.md", text, tmp_path)
    _assert_scrubbed(text)
    assert not (world.logs / "pre_restart.log").exists()


def test_user_report_saved_file_is_byte_identical(world, tmp_path):
    (world.logs / "pre_restart.log").write_bytes(
        f"11:59:59 INFO localm: before restart Bearer {BEARER}\n".encode())
    _write_hang(world.logs, os.getpid())
    _write_run_log(world.logs, os.getpid())
    decoy = world.logs / "localm_2026-10-09_120500_1.log"
    decoy.write_bytes(b"2026-10-09 12:05:00,000 ERROR    localm: DECOY from another run\n")
    later = time.time() + 60
    os.utime(decoy, (later, later))
    path = bugreport.save_user_report(
        f"I opened {WIN_PATH} with token={QV}",
        what_i_expected=f"It should load {POSIX_PATH}",
        what_happened=(f"Crashed posting to http://admin:{PW}@{HOST}/ with "
                       f"Bearer {BEARER}\nsecond line"),
        include_log=True,
        client={"userAgent": "Mozilla/5.0 (contract)",
                "console": [f"Error: Bearer {BEARER}", f'{{"password":"{JV}"}}']},
        extra_hang_trace=f'Thread 0x0003\n  File "{MAC_PATH}", line 1 in run\n')
    assert path == world.home / "bug-reports" / "bug-20261009-120000.md"
    text = _assert_file_bytes(path, "user_report.md", tmp_path)
    _assert_scrubbed(text)
    assert not (world.logs / "pre_restart.log").exists()
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_user_report_with_an_explicit_summary_and_no_log(world, tmp_path):
    path = bugreport.save_user_report(
        "", summary=f"  custom title {OPENAI_KEY}  ",
        what_happened="it broke", include_log=True)
    text = path.read_bytes().decode("utf-8").replace(os.linesep, "\n")
    _assert_golden("user_report_no_log.md", text, tmp_path)
    _assert_scrubbed(text)


def test_an_unreadable_log_is_reported_without_its_path(world, monkeypatch):
    from localm import _log_digest
    _write_run_log(world.logs, os.getpid())

    def _boom(*a, **k):
        raise OSError(13, "Permission denied", WIN_PATH)

    monkeypatch.setattr(_log_digest, "build_digest", _boom)
    path = bugreport.save_user_report("doing a thing", include_log=True)
    text = path.read_text(encoding="utf-8")
    assert ("## Recent log (tail)\n(not collected: the log file could not be read "
            "(PermissionError: Permission denied))\n") in text
    _assert_scrubbed(text)


def test_log_digest_is_scrubbed_at_its_source(world):
    _write_run_log(world.logs, os.getpid())
    digest, reason = bugreport._recent_log_tail_result(pid=os.getpid())
    assert reason == ""
    assert "localm.engine: load failed" in digest
    _assert_scrubbed(digest)
    assert bugreport._recent_log_tail(pid=os.getpid()) == digest


def test_a_missing_log_says_so_and_an_opt_out_says_nothing(world):
    asked = bugreport.save_user_report("doing a thing", include_log=True)
    assert "(not collected: no log file was found for that run)" in \
        asked.read_text(encoding="utf-8")
    asked.unlink()
    opted_out = bugreport.save_user_report("doing a thing", include_log=False)
    assert "## Recent log (tail)" not in opted_out.read_text(encoding="utf-8")


def test_save_report_fallback_location(world, monkeypatch):
    import localm
    from localm import config

    def _no_home():
        raise RuntimeError("no config")

    seen = []

    def _mkdir(self, *a, **k):
        seen.append(self)
        raise OSError("refused")

    monkeypatch.setattr(config, "home_dir", _no_home)
    monkeypatch.setattr(Path, "mkdir", _mkdir)
    assert bugreport.save_report("text", when="x") is None
    assert seen == [Path(localm.__file__).resolve().parents[1] / "home" / "bug-reports"]


def test_exception_types_keep_their_public_name():
    import importlib
    for cls in (bugreport.LocalmError, bugreport.RateLimitedError):
        assert cls.__module__ == "localm.bugreport"
        assert getattr(importlib.import_module(cls.__module__), cls.__qualname__) is cls
    assert bugreport._format_error(bugreport.LocalmError("setup failed")) == (
        "localm.bugreport.LocalmError: setup failed")
    assert bugreport._format_error(bugreport.RateLimitedError(7)) == (
        "localm.bugreport.RateLimitedError: the bug-report server is rate limiting reports")


def test_report_title_and_mailto(world, tmp_path):
    assert bugreport.report_title("", "", "") == "user-reported issue"
    assert bugreport.report_title("", "first\nsecond", "desc") == "first"
    assert bugreport.report_title("", "", "desc line\nmore") == "desc line"
    assert bugreport.report_title("  given  ", "x", "y") == "given"
    assert bugreport.report_title("", "z" * 300, "") == "z" * 120
    body = "\n".join(f"line {i} {OTHER_EMAIL}" for i in range(200))
    _assert_golden("mailto.txt", bugreport.mailto_url("crash on startup", body) + "\n",
                   tmp_path)


# --------------------------------------------------------------------------- #
#  Upload preparation and consent                                             #
# --------------------------------------------------------------------------- #

def _request_text(req: dict) -> str:
    lines = [f"{req['method']} {req['url']}", f"timeout: {req['timeout']}"]
    lines += [f"{k}: {v}" for k, v in sorted(req["headers"].items())]
    return "\n".join(lines) + "\n\n" + req["data"].decode("utf-8") + "\n"


def test_upload_request_is_byte_identical(world, tmp_path):
    footer = bugreport._report_footer()
    res = bugreport.upload_report(
        f"Crash in {WIN_PATH} token={QV} {OPENAI_KEY} on {HOST}",
        f"# report\n\nalready scrubbed body\n\n---\n{footer}\n")
    assert res == {"url": "https://example.invalid/issues/42"}
    res = bugreport.upload_report(
        "", f"body without newline\n\n---\n{footer}", url="https://other.example.invalid/r",
        token=None, timeout=3.0)
    res = bugreport.upload_report(
        "t", f"edited footer\n\n---\n{footer} (edited)\n",
        url="https://other.example.invalid/r", token="explicit-token")
    text = "".join(_request_text(r) for r in world.transport.requests)
    _assert_golden("upload_requests.txt", text, tmp_path)
    assert all(QV not in r["data"].decode() and OPENAI_KEY not in r["data"].decode()
               for r in world.transport.requests)


def test_upload_without_an_endpoint_refuses_and_sends_nothing(world):
    world.cfg = _config(with_upload=False)
    with pytest.raises(bugreport.LocalmError) as ei:
        bugreport.upload_report("t", "b")
    assert (ei.value.summary, ei.value.stage) == (
        "no upload endpoint is configured", "no_endpoint")
    assert world.transport.requests == []
    assert bugreport.upload_available() is False
    world.cfg = _config()
    assert bugreport.upload_available() is True


@pytest.mark.parametrize("exc, stage", [
    (urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed")), "offline_or_dns"),
    (urllib.error.URLError(ssl.SSLError("bad cert")), "tls"),
    (urllib.error.URLError(TimeoutError("timed out")), "timeout"),
    (ConnectionRefusedError(10061, "refused"), "unreachable"),
    (OSError("Network is unreachable"), "offline_or_dns"),
    (OSError("something else"), "unreachable"),
])
def test_upload_failures_are_diagnosed(world, exc, stage):
    world.transport.responses = [exc]
    with pytest.raises(bugreport.LocalmError) as ei:
        bugreport.upload_report("t", "b")
    assert ei.value.stage == stage
    assert ei.value.summary == "could not reach the bug-report server"
    assert ei.value.hint


def test_upload_rejection_and_rate_limit(world):
    world.transport.responses = [_http_error(503, "down"), _http_error(429, "", 7),
                                 _http_error(429, '{"retry_after": 9}'), (500, "nope"),
                                 (429, '{"retry_after": 4}'), (200, "not json")]
    with pytest.raises(bugreport.LocalmError) as ei:
        bugreport.upload_report("t", "b")
    assert (ei.value.stage, ei.value.reason) == ("server_rejected", "HTTP 503: down")
    for want in (7, 9):
        with pytest.raises(bugreport.RateLimitedError) as ei:
            bugreport.upload_report("t", "b")
        assert ei.value.retry_after == want
    with pytest.raises(bugreport.LocalmError) as ei:
        bugreport.upload_report("t", "b")
    assert (ei.value.stage, ei.value.reason) == ("server_rejected", "HTTP 500: nope")
    with pytest.raises(bugreport.RateLimitedError) as ei:
        bugreport.upload_report("t", "b")
    assert ei.value.retry_after == 4
    assert bugreport.upload_report("t", "b") == {"raw": "not json"}


def _offer(path, text, answers, **kw):
    asked, opened = [], []
    it = iter(answers)

    def prompt(question):
        asked.append(question)
        return next(it, "")

    bugreport.offer_to_send("contract summary", path, text, prompt=prompt,
                            open_browser=opened.append, **kw)
    return asked, opened


def test_consent_flow_transcripts(world, capsys, tmp_path):
    path = bugreport.save_user_report("doing a thing", what_happened="it broke")
    text = path.read_text(encoding="utf-8")
    capsys.readouterr()
    footer_free = bugreport._strip_report_footer(text)
    out = []

    def case(name, answers, responses=(), cfg=None, edit=None, offer_path=path, **kw):
        world.cfg = cfg if cfg is not None else _config()
        world.transport.requests.clear()
        world.transport.responses = list(responses)
        world.sleeps.clear()
        if edit is not None:
            path.write_text(edit, encoding="utf-8")
        asked, opened = _offer(offer_path, text, answers, **kw)
        bodies = [json.loads(r["data"])["body"] for r in world.transport.requests]
        out.append(f"=== {name}\n"
                   f"asked: {asked}\n"
                   f"opened: {len(opened)}\n"
                   f"requests: {len(bodies)}\n"
                   f"bodies match the saved file: "
                   f"{[b == footer_free for b in bodies]}\n"
                   f"slept: {world.sleeps}\n"
                   + _transcript(capsys, path))
        if edit is not None:
            path.write_text(text, encoding="utf-8")
        return asked, opened, bodies

    case("no endpoint, pick 1", ["1"], cfg=_config(with_upload=False))
    case("not now", [""])
    case("self", ["3"])
    _, opened, _ = case("email", ["2"])
    assert opened == [bugreport.mailto_url("contract summary", text)]
    case("upload", ["1"])
    case("upload after a rejection, retry yes", ["1", "y"],
         responses=[_http_error(503, "down")])
    case("upload rejected, retry no", ["1", "n"], responses=[_http_error(503, "down")])
    case("upload rejected three times", ["1", "y", "y"],
         responses=[_http_error(503, "a"), _http_error(503, "b"), _http_error(503, "c")])
    case("rate limited then sent", ["1"], responses=[_http_error(429, "", 7)])
    case("rate limited twice", ["1"],
         responses=[_http_error(429, "", 7), _http_error(429, "", 8)])
    case("non-interactive", ["1"], interactive=False)
    case("non-interactive without endpoint", ["1"], interactive=False,
         cfg=_config(with_upload=False))
    case("assume yes", ["1"], assume_yes=True)
    case("auto send without endpoint", [], auto_send=True, cfg=_config(with_upload=False))
    _, _, bodies = case("auto send", [], auto_send=True)
    assert bodies == [footer_free]
    case("auto send rate limited then sent", [], auto_send=True,
         responses=[_http_error(429, "", 5)])
    case("auto send rate limited then refused", [], auto_send=True,
         responses=[_http_error(429, "", 5), _http_error(503, "x")])
    case("auto send refused", [], auto_send=True, responses=[_http_error(503, "x")])
    _, _, bodies = case("edited file is what is sent", ["1"],
                        edit=text.replace("it broke", "it broke (edited)"))
    assert bodies[0] == footer_free.replace("it broke", "it broke (edited)")
    _, _, bodies = case("save failed, upload from memory", ["1"], offer_path=None)
    assert bodies == [footer_free]
    case("save failed, self", ["3"], offer_path=None)
    _assert_golden("consent_transcripts.txt", "".join(out), tmp_path)


def test_report_failure_never_sends_unless_asked(world, capsys, tmp_path):
    path = bugreport.report_failure(
        summary="setup failed", reason=f"see {WIN_PATH}", error=_automatic_error(),
        context={"operation": "setup-llama"}, interactive=False)
    assert world.transport.requests == []
    first = _transcript(capsys, path)
    path.unlink()
    path = bugreport.report_failure(summary="user filed", as_failure=False,
                                    interactive=False)
    second = _transcript(capsys, path)
    path.unlink()
    asked = []
    path = bugreport.report_failure(summary="asked", prompt=lambda q: asked.append(q) or "",
                                    open_browser=lambda u: None)
    third = _transcript(capsys, path)
    assert asked == ["  Pick a number"]
    assert world.transport.requests == []
    _assert_golden("report_failure_transcripts.txt",
                   "=== automatic\n" + first + "=== user\n" + second
                   + "=== interactive, not now\n" + third, tmp_path)


def test_background_crashes_are_saved_never_sent(world, capsys):
    exc = ValueError(f"worker failed with token={QV}")
    bugreport._handle_thread_exception(SimpleNamespace(
        exc_type=ValueError, exc_value=exc, exc_traceback=None,
        thread=SimpleNamespace(name="preload")))
    bugreport._handle_main_exception(RuntimeError, RuntimeError("main boom"), None)

    class _Loop:
        handler = None
        defaults: list = []

        def set_exception_handler(self, h):
            self.handler = h

        def default_exception_handler(self, ctx):
            self.defaults.append(ctx)

    loop = _Loop()
    assert bugreport.install_asyncio_handler(loop) is True
    loop.handler(loop, {"message": "Task exception was never retrieved",
                        "exception": KeyError("k")})
    loop.handler(loop, {"message": "only a message"})
    import asyncio
    loop.handler(loop, {"exception": asyncio.CancelledError()})
    assert loop.defaults == [{"message": "only a message"}]
    reports = _saved_reports(world)
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert text.startswith("# localm bug report: an async task crashed\n")
    assert world.transport.requests == []
    out = capsys.readouterr().out
    assert "Sorry - a background task crashed (thread 'preload')." in out
    assert "Sorry - localm hit an unexpected error." in out
    assert "Sorry - an async task crashed." in out
    assert "How would you like to send it" not in out


def test_global_handlers_install_once(world, monkeypatch):
    monkeypatch.setattr(bugreport, "_handlers_installed", False)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    assert bugreport.install_global_handlers() is False
    assert bugreport.install_global_handlers(force=True) is True
    assert sys.excepthook is bugreport._handle_main_exception
    assert threading.excepthook is bugreport._handle_thread_exception
    assert bugreport._handlers_installed is True
    assert bugreport.install_global_handlers(force=True) is False


# --------------------------------------------------------------------------- #
#  A scrubber that fails never yields a report                                #
# --------------------------------------------------------------------------- #

@pytest.fixture
def failing_scrub(monkeypatch):
    def _boom(text):
        raise RuntimeError("scrub exploded")

    monkeypatch.setattr(bugreport, "_scrub_secrets", _boom)


def test_failing_scrub_builds_saves_and_uploads_nothing(world, failing_scrub, capsys):
    with pytest.raises(RuntimeError, match="scrub exploded"):
        bugreport.build_report("summary")
    with pytest.raises(RuntimeError, match="scrub exploded"):
        bugreport.save_user_report("doing a thing", include_log=True)
    with pytest.raises(RuntimeError, match="scrub exploded"):
        bugreport.upload_report("title", "body")
    with pytest.raises(RuntimeError, match="scrub exploded"):
        bugreport.report_failure(summary="x", interactive=False)
    capsys.readouterr()
    _offer(None, "already built", ["1"])
    assert "Could not open that automatically" in capsys.readouterr().out
    bugreport._handle_thread_exception(SimpleNamespace(
        exc_type=ValueError, exc_value=ValueError("v"), exc_traceback=None,
        thread=SimpleNamespace(name="t")))
    assert _saved_reports(world) == []
    assert world.transport.requests == []


def test_failing_scrub_files_no_crash_report(world, failing_scrub, tmp_path):
    home = tmp_path / "crash-home"
    run = home / "run"
    run.mkdir(parents=True)
    marker = run / "server-crash.inst-a.marker"
    marker.write_text(json.dumps({"pid": CRASH_PID, "diagnostics": True}),
                      encoding="utf-8")
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert not marker.exists()
    assert _saved_reports(world) == []


# --------------------------------------------------------------------------- #
#  Crash guard and prior-crash reports                                        #
# --------------------------------------------------------------------------- #

def _crash_home(tmp_path: Path) -> Path:
    home = tmp_path / "crash-home"
    (home / "run").mkdir(parents=True)
    (home / "logs").mkdir()
    return home


def test_crash_guard_arms_and_disarms(world, tmp_path):
    home = _crash_home(tmp_path)
    run = home / "run"
    ctx = {"host": HOST, "port": 8765}
    assert bugreport.arm_crash_guard(context=ctx, home=home, instance_id="inst-x") is True
    assert json.loads((run / "server-crash.inst-x.marker").read_text(encoding="utf-8")) == {
        "pid": os.getpid(), "context": ctx, "diagnostics": True}
    assert (run / "server-crash-trace.inst-x.txt").exists()
    assert bugreport._crash_trace_fh is not None
    assert bugreport._crash_trace_instance_id == "inst-x"
    assert bugreport.armed_instance_id() == "inst-x"
    assert [c[0] for c in world.faulthandler] == ["enable"]

    bugreport.clear_crash_marker(home=home, instance_id="inst-x")
    assert sorted(p.name for p in run.iterdir()) == [
        "server-crash-trace.inst-x.txt", "server-crash.inst-x.stopping"]
    assert bugreport.armed_instance_id() is None
    bugreport.release_crash_trace(home=home, instance_id="other")
    assert bugreport._crash_trace_fh is not None
    bugreport.release_crash_trace(home=home, instance_id="inst-x")
    assert bugreport._crash_trace_fh is None
    assert bugreport._crash_trace_instance_id is None
    assert list(run.iterdir()) == []
    assert [c[0] for c in world.faulthandler] == ["enable", "disable"]

    world.diagnostics = False
    assert bugreport.arm_crash_guard(home=home, instance_id=None) is True
    assert json.loads((run / "server-crash.marker").read_text(encoding="utf-8")) == {
        "pid": os.getpid(), "context": {}, "diagnostics": False}
    assert not (run / "server-crash-trace.txt").exists()
    assert bugreport._crash_trace_fh is None
    bugreport.disarm_crash_guard(home=home, instance_id=None)
    assert list(run.iterdir()) == []


def _arm_dead_run(home: Path, *, diagnostics=True) -> None:
    run = home / "run"
    (run / "server-crash.inst-a.marker").write_text(json.dumps({
        "pid": CRASH_PID, "context": {"host": HOST, "cwd": _own_path()},
        "diagnostics": diagnostics}), encoding="utf-8")
    (run / "server-crash-trace.inst-a.txt").write_text(
        "Windows fatal exception: code 0x8001010d\n\n"
        "Windows fatal exception: access violation\n\n"
        "Current thread 0x0000beef (most recent call first):\n"
        f'  File "{WIN_PATH}", line 99 in decode\n'
        f"  X-Api-Key: {HV}\n", encoding="utf-8")
    (run / "server-crash-exit.inst-a.json").write_text(json.dumps(
        {"pid": CRASH_PID, "exit_code": 3221225477, "watched_for_s": 30}),
        encoding="utf-8")
    _write_hang(home / "logs", CRASH_PID)
    _write_run_log(home / "logs", CRASH_PID, truncated=True)


def test_prior_crash_report_is_byte_identical(world, capsys, tmp_path):
    home = _crash_home(tmp_path)
    _arm_dead_run(home)
    path = bugreport.check_and_report_prior_crash(home=home)
    assert path == world.home / "bug-reports" / "bug-20261009-120000.md"
    text = _assert_file_bytes(path, "prior_crash_report.md", tmp_path)
    _assert_scrubbed(text)
    assert list((home / "run").iterdir()) == []
    assert world.transport.requests == []
    _assert_golden("prior_crash_transcript.txt", _transcript(capsys, path), tmp_path)


@pytest.mark.parametrize("trace, fatal", [
    ("Windows fatal exception: code 0x8001010d\n", False),
    ("Windows fatal exception: code 0xc0000409\n", True),
    ("Fatal Python error: Segmentation fault\n", True),
    ("", False),
])
def test_prior_crash_classification(world, tmp_path, trace, fatal):
    home = _crash_home(tmp_path)
    run = home / "run"
    (run / "server-crash.inst-a.marker").write_text(
        json.dumps({"pid": CRASH_PID}), encoding="utf-8")
    if trace:
        (run / "server-crash-trace.inst-a.txt").write_text(trace, encoding="utf-8")
    path = bugreport.check_and_report_prior_crash(home=home)
    head = path.read_text(encoding="utf-8").splitlines()[0]
    if fatal:
        assert head == ("# localm bug report: localm server crashed - native fault "
                        f"captured: {trace.strip()}")
    else:
        assert head == ("# localm bug report: localm server crashed "
                        "(recovered on the next start)")


def test_prior_crash_variants(world, tmp_path):
    home = _crash_home(tmp_path)
    run = home / "run"
    _arm_dead_run(home)
    world.live_pids.add(CRASH_PID)
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert (run / "server-crash.inst-a.marker").exists()

    world.live_pids.clear()
    world.diagnostics = False
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert not (run / "server-crash.inst-a.marker").exists()
    assert not (run / "server-crash-trace.inst-a.txt").exists()
    assert _saved_reports(world) == []

    world.diagnostics = True
    _arm_dead_run(home, diagnostics=False)
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert _saved_reports(world) == []

    (run / "server-crash.marker").write_text("{half a marker", encoding="utf-8")
    path = bugreport.check_and_report_prior_crash(home=home)
    assert path is not None
    assert path.read_text(encoding="utf-8").startswith(
        "# localm bug report: localm server crashed during model load/construction "
        "(native crash suspected, no trace captured)\n")
    assert world.transport.requests == []


def test_stopping_record_report_is_byte_identical(world, tmp_path):
    home = _crash_home(tmp_path)
    run = home / "run"
    (run / "server-crash.inst-b.stopping").write_text(json.dumps(
        {"pid": STOP_PID, "context": {"host": HOST}, "diagnostics": True}),
        encoding="utf-8")
    (run / "server-crash-trace.inst-b.txt").write_text(
        "Fatal Python error: Segmentation fault\n\n"
        "Thread 0x00007f00 (most recent call first):\n"
        f'  File "{POSIX_PATH}", line 1 in close\n'
        f"  Authorization: Bearer {BEARER}\n", encoding="utf-8")
    _write_run_log(home / "logs", STOP_PID)
    path = bugreport.check_and_report_prior_crash(home=home)
    text = _assert_file_bytes(path, "stopping_report.md", tmp_path)
    _assert_scrubbed(text)
    assert list(run.iterdir()) == []


def test_stopping_record_variants(world, tmp_path):
    home = _crash_home(tmp_path)
    run = home / "run"
    record = run / "server-crash.inst-b.stopping"
    trace = run / "server-crash-trace.inst-b.txt"
    record.write_text(json.dumps({"pid": STOP_PID}), encoding="utf-8")
    trace.write_text("Windows fatal exception: code 0x8001010d\n", encoding="utf-8")
    world.live_pids.add(STOP_PID)
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert record.exists() and trace.exists()
    world.live_pids.clear()
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert list(run.iterdir()) == []
    record.write_text(json.dumps({"pid": STOP_PID}), encoding="utf-8")
    trace.write_text("Fatal Python error: Aborted\n", encoding="utf-8")
    world.diagnostics = False
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert list(run.iterdir()) == []
    assert _saved_reports(world) == []


def test_live_server_hang_trace(world, tmp_path):
    home = tmp_path / "live-home"
    (home / "run").mkdir(parents=True)
    (home / "logs").mkdir()
    for iid, pid, started in (("a", LIVE_PID, "2026-10-09T10:00:00"),
                              ("b", CRASH_PID, "2026-10-09T11:00:00")):
        (home / "run" / f"{iid}.json").write_text(json.dumps(
            {"pid": pid, "started": started, "port": 1}), encoding="utf-8")
        _write_hang(home / "logs", pid, label=f"0x000{iid}")
    world.live_pids.add(LIVE_PID)
    trace = bugreport.live_server_hang_trace(home)
    assert trace == (
        "Thread 0x000a (most recent call first):\n"
        '  File "C:\\Users\\<redacted>\\Documents\\localm\\notes.txt", line 40 in _run_once\n'
        "  Authorization: <redacted>")
    world.live_pids.clear()
    assert bugreport.live_server_hang_trace(home) == ""
