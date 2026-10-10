# SPDX-License-Identifier: AGPL-3.0-or-later
"""The managed KoboldCpp music server: one process per model set, resident
between jobs, bound to 127.0.0.1 with a fresh random API key per start.

Every public function here is serialised by one module lock, so two music jobs
never share or race the process; a job that arrives while another runs waits
for it. The server exits on :func:`stop` (the chat/media VRAM handover), after
:data:`IDLE_SECONDS` without a job, when a job needs a different model set or
backend, and with localm (see ``_proc``).

Progress comes from localm's own phase lines and a heartbeat while a request
runs: KoboldCpp's native output is block-buffered on a pipe, so it arrives too
late to report stages from. That output is kept in memory only, for the reason
shown when the process fails.
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import _proc
from .runtime import Runtime

IDLE_SECONDS = 600.0
START_TIMEOUT = 900.0
HEARTBEAT_SECONDS = 15.0
# How long a dropped connection waits for the process to exit before the
# failure is reported as a request error rather than the process stopping.
_EXIT_GRACE = 3.0
_POLL = 0.25
_LOG_LINES = 200

Progress = Callable[[str], None]
CancelCheck = Callable[[], bool]


class ServerError(RuntimeError):
    """The server could not start, or a request to it failed."""


class StartError(ServerError):
    """The server did not start or did not load the music models. ``crashed`` is
    True when the process exited while loading for a reason other than running
    out of memory."""

    def __init__(self, message: str, *, crashed: bool = False) -> None:
        super().__init__(message)
        self.crashed = crashed


_MEMORY_RE = re.compile(
    r"out of memory|outofmemory|\boom\b|failed to allocate|cudamalloc|vk_error_out_of"
    r"|insufficient memory", re.IGNORECASE)


def _mentions_memory(text: str) -> bool:
    return _MEMORY_RE.search(text) is not None


class Cancelled(Exception):
    """The job was cancelled; the server was stopped to end it."""


@dataclass(frozen=True)
class ModelSet:
    """The ACE-Step component files. ``lm`` is optional: without it the DiT
    generates from the caption and lyrics alone."""
    text_encoder: str
    dit: str
    vae: str
    lm: Optional[str] = None


@dataclass(frozen=True)
class ServerKey:
    """Everything a running server was started with; a job whose key differs
    gets a fresh server."""
    launcher: str
    backend: str
    models: ModelSet
    lowvram: bool = False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_argv(key: ServerKey, port: int) -> list:
    """The launcher command line for *key*: music models only, loopback. The API
    key is passed in the ``KCPP_PASSWORD`` environment variable, not here."""
    m = key.models
    argv = [key.launcher,
            "--musicembeddings", m.text_encoder,
            "--musicdiffusion", m.dit,
            "--musicvae", m.vae]
    if m.lm:
        argv += ["--musicllm", m.lm]
    argv += ["--host", "127.0.0.1", "--port", str(port), "--skiplauncher", "--quiet"]
    if key.lowvram:
        argv.append("--musiclowvram")
    if key.backend == "cuda":
        argv.append("--usecuda")
    elif key.backend == "vulkan":
        argv.append("--usevulkan")
    elif key.backend == "cpu":
        argv.append("--usecpu")
    return argv


class _Server:
    def __init__(self, key: ServerKey, work_dir: Path) -> None:
        self.key = key
        self.work_dir = work_dir
        self.port = 0
        self.password = ""
        self.mp: Optional[_proc.ManagedProcess] = None
        self.log: collections.deque = collections.deque(maxlen=_LOG_LINES)
        self._reader: Optional[threading.Thread] = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.mp is not None and self.mp.poll() is None

    def log_tail(self, n: int = 8) -> str:
        lines = [ln for ln in list(self.log) if ln.strip()]
        return "\n".join(lines[-n:])

    def _read(self, mp: _proc.ManagedProcess) -> None:
        if mp.proc.stdout is None:
            return
        try:
            for line in mp.proc.stdout:
                self.log.append(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass

    def start(self, on_progress: Progress, cancel_check: CancelCheck) -> None:
        """Start the process and wait until it serves with the music models
        loaded. Raises :class:`StartError` or :class:`Cancelled`; the process is
        stopped on any failure."""
        self.port = _free_port()
        self.password = secrets.token_urlsafe(32)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for k in ("TEMP", "TMP", "TMPDIR"):
            env[k] = str(self.work_dir)
        env["KCPP_PASSWORD"] = self.password
        env["PYTHONUNBUFFERED"] = "1"
        t0 = last_beat = time.monotonic()
        try:
            self.mp = _proc.start(build_argv(self.key, self.port),
                                  cwd=str(self.work_dir), env=env)
        except OSError as e:
            raise StartError(f"the music runtime could not be started: {e}") from e
        self._reader = threading.Thread(target=self._read, args=(self.mp,),
                                        name="koboldcpp-log", daemon=True)
        self._reader.start()
        deadline = t0 + START_TIMEOUT
        while True:
            if cancel_check():
                self.stop()
                raise Cancelled()
            code = self.mp.poll()
            if code is not None:
                self.stop()
                tail = self.log_tail()
                raise StartError(
                    f"the music runtime exited while loading (exit code {code}): "
                    f"{tail or 'no output'}",
                    crashed=not _mentions_memory("\n".join(self.log)))
            info = self._version()
            if info is not None:
                if self._listener_is_ours() is False:
                    self.stop()
                    raise StartError(
                        f"another program is listening on port {self.port}, which the music "
                        "runtime was started on; retry")
                if not info.get("music"):
                    self.stop()
                    raise StartError(
                        "the music runtime started but did not load the music models: "
                        f"{self.log_tail() or 'no output'}")
                on_progress(f"Music models loaded in {time.monotonic() - t0:.0f} s.")
                return
            now = time.monotonic()
            if now > deadline:
                self.stop()
                raise StartError(f"the music runtime did not finish loading in "
                                 f"{START_TIMEOUT:.0f} s: {self.log_tail() or 'no output'}")
            if now - last_beat >= HEARTBEAT_SECONDS:
                last_beat = now
                on_progress(f"Still loading the music models ({now - t0:.0f} s)...")
            time.sleep(_POLL)

    def _listener_is_ours(self) -> Optional[bool]:
        """Whether the socket listening on ``self.port`` belongs to the started
        process or its children: True, False, or None when it cannot be told
        (psutil missing or access denied)."""
        try:
            import psutil
        except ImportError:
            return None
        if self.mp is None:
            return None
        try:
            root = psutil.Process(self.mp.pid)
            procs = [root] + root.children(recursive=True)
        except psutil.Error:
            return None
        for proc in procs:
            try:
                get = getattr(proc, "net_connections", None) or proc.connections
                conns = get(kind="tcp")
            except psutil.NoSuchProcess:
                continue
            except psutil.Error:
                return None
            for c in conns:
                if c.laddr and c.laddr.port == self.port and not c.raddr:
                    return True
        return False

    def _version(self) -> Optional[dict]:
        try:
            with self._opener.open(self.base + "/api/extra/version", timeout=2) as r:
                data = json.loads(r.read())
        except (OSError, ValueError, RecursionError):
            return None
        return data if isinstance(data, dict) else None

    def request(self, path: str, body: dict, *, timeout: float,
                cancel_check: CancelCheck, on_progress: Optional[Progress] = None,
                activity: str = "Working") -> tuple[str, bytes]:
        """POST *body* to *path*; returns (content type, body). While it runs,
        *on_progress* gets "<activity> (N s)..." every ``HEARTBEAT_SECONDS``. A
        cancel stops the server, which ends the request."""
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.password})
        result: dict = {}

        def _call() -> None:
            try:
                with self._opener.open(req, timeout=timeout) as r:
                    result["ctype"] = r.headers.get("Content-Type") or ""
                    result["data"] = r.read()
            except urllib.error.HTTPError as e:
                result["error"] = f"HTTP {e.code}"
                result["http"] = True
            except (OSError, ValueError) as e:
                result["error"] = str(e) or type(e).__name__

        t = threading.Thread(target=_call, name="koboldcpp-request", daemon=True)
        t0 = last_beat = time.monotonic()
        t.start()
        while t.is_alive():
            t.join(_POLL)
            now = time.monotonic()
            if on_progress is not None and now - last_beat >= HEARTBEAT_SECONDS:
                last_beat = now
                on_progress(f"{activity} ({now - t0:.0f} s)...")
            if cancel_check():
                self.stop()
                t.join(10)
                raise Cancelled()
        if "error" in result:
            if not result.get("http") and self.mp is not None:
                try:
                    self.mp.proc.wait(timeout=_EXIT_GRACE)
                except subprocess.TimeoutExpired:
                    pass
            tail = self.log_tail()
            if not self.alive():
                raise ServerError(f"the music runtime stopped during the request "
                                  f"({result['error']}): {tail or 'no output'}")
            raise ServerError(f"the music runtime request failed ({result['error']})"
                              + (f": {tail}" if tail else ""))
        return result["ctype"], result["data"]

    def stop(self) -> None:
        mp, self.mp = self.mp, None
        if mp is not None:
            _proc.kill(mp)
        if self._reader is not None:
            self._reader.join(timeout=5)
            self._reader = None


_lock = threading.Lock()
_server: Optional[_Server] = None
_idle_timer: Optional[threading.Timer] = None
_last_used = 0.0


def _cancel_idle_timer() -> None:
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def _arm_idle_timer() -> None:
    global _idle_timer
    _cancel_idle_timer()
    t = threading.Timer(IDLE_SECONDS, _idle_check)
    t.daemon = True
    t.start()
    _idle_timer = t


def _idle_check() -> None:
    global _server
    if not _lock.acquire(blocking=False):
        return
    try:
        if _server is not None and time.monotonic() - _last_used >= IDLE_SECONDS - 1:
            _server.stop()
            _server = None
    finally:
        _lock.release()


def _acquire(on_progress: Progress, cancel_check: CancelCheck) -> None:
    if _lock.acquire(blocking=False):
        return
    on_progress("Waiting for the music generation already running...")
    while not _lock.acquire(timeout=_POLL):
        if cancel_check():
            raise Cancelled()


def run(runtime: Runtime, backend: str, models: ModelSet, work_dir: Path, *,
        prepare: bool, request: dict, timeout: float, lowvram: bool = False,
        on_progress: Optional[Progress] = None,
        cancel_check: Optional[CancelCheck] = None) -> bytes:
    """Generate one track and return the WAV bytes KoboldCpp produced.

    Starts (or reuses) the server for this runtime, backend and model set. When
    *prepare* is true and an LM is loaded, the request first goes through
    ``/api/extra/music/prepare`` with audio-code generation on and caption
    rewriting off; the planned fields (tempo, key, time signature, audio codes)
    are then merged under the caller's own request fields. The planner ends the
    song itself, so the track can be slightly shorter than ``duration``. Raises
    :class:`StartError` when the server cannot start, :class:`ServerError` for
    any other failure, or :class:`Cancelled`."""
    global _server, _last_used
    say = on_progress or (lambda _m: None)
    cancelled = cancel_check or (lambda: False)
    _acquire(say, cancelled)
    try:
        _cancel_idle_timer()
        key = ServerKey(str(runtime.launcher), backend, models, lowvram)
        if _server is not None and (_server.key != key or not _server.alive()):
            _server.stop()
            _server = None
        if _server is None:
            say(f"Starting the native music runtime ({backend})...")
            srv = _Server(key, work_dir)
            srv.start(say, cancelled)
            _server = srv
        srv = _server
        try:
            body = dict(request)
            if prepare and models.lm:
                say("Planning the track (tempo, key and structure)...")
                plan_body = dict(request, gen_codes=True, rewrite_caption=False)
                _ctype, raw = srv.request("/api/extra/music/prepare", plan_body,
                                          timeout=timeout, cancel_check=cancelled,
                                          on_progress=say, activity="Still planning")
                try:
                    planned = json.loads(raw)
                except (ValueError, RecursionError) as e:
                    raise ServerError("the music planner returned an unreadable reply") from e
                if isinstance(planned, dict) and not planned.get("error"):
                    body = dict(planned)
                    body.update(request)
                elif isinstance(planned, dict):
                    say(f"The music planner failed ({planned.get('error')}); "
                        "generating from the prompt as given.")
            say("Generating the audio...")
            t_gen = time.monotonic()
            ctype, data = srv.request("/api/extra/music/generate", body,
                                      timeout=timeout, cancel_check=cancelled,
                                      on_progress=say, activity="Still generating")
            say(f"Audio generated in {time.monotonic() - t_gen:.1f} s.")
        except Cancelled:
            _server = None
            raise
        except ServerError:
            if not srv.alive():
                srv.stop()
                _server = None
            raise
        if not data:
            raise ServerError("the music runtime returned no audio" +
                              (f": {srv.log_tail()}" if srv.log_tail() else ""))
        if "wav" not in ctype.lower() and not data.startswith(b"RIFF"):
            raise ServerError(f"the music runtime returned {ctype or 'unknown content'}, "
                              "not WAV audio")
        return data
    finally:
        _last_used = time.monotonic()
        if _server is not None:
            _arm_idle_timer()
        _lock.release()


def _stop_at_exit() -> None:
    global _server
    if not _lock.acquire(timeout=5):
        return
    try:
        _cancel_idle_timer()
        if _server is not None:
            _server.stop()
            _server = None
    finally:
        _lock.release()


atexit.register(_stop_at_exit)


def stop() -> bool:
    """Stop the music server if one is running. True when one was stopped."""
    global _server
    with _lock:
        _cancel_idle_timer()
        if _server is None:
            return False
        _server.stop()
        _server = None
        return True


def running_pid() -> Optional[int]:
    """The PID of the music server localm started, if it is running."""
    srv = _server
    if srv is None or srv.mp is None or not srv.alive():
        return None
    return srv.mp.pid
