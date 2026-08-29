# SPDX-License-Identifier: AGPL-3.0-or-later
"""The standalone crash-recovery watchdog (scripts/crash_recovery_watchdog.py): watches
a server pid and relaunches it if it dies without clearing its own crash marker.
Loaded by file path, matching test_update_watchdog.py's convention for the sibling
watchdog script. Every test drives REAL child processes and a REAL /whoami HTTP
server on a throwaway port; nothing here mocks pid liveness or the network call."""

from __future__ import annotations

import http.server
import importlib.util
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WATCHDOG_SCRIPT = REPO / "scripts" / "crash_recovery_watchdog.py"


def _load_wd():
    spec = importlib.util.spec_from_file_location("crash_recovery_watchdog", WATCHDOG_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _whoami_server(instance_id_holder: list):
    """A ThreadingHTTPServer whose /whoami reports instance_id_holder[0], or 503s
    when it is None (nothing bound yet). Returns (server, thread, port); caller
    must call server.shutdown()."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/whoami":
                self.send_response(404)
                self.end_headers()
                return
            if instance_id_holder[0] is None:
                self.send_response(503)
                self.end_headers()
                return
            body = json.dumps({"app": "localm", "instance_id": instance_id_holder[0]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, server.server_address[1]


def _spawn_sleeper(seconds: float) -> subprocess.Popen:
    """A real child process that stays alive for *seconds* then exits 0."""
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def _write_marker(crash_dir: Path, instance_id: str, pid: int) -> Path:
    crash_dir.mkdir(parents=True, exist_ok=True)
    p = crash_dir / f"server-crash.{instance_id}.marker"
    p.write_text(json.dumps({"pid": pid, "context": {}}), encoding="utf-8")
    return p


class TestMarkerPathMatchesBugreport:
    def test_marker_path_matches_bugreport(self):
        wd = _load_wd()
        from localm import bugreport
        d = Path("D:/does/not/need/to/exist/run")
        for instance_id in ("abc123", "0" * 16, "with-dashes-99"):
            assert wd.marker_path(d, instance_id) == bugreport._crash_marker_path(d, instance_id)


class TestPidAlive:
    def test_true_for_self(self):
        import os
        wd = _load_wd()
        assert wd.pid_alive(os.getpid()) is True

    def test_false_after_real_process_exits(self):
        wd = _load_wd()
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=10)
        deadline = time.monotonic() + 5
        while wd.pid_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert wd.pid_alive(proc.pid) is False


class TestReadMarkerPid:
    def test_missing_file_returns_none(self, tmp_path):
        wd = _load_wd()
        assert wd.read_marker_pid(tmp_path / "nope.marker") is None

    def test_reads_the_recorded_pid(self, tmp_path):
        wd = _load_wd()
        marker = _write_marker(tmp_path, "inst1", 4242)
        assert wd.read_marker_pid(marker) == 4242


class TestRunCleanShutdown:
    """A watched process exits and its marker is already gone (disarm_crash_guard
    already ran) - no replacement ever answers /whoami. run() must return
    EXIT_WATCHDOG_STOPPED_CLEAN and must NOT relaunch."""

    def test_clean_exit_is_never_relaunched(self, tmp_path):
        wd = _load_wd()
        holder = [None]
        server, thread, port = _whoami_server(holder)
        try:
            proc = _spawn_sleeper(0.3)
            sentinel = tmp_path / "relaunched.txt"
            relaunch_argv = [sys.executable, "-c",
                             f"open(r'{sentinel}', 'w').write('x')"]
            # No marker written at all: mirrors disarm_crash_guard having already
            # removed it as part of a clean shutdown.
            code = wd.run(pid=proc.pid, host="127.0.0.1", port=port, scheme="http",
                          instance_id="clean-inst", crash_dir=tmp_path,
                          relaunch_argv=relaunch_argv, poll_interval=0.05,
                          grace_s=0.5, request_timeout=1.0, log_path=None)
            proc.wait(timeout=5)
            assert code == wd.EXIT_WATCHDOG_STOPPED_CLEAN
            assert not sentinel.exists()
        finally:
            server.shutdown()
            thread.join(timeout=5)


class TestRunCrashRelaunches:
    """A watched process exits WITHOUT clearing its marker (a crash). run() must
    invoke the relaunch command and then track whatever instance answers next."""

    def test_crash_triggers_a_real_relaunch(self, tmp_path):
        wd = _load_wd()
        holder = [None]
        server, thread, port = _whoami_server(holder)
        try:
            proc = _spawn_sleeper(0.2)
            _write_marker(tmp_path, "dead-inst", proc.pid)
            sentinel = tmp_path / "relaunched.txt"
            relaunch_argv = [sys.executable, "-c",
                             f"open(r'{sentinel}', 'w').write('x')"]

            done = {}

            def _drive():
                done["code"] = wd.run(
                    pid=proc.pid, host="127.0.0.1", port=port, scheme="http",
                    instance_id="dead-inst", crash_dir=tmp_path,
                    relaunch_argv=relaunch_argv, poll_interval=0.05,
                    grace_s=0.5, request_timeout=1.0, log_path=None)

            t = threading.Thread(target=_drive, daemon=True)
            t.start()

            deadline = time.monotonic() + 10
            while not sentinel.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert sentinel.exists(), "the crash was never relaunched"

            # Give the never-answering relaunch its grace window, then bring up
            # a real, live process as the recovered instance so run() settles
            # into steady watching instead of treating a fake pid as an
            # instant second crash.
            recovered = _spawn_sleeper(5.0)
            try:
                _write_marker(tmp_path, "recovered-inst", recovered.pid)
                holder[0] = "recovered-inst"
                time.sleep(0.3)
                assert t.is_alive(), (
                    "run() exited instead of settling into watching the "
                    "recovered instance")
            finally:
                recovered.kill()
                recovered.wait(timeout=5)
        finally:
            server.shutdown()
            thread.join(timeout=5)


class TestRunStormCap:
    """A relaunch that ALSO crashes immediately, every time, must eventually stop
    being retried rather than loop forever."""

    def test_repeated_crash_is_capped(self, tmp_path):
        wd = _load_wd()
        wd._STORM_LIMIT = 2
        wd._STORM_WINDOW_S = 300.0
        holder = [None]
        server, thread, port = _whoami_server(holder)
        try:
            proc = _spawn_sleeper(0.2)
            _write_marker(tmp_path, "crashy", proc.pid)
            # Every relaunch is itself an instant no-op process - it never
            # answers /whoami, so poll_for_new_instance always times out and
            # run() re-checks the SAME still-present marker on each pass.
            relaunch_argv = [sys.executable, "-c", "pass"]

            code = wd.run(pid=proc.pid, host="127.0.0.1", port=port, scheme="http",
                          instance_id="crashy", crash_dir=tmp_path,
                          relaunch_argv=relaunch_argv, poll_interval=0.02,
                          grace_s=0.1, request_timeout=0.5, log_path=None)
            assert code == wd.EXIT_STORM_CAPPED
        finally:
            server.shutdown()
            thread.join(timeout=5)

    def test_fires_control_without_the_cap_it_never_returns(self, tmp_path):
        """Proves the test above actually exercises the cap: with the storm
        check disabled, the identical scenario never reaches a return at all
        within a generous bound."""
        wd = _load_wd()
        wd._STORM_LIMIT = 10 ** 9
        wd._STORM_WINDOW_S = 300.0
        holder = [None]
        server, thread, port = _whoami_server(holder)
        try:
            proc = _spawn_sleeper(0.2)
            _write_marker(tmp_path, "crashy2", proc.pid)
            relaunch_argv = [sys.executable, "-c", "pass"]

            result = {}

            def _drive():
                result["code"] = wd.run(
                    pid=proc.pid, host="127.0.0.1", port=port, scheme="http",
                    instance_id="crashy2", crash_dir=tmp_path,
                    relaunch_argv=relaunch_argv, poll_interval=0.02,
                    grace_s=0.1, request_timeout=0.5, log_path=None)

            t = threading.Thread(target=_drive, daemon=True)
            t.start()
            t.join(timeout=3)
            assert t.is_alive(), (
                "run() returned even with the storm cap effectively disabled - "
                "the cap test above is not exercising what it claims to")
        finally:
            server.shutdown()
            thread.join(timeout=5)


class TestPollForNewInstance:
    def test_times_out_when_nothing_answers(self, tmp_path):
        wd = _load_wd()
        holder = [None]
        server, thread, port = _whoami_server(holder)
        try:
            result = wd.poll_for_new_instance(
                "127.0.0.1", port, "http", "old-id", tmp_path,
                deadline_s=0.2, poll_interval=0.02, request_timeout=0.5,
                log_path=None)
            assert result is None
        finally:
            server.shutdown()
            thread.join(timeout=5)

    def test_finds_a_new_instance_once_its_marker_exists(self, tmp_path):
        wd = _load_wd()
        holder = [None]
        server, thread, port = _whoami_server(holder)
        try:
            def _flip_after_delay():
                time.sleep(0.1)
                _write_marker(tmp_path, "new-id", 5555)
                holder[0] = "new-id"

            threading.Thread(target=_flip_after_delay, daemon=True).start()
            result = wd.poll_for_new_instance(
                "127.0.0.1", port, "http", "old-id", tmp_path,
                deadline_s=2.0, poll_interval=0.02, request_timeout=0.5,
                log_path=None)
            assert result == ("new-id", 5555)
        finally:
            server.shutdown()
            thread.join(timeout=5)

    def test_ignores_the_same_old_instance_id(self, tmp_path):
        wd = _load_wd()
        holder = ["old-id"]
        _write_marker(tmp_path, "old-id", 111)
        server, thread, port = _whoami_server(holder)
        try:
            result = wd.poll_for_new_instance(
                "127.0.0.1", port, "http", "old-id", tmp_path,
                deadline_s=0.2, poll_interval=0.02, request_timeout=0.5,
                log_path=None)
            assert result is None
        finally:
            server.shutdown()
            thread.join(timeout=5)
