# SPDX-License-Identifier: AGPL-3.0-or-later
"""The standalone post-restart health watchdog (scripts/update_watchdog.py,
LM-DA-011): polls a restarted instance's own /whoami for the applied version and
auto-invokes scripts/rollback_update.py on failure/timeout - loaded by file path,
exactly like rollback_update.py itself loads _apply_update.py, so a broken new
build cannot break the thing meant to detect it. Every test runs against a
THROWAWAY install + home + fake HTTP server, never the real repo/install."""

from __future__ import annotations

import http.server
import importlib.util
import json
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WATCHDOG_SCRIPT = REPO / "scripts" / "update_watchdog.py"


def _load_wd():
    spec = importlib.util.spec_from_file_location("update_watchdog", WATCHDOG_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _whoami_server(responder, port: int = 0):
    """A ThreadingHTTPServer on 127.0.0.1 whose /whoami handler calls
    ``responder() -> dict`` per request. Returns (server, thread, port); caller
    must call server.shutdown() when done."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/whoami":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(responder()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):
            pass   # keep test output quiet

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, server.server_address[1]


def _stage(tmp_path, monkeypatch, *, manifest, with_helper=True):
    """Mirrors tests/test_rollback_standalone.py's _stage(): a fake install (with
    the REAL rollback_update.py, and optionally _apply_update.py, copied in) and a
    fake home whose updates/backup holds the pre-apply content."""
    inst = tmp_path / "install"
    (inst / "scripts").mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "rollback_update.py",
                inst / "scripts" / "rollback_update.py")
    if with_helper:
        (inst / "localm").mkdir()
        shutil.copy2(REPO / "localm" / "_apply_update.py",
                    inst / "localm" / "_apply_update.py")
    home = tmp_path / "home"
    (home / "updates" / "backup").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    (inst / "existing.txt").write_text("NEW-from-update", encoding="utf-8")
    (home / "updates" / "backup" / "existing.txt").write_text("OLD-preapply", encoding="utf-8")
    if manifest is not None:
        (home / "updates" / "applied_names.json").write_text(
            json.dumps(manifest), encoding="utf-8")
    return inst, home


# ------------------------------- health check ------------------------------

def test_health_check_passes_no_rollback(tmp_path, monkeypatch):
    wd = _load_wd()
    server, thread, port = _whoami_server(lambda: {"app": "localm", "version": "1.2.3"})
    try:
        def _boom(_install_root):
            raise AssertionError("rollback must not be invoked on a healthy check")
        monkeypatch.setattr(wd, "_load_rollback_module", _boom)
        code = wd.main([
            "--host", "127.0.0.1", "--port", str(port),
            "--expect-version", "1.2.3", "--install-root", str(tmp_path),
            "--timeout", "5", "--poll-interval", "0.2", "--request-timeout", "1"])
        assert code == wd.EXIT_HEALTHY
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_wrong_version_then_correct_eventually_succeeds(tmp_path):
    wd = _load_wd()
    calls = {"n": 0}

    def respond():
        calls["n"] += 1
        return {"app": "localm", "version": "0.1.0" if calls["n"] < 3 else "0.2.0"}

    server, thread, port = _whoami_server(respond)
    try:
        code = wd.main([
            "--host", "127.0.0.1", "--port", str(port),
            "--expect-version", "0.2.0", "--install-root", str(tmp_path),
            "--timeout", "5", "--poll-interval", "0.1", "--request-timeout", "1"])
        assert code == wd.EXIT_HEALTHY
        # Did not succeed on the FIRST wrong-version answer - proves it kept polling.
        assert calls["n"] >= 3
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_connection_refused_then_recovers(tmp_path):
    wd = _load_wd()
    port = _free_port()   # nothing listens here yet
    result = {}

    def _start_late():
        time.sleep(0.6)
        server, thread, _ = _whoami_server(
            lambda: {"app": "localm", "version": "0.2.0"}, port=port)
        result["server"], result["thread"] = server, thread

    starter = threading.Thread(target=_start_late, daemon=True)
    starter.start()
    try:
        code = wd.main([
            "--host", "127.0.0.1", "--port", str(port),
            "--expect-version", "0.2.0", "--install-root", str(tmp_path),
            "--timeout", "5", "--poll-interval", "0.2", "--request-timeout", "1"])
        assert code == wd.EXIT_HEALTHY
    finally:
        starter.join(timeout=3)
        if "server" in result:
            result["server"].shutdown()
            result["thread"].join(timeout=2)


# --------------------------- timeout -> rollback ----------------------------

def test_timeout_triggers_rollback_and_restores_backup(tmp_path, monkeypatch):
    wd = _load_wd()
    inst, _home = _stage(tmp_path, monkeypatch, manifest=["existing.txt"])
    closed_port = _free_port()   # nothing listens here for the duration of this test
    logfile = tmp_path / "watchdog.log"
    code = wd.main([
        "--host", "127.0.0.1", "--port", str(closed_port),
        "--expect-version", "9.9.9", "--install-root", str(inst),
        "--timeout", "1.0", "--poll-interval", "0.2", "--request-timeout", "0.3",
        "--log-file", str(logfile)])
    assert code == wd.EXIT_ROLLED_BACK
    assert (inst / "existing.txt").read_text(encoding="utf-8") == "OLD-preapply"
    assert "watchdog exiting with code 1" in logfile.read_text(encoding="utf-8")


def test_rollback_itself_failing_returns_distinct_code(tmp_path, monkeypatch):
    """A build so broken the restore helper cannot even load must fail loud with a
    DIFFERENT code than a successful rollback - never confused with recovery."""
    wd = _load_wd()
    inst, _home = _stage(tmp_path, monkeypatch, manifest=["existing.txt"],
                         with_helper=False)
    closed_port = _free_port()
    code = wd.main([
        "--host", "127.0.0.1", "--port", str(closed_port),
        "--expect-version", "9.9.9", "--install-root", str(inst),
        "--timeout", "1.0", "--poll-interval", "0.2", "--request-timeout", "0.3"])
    assert code == wd.EXIT_ROLLBACK_FAILED
    assert code != wd.EXIT_ROLLED_BACK
    # The install was never touched (rollback couldn't even load).
    assert (inst / "existing.txt").read_text(encoding="utf-8") == "NEW-from-update"


# -------------------------------- misc units --------------------------------

def test_insecure_ssl_context_disables_verification():
    wd = _load_wd()
    ctx = wd._insecure_ssl_context()
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_NONE


# ------------------------- detachment survival ------------------------------

def test_survives_after_launcher_process_exits(tmp_path):
    """The watchdog must be a genuinely DETACHED OS process, not tied to its
    immediate launcher's lifetime - the whole point is surviving a server that
    crashes right after spawning it. A throwaway launcher spawns the watchdog with
    the exact same detachment flags updater.spawn_health_watchdog() uses, then
    exits immediately; the watchdog must keep running (and finish) afterward."""
    closed_port = _free_port()
    logfile = tmp_path / "watchdog.log"
    inst = tmp_path / "install"   # deliberately has no rollback helper - this test
    inst.mkdir()                  # only cares that the watchdog ran to completion.
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import subprocess, sys\n"
        f"argv = [sys.executable, {str(WATCHDOG_SCRIPT)!r}, "
        "'--host', '127.0.0.1', '--port', " + repr(str(closed_port)) + ", "
        "'--expect-version', 'irrelevant', "
        f"'--install-root', {str(inst)!r}, "
        "'--timeout', '1.5', '--poll-interval', '0.3', "
        "'--request-timeout', '0.3', "
        f"'--log-file', {str(logfile)!r}]\n"
        "kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL, close_fds=True)\n"
        "if sys.platform == 'win32':\n"
        "    kwargs['creationflags'] = 0x00000008 | 0x00000200\n"
        "else:\n"
        "    kwargs['start_new_session'] = True\n"
        "subprocess.Popen(argv, **kwargs)\n",
        encoding="utf-8")

    subprocess.run([sys.executable, str(launcher)], check=True, timeout=10)
    # subprocess.run() has now returned - the launcher process is confirmed gone.
    # The watchdog it spawned must keep running independently and finish on its own.
    #
    # This window is deliberately generous (verified, not guessed): under this
    # suite's own full -n auto run, a captured failure here showed the watchdog
    # WAS alive and progressing past its launcher's exit (its log had two real
    # "attempt N" lines logged strictly after the trivial launcher had already
    # returned - proof the detached process was not killed), but then went
    # unscheduled for 7+ seconds under 16-way CPU contention before this test's
    # old 8s window gave up - a low-priority detached background process getting
    # starved of CPU time under heavy parallel load, not a survival/detachment
    # failure. 45s leaves large margin above that observed worst case while the
    # common (uncontended) case still exits this loop in well under a second via
    # the early break.
    deadline = time.monotonic() + 45.0
    content = ""
    while time.monotonic() < deadline:
        if logfile.exists():
            content = logfile.read_text(encoding="utf-8")
            if "watchdog exiting with code" in content:
                break
        time.sleep(0.2)
    assert "watchdog exiting with code" in content, (
        f"watchdog did not complete after its launcher exited; log so far: {content!r}")
