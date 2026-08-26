# SPDX-License-Identifier: AGPL-3.0-or-later
"""A captured event-loop hang trace (from the always-on watchdog) must be bundled
into a bug report automatically, so a non-technical tester who just files their
normal report carries the freeze diagnosis with them - no env var, no py-spy."""

import json
import os

import localm.config as cfg
from localm import bugreport


def _seed_registry(home, *, pid, instance_id="testsrv",
                   started="2026-07-10T12:00:00+00:00"):
    """Seed a live-instance registry entry the way a running server advertises one
    (<home>/run/<id>.json). Only the fields live_server_hang_trace reads (pid,
    started) matter here; the file is a plain JSON dict, exactly what
    instances.list_entries parses."""
    run = home / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{instance_id}.json").write_text(json.dumps({
        "instance_id": instance_id, "pid": pid, "port": 1, "host": "127.0.0.1",
        "root_dir": str(home), "mode": "full", "started": started,
    }), encoding="utf-8")


def _seed_hang(home, name=None, body=None):
    """Seed a hang capture the way the product actually produces one: the stall
    watchdog runs INSIDE the server process (http_server.py) and names the file
    hang_<date>_<os.getpid()>.log, and the only caller of save_user_report is a
    route in that same process. So the trace of the run being reported carries
    THIS process's pid. A foreign pid would be a file the product never creates
    on this path, and one a report must NOT attach, since it belongs to some
    other run."""
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    name = name or f"hang_2026-07-10_120000_{os.getpid()}.log"
    body = body or (
        f"===== LOCALM HANG WATCHDOG: event loop stalled 31.0s (pid {os.getpid()}) =====\n"
        'File "engine.py", line 302, in _list_gpus_probe\n'
        "    free, total = torch.cuda.mem_get_info(i)\n")
    (logs / name).write_text(body, encoding="utf-8")
    return logs / name


def test_recent_hang_traces_reads_the_capture(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    _seed_hang(home)
    ht = bugreport._recent_hang_traces(pid=os.getpid())
    assert "LOCALM HANG WATCHDOG" in ht
    assert "mem_get_info" in ht        # the culprit frame is carried through


def test_recent_hang_traces_empty_when_no_freeze(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    (home / "logs").mkdir(parents=True)
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    assert bugreport._recent_hang_traces(pid=os.getpid()) == ""   # healthy run leaves nothing


def test_build_report_renders_hang_section():
    text = bugreport.build_report(
        "x", context={"hang_traces": "MARKER-Z event loop stalled 40s\n frame"})
    assert "Server hang trace" in text
    assert "MARKER-Z" in text


def test_save_user_report_attaches_hang_trace_without_include_log(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    _seed_hang(home, body="HANG-MARKER event loop stalled 30.0s\n frame here\n")
    path = bugreport.save_user_report("the app froze", include_log=False)
    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "Server hang trace" in text     # attached even with include_log=False
    assert "HANG-MARKER" in text


# --------------------------------------------------------------------------- #
#  The `localm bug-report` CLI files from a SEPARATE process than the frozen  #
#  server, so it finds the live server via the run registry rather than by    #
#  its own pid.                                                               #
# --------------------------------------------------------------------------- #

def test_live_server_hang_trace_attaches_live_servers_freeze(tmp_path, monkeypatch):
    from localm import instances
    home = tmp_path / ".localm"
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    # The frozen server is a DIFFERENT process than this one (its pid is not
    # os.getpid()). Simulate that pid being alive: the freeze belongs to a server
    # still running while the reporter runs `localm bug-report`.
    server_pid = 4242424
    monkeypatch.setattr(instances, "pid_alive", lambda pid: pid == server_pid)
    _seed_registry(home, pid=server_pid)
    _seed_hang(home, name=f"hang_2026-07-10_120000_{server_pid}.log",
               body="LIVE-SRV-MARKER event loop stalled 30.0s\n frame\n")
    trace = bugreport.live_server_hang_trace()
    # Matched by the REGISTRY pid, not this process's pid.
    assert "LIVE-SRV-MARKER" in trace


def test_live_server_hang_trace_empty_without_a_registered_server(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    # A freeze on disk named for THIS process, but no run/*.json entry. The CLI
    # does not fall back to its own pid; with no live server registered there is
    # nothing to attach.
    _seed_hang(home, body="SELF-PID-TRACE event loop stalled 30.0s\n")
    assert bugreport.live_server_hang_trace() == ""


def test_live_server_hang_trace_skips_a_dead_registered_server(tmp_path, monkeypatch):
    from localm import instances
    home = tmp_path / ".localm"
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    # A stale registry entry for a server that already exited (pid not alive). Its
    # freeze belongs to the crash-recovery path on the next start, not this one.
    dead_pid = 4242424
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    _seed_registry(home, pid=dead_pid)
    _seed_hang(home, name=f"hang_2026-07-10_120000_{dead_pid}.log",
               body="DEAD-SRV-TRACE event loop stalled 30.0s\n")
    assert bugreport.live_server_hang_trace() == ""


def test_bug_report_cli_bundles_live_server_hang_trace(cli_runner):
    """End-to-end through the REAL `localm bug-report` command: a live server has
    registered itself and captured a freeze; the generated report must carry the
    hang-trace section. Drives the actual CLI + registry + collector, with no
    mock of the collector.
    """
    from localm import instances
    from localm.cli import main
    home = cfg.HOME_DIR   # cli_runner points HOME_DIR (and LOCALM_HOME) at a throwaway
    # register_instance stamps pid=os.getpid(), which is alive, so the real
    # pid_alive path runs (no monkeypatch).
    instances.register_instance(home, instance_id="livesrv", port=8080,
                                host="127.0.0.1", root_dir=str(home),
                                mode="full", token="tok")
    _seed_hang(home, body="CLI-E2E-MARKER event loop stalled 42.0s\n frame\n")

    result = cli_runner.invoke(main, ["bug-report", "-m", "server froze during test"])
    assert result.exit_code == 0, result.output

    reports = sorted((home / "bug-reports").glob("bug-*.md"))
    assert reports, f"no report file written; CLI output:\n{result.output}"
    text = reports[-1].read_text(encoding="utf-8")
    assert "Server hang trace" in text
    assert "CLI-E2E-MARKER" in text
