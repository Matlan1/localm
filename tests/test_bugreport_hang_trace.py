# SPDX-License-Identifier: AGPL-3.0-or-later
"""A captured event-loop hang trace (from the always-on watchdog) must be bundled
into a bug report automatically, so a non-technical tester who just files their
normal report carries the freeze diagnosis with them - no env var, no py-spy."""

import os

import localm.config as cfg
from localm import bugreport


def _seed_hang(home, name=None, body=None):
    """Seed a hang capture the way the product actually produces one: the stall
    watchdog runs INSIDE the server process (http_server.py) and names the file
    hang_<date>_<os.getpid()>.log, and the only caller of save_user_report is a
    route in that same process. So the trace of the run being reported carries
    THIS process's pid - seeding a foreign pid would test a file the product
    never creates on this path (and one a report must NOT attach, since it would
    belong to some other run - REG-542)."""
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
