# SPDX-License-Identifier: AGPL-3.0-or-later
"""Three defects that make localm's own diagnostics lie about what went wrong.

localm/_log_digest.py - when the errors alone do not fit the budget, _fit_budget
only keeps a block that fits WHOLE. A single error bigger than the budget
therefore never fits, is counted as "omitted", and the digest comes back as JUST
the notice string with ZERO bytes of the actual crash. A plain last-max_chars
tail always included the innermost exception type+message - the most actionable
line there is - so a bug report for exactly the deep native-load crash this code
targets must not surface no error at all.

localm/bugreport.py - _recent_hang_traces attaching the newest hang_*.log with no
recency or run filter, while nothing ever prunes them. The watchdog is on by
default, so any transient >10s stall writes one. Weeks later an unrelated report
("the model gave a wrong answer") renders that stale freeze under "## Server hang
trace" with assertive text claiming it is "the server froze", sending triage down
a false path.

localm/inference/routes/admin.py - the post-update health watchdog probes
app.state.instance_port, but _restart_argv() re-execs with no port token, so the
new process re-runs pick_port() and can bind a DIFFERENT port (the one the old
instance was auto-bumped off). The watchdog then polls a dead port, times out at
90s, and auto-rolls back a perfectly healthy update.
"""

import os
import time

import pytest


# --------------------------------------------------------------------------- #
#  An oversized error is truncated, never dropped whole                        #
# --------------------------------------------------------------------------- #

def _native_crash_block(pad_to, stamp="2026-07-13 15:24:50,000"):
    """A realistic deep native-load traceback, in the REAL log-line format the
    digest's own parser keys off (a timestamped LEVEL line starts a record, and
    unprefixed lines continue it) - otherwise this would parse as one anonymous
    blob and never exercise the multi-record path.

    The actionable part (the innermost exception type + message) is at the END,
    as it always is in a traceback."""
    head = (f"{stamp} ERROR   localm: model load failed\n"
            "Traceback (most recent call last):\n")
    filler = "".join(
        f'  File "localm/inference/backends/llamacpp/_runner.py", line {i}, '
        f"in _load_native\n    self._lib.llama_load_model_from_file(path)\n"
        for i in range(400))
    tail = ("RuntimeError: Native llama runtime failed to load: [WinError 2] "
            "The system cannot find the file specified\n")
    block = head + filler + tail
    assert len(block) > pad_to, "the fixture must exceed the budget to be the case"
    return block


def test_single_oversized_error_is_truncated_not_dropped_whole():
    """THE REGRESSION. One error larger than the budget must still surface its
    most actionable part. Dropping it whole leaves the digest with nothing but
    "(1 earlier error(s) omitted)" - strictly worse than the tail behaviour this
    code replaced, and useless for the crash it exists to report."""
    from localm._log_digest import build_digest

    block = _native_crash_block(6000)
    digest = build_digest(block, max_chars=6000)

    assert "RuntimeError: Native llama runtime failed to load" in digest, (
        "the entire crash error was dropped - the bug report carries no error "
        f"at all (digest is {len(digest)} chars)")
    assert "WinError 2" in digest, "the innermost, most actionable detail was lost"
    assert len(digest) <= 6000, f"the digest blew its budget ({len(digest)} chars)"


def test_truncated_error_says_it_was_truncated():
    """Rule 5: a cut-down error must not look like the whole error - the reader
    has to know bytes are missing, or they will diagnose from a partial trace
    believing it complete."""
    from localm._log_digest import build_digest

    digest = build_digest(_native_crash_block(6000), max_chars=6000)
    assert "truncat" in digest.lower(), (
        "the error was silently cut with no marker saying so")


def test_oversized_error_keeps_its_tail_not_its_head():
    """The tail is the actionable end of a traceback (the innermost exception),
    and is what the _recent_log_tail this replaced always surfaced. A head-first
    truncation would keep 6000 chars of stack frames and cut the one line that
    names the failure."""
    from localm._log_digest import build_digest

    digest = build_digest(_native_crash_block(6000), max_chars=6000)
    assert digest.rstrip().endswith(
        "The system cannot find the file specified"), (
        "truncation kept the head; the innermost exception was cut")


def test_newest_oversized_error_wins_over_older_ones():
    """With several errors and no room, the MOST RECENT is the one to keep
    (partially): it is the one the user is reporting."""
    from localm._log_digest import build_digest

    from localm._log_digest import is_error_record, parse_records

    old = ("2026-07-13 15:20:00,000 ERROR   localm: an older unrelated failure\n"
           "ValueError: ancient\n")
    raw = old + _native_crash_block(6000)
    assert sum(1 for r in parse_records(raw) if is_error_record(r)) == 2, (
        "the fixture must produce TWO error records or this proves nothing")

    digest = build_digest(raw, max_chars=6000)
    assert "RuntimeError: Native llama runtime failed to load" in digest
    assert "ancient" not in digest, "an older error displaced the newest one"
    assert "omitted" in digest, "the dropped older error must still be declared"


# NEGATIVE CASES ------------------------------------------------------------ #

def test_errors_that_fit_are_still_kept_whole():
    """NEGATIVE CASE: the fix must not start truncating errors that fit - the
    path the log-digest tests already cover."""
    from localm._log_digest import build_digest

    raw = "ERROR localm: boom\nValueError: a small error\n"
    digest = build_digest(raw, max_chars=6000)
    assert "ValueError: a small error" in digest
    assert "truncat" not in digest.lower(), "a fitting error must not be cut"


def test_several_fitting_errors_are_all_kept_and_none_marked_truncated():
    """NEGATIVE CASE: multiple errors that collectively fit stay intact."""
    from localm._log_digest import build_digest

    raw = "".join(f"ERROR localm: e{i}\nValueError: small-{i}\n" for i in range(3))
    digest = build_digest(raw, max_chars=6000)
    for i in range(3):
        assert f"small-{i}" in digest
    assert "truncat" not in digest.lower()


def test_budget_is_never_exceeded_by_the_truncation_path():
    """NEGATIVE CASE: truncating must respect the budget it exists to satisfy -
    across a range of budgets, including tight ones."""
    from localm._log_digest import build_digest

    block = _native_crash_block(6000)
    for budget in (200, 500, 1000, 2500, 6000, 9000):
        digest = build_digest(block, max_chars=budget)
        assert len(digest) <= budget, (
            f"budget {budget} exceeded: digest is {len(digest)} chars")


# --------------------------------------------------------------------------- #
#  A hang trace must belong to the run being reported                          #
# --------------------------------------------------------------------------- #

def _write_hang(home, pid, age_s, text="MainThread stack:\n  frozen_call()\n"):
    """A hang trace exactly as debuglog.hang_trace_path() names it:
    hang_<date>_<pid>.log."""
    d = home / "logs"
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(time.time() - age_s))
    p = d / f"hang_{stamp}_{pid}.log"
    p.write_text(text, encoding="utf-8")
    old = time.time() - age_s
    os.utime(p, (old, old))
    return p


def test_stale_hang_trace_from_a_previous_run_is_not_attached(tmp_path):
    """THE REGRESSION. A freeze captured weeks ago by a long-dead process must
    not be attached to today's unrelated report. The report renders it under
    "## Server hang trace (event-loop stall)" asserting "the server froze", which
    turns an old, irrelevant trace into a confident false diagnosis."""
    from localm import bugreport

    _write_hang(tmp_path, pid=999001, age_s=30 * 24 * 3600)   # a month old
    assert bugreport._recent_hang_traces(tmp_path, pid=os.getpid()) == "", (
        "a month-old freeze from a different run was attached to this report")


def test_hang_trace_from_this_run_is_still_attached(tmp_path):
    """NEGATIVE CASE, the important one: the fix must not throw the baby out. A
    freeze captured by THIS run is exactly what diagnoses an "it hung" report and
    must still be attached."""
    from localm import bugreport

    _write_hang(tmp_path, pid=os.getpid(), age_s=5)
    out = bugreport._recent_hang_traces(tmp_path, pid=os.getpid())
    assert "frozen_call" in out, "this run's own freeze trace was dropped"


def test_hang_trace_from_the_crashed_run_is_attached_on_recovery(tmp_path):
    """NEGATIVE CASE: the crash-recovery path reports a PRIOR run, so it must
    attach THAT run's hang trace (matched by its pid, the way _recent_log_tail
    already matches its log), not this recovering process's."""
    from localm import bugreport

    _write_hang(tmp_path, pid=4242, age_s=30, text="stack:\n  crashed_run_call()\n")
    out = bugreport._recent_hang_traces(tmp_path, pid=4242)
    assert "crashed_run_call" in out


def test_a_reused_pid_on_an_old_trace_is_not_mistaken_for_this_run(tmp_path):
    """NEGATIVE CASE: pids are reused, so a pid match alone cannot prove a trace
    belongs to this run. An ancient trace carrying our own pid must still be
    rejected on age."""
    from localm import bugreport

    _write_hang(tmp_path, pid=os.getpid(), age_s=30 * 24 * 3600)
    assert bugreport._recent_hang_traces(tmp_path, pid=os.getpid()) == "", (
        "an ancient trace was attached just because a dead process once had our pid")


def test_recent_trace_from_another_live_run_is_not_attached(tmp_path):
    """NEGATIVE CASE: recency alone is not enough either - a freeze in a DIFFERENT
    localm instance minutes ago is not this report's problem."""
    from localm import bugreport

    _write_hang(tmp_path, pid=999002, age_s=10)
    assert bugreport._recent_hang_traces(tmp_path, pid=os.getpid()) == ""


def test_no_hang_files_is_still_empty_and_never_raises(tmp_path):
    """NEGATIVE CASE: the no-freeze case (the overwhelming majority) is unchanged
    and must never raise."""
    from localm import bugreport

    assert bugreport._recent_hang_traces(tmp_path, pid=os.getpid()) == ""
    assert bugreport._recent_hang_traces(tmp_path / "nonexistent", pid=1) == ""


# --------------------------------------------------------------------------- #
#  The watchdog probes the port the restart actually binds                     #
# --------------------------------------------------------------------------- #

def _port_from_argv(argv):
    """The port a click command would resolve from *argv* (last -p/--port wins,
    which is click's own precedence), or None when no port token is present -
    exactly what the re-exec'd process passes to pick_port()."""
    port = None
    for i, tok in enumerate(argv):
        if tok in ("-p", "--port") and i + 1 < len(argv):
            port = int(argv[i + 1])
        elif tok.startswith("--port="):
            port = int(tok.split("=", 1)[1])
    return port


@pytest.fixture()
def restart_harness(tmp_path, monkeypatch):
    """Capture what _do_restart hands to the watchdog and to execv, without
    re-execing or spawning anything."""
    import localm.config as cfg
    import localm.inference.http_server as hs
    from localm import updater
    from localm.inference import embedder as emb

    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(hs, "_engines", {})
    monkeypatch.setattr(emb, "active_requests", lambda: 1)   # skip embedder teardown

    seen = {}
    monkeypatch.setattr(updater, "spawn_health_watchdog",
                        lambda **kw: (seen.__setitem__("watchdog", kw), True)[1])
    monkeypatch.setattr(
        os, "execv",
        lambda exe, argv: (seen.__setitem__("argv", argv),
                           (_ for _ in ()).throw(SystemExit(0)))[1])
    return seen


def test_restart_pins_the_port_the_watchdog_will_probe(restart_harness, monkeypatch):
    """THE REGRESSION. The user started this instance without -p while another
    localm held 8642, so pick_port() auto-bumped it to 8643 and the watchdog
    captured 8643. _restart_argv() re-execs `python -m localm <argv[1:]>` with NO
    port token, so the new process calls pick_port(None), finds 8642 free again
    (the other instance closed) and binds 8642. The watchdog polls 8643 forever,
    times out at 90s, and auto-rolls back a perfectly healthy update.

    The restart must therefore hand the new process the port the watchdog is
    about to probe, which is what _do_restart's own docstring promises ("the
    server comes back on the same port")."""
    import sys

    import localm.inference.http_server as hs

    monkeypatch.setattr(sys, "argv", ["localm", "gui", "--no-browser"])
    bound = 8643                                   # what THIS instance is on
    with pytest.raises(SystemExit):
        hs._do_restart(port=bound, update_watchdog={
            "host": "127.0.0.1", "port": bound, "scheme": "http",
            "expect_version": "9.9.9"})

    argv = restart_harness["argv"]
    assert _port_from_argv(argv) == bound, (
        f"the restart argv {argv} carries no port, so the new process re-runs "
        "pick_port() and can bind a different port than the watchdog probes")
    assert restart_harness["watchdog"]["port"] == bound


def test_restarted_process_binds_exactly_the_probed_port(restart_harness, monkeypatch):
    """The same defect proven through the REAL pick_port(), which is what
    actually decides the new process's port - not just the presence of a token.

    Reproduces the drift: the default 8642 is free (the other instance closed),
    so pick_port(None) - what an argv carrying no port token produces - returns
    8642 while the watchdog probes 8643. They must agree."""
    import sys

    from localm.config import PORT_RANGE, pick_port
    import localm.inference.http_server as hs

    monkeypatch.setattr(sys, "argv", ["localm", "gui"])
    bound = PORT_RANGE[0] + 1                      # auto-bumped off the default
    with pytest.raises(SystemExit):
        hs._do_restart(port=bound, update_watchdog={
            "host": "127.0.0.1", "port": bound, "scheme": "http",
            "expect_version": "9.9.9"})

    requested = _port_from_argv(restart_harness["argv"])
    new_port, _busy = pick_port(requested, host="127.0.0.1")
    assert new_port == restart_harness["watchdog"]["port"], (
        f"the restarted process binds {new_port} but the watchdog probes "
        f"{restart_harness['watchdog']['port']} - it will never answer, and a "
        "healthy update gets auto-rolled back")


# NEGATIVE CASES ------------------------------------------------------------ #

def test_an_explicit_user_port_is_not_duplicated_or_contradicted(restart_harness,
                                                                 monkeypatch):
    """NEGATIVE CASE: when the user DID pass -p, the resolved port is the same
    one, and the argv must still resolve to it (click takes the last occurrence,
    so an appended token is consistent, not contradictory)."""
    import sys

    import localm.inference.http_server as hs

    monkeypatch.setattr(sys, "argv", ["localm", "serve", "mymodel", "-p", "9000"])
    with pytest.raises(SystemExit):
        hs._do_restart(port=9000, update_watchdog={
            "host": "127.0.0.1", "port": 9000, "scheme": "http",
            "expect_version": "9.9.9"})
    argv = restart_harness["argv"]
    assert _port_from_argv(argv) == 9000
    assert argv[:5] == [sys.executable, "-m", "localm", "serve", "mymodel"], (
        "the original command line must be preserved")


def test_plain_restart_without_an_update_also_keeps_its_port(restart_harness,
                                                             monkeypatch):
    """The /v1/server/restart button has the same port-drift bug (its docstring
    makes the same 'comes back on the same port' promise), just without a
    watchdog to turn it into a rollback: today a plain restart can silently move
    the server, stranding the user's open GUI tab on a dead port."""
    import sys

    import localm.inference.http_server as hs

    monkeypatch.setattr(sys, "argv", ["localm", "gui"])
    with pytest.raises(SystemExit):
        hs._do_restart(port=8677)
    assert _port_from_argv(restart_harness["argv"]) == 8677
    assert "watchdog" not in restart_harness, (
        "a plain restart must not spawn an update watchdog")


def test_restart_without_a_known_port_is_unchanged(restart_harness, monkeypatch):
    """NEGATIVE CASE: a bare create_app() harness that never advertised has no
    instance_port, so there is no port to pin - the restart must proceed
    unchanged rather than inventing one."""
    import sys

    import localm.inference.http_server as hs

    monkeypatch.setattr(sys, "argv", ["localm", "gui"])
    with pytest.raises(SystemExit):
        hs._do_restart()
    assert _port_from_argv(restart_harness["argv"]) is None
    assert restart_harness["argv"][:4] == [sys.executable, "-m", "localm", "gui"]
