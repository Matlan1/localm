# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background shell execution (run_shell_background / check_shell_job /
kill_shell_job) and the generic job registry behind them.

These drive the REAL tools against REAL OS processes - the point of the feature
is process lifecycle, and a mocked subprocess would prove nothing about the two
things that actually break: whether a kill reaps the process TREE, and whether
the buffer stays bounded. The only mock here is a spy on ShellJob's constructor,
used to assert the argv-vs-shell ROUTING decision (checking a security decision,
not standing in for the thing under test).
"""

import inspect
import sys
import threading
import time

import pytest

from localm.plugins.coder import background as bg
from localm.plugins.coder.background import (
    BackgroundJob, JobCapacityError, JobError, JobRegistry, RingBuffer, ShellJob,
    get_registry, reset_registry,
)
from localm.plugins.coder.tools.shell import (
    _shell_argv, tool_check_shell_job, tool_kill_shell_job,
    tool_run_shell_background,
)

@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is a process-wide singleton; never let jobs leak between
    tests (or between xdist tests sharing a worker process)."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def make_registry():
    """Build standalone registries that are torn down with the test.

    A JobRegistry arms an atexit hook, so building them ad hoc would leak both
    live processes and hooks across the session.
    """
    import atexit
    made = []

    def _make(**kwargs):
        reg = JobRegistry(**kwargs)
        made.append(reg)
        return reg

    yield _make
    for reg in made:
        reg.shutdown_all()
        try:
            atexit.unregister(reg.shutdown_all)
        except Exception:
            pass


@pytest.fixture
def _py(tmp_path):
    """Build a command string that runs *code* with this interpreter.

    The code goes to a script file rather than ``-c "..."`` because on Windows
    these commands are routed through ``cmd /C`` (any absolute path contains
    backslashes, which force shell mode), and cmd cannot parse a command line
    with two separately-quoted tokens. That is pre-existing ``run_shell``
    behaviour, not something the background variant introduces - verified by
    running the same command string through tool_run_shell.
    """
    if sys.platform == "win32" and " " in sys.executable:
        pytest.skip(
            "interpreter path contains a space; cmd /C cannot parse the "
            "resulting command line (a run_shell quoting limitation, not a "
            "background-execution one)")
    counter = [0]

    def _make(code: str) -> str:
        counter[0] += 1
        name = f"_bgjob_{counter[0]}.py"
        (tmp_path / name).write_text(code, encoding="utf-8")
        # Bare interpreter path + a cwd-relative script: one unquoted token each,
        # which cmd /C and /bin/sh both handle.
        return f"{sys.executable} -u {name}"

    return _make


def _wait_for(predicate, timeout=20.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _pid_alive(pid: int) -> bool:
    """Ground truth from the OS, not from our own job bookkeeping."""
    import subprocess
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie is not a running process for our purposes, but our own children
    # are reaped by the registry, so anything still visible here is real.
    return True


# --------------------------------------------------------------------------- #
#  Poll mid-run / poll after completion                                        #
# --------------------------------------------------------------------------- #

def test_poll_mid_run_reports_still_running(tmp_path, _py):
    res = tool_run_shell_background(
        tmp_path, _py("import time; print('booted'); time.sleep(30)"))
    assert res.ok, res.output
    job_id = _job_id(res)

    # Wait for the first output so we are genuinely mid-run, not pre-start.
    assert _wait_for(
        lambda: "booted" in tool_check_shell_job(tmp_path, job_id).output)

    check = tool_check_shell_job(tmp_path, job_id)
    assert "<state>running</state>" in check.output
    assert "<exit_code>" not in check.output    # not finished -> no exit code
    assert check.ok                             # running is not a failure

    tool_kill_shell_job(tmp_path, job_id)


def test_poll_after_completion_has_exit_code_and_buffered_output(tmp_path, _py):
    res = tool_run_shell_background(tmp_path, _py(
        "import sys; print('to-stdout'); "
        "print('to-stderr', file=sys.stderr); sys.exit(3)"))
    job_id = _job_id(res)

    assert _wait_for(
        lambda: "<state>done</state>" in tool_check_shell_job(tmp_path, job_id).output)

    check = tool_check_shell_job(tmp_path, job_id)
    assert "<exit_code>3</exit_code>" in check.output
    assert "to-stdout" in check.output
    assert "STDERR:" in check.output and "to-stderr" in check.output
    # A finished job that failed must not report ok - same contract as run_shell.
    assert not check.ok

    # The result stays queryable after completion, repeatedly.
    again = tool_check_shell_job(tmp_path, job_id)
    assert "<exit_code>3</exit_code>" in again.output


def test_completed_job_exit_zero_reports_ok(tmp_path, _py):
    res = tool_run_shell_background(tmp_path, _py("print('fine')"))
    job_id = _job_id(res)
    assert _wait_for(
        lambda: "<state>done</state>" in tool_check_shell_job(tmp_path, job_id).output)
    check = tool_check_shell_job(tmp_path, job_id)
    assert check.ok
    assert "<exit_code>0</exit_code>" in check.output


# --------------------------------------------------------------------------- #
#  Kill: the OS process must actually be gone                                  #
# --------------------------------------------------------------------------- #

def test_kill_mid_run_actually_kills_the_os_process(tmp_path, _py):
    res = tool_run_shell_background(tmp_path, _py("import time; time.sleep(120)"))
    job_id = _job_id(res)
    job = get_registry().get(job_id)
    pid = job.pid
    assert _pid_alive(pid), "the job process should be running before the kill"

    killed = tool_kill_shell_job(tmp_path, job_id)
    assert killed.ok, killed.output
    assert "<state>killed</state>" in killed.output

    # The claim under test: the OS process is GONE, not merely marked dead.
    assert _wait_for(lambda: not _pid_alive(pid), timeout=15), (
        f"pid {pid} is still alive after kill_shell_job reported success")

    # And the job now reads as killed, not running.
    assert "<state>killed</state>" in tool_check_shell_job(tmp_path, job_id).output


def test_kill_reaps_the_whole_process_tree(tmp_path, _py):
    """A spawned build leaves orphans if only the direct child is killed."""
    code = (
        "import subprocess,sys,time; "
        "g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "print(g.pid, flush=True); time.sleep(120)"
    )
    res = tool_run_shell_background(tmp_path, _py(code))
    job_id = _job_id(res)
    job = get_registry().get(job_id)

    grandchild = None
    def _got_pid():
        nonlocal grandchild
        out, _err, _d = job.output()
        digits = out.strip().split("\n")[0].strip() if out.strip() else ""
        if digits.isdigit():
            grandchild = int(digits)
            return True
        return False

    assert _wait_for(_got_pid, timeout=30), "grandchild never reported its pid"
    assert _pid_alive(grandchild), "grandchild should be running before the kill"

    tool_kill_shell_job(tmp_path, job_id)

    assert _wait_for(lambda: not _pid_alive(job.pid), timeout=15), "child survived"
    assert _wait_for(lambda: not _pid_alive(grandchild), timeout=15), (
        f"ORPHAN: grandchild {grandchild} survived the kill - the tree was not reaped")


def test_killing_an_already_finished_job_is_not_an_error(tmp_path, _py):
    res = tool_run_shell_background(tmp_path, _py("print('quick')"))
    job_id = _job_id(res)
    assert _wait_for(
        lambda: "<state>done</state>" in tool_check_shell_job(tmp_path, job_id).output)

    killed = tool_kill_shell_job(tmp_path, job_id)
    assert killed.ok
    assert "already" in killed.output
    # The natural exit code survives - a late kill must not rewrite history.
    assert "<exit_code>0</exit_code>" in killed.output


def test_registry_shutdown_kills_running_jobs(tmp_path, _py):
    """Nothing the coder started may outlive the process (orphan prevention)."""
    res = tool_run_shell_background(tmp_path, _py("import time; time.sleep(120)"))
    pid = get_registry().get(_job_id(res)).pid
    assert _pid_alive(pid)

    assert get_registry().shutdown_all() == 1
    assert _wait_for(lambda: not _pid_alive(pid), timeout=15)


# --------------------------------------------------------------------------- #
#  Concurrency cap                                                             #
# --------------------------------------------------------------------------- #

def test_concurrency_cap_rejects_with_a_clear_error(tmp_path, _py):
    reg = get_registry()
    assert reg.cap_for("shell") == 4, "default shell cap changed - update this test"

    ids = []
    for _ in range(4):
        res = tool_run_shell_background(tmp_path, _py("import time; time.sleep(60)"))
        assert res.ok, res.output
        ids.append(_job_id(res))

    rejected = tool_run_shell_background(tmp_path, _py("import time; time.sleep(60)"))
    assert not rejected.ok
    assert "4/4" in rejected.output
    assert "kill_shell_job" in rejected.output      # tells the model how to recover
    # Rejected, NOT silently queued: no fifth job exists.
    assert len(reg.running("shell")) == 4

    # Freeing a slot lets the next one through.
    tool_kill_shell_job(tmp_path, ids[0])
    accepted = tool_run_shell_background(tmp_path, _py("print('slot freed')"))
    assert accepted.ok, accepted.output


def test_finished_jobs_do_not_occupy_a_slot(tmp_path, make_registry):
    reg = make_registry(kind_caps={"shell": 1})
    job = reg.submit(lambda: ShellJob(
        _argv("print('done')"), tmp_path, label="quick"), kind="shell")
    assert _wait_for(lambda: job.state == "done")
    # A finished job must not hold the slot forever.
    reg.submit(lambda: ShellJob(_argv("print('second')"), tmp_path, label="second"),
               kind="shell")
    assert len(reg.running()) <= 1


def test_cap_is_enforced_before_the_process_starts(tmp_path, make_registry):
    """The cap must reject, not spawn-then-refuse (which would leak a process)."""
    reg = make_registry(kind_caps={"shell": 1})
    reg.submit(lambda: ShellJob(
        _argv("import time; time.sleep(60)"), tmp_path, label="hog"), kind="shell")

    spawned = []
    def _factory():
        spawned.append(1)
        return ShellJob(_argv("print('x')"), tmp_path, label="nope")

    with pytest.raises(JobCapacityError):
        reg.submit(_factory, kind="shell")
    assert spawned == [], "the factory ran, so a process was spawned despite the cap"


# --------------------------------------------------------------------------- #
#  Registry generality - a second job KIND must fit without reshaping anything  #
#  (the background sub-agent PR stores its jobs in this same table)             #
# --------------------------------------------------------------------------- #

class _FakeAgentJob(BackgroundJob):
    """A non-shell job kind, standing in for a background sub-agent.

    Deliberately has no process, no exit code and no stdout: if the registry
    only works for ShellJob, these tests fail.
    """

    kind = "agent"

    def __init__(self, label="analyse auth.py"):
        super().__init__(label)
        self._done_with = None
        self.terminated = False
        self.start_watcher()

    def finish_now(self, payload):
        with self._lock:
            self._done_with = payload

    def _poll(self):
        return self._done_with

    def _result_for(self, poll_value):
        return {"summary": poll_value}

    def _terminate(self, *, force: bool) -> None:
        self.terminated = True
        self._done_with = "cancelled"


def test_registry_holds_a_non_shell_kind_with_an_opaque_payload(make_registry):
    reg = make_registry(kind_caps={"agent": 2})
    job = reg.submit(_FakeAgentJob, kind="agent")
    job.finish_now("found 2 issues")
    assert _wait_for(lambda: job.state == "done")

    st = job.status()
    # The generic record shape, with NO shell fields hardcoded in it.
    assert set(st) >= {"id", "kind", "label", "state", "started_at",
                       "finished_at", "result", "error", "warnings"}
    assert st["kind"] == "agent"
    assert st["result"] == {"summary": "found 2 issues"}   # opaque, per-kind
    assert "exit_code" not in st, "the base record must not be shell-shaped"


def test_caps_are_per_kind_not_global(tmp_path, make_registry):
    """Shell jobs and agent jobs exhaust different resources."""
    reg = make_registry(kind_caps={"shell": 4, "agent": 2})
    assert reg.cap_for("shell") == 4
    assert reg.cap_for("agent") == 2

    agents = [reg.submit(_FakeAgentJob, kind="agent") for _ in range(2)]
    with pytest.raises(JobCapacityError) as exc:
        reg.submit(_FakeAgentJob, kind="agent")
    assert "2/2" in str(exc.value)
    assert "agent" in str(exc.value)

    # A full agent quota must NOT block a shell job - that is the whole point.
    shell = reg.submit(
        lambda: ShellJob(_argv("print('unblocked')"), tmp_path, label="s"),
        kind="shell")
    assert shell.state in ("running", "done")

    for a in agents:
        a.finish_now("done")


def test_cap_check_and_insert_are_atomic_under_one_lock(make_registry):
    """No TOCTOU window: concurrent submits must not both see the free slot."""
    reg = make_registry(kind_caps={"agent": 2})
    admitted, rejected = [], []
    barrier = threading.Barrier(8)

    def _try():
        barrier.wait()                       # maximise the overlap
        try:
            admitted.append(reg.submit(_FakeAgentJob, kind="agent"))
        except JobCapacityError:
            rejected.append(1)

    threads = [threading.Thread(target=_try) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(admitted) == 2, f"cap breached: {len(admitted)} admitted, want 2"
    assert len(rejected) == 6
    for job in admitted:
        job.finish_now("done")


def test_drain_finished_returns_each_completion_exactly_once(make_registry):
    """A turn-boundary consumer needs 'everything since I last asked'."""
    reg = make_registry(kind_caps={"agent": 4})
    a, b = reg.submit(_FakeAgentJob, kind="agent"), reg.submit(_FakeAgentJob, kind="agent")

    assert reg.drain_finished() == [], "nothing has finished yet"

    a.finish_now("a-result")
    assert _wait_for(lambda: a.state == "done")
    first = reg.drain_finished()
    assert [j["id"] for j in first] == [a.id]
    assert first[0]["result"] == {"summary": "a-result"}

    # Already drained -> not handed out a second time.
    assert reg.drain_finished() == []

    b.finish_now("b-result")
    assert _wait_for(lambda: b.state == "done")
    assert [j["id"] for j in reg.drain_finished()] == [b.id]
    assert reg.drain_finished() == []

    # Draining does not remove the job: poll-by-id still works afterwards.
    assert reg.get(a.id) is not None


def test_drain_can_filter_by_kind(tmp_path, make_registry):
    reg = make_registry(kind_caps={"agent": 2, "shell": 2})
    agent = reg.submit(_FakeAgentJob, kind="agent")
    reg.submit(lambda: ShellJob(_argv("print('s')"), tmp_path, label="s"),
               kind="shell")
    agent.finish_now("done")
    assert _wait_for(lambda: agent.state == "done")

    drained = reg.drain_finished(kind="agent")
    assert [j["id"] for j in drained] == [agent.id]
    # The shell job's completion is still pending for its own consumer.
    assert reg.drain_finished(kind="agent") == []


def test_pruning_evicts_drained_jobs_before_undrained_ones(make_registry):
    """A completion nobody has collected must outlive one already handed over.

    The drained job is deliberately NOT the oldest. If it were, "evict drained
    first" and plain "evict oldest first" would pick the same victim and this
    test would pass either way - proving nothing about the ordering it names.
    """
    reg = make_registry(kind_caps={"agent": 50}, keep_finished=2)

    # oldest, and left RUNNING so the drain below cannot collect it
    oldest = reg.submit(_FakeAgentJob, kind="agent")

    collected = reg.submit(_FakeAgentJob, kind="agent")
    collected.finish_now("collected")
    assert _wait_for(lambda: collected.state == "done")
    assert [j["id"] for j in reg.drain_finished()] == [collected.id]

    # now let the oldest finish, still uncollected
    oldest.finish_now("never collected")
    assert _wait_for(lambda: oldest.state == "done")

    newest = reg.submit(_FakeAgentJob, kind="agent")
    newest.finish_now("also uncollected")
    assert _wait_for(lambda: newest.state == "done")

    # The table only GROWS on submit, so that is where pruning runs. Trigger one
    # so the eviction ORDER is actually exercised. 3 finished, keep 2 -> 1 goes.
    reg.submit(_FakeAgentJob, kind="agent")

    surviving = {j["id"] for j in reg.list_status()}
    assert collected.id not in surviving, (
        "the DRAINED job should have been evicted, even though it is not the oldest")
    assert oldest.id in surviving, (
        "the oldest job was evicted despite nobody having collected it - "
        "pruning is ordering by age, not by whether the result was handed over")
    assert newest.id in surviving
    assert reg.dropped_undrained == 0
    assert {j["id"] for j in reg.drain_finished()} == {oldest.id, newest.id}


def test_dropping_an_uncollected_completion_is_counted_not_hidden(make_registry):
    """The table must stay bounded, so once every retained completion is
    undrained something has to go - but a lost result must never look the same
    as 'nothing finished'."""
    reg = make_registry(kind_caps={"agent": 50}, keep_finished=2)
    for i in range(6):
        job = reg.submit(_FakeAgentJob, kind="agent")
        job.finish_now(f"r{i}")
        assert _wait_for(lambda j=job: j.state == "done")

    assert len(reg.list_status()) <= 3        # bounded (2 kept + the newest)
    assert reg.dropped_undrained > 0, (
        "completions were discarded without being counted anywhere")


def test_submit_rejects_a_factory_that_returns_the_wrong_kind(make_registry):
    reg = make_registry(kind_caps={"agent": 2})
    with pytest.raises(JobError):
        reg.submit(_FakeAgentJob, kind="shell")


# --------------------------------------------------------------------------- #
#  Security posture: same argv/shell routing as run_shell, no naive shell=True #
# --------------------------------------------------------------------------- #

def test_background_uses_the_same_argv_routing_as_run_shell(tmp_path, monkeypatch):
    seen = {}
    real = bg.ShellJob

    class _Spy(real):
        def __init__(self, argv, cwd, **kw):
            seen["argv"] = argv
            super().__init__(argv, cwd, **kw)

    monkeypatch.setattr(bg, "ShellJob", _Spy)

    # A bare on-PATH command with no shell metacharacters must reach the OS as an
    # ARGUMENT LIST, never wrapped in a shell (that is the property run_shell has
    # and a naive shell=True background variant would silently give up).
    plain = "tasklist" if sys.platform == "win32" else "env"
    res = tool_run_shell_background(tmp_path, plain)
    assert res.ok, res.output
    assert seen["argv"] == [plain], seen["argv"]
    assert seen["argv"] == _shell_argv(plain), "diverged from run_shell's routing"

    # Shell metacharacters -> the platform shell, same as run_shell.
    piped = f"{plain} | more" if sys.platform == "win32" else f"{plain} | cat"
    res2 = tool_run_shell_background(tmp_path, piped)
    assert res2.ok, res2.output
    assert seen["argv"] == _shell_argv(piped)
    assert seen["argv"][0] in ("cmd", "/bin/sh")


def test_background_and_blocking_shell_share_one_routing_function():
    """The security decision must live in exactly one place, so the two tools
    cannot drift apart. tools/shell.py is the only definition of it."""
    from localm.plugins.coder.tools import shell as shell_mod

    run_shell_src = inspect.getsource(shell_mod.tool_run_shell)
    background_src = inspect.getsource(shell_mod.tool_run_shell_background)
    assert "_shell_argv(command)" in run_shell_src
    assert "_shell_argv(command)" in background_src
    # Neither may hand a raw string to a shell=True subprocess.
    for src in (run_shell_src, background_src):
        assert "shell=True" not in src


def test_privacy_mode_zeroes_shell_history_env(tmp_path, _py):
    code = "import os; print('HISTFILE=' + os.environ.get('HISTFILE', 'UNSET'))"
    res = tool_run_shell_background(tmp_path, _py(code), _privacy=True)
    job_id = _job_id(res)
    assert _wait_for(
        lambda: "<state>done</state>" in tool_check_shell_job(tmp_path, job_id).output)

    out = tool_check_shell_job(tmp_path, job_id).output
    null = "NUL" if sys.platform == "win32" else "/dev/null"
    assert f"HISTFILE={null}" in out, out


def test_disabling_run_shell_also_disables_the_background_variant():
    """Otherwise a shareable, deliberately shell-less session keeps RCE."""
    from localm.plugins.coder.agent.core import _expand_shell_disable
    expanded = _expand_shell_disable(frozenset({"run_shell"}))
    for name in ("run_shell", "run_shell_background",
                 "check_shell_job", "kill_shell_job"):
        assert name in expanded, f"{name} stayed enabled after run_shell was disabled"
    # Unrelated disables are untouched.
    assert _expand_shell_disable(frozenset({"fetch_url"})) == frozenset({"fetch_url"})


def test_background_tools_are_not_available_to_restricted_sessions():
    from localm.plugins.coder.tools import SAFE_RESTRICTED_TOOLS
    for name in ("run_shell_background", "check_shell_job", "kill_shell_job"):
        assert name not in SAFE_RESTRICTED_TOOLS


def test_background_tools_are_destructive_and_unscoped():
    from localm.plugins.coder.agent.constants import _INTENTIONALLY_UNSCOPED
    from localm.plugins.coder.tools import TOOL_REGISTRY
    for name in ("run_shell_background", "check_shell_job", "kill_shell_job"):
        assert TOOL_REGISTRY[name].destructive, f"{name} must hit the confirm gate"
        assert name in _INTENTIONALLY_UNSCOPED


def test_unattended_one_shot_gate_covers_the_background_variant():
    """R19a: the CLI forces confirmation on shell execution for an unattended
    one-shot. A background variant outside that set would bypass the gate."""
    from localm.plugins.coder.agent.constants import _SHELL_EXEC_TOOLS
    from localm.plugins.coder.cli import _main

    assert _SHELL_EXEC_TOOLS == frozenset({"run_shell", "run_shell_background"})
    src = inspect.getsource(_main)
    assert "always_confirm = set(always_confirm) | set(_SHELL_EXEC_TOOLS)" in src, (
        "the R19a unattended gate no longer covers the whole shell-exec family")
    assert "always_confirm.update(_SHELL_EXEC_TOOLS)" in src, (
        "--interactive-confirm no longer covers the whole shell-exec family")


# --------------------------------------------------------------------------- #
#  Bounded memory                                                              #
# --------------------------------------------------------------------------- #

def test_ring_buffer_caps_and_counts_what_it_dropped():
    ring = RingBuffer(max_chars=100)
    for _ in range(50):
        ring.append("x" * 10)          # 500 chars into a 100-char ring
    text, dropped = ring.read()
    assert len(text) <= 100
    assert dropped == 400, dropped     # accounted for, not silently discarded


def test_ring_buffer_trims_a_single_oversized_chunk():
    """A '\\r' progress bar emits no newlines - one chunk must not grow forever."""
    ring = RingBuffer(max_chars=50)
    ring.append("a" * 500)
    text, dropped = ring.read()
    assert len(text) == 50
    assert text == "a" * 50
    assert dropped == 450


def test_chatty_process_output_stays_bounded_and_reports_the_drop(tmp_path, make_registry):
    reg = make_registry()
    job = reg.submit(lambda: ShellJob(
        _argv("[print('y' * 200) for _ in range(2000)]"),
        tmp_path, label="chatty", max_chars=5_000), kind="shell")
    assert _wait_for(lambda: job.state == "done", timeout=60)

    out, _err, dropped = job.output()
    assert len(out) <= 5_000, "buffer exceeded its cap"
    assert dropped > 0
    # The tool surface must SAY output was dropped rather than present the tail
    # as the whole story.
    from localm.plugins.coder.tools.shell import _render_job
    text, _summary, _trunc = _render_job(job)
    assert "<dropped_chars>" in text


# --------------------------------------------------------------------------- #
#  Error paths                                                                 #
# --------------------------------------------------------------------------- #

def test_unknown_job_id_lists_the_known_ids(tmp_path, _py):
    res = tool_run_shell_background(tmp_path, _py("import time; time.sleep(30)"))
    real_id = _job_id(res)

    for tool in (tool_check_shell_job, tool_kill_shell_job):
        bad = tool(tmp_path, "job_doesnotexist")
        assert not bad.ok
        assert "job_doesnotexist" in bad.output
        assert real_id in bad.output, "the error should name the ids that DO exist"

    tool_kill_shell_job(tmp_path, real_id)


def test_missing_executable_reports_a_clear_error(tmp_path):
    res = tool_run_shell_background(tmp_path, "definitely-not-a-real-binary-xyz")
    # Routed through the shell (not on PATH), so the failure shows up as a
    # non-zero exit rather than a launch error - either way it must not look ok.
    if res.ok:
        job_id = _job_id(res)
        assert _wait_for(
            lambda: "<state>done</state>"
            in tool_check_shell_job(tmp_path, job_id).output)
        assert not tool_check_shell_job(tmp_path, job_id).ok
    else:
        assert "definitely-not-a-real-binary-xyz" in res.output


def test_job_id_is_returned_immediately_without_waiting(tmp_path, _py):
    """The whole point: the call returns before the command finishes."""
    t0 = time.time()
    res = tool_run_shell_background(tmp_path, _py("import time; time.sleep(20)"))
    elapsed = time.time() - t0
    assert res.ok
    assert elapsed < 5, f"run_shell_background blocked for {elapsed:.1f}s"
    assert "<state>running</state>" in res.output
    tool_kill_shell_job(tmp_path, _job_id(res))


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #

def _job_id(result) -> str:
    import re
    m = re.search(r"<job>(job_[0-9a-f]+)</job>", result.output)
    assert m, f"no job id in: {result.output}"
    return m.group(1)


def _argv(code: str) -> list:
    return [sys.executable, "-u", "-c", code]
