# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background shell execution (run_shell_background / check_shell_job /
kill_shell_job) and the generic job registry behind them.

These drive the REAL tools against REAL OS processes, so the two properties that
matter - a kill reaping the process TREE, and the buffer staying bounded - are
exercised for real. The only mock here is a spy on ShellJob's constructor, used
to assert the argv-vs-shell ROUTING decision.
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
    with two separately-quoted tokens. That is ``run_shell`` behaviour, not
    something the background variant introduces.
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
    # A zombie counts as not running.
    return True


def _unexplained_taskkill_failures(warnings: list) -> list:
    """taskkill "exited" warnings whose reason is NOT the known-benign race: a
    descendant that legitimately exits between taskkill's tree snapshot and
    taskkill reaching that specific pid, reported as "There is no running
    instance of the task" even though the whole tree ends up fully dead.
    Anything else (access-denied, an unseen reason) counts as unexplained.
    """
    benign = "there is no running instance of the task"
    return [w for w in warnings if "taskkill exited" in w and benign not in w.lower()]


# --------------------------------------------------------------------------- #
#  Poll mid-run / poll after completion                                        #
# --------------------------------------------------------------------------- #

def test_poll_mid_run_reports_still_running(tmp_path, _py):
    res = tool_run_shell_background(
        tmp_path, _py("import time; print('booted'); time.sleep(30)"))
    assert res.ok, res.output
    job_id = _job_id(res)

    # Wait for the first output so the job is mid-run.
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
    # A finished job that failed does not report ok.
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

    # The OS process is gone, not merely marked dead.
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
    # The natural exit code survives a late kill.
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
    assert "kill_shell_job" in rejected.output
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
#  Registry generality - a second job KIND fits without reshaping anything     #
# --------------------------------------------------------------------------- #

class _FakeAgentJob(BackgroundJob):
    """A non-shell job kind, standing in for a background sub-agent.

    Has no process, no exit code and no stdout, so a registry that only works
    for ShellJob fails these tests.
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

    # A full agent quota does not block a shell job.
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

    The drained job is NOT the oldest. If it were, "evict drained first" and
    plain "evict oldest first" would pick the same victim.
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

    # The table only grows on submit, so that is where pruning runs. Trigger
    # one so the eviction order is exercised. 3 finished, keep 2 -> 1 goes.
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

    # A bare on-PATH command with no shell metacharacters reaches the OS as an
    # ARGUMENT LIST, never wrapped in a shell.
    plain = "tasklist" if sys.platform == "win32" else "env"
    res = tool_run_shell_background(tmp_path, plain)
    assert res.ok, res.output
    assert seen["argv"] == [plain], seen["argv"]
    assert seen["argv"] == _shell_argv(plain), "diverged from run_shell's routing"

    # Shell metacharacters route to the platform shell. The launch form differs
    # by platform: a raw command-line STRING on Windows, an argv list on POSIX,
    # so compare against _shell_argv rather than assuming a list here.
    piped = f"{plain} | more" if sys.platform == "win32" else f"{plain} | cat"
    res2 = tool_run_shell_background(tmp_path, piped)
    assert res2.ok, res2.output
    assert seen["argv"] == _shell_argv(piped)
    launched = seen["argv"]
    first = launched.split()[0] if isinstance(launched, str) else launched[0]
    assert first in ("cmd", "/bin/sh")


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
    """A shareable, shell-less session must lose the whole shell family."""
    from localm.plugins.coder.agent.constants import expand_shell_disable
    expanded = expand_shell_disable(frozenset({"run_shell"}))
    for name in ("run_shell", "run_shell_background",
                 "check_shell_job", "kill_shell_job"):
        assert name in expanded, f"{name} stayed enabled after run_shell was disabled"
    # Unrelated disables are untouched.
    assert expand_shell_disable(frozenset({"fetch_url"})) == frozenset({"fetch_url"})


def test_disabling_the_shell_family_yields_a_clean_system_prompt(tmp_path):
    """The prompt builders must expand the shell family the same way the Agent
    does: build_system_prompt(disabled_tools={"run_shell"}) must not advertise
    run_shell_background, whose NAME contains 'run_shell'.
    """
    from localm.plugins.coder.prompts import (
        build_subagent_system_prompt, build_system_prompt,
    )
    # This test's own NAME must not contain "run_shell": pytest derives tmp_path
    # from it and the sub-agent prompt embeds the absolute cwd, so such a name
    # would match on the directory rather than on any tool doc.
    cwd = tmp_path / "proj"
    cwd.mkdir()
    off = frozenset({"run_shell"})
    for prompt in (build_system_prompt(cwd, model_name="generic",
                                       disabled_tools=off),
                   build_subagent_system_prompt(cwd, "helper",
                                                disabled_tools=off)):
        assert "run_shell" not in prompt
        assert "check_shell_job" not in prompt
        assert "kill_shell_job" not in prompt


def test_background_tools_are_not_available_to_restricted_sessions():
    from localm.plugins.coder.tools import SAFE_RESTRICTED_TOOLS
    for name in ("run_shell_background", "check_shell_job", "kill_shell_job"):
        assert name not in SAFE_RESTRICTED_TOOLS


def test_concurrent_polls_are_safe(tmp_path, _py):
    """check_shell_job is non-destructive, so the agent may batch several polls
    into ONE parallel tool batch. Prove concurrent polls neither raise nor
    disturb the job.

    Every read is under a lock and none mutate job state: registry.get() takes
    the registry lock, status() the job lock, and output() each ring's own lock.
    (Note it is NOT _poll that protects this - check_shell_job never polls the
    process; only the watcher thread and kill() do, both under the job lock.)
    """
    res = tool_run_shell_background(
        tmp_path, _py("import time\nfor i in range(40):\n"
                      "    print('line', i)\n    time.sleep(0.05)"))
    job_id = _job_id(res)

    results, errors = [], []

    def _poll_many():
        try:
            for _ in range(15):
                results.append(tool_check_shell_job(tmp_path, job_id).output)
        except Exception as e:                      # noqa: BLE001 - recorded, asserted below
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=_poll_many) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == [], f"concurrent polls raised: {errors}"
    assert len(results) == 8 * 15
    assert all("<job>" in r and job_id in r for r in results), "a poll returned a malformed body"

    # The job is unharmed by being polled 120 times concurrently: it still
    # finishes normally with its real exit code and its output intact.
    assert _wait_for(
        lambda: "<state>done</state>" in tool_check_shell_job(tmp_path, job_id).output,
        timeout=60)
    final = tool_check_shell_job(tmp_path, job_id)
    assert "<exit_code>0</exit_code>" in final.output
    assert "line 39" in final.output


def test_all_background_tools_are_unscoped():
    """A path-arg check cannot confine arbitrary code, so none of these are
    scope-confined - that is deliberate and must stay explicit, not accidental."""
    from localm.plugins.coder.agent.constants import _INTENTIONALLY_UNSCOPED
    for name in ("run_shell_background", "check_shell_job", "kill_shell_job"):
        assert name in _INTENTIONALLY_UNSCOPED


def test_starting_and_killing_are_gated_but_polling_is_not(tmp_path):
    """Starting a job is arbitrary code execution and killing one tears down a
    process tree, so both must be confirmed. check_shell_job only reads a status
    field and an output buffer, and is not gated.

    Drives the REAL dispatch gate rather than reading ToolDef.destructive back,
    so the gate is observed rather than the declaration. Every call is REJECTED,
    so nothing is started or killed.
    """
    from localm.plugins.coder.agent import Agent
    from localm.plugins.coder.parser import ToolCall

    class _StubBackend:
        model_id = "stub-model"
        native_tools = False

        def set_tools(self, defs):
            pass

    asked: list = []

    def _recording_confirm(call):
        asked.append(call.name)
        return False                      # reject: nothing may actually run

    agent = Agent(_StubBackend(), cwd=tmp_path, auto_approve=False,
                  confirm_handler=_recording_confirm)

    def _dispatch(name, **args):
        return agent._execute_tool(
            ToolCall(name=name, args=args, raw="", start=0, end=0),
            interactive=False)

    # Reading a job's status must reach the tool WITHOUT a confirmation.
    res = _dispatch("check_shell_job", job_id="job_nonexistent")
    assert asked == [], "polling a job asked for confirmation"
    assert "Rejected by user" not in res.output
    assert "job_nonexistent" in res.output   # it really ran and reported

    # Starting and killing must BOTH be stopped at the gate.
    started = _dispatch("run_shell_background", command="echo should-not-run")
    assert asked == ["run_shell_background"]
    assert "Rejected by user" in started.output
    assert not get_registry().running(), "a rejected start still spawned a job"

    killed = _dispatch("kill_shell_job", job_id="job_nonexistent")
    assert asked == ["run_shell_background", "kill_shell_job"]
    assert "Rejected by user" in killed.output


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
    # The tool surface says output was dropped rather than presenting the tail
    # as the whole story.
    from localm.plugins.coder.tools.shell import _render_job
    text, _summary, _trunc, _st = _render_job(job)
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
    # non-zero exit rather than a launch error. Neither reads as ok.
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
#  Honesty of reporting                                                        #
# --------------------------------------------------------------------------- #

class _FakeShellJob(_FakeAgentJob):
    """A process-less job of the SHELL kind, for registry-level retention tests.

    The retention rules under test are registry bookkeeping, not process
    lifecycle.
    """

    kind = "shell"


class _UnkillableJob(BackgroundJob):
    """A job that nothing can stop.

    kill() signals this failure by RETURN VALUE ("kill FAILED - ..."), never by
    raising.
    """

    kind = "shell"

    def __init__(self, label="stubborn"):
        super().__init__(label)
        self.start_watcher()

    def _poll(self):
        return None                     # never exits, however hard we try

    def _terminate(self, *, force: bool) -> None:
        pass                            # signals go nowhere


class _RaisingKillJob(_UnkillableJob):
    """A job whose termination blows up rather than failing quietly."""

    def _terminate(self, *, force: bool) -> None:
        raise OSError("cannot signal this process")


@pytest.fixture
def _fast_kill(monkeypatch):
    """Shrink the kill grace so the FAILURE paths are quick to exercise."""
    monkeypatch.setattr(bg, "_KILL_GRACE", 0.05)
    monkeypatch.setattr(bg, "_POLL_INTERVAL", 0.01)


# -- a failed kill at exit is not counted as a success ----------------------- #

def test_shutdown_does_not_count_a_failed_kill_as_a_success(
        make_registry, capsys, _fast_kill):
    """This is the atexit hook, so it is the last chance to mention an orphan."""
    reg = make_registry(kind_caps={"shell": 4})
    stubborn = reg.submit(_UnkillableJob, kind="shell")

    assert reg.shutdown_all() == 0, (
        "a kill that FAILED was counted as a job successfully killed - the exit "
        "hook then reports a clean shutdown while the process lives on")
    err = capsys.readouterr().err
    assert stubborn.id in err and "still running" in err, (
        f"the surviving job was never reported at exit; stderr was: {err!r}")


def test_shutdown_reports_a_kill_that_raised_instead_of_discarding_it(
        make_registry, capsys, _fast_kill):
    """An exception here means we do not even KNOW whether the process died."""
    reg = make_registry(kind_caps={"shell": 4})
    job = reg.submit(_RaisingKillJob, kind="shell")

    assert reg.shutdown_all() == 0
    err = capsys.readouterr().err
    assert job.id in err and "OSError" in err, (
        f"the exception that stopped the kill was discarded; stderr: {err!r}")

    # kill() raised before it could finish the job, so it is still "running" and
    # its watcher would spin on. Retire it by hand so no busy thread is left.
    with job._lock:
        job._finish("failed", None, error="retired by the test")


def test_shutdown_still_counts_a_kill_that_WORKED_and_stays_quiet(
        tmp_path, make_registry, capsys):
    """A kill that WORKS is counted and prints nothing, so ``shutdown_all() ==
    0`` cannot be satisfied by a broken counter and "something was printed"
    cannot be satisfied by code that warns unconditionally.
    """
    reg = make_registry(kind_caps={"shell": 4})
    job = reg.submit(lambda: ShellJob(
        _argv("import time; time.sleep(120)"), tmp_path, label="normal"),
        kind="shell")
    assert _pid_alive(job.pid)

    assert reg.shutdown_all() == 1, "a kill that worked must still be counted"
    assert _wait_for(lambda: not _pid_alive(job.pid), timeout=15)
    err = capsys.readouterr().err
    assert "WARNING" not in err, (
        f"a clean shutdown reported a failure it did not have: {err!r}")


def test_exit_teardown_says_it_is_stopping_jobs(tmp_path, make_registry, capsys):
    """Bounded, but a silent multi-second pause at exit reads as a hang."""
    reg = make_registry(kind_caps={"shell": 4})
    reg.submit(lambda: ShellJob(_argv("import time; time.sleep(120)"),
                                tmp_path, label="slow"), kind="shell")
    reg.shutdown_all()
    assert "stopping 1 background job" in capsys.readouterr().err


def test_exit_teardown_is_silent_when_there_is_nothing_to_stop(make_registry, capsys):
    """Fires-control: the notice must not print on every clean exit."""
    reg = make_registry()
    assert reg.shutdown_all() == 0
    assert capsys.readouterr().err == ""


# -- one kind's undrained pile-up must not evict another kind's result ------- #

def test_one_kind_cannot_evict_another_kinds_uncollected_completion(make_registry):
    """The agent kind's completions ARE drained, while nothing drains the shell
    kind at all (both drain call sites filter kind="agent"). Against ONE global
    budget the undrained shell pile-up would evict the sub-agent completion the
    parent is about to absorb, and absorption is drain-only, so that child's
    summary, branch and diff would be unrecoverable.
    """
    reg = make_registry(kind_caps={"agent": 4, "shell": 50}, keep_finished=2)

    agent = reg.submit(_FakeAgentJob, kind="agent")
    agent.finish_now("the child's summary")
    assert _wait_for(lambda: agent.state == "done")

    # Flood the OTHER kind. Nothing drains these, so they stay undrained and
    # compete on submit order alone.
    for i in range(8):
        job = reg.submit(_FakeShellJob, kind="shell")
        job.finish_now(f"s{i}")
        assert _wait_for(lambda j=job: j.state == "done")

    drained = reg.drain_finished(kind="agent")
    assert [j["id"] for j in drained] == [agent.id], (
        "the sub-agent completion was evicted by unrelated SHELL jobs before the "
        "parent's turn-boundary drain could collect it - its payload is lost")
    assert drained[0]["result"] == {"summary": "the child's summary"}
    assert reg.take_dropped_undrained("agent") == 0


def test_retention_is_budgeted_per_kind(make_registry):
    """Each kind keeps its OWN budget of 2, so neither starves the other.

    Under one shared budget of 2 the interleaved stream below leaves a single
    completion of each kind; per kind it leaves two of each.
    """
    reg = make_registry(kind_caps={"agent": 50, "shell": 50}, keep_finished=2)
    for i in range(5):
        for factory, kind in ((_FakeAgentJob, "agent"), (_FakeShellJob, "shell")):
            job = reg.submit(factory, kind=kind)
            job.finish_now(f"r{i}")
            assert _wait_for(lambda j=job: j.state == "done")

    # Pruning runs only on submit, and the last few completions have not faced
    # one yet. Trigger a final prune with a job that stays RUNNING, so it is not
    # itself a candidate and the counts below are exact.
    reg.submit(_FakeAgentJob, kind="agent")

    counts: dict = {}
    for row in reg.list_status():
        if row["state"] != "running":
            counts[row["kind"]] = counts.get(row["kind"], 0) + 1

    assert counts.get("agent", 0) == 2, (
        f"the agent kind kept {counts.get('agent', 0)} completions, want its own "
        "budget of 2")
    assert counts.get("shell", 0) == 2, (
        f"the shell kind kept {counts.get('shell', 0)} completions, want its own "
        "budget of 2 - one shared budget would leave only 1")


def test_a_lost_completion_is_reported_to_its_consumer_exactly_once(make_registry):
    reg = make_registry(kind_caps={"agent": 50}, keep_finished=2)
    for i in range(6):
        job = reg.submit(_FakeAgentJob, kind="agent")
        job.finish_now(f"r{i}")
        assert _wait_for(lambda j=job: j.state == "done")

    assert reg.dropped_undrained > 0, "completions vanished without being counted"
    lost = reg.take_dropped_undrained("agent")
    assert lost == reg.dropped_undrained, "the report undercounts the real losses"
    assert reg.take_dropped_undrained("agent") == 0, (
        "the same loss was handed out twice - a turn-boundary consumer would "
        "warn about it every turn forever")
    # The cumulative total is not consumed: /bg shows it all session.
    assert reg.dropped_undrained > 0


def test_a_lost_sub_agent_completion_reaches_the_user(
        monkeypatch, capsys, make_registry):
    """A counter no product code reads is bookkeeping, not honesty."""
    from localm.plugins.coder.cli._main import _warn_unfinished_background

    reg = make_registry(kind_caps={"agent": 50}, keep_finished=1)
    monkeypatch.setattr(bg, "_registry", reg)
    for i in range(4):
        job = reg.submit(_FakeAgentJob, kind="agent")
        job.finish_now(f"r{i}")
        assert _wait_for(lambda j=job: j.state == "done")
    assert reg.dropped_undrained > 0

    _warn_unfinished_background(None)
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "discarded" in text and "lost" in text, (
        f"a lost sub-agent completion was never surfaced to the user: {text!r}")


def test_the_turn_boundary_note_tells_the_MODEL_what_it_lost(
        monkeypatch, make_registry):
    """The other product surface: what the parent agent puts in front of the model
    at the top of its turn, which is where a lost delegation actually matters."""
    from localm.plugins.coder.agent.persistence import _PersistenceMixin

    reg = make_registry(kind_caps={"agent": 50}, keep_finished=1)
    monkeypatch.setattr(bg, "_registry", reg)
    for i in range(4):
        job = reg.submit(_FakeAgentJob, kind="agent")
        job.finish_now(f"r{i}")
        assert _wait_for(lambda j=job: j.state == "done")
    assert reg.dropped_undrained > 0

    class _FakeParent:
        _error_trace: list = []

    notes = _PersistenceMixin._drain_background_agents(_FakeParent())
    assert any("discarded" in n and "lost" in n for n in notes), (
        f"the model was never told a delegation had been lost: {notes}")


def test_bg_calls_a_lost_sub_agent_a_LOSS_and_shell_pruning_HOUSEKEEPING(
        monkeypatch, capsys, make_registry):
    """/bg renders the two kinds differently. A shell completion aged out of the
    table is NOT a silent loss - check_shell_job answers "No background job with
    id ...", listing the ids that do exist."""
    from localm.plugins.coder.cli.repl import _handle_command_extended

    reg = make_registry(kind_caps={"agent": 50, "shell": 50}, keep_finished=1)
    monkeypatch.setattr(bg, "_registry", reg)
    for factory, kind in ((_FakeShellJob, "shell"), (_FakeShellJob, "shell"),
                          (_FakeShellJob, "shell")):
        job = reg.submit(factory, kind=kind)
        job.finish_now("s")
        assert _wait_for(lambda j=job: j.state == "done")

    assert reg.dropped_undrained_by_kind.get("shell", 0) > 0, "no shell drop to report"
    assert reg.dropped_undrained_by_kind.get("agent", 0) == 0

    _handle_command_extended("bg", "", None)
    shell_only = capsys.readouterr().out
    assert "aged out" in shell_only, f"shell pruning went unmentioned: {shell_only!r}"
    assert "lost" not in shell_only, (
        f"routine shell pruning was reported as a LOST result, which cries wolf "
        f"on every long session: {shell_only!r}")

    # Now lose an AGENT completion: that one IS unrecoverable and must say so.
    for i in range(3):
        job = reg.submit(_FakeAgentJob, kind="agent")
        job.finish_now(f"a{i}")
        assert _wait_for(lambda j=job: j.state == "done")
    assert reg.dropped_undrained_by_kind.get("agent", 0) > 0

    _handle_command_extended("bg", "", None)
    with_agent = capsys.readouterr().out
    assert "lost" in with_agent and "sub-agent" in with_agent, (
        f"a genuinely lost sub-agent result was not called a loss: {with_agent!r}")


def test_no_loss_means_no_scary_message(monkeypatch, capsys, make_registry):
    """Fires-control: the loss warning must not fire on an ordinary exit."""
    from localm.plugins.coder.cli._main import _warn_unfinished_background

    reg = make_registry(kind_caps={"agent": 50}, keep_finished=50)
    monkeypatch.setattr(bg, "_registry", reg)
    job = reg.submit(_FakeAgentJob, kind="agent")
    job.finish_now("collected normally")
    assert _wait_for(lambda: job.state == "done")

    _warn_unfinished_background(None)
    captured = capsys.readouterr()
    assert "discarded" not in (captured.out + captured.err)


# -- taskkill's exit status is reported -------------------------------------- #

def test_a_failed_taskkill_is_reported_and_falls_through(
        tmp_path, make_registry, monkeypatch):
    """taskkill signals its ORDINARY failures by exit code, not by raising: the
    direct child exiting between the poll and the call, or access-denied against
    a higher-integrity process. Returning unconditionally reports a tree kill
    that never ran and skips the fallback sweep."""
    import subprocess as _sp

    reg = make_registry()
    job = reg.submit(lambda: ShellJob(_argv("print('x')"), tmp_path, label="tk"),
                     kind="shell")
    assert _wait_for(lambda: job.state == "done")

    monkeypatch.setattr(bg.subprocess, "run", lambda argv, **kw: _sp.CompletedProcess(
        argv, 128, b"", b'ERROR: The process "1234" not found.'))
    swept = []
    monkeypatch.setattr(ShellJob, "_kill_children_via_psutil",
                        lambda self: swept.append("psutil"))
    monkeypatch.setattr(ShellJob, "_kill_via_handle",
                        lambda self, *, force: swept.append("handle"))

    job._terminate_tree_windows()

    assert any("taskkill" in w and "128" in w for w in job.warnings), (
        f"taskkill's failure was discarded; warnings were: {job.warnings}")
    assert any("not found" in w for w in job.warnings), (
        "taskkill's stderr was dropped, so the reason is unrecoverable")
    assert swept, "a failed taskkill must still fall through to the fallback kill"


def test_a_taskkill_that_SUCCEEDED_is_silent(tmp_path, make_registry, monkeypatch):
    """Fires-control: the warning keys on the exit code, not on every call."""
    import subprocess as _sp

    reg = make_registry()
    job = reg.submit(lambda: ShellJob(_argv("print('x')"), tmp_path, label="tk"),
                     kind="shell")
    assert _wait_for(lambda: job.state == "done")

    monkeypatch.setattr(bg.subprocess, "run",
                        lambda argv, **kw: _sp.CompletedProcess(argv, 0, b"", b""))
    swept = []
    monkeypatch.setattr(ShellJob, "_kill_children_via_psutil",
                        lambda self: swept.append("psutil"))

    job._terminate_tree_windows()

    assert job.warnings == [], f"a successful taskkill warned anyway: {job.warnings}"
    assert swept == [], "a successful taskkill must not run the fallback sweep"


def test_the_benign_taskkill_descendant_race_is_not_flagged_unexplained(
        tmp_path, make_registry, monkeypatch):
    """taskkill's /T walk snapshots the tree once and terminates each pid in
    turn, so a descendant can legitimately exit in that gap. Windows then
    reports exit 255 naming that one pid as "There is no running instance of
    the task" even though the fallback sweep (asserted below) still runs and
    the tree ends up fully dead. That is the benign race, not a partial
    failure."""
    import subprocess as _sp

    reg = make_registry()
    job = reg.submit(lambda: ShellJob(_argv("print('x')"), tmp_path, label="tk"),
                     kind="shell")
    assert _wait_for(lambda: job.state == "done")

    monkeypatch.setattr(bg.subprocess, "run", lambda argv, **kw: _sp.CompletedProcess(
        argv, 255, b"",
        b'ERROR: The process with PID 35352 (child process of PID 14540) could '
        b'not be terminated. Reason: There is no running instance of the task.'))
    swept = []
    monkeypatch.setattr(ShellJob, "_kill_children_via_psutil",
                        lambda self: swept.append("psutil"))
    monkeypatch.setattr(ShellJob, "_kill_via_handle",
                        lambda self, *, force: swept.append("handle"))

    job._terminate_tree_windows()

    assert swept, "the benign race must still fall through to the fallback kill"
    assert _unexplained_taskkill_failures(job.warnings) == [], (
        f"the known-benign taskkill race was wrongly treated as unexplained: "
        f"{job.warnings}")


def test_a_genuinely_different_taskkill_failure_is_still_flagged_unexplained(
        tmp_path, make_registry, monkeypatch):
    """Fires-control for the relaxation above: an UNRELATED taskkill failure
    (e.g. access-denied) must still read as unexplained, proving the benign-
    race allowance is not a blanket bypass of the exit-code contract."""
    import subprocess as _sp

    reg = make_registry()
    job = reg.submit(lambda: ShellJob(_argv("print('x')"), tmp_path, label="tk"),
                     kind="shell")
    assert _wait_for(lambda: job.state == "done")

    monkeypatch.setattr(bg.subprocess, "run", lambda argv, **kw: _sp.CompletedProcess(
        argv, 1, b"", b"ERROR: Access is denied."))

    job._terminate_tree_windows()

    unexplained = _unexplained_taskkill_failures(job.warnings)
    assert unexplained, (
        "a genuinely different taskkill failure must still be flagged unexplained")
    assert any("Access is denied" in w for w in unexplained)


# -- escalation and the "killed" report cover the whole tree ----------------- #

def test_kill_reports_descendants_that_outlived_the_direct_child(
        tmp_path, make_registry, monkeypatch):
    """_wait_for_exit only ever observes the direct child, so a descendant that
    handles SIGTERM and then hangs satisfies it while still holding its port.

    Does NOT use _fast_kill: this kills a REAL process, and _KILL_GRACE also
    bounds the real SIGTERM-then-SIGKILL wait, so shrinking it would race the
    real kill.
    """
    reg = make_registry()
    job = reg.submit(lambda: ShellJob(_argv("import time; time.sleep(120)"),
                                      tmp_path, label="tree"), kind="shell")

    class _Zombie:
        pid = 4242

        def kill(self):
            pass                        # ignores the kill, like the real case

    monkeypatch.setattr(ShellJob, "_surviving_descendants",
                        lambda self: [_Zombie()])

    assert job.kill() == "killed"
    assert any("survived the kill" in w and "4242" in w for w in job.warnings), (
        f"a surviving descendant went unreported; warnings: {job.warnings}")


def test_kill_stays_quiet_when_the_tree_really_died(
        tmp_path, make_registry, monkeypatch):
    """Fires-control: the survivor warning must key on real survivors."""
    reg = make_registry()
    job = reg.submit(lambda: ShellJob(_argv("import time; time.sleep(120)"),
                                      tmp_path, label="tree"), kind="shell")
    monkeypatch.setattr(ShellJob, "_surviving_descendants", lambda self: [])

    assert job.kill() == "killed"
    assert [w for w in job.warnings if "survived" in w] == [], job.warnings


def test_kill_says_so_when_it_cannot_verify_the_tree(
        tmp_path, make_registry, monkeypatch):
    """psutil is an OPTIONAL dependency, so "we did not look" must never be
    reported as "the tree is clean"."""
    reg = make_registry()
    job = reg.submit(lambda: ShellJob(_argv("import time; time.sleep(120)"),
                                      tmp_path, label="tree"), kind="shell")
    monkeypatch.setattr(ShellJob, "_surviving_descendants", lambda self: None)

    assert job.kill() == "killed"
    assert any("could not verify" in w for w in job.warnings), job.warnings


def test_a_missing_psutil_and_a_failed_lookup_report_DIFFERENT_reasons(
        tmp_path, make_registry, monkeypatch):
    """Both mean "unverified", but naming the wrong cause sends whoever reads the
    warning hunting the wrong thing."""
    pytest.importorskip("psutil")
    reg = make_registry()
    # The job must be FINISHED first: the second half of this test patches
    # builtins.__import__ process-wide, which must not happen while this job's
    # watcher and reader threads are still live.
    job = reg.submit(lambda: ShellJob(_argv("print('x')"), tmp_path, label="tree"),
                     kind="shell")
    assert _wait_for(lambda: job.state == "done")

    # The lookup itself fails, with psutil perfectly well installed.
    import psutil

    def _boom(*a, **kw):
        raise RuntimeError("tree read failed")

    monkeypatch.setattr(psutil, "Process", _boom)
    job._snapshot_tree()
    assert job._tree_snapshot is None
    assert "psutil is not installed" not in (job._tree_unverified_reason or ""), (
        "a failed lookup was blamed on a missing psutil")
    assert "RuntimeError" in (job._tree_unverified_reason or "")

    # psutil genuinely absent, which is the ordinary case on a core install.
    import builtins
    real_import = builtins.__import__

    def _no_psutil(name, *a, **kw):
        if name == "psutil":
            raise ImportError("no module named psutil")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    job._snapshot_tree()
    assert job._tree_unverified_reason == "psutil is not installed"


def test_tree_snapshot_pins_a_REAL_descendant_and_clears_once_it_dies(tmp_path, _py):
    """The three tests above stub the lookup; this one proves the lookup works
    against a real grandchild, so the seam they exercise is not a fiction."""
    pytest.importorskip("psutil")
    code = (
        "import subprocess,sys,time; "
        "g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "print(g.pid, flush=True); time.sleep(120)"
    )
    res = tool_run_shell_background(tmp_path, _py(code))
    job = get_registry().get(_job_id(res))

    grandchild = None

    def _got_pid():
        nonlocal grandchild
        out, _err, _d = job.output()
        first = out.strip().split("\n")[0].strip() if out.strip() else ""
        if first.isdigit():
            grandchild = int(first)
            return True
        return False

    assert _wait_for(_got_pid, timeout=30), "grandchild never reported its pid"

    job._snapshot_tree()
    assert job._tree_snapshot, "the snapshot pinned no descendants at all"
    assert grandchild in [pid for pid, _ct in job._tree_snapshot], (
        "the real grandchild was not pinned, so a survivor could never be seen")
    assert job._surviving_descendants(), "a live descendant read as already gone"

    assert job.kill() == "killed"
    assert _wait_for(lambda: not _pid_alive(grandchild), timeout=15), (
        f"ORPHAN: grandchild {grandchild} survived the kill")
    assert job._surviving_descendants() == [], (
        "the tree is dead but the verifier still reports survivors")
    assert [w for w in job.warnings if "survived" in w] == [], job.warnings
    # On Windows this runs the real `taskkill /F /T`, so this is the only place
    # the exit-code contract is exercised against a live tree. Exit 255 naming a
    # single pid with "there is no running instance of the task" is the known
    # snapshot race and is tolerated; any other reason counts as unexplained.
    unexplained = _unexplained_taskkill_failures(job.warnings)
    assert unexplained == [], (
        f"a REAL successful tree kill reported an UNEXPLAINED taskkill "
        f"failure: {unexplained}")


# -- two independent status snapshots must not tear -------------------------- #

class _TearingJob:
    """A job whose state changes BETWEEN two status() reads.

    Stands in for the real race (the watcher thread finishing a job mid-render),
    which cannot be scheduled deterministically against a real process.
    """

    id = "job_tearing"
    kind = "shell"
    label = "flaky"
    # This stand-in is inserted straight into the registry table, which the
    # fixture sweeps on teardown: running() reads .state and pruning reads
    # .drained, so both attributes must exist or shutdown_all() raises.
    state = "done"
    drained = False

    def __init__(self):
        self.calls = 0

    def status(self) -> dict:
        self.calls += 1
        base = {"id": self.id, "kind": "shell", "label": self.label,
                "started_at": 0.0, "elapsed": 1.0, "error": None,
                "warnings": [], "pid": 1234}
        if self.calls == 1:
            return {**base, "state": "running", "finished_at": None, "result": None}
        return {**base, "state": "done", "finished_at": 1.0,
                "result": {"exit_code": 1}}

    def output(self):
        return ("", "", 0)


def test_check_takes_exactly_one_status_snapshot(tmp_path, make_registry, monkeypatch):
    """A second, independent read can disagree with the body already rendered:
    the model is handed a result that reads "still running" but is flagged as a
    failure, and that failure feeds the consecutive-failure circuit breaker."""
    job = _TearingJob()
    reg = make_registry()
    monkeypatch.setattr(bg, "_registry", reg)
    with reg._lock:
        reg._jobs[job.id] = job

    res = tool_check_shell_job(tmp_path, job.id)

    assert job.calls == 1, (
        f"check_shell_job read the status {job.calls} times; the second read can "
        "describe a different job than the one it rendered")
    assert "<state>running</state>" in res.output
    assert res.ok, "the body says running while ok says the job failed"


# -- the kind guard fires before the factory spawns -------------------------- #

def test_a_wrong_kind_job_is_STOPPED_not_merely_rejected(make_registry):
    """The guard fires after the factory produced a LIVE job. Rejecting without
    stopping it leaks exactly what the cap check prevents, and the leak is
    unreachable: kill_shell_job and shutdown_all only see REGISTERED jobs."""
    reg = make_registry(kind_caps={"agent": 2, "shell": 2})
    made = []

    def _factory():
        job = _FakeAgentJob()            # kind "agent"...
        made.append(job)
        return job

    with pytest.raises(JobError):
        reg.submit(_factory, kind="shell")        # ...but the slot says shell

    assert made, "the factory never ran, so this test proves nothing"
    stray = made[0]
    assert stray.terminated, "the stray job was never told to stop"
    assert _wait_for(lambda: stray.state != "running"), (
        f"the stray job is still running ({stray.state}) - it leaked")
    assert reg.get(stray.id) is None, "a rejected job must not stay registered"


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
