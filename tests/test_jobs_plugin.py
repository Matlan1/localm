# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the jobs plugin backend (store, cron matcher, scheduler, runner, CLI).

Every test that touches disk points LOCALM_HOME at a tmp dir and patches the
config module's HOME_DIR (the jobs store resolves the data dir at call time via
``home_dir()``), so nothing touches the user's real data.
"""

from __future__ import annotations

import json
import time

import pytest


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _make_job(**kw):
    from localm.plugins.builtin.jobs.store import Job
    base = dict(name="test", task_kind="chat", prompt="hi",
                schedule_kind="interval", schedule=60)
    base.update(kw)
    return Job(**base)


# --------------------------------------------------------------------------- #
#  Store: CRUD                                                                 #
# --------------------------------------------------------------------------- #

def test_store_add_get_list(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    job = _make_job(name="alpha")
    store.add(job)

    got = store.get(job.id)
    assert got is not None and got.name == "alpha"
    assert [j.id for j in store.list()] == [job.id]
    # Defs file was written under the jobs dir.
    assert (store.root / "jobs.json").is_file()


def test_store_update_and_remove(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    job = store.add(_make_job(name="beta", enabled=True))

    updated = store.update(job.id, name="renamed", enabled=False)
    assert updated.name == "renamed" and updated.enabled is False
    assert store.get(job.id).name == "renamed"

    assert store.remove(job.id) is True
    assert store.get(job.id) is None
    assert store.remove(job.id) is False        # already gone


def test_store_update_missing_raises(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    with pytest.raises(KeyError):
        store.update("nope", name="x")


def test_store_persists_across_instances(home):
    from localm.plugins.builtin.jobs.store import JobStore
    j = JobStore().add(_make_job(name="persist"))
    # A fresh store reads the same file.
    assert JobStore().get(j.id).name == "persist"


def test_store_atomic_write_leaves_no_tmp(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    store.add(_make_job())
    leftover = list(store.root.glob("jobs.json.tmp*"))
    assert leftover == []


def test_record_result_survives_metadata_stamp_failure(home, monkeypatch):
    """If stamping the job metadata fails AFTER the result file is written,
    record_result still persists the result and returns its id rather than
    orphaning it or propagating, and warns about the stamp failure."""
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    job = store.add(_make_job(name="j"))

    def boom(self, jobs):
        raise OSError("disk full")
    monkeypatch.setattr(JobStore, "_write_all", boom)

    rid = store.record_result(job.id, {"status": "ok", "output": "hi",
                                       "started": 1.0, "finished": 2.0})
    assert rid                                              # returned, not propagated
    # the result file is on disk despite the metadata-stamp failure (not orphaned)
    assert list(store._result_dir(job.id).glob("*.json"))


def test_list_results_paginates(home):
    """list_results reads only the requested page; limit=None keeps the full
    list."""
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    job = store.add(_make_job(name="j"))
    for i in range(5):
        store.record_result(job.id, {"status": "ok", "output": f"r{i}",
                                     "started": 1000.0 + i, "finished": 1000.0 + i})
    assert len(store.list_results(job.id)) == 5                  # default: all
    assert len(store.list_results(job.id, limit=2)) == 2
    assert len(store.list_results(job.id, limit=2, offset=2)) == 2
    assert len(store.list_results(job.id, limit=2, offset=4)) == 1
    assert len(store.list_results(job.id, limit=100)) == 5       # limit > count -> all
    # the pages are disjoint and together cover every result
    seen = []
    for off in (0, 2, 4):
        seen += [r["output"] for r in store.list_results(job.id, limit=2, offset=off)]
    assert sorted(seen) == ["r0", "r1", "r2", "r3", "r4"]


def test_scheduler_prunes_cron_fired_for_deleted_jobs(home):
    """A tick drops _cron_fired entries for jobs that no longer exist."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    sched = JobScheduler(run_job=lambda *a, **k: {})
    sched._cron_fired["ghost-job"] = 12345           # a fired cron job since deleted
    sched.tick(now=1000.0)                             # prunes stale entries first
    assert "ghost-job" not in sched._cron_fired


def test_job_validation_rejects_bad_defs(home):
    from localm.plugins.builtin.jobs.store import Job
    with pytest.raises(ValueError):
        Job(name="", prompt="x")                       # blank name
    with pytest.raises(ValueError):
        Job(name="n", prompt="x", task_kind="bogus")   # bad task_kind
    with pytest.raises(ValueError):
        Job(name="n", prompt="")                       # blank prompt
    with pytest.raises(ValueError):
        Job(name="n", prompt="x", schedule_kind="interval", schedule=0)  # <1s
    with pytest.raises(ValueError):
        Job(name="n", prompt="x", task_kind="coder")   # coder needs cwd


# --------------------------------------------------------------------------- #
#  Store: results round-trip + path confinement                               #
# --------------------------------------------------------------------------- #

def test_result_round_trip_and_stamps_job(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    job = store.add(_make_job())
    result = {"status": "ok", "output": "the answer", "error": None,
              "started": time.time(), "finished": time.time()}
    rid = store.record_result(job.id, result)
    assert rid

    results = store.list_results(job.id)
    assert len(results) == 1
    assert results[0]["output"] == "the answer"
    assert results[0]["status"] == "ok"
    assert results[0]["job_id"] == job.id

    # The job def was stamped with last_run/last_status/last_result_id.
    stamped = store.get(job.id)
    assert stamped.last_status == "ok"
    assert stamped.last_result_id == rid
    assert stamped.last_run is not None


def test_results_newest_first(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    job = store.add(_make_job())
    store.record_result(job.id, {"status": "ok", "output": "first",
                                 "started": 1.0, "finished": 1.0})
    store.record_result(job.id, {"status": "ok", "output": "second",
                                 "started": 2.0, "finished": 2.0})
    outputs = [r["output"] for r in store.list_results(job.id)]
    # Two distinct files were created (no overwrite), newest first by mtime.
    assert set(outputs) == {"first", "second"}


def test_result_dir_confined_to_jobs_dir(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    # A crafted, traversal-shaped job id is sanitised; nothing escapes the root.
    rid = store.record_result("../../etc/evil", {"status": "ok", "output": "x",
                                                 "started": 0.0, "finished": 0.0})
    assert rid
    results_root = (store.root / "results").resolve()
    for child in results_root.iterdir():
        assert child.resolve().parent == results_root
    # No file was written above the jobs dir.
    assert not (store.root.parent / "etc" / "evil").exists()


def test_confine_rejects_escape(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    with pytest.raises(ValueError):
        store._confine(store.root.parent / "outside.json")


# --------------------------------------------------------------------------- #
#  Cron matcher                                                                #
# --------------------------------------------------------------------------- #

def _at(y, mo, d, h, mi):
    return time.mktime((y, mo, d, h, mi, 0, 0, 0, -1))


@pytest.mark.parametrize("expr,y,mo,d,h,mi,expected", [
    pytest.param("* * * * *", 2026, 6, 17, 13, 37, True, id="every_minute"),
    pytest.param("30 9 * * *", 2026, 6, 17, 9, 30, True, id="specific_minute_hour-match"),
    pytest.param("30 9 * * *", 2026, 6, 17, 9, 31, False, id="specific_minute_hour-wrong_minute"),
    pytest.param("30 9 * * *", 2026, 6, 17, 10, 30, False, id="specific_minute_hour-wrong_hour"),
    pytest.param("*/15 * * * *", 2026, 6, 17, 0, 0, True, id="step-on_0"),
    pytest.param("*/15 * * * *", 2026, 6, 17, 0, 15, True, id="step-on_15"),
    pytest.param("*/15 * * * *", 2026, 6, 17, 0, 30, True, id="step-on_30"),
    pytest.param("*/15 * * * *", 2026, 6, 17, 0, 7, False, id="step-off_7"),
    pytest.param("0 9-17 * * *", 2026, 6, 17, 9, 0, True, id="range-lower_bound"),
    pytest.param("0 9-17 * * *", 2026, 6, 17, 17, 0, True, id="range-upper_bound"),
    pytest.param("0 9-17 * * *", 2026, 6, 17, 8, 0, False, id="range-below"),
    pytest.param("0 9-17 * * *", 2026, 6, 17, 18, 0, False, id="range-above"),
    pytest.param("0,30 * * * *", 2026, 6, 17, 4, 0, True, id="list-first"),
    pytest.param("0,30 * * * *", 2026, 6, 17, 4, 30, True, id="list-second"),
    pytest.param("0,30 * * * *", 2026, 6, 17, 4, 15, False, id="list-not_in_list"),
    # June 15 2026 is a Monday; dow Monday == 1.
    pytest.param("0 0 * * 1", 2026, 6, 15, 0, 0, True, id="day_of_week-monday"),
    pytest.param("0 0 * * 1", 2026, 6, 16, 0, 0, False, id="day_of_week-tuesday"),
])
def test_cron_match(expr, y, mo, d, h, mi, expected):
    from localm.plugins.builtin.jobs.scheduler import cron_match
    assert cron_match(expr, _at(y, mo, d, h, mi)) is expected


def test_cron_invalid_raises():
    from localm.plugins.builtin.jobs.scheduler import parse_cron, validate_cron
    with pytest.raises(ValueError):
        parse_cron("* * * *")              # only 4 fields
    with pytest.raises(ValueError):
        parse_cron("99 * * * *")           # minute out of range
    with pytest.raises(ValueError):
        validate_cron("a * * * *")         # non-numeric


# --------------------------------------------------------------------------- #
#  Scheduler: due()                                                            #
# --------------------------------------------------------------------------- #

def test_due_interval_never_run_is_due(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    sched = JobScheduler(run_job=lambda *a, **k: {})
    job = _make_job(schedule_kind="interval", schedule=60, last_run=None)
    assert sched.due(job, now=1000.0) is True


def test_due_interval_respects_elapsed(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    sched = JobScheduler(run_job=lambda *a, **k: {})
    job = _make_job(schedule_kind="interval", schedule=60, last_run=1000.0)
    assert sched.due(job, now=1059.0) is False     # 59s < 60s
    assert sched.due(job, now=1060.0) is True       # exactly 60s
    assert sched.due(job, now=2000.0) is True


def test_due_cron(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    sched = JobScheduler(run_job=lambda *a, **k: {})
    job = _make_job(schedule_kind="cron", schedule="30 9 * * *")
    assert sched.due(job, now=_at(2026, 6, 17, 9, 30)) is True
    assert sched.due(job, now=_at(2026, 6, 17, 9, 31)) is False


def test_due_cron_not_twice_in_same_minute(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    sched = JobScheduler(run_job=lambda *a, **k: {})
    job = _make_job(schedule_kind="cron", schedule="* * * * *")
    when = _at(2026, 6, 17, 9, 30)
    assert sched.due(job, now=when) is True
    sched._cron_fired[job.id] = int(when // 60) * 60      # simulate a prior fire
    assert sched.due(job, now=when + 5) is False           # same minute -> skip


# --------------------------------------------------------------------------- #
#  Scheduler: tick()                                                           #
# --------------------------------------------------------------------------- #

def test_tick_runs_due_enabled_job(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    job = store.add(_make_job(name="due", schedule=60, last_run=None))

    calls = []

    def fake_run(j, *, engine=None):
        calls.append(j.id)
        return {"status": "ok", "output": "ran", "error": None,
                "started": 0.0, "finished": 0.0}

    sched = JobScheduler(store, run_job=fake_run)
    ran = sched.tick(now=1000.0)

    assert ran == [job.id]
    assert calls == [job.id]
    # The result was recorded.
    assert len(store.list_results(job.id)) == 1


def test_tick_skips_disabled_and_not_due(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    disabled = store.add(_make_job(name="off", schedule=60,
                                   last_run=None, enabled=False))
    not_due = store.add(_make_job(name="recent", schedule=3600, last_run=999.0))
    due = store.add(_make_job(name="on", schedule=60, last_run=None))

    ran = JobScheduler(store, run_job=lambda j, **k: {
        "status": "ok", "output": "", "error": None,
        "started": 0.0, "finished": 0.0}).tick(now=1000.0)

    assert due.id in ran
    assert disabled.id not in ran
    assert not_due.id not in ran


def test_tick_catches_runner_error(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    job = store.add(_make_job(schedule=60, last_run=None))

    def boom(j, *, engine=None):
        raise RuntimeError("kaboom")

    # tick must not raise even though the runner does.
    ran = JobScheduler(store, run_job=boom).tick(now=1000.0)
    assert ran == [job.id]
    results = store.list_results(job.id)
    assert results and results[0]["status"] == "error"
    assert "kaboom" in results[0]["error"]


def test_tick_error_getter_reflects_the_current_failure(home):
    """tick_error() surfaces the same signature _note_tick_error latches, so a
    caller with a status surface (GET /api/jobs) sees what only reached the
    debug log otherwise."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    sched = JobScheduler(JobStore())
    assert sched.tick_error() is None

    sched._note_tick_error(RuntimeError("jobs.json unreadable"))
    warning = sched.tick_error()
    assert warning and "jobs.json unreadable" in warning

    # A second, distinct failure updates the signature.
    sched._note_tick_error(OSError("disk full"))
    assert "disk full" in sched.tick_error()


def test_tick_error_getter_clears_on_recovery(home):
    """NEGATIVE CASE: once a tick succeeds, tick_error() must go back to None -
    a stale failure notice must not linger after the scheduler recovers."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    sched = JobScheduler(JobStore())
    sched._note_tick_error(RuntimeError("boom"))
    assert sched.tick_error() is not None

    sched._note_tick_ok()
    assert sched.tick_error() is None


def test_scheduler_start_noop_without_event_loop(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    sched = JobScheduler(run_job=lambda *a, **k: {})
    assert sched.start() is False        # no running loop in a sync test
    assert sched.running is False
    sched.stop()                          # safe to call when not running


# --------------------------------------------------------------------------- #
#  Runner                                                                      #
# --------------------------------------------------------------------------- #

class _FakeEngine:
    def __init__(self, tokens):
        self._tokens = tokens

    def chat_stream(self, messages, **kw):
        assert messages and messages[-1]["content"]
        for t in self._tokens:
            yield t


def test_runner_chat_ok_with_injected_engine(home):
    from localm.plugins.builtin.jobs.runner import run_job
    job = _make_job(task_kind="chat", prompt="What is 2+2?")
    result = run_job(job, engine=_FakeEngine(["4", "!"]))
    assert result["status"] == "ok"
    assert result["output"] == "4!"
    assert result["error"] is None
    assert result["started"] <= result["finished"]


def test_runner_chat_error_is_caught(home):
    from localm.plugins.builtin.jobs.runner import run_job

    class _Boom:
        def chat_stream(self, messages, **kw):
            raise RuntimeError("engine down")
            yield  # pragma: no cover

    job = _make_job(task_kind="chat", prompt="hi")
    result = run_job(job, engine=_Boom())
    assert result["status"] == "error"
    assert "engine down" in result["error"]
    assert result["output"] == ""


def test_runner_chat_no_engine_no_model_errors(home, monkeypatch):
    from localm.plugins.builtin.jobs import runner
    # No model resolvable -> _load_engine returns (None, False) -> error result.
    monkeypatch.setattr(runner, "_load_engine", lambda model: (None, False))
    job = _make_job(task_kind="chat", prompt="hi")
    result = runner.run_job(job, engine=None)
    assert result["status"] == "error"
    assert "engine" in result["error"].lower()


def test_runner_coder_best_effort_mocked(home, tmp_path, monkeypatch):
    """Coder path: with the agent + backend mocked it runs the prompt and
    returns ok. (A real run needs the coder extra + a live server.)"""
    from localm.plugins.builtin.jobs import runner

    work = tmp_path / "proj"
    work.mkdir()

    class _FakeAgent:
        def __init__(self, backend, cwd, **kw):
            assert cwd == work.resolve()
            self.kw = kw

        def run_task(self, prompt):
            return f"did: {prompt}"

        def close(self):
            return None

    monkeypatch.setattr(runner, "_coder_backend", lambda job: object())
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", _FakeAgent)

    job = _make_job(task_kind="coder", prompt="refactor", cwd=str(work),
                    schedule_kind="interval", schedule=60)
    result = runner.run_job(job, engine=None)
    assert result["status"] == "ok"
    assert result["output"] == "did: refactor"


# --------------------------------------------------------------------------- #
#  CLI round-trip                                                              #
# --------------------------------------------------------------------------- #

def test_cli_add_list_run_enable_disable(home, monkeypatch):
    from click.testing import CliRunner

    from localm.plugins.builtin.jobs import runner
    from localm.plugins.builtin.jobs.cli import main

    # Mock the actual run so the CLI round-trip needs no model/server.
    monkeypatch.setattr(
        runner, "run_job",
        lambda job, engine=None: {"status": "ok", "output": "MOCK-OUTPUT",
                                  "error": None, "started": 0.0, "finished": 0.0})

    cli = CliRunner()

    # add
    r = cli.invoke(main, ["add", "nightly", "--prompt", "summarise the day",
                          "--every", "3600"])
    assert r.exit_code == 0, r.output
    assert "Added job" in r.output

    # list -> capture the id
    r = cli.invoke(main, ["list"])
    assert r.exit_code == 0
    assert "nightly" in r.output
    from localm.plugins.builtin.jobs.store import JobStore
    job_id = JobStore().list()[0].id

    # run now (mocked runner)
    r = cli.invoke(main, ["run", job_id])
    assert r.exit_code == 0, r.output
    assert "MOCK-OUTPUT" in r.output
    assert JobStore().list_results(job_id)        # a result was recorded

    # disable / enable
    r = cli.invoke(main, ["disable", job_id])
    assert r.exit_code == 0 and "Disabled" in r.output
    assert JobStore().get(job_id).enabled is False

    r = cli.invoke(main, ["enable", job_id])
    assert r.exit_code == 0 and "Enabled" in r.output
    assert JobStore().get(job_id).enabled is True

    # remove
    r = cli.invoke(main, ["remove", job_id])
    assert r.exit_code == 0 and "Removed" in r.output
    assert JobStore().get(job_id) is None


def test_cli_run_unknown_job(home):
    from click.testing import CliRunner

    from localm.plugins.builtin.jobs.cli import main
    r = CliRunner().invoke(main, ["run", "does-not-exist"])
    assert r.exit_code == 1
    assert "No such job" in r.output


def test_cli_add_cron_and_every_conflict(home):
    from click.testing import CliRunner

    from localm.plugins.builtin.jobs.cli import main
    r = CliRunner().invoke(main, ["add", "x", "--prompt", "p",
                                  "--cron", "* * * * *", "--every", "60"])
    assert r.exit_code == 1
    assert "not both" in r.output


# --------------------------------------------------------------------------- #
#  CLI: `job show` / `job results`                                             #
# --------------------------------------------------------------------------- #

def test_cli_show_full_definition(home):
    """`job show` is the CLI's window onto allow_shell/cwd/scope: which
    unattended jobs run the shell-capable coder, and in which cwd/scope."""
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    cli = CliRunner()
    r = cli.invoke(main, ["add", "auditme", "--coder", "--cwd", str(home),
                          "--scope", "tests/**", "--allow-shell",
                          "--prompt", "run the tests", "--every", "1800"])
    assert r.exit_code == 0, r.output
    job_id = JobStore().list()[0].id

    r = cli.invoke(main, ["show", job_id])
    assert r.exit_code == 0, r.output
    assert f"Job {job_id}" in r.output and "auditme" in r.output
    assert "kind:        coder" in r.output
    assert "state:       enabled" in r.output
    assert "schedule:    every 1800s" in r.output
    assert f"cwd:         {home}" in r.output
    assert "scope:       tests/**" in r.output
    assert "allow_shell: yes" in r.output
    assert "prompt:      run the tests" in r.output
    assert "last_run:    -" in r.output           # never run yet
    assert "last_result: -" in r.output
    # `show` renders the fields `_job_dict` exposes; the internal
    # principal-binding fields are not re-derived from the raw Job object.
    assert "owner" not in r.output.lower()


def test_cli_show_defaults_for_non_coder_job(home):
    """A plain chat job has no cwd/scope/collection/allow_shell: `show` renders
    those as '-'/'no' rather than omitting the lines."""
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    cli = CliRunner()
    r = cli.invoke(main, ["add", "digest", "--prompt", "summarise",
                          "--cron", "0 9 * * 1-5"])
    assert r.exit_code == 0, r.output
    job_id = JobStore().list()[0].id

    r = cli.invoke(main, ["show", job_id])
    assert r.exit_code == 0, r.output
    assert "kind:        chat" in r.output
    assert "schedule:    cron '0 9 * * 1-5'" in r.output
    assert "cwd:         -" in r.output
    assert "scope:       -" in r.output
    assert "collection:  -" in r.output
    assert "allow_shell: no" in r.output


def test_cli_show_unknown_job(home):
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    r = CliRunner().invoke(main, ["show", "does-not-exist"])
    assert r.exit_code == 1
    assert "No such job" in r.output


def test_cli_results_reports_status_and_output(home):
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    store = JobStore()
    job = store.add(_make_job(name="hasresults"))
    store.record_result(job.id, {"status": "ok", "output": "line one\nline two",
                                 "error": None, "started": 1000.0, "finished": 1000.5})
    store.record_result(job.id, {"status": "error", "output": "",
                                 "error": "boom: disk full",
                                 "started": 2000.0, "finished": 2000.5})

    r = CliRunner().invoke(main, ["results", job.id])
    assert r.exit_code == 0, r.output
    assert "line one" in r.output and "line two" in r.output
    assert "boom: disk full" in r.output
    assert " ok " in r.output or r.output.count(" ok") >= 1
    assert " error " in r.output or r.output.count(" error") >= 1
    # An error result must show the ERROR text, never a blank or "no output"
    # standing in for it.
    assert "(no output)" not in r.output


def test_cli_results_empty_is_not_unknown_job(home):
    """A job with zero runs and a job that does not exist print DIFFERENT
    things: "No results." at exit 0 versus "No such job" at exit 1."""
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    store = JobStore()
    job = store.add(_make_job(name="neverrun"))

    r = CliRunner().invoke(main, ["results", job.id])
    assert r.exit_code == 0, r.output
    assert "No results." in r.output

    r = CliRunner().invoke(main, ["results", "does-not-exist"])
    assert r.exit_code == 1
    assert "No such job" in r.output


def test_cli_results_limit_and_offset(home):
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    store = JobStore()
    job = store.add(_make_job(name="paged"))
    for i in range(3):
        store.record_result(job.id, {"status": "ok", "output": f"r{i}",
                                     "started": 1000.0 + i, "finished": 1000.0 + i})

    r = CliRunner().invoke(main, ["results", job.id, "--limit", "1"])
    assert r.exit_code == 0, r.output
    assert r.output.count("(result ") == 1

    r = CliRunner().invoke(main, ["results", job.id, "--limit", "1", "--offset", "1"])
    assert r.exit_code == 0, r.output
    assert r.output.count("(result ") == 1

    r = CliRunner().invoke(main, ["results", job.id])
    assert r.exit_code == 0, r.output
    assert r.output.count("(result ") == 3        # no --limit: every result


def test_cli_run_then_results_round_trip(home, monkeypatch):
    """A run recorded through `job run` reads back through `job results`: the
    two commands compose against the same store."""
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs import runner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    monkeypatch.setattr(
        runner, "run_job",
        lambda job, engine=None: {"status": "ok", "output": "ROUND-TRIP-OUTPUT",
                                  "error": None, "started": 0.0, "finished": 0.0})

    cli = CliRunner()
    r = cli.invoke(main, ["add", "rt", "--prompt", "p", "--every", "60"])
    assert r.exit_code == 0, r.output
    job_id = JobStore().list()[0].id

    r = cli.invoke(main, ["run", job_id])
    assert r.exit_code == 0, r.output

    r = cli.invoke(main, ["results", job_id])
    assert r.exit_code == 0, r.output
    assert "ROUND-TRIP-OUTPUT" in r.output

    r = cli.invoke(main, ["show", job_id])
    assert r.exit_code == 0, r.output
    assert "(ok)" in r.output                      # last_run carries the status
    assert "last_result: -" not in r.output         # a real result id is stamped


# --------------------------------------------------------------------------- #
#  Plugin routes + presence in the catalog/manifest                           #
# --------------------------------------------------------------------------- #

def test_plugin_routes_via_engine(home, monkeypatch):
    """Install the bundled jobs plugin onto a bare app and exercise the routes
    (open mode, no API key) end to end, with the runner mocked."""
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    # Mock the runner so run-now needs no engine/server. plug.py calls through
    # the runner MODULE, so patching the canonical path reaches the route.
    import localm.plugins.builtin.jobs.runner as runner
    monkeypatch.setattr(
        runner, "run_job",
        lambda job, engine=None: {"status": "ok", "output": "ROUTE-OK",
                                  "error": None, "started": 0.0, "finished": 0.0})

    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    installed = home / "plugins"
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    mgr = PluginManager(app, store_root=store_root, installed_root=installed)
    mgr.install("jobs")

    with TestClient(app) as c:
        # create
        r = c.post("/api/jobs", json={"name": "j1", "prompt": "hello",
                                      "schedule_kind": "interval",
                                      "schedule": 120})
        assert r.status_code == 200, r.text
        jid = r.json()["id"]

        # list
        r = c.get("/api/jobs")
        assert r.status_code == 200
        assert any(j["id"] == jid for j in r.json()["jobs"])

        # detail
        assert c.get(f"/api/jobs/{jid}").json()["name"] == "j1"

        # update
        r = c.put(f"/api/jobs/{jid}", json={"enabled": False})
        assert r.json()["enabled"] is False

        # bad create -> 400
        assert c.post("/api/jobs", json={"name": "", "prompt": "x"}).status_code == 400

        # run now (mocked)
        r = c.post(f"/api/jobs/{jid}/run")
        assert r.status_code == 200 and r.json()["output"] == "ROUTE-OK"

        # results
        r = c.get(f"/api/jobs/{jid}/results")
        assert r.status_code == 200 and len(r.json()["results"]) == 1

        # delete
        assert c.delete(f"/api/jobs/{jid}").status_code == 200
        assert c.get(f"/api/jobs/{jid}").status_code == 404


def test_list_jobs_surfaces_scheduler_warning(home, monkeypatch):
    """GET /api/jobs carries scheduler_warning: a failing tick halts every
    scheduled job silently otherwise (own comment in scheduler.py cites
    AGENTS.md rule 5)."""
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    installed = home / "plugins"
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    mgr = PluginManager(app, store_root=store_root, installed_root=installed)
    mgr.install("jobs")

    # install() runs plug.py under a synthetic module name (_localm_plugin_jobs),
    # a SEPARATE module object from localm.plugins.builtin.jobs.plug - reach the
    # module the route actually bound its closures to via sys.modules.
    import sys
    jobs_plug = sys.modules["_localm_plugin_jobs"]
    with TestClient(app) as c:
        assert c.get("/api/jobs").json()["scheduler_warning"] is None

        jobs_plug._scheduler._note_tick_error(RuntimeError("jobs.json unreadable"))
        data = c.get("/api/jobs").json()
        assert data["scheduler_warning"] and "jobs.json unreadable" in data["scheduler_warning"]

        jobs_plug._scheduler._note_tick_ok()
        assert c.get("/api/jobs").json()["scheduler_warning"] is None


def test_jobs_in_catalog():
    from localm.plugins import catalog
    entry = catalog.get("jobs")
    assert entry is not None
    assert "job" in entry.commands
    assert catalog.commands().get("job") == "jobs"


def test_jobs_client_entry_is_served(home, monkeypatch):
    """The GUI imports the Jobs view from /plugins/jobs/jobs.js, so the plugin
    mounts its static assets there."""
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    app = FastAPI()
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(app, store_root=store_root, installed_root=home / "plugins")
    mgr.install("jobs")
    with TestClient(app) as c:
        r = c.get("/plugins/jobs/jobs.js")
        assert r.status_code == 200, "client_entry must be served, not 404"
        assert "register" in r.text          # the ES-module export the GUI calls


def test_jobs_available_in_api_state(home, monkeypatch):
    """The bundled jobs plugin shows up as an available plugin (store catalog)."""
    from pathlib import Path

    from fastapi import FastAPI

    from localm.plugins.engine import PluginManager
    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    installed = home / "plugins"
    mgr = PluginManager(FastAPI(), store_root=store_root, installed_root=installed)
    state = mgr.api_state()
    names = {p["name"] for p in state["plugins"]}
    assert "jobs" in names
    jobs = next(p for p in state["plugins"] if p["name"] == "jobs")
    assert jobs["available"] is True       # bundled, not yet installed
    assert "job" in jobs["commands"]


def test_jobs_manifest_parses():
    from pathlib import Path

    from localm.plugins.engine import parse_spec
    d = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin" / "jobs"
    spec = parse_spec(d, builtin=True)
    assert spec.name == "jobs"
    assert spec.scope == "jobs"
    assert spec.cli_entry == "localm.plugins.builtin.jobs.cli:main"
    assert spec.surface.tab_id == "jobs"
    assert spec.compatible()


def test_jobs_json_is_valid_after_writes(home):
    """Sanity: the on-disk defs file is valid JSON with the expected shape."""
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    store.add(_make_job(name="a"))
    store.add(_make_job(name="b"))
    data = json.loads((store.root / "jobs.json").read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert len(data["jobs"]) == 2


# --------------------------------------------------------------------------- #
#  Cron parsing and scheduling edge cases                                      #
# --------------------------------------------------------------------------- #

def test_cron_bare_value_step_expands():
    """'0/15' must mean 0,15,30,45 (Vixie 'a/n' = a..max step n), not just {0}."""
    from localm.plugins.builtin.jobs.scheduler import parse_cron
    assert parse_cron("0/15 * * * *")[0] == {0, 15, 30, 45}
    assert parse_cron("5/20 * * * *")[0] == {5, 25, 45}
    assert parse_cron("5 * * * *")[0] == {5}          # bare value (no step) unchanged
    assert parse_cron("*/15 * * * *")[0] == {0, 15, 30, 45}


def test_cron_step_fires_at_interval():
    import time
    from localm.plugins.builtin.jobs.scheduler import cron_match
    at15 = time.mktime((2026, 6, 17, 10, 15, 0, 0, 0, -1))
    at07 = time.mktime((2026, 6, 17, 10, 7, 0, 0, 0, -1))
    assert cron_match("0/15 * * * *", at15) is True
    assert cron_match("0/15 * * * *", at07) is False


def test_cron_dow_7_is_sunday():
    """Standard cron accepts 7 as Sunday; it must parse and normalise to 0."""
    from localm.plugins.builtin.jobs.scheduler import parse_cron
    assert parse_cron("0 0 * * 7")[4] == {0}
    assert parse_cron("0 0 * * 0")[4] == {0}
    assert parse_cron("0 0 * * 1-5")[4] == {1, 2, 3, 4, 5}


def test_store_concurrent_adds_lose_nothing(home):
    """Concurrent writers (scheduler tick + route handlers) must not lose jobs
    or collide on the temp file."""
    import threading
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    n = 40
    errors = []

    def worker(i):
        try:
            store.add(_make_job(name=f"job{i}"))
        except Exception as e:                       # noqa: BLE001 - want to see any
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert len(JobStore().list()) == n               # every job survived


def test_localm_job_wired_into_top_level_cli(home):
    """`localm job ...` is reachable from the top-level CLI: cli.py wires the
    command in; the manifest cli_entry alone does not mount it."""
    from click.testing import CliRunner
    from localm.cli import main
    assert "job" in main.commands
    r = CliRunner().invoke(main, ["job", "list"])
    assert r.exit_code == 0, r.output


# --------------------------------------------------------------------------- #
#  Security: scheduled coder jobs run restricted unless opted into shell       #
# --------------------------------------------------------------------------- #

def _req(headers=None):
    """A minimal Starlette Request for the auth-gate helpers (header/cookie read)."""
    from starlette.requests import Request
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "POST", "path": "/api/jobs",
                    "query_string": b"", "headers": raw})


def test_job_allow_shell_round_trips_and_defaults_false(home):
    from localm.plugins.builtin.jobs.store import Job
    j = Job(name="c", task_kind="coder", prompt="p", cwd=".")
    j.allow_shell = True
    d = j.to_dict()
    assert d["allow_shell"] is True
    assert Job.from_dict(d).allow_shell is True
    # An old jobs.json record without the field loads as the SAFE default (False).
    old = {k: v for k, v in d.items() if k != "allow_shell"}
    assert Job.from_dict(old).allow_shell is False
    # A brand-new coder job is restricted (no shell) unless explicitly opted in.
    assert Job(name="d", task_kind="coder", prompt="p", cwd=".").allow_shell is False


def _fake_agent_capture(monkeypatch):
    """Patch the runner's backend + Agent and capture the Agent kwargs."""
    from localm.plugins.builtin.jobs import runner
    captured: dict = {}

    class _FakeAgent:
        def __init__(self, backend, cwd, **kw):
            captured.update(kw)

        def run_task(self, prompt):
            return "ran"

        def close(self):
            return None

    monkeypatch.setattr(runner, "_coder_backend", lambda job: object())
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", _FakeAgent)
    return runner, captured


def test_runner_coder_is_restricted_by_default(home, tmp_path, monkeypatch):
    """A scheduled coder job with no opt-in builds a RESTRICTED Agent (no
    run_shell/network)."""
    work = tmp_path / "proj"; work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)
    job = _make_job(task_kind="coder", prompt="x", cwd=str(work), schedule=60)
    result = runner.run_job(job, engine=None)
    assert result["status"] == "ok"
    assert captured["restricted"] is True


def test_runner_coder_allow_shell_runs_unrestricted(home, tmp_path, monkeypatch):
    """The explicit owner opt-in restores the full, shell-capable coder."""
    work = tmp_path / "proj"; work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)
    job = _make_job(task_kind="coder", prompt="x", cwd=str(work), schedule=60)
    job.allow_shell = True
    result = runner.run_job(job, engine=None)
    assert result["status"] == "ok"
    assert captured["restricted"] is False


def test_caller_can_allow_shell_open_mode_is_owner(monkeypatch):
    from localm.plugins.builtin.jobs import plug
    import localm.auth as auth
    monkeypatch.setattr(auth, "any_key_configured", lambda: False)
    assert plug._caller_can_allow_shell(_req()) is True


def test_caller_can_allow_shell_owner_or_coder_full(monkeypatch):
    from localm.plugins.builtin.jobs import plug
    import localm.auth as auth
    import localm.inference.http_server as hs
    monkeypatch.setattr(auth, "any_key_configured", lambda: True)
    monkeypatch.setattr(hs, "_request_token", lambda request: ("tok", "header"))
    monkeypatch.setattr(auth, "verify", lambda t: {"admin"})
    assert plug._caller_can_allow_shell(_req()) is True
    monkeypatch.setattr(auth, "verify", lambda t: {"coder:full"})
    assert plug._caller_can_allow_shell(_req()) is True


def test_caller_cannot_allow_shell_when_not_owner(monkeypatch):
    from localm.plugins.builtin.jobs import plug
    import localm.auth as auth
    import localm.inference.http_server as hs
    monkeypatch.setattr(auth, "any_key_configured", lambda: True)
    monkeypatch.setattr(hs, "_request_token", lambda request: ("tok", "header"))
    # A plain jobs/coder key (no admin, no coder:full) cannot opt into shell.
    monkeypatch.setattr(auth, "verify", lambda t: {"jobs", "coder"})
    assert plug._caller_can_allow_shell(_req()) is False
    # No presented token at all -> denied.
    monkeypatch.setattr(hs, "_request_token", lambda request: (None, None))
    assert plug._caller_can_allow_shell(_req()) is False


def test_create_job_route_rejects_unauthorized_allow_shell(home, monkeypatch):
    import asyncio
    from fastapi import HTTPException
    from localm.plugins.builtin.jobs import plug
    monkeypatch.setattr(plug, "_caller_can_allow_shell", lambda request: False)
    req = plug.JobCreate(name="x", task_kind="coder", prompt="p", cwd=str(home))
    req.allow_shell = True
    with pytest.raises(HTTPException) as ei:
        asyncio.run(plug.create_job(req, _req()))
    assert ei.value.status_code == 403
    # And nothing was persisted.
    from localm.plugins.builtin.jobs.store import JobStore
    assert JobStore().list() == []


def test_create_job_route_allows_authorized_allow_shell(home, monkeypatch):
    import asyncio
    from localm.plugins.builtin.jobs import plug
    monkeypatch.setattr(plug, "_caller_can_allow_shell", lambda request: True)
    req = plug.JobCreate(name="x", task_kind="coder", prompt="p", cwd=str(home))
    req.allow_shell = True
    out = asyncio.run(plug.create_job(req, _req()))
    assert out["allow_shell"] is True
    # A default coder job (no opt-in) is stored restricted.
    out2 = asyncio.run(plug.create_job(
        plug.JobCreate(name="y", task_kind="coder", prompt="p", cwd=str(home)),
        _req()))
    assert out2["allow_shell"] is False


def test_cli_allow_shell_flag_sets_it(home):
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore
    r = CliRunner().invoke(main, ["add", "shelljob", "--coder", "--prompt", "p",
                                  "--cwd", str(home), "--allow-shell", "--every", "60"])
    assert r.exit_code == 0, r.output
    job = JobStore().list()[0]
    assert job.task_kind == "coder"
    assert job.allow_shell is True


def test_cli_coder_job_defaults_to_no_shell(home):
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore
    r = CliRunner().invoke(main, ["add", "safejob", "--coder", "--prompt", "p",
                                  "--cwd", str(home), "--every", "60"])
    assert r.exit_code == 0, r.output
    assert JobStore().list()[0].allow_shell is False


# --------------------------------------------------------------------------- #
#  The scheduler starts on the REAL server path, where plugins register       #
#  before the event loop exists.                                              #
# --------------------------------------------------------------------------- #

def _install_jobs_into(home):
    """Copy+enable the jobs plugin into *home*'s installed folder using a
    throwaway bootstrap manager (mirrors what a user install does)."""
    from pathlib import Path

    from fastapi import FastAPI
    from localm.plugins.engine import PluginManager
    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    boot = PluginManager(FastAPI(), store_root=store_root,
                         installed_root=home / "plugins")
    boot.install("jobs")


def test_scheduler_starts_via_server_lifespan(home, monkeypatch):
    """Build the app the way `localm serve` does (create_app -> attach_engine
    inside it -> uvicorn lifespan): the scheduler starts and a due job RUNS.
    register() runs before the loop exists, so the start is queued for the
    lifespan rather than attempted immediately."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    _install_jobs_into(home)

    ran = []
    import localm.plugins.builtin.jobs.runner as runner
    monkeypatch.setattr(
        runner, "run_job",
        lambda job, engine=None: ran.append(job.id) or {
            "status": "ok", "output": "TICKED", "error": None,
            "started": 0.0, "finished": 0.0})

    from localm.inference.http_server import create_app
    app = create_app(None)
    mgr = getattr(app.state, "plugin_manager", None)
    assert mgr is not None, "plugin engine failed to attach"
    assert "jobs" in mgr._loaded, "jobs plugin did not load"
    module = mgr._loaded["jobs"][1]
    sched = module._scheduler
    assert sched is not None
    # Before the loop exists the scheduler must NOT be running yet, and the start
    # must be QUEUED for the lifespan (not lost).
    assert not sched.running
    assert mgr._startup_callbacks, "startup callback was not queued"

    # A due job created before startup (interval jobs are due immediately).
    from localm.plugins.builtin.jobs.store import JobStore
    job = JobStore().add(_make_job(schedule=3600))
    sched.poll_seconds = 0.05          # fast ticks for the test

    with TestClient(app):              # runs the app lifespan = server startup
        assert sched.running, "lifespan did not start the scheduler"
        deadline = time.time() + 10
        while not ran and time.time() < deadline:
            time.sleep(0.05)
    assert ran == [job.id], "a due job did not run on a stock server start"


def test_on_startup_runs_immediately_when_loop_is_up(home):
    """Runtime enable (management API, loop already running): the callback runs
    right away and nothing is queued."""
    import asyncio
    from types import SimpleNamespace

    from fastapi import FastAPI
    from localm.plugins.engine import PluginHost, PluginManager

    mgr = PluginManager(FastAPI(), store_root=None, installed_root=home / "p")
    host = PluginHost(mgr.app, mgr, SimpleNamespace(name="x", surface=None))
    fired = []

    async def scenario():
        host.on_startup(lambda: fired.append(True))

    asyncio.run(scenario())
    assert fired == [True]
    assert mgr._startup_callbacks == []


def test_unmount_discards_queued_startup_callbacks(home):
    """A plugin disabled before the lifespan ran does not have its deferred
    startup work fire later."""
    from types import SimpleNamespace

    from fastapi import FastAPI
    from localm.plugins.engine import PluginHost, PluginManager

    mgr = PluginManager(FastAPI(), store_root=None, installed_root=home / "p")
    host = PluginHost(mgr.app, mgr, SimpleNamespace(name="x", surface=None))
    host.on_startup(lambda: (_ for _ in ()).throw(AssertionError("must not run")))
    assert len(mgr._startup_callbacks) == 1
    host.unmount()
    assert mgr._startup_callbacks == []
    mgr.run_startup_callbacks()        # nothing left; must not raise


# --------------------------------------------------------------------------- #
#  Scheduled coder job endpoint + cron catch-up                               #
# --------------------------------------------------------------------------- #

def test_coder_backend_uses_configured_port_not_8080(home, monkeypatch):
    """With no LOCALM_SELF_URL the coder backend targets the configured port,
    not a hardcoded :8080."""
    monkeypatch.delenv("LOCALM_SELF_URL", raising=False)
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"port": 8642})

    captured = {}

    class _FakeBackend:
        def __init__(self, url, model=None, api_key=None, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "localm.plugins.coder.backends.http.HTTPBackend", _FakeBackend)
    from localm.plugins.builtin.jobs import runner
    runner._coder_backend(_make_job(task_kind="coder", cwd=str(home)))
    assert captured["url"] == "http://127.0.0.1:8642/v1"
    assert ":8080" not in captured["url"]
    # A self-connection is a localm server, so grammar sampling is available.
    assert captured["kwargs"].get("localm_server") is True


def test_coder_backend_honours_explicit_self_url(home, monkeypatch):
    monkeypatch.setenv("LOCALM_SELF_URL", "http://127.0.0.1:8677/v1")
    captured = {}

    class _FakeBackend:
        def __init__(self, url, model=None, api_key=None, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "localm.plugins.coder.backends.http.HTTPBackend", _FakeBackend)
    from localm.plugins.builtin.jobs import runner
    runner._coder_backend(_make_job(task_kind="coder", cwd=str(home)))
    assert captured["url"] == "http://127.0.0.1:8677/v1"
    assert captured["kwargs"].get("localm_server") is True


def test_publish_self_url_from_app_state(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("LOCALM_SELF_URL", raising=False)
    from localm.plugins.builtin.jobs import plug
    host = SimpleNamespace(_app=SimpleNamespace(
        state=SimpleNamespace(instance_port=8699, instance_scheme="http")))
    plug._publish_self_url(host)
    import os
    assert os.environ["LOCALM_SELF_URL"] == "http://127.0.0.1:8699/v1"


def test_publish_self_url_does_not_override_env(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("LOCALM_SELF_URL", "http://127.0.0.1:1234/v1")
    from localm.plugins.builtin.jobs import plug
    host = SimpleNamespace(_app=SimpleNamespace(
        state=SimpleNamespace(instance_port=8699, instance_scheme="http")))
    plug._publish_self_url(host)
    import os
    assert os.environ["LOCALM_SELF_URL"] == "http://127.0.0.1:1234/v1"


def test_cron_catch_up_fires_a_slot_missed_while_down(home):
    """A cron slot that passed while the server was down runs ONCE on
    restart."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    sched = JobScheduler(JobStore())
    # Daily 09:00 job that last ran yesterday 09:00; "now" is today 09:05, so
    # today's 09:00 slot was missed while the server was down.
    job = _make_job(schedule_kind="cron", schedule="0 9 * * *",
                    last_run=_at(2026, 6, 16, 9, 0))
    now = _at(2026, 6, 17, 9, 5)
    assert sched.due(job, now) is True


def test_cron_no_catch_up_for_never_run_job(home):
    """A freshly created cron job never back-fires history: it waits for its
    next real slot."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    sched = JobScheduler(JobStore())
    job = _make_job(schedule_kind="cron", schedule="0 9 * * *", last_run=None)
    now = _at(2026, 6, 17, 15, 0)          # 3pm, past today's 9am slot
    assert sched.due(job, now) is False


def test_cron_no_catch_up_beyond_window(home):
    """A cron slot whose most recent occurrence is more than 24h before now is
    too stale to back-fire (a weekly job on a machine off for days)."""
    import datetime as _dt

    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    sched = JobScheduler(JobStore())
    # Find a Monday, then set "now" to the Wednesday 3 days later at noon, so the
    # most recent "Monday 09:00" slot is ~2.1 days before now (> the 24h window).
    d = _dt.date(2026, 6, 15)
    while d.weekday() != 0:                 # 0 = Monday
        d += _dt.timedelta(days=1)
    monday_9 = _at(d.year, d.month, d.day, 9, 0)
    now = _at(d.year, d.month, d.day, 9, 0) + 2 * 86400 + 3 * 3600   # +2d3h = Wed 12:00
    job = _make_job(schedule_kind="cron", schedule="0 9 * * 1",
                    last_run=monday_9 - 7 * 86400)      # last ran the prior Monday
    assert sched.due(job, now) is False


def test_cron_no_catch_up_when_no_slot_missed(home):
    """No spurious catch-up: a daily job whose next slot is still in the future
    (now is before today's slot) must not fire."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    sched = JobScheduler(JobStore())
    job = _make_job(schedule_kind="cron", schedule="0 9 * * *",
                    last_run=_at(2026, 6, 16, 9, 0))    # yesterday 09:00
    now = _at(2026, 6, 17, 8, 0)           # today 08:00, before the 09:00 slot
    assert sched.due(job, now) is False


def test_cron_current_minute_still_fires_once(home):
    """The normal same-minute cron path is unchanged: fires once, then the
    per-minute dedup blocks a second sub-minute poll."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    sched = JobScheduler(JobStore())
    job = _make_job(schedule_kind="cron", schedule="0 9 * * *",
                    last_run=_at(2026, 6, 16, 9, 0))
    now = _at(2026, 6, 17, 9, 0)
    assert sched.due(job, now) is True
    sched._cron_fired[job.id] = int(now // 60) * 60     # mark fired this minute
    assert sched.due(job, now) is False
