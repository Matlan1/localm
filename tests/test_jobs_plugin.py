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


def test_cron_every_minute_matches_always():
    from localm.plugins.builtin.jobs.scheduler import cron_match
    assert cron_match("* * * * *", _at(2026, 6, 17, 13, 37)) is True


def test_cron_specific_minute_hour():
    from localm.plugins.builtin.jobs.scheduler import cron_match
    expr = "30 9 * * *"
    assert cron_match(expr, _at(2026, 6, 17, 9, 30)) is True
    assert cron_match(expr, _at(2026, 6, 17, 9, 31)) is False
    assert cron_match(expr, _at(2026, 6, 17, 10, 30)) is False


def test_cron_step():
    from localm.plugins.builtin.jobs.scheduler import cron_match
    expr = "*/15 * * * *"          # 0,15,30,45
    assert cron_match(expr, _at(2026, 6, 17, 0, 0)) is True
    assert cron_match(expr, _at(2026, 6, 17, 0, 15)) is True
    assert cron_match(expr, _at(2026, 6, 17, 0, 30)) is True
    assert cron_match(expr, _at(2026, 6, 17, 0, 7)) is False


def test_cron_range():
    from localm.plugins.builtin.jobs.scheduler import cron_match
    expr = "0 9-17 * * *"          # on the hour, 9am-5pm
    assert cron_match(expr, _at(2026, 6, 17, 9, 0)) is True
    assert cron_match(expr, _at(2026, 6, 17, 17, 0)) is True
    assert cron_match(expr, _at(2026, 6, 17, 8, 0)) is False
    assert cron_match(expr, _at(2026, 6, 17, 18, 0)) is False


def test_cron_list():
    from localm.plugins.builtin.jobs.scheduler import cron_match
    expr = "0,30 * * * *"
    assert cron_match(expr, _at(2026, 6, 17, 4, 0)) is True
    assert cron_match(expr, _at(2026, 6, 17, 4, 30)) is True
    assert cron_match(expr, _at(2026, 6, 17, 4, 15)) is False


def test_cron_day_of_week():
    from localm.plugins.builtin.jobs.scheduler import cron_match
    # 2026-06-15 is a Monday; dow Monday == 1.
    expr = "0 0 * * 1"
    assert cron_match(expr, _at(2026, 6, 15, 0, 0)) is True     # Monday
    assert cron_match(expr, _at(2026, 6, 16, 0, 0)) is False    # Tuesday


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
    # No model resolvable -> _load_engine returns None -> error result.
    monkeypatch.setattr(runner, "_load_engine", lambda model: None)
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
    # the runner MODULE, so patching the canonical path reaches the route even
    # though the engine imports plug.py under a synthetic module name.
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


def test_jobs_in_catalog():
    from localm.plugins import catalog
    entry = catalog.get("jobs")
    assert entry is not None
    assert "job" in entry.commands
    assert catalog.commands().get("job") == "jobs"


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
    assert spec.cli_entry == "cli:main"
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
