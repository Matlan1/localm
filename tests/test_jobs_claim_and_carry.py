# SPDX-License-Identifier: AGPL-3.0-or-later
"""Jobs: one run at a time across processes, entries the build cannot parse are
carried through every write, scope and creator checks, exclusive result files."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _job(**kw):
    from localm.plugins.builtin.jobs.store import Job
    base = dict(name="t", task_kind="chat", prompt="hi",
                schedule_kind="interval", schedule=3600)
    base.update(kw)
    return Job(**base)


def _entries(store):
    return json.loads(store._defs_file.read_text(encoding="utf-8"))["jobs"]


# --------------------------------------------------------------------------- #
#  Entries the build cannot parse survive every write                          #
# --------------------------------------------------------------------------- #

FUTURE = {"id": "future-job", "name": "later", "task_kind": "future-kind",
          "schedule_kind": "interval", "schedule": 60, "prompt": "p",
          "owner": "a" * 64}


def _store_with_future_entry(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore(home / "jobs")
    job = store.add(_job())
    data = json.loads(store._defs_file.read_text(encoding="utf-8"))
    data["jobs"].append(dict(FUTURE))
    data["jobs"].append("not an object")
    data["note"] = {"kept": True}
    store._defs_file.write_text(json.dumps(data), encoding="utf-8")
    return store, job


def _assert_carried(store):
    raw = json.loads(store._defs_file.read_text(encoding="utf-8"))
    assert FUTURE in raw["jobs"]
    assert "not an object" in raw["jobs"]
    assert raw["note"] == {"kept": True}


def test_record_result_keeps_unparsed_entries(home):
    store, job = _store_with_future_entry(home)
    store.record_result(job.id, {"status": "ok", "finished": time.time()})
    _assert_carried(store)
    assert [j.id for j in store.list()] == [job.id]


def test_add_update_remove_keep_unparsed_entries(home):
    store, job = _store_with_future_entry(home)
    other = store.add(_job(name="other"))
    _assert_carried(store)
    store.update(other.id, name="renamed")
    _assert_carried(store)
    assert store.remove(other.id) is True
    _assert_carried(store)


def test_claim_and_release_keep_unparsed_entries(home):
    store, job = _store_with_future_entry(home)
    claimed = store.claim(job.id)
    assert claimed is not None
    _assert_carried(store)
    store.release_claim(job.id, claimed.claim)
    _assert_carried(store)


# --------------------------------------------------------------------------- #
#  Claims                                                                      #
# --------------------------------------------------------------------------- #

def test_claim_is_exclusive_until_the_result_is_recorded(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore(home / "jobs")
    job = store.add(_job())
    first = store.claim(job.id)
    assert first is not None and first.claim["pid"] == os.getpid()
    assert JobStore(home / "jobs").claim(job.id) is None
    store.record_result(job.id, {"status": "ok", "finished": time.time()})
    assert store.get(job.id).claim is None
    assert store.claim(job.id) is not None


def test_a_live_claim_makes_the_job_not_due(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore(home / "jobs")
    job = store.add(_job())
    sched = JobScheduler(store)
    assert sched.due(store.get(job.id), time.time()) is True
    store.claim(job.id)
    assert sched.due(store.get(job.id), time.time()) is False


def test_claim_of_a_dead_process_is_reclaimed(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore(home / "jobs")
    job = store.add(_job())
    proc = subprocess.Popen([sys.executable, "-c", "pass"],
                            creationflags=NO_WINDOW)
    proc.wait()
    from localm import instances
    stale = {"pid": proc.pid, "start": instances.process_start_identity(os.getpid()),
             "token": "dead"}
    data = json.loads(store._defs_file.read_text(encoding="utf-8"))
    data["jobs"][0]["claim"] = stale
    store._defs_file.write_text(json.dumps(data), encoding="utf-8")
    assert store.claim(job.id) is not None


def test_claim_of_a_live_process_with_another_start_identity_is_reclaimed(home):
    from localm import instances
    from localm.plugins.builtin.jobs.store import claim_is_live
    ident = instances.process_start_identity(os.getppid())
    if ident is None:
        pytest.skip("no process start identity on this platform")
    wrong = {"created": 1.0} if "created" in ident else {
        "boot": ident["boot"], "ticks": ident["ticks"] + 99999}
    assert claim_is_live({"pid": os.getppid(), "start": ident, "token": "x"}) is True
    assert claim_is_live({"pid": os.getppid(), "start": wrong, "token": "x"}) is False


def test_cli_run_refuses_a_job_with_a_live_claim(home, monkeypatch):
    from click.testing import CliRunner

    from localm.plugins.builtin.jobs import cli
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore(home / "jobs")
    job = store.add(_job())
    monkeypatch.setattr(cli, "_store", lambda: store)
    calls = []
    import localm.plugins.builtin.jobs.runner as runner
    monkeypatch.setattr(runner, "run_job", lambda j, engine=None: calls.append(j) or {
        "status": "ok", "output": "x", "error": None, "started": 0, "finished": 0})
    store.claim(job.id)
    res = CliRunner().invoke(cli.main, ["run", job.id])
    assert res.exit_code == 1 and not calls
    store.record_result(job.id, {"status": "ok", "finished": time.time()})
    res = CliRunner().invoke(cli.main, ["run", job.id])
    assert res.exit_code == 0 and len(calls) == 1
    assert store.get(job.id).claim is None


TICK_SCRIPT = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore

    counter = Path(sys.argv[3])

    def run(job, engine=None):
        with open(counter, "a") as fh:
            fh.write("run\\n")
        time.sleep(2)
        return {"status": "ok", "output": "", "error": None,
                "started": time.time(), "finished": time.time()}

    JobScheduler(JobStore(Path(sys.argv[2])), run_job=run).tick()
""")


def test_two_interpreters_ticking_one_folder_run_a_due_job_once(home):
    from localm.plugins.builtin.jobs.store import JobStore
    root = home / "jobs"
    JobStore(root).add(_job())
    counter = home / "counter.txt"
    script = home / "tick.py"
    script.write_text(TICK_SCRIPT, encoding="utf-8")
    env = {**os.environ, "LOCALM_HOME": str(home), "PYTHONPATH": str(REPO)}
    procs = [subprocess.Popen(
        [sys.executable, str(script), str(REPO), str(root), str(counter)],
        env=env, creationflags=NO_WINDOW, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE) for _ in range(2)]
    outs = [p.communicate(timeout=120) for p in procs]
    for p, (_o, e) in zip(procs, outs, strict=True):
        assert p.returncode == 0, e.decode(errors="replace")
    assert counter.read_text().splitlines() == ["run"]


# --------------------------------------------------------------------------- #
#  Scope for the task kind, creator liveness                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def jobs_app(home):
    from localm.plugins.engine import PluginManager
    store_root = REPO / "localm" / "plugins" / "builtin"
    app = FastAPI()
    PluginManager(app, store_root=store_root,
                  installed_root=home / "plugins").install("jobs")
    return app


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _key(scopes):
    from localm import auth
    return auth.create_key("k", scopes)["key"]


def test_create_requires_the_scope_matching_the_task_kind(jobs_app):
    jobs_only = _key(["jobs"])
    with_chat = _key(["jobs", "chat"])
    with_coder = _key(["jobs", "coder"])
    body = {"name": "j", "prompt": "hi", "schedule": 120}
    coder_body = {**body, "task_kind": "coder", "cwd": "."}
    with TestClient(jobs_app) as c:
        assert c.post("/api/jobs", headers=_h(jobs_only), json=body).status_code == 403
        assert c.post("/api/jobs", headers=_h(jobs_only),
                      json=coder_body).status_code == 403
        assert c.post("/api/jobs", headers=_h(with_coder),
                      json=body).status_code == 403
        assert c.post("/api/jobs", headers=_h(with_chat), json=body).status_code == 200
        assert c.post("/api/jobs", headers=_h(with_coder),
                      json=coder_body).status_code == 200


def test_runner_skips_a_job_whose_creator_key_is_gone(home):
    from localm import auth
    from localm.plugins.builtin.jobs import runner
    key = auth.create_key("k", ["jobs", "chat"])
    owner = auth._hash_key(key["key"])
    job = _job(owner=owner)
    called = []
    orig = runner._run_chat
    runner._run_chat = lambda j, engine=None: called.append(j) or "ran"
    try:
        assert runner.run_job(job, engine=object())["status"] == "ok"
        auth.revoke_key(key.get("id") or key["name"]) if hasattr(
            auth, "revoke_key") else None
        called.clear()
        result = runner.run_job(job, engine=object())
    finally:
        runner._run_chat = orig
    assert result["status"] == "error"
    assert "could not be re-confirmed" in result["error"] and not called


def test_runner_skips_a_job_whose_creator_lost_the_task_kind_scope(home):
    from localm import auth
    from localm.plugins.builtin.jobs import runner
    key = auth.create_key("k", ["jobs"])
    job = _job(owner=auth._hash_key(key["key"]))
    result = runner.run_job(job, engine=object())
    assert result["status"] == "error" and "'chat' scope" in result["error"]


# --------------------------------------------------------------------------- #
#  Result files are created exclusively                                        #
# --------------------------------------------------------------------------- #

def test_record_result_never_overwrites_an_existing_result_file(home, monkeypatch):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore(home / "jobs")
    job = store.add(_job())
    result = {"status": "ok", "output": "new", "finished": 1_700_000_000.0}
    rid = store.record_result(job.id, result)
    path = store._result_dir(job.id) / f"{rid}.json"
    path.write_text('{"output": "original"}', encoding="utf-8")
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists",
                        lambda self, **kw: False if self == path else real_exists(self, **kw))
    rid2 = store.record_result(job.id, result)
    assert rid2 != rid
    assert json.loads(path.read_text(encoding="utf-8")) == {"output": "original"}
