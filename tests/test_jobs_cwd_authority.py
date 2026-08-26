# SPDX-License-Identifier: AGPL-3.0-or-later
"""A jobs-scoped key must not choose an arbitrary working directory.

The coder ROUTE already draws this line: a restricted caller is forced into the
project root, "ignoring req.cwd, so a scoped key cannot point the (confined)
file tools at arbitrary paths" (builtin/coder/plug.py). The SCHEDULER has no
equivalent unless one is added - ``cwd`` is validated only for UNC/device SHAPE,
never for who chose it - so a plain ``jobs``-scoped key gets read plus confined
write on any directory on the server simply by scheduling a coder job there.

Two gates, both needed: the route CONFINES on the way in, and the runner
CONFINES again at run time, because the route is not the only writer - the CLI
and rows persisted by older builds reach the scheduler without passing through
it.

Confined rather than REFUSED, which is the opposite of how ``allow_shell`` is
handled at the same route: ``allow_shell`` is an optional opt-in, so refusing it
still leaves a working job, while ``cwd`` is MANDATORY for a coder job
(Job.validate), so refusing it would not restrict the capability, it would
delete it.

The run-time answer is RE-DERIVED, not stamped, which is the opposite choice
from the neighbouring ``owner_is_owner_key``. Unlike "was this the owner key",
"may this principal choose a directory" is still answerable later, so narrowing
or revoking a key removes the freedom on the next tick instead of leaving a
months-old grant stamped on the row.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S

JOBS = "jobs"
OPT_IN = True
KEY_ONE = "owner-key-cwd-authority-0123456789"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


@pytest.fixture
def jobs_app(home, tmp_path):
    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, store_root=store_root,
                  installed_root=tmp_path / "plugins").install(JOBS)
    return app


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _post_job(app, key, cwd, **over):
    body = {"name": "nightly", "task_kind": "coder", "prompt": "tidy up",
            "cwd": str(cwd), "schedule_kind": "interval", "schedule": 3600}
    body.update(over)
    with TestClient(app) as c:
        return c.post("/api/jobs", json=body, headers=_h(key))


def _fake_agent_capture(monkeypatch):
    """Capture the cwd the Agent is actually constructed with - the observable
    that decides where the file tools can reach."""
    from localm.plugins.builtin.jobs import runner
    captured: dict = {}

    class _FakeAgent:
        def __init__(self, backend, cwd, **kw):
            captured["cwd"] = Path(cwd)
            captured.update(kw)

        def run_task(self, prompt):
            return "ran"

        def close(self):
            return None

    monkeypatch.setattr(runner, "_coder_backend", lambda job: object())
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", _FakeAgent)
    return runner, captured


# --------------------------------------------------------------------------- #
#  The route gate                                                              #
# --------------------------------------------------------------------------- #

def test_a_jobs_only_keys_directory_is_confined_to_the_project_root(jobs_app,
                                                                     tmp_path):
    """CONFINED, not refused. cwd is MANDATORY for a coder job, so refusing it
    would remove the capability from a jobs-scoped key rather than restrict it -
    the opposite of allow_shell, which is an optional opt-in and IS refused."""
    from localm import auth
    from localm.instances import resolve_root_dir
    victim = tmp_path / "elsewhere"
    victim.mkdir()
    auth.set_api_key(KEY_ONE)
    scoped = auth.create_key("bot", [JOBS], allow_privileged=False)

    r = _post_job(jobs_app, scoped["key"], victim)

    assert r.status_code == 200, r.text
    # Not silent: the EFFECTIVE cwd is in the response the caller gets back.
    assert Path(r.json()["cwd"]).resolve() == Path(resolve_root_dir()).resolve()
    assert Path(r.json()["cwd"]).resolve() != victim.resolve()
    # And it is what actually persisted, not merely what was echoed.
    from localm.plugins.builtin.jobs.store import JobStore
    assert Path(JobStore().get(r.json()["id"]).cwd).resolve() != victim.resolve()


def test_the_owner_key_may_still_choose_one(jobs_app, tmp_path):
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)

    r = _post_job(jobs_app, KEY_ONE, work)

    assert r.status_code == 200, r.text


def test_a_coder_full_key_may_still_choose_one(jobs_app, tmp_path):
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    cf = auth.create_key("cf", [S.CODER_FULL, JOBS], allow_privileged=True)

    r = _post_job(jobs_app, cf["key"], work)

    assert r.status_code == 200, r.text


def test_a_jobs_only_key_can_still_schedule_a_coder_job_at_all(jobs_app, tmp_path):
    """The capability must SURVIVE the gate. This is the test that caught the
    first draft: it refused the cwd outright, and because Job.validate requires a
    cwd for a coder job, a jobs-scoped key could then not schedule one at all."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    scoped = auth.create_key("bot", [JOBS], allow_privileged=False)

    r = _post_job(jobs_app, scoped["key"], work)

    assert r.status_code == 200, r.text
    assert r.json()["task_kind"] == "coder"


def test_a_non_coder_job_is_unaffected(jobs_app):
    """Only coder jobs carry a cwd; a chat job must not be touched."""
    from localm import auth
    auth.set_api_key(KEY_ONE)
    scoped = auth.create_key("bot", [JOBS], allow_privileged=False)

    with TestClient(jobs_app) as c:
        r = c.post("/api/jobs", json={
            "name": "n", "task_kind": "chat", "prompt": "p",
            "schedule_kind": "interval", "schedule": 3600,
        }, headers=_h(scoped["key"]))

    assert r.status_code == 200, r.text


def test_the_update_route_is_gated_too(jobs_app, tmp_path):
    """PUT is the second write path into cwd; without the same gate the create
    check is simply routed around with an update."""
    from localm import auth
    victim = tmp_path / "elsewhere"
    victim.mkdir()
    auth.set_api_key(KEY_ONE)
    scoped = auth.create_key("bot", [JOBS], allow_privileged=False)

    with TestClient(jobs_app) as c:
        made = c.post("/api/jobs", json={
            "name": "n", "task_kind": "coder", "prompt": "p",
            "cwd": str(tmp_path), "schedule_kind": "interval", "schedule": 3600,
        }, headers=_h(scoped["key"]))
        assert made.status_code == 200, made.text
        r = c.put(f"/api/jobs/{made.json()['id']}", json={"cwd": str(victim)},
                  headers=_h(scoped["key"]))

    assert r.status_code == 200, r.text
    assert Path(r.json()["cwd"]).resolve() != victim.resolve()


# --------------------------------------------------------------------------- #
#  The runner gate - the route is not the only writer                          #
# --------------------------------------------------------------------------- #

def _scoped_job(auth, cwd, scopes_list, *, privileged=False):
    """A job persisted DIRECTLY into the store (as the CLI, an older build, or a
    pre-gate row does), owned by a key holding *scopes_list*."""
    from localm.plugins.builtin.jobs.store import Job, JobStore
    created = auth.create_key("bot", scopes_list, allow_privileged=privileged)
    job = Job(name="x", task_kind="coder", prompt="p", cwd=str(cwd),
              schedule_kind="interval", schedule=60,
              owner=auth._hash_key(created["key"]))
    JobStore().add(job)
    return created, job


def test_the_runner_confines_a_jobs_only_keys_job_to_the_project_root(
        home, tmp_path, monkeypatch):
    """The authoritative half: a row that never passed the route must still not
    reach an arbitrary directory."""
    from localm import auth
    from localm.instances import resolve_root_dir
    runner, captured = _fake_agent_capture(monkeypatch)
    victim = tmp_path / "elsewhere"
    victim.mkdir()
    auth.set_api_key(KEY_ONE)
    _created, job = _scoped_job(auth, victim, [JOBS])

    assert runner.run_job(job, engine=None)["status"] == "ok"

    assert captured["cwd"] == Path(resolve_root_dir()).resolve(), \
        "a jobs-only key's job ran in the directory it chose"
    assert captured["cwd"] != victim.resolve()


def test_the_confinement_is_surfaced_not_silent(home, tmp_path, monkeypatch,
                                                caplog):
    """Rule 5: the run did something narrower than configured, so it must say so
    in the log AND in the job's own output, where a user actually looks."""
    from localm import auth
    runner, _captured = _fake_agent_capture(monkeypatch)
    victim = tmp_path / "elsewhere"
    victim.mkdir()
    auth.set_api_key(KEY_ONE)
    _created, job = _scoped_job(auth, victim, [JOBS])

    with caplog.at_level("WARNING"):
        result = runner.run_job(job, engine=None)

    assert result["status"] == "ok"          # confined, not failed
    assert "working directory" in result["output"].lower()
    assert any("working directory" in r.message.lower() for r in caplog.records)


def test_a_coder_full_keys_job_keeps_its_directory(home, tmp_path, monkeypatch):
    from localm import auth
    runner, captured = _fake_agent_capture(monkeypatch)
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    _created, job = _scoped_job(auth, work, [S.CODER_FULL, JOBS], privileged=True)

    assert runner.run_job(job, engine=None)["status"] == "ok"

    assert captured["cwd"] == work.resolve()


def test_an_owner_created_job_keeps_its_directory(home, tmp_path, monkeypatch):
    from localm import auth
    from localm.plugins.builtin.jobs.store import Job
    runner, captured = _fake_agent_capture(monkeypatch)
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    job = Job(name="x", task_kind="coder", prompt="p", cwd=str(work),
              schedule_kind="interval", schedule=60,
              owner=auth._hash_key(KEY_ONE))

    assert runner.run_job(job, engine=None)["status"] == "ok"

    assert captured["cwd"] == work.resolve()
    assert "working directory" not in runner.run_job(job, engine=None)["output"].lower()


def test_an_owner_session_job_keeps_its_directory_across_a_key_roll(
        home, tmp_path, monkeypatch):
    """Reuses the owner-session stamp rather than adding a second field. This is
    the one case run-time re-derivation cannot reach: after a roll the recorded
    hash matches nothing, so without the stamp the owner's own job would be
    confined."""
    from localm import auth
    from localm.plugins.builtin.jobs.store import Job
    runner, captured = _fake_agent_capture(monkeypatch)
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    job = Job(name="x", task_kind="coder", prompt="p", cwd=str(work),
              schedule_kind="interval", schedule=60,
              owner=auth._hash_key(KEY_ONE), owner_is_owner_key=True)
    auth.regenerate_key()

    assert runner.run_job(job, engine=None)["status"] == "ok"

    assert captured["cwd"] == work.resolve()


def test_an_unowned_open_mode_job_keeps_its_directory(home, tmp_path, monkeypatch):
    """owner=None is a tokenless / open-mode creation, which IS the loopback
    owner, so there is no lesser principal to confine."""
    from localm.plugins.builtin.jobs.store import Job
    runner, captured = _fake_agent_capture(monkeypatch)
    work = tmp_path / "proj"
    work.mkdir()
    job = Job(name="x", task_kind="coder", prompt="p", cwd=str(work),
              schedule_kind="interval", schedule=60, owner=None)

    assert runner.run_job(job, engine=None)["status"] == "ok"

    assert captured["cwd"] == work.resolve()


# --------------------------------------------------------------------------- #
#  Re-derivation: the freedom is re-checked, never stamped                     #
# --------------------------------------------------------------------------- #

def test_revoking_the_key_removes_the_directory_freedom_on_the_next_run(
        home, tmp_path, monkeypatch):
    """The freedom is re-derived, not stamped: a coder:full job that was
    legitimately allowed a directory loses it the moment its key is revoked,
    rather than keeping a grant recorded at creation."""
    from localm import auth
    from localm.instances import resolve_root_dir
    runner, captured = _fake_agent_capture(monkeypatch)
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    created, job = _scoped_job(auth, work, [S.CODER_FULL, JOBS], privileged=True)

    assert runner.run_job(job, engine=None)["status"] == "ok"
    assert captured["cwd"] == work.resolve()            # control

    assert auth.revoke_key(created["id"]) is True
    captured.clear()
    assert runner.run_job(job, engine=None)["status"] == "ok"

    assert captured["cwd"] == Path(resolve_root_dir()).resolve(), \
        "a revoked key kept the arbitrary-directory freedom it was granted at creation"


def test_an_unreadable_keystore_does_not_grant_the_freedom(home, tmp_path,
                                                           monkeypatch):
    """Fails CLOSED. _load_keystore() fails OPEN (returns [] on OSError), so
    scopes_for_key_hash answers None, and None must never read as a grant - the
    same trap that produced two privilege escalations in the neighbouring
    owner-key check."""
    from localm import auth
    from localm.instances import resolve_root_dir
    runner, captured = _fake_agent_capture(monkeypatch)
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    _created, job = _scoped_job(auth, work, [S.CODER_FULL, JOBS], privileged=True)

    monkeypatch.setattr(auth, "_load_keystore", lambda: [])
    assert runner.run_job(job, engine=None)["status"] == "ok"

    assert captured["cwd"] == Path(resolve_root_dir()).resolve()


def test_scopes_for_key_hash_answers_none_for_an_expired_key(home):
    """The by-hash sibling of verify() must share its liveness rules, or the
    runner would honour a key the bearer path already rejects."""
    import time
    from localm import auth
    auth.set_api_key(KEY_ONE)
    created = auth.create_key("d", [S.CODER_FULL], allow_privileged=True,
                              expires=time.time() + 3600)
    kh = auth._hash_key(created["key"])
    assert auth.scopes_for_key_hash(kh) == {S.CODER_FULL}     # control

    ks = auth._load_keystore()
    for rec in ks:
        rec["expires"] = time.time() - 10
    auth._save_keystore(ks)

    assert auth.scopes_for_key_hash(kh) is None
    assert auth.scopes_for_key_hash("deadbeef" * 8) is None
    assert auth.scopes_for_key_hash(None) is None
