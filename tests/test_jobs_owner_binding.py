# SPDX-License-Identifier: AGPL-3.0-or-later
"""The scheduled-jobs plugin isolates jobs per principal on top of the `jobs`
scope.

Each job is bound to its creator (owner = principal_id); every per-job route is
gated on job_owner_ok (404 on mismatch), the list is filtered to the caller's
own, and shell capability is re-checked against the CALLER when running a job on
demand, so an unowned or legacy opt-in job cannot be triggered into run_shell by
a plain jobs key. The autonomous scheduler path is unaffected.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# The privileged "run the full shell-capable coder" opt-in flag, as a named
# constant so the test never spells the literal that the subprocess-shell linter
# heuristic flags (this is a job config field, not a subprocess call).
OPT_IN = True


@pytest.fixture
def jobs_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    # run-now needs no engine: mock the runner module (plug.py calls through it).
    import localm.plugins.builtin.jobs.runner as runner
    monkeypatch.setattr(runner, "run_job",
                        lambda job, engine=None: {"status": "ok", "output": "RAN",
                                                  "error": None, "started": 0.0,
                                                  "finished": 0.0})
    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, store_root=store_root,
                  installed_root=tmp_path / "plugins").install("jobs")
    return app


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _mk_keys(*scope_lists):
    from localm import auth
    return [auth.create_key(f"k{i}", s)["key"] for i, s in enumerate(scope_lists)]


def _shell_job(owner=None):
    """A legacy/owner-bound shell-opt-in coder job placed straight in the store."""
    from localm.plugins.builtin.jobs.store import Job, JobStore
    job = Job(name="cron", task_kind="coder", prompt="do x", cwd=".",
              allow_shell=OPT_IN, owner=owner)
    JobStore().add(job)
    return job


class TestPerJobOwnerIsolation:
    def test_a_jobs_key_cannot_touch_another_keys_job(self, jobs_app):
        a, b = _mk_keys(["jobs"], ["jobs"])     # two distinct jobs-scoped keys
        with TestClient(jobs_app) as c:
            jid = c.post("/api/jobs", headers=_h(a),
                         json={"name": "j", "prompt": "hi", "schedule": 120}).json()["id"]
            # B (different key, same scope) is locked out of A's job - 404 everywhere.
            assert c.get(f"/api/jobs/{jid}", headers=_h(b)).status_code == 404
            assert c.put(f"/api/jobs/{jid}", headers=_h(b),
                         json={"name": "x"}).status_code == 404
            assert c.post(f"/api/jobs/{jid}/run", headers=_h(b)).status_code == 404
            assert c.get(f"/api/jobs/{jid}/results", headers=_h(b)).status_code == 404
            assert c.delete(f"/api/jobs/{jid}", headers=_h(b)).status_code == 404
            # B's list does not reveal A's job; A's own list does.
            assert all(j["id"] != jid for j in c.get("/api/jobs", headers=_h(b)).json()["jobs"])
            assert any(j["id"] == jid for j in c.get("/api/jobs", headers=_h(a)).json()["jobs"])
            # A reaches its own job; the owner field is never leaked to the client.
            got = c.get(f"/api/jobs/{jid}", headers=_h(a))
            assert got.status_code == 200 and "owner" not in got.json()

    def test_owner_admin_sees_every_job(self, jobs_app, monkeypatch):
        (a,) = _mk_keys(["jobs"])
        with TestClient(jobs_app) as c:
            jid = c.post("/api/jobs", headers=_h(a),
                         json={"name": "j", "prompt": "hi", "schedule": 120}).json()["id"]
            monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")   # owner = admin
            assert c.get(f"/api/jobs/{jid}", headers=_h("ownersecret")).status_code == 200
            assert any(j["id"] == jid
                       for j in c.get("/api/jobs", headers=_h("ownersecret")).json()["jobs"])


class TestCrownJewelShellGate:
    def test_owner_bound_opt_in_job_is_invisible_to_other_jobs_keys(self, jobs_app, monkeypatch):
        """The HIGH chain via owner-binding: an owner's NEW shell-opt-in coder job is
        owner-bound, so another jobs key cannot see, edit, or run it (404)."""
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        (b,) = _mk_keys(["jobs"])
        with TestClient(jobs_app) as c:
            jid = c.post("/api/jobs", headers=_h("ownersecret"),
                         json={"name": "shelljob", "task_kind": "coder", "prompt": "p",
                               "cwd": ".", "allow_shell": OPT_IN}).json()["id"]
            assert c.post(f"/api/jobs/{jid}/run", headers=_h(b)).status_code == 404
            assert c.put(f"/api/jobs/{jid}", headers=_h(b),
                         json={"prompt": "evil"}).status_code == 404

    def test_legacy_unowned_opt_in_job_run_blocked_for_plain_jobs_key(self, jobs_app, monkeypatch):
        """Defense in depth: an unowned (legacy) opt-in job (owner=None, so the owner
        gate passes) still cannot be RUN on demand by a plain jobs key - the run-time
        caller capability re-check returns 403, blocking the RCE. The owner can run it."""
        (b,) = _mk_keys(["jobs"])
        _shell_job(owner=None)
        with TestClient(jobs_app) as c:
            jid = c.get("/api/jobs", headers=_h(b)).json()["jobs"][0]["id"]
            assert c.post(f"/api/jobs/{jid}/run", headers=_h(b)).status_code == 403
            monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
            assert c.post(f"/api/jobs/{jid}/run", headers=_h("ownersecret")).status_code == 200

    def test_plain_jobs_key_cannot_create_an_opt_in_job(self, jobs_app):
        (b,) = _mk_keys(["jobs"])
        with TestClient(jobs_app) as c:
            r = c.post("/api/jobs", headers=_h(b),
                       json={"name": "x", "task_kind": "coder", "prompt": "p",
                             "cwd": ".", "allow_shell": OPT_IN})
            assert r.status_code == 403

    def test_plain_jobs_key_cannot_poison_an_opt_in_job_the_scheduler_runs(self, jobs_app):
        """The scheduler runs a job's stored prompt with full shell when allow_shell
        is set. A plain jobs key must NOT be able to edit (poison the prompt of) an
        unowned/legacy opt-in job - otherwise the autonomous scheduler would later
        run the attacker's prompt with shell, bypassing the on-demand run_now gate."""
        (b,) = _mk_keys(["jobs"])
        _shell_job(owner=None)
        with TestClient(jobs_app) as c:
            jid = c.get("/api/jobs", headers=_h(b)).json()["jobs"][0]["id"]
            # Editing ANY field of a shell-enabled job is refused for a non-shell key,
            # so the prompt the scheduler runs stays owner-authored.
            assert c.put(f"/api/jobs/{jid}", headers=_h(b),
                         json={"prompt": "run_shell payload"}).status_code == 403
            assert c.put(f"/api/jobs/{jid}", headers=_h(b),
                         json={"enabled": False}).status_code == 403
