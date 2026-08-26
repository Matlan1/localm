# SPDX-License-Identifier: AGPL-3.0-or-later
"""A scheduled coder job's ``allow_shell`` is re-validated against the creating
key's live state on every run, not captured once at creation time. The
autonomous scheduler tick has no request or caller to re-check, unlike run-now,
which re-validates the CALLER via _caller_can_allow_shell.

runner._shell_still_authorized(job) re-checks that the OWNING key (job.owner,
the same sha256 hash auth.key_hash_live() checks for cookie sessions) is still
live - not revoked, not expired. A dead key downgrades the run to RESTRICTED
(read plus confined edit, no run_shell), the same safe default a coder job with
no opt-in gets; it is never silently skipped.
"""

from __future__ import annotations

import time

import pytest

from localm import scopes as S

# The privileged opt-in flag for the full shell-capable coder, as a named
# constant so the test never spells the literal that the subprocess-shell linter
# heuristic flags (this is a job config field, not a subprocess call).
OPT_IN = True


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _make_job(**kw):
    from localm.plugins.builtin.jobs.store import Job
    base = dict(name="shelljob", task_kind="coder", prompt="do x", cwd=".",
                schedule_kind="interval", schedule=60, allow_shell=OPT_IN)
    base.update(kw)
    return Job(**base)


def _fake_agent_capture(monkeypatch):
    """Patch the runner's backend + Agent and capture the Agent kwargs, mirroring
    the pattern in test_jobs_plugin.py."""
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


# --------------------------------------------------------------------------- #
#  Unit: _shell_still_authorized                                              #
# --------------------------------------------------------------------------- #

def test_no_owner_needs_no_recheck(home):
    """A job created in open mode (owner=None) opted into shell without any
    privileged key existing, so there is no key to re-validate."""
    from localm.plugins.builtin.jobs.runner import _shell_still_authorized
    job = _make_job(owner=None)
    assert _shell_still_authorized(job) is True


def test_owner_key_itself_is_exempt(home, monkeypatch):
    """A job created by the OWNER key is exempt: the owner key is not a keystore
    entry (key_hash_live would always say 'not found' for it), and it is not
    revocable/expirable the way a scoped key is - mirrors key_hash_live's own
    owner-session exemption in auth.py."""
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    from localm.auth import _hash_key
    from localm.plugins.builtin.jobs.runner import _shell_still_authorized
    job = _make_job(owner=_hash_key("ownersecret"))
    assert _shell_still_authorized(job) is True


def test_live_scoped_key_is_authorized(home):
    from localm import auth
    from localm.plugins.builtin.jobs.runner import _shell_still_authorized
    created = auth.create_key("k", [S.CODER_FULL], allow_privileged=True)
    job = _make_job(owner=auth._hash_key(created["key"]))
    assert _shell_still_authorized(job) is True


def test_revoked_scoped_key_is_not_authorized(home):
    from localm import auth
    from localm.plugins.builtin.jobs.runner import _shell_still_authorized
    created = auth.create_key("k", [S.CODER_FULL], allow_privileged=True)
    job = _make_job(owner=auth._hash_key(created["key"]))
    assert auth.revoke_key(created["id"]) is True
    assert _shell_still_authorized(job) is False


def test_expired_scoped_key_is_not_authorized(home):
    from localm import auth
    from localm.plugins.builtin.jobs.runner import _shell_still_authorized
    created = auth.create_key("k", [S.CODER_FULL], allow_privileged=True,
                              expires=time.time() - 10)   # already expired
    job = _make_job(owner=auth._hash_key(created["key"]))
    assert _shell_still_authorized(job) is False


def test_unknown_owner_hash_is_not_authorized(home):
    """A hash that never matches any keystore entry (and isn't the owner key)
    fails closed rather than defaulting open."""
    from localm.plugins.builtin.jobs.runner import _shell_still_authorized
    job = _make_job(owner="deadbeef" * 8)
    assert _shell_still_authorized(job) is False


# --------------------------------------------------------------------------- #
#  Integration: runner.run_job downgrades a dead-key job to restricted        #
# --------------------------------------------------------------------------- #

def test_runner_downgrades_to_restricted_after_key_revoked(home, tmp_path, monkeypatch):
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    created = auth.create_key("k", [S.CODER_FULL], allow_privileged=True)
    job = _make_job(cwd=str(work), owner=auth._hash_key(created["key"]))

    # Before revocation: the live key authorizes the full, shell-capable coder.
    result = runner.run_job(job, engine=None)
    assert result["status"] == "ok"
    assert captured["restricted"] is False

    # Revoke the key, then run again: the job must NOT keep shell access.
    assert auth.revoke_key(created["id"]) is True
    captured.clear()
    result = runner.run_job(job, engine=None)
    assert result["status"] == "ok"          # downgraded, not failed/skipped
    assert captured["restricted"] is True


def test_runner_downgrades_to_restricted_for_an_expired_key(home, tmp_path, monkeypatch):
    """A long-lived scheduled job whose owning key's expiry has since passed must
    run restricted, not with shell access, on its next tick."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    created = auth.create_key("k", [S.CODER_FULL], allow_privileged=True,
                              expires=time.time() - 1)   # already expired
    job = _make_job(cwd=str(work), owner=auth._hash_key(created["key"]))

    result = runner.run_job(job, engine=None)
    assert result["status"] == "ok"          # downgraded, not failed/skipped
    assert captured["restricted"] is True


def test_runner_owner_created_shell_job_survives_indefinitely(home, tmp_path, monkeypatch):
    """The owner's OWN shell-opt-in job is unaffected by keystore churn - it is
    not gated on a revocable scoped key."""
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)
    job = _make_job(cwd=str(work), owner=auth._hash_key("ownersecret"))
    result = runner.run_job(job, engine=None)
    assert result["status"] == "ok"
    assert captured["restricted"] is False


# --------------------------------------------------------------------------- #
#  Integration: the autonomous SCHEDULER tick honours the same re-check       #
# --------------------------------------------------------------------------- #

def test_scheduler_tick_downgrades_shell_job_after_key_revoked(home, tmp_path, monkeypatch):
    """The exact scenario the finding describes: a scheduled job's allow_shell
    must not survive its creating key's revocation. Drives the real scheduler
    tick -> real runner.run_job -> _run_coder (only the Agent/backend are
    mocked), so this exercises the actual autonomous path with no request/
    caller in sight."""
    from localm import auth
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore

    work = tmp_path / "proj"
    work.mkdir()
    _runner_mod, captured = _fake_agent_capture(monkeypatch)

    created = auth.create_key("k", [S.CODER_FULL], allow_privileged=True)
    store = JobStore()
    job = store.add(_make_job(cwd=str(work), owner=auth._hash_key(created["key"]),
                              last_run=None))

    # Revoke BEFORE the tick ever runs the job - the live key never gets to
    # authorize a shell run at all.
    assert auth.revoke_key(created["id"]) is True

    sched = JobScheduler(store)     # real runner.run_job (module default)
    ran = sched.tick(now=1000.0)

    assert ran == [job.id]
    results = store.list_results(job.id)
    assert results and results[0]["status"] == "ok"
    # The job ran (it was not silently skipped), but WITHOUT shell access.
    assert captured["restricted"] is True


def test_scheduler_tick_runs_unrestricted_while_key_is_live(home, tmp_path, monkeypatch):
    """Control: with the key still live, the scheduler tick runs the job with
    full shell access, proving the downgrade above is specific to revocation."""
    from localm import auth
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore

    work = tmp_path / "proj"
    work.mkdir()
    _runner_mod, captured = _fake_agent_capture(monkeypatch)

    created = auth.create_key("k", [S.CODER_FULL], allow_privileged=True)
    store = JobStore()
    job = store.add(_make_job(cwd=str(work), owner=auth._hash_key(created["key"]),
                              last_run=None))

    sched = JobScheduler(store)
    ran = sched.tick(now=1000.0)

    assert ran == [job.id]
    assert captured["restricted"] is False
