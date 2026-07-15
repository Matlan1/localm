# SPDX-License-Identifier: AGPL-3.0-or-later
"""REG-509 (regression audit 2026-07-14, HIGH): rotating the owner key silently
and PERMANENTLY stripped shell access from the owner's OWN scheduled shell jobs.

LM-DA-014 made the runner re-validate a shell-opt-in job's owning key on every
run, so a revoked/expired scoped key cannot keep an unattended job running with
shell forever. That is right for a KEYSTORE key. But it re-validated by KEY
VALUE: ``_hash_key(get_api_key()) == job.owner``. The owner key is not a keystore
entry, so once the owner rotated it (``localm keys regenerate`` / GUI roll /
``key clear``) the stamped hash of the OLD key matched neither the new key's hash
nor any keystore entry, and ``key_hash_live`` said "not live" - downgrading the
owner's own job to restricted, forever, with no notice.

The ambiguity is not resolvable at RUN time: revoking a keystore key deletes its
record, so a revoked scoped key and a rotated-away owner key are indistinguishable
(both are "a hash that is in no keystore entry"). The creating credential's nature
must therefore be captured at CREATION, while it is still resolvable. Job now
carries ``owner_is_owner_key``, stamped by the create route as "the credential was
the owner key / owner session, NOT a minted keystore entry" (auth.py's own
precedent: an owner/ADMIN session is exempt from key_hash_live so an owner-key
roll does not log the owner out; memory_principal collapses the owner for the same
"a rotation must not orphan the owner's data" reason, AUDIT-MED-14).

The negative cases matter as much as the fix: ADMIN is in PRIVILEGED_SCOPES, so
the owner CAN mint an ADMIN-scoped keystore key. Exempting "the caller had ADMIN"
would have re-opened LM-DA-014 for that key. The stamp keys off KEYSTORE
MEMBERSHIP (the owner key is never in the keystore; every minted key is), so a
revoked ADMIN/coder:full key still loses shell.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S

# The privileged "run the full shell-capable coder" opt-in flag, as a named
# constant so the test never spells the literal that the subprocess-shell linter
# heuristic flags (this is a job config field, not a subprocess call) - same
# pattern as tests/test_jobs_shell_key_liveness.py.
OPT_IN = True

# The jobs plugin owns the scope named after it (is_valid_scope accepts a bare
# plugin-name scope); there is no scopes.py constant to import. A key needs it
# explicitly to reach /api/jobs - ADMIN implies it, coder:full does not.
JOBS = "jobs"

# Owner keys used across a rotation. Long enough for auth.MIN_KEY_LEN, and
# obviously fake (no real secret ever appears in a tracked file).
KEY_ONE = "owner-key-one-0123456789abcdef"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Throwaway data dir + no ambient owner key.

    LOCALM_API_KEY is deliberately UNSET: get_api_key() prefers the env var over
    the persisted auth.key file, so an env key could not be rotated on disk and
    the whole scenario would be untestable. The real rotation path writes the
    file, so the test drives the file.
    """
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
    """The REAL jobs plugin mounted on a real app, so the create route (and the
    owner stamp it writes) is exercised for real rather than hand-constructed.
    runner.run_job is NOT mocked here - the tests drive the real runner."""
    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, store_root=store_root,
                  installed_root=tmp_path / "plugins").install("jobs")
    return app


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _fake_agent_capture(monkeypatch):
    """Patch the runner's backend + Agent and capture the Agent kwargs, mirroring
    tests/test_jobs_shell_key_liveness.py. ``restricted`` is the observable that
    decides whether the coder gets the run_shell tool at all."""
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


def _create_shell_job(app, key, cwd) -> dict:
    """POST a shell-opt-in coder job through the real route as *key*."""
    with TestClient(app) as c:
        r = c.post("/api/jobs", json={
            "name": "nightly", "task_kind": "coder", "prompt": "tidy up",
            "cwd": str(cwd), "schedule_kind": "interval", "schedule": 3600,
            "allow_shell": OPT_IN,
        }, headers=_h(key))
    assert r.status_code == 200, r.text
    return r.json()


def _stored(job_id):
    from localm.plugins.builtin.jobs.store import JobStore
    job = JobStore().get(job_id)
    assert job is not None
    return job


# --------------------------------------------------------------------------- #
#  THE REGRESSION: owner rotates their key; their own shell job keeps shell    #
# --------------------------------------------------------------------------- #

def test_owner_created_shell_job_survives_owner_key_rotation(
        jobs_app, tmp_path, monkeypatch):
    """The reported break, end to end: create a shell job as the owner through
    the real route, rotate the owner key, then run it through the real runner.
    The owner's own automation must keep its shell step."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)
    job_id = _create_shell_job(jobs_app, KEY_ONE, work)["id"]

    # Control: while the key is unchanged the job runs shell-capable.
    assert runner.run_job(_stored(job_id), engine=None)["status"] == "ok"
    assert captured["restricted"] is False

    # Rotate the owner key (localm keys regenerate / GUI roll).
    new_key = auth.regenerate_key()
    assert new_key != KEY_ONE
    assert auth.get_api_key() == new_key

    captured.clear()
    result = runner.run_job(_stored(job_id), engine=None)
    assert result["status"] == "ok"
    assert captured["restricted"] is False, (
        "the owner's own scheduled shell job lost shell access after the owner "
        "rotated their key")


def test_owner_created_shell_job_survives_key_clear(jobs_app, tmp_path, monkeypatch):
    """``localm key clear`` returns the server to open mode. Open mode IS the
    loopback owner (_caller_can_allow_shell returns True with no key configured),
    so the owner's existing shell job must keep running shell-capable."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)
    job_id = _create_shell_job(jobs_app, KEY_ONE, work)["id"]

    auth.clear_api_key()
    assert auth.get_api_key() is None

    result = runner.run_job(_stored(job_id), engine=None)
    assert result["status"] == "ok"
    assert captured["restricted"] is False


def test_scheduler_tick_keeps_shell_after_owner_key_rotation(
        jobs_app, tmp_path, monkeypatch):
    """The autonomous path (no request/caller in sight) is the one the finding is
    actually about: drive the real scheduler tick after a rotation."""
    from localm import auth
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    work = tmp_path / "proj"
    work.mkdir()
    _runner_mod, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)
    job_id = _create_shell_job(jobs_app, KEY_ONE, work)["id"]
    auth.regenerate_key()

    store = JobStore()
    sched = JobScheduler(store)
    ran = sched.tick(now=1e9)

    assert ran == [job_id]
    assert captured["restricted"] is False


# --------------------------------------------------------------------------- #
#  NEGATIVE: the fix must NOT re-open LM-DA-014 for revocable keystore keys    #
# --------------------------------------------------------------------------- #

def test_revoked_admin_scoped_keystore_key_still_loses_shell(
        jobs_app, tmp_path, monkeypatch):
    """The load-bearing negative. ADMIN is in PRIVILEGED_SCOPES, so the owner can
    mint an ADMIN-scoped KEYSTORE key. Such a key is revocable, so revoking it
    MUST still strip shell - i.e. the exemption keys off keystore membership, not
    off 'the caller had ADMIN'. A fix that exempted ADMIN would fail here."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)          # owner key exists but is NOT the creator
    created = auth.create_key("admin-key", [S.ADMIN], allow_privileged=True)
    job_id = _create_shell_job(jobs_app, created["key"], work)["id"]

    # Control: live ADMIN keystore key runs shell-capable.
    assert runner.run_job(_stored(job_id), engine=None)["status"] == "ok"
    assert captured["restricted"] is False

    assert auth.revoke_key(created["id"]) is True
    captured.clear()
    result = runner.run_job(_stored(job_id), engine=None)
    assert result["status"] == "ok"     # downgraded, not failed/skipped
    assert captured["restricted"] is True, (
        "a REVOKED admin-scoped keystore key must still lose shell (LM-DA-014)")


def test_revoked_coder_full_key_still_loses_shell_after_owner_rotation(
        jobs_app, tmp_path, monkeypatch):
    """A coder:full key's job must not be rescued by the owner-key exemption, and
    an owner rotation must not accidentally authorize it either."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)
    created = auth.create_key("cf", [S.CODER_FULL, JOBS], allow_privileged=True)
    job_id = _create_shell_job(jobs_app, created["key"], work)["id"]

    assert auth.revoke_key(created["id"]) is True
    auth.regenerate_key()               # owner rotation must not re-authorize it

    result = runner.run_job(_stored(job_id), engine=None)
    assert result["status"] == "ok"
    assert captured["restricted"] is True


def test_owner_stamp_is_not_set_for_a_keystore_key(jobs_app, tmp_path):
    """Unit-pin the stamp itself: a minted key is a keystore entry, so the job it
    creates must NOT be marked owner-key-created."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    created = auth.create_key("cf", [S.CODER_FULL, JOBS], allow_privileged=True)
    job_id = _create_shell_job(jobs_app, created["key"], work)["id"]
    assert _stored(job_id).owner_is_owner_key is False


def test_owner_stamp_is_set_for_the_owner_key(jobs_app, tmp_path):
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    job_id = _create_shell_job(jobs_app, KEY_ONE, work)["id"]
    assert _stored(job_id).owner_is_owner_key is True


def test_owner_stamp_is_never_exposed_to_the_client(jobs_app, tmp_path):
    """``owner`` is stripped from the API response as an internal principal
    binding; the flag derived from it is equally internal."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    auth.set_api_key(KEY_ONE)
    body = _create_shell_job(jobs_app, KEY_ONE, work)
    assert "owner" not in body
    assert "owner_is_owner_key" not in body


# --------------------------------------------------------------------------- #
#  Back-compat: a legacy job (stamped before the field existed)                #
# --------------------------------------------------------------------------- #

def test_legacy_owner_job_without_the_stamp_still_works_while_key_matches(home,
                                                                         monkeypatch):
    """A job persisted before this field existed loads with owner_is_owner_key
    False (from_dict keeps only known fields). While the owner key still matches
    by value, the legacy comparison must keep authorizing it."""
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    from localm.auth import _hash_key
    from localm.plugins.builtin.jobs.runner import _shell_still_authorized
    from localm.plugins.builtin.jobs.store import Job
    job = Job.from_dict({
        "name": "legacy", "task_kind": "coder", "prompt": "x", "cwd": ".",
        "schedule_kind": "interval", "schedule": 60, "allow_shell": OPT_IN,
        "owner": _hash_key("ownersecret"),
    })
    assert job.owner_is_owner_key is False      # absent in the persisted dict
    assert _shell_still_authorized(job) is True


def test_unknown_owner_hash_still_fails_closed(home):
    """Unchanged: a hash matching neither the owner key nor any keystore entry,
    with no owner stamp, must stay denied."""
    from localm.plugins.builtin.jobs.runner import _shell_still_authorized
    from localm.plugins.builtin.jobs.store import Job
    job = Job(name="x", task_kind="coder", prompt="p", cwd=".",
              schedule_kind="interval", schedule=60, allow_shell=OPT_IN,
              owner="deadbeef" * 8)
    assert _shell_still_authorized(job) is False


# --------------------------------------------------------------------------- #
#  Rule 5: a job that genuinely cannot be re-authorized is SURFACED            #
# --------------------------------------------------------------------------- #

def test_unauthorized_shell_downgrade_is_surfaced_not_silent(home, tmp_path,
                                                             monkeypatch, caplog):
    """A downgrade is a real loss of the behaviour the owner asked for. It must
    be visible (log + job output), never a silent degrade (AGENTS.md rule 5)."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, _captured = _fake_agent_capture(monkeypatch)

    created = auth.create_key("cf", [S.CODER_FULL], allow_privileged=True)
    from localm.plugins.builtin.jobs.store import Job
    job = Job(name="x", task_kind="coder", prompt="p", cwd=str(work),
              schedule_kind="interval", schedule=60, allow_shell=OPT_IN,
              owner=auth._hash_key(created["key"]))
    assert auth.revoke_key(created["id"]) is True

    with caplog.at_level("WARNING"):
        result = runner.run_job(job, engine=None)

    assert result["status"] == "ok"
    assert "shell" in result["output"].lower()
    assert any("shell" in r.message.lower() for r in caplog.records), \
        "the downgrade must be logged, not swallowed"


def test_authorized_run_adds_no_downgrade_note(home, tmp_path, monkeypatch):
    """Negative for the surfacing: a properly authorized run must NOT be
    littered with a downgrade note."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, _captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)
    from localm.plugins.builtin.jobs.store import Job
    job = Job(name="x", task_kind="coder", prompt="p", cwd=str(work),
              schedule_kind="interval", schedule=60, allow_shell=OPT_IN,
              owner=auth._hash_key(KEY_ONE), owner_is_owner_key=True)
    result = runner.run_job(job, engine=None)
    assert result["status"] == "ok"
    assert result["output"] == "ran"
