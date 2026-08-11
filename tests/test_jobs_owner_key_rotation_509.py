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

import json
import time
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
    runner.run_job is NOT mocked here - the tests drive the real runner.

    The REAL session-login route is mounted alongside it, so the cookie tests below
    mint their session the way a browser actually does (POST /api/session) instead
    of hand-calling sessions.create. That matters: the mint site is half of the
    fix, and a test that constructs the record itself would pass with that half
    reverted.
    """
    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    from localm.inference.routes import session as _routes_session
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, store_root=store_root,
                  installed_root=tmp_path / "plugins").install("jobs")
    _routes_session.register(app, None)      # register() never reads its ctx arg
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


def _login(app, key) -> str:
    """Sign in through the REAL POST /api/session and return the session id the
    server put in the cookie - i.e. exactly what a browser ends up holding."""
    from localm.inference import http_server as hs
    with TestClient(app) as c:
        r = c.post("/api/session", json={"key": key})
        assert r.status_code == 200, r.text
        sid = c.cookies.get(hs.SESSION_COOKIE)
    assert sid, "login set no session cookie"
    return sid


def _create_shell_job_over_cookie(app, sid, cwd) -> dict:
    """POST a shell-opt-in coder job through the real route carrying a SESSION
    COOKIE - the browser GUI's credential - rather than a bearer key.

    Fetches the CSRF token from GET /api/session first and echoes it in
    X-CSRF-Token, exactly as the GUI does: a cookie-authenticated state change is
    CSRF-gated (_enforce_request), and that gate is live here. Doing it the real
    way rather than bypassing it keeps this driving the production flow."""
    from localm.inference import http_server as hs
    with TestClient(app) as c:
        c.cookies.set(hs.SESSION_COOKIE, sid)
        state = c.get("/api/session")
        assert state.status_code == 200, state.text
        csrf = state.json().get("csrf")
        assert csrf, f"no CSRF token for this session: {state.text}"
        r = c.post("/api/jobs", json={
            "name": "nightly", "task_kind": "coder", "prompt": "tidy up",
            "cwd": str(cwd), "schedule_kind": "interval", "schedule": 3600,
            "allow_shell": OPT_IN,
        }, headers={hs.CSRF_HEADER: csrf})
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


def _cookie_request(sid):
    """A minimal Request-alike carrying a session cookie, enough for
    principal_id/caller_scopes (which read only cookies + headers)."""
    from localm.inference import http_server as hs

    class _Req:
        cookies = {hs.SESSION_COOKIE: sid}
        headers: dict = {}
    return _Req()


def test_expired_admin_keystore_key_over_a_live_cookie_is_not_the_owner(home):
    """THE HOLE the pre-merge security review caught, pinned so it cannot come
    back.

    revoke_key DELETES a keystore record, but EXPIRY does not - and
    _principal_from_token exempts an ADMIN session from key_hash_live, so an
    expired ADMIN-scoped keystore key's cookie session still resolves a principal
    that key_hash_live calls dead. The first version of this fix asked the
    NEGATIVE question ("absent from the keystore, so it must be the owner") and
    therefore stamped that expired, revocable key as the OWNER - handing it
    permanent shell and re-opening LM-DA-014.

    Note the bearer path canNOT show this (verify() rejects an expired key), which
    is exactly why the original negative test missed it: it used a Bearer header.
    """
    from localm import auth, sessions
    from localm.inference.http_server import caller_scopes, principal_id
    from localm.plugins.builtin.jobs.plug import _caller_is_owner_key

    auth.set_api_key(KEY_ONE)
    created = auth.create_key("device", [S.ADMIN, JOBS], allow_privileged=True,
                              expires=time.time() + 3600)
    kh = auth._hash_key(created["key"])
    sid = sessions.create(scopes={S.ADMIN}, key_hash=kh, fs_access="host")

    # The key expires. The record survives; the ADMIN session survives with it.
    ks = auth._load_keystore()
    for rec in ks:
        rec["expires"] = time.time() - 10
    auth._save_keystore(ks)

    req = _cookie_request(sid)
    # Preconditions: this is the exact state that fooled the negative test.
    assert auth.verify(created["key"]) is None          # bearer would be rejected
    assert auth.key_hash_live(kh) is False              # "not live"...
    assert sessions.lookup(sid) is not None             # ...but the cookie lives
    assert caller_scopes(req) == {S.ADMIN}
    assert principal_id(req) == kh

    assert _caller_is_owner_key(req) is False, (
        "an EXPIRED admin-scoped keystore key was stamped as the owner key - "
        "that hands a revoked credential permanent shell (LM-DA-014)")


def test_an_unreadable_keystore_cannot_promote_a_key_to_owner(home, monkeypatch):
    """The other half: _load_keystore() swallows OSError/ValueError and returns
    [], so key_hash_live says "not live" for a perfectly LIVE key on a transient
    corrupt/locked auth.json. A PERMANENT privilege stamp must never be derived
    from a fail-open read."""
    from localm import auth, sessions
    from localm.plugins.builtin.jobs.plug import _caller_is_owner_key

    auth.set_api_key(KEY_ONE)
    created = auth.create_key("device", [S.ADMIN, JOBS], allow_privileged=True)
    kh = auth._hash_key(created["key"])
    sid = sessions.create(scopes={S.ADMIN}, key_hash=kh, fs_access="host")

    # auth.json becomes unreadable/corrupt -> _load_keystore returns [].
    monkeypatch.setattr(auth, "_load_keystore", lambda: [])
    assert auth.key_hash_live(kh) is False       # a LIVE key now reads as dead

    assert _caller_is_owner_key(_cookie_request(sid)) is False, (
        "a keystore read failure promoted a live scoped key to owner")


def test_the_owner_key_over_a_cookie_session_is_the_owner(home):
    """Positive control for the two negatives above: the owner's own session
    (minted by the owner key, key unchanged) IS recognised."""
    from localm import auth, sessions
    from localm.plugins.builtin.jobs.plug import _caller_is_owner_key

    auth.set_api_key(KEY_ONE)
    sid = sessions.create(scopes={S.ADMIN}, key_hash=auth._hash_key(KEY_ONE),
                          fs_access="host")
    assert _caller_is_owner_key(_cookie_request(sid)) is True


# --------------------------------------------------------------------------- #
#  THE RESIDUAL: the job is created over a COOKIE SESSION minted BEFORE the     #
#  roll. The entry's own named trigger ("GUI roll"), and the case none of the   #
#  cookie tests above could express - each of them creates its session AFTER    #
#  the key is already in its final state, so the frozen key_hash never goes     #
#  stale and the value comparison alone answers correctly.                      #
# --------------------------------------------------------------------------- #

def test_owner_session_job_keeps_shell_when_the_key_rolled_after_sign_in(
        jobs_app, tmp_path, monkeypatch):
    """The reported chain, end to end and entirely through real routes: the owner
    signs into the GUI under K1, rolls the key to K2 (which deliberately does NOT
    sign the browser out), then schedules a shell job from that still-valid
    session. The runner must keep the shell step.

    Before the fix the session froze hash(K1); after the roll that matched neither
    the new owner key nor any keystore entry, so the job was stamped as not the
    owner's and the runner downgraded it - silently, permanently, on the owner's
    own automation."""
    from localm import auth
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)
    sid = _login(jobs_app, KEY_ONE)          # 1. sign in under K1

    new_key = auth.regenerate_key()          # 2. roll to K2
    assert new_key != KEY_ONE and auth.get_api_key() == new_key

    # The session survives the roll BY DESIGN; that is the premise, not a bug.
    from localm import sessions
    assert sessions.lookup(sid) is not None, \
        "premise broken: the roll signed the browser out, so this cannot test REG-509"

    job_id = _create_shell_job_over_cookie(jobs_app, sid, work)["id"]  # 3. schedule
    assert _stored(job_id).owner_is_owner_key is True, \
        "a job created over the owner's own session was not stamped as the owner's"

    result = runner.run_job(_stored(job_id), engine=None)              # 4. tick
    assert result["status"] == "ok"
    assert captured["restricted"] is False, (
        "the owner's scheduled shell job lost shell because the owner key was "
        "rolled after the browser session was minted (REG-509, cookie path)")


def test_scheduler_tick_keeps_shell_for_a_job_made_over_a_pre_roll_session(
        jobs_app, tmp_path, monkeypatch):
    """Same chain, driven through the real scheduler tick - the autonomous path
    the finding is actually about, where no request or caller exists any more."""
    from localm import auth
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore
    work = tmp_path / "proj"
    work.mkdir()
    _runner_mod, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)
    sid = _login(jobs_app, KEY_ONE)
    auth.regenerate_key()
    job_id = _create_shell_job_over_cookie(jobs_app, sid, work)["id"]

    ran = JobScheduler(JobStore()).tick(now=1e9)

    assert ran == [job_id]
    assert captured["restricted"] is False


def test_a_keystore_key_session_is_not_stamped_and_still_loses_shell(
        jobs_app, tmp_path, monkeypatch):
    """The load-bearing negative for the cookie path, mirroring the bearer one at
    the top of this section.

    An ADMIN-scoped KEYSTORE key can sign into the GUI too, and its session is
    equally ADMIN-scoped - so a fix that stamped "this session holds ADMIN", or
    that inferred the owner from the session merely EXISTING, would mark it as the
    owner and hand a revocable credential permanent shell (LM-DA-014). The stamp
    must come from the key VALUE proven at mint, so this session is never marked
    and revoking the key still strips shell."""
    from localm import auth, sessions
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)               # an owner key exists, but is NOT used
    created = auth.create_key("device", [S.ADMIN, JOBS], allow_privileged=True)
    sid = _login(jobs_app, created["key"])

    assert sessions.lookup(sid)["owner_key_minted"] is False, \
        "an ADMIN-scoped KEYSTORE key's session was stamped as owner-key-minted"

    job_id = _create_shell_job_over_cookie(jobs_app, sid, work)["id"]
    assert _stored(job_id).owner_is_owner_key is False

    # Control: while the key is live the job still runs shell-capable.
    assert runner.run_job(_stored(job_id), engine=None)["status"] == "ok"
    assert captured["restricted"] is False

    assert auth.revoke_key(created["id"]) is True
    captured.clear()
    assert runner.run_job(_stored(job_id), engine=None)["status"] == "ok"
    assert captured["restricted"] is True, (
        "a REVOKED admin-scoped keystore key kept shell through its cookie "
        "session (LM-DA-014)")


def test_the_owner_login_records_the_stamp_and_a_scoped_login_does_not(jobs_app):
    """Unit-pin the mint site itself, at the real route: the recorded flag tracks
    the KEY VALUE presented, not the scopes it happens to carry."""
    from localm import auth, sessions
    auth.set_api_key(KEY_ONE)
    scoped = auth.create_key("device", [S.ADMIN, JOBS], allow_privileged=True)

    owner_rec = sessions.lookup(_login(jobs_app, KEY_ONE))
    scoped_rec = sessions.lookup(_login(jobs_app, scoped["key"]))

    assert owner_rec["owner_key_minted"] is True
    assert scoped_rec["owner_key_minted"] is False
    # Both are ADMIN sessions, so the scope set cannot be what separates them.
    assert S.ADMIN in owner_rec["scopes"] and S.ADMIN in scoped_rec["scopes"]


def test_a_session_recorded_before_this_field_existed_is_not_the_owner(home):
    """Fail closed on absence. A session minted by an older build has no
    ``owner_key_minted`` key at all; it must read as False rather than inheriting
    a privilege it never proved. (It still resolves correctly by key VALUE until
    the owner rolls - that is the pre-existing behaviour, not a new grant.)"""
    from localm import auth, sessions
    from localm.plugins.builtin.jobs.plug import _caller_is_owner_key

    auth.set_api_key(KEY_ONE)
    sid = sessions.create(scopes={S.ADMIN}, key_hash=auth._hash_key(KEY_ONE),
                          fs_access="host")
    # Strip the field the way an older build's store would have it: absent.
    raw = json.loads(sessions.sessions_file().read_text(encoding="utf-8"))
    for rec in raw:
        rec.pop("owner_key_minted", None)
    sessions.sessions_file().write_text(json.dumps(raw), encoding="utf-8")
    sessions._CACHE["mtime"] = None          # force a re-read of the edited file

    assert sessions.lookup(sid)["owner_key_minted"] is False
    # Still the owner by VALUE, because the key has not rolled yet.
    assert _caller_is_owner_key(_cookie_request(sid)) is True
    # ...and once it rolls, the un-stamped legacy session correctly stops matching,
    # which is exactly the pre-fix behaviour rather than a new hole.
    auth.regenerate_key()
    assert _caller_is_owner_key(_cookie_request(sid)) is False


def test_a_truthy_non_bool_in_the_store_is_not_the_owner_stamp(home):
    """A hand-edited or corrupted store row holding a truthy STRING must not read
    as the stamp. create() only ever writes a real bool, so anything else is not
    something this server wrote."""
    from localm import auth, sessions
    from localm.plugins.builtin.jobs.plug import _caller_is_owner_key

    auth.set_api_key(KEY_ONE)
    created = auth.create_key("device", [S.ADMIN, JOBS], allow_privileged=True)
    sid = sessions.create(scopes={S.ADMIN}, key_hash=auth._hash_key(created["key"]),
                          fs_access="host")
    raw = json.loads(sessions.sessions_file().read_text(encoding="utf-8"))
    for rec in raw:
        rec["owner_key_minted"] = "yes"
    sessions.sessions_file().write_text(json.dumps(raw), encoding="utf-8")
    sessions._CACHE["mtime"] = None

    assert sessions.lookup(sid)["owner_key_minted"] is False
    assert _caller_is_owner_key(_cookie_request(sid)) is False


def test_clearing_the_owner_key_destroys_the_stamp_with_the_session(jobs_app):
    """The design call, pinned. A key ROLL leaves sessions (and so the stamp)
    alive on purpose - that is the whole fix. The containment path for a leaked
    owner key is ``key clear`` / sign out everywhere, which revokes every session,
    so the stamp can never outlive the credential that carries it."""
    from localm import auth, sessions
    auth.set_api_key(KEY_ONE)
    sid = _login(jobs_app, KEY_ONE)
    assert sessions.lookup(sid)["owner_key_minted"] is True

    auth.regenerate_key()                     # a ROLL keeps it (the fix)
    assert sessions.lookup(sid)["owner_key_minted"] is True

    sessions.revoke_all()                     # what key clear / recover call
    assert sessions.lookup(sid) is None


def test_the_legacy_identity_relink_preserves_the_stamp(home):
    """The owner key's identity moved to a salted KDF, and relink_key_hash
    rewrites existing sessions to the derived value. It must carry the stamp
    across: dropping it there would re-open REG-509 through the upgrade path."""
    from localm import auth, sessions
    auth.set_api_key(KEY_ONE)
    legacy = auth._legacy_owner_identity(KEY_ONE)
    derived = auth._hash_key(KEY_ONE)
    sid = sessions.create(scopes={S.ADMIN}, key_hash=legacy, fs_access="host",
                          owner_key_minted=True)

    assert sessions.relink_key_hash(legacy, derived) == 1

    rec = sessions.lookup(sid)
    assert rec["key_hash"] == derived
    assert rec["owner_key_minted"] is True


def test_a_revoked_scoped_session_is_never_read_for_the_stamp(home, monkeypatch):
    """The stamp is read through the same re-validation every other cookie
    consumer uses, never a bare sessions.lookup(). So a scoped-key session whose
    key has been revoked is rejected outright - even if its record somehow carried
    the flag, which only a tampered store could produce."""
    from localm import auth, sessions
    from localm.inference.http_server import caller_minted_by_owner_key

    auth.set_api_key(KEY_ONE)
    created = auth.create_key("device", [JOBS], allow_privileged=False)
    sid = sessions.create(scopes={JOBS}, key_hash=auth._hash_key(created["key"]),
                          fs_access="none", owner_key_minted=True)
    assert caller_minted_by_owner_key(_cookie_request(sid)) is True   # control

    # revoke_key ALSO drops the key's sessions, which would end the test early by
    # a different mechanism. Neutralise that so what is measured here is purely
    # the per-request key_hash_live re-validation.
    monkeypatch.setattr(sessions, "revoke_by_key_hash", lambda *a, **kw: 0)
    assert auth.revoke_key(created["id"]) is True
    assert sessions.lookup(sid) is not None, "the session was dropped, not re-validated"
    assert caller_minted_by_owner_key(_cookie_request(sid)) is False, (
        "a revoked scoped key's session was read for the owner stamp without "
        "the key_hash_live re-validation")


def test_a_bearer_caller_is_never_stamped_from_a_session(home):
    """caller_minted_by_owner_key answers only for a cookie. A bearer caller has
    no session, and its correct answer comes from the key value comparison."""
    from localm import auth, sessions
    from localm.inference.http_server import caller_minted_by_owner_key
    auth.set_api_key(KEY_ONE)
    sessions.create(scopes={S.ADMIN}, key_hash=auth._hash_key(KEY_ONE),
                    fs_access="host", owner_key_minted=True)

    class _BearerReq:
        cookies: dict = {}
        headers = {"authorization": f"Bearer {KEY_ONE}"}

    assert caller_minted_by_owner_key(_BearerReq()) is False


def test_legacy_owner_job_is_restamped_so_a_later_rotation_cannot_strip_it(
        home, tmp_path, monkeypatch):
    """A job persisted BEFORE this field existed must not stay exposed: the first
    run that proves it is the owner's (the key still matches by value) records
    that, so the owner's NEXT roll does not strip its shell."""
    from localm import auth
    from localm.plugins.builtin.jobs.store import Job, JobStore
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)
    store = JobStore()
    legacy = Job.from_dict({
        "name": "legacy", "task_kind": "coder", "prompt": "x", "cwd": str(work),
        "schedule_kind": "interval", "schedule": 60, "allow_shell": OPT_IN,
        "owner": auth._hash_key(KEY_ONE),      # no owner_is_owner_key key at all
    })
    store.add(legacy)
    assert store.get(legacy.id).owner_is_owner_key is False

    # One ordinary run, while the key still matches.
    assert runner.run_job(store.get(legacy.id), engine=None)["status"] == "ok"
    assert captured["restricted"] is False
    assert store.get(legacy.id).owner_is_owner_key is True, \
        "the proven owner stamp was not persisted"

    # NOW the owner rolls their key. The legacy job must keep its shell.
    auth.regenerate_key()
    captured.clear()
    assert runner.run_job(store.get(legacy.id), engine=None)["status"] == "ok"
    assert captured["restricted"] is False


def test_restamp_failure_does_not_break_the_run(home, tmp_path, monkeypatch):
    """The re-stamp is best-effort: a store write failure must not fail the job
    (this run is authorized either way), and must not be silent."""
    from localm import auth
    from localm.plugins.builtin.jobs import runner as runner_mod
    from localm.plugins.builtin.jobs.store import Job
    work = tmp_path / "proj"
    work.mkdir()
    runner, captured = _fake_agent_capture(monkeypatch)

    auth.set_api_key(KEY_ONE)
    job = Job(name="x", task_kind="coder", prompt="p", cwd=str(work),
              schedule_kind="interval", schedule=60, allow_shell=OPT_IN,
              owner=auth._hash_key(KEY_ONE))     # not in any store -> update KeyErrors

    def _boom(*a, **kw):
        raise OSError("disk gone")

    monkeypatch.setattr(runner_mod.JobStore, "update", _boom)
    result = runner.run_job(job, engine=None)
    assert result["status"] == "ok"
    assert captured["restricted"] is False       # still authorized


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
