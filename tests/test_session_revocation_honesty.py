# SPDX-License-Identifier: AGPL-3.0-or-later
"""A session revocation that FAILED must never report success.

Every revocation helper distinguishes a store-write failure from the benign
"there was nothing to revoke" case (``0`` / ``False``), so a caller cannot
report a completed sign-out while the sessions stay live. ``localm key recover``
is the sharpest instance: unlike ``key clear`` it always configures a NEW key,
so a surviving ADMIN cookie resolves against the fresh key immediately.

These drive the real functions and assert from OUTSIDE the call. The failure is
injected at the lowest point - the store WRITE raising OSError, which is what a
locked or read-only file does - rather than by patching the function under test,
so every layer above it runs for real.
"""

from __future__ import annotations

import pytest

from localm import scopes as S

KEY = "owner-key-for-revocation-honesty-abc"


@pytest.fixture
def runner(cli_runner, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    # The store cache is process-wide; reset it so a prior test's cached view is
    # never read through this test's throwaway store.
    from localm import sessions
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    return cli_runner


def _store_writes_fail(monkeypatch):
    """Make the session store's write raise, as a locked/read-only file does.

    Patches the lowest layer that actually touches the disk, NOT the revocation
    functions under test, which would delete the behaviour being measured.
    """
    from localm import sessions

    def boom(_records):
        raise OSError(13, "The process cannot access the file")

    monkeypatch.setattr(sessions, "_save", boom)


def _mint_owner_session():
    from localm import auth, sessions
    auth.set_api_key(KEY)
    return sessions.create(scopes={S.ADMIN}, key_hash=auth._hash_key(KEY),
                           fs_access="host")


def _flat(output: str) -> str:
    """CLI output as one lowercase whitespace-normalised line.

    rich wraps console output at the terminal width, so a sentence the user
    plainly sees can carry a newline in the middle and a naive ``in`` check fails
    on a string that IS displayed. Prose wraps at spaces, so collapsing runs of
    whitespace restores it. (Do not use this to assert on a PATH - rich splits a
    long space-free string mid-word, and collapsing would insert a space.)
    """
    return " ".join(output.split()).lower()


# --------------------------------------------------------------------------- #
#  The store layer: a failure must be distinguishable from "nothing to do"     #
# --------------------------------------------------------------------------- #

class TestRevocationReportsFailureDistinguishably:
    def test_revoke_all_returns_none_on_a_write_failure(self, runner, monkeypatch):
        from localm import sessions
        sid = _mint_owner_session()
        _store_writes_fail(monkeypatch)

        assert sessions.revoke_all() is None, (
            "a failed sign-out returning 0 is indistinguishable from an empty "
            "store, which is what let every caller claim success")
        # Not merely cosmetic: prove the session genuinely SURVIVED.
        assert sessions.lookup(sid) is not None

    def test_revoke_all_returns_zero_when_there_was_genuinely_nothing(self, runner):
        from localm import sessions
        assert sessions.revoke_all() == 0      # a real answer, not a failure

    def test_revoke_returns_none_on_a_write_failure(self, runner, monkeypatch):
        from localm import sessions
        sid = _mint_owner_session()
        _store_writes_fail(monkeypatch)

        assert sessions.revoke(sid) is None
        assert sessions.lookup(sid) is not None

    def test_revoke_still_returns_false_for_an_unknown_session(self, runner):
        from localm import sessions
        _mint_owner_session()
        assert sessions.revoke("no-such-session-id") is False

    def test_revoke_by_key_hash_returns_none_on_a_write_failure(self, runner,
                                                                monkeypatch):
        from localm import auth, sessions
        auth.set_api_key(KEY)
        created = auth.create_key("device", [S.ADMIN], allow_privileged=True)
        kh = auth._hash_key(created["key"])
        sid = sessions.create(scopes={S.ADMIN}, key_hash=kh, fs_access="host")
        _store_writes_fail(monkeypatch)

        assert sessions.revoke_by_key_hash(kh) is None
        assert sessions.lookup(sid) is not None

    def test_none_stays_falsy_so_existing_truthiness_callers_are_unchanged(
            self, runner, monkeypatch):
        """None is falsy, so `if revoked:` still means "say devices were signed
        out" and never fires on a failure. Only an `is None` test learns the
        difference."""
        from localm import sessions
        _mint_owner_session()
        _store_writes_fail(monkeypatch)
        assert not sessions.revoke_all()
        assert not sessions.revoke("anything")
        assert not sessions.revoke_by_key_hash("deadbeef" * 8)

    def test_the_failure_reaches_the_local_log(self, runner, monkeypatch, caplog):
        import logging
        from localm import sessions
        _mint_owner_session()
        _store_writes_fail(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="localm"):
            sessions.revoke_all()

        warned = [r.getMessage() for r in caplog.records
                  if r.levelno >= logging.WARNING]
        assert warned, "a failed sign-out must not be silent"
        assert any("remain live" in m.lower() for m in warned), \
            "and must say that the sessions survived, not merely that a write failed"


# --------------------------------------------------------------------------- #
#  The HTTP route                                                              #
# --------------------------------------------------------------------------- #

class TestClearRouteDoesNotClaimSuccess:
    def _client(self):
        from fastapi.testclient import TestClient
        from localm.inference.http_server import create_app
        return TestClient(create_app(None))

    def test_a_clean_clear_still_reports_cleared(self, runner):
        from localm import auth
        auth.set_api_key(KEY)
        r = self._client().post("/api/auth/key/clear",
                                headers={"Authorization": f"Bearer {KEY}"})
        assert r.status_code == 200
        assert r.json() == {"cleared": True, "warnings": []}

    def test_a_failed_session_revocation_is_not_reported_as_cleared(
            self, runner, monkeypatch):
        """The credential half succeeds and the SESSION half fails: the route
        must not return cleared:true, which the GUI renders as open mode while a
        live ADMIN cookie still has full access."""
        from localm import auth
        _mint_owner_session()
        auth.set_api_key(KEY)
        _store_writes_fail(monkeypatch)

        r = self._client().post("/api/auth/key/clear",
                                headers={"Authorization": f"Bearer {KEY}"})

        assert r.status_code == 200
        body = r.json()
        assert body["cleared"] is False, (
            "cleared:true while a live admin session survives is the rule-5 "
            "violation itself")
        assert body["warnings"], "the caller must learn WHAT survived"
        assert any("session" in w.lower() for w in body["warnings"])

    def test_the_route_discloses_no_path_and_no_exception_text(
            self, runner, monkeypatch):
        """The store path carries the account name and raw OS text is
        stack-trace exposure; neither rides out on an HTTP response, matching the
        credential half."""
        from localm import auth, sessions
        _mint_owner_session()
        auth.set_api_key(KEY)
        home = str(sessions.sessions_file().parent)
        _store_writes_fail(monkeypatch)

        r = self._client().post("/api/auth/key/clear",
                                headers={"Authorization": f"Bearer {KEY}"})

        assert home not in r.text, "the data dir path must not go on the wire"
        assert "OSError" not in r.text and "cannot access" not in r.text
        assert "session" in r.text.lower(), "but it must still say WHAT survived"

    def test_logout_reports_a_failed_server_side_revocation(self, runner,
                                                            monkeypatch):
        """Deleting the cookie alone leaves a replayable server session, so a
        logout whose store write failed does not read as a clean sign-out."""
        from localm import auth, sessions
        from localm.inference import http_server as hs
        auth.set_api_key(KEY)
        sid = _mint_owner_session()
        c = self._client()
        c.cookies.set(hs.SESSION_COOKIE, sid)
        csrf = c.get("/api/session").json()["csrf"]
        _store_writes_fail(monkeypatch)

        r = c.post("/api/session/logout", headers={hs.CSRF_HEADER: csrf})

        assert r.status_code == 200
        assert r.json()["warnings"], \
            "a logout that left the session live on the server must say so"
        assert sessions.lookup(sid) is not None

    def test_a_clean_logout_reports_no_warnings(self, runner):
        from localm import auth, sessions
        from localm.inference import http_server as hs
        auth.set_api_key(KEY)
        sid = _mint_owner_session()
        c = self._client()
        c.cookies.set(hs.SESSION_COOKIE, sid)
        csrf = c.get("/api/session").json()["csrf"]

        r = c.post("/api/session/logout", headers={hs.CSRF_HEADER: csrf})

        assert r.json() == {"authed": False, "warnings": []}
        assert sessions.lookup(sid) is None


# --------------------------------------------------------------------------- #
#  The CLI                                                                     #
# --------------------------------------------------------------------------- #

class TestKeyCliDoesNotClaimSessionsWereSignedOut:
    def test_clear_success_still_says_cleared(self, runner):
        from localm import auth
        from localm.cli import main
        auth.set_api_key(KEY)
        _mint_owner_session()
        r = runner.invoke(main, ["key", "clear", "--yes"])
        assert r.exit_code == 0
        out = _flat(r.output)
        assert "were signed out" in out
        assert "not signed out" not in out

    def test_clear_does_not_print_a_success_tick_when_sessions_survive(
            self, runner, monkeypatch):
        from localm import auth
        from localm.cli import main
        auth.set_api_key(KEY)
        _mint_owner_session()
        _store_writes_fail(monkeypatch)

        r = runner.invoke(main, ["key", "clear", "--yes"])

        out = _flat(r.output)
        assert "not signed out" in out, \
            "a failed sign-out reported as success is the rule-5 violation itself"
        assert "may still have access" in out
    # The unconditional success wording is pinned as absent.
        assert "api key cleared - open mode" not in out

    def test_recover_does_not_claim_a_completed_recovery(self, runner, monkeypatch):
        """The strongest instance: recovery always configures a NEW key, so a
        surviving ADMIN cookie resolves against it immediately."""
        from localm import auth, sessions
        from localm.cli import main
        auth.set_api_key(KEY)
        sid = _mint_owner_session()
        _store_writes_fail(monkeypatch)

        r = runner.invoke(main, ["key", "recover"])

        out = _flat(r.output)
        assert "not signed out" in out
        assert "may still have access" in out
        assert "browser sessions were reset" not in out, \
            "the reassurance must not print when the sign-out did not happen"
        assert sessions.lookup(sid) is not None

    def test_recover_still_prints_the_new_key_on_a_session_failure(
            self, runner, monkeypatch):
        """Honesty must not cost the user the key: it is shown exactly once and
        is unrecoverable, and it IS usable. Reported, not failed."""
        from localm import auth
        from localm.cli import main
        auth.set_api_key(KEY)
        _mint_owner_session()
        _store_writes_fail(monkeypatch)

        r = runner.invoke(main, ["key", "recover"])

        assert r.exit_code == 0, "the exit-code contract is not this fix's to change"
        assert auth.get_api_key() is not None
        assert auth.get_api_key() in r.output, "the once-only key must still be shown"

    def test_recover_success_still_reports_the_reset(self, runner):
        from localm import auth
        from localm.cli import main
        auth.set_api_key(KEY)
        _mint_owner_session()
        r = runner.invoke(main, ["key", "recover"])
        assert r.exit_code == 0
        assert "browser sessions were reset" in _flat(r.output)


# --------------------------------------------------------------------------- #
#  The two properties must not be satisfied by breaking each other             #
# --------------------------------------------------------------------------- #

def test_a_key_roll_still_leaves_sessions_alone(runner):
    """A key ROLL (regenerate_key) does NOT sign the browser out; a scheduled
    job created over the surviving session depends on that. Revocation happens
    only when something ASKED for it (clear / recover / logout)."""
    from localm import auth, sessions
    sid = _mint_owner_session()

    auth.regenerate_key()

    assert sessions.lookup(sid) is not None, \
        "a key roll must not sign the browser out"
    assert sessions.lookup(sid)["owner_key_minted"] is False  # created() default here
    assert sessions.revoke_all() == 1                          # but an ASK still works
    assert sessions.lookup(sid) is None
