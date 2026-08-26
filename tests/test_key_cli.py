"""The `localm key` CLI group.

show / generate / set / clear / list / create / rm over the owner key and the
scoped named-key store. Secrets are masked by default and shown in full only
once at creation; privileged scopes can never be self-minted.
"""

import re

import pytest

from localm import auth
from localm.cli import main

_SECRET_RE = re.compile(r"[A-Za-z0-9_\-]{30,}")


@pytest.fixture
def runner(cli_runner, monkeypatch):
    # Drop any ambient owner key / require-auth so "open mode" assertions are
    # about the throwaway home, not the developer's environment.
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    return cli_runner


# --------------------------------------------------------------------------- #
#  Owner key: show / generate / set / clear                                    #
# --------------------------------------------------------------------------- #

class TestOwnerKey:
    def test_show_open_mode(self, runner):
        r = runner.invoke(main, ["key", "show"])
        assert r.exit_code == 0
        assert "open mode" in r.output.lower()

    def test_generate_persists_and_shows_once(self, runner):
        r = runner.invoke(main, ["key", "generate"])
        assert r.exit_code == 0
        key = auth.get_api_key()
        assert key is not None                      # a key was generated
        assert key in r.output                      # shown so the user can copy
        assert r.output.count(key) == 1             # never echoed twice

    def test_show_masks_by_default(self, runner):
        runner.invoke(main, ["key", "generate"])
        key = auth.get_api_key()
        r = runner.invoke(main, ["key", "show"])
        assert r.exit_code == 0
        assert key not in r.output                  # full secret not leaked
        assert key[:4] in r.output                  # masked preview present

    def test_show_reveal_prints_full(self, runner):
        runner.invoke(main, ["key", "generate"])
        key = auth.get_api_key()
        r = runner.invoke(main, ["key", "show", "--reveal"])
        assert r.exit_code == 0
        assert key in r.output

    def test_set_persists_specific_key(self, runner):
        secret = "my-explicit-owner-key-1234567890"
        r = runner.invoke(main, ["key", "set", secret])
        assert r.exit_code == 0
        assert auth.get_api_key() == secret
        assert secret not in r.output               # set echoes a masked preview

    def test_set_refuses_non_ascii_key_and_states_allowed_chars(self, runner):
        """A non-ASCII key cannot ride in an HTTP Authorization header (clients
        send UTF-8, RFC 7230 obs-text decodes latin-1), so `key set` refuses it and
        says which characters ARE allowed. It must read as a clean user error:
        letting set_api_key's ValueError escape routes it to the crash handler,
        which asks the user to file their own typo as a localm bug."""
        r = runner.invoke(main, ["key", "set", "pässwort-key"])
        assert r.exit_code == 1
        assert "letters, numbers" in r.output        # states what IS allowed
        assert "'-'" in r.output and "'_'" in r.output
        assert auth.get_api_key() is None            # NEGATIVE: nothing persisted
        assert "unexpected error" not in r.output    # not the crash/bug-report path
        assert "bug" not in r.output.lower()

    def test_set_accepts_a_generated_key_verbatim(self, runner):
        """`key generate` emits dashes AND underscores, so `key set` must accept
        both - a charset allowing only "-" would reject ~half the generated keys."""
        r = runner.invoke(main, ["key", "set", "Gen_key-With_Both-123456"])
        assert r.exit_code == 0
        assert auth.get_api_key() == "Gen_key-With_Both-123456"

    def test_set_refuses_short_key_as_a_user_error(self, runner):
        """Same presentation for the pre-existing MIN_KEY_LEN rejection."""
        r = runner.invoke(main, ["key", "set", "short"])
        assert r.exit_code == 1
        assert "8 characters" in r.output
        assert auth.get_api_key() is None
        assert "unexpected error" not in r.output

    def test_clear_returns_to_open_mode(self, runner):
        runner.invoke(main, ["key", "generate"])
        assert auth.get_api_key() is not None
        r = runner.invoke(main, ["key", "clear", "--yes"])
        assert r.exit_code == 0
        assert auth.get_api_key() is None           # NEGATIVE: clear truly clears
        assert "open mode" in runner.invoke(main, ["key", "show"]).output.lower()

    def test_clear_open_mode_is_noop(self, runner):
        r = runner.invoke(main, ["key", "clear", "--yes"])
        assert r.exit_code == 0
        assert "already open mode" in r.output.lower()


# --------------------------------------------------------------------------- #
#  recover / clear must revoke live browser sessions                           #
# --------------------------------------------------------------------------- #

class TestOwnerSessionRevocation:
    """`key recover` / `key clear` must sign out live browser sessions.

    Owner (ADMIN) browser sessions are decoupled from the key value and are
    exempt from the per-request keystore recheck, so they SURVIVE an owner-key roll
    by design - correct for the GUI's own roll, but the CLI recovery path exists to
    lock a compromised owner OUT, so a captured owner cookie must not outlive it.
    Both commands mirror /api/auth/key/clear and call sessions.revoke_all(); scoped
    DEVICE keys in the keystore stay untouched (only browser SESSIONS are dropped).
    """

    _KEY = "owner-key-for-revoke-test-abcdef"

    def _mint_owner_session(self):
        from localm import auth, sessions
        from localm import scopes as S
        auth.set_api_key(self._KEY)
        return sessions.create(scopes={S.ADMIN},
                               key_hash=auth._hash_key(self._KEY),
                               fs_access="host")

    def test_recover_revokes_browser_sessions(self, runner, monkeypatch):
        from localm import auth, sessions
        # Module-level session cache is process-wide; reset it so this test's
        # throwaway store is never read through a prior test's cached view.
        monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
        sid = self._mint_owner_session()
        # A scoped DEVICE key exists too; recovery must rotate the owner key and
        # drop sessions but leave the device key working.
        auth.create_key("phone", ["models:read"], allow_privileged=False)
        assert sessions.lookup(sid) is not None            # session valid pre-recover
        r = runner.invoke(main, ["key", "recover"])
        assert r.exit_code == 0, r.output
        assert sessions.lookup(sid) is None                # NEGATIVE: owner cookie dead
        assert len(auth.list_keys()) == 1                  # device key untouched
        assert "session" in r.output.lower()               # the user is told

    def test_clear_revokes_browser_sessions(self, runner, monkeypatch):
        from localm import auth, sessions
        monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
        sid = self._mint_owner_session()
        assert sessions.lookup(sid) is not None
        r = runner.invoke(main, ["key", "clear", "--yes"])
        assert r.exit_code == 0, r.output
        assert auth.get_api_key() is None                  # clear truly clears
        assert sessions.lookup(sid) is None                # NEGATIVE: owner cookie dead


# --------------------------------------------------------------------------- #
#  Named scoped keys: list / create / rm                                       #
# --------------------------------------------------------------------------- #

class TestNamedKeys:
    def test_list_empty(self, runner):
        r = runner.invoke(main, ["key", "list"])
        assert r.exit_code == 0
        assert "no named keys" in r.output.lower()

    def test_create_scoped_key_shown_once(self, runner):
        r = runner.invoke(main, ["key", "create", "dashboard",
                                 "--scope", "models:read"])
        assert r.exit_code == 0
        keys = auth.list_keys()
        assert len(keys) == 1
        assert keys[0]["scopes"] == ["models:read"]
        secret = _SECRET_RE.findall(r.output)
        assert secret, "the new key must be printed once"

    def test_create_with_rag_roots_round_trips_and_lists(self, runner):
        # --rag-root is repeatable, following the exact CLI shape as --fs-access:
        # a key created with it is CONFINED to exactly those folders for RAG.
        # Short synthetic paths, not tmp_path: the RAG-roots column is a Rich
        # table cell and gets ellipsized past a certain width.
        root_a, root_b = "C:/docs/a", "C:/docs/b"
        r = runner.invoke(main, ["key", "create", "scoped-rag",
                                 "--scope", "rag",
                                 "--rag-root", root_a, "--rag-root", root_b])
        assert r.exit_code == 0, r.output
        assert "rag-roots" in r.output.lower()
        keys = auth.list_keys()
        assert len(keys) == 1
        assert keys[0]["rag_roots"] == [root_a, root_b]
        listed = runner.invoke(main, ["key", "list"])
        assert root_a in listed.output and root_b in listed.output

    def test_create_without_rag_roots_defaults_to_unrestricted(self, runner):
        r = runner.invoke(main, ["key", "create", "plain", "--scope", "rag"])
        assert r.exit_code == 0
        assert auth.list_keys()[0]["rag_roots"] == []
        # "rag-roots" (the note) must NOT appear when none were granted.
        assert "rag-roots" not in r.output.lower()
        listed = runner.invoke(main, ["key", "list"])
        assert "unrestricted" in listed.output.lower()

    def test_create_privileged_scope_refused(self, runner):
        # NEGATIVE: a non-owner mint of keys:admin must fail and store nothing.
        r = runner.invoke(main, ["key", "create", "evil",
                                 "--scope", "keys:admin"])
        assert r.exit_code == 1
        assert auth.list_keys() == []

    def test_create_allow_privileged_mints_privileged_scope(self, runner):
        # --allow-privileged is the explicit opt-in: the same scope refused above
        # succeeds once asked for, and the confirmation message names it.
        r = runner.invoke(main, ["key", "create", "manager",
                                 "--scope", "keys:admin", "--allow-privileged"])
        assert r.exit_code == 0, r.output
        assert auth.list_keys()[0]["scopes"] == ["keys:admin"]
        assert "privileged" in r.output.lower()

    def test_create_without_allow_privileged_flag_default_unchanged(self, runner):
        # A plain, ordinary-scope create is unaffected by the new flag existing.
        r = runner.invoke(main, ["key", "create", "plain",
                                 "--scope", "models:read"])
        assert r.exit_code == 0
        assert "privileged" not in r.output.lower()

    def test_create_with_expires_in_sets_deadline_and_lists(self, runner):
        import time as _time

        before = _time.time()
        r = runner.invoke(main, ["key", "create", "temp",
                                 "--scope", "models:read", "--expires-in", "3600"])
        assert r.exit_code == 0, r.output
        assert "expires" in r.output.lower()
        rec = auth.list_keys()[0]
        assert before + 3600 - 5 <= rec["expires"] <= _time.time() + 3600 + 5
        listed = runner.invoke(main, ["key", "list"])
        assert "never" not in listed.output.lower()

    def test_create_without_expires_in_never_expires(self, runner):
        r = runner.invoke(main, ["key", "create", "perm",
                                 "--scope", "models:read"])
        assert r.exit_code == 0
        assert "expires" not in r.output.lower()   # note only appears when set
        assert auth.list_keys()[0]["expires"] is None
        listed = runner.invoke(main, ["key", "list"])
        assert "never" in listed.output.lower()

    def test_list_flags_an_expired_key(self, runner):
        r = runner.invoke(main, ["key", "create", "stale",
                                 "--scope", "models:read",
                                 "--expires-in", "-1"])   # deadline already past
        assert r.exit_code == 0, r.output
        listed = runner.invoke(main, ["key", "list"])
        assert "expired" in listed.output.lower()

    def test_list_shows_age_expires_used_columns(self, runner):
        # Columns are named Age/Expires/Used, not Created/Last used.
        runner.invoke(main, ["key", "create", "dash", "--scope", "models:read"])
        r = runner.invoke(main, ["key", "list"])
        assert r.exit_code == 0
        assert "Age" in r.output
        assert "Expires" in r.output
        assert "Used" in r.output
        assert "never" in r.output.lower()      # no --expires-in given
        # a brand-new key's age is a real short duration, not the "-" placeholder
        assert re.search(r"\b\d+s\b", r.output), r.output

    def test_create_unknown_scope_refused(self, runner):
        r = runner.invoke(main, ["key", "create", "x", "--scope", "not:a:scope"])
        assert r.exit_code == 1
        assert auth.list_keys() == []

    def test_list_shows_metadata_not_secret(self, runner):
        created = runner.invoke(main, ["key", "create", "dash",
                                       "--scope", "models:read"])
        secrets = _SECRET_RE.findall(created.output)
        assert secrets
        r = runner.invoke(main, ["key", "list"])
        assert r.exit_code == 0
        assert "models:read" in r.output
        for s in secrets:
            assert s not in r.output                # plaintext never re-shown

    def test_rm_revokes_named_key(self, runner):
        runner.invoke(main, ["key", "create", "dash", "--scope", "models:read"])
        key_id = auth.list_keys()[0]["id"]
        r = runner.invoke(main, ["key", "rm", key_id, "--yes"])
        assert r.exit_code == 0
        assert auth.list_keys() == []

    def test_rm_unknown_id_errors(self, runner):
        r = runner.invoke(main, ["key", "rm", "deadbeef", "--yes"])
        assert r.exit_code == 1


# --------------------------------------------------------------------------- #
#  Timestamp formatting helpers used by `key list`                             #
# --------------------------------------------------------------------------- #

class TestTimestampFormatting:
    def test_fmt_ts_none_is_dash(self):
        from localm.cli.keys import _fmt_ts
        assert _fmt_ts(None) == "-"

    def test_fmt_ts_formats_a_real_epoch(self):
        from localm.cli.keys import _fmt_ts
        # A fixed epoch timestamp; only checking it renders the year/date
        # shape, not a specific timezone-dependent clock reading.
        out = _fmt_ts(1767225600)
        assert out != "-"
        assert "2025" in out or "2026" in out

    def test_fmt_expires_none_is_never(self):
        from localm.cli.keys import _fmt_expires
        assert _fmt_expires(None) == "never"

    def test_fmt_expires_future_has_no_expired_marker(self):
        import time

        from localm.cli.keys import _fmt_expires
        out = _fmt_expires(time.time() + 3600)
        assert "expired" not in out.lower()

    def test_fmt_expires_past_is_flagged(self):
        import time

        from localm.cli.keys import _fmt_expires
        out = _fmt_expires(time.time() - 3600)
        assert "expired" in out.lower()
