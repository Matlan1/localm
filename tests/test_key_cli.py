"""H10: the `localm key` CLI group (advertised in auth.py but previously missing).

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
        assert key is not None                      # NEGATIVE pre-fix: no command
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

    def test_create_privileged_scope_refused(self, runner):
        # NEGATIVE: a non-owner mint of keys:admin must fail and store nothing.
        r = runner.invoke(main, ["key", "create", "evil",
                                 "--scope", "keys:admin"])
        assert r.exit_code == 1
        assert auth.list_keys() == []

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
