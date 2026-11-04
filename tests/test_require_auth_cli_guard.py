# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm config require_auth true` refuses a lockout the same way PATCH
/v1/config does: with no key configured it would answer 401 to every request."""

from localm.cli import main


def _saved(key):
    from localm.config import load_config
    return load_config().get(key)


def test_cli_refuses_to_enable_with_no_key(cli_runner, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    result = cli_runner.invoke(main, ["config", "require_auth", "true"])
    assert result.exit_code != 0
    assert "Cannot enable require_auth" in result.output
    assert _saved("require_auth") is False


def test_cli_enables_with_a_key_and_turning_off_is_always_allowed(cli_runner, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "a-long-enough-test-key")
    result = cli_runner.invoke(main, ["config", "require_auth", "true"])
    assert result.exit_code == 0, result.output
    assert _saved("require_auth") is True
    monkeypatch.delenv("LOCALM_API_KEY")
    result = cli_runner.invoke(main, ["config", "require_auth", "false"])
    assert result.exit_code == 0, result.output
    assert _saved("require_auth") is False
