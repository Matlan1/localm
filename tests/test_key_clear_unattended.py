"""Removing the last API key needs the owner at an interactive terminal.

`localm key clear` and `localm key rm` of the last key return the server to open
mode; a script piping 'y' or passing --yes (an auto-approved coder shell, an
unattended job) must not be able to do that.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from localm import auth
from localm.cli import main
from localm.cli import keys as keys_cli

STRONG_KEY = "w1-strong-test-key-0123456789abcdef"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)


def test_clear_with_yes_from_piped_stdin_is_refused():
    auth.set_api_key(STRONG_KEY)
    r = CliRunner().invoke(main, ["key", "clear", "--yes"], input="y\n")
    assert r.exit_code == 1, r.output
    assert "Refusing to clear the API key" in r.output
    assert auth.key_file().read_text(encoding="utf-8").strip() == STRONG_KEY


def test_clear_piped_confirmation_is_refused():
    auth.set_api_key(STRONG_KEY)
    r = CliRunner().invoke(main, ["key", "clear"], input="y\n")
    assert r.exit_code == 1, r.output
    assert auth.get_api_key() == STRONG_KEY


def test_clear_at_an_interactive_terminal_still_works(monkeypatch):
    monkeypatch.setattr(keys_cli, "_stdin_is_interactive", lambda: True)
    auth.set_api_key(STRONG_KEY)
    r = CliRunner().invoke(main, ["key", "clear", "--yes"], input="y\n")
    assert r.exit_code == 0, r.output
    assert "Clear the API key?" in r.output
    assert "API key cleared" in r.output
    assert not auth.key_file().exists()


@pytest.mark.parametrize("argv", [["key", "clear", "--yes"], ["key", "rm", "{id}", "--yes"]])
def test_yes_does_not_skip_the_question_for_the_last_key(monkeypatch, argv):
    monkeypatch.setattr(keys_cli, "_stdin_is_interactive", lambda: True)
    rec = auth.create_key("only", ["models:read"])
    if argv[1] == "clear":
        auth.set_api_key(STRONG_KEY)
    argv = [a.format(id=rec["id"]) for a in argv]
    r = CliRunner().invoke(main, argv, input="n\n")
    assert "?" in r.output, r.output
    assert "Cancelled" in r.output, r.output
    assert [k["id"] for k in auth.list_keys()] == [rec["id"]]
    if argv[1] == "clear":
        assert auth.get_api_key() == STRONG_KEY


def test_yes_still_skips_the_question_when_a_key_remains():
    keep = auth.create_key("keep", ["models:read"])
    gone = auth.create_key("gone", ["models:read"])
    r = CliRunner().invoke(main, ["key", "rm", gone["id"], "--yes"])
    assert r.exit_code == 0, r.output
    assert "?" not in r.output
    assert [k["id"] for k in auth.list_keys()] == [keep["id"]]


def test_clear_is_not_refused_while_an_env_key_survives(monkeypatch):
    auth.set_api_key(STRONG_KEY)
    monkeypatch.setenv("LOCALM_API_KEY", "w1-env-key-0123456789abcdefghij")
    r = CliRunner().invoke(main, ["key", "clear", "--yes"])
    assert r.exit_code == 0, r.output
    assert not auth.key_file().exists()


def test_rm_of_the_last_key_from_a_script_is_refused():
    rec = auth.create_key("only", ["models:read"])
    r = CliRunner().invoke(main, ["key", "rm", rec["id"], "--yes"])
    assert r.exit_code == 1, r.output
    assert "Refusing to revoke" in r.output
    assert [k["id"] for k in auth.list_keys()] == [rec["id"]]


def test_rm_of_a_key_that_is_not_the_last_still_works():
    keep = auth.create_key("keep", ["models:read"])
    gone = auth.create_key("gone", ["models:read"])
    r = CliRunner().invoke(main, ["key", "rm", gone["id"], "--yes"])
    assert r.exit_code == 0, r.output
    assert [k["id"] for k in auth.list_keys()] == [keep["id"]]


def test_rm_while_an_owner_key_exists_still_works():
    auth.set_api_key(STRONG_KEY)
    rec = auth.create_key("only", ["models:read"])
    r = CliRunner().invoke(main, ["key", "rm", rec["id"], "--yes"])
    assert r.exit_code == 0, r.output
    assert auth.list_keys() == []


@pytest.mark.parametrize("stdin", ["pipe", "null-device"])
def test_a_real_process_without_a_terminal_cannot_clear_the_key(stdin):
    auth.set_api_key(STRONG_KEY)
    env = dict(os.environ)
    env.pop("LOCALM_API_KEY", None)
    env["PYTHONPATH"] = str(REPO)
    kw = ({"input": "y\n"} if stdin == "pipe" else {"stdin": subprocess.DEVNULL})
    p = subprocess.run([sys.executable, "-m", "localm", "key", "clear", "--yes"],
                       capture_output=True, text=True, env=env,
                       cwd=str(REPO), timeout=120, **kw)
    assert p.returncode == 1, (p.stdout, p.stderr)
    assert "Refusing to clear the API key" in p.stdout
    assert auth.key_file().read_text(encoding="utf-8").strip() == STRONG_KEY
