# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm serve` must REFUSE a keyless network bind, matching `localm gui`.

`serve -H 0.0.0.0` with no API key would serve the OpenAI API - plus any enabled
plugin routes, e.g. the coder agent's history - unauthenticated to the whole
LAN. It refuses unless --insecure is passed, and the refusal happens before any
model work.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from localm.cli import main


@pytest.fixture
def open_home(tmp_path, monkeypatch):
    """No API key configured anywhere (open mode) + an isolated home."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


def test_serve_refuses_keyless_network_bind(open_home):
    r = CliRunner().invoke(main, ["serve", "anymodel", "-H", "0.0.0.0"])
    assert r.exit_code == 2, r.output
    assert "Refusing to start" in r.output


def test_serve_insecure_overrides_the_refusal(open_home):
    # --insecure gets PAST the bind gate; it then fails on the missing model
    # (exit 1), which proves the bind gate did not block it.
    r = CliRunner().invoke(main, ["serve", "no-such-model", "-H", "0.0.0.0", "--insecure"])
    assert r.exit_code != 2, r.output
    assert "Model not found" in r.output


def test_serve_loopback_is_not_refused(open_home):
    # Loopback is safe; serve proceeds to the model lookup (exit 1, not refused).
    r = CliRunner().invoke(main, ["serve", "no-such-model", "-H", "127.0.0.1"])
    assert r.exit_code != 2, r.output
    assert "Model not found" in r.output


def test_serve_with_api_key_is_not_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("LOCALM_API_KEY", "a-real-secret")
    r = CliRunner().invoke(main, ["serve", "no-such-model", "-H", "0.0.0.0"])
    assert r.exit_code != 2, r.output
