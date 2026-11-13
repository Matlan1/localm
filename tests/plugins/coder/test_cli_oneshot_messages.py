# SPDX-License-Identifier: AGPL-3.0-or-later
"""What `localm coder` says on its edge paths: a server that does not answer or
refuses, a provider key, a resumed id that matches nothing, an unknown episode
id, --estimate, --ci colours, a disabled plugin, and a reply another model gave."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests
from click.testing import CliRunner

from localm.plugins.coder.backends.http import (
    CoderAuthError,
    CoderServerError,
    HTTPBackend,
    _raise_for_status,
)
from localm.plugins.coder.cli import _main


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


@pytest.fixture
def proj(home, tmp_path, monkeypatch):
    from localm.plugins.engine import PluginManager
    PluginManager(None).set_installed_state("coder", True)
    p = tmp_path / "proj"
    p.mkdir()
    backend = MagicMock()
    backend.model_id = "stub-model"
    backend._base_url = "http://127.0.0.1:9/v1"
    agent = MagicMock()
    agent._patch_chunks = []
    agent.has_patch.return_value = False
    monkeypatch.setattr(_main, "_build_backend", lambda *a, **k: backend)
    monkeypatch.setattr(_main, "build_agent", lambda *a, **k: agent)
    monkeypatch.setattr(_main, "run_single_task", MagicMock())
    monkeypatch.setattr(_main, "finish_agent", lambda a: None)
    monkeypatch.setattr(_main, "_repl", MagicMock())
    p.agent = agent
    return p


def _invoke(proj, *args):
    return CliRunner().invoke(_main.main, ["-m", "stub-model", "--cwd", str(proj), *args])


class TestAOneShotThatCannotReachOrIsRefused:
    def test_no_server_answering_is_one_line_not_a_crash(self, proj, monkeypatch):
        monkeypatch.setattr(_main, "run_single_task", MagicMock(
            side_effect=requests.ConnectionError("Max retries exceeded")))
        r = _invoke(proj, "--yes", "do x")
        assert r.exit_code == 1
        assert "No model server answers at http://127.0.0.1:9/v1" in r.output
        assert not isinstance(r.exception, requests.ConnectionError)

    def test_a_server_refusal_is_one_line_not_a_crash(self, proj, monkeypatch):
        monkeypatch.setattr(_main, "run_single_task", MagicMock(
            side_effect=CoderServerError("HTTP 302 redirect from x refused")))
        r = _invoke(proj, "--yes", "--ci", "do x")
        assert r.exit_code == 2
        assert "HTTP 302 redirect from x refused" in r.output


class TestMessages:
    def test_estimate_has_no_unattended_warning(self, proj, monkeypatch):
        monkeypatch.setattr(_main, "_run_estimate", MagicMock())
        r = _invoke(proj, "--estimate", "do x")
        assert r.exit_code == 0, r.output
        assert "Unattended one-shot" not in r.output

    def test_a_resumed_id_that_matches_nothing_explains_the_task_order(self, proj):
        proj.agent.load_checkpoint.return_value = None
        r = _invoke(proj, "-r", "fix")
        assert "No saved session with id 'fix'" in r.output
        assert 'localm coder "fix" --resume' in r.output

    def test_ci_turns_off_colour_on_the_existing_consoles(self, proj, monkeypatch):
        from localm.plugins.coder import display
        monkeypatch.setattr(display.console, "no_color", False)
        monkeypatch.setattr(display.err_console, "no_color", False)
        r = _invoke(proj, "--ci", "do x")
        assert r.exit_code == 0, r.output
        assert display.console.no_color is True and display.err_console.no_color is True


class TestEpisodeIdsThatDoNotExist:
    @pytest.mark.parametrize("flag", ["--forget-episode", "--restore-episode"])
    def test_exit_code_is_one(self, home, tmp_path, flag):
        from localm.plugins.engine import PluginManager
        PluginManager(None).set_installed_state("coder", True)
        r = CliRunner().invoke(_main.main, ["--cwd", str(tmp_path), flag, "nope"])
        assert r.exit_code == 1, r.output
        assert "nope" in r.output


def test_an_installed_but_disabled_plugin_is_told_to_enable(home, monkeypatch, tmp_path):
    from localm.plugins.engine import PluginManager
    monkeypatch.setattr(PluginManager, "is_active", lambda self, name: False)
    monkeypatch.setattr(PluginManager, "is_installed", lambda self, name: True)
    r = CliRunner().invoke(_main.main, ["--cwd", str(tmp_path), "do x"])
    assert r.exit_code == 1
    assert "localm plugin enable coder" in r.output
    assert "plugin install" not in r.output


class _Resp:
    def __init__(self, code, url):
        self.status_code = code
        self.url = url


class TestKeyAdvice:
    def test_openai_names_its_key_variable(self):
        be = HTTPBackend("https://api.openai.com/v1", "gpt-4o", trusted_peer=True)
        with pytest.raises(CoderAuthError) as exc:
            _raise_for_status(_Resp(401, "https://api.openai.com/v1/chat"),
                              be._auth_advice())
        assert "OPENAI_API_KEY" in str(exc.value) and "localm key" not in str(exc.value)

    def test_anthropic_names_its_key_variable(self):
        be = HTTPBackend("https://api.anthropic.com/v1", "claude", anthropic=True,
                         trusted_peer=True)
        assert "ANTHROPIC_API_KEY" in be._auth_advice()

    def test_localm_keeps_its_key_advice(self):
        be = HTTPBackend("http://127.0.0.1:8642/v1", "m", localm_server=True,
                         trusted_peer=True)
        assert "localm key show --reveal" in be._auth_advice()


class TestAnotherModelAnswered:
    def _headers(self, **payload):
        return {"X-Localm-Model-Routing": json.dumps(payload)}

    def test_a_routed_reply_is_said_once_per_model(self):
        be = HTTPBackend("http://127.0.0.1:8642/v1", "small", localm_server=True,
                         trusted_peer=True)
        seen = []
        be.on_routing_note = seen.append
        routed = self._headers(resolved="big-vision", requested="small", routed=True,
                               pinned=False)
        be._note_routing(routed)
        be._note_routing(routed)
        assert len(seen) == 1 and "big-vision" in seen[0] and "small" in seen[0]
        assert be.routing_note is None

    def test_a_reply_from_the_named_model_says_nothing(self):
        be = HTTPBackend("http://127.0.0.1:8642/v1", "small", localm_server=True,
                         trusted_peer=True)
        seen = []
        be.on_routing_note = seen.append
        be._note_routing(self._headers(resolved="small", requested="small",
                                       routed=False, pinned=True))
        assert seen == []
