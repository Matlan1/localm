"""C1 / H13: the coder's local backend must present the real API key
(LOCALM_API_KEY or --api-key), not the hardcoded "localm" that 401s against a
require_auth server."""

from click.testing import CliRunner

import localm.plugins.coder.cli as ccli
from localm.plugins.coder.backends.http import make_localm_backend


class TestMakeLocalmBackendKey:
    def test_defaults_to_localm_when_no_key(self, monkeypatch):
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        assert make_localm_backend("m", port=8642)._api_key == "localm"

    def test_reads_localm_api_key_env(self, monkeypatch):
        # FAILS pre-fix: make_localm_backend hardcoded "localm".
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        assert make_localm_backend("m", port=8642)._api_key == "REALKEY123"

    def test_explicit_api_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "envkey")
        assert make_localm_backend("m", port=8642, api_key="flagkey")._api_key == "flagkey"

    def test_empty_explicit_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "envkey")
        assert make_localm_backend("m", port=8642, api_key="")._api_key == "envkey"


class TestCoderCliThreadsKey:
    """The local (auto-started / --no-server) path must pass the resolved key
    into make_localm_backend, honouring $LOCALM_API_KEY and --api-key."""

    def _capture(self, monkeypatch):
        captured = {}
        # the coder CLI refuses to run unless the plugin is active; bypass that
        # gate so this unit-level CLI test can reach the backend construction.
        monkeypatch.setattr("localm.plugins.engine.PluginManager.is_active",
                            lambda self, name: True)

        def fake_backend(model, port=8642, *, api_key="", **kw):
            captured["model"] = model
            captured["api_key"] = api_key
            raise SystemExit(0)   # stop before the Agent / REPL

        monkeypatch.setattr(ccli, "make_localm_backend", fake_backend)
        return captured

    def test_env_key_threaded(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY")
        captured = self._capture(monkeypatch)
        CliRunner().invoke(ccli.main, ["--model", "m", "--no-server", "hi"])
        assert captured.get("api_key") == "REALKEY"

    def test_flag_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "envkey")
        captured = self._capture(monkeypatch)
        CliRunner().invoke(ccli.main, ["--model", "m", "--no-server",
                                       "--api-key", "flagkey", "hi"])
        assert captured.get("api_key") == "flagkey"
