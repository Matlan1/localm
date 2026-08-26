# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localcoder` attaching to an already-running localm instance must
authenticate with the OWNER KEY when this install has one configured, not the
discovered instance's raw per-instance attach token (``auth.verify()`` has no
notion of instance tokens at all and 401s once any key is configured), and
not the ``--api-key`` option's literal ``"localm"`` placeholder default
either. The same class applies to `localm run`'s attach and to cli/models.py's
unload_cmd/stop_cmd.

An EXPLICIT ``--api-key`` / ``$LOCALM_API_KEY`` must still win (never silently
override an explicit user choice) - covered here too, for both the
discovered-instance attach and the newly-autostarted-server attach (two
separate call sites in _build_backend that share the same _api_key_explicit
computation).
"""

from click.testing import CliRunner

import localm.instances as instances
import localm.plugins.coder.cli as ccli
import localm.plugins.coder.cli._main as cli_main
from localm import auth


def _bypass_plugin_gate(monkeypatch):
    monkeypatch.setattr("localm.plugins.engine.PluginManager.is_active",
                        lambda self, name: True)


def _existing_instance(monkeypatch, token="raw-instance-token"):
    fake_target = {"base_url": "http://127.0.0.1:8642/v1", "port": 8642,
                   "mode": "chat", "token": token}
    monkeypatch.setattr(instances, "attach_target", lambda *a, **k: fake_target)
    monkeypatch.setattr(instances, "resolve_root_dir", lambda *a, **k: ".")
    return fake_target


def _no_existing_instance(monkeypatch):
    monkeypatch.setattr(instances, "attach_target", lambda *a, **k: None)
    monkeypatch.setattr(instances, "resolve_root_dir", lambda *a, **k: ".")


def _capture_http_backend(monkeypatch, module=cli_main):
    """HTTPBackend is a plain top-level import in _main.py (unlike
    make_localm_backend, which has an explicit live-attribute-access indirection
    for exactly this reason) - patch it on the submodule that actually holds
    the reference, not the localm.plugins.coder.cli package re-export."""
    captured = {}

    def fake_backend(base_url, model, api_key="localm", **kw):
        captured.update(base_url=base_url, model=model, api_key=api_key)
        raise SystemExit(0)   # stop before the Agent / REPL, like TestCoderCliThreadsKey

    monkeypatch.setattr(module, "HTTPBackend", fake_backend)
    return captured


class TestAttachToDiscoveredInstance:
    """The `_tgt` branch: a localm is already running for this project dir."""

    def test_owner_key_used_when_configured(self, monkeypatch):
        _bypass_plugin_gate(monkeypatch)
        _existing_instance(monkeypatch)
        auth.set_api_key("real-owner-key-0123456789")
        captured = _capture_http_backend(monkeypatch)

        CliRunner().invoke(ccli.main, ["--model", "m", "hi"])

        assert captured.get("api_key") == "real-owner-key-0123456789"

    def test_instance_token_used_when_open(self, monkeypatch):
        _bypass_plugin_gate(monkeypatch)
        _existing_instance(monkeypatch)
        assert auth.get_api_key() is None
        captured = _capture_http_backend(monkeypatch)

        CliRunner().invoke(ccli.main, ["--model", "m", "hi"])

        assert captured.get("api_key") == "raw-instance-token"

    def test_explicit_api_key_flag_still_wins(self, monkeypatch):
        _bypass_plugin_gate(monkeypatch)
        _existing_instance(monkeypatch)
        auth.set_api_key("real-owner-key-0123456789")
        captured = _capture_http_backend(monkeypatch)

        CliRunner().invoke(
            ccli.main, ["--model", "m", "--api-key", "explicit-flag-key", "hi"])

        assert captured.get("api_key") == "explicit-flag-key"

    def test_explicit_env_key_still_wins(self, monkeypatch):
        _bypass_plugin_gate(monkeypatch)
        _existing_instance(monkeypatch)
        auth.set_api_key("real-owner-key-0123456789")
        monkeypatch.setenv("LOCALM_API_KEY", "explicit-env-key")
        captured = _capture_http_backend(monkeypatch)

        CliRunner().invoke(ccli.main, ["--model", "m", "hi"])

        assert captured.get("api_key") == "explicit-env-key"


class TestAttachToAutoStartedInstance:
    """The other _tgt branch: no instance was running, this process spawns
    `localm gui` itself and then attaches once it comes up - same
    _api_key_explicit computation, a separate HTTPBackend construction site
    that must not silently diverge from the branch above."""

    def _drive_autostart(self, monkeypatch, target):
        _bypass_plugin_gate(monkeypatch)
        _no_existing_instance(monkeypatch)
        monkeypatch.setattr("time.sleep", lambda s: None)

        class _AliveProc:
            def poll(self):
                return None   # never exits on its own during the test

        calls = {"n": 0}

        def fake_popen(cmd, **kw):
            return _AliveProc()

        def fake_attach_target(*a, **k):
            # Returns None on the first probe, the real target from then on.
            calls["n"] += 1
            return target if calls["n"] > 1 else None

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr(instances, "attach_target", fake_attach_target)

    def test_owner_key_used_when_configured(self, monkeypatch):
        target = {"base_url": "http://127.0.0.1:8642/v1", "port": 8642,
                  "mode": "chat", "token": "raw-instance-token"}
        self._drive_autostart(monkeypatch, target)
        auth.set_api_key("real-owner-key-0123456789")
        captured = _capture_http_backend(monkeypatch)

        CliRunner().invoke(ccli.main, ["--model", "m", "hi"])

        assert captured.get("api_key") == "real-owner-key-0123456789"

    def test_instance_token_used_when_open(self, monkeypatch):
        target = {"base_url": "http://127.0.0.1:8642/v1", "port": 8642,
                  "mode": "chat", "token": "raw-instance-token"}
        self._drive_autostart(monkeypatch, target)
        assert auth.get_api_key() is None
        captured = _capture_http_backend(monkeypatch)

        CliRunner().invoke(ccli.main, ["--model", "m", "hi"])

        assert captured.get("api_key") == "raw-instance-token"
