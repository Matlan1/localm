# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI must warn before binding past loopback without auth: `localm gui`
serves the shell-capable coder agent, so a LAN bind without a key is warned
about the same way `localm serve` warns.
"""

from unittest.mock import patch

from localm.plugins.gui.cli import _gui_bind_warning


class TestGuiBindWarning:
    def test_loopback_is_silent(self):
        assert _gui_bind_warning("127.0.0.1") is None
        assert _gui_bind_warning("localhost") is None

    def test_lan_bind_warns_about_coder_agent(self):
        with patch.dict("os.environ", {}, clear=False) as _env:
            import os
            os.environ.pop("LOCALM_API_KEY", None)
            msg = _gui_bind_warning("0.0.0.0")
        assert msg is not None
        assert "0.0.0.0" in msg
        # The warning names the coder-agent reach.
        assert "coder agent" in msg
        assert "shell" in msg
        # Built-in TLS encrypts a network bind, so the warning must NOT push a
        # reverse proxy as the way to get encryption.
        assert "reverse proxy" not in msg

    def test_api_key_silences_warning(self):
        # Setting an API key is the documented precaution; stay quiet then,
        # consistent with `localm serve`.
        with patch.dict("os.environ", {"LOCALM_API_KEY": "a-strong-secret"}):
            assert _gui_bind_warning("0.0.0.0") is None

    def test_an_api_only_server_does_not_mention_the_coder_agent(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("LOCALM_API_KEY", None)
            msg = _gui_bind_warning("0.0.0.0", api_mode=True)
        assert msg is not None
        assert "0.0.0.0" in msg
        assert "coder agent" not in msg


def test_a_console_shared_with_a_shell_is_not_hidden(monkeypatch):
    from localm import winconsole
    monkeypatch.setattr(winconsole, "_console_processes",
                        lambda: [(10, 1, "cmd.exe"), (11, 10, "python.exe")])
    assert winconsole.console_has_a_shell() is True
    monkeypatch.setattr(winconsole, "_console_processes",
                        lambda: [(10, 1, "localm.exe"), (11, 10, "python.exe")])
    assert winconsole.console_has_a_shell() is False


def test_the_launchers_own_cmd_wrapper_does_not_count_as_a_shell(monkeypatch):
    from localm import winconsole
    monkeypatch.setattr(winconsole, "_console_processes",
                        lambda: [(20, 777, "cmd.exe"), (21, 20, "python.exe")])
    assert winconsole.console_has_a_shell(777) is False
    assert winconsole.console_has_a_shell(778) is True


def test_an_unreadable_process_list_keeps_the_console_visible(monkeypatch):
    from localm import winconsole

    def fail():
        raise OSError("snapshot refused")
    monkeypatch.setattr(winconsole, "_console_processes", fail)
    assert winconsole.console_has_a_shell() is True


def test_the_launcher_passes_its_pid_as_the_own_console_value():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "launcher.pyw").read_text(encoding="utf-8")
    assert 'env["LOCALM_OWN_CONSOLE"] = str(os.getpid())' in text
