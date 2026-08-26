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
