# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-O: a weak owner key must not satisfy the network-bind auth gate."""

import pytest

from localm import auth
from localm.cli import _exposed_bind_warning


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)


class TestWeakKeyBindGate:
    def test_loopback_with_weak_key_is_allowed(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "1")
        assert _exposed_bind_warning("127.0.0.1") is None
        assert _exposed_bind_warning("localhost") is None
        assert _exposed_bind_warning("::1") is None

    def test_network_with_no_key_warns_unauthenticated(self):
        w = _exposed_bind_warning("0.0.0.0")
        assert w is not None and "WITHOUT authentication" in w

    @pytest.mark.parametrize("weak", ["1", "1234567"])  # 1 and 7 chars (< 8)
    def test_network_with_weak_key_is_refused(self, monkeypatch, weak):
        monkeypatch.setenv("LOCALM_API_KEY", weak)
        w = _exposed_bind_warning("0.0.0.0")
        assert w is not None and "WEAK" in w

    def test_network_with_strong_key_is_allowed(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "a-strong-secret-123")
        assert _exposed_bind_warning("0.0.0.0") is None

    def test_weak_key_via_auth_file_also_refused(self, monkeypatch):
        # A hand-edited auth.key (bypassing set_api_key's floor) must also be
        # caught at the bind gate, not just an env var.
        auth.key_file().parent.mkdir(parents=True, exist_ok=True)
        auth.key_file().write_text("1\n", encoding="utf-8")
        w = _exposed_bind_warning("192.168.0.10")
        assert w is not None and "WEAK" in w


class TestSetKeyFloorUnchanged:
    def test_set_api_key_still_rejects_short(self):
        with pytest.raises(ValueError, match="at least 8"):
            auth.set_api_key("1")

    def test_set_api_key_accepts_strong(self):
        auth.set_api_key("a-strong-secret-123")
        assert auth.get_api_key() == "a-strong-secret-123"
