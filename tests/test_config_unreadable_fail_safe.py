# SPDX-License-Identifier: AGPL-3.0-or-later
"""A config.json that exists and cannot be read must not loosen a security
setting.

load_config() never raises: an unreadable config.json (and its .bak) comes back
as the defaults with read_ok False. These tests write real undecodable bytes
into the files, which is the only way production produces that state, and check
that the network policy and require_auth fail to the restrictive side.
"""

import logging

import pytest

import localm.config as cfg
from localm import auth
from localm.netpolicy import NetworkPolicyError, check_url, network_mode

PUBLIC_IP = "93.184.216.34"
OTHER_PUBLIC_IP = "1.1.1.1"
DENIED_PUBLIC_IP = "8.8.8.8"
UNDECODABLE = b"\xff\xfe\x00 not json \xc3\x28"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


def _store(home, **values):
    import json
    (home / "config.json").write_text(json.dumps(values), encoding="utf-8")


def _corrupt(home, *, bak=True):
    (home / "config.json").write_bytes(UNDECODABLE)
    if bak:
        (home / "config.json.bak").write_bytes(UNDECODABLE)


def test_the_corruption_fixture_really_makes_the_config_unreadable(home):
    _store(home, net_mode="off")
    _corrupt(home)
    conf, read_ok = cfg.load_config_checked()
    assert read_ok is False
    assert conf["net_mode"] == cfg.DEFAULT_CONFIG["net_mode"] != "off"


class TestNetworkMode:
    def test_an_explicit_off_survives_an_unreadable_config(self, home, caplog):
        _store(home, net_mode="off")
        assert network_mode() == "off"
        _corrupt(home)
        with caplog.at_level(logging.WARNING, logger="localm"):
            assert network_mode() == "off"
        assert any("unreadable" in r.message and "off" in r.message
                   for r in caplog.records), "resolved to off without saying why"

    def test_a_missing_config_is_the_ordinary_default(self, home):
        assert not (home / "config.json").exists()
        assert network_mode() == "ask"

    def test_a_good_backup_keeps_the_stored_mode(self, home):
        _store(home, net_mode="allow")
        (home / "config.json.bak").write_bytes((home / "config.json").read_bytes())
        (home / "config.json").write_bytes(UNDECODABLE)
        assert network_mode() == "allow"

    def test_the_env_override_still_wins(self, home, monkeypatch):
        _corrupt(home)
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")
        assert network_mode() == "allow"


class TestCheckUrl:
    def test_the_allow_list_is_not_lifted_by_an_unreadable_config(self, home):
        _store(home, net_mode="allow", net_allow=[PUBLIC_IP])
        check_url(f"http://{PUBLIC_IP}/")
        with pytest.raises(NetworkPolicyError, match="allow list"):
            check_url(f"http://{OTHER_PUBLIC_IP}/")
        _corrupt(home)
        with pytest.raises(NetworkPolicyError, match="could not be read"):
            check_url(f"http://{OTHER_PUBLIC_IP}/")

    def test_the_deny_list_is_not_dropped_by_an_unreadable_config(self, home):
        _store(home, net_mode="allow", net_deny=[DENIED_PUBLIC_IP])
        with pytest.raises(NetworkPolicyError, match="deny list"):
            check_url(f"http://{DENIED_PUBLIC_IP}/")
        _corrupt(home)
        with pytest.raises(NetworkPolicyError, match="could not be read"):
            check_url(f"http://{DENIED_PUBLIC_IP}/")

    def test_net_mode_off_is_not_downgraded_to_ask(self, home):
        _store(home, net_mode="off")
        with pytest.raises(NetworkPolicyError, match="net_mode=off"):
            check_url(f"http://{PUBLIC_IP}/")
        _corrupt(home)
        with pytest.raises(NetworkPolicyError, match="could not be read"):
            check_url(f"http://{PUBLIC_IP}/")

    def test_an_env_override_does_not_bypass_the_refusal(self, home, monkeypatch):
        _corrupt(home)
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")
        with pytest.raises(NetworkPolicyError, match="could not be read"):
            check_url(f"http://{PUBLIC_IP}/")

    def test_a_missing_config_is_not_a_refusal(self, home):
        assert not (home / "config.json").exists()
        check_url(f"http://{PUBLIC_IP}/")

    def test_a_refusal_is_logged(self, home, caplog):
        _corrupt(home)
        with caplog.at_level(logging.WARNING, logger="localm"):
            with pytest.raises(NetworkPolicyError):
                check_url(f"http://{PUBLIC_IP}/")
        assert any("unreadable" in r.message for r in caplog.records)


class TestRequireAuth:
    def test_an_unreadable_config_requires_auth(self, home, caplog):
        _store(home, require_auth=False)
        assert auth.require_auth_enabled() is False
        _corrupt(home)
        with caplog.at_level(logging.WARNING, logger="localm"):
            assert auth.require_auth_enabled() is True
        assert any("require_auth" in r.message and "unreadable" in r.message
                   for r in caplog.records)

    def test_a_missing_config_does_not_require_auth(self, home):
        assert not (home / "config.json").exists()
        assert auth.require_auth_enabled() is False

    def test_a_good_backup_keeps_the_stored_flag(self, home):
        _store(home, require_auth=False)
        (home / "config.json.bak").write_bytes((home / "config.json").read_bytes())
        (home / "config.json").write_bytes(UNDECODABLE)
        assert auth.require_auth_enabled() is False

