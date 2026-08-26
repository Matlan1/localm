# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI-settable bind: config 'bind_host' + the TLS trio.

Settings > Server > Bind address lets a browser-only user bind the server past
loopback (the phone/Companion feature) without a terminal. The security
property under test throughout: a CONFIG-driven network bind is fail-closed
WITHOUT being able to kill the server - no strong API key means the server
IGNORES the configured bind and stays on loopback (loudly, with the reason
surfaced via app.state.bind_fallback), it never exits and never serves the
network unauthenticated. Only the CLI's --insecure flag, which has NO config
form, can override the key requirement, so an unauthenticated
network bind always requires a terminal. Explicit -H keeps its historical
fail-hard behavior (exit 2).
"""

import ssl

import pytest
from click.testing import CliRunner

from localm import settings_schema as ss
from localm.bindhost import is_valid_bind_host
from localm.cli import _bind_preflight_error, _resolve_bind_host, _resolve_tls

# TEST-NET-2 (RFC 5737): reserved for documentation and never assigned to a real
# interface, so binding it fails deterministically on any machine.
# Syntactically valid, currently unbindable.
STALE_IP = "198.51.100.23"


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    """Throwaway LOCALM_HOME that load_config/save_config actually read.

    config.py freezes HOME_DIR/CONFIG_FILE/REGISTRY_FILE at import, so the
    autouse LOCALM_HOME env fixture alone does not redirect them."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return cfg


def _set(cfg_mod, **updates):
    c = cfg_mod.load_config()
    c.update(updates)
    cfg_mod.save_config(c)


# ------------------------------------------------------------------ #
#  is_valid_bind_host                                                 #
# ------------------------------------------------------------------ #

class TestIsValidBindHost:
    @pytest.mark.parametrize("good", [
        "127.0.0.1", "0.0.0.0", "192.168.1.20", "10.0.0.7", "localhost",
        # IPv6 literals are accepted.
        "::", "::1", "2001:db8::5", "fe80::1",
    ])
    def test_accepts_bindable_literals(self, good):
        assert is_valid_bind_host(good) is True

    @pytest.mark.parametrize("bad", [
        "", None, 0, "myhouse", "example.com", "0.0.0.0:8642",
        "http://0.0.0.0", "127.0.0.1 ", "0.0.0.0/0",
        # A ZONE ID stays rejected: the index names an interface as numbered on
        # ONE machine, and the scoped form does not survive the
        # getaddrinfo(AI_PASSIVE) the server binds through. Plain IPv6 literals
        # are accepted - see TestAcceptsIPv6 below.
        "fe80::1%eth0", "[::1]", "[::]",
    ])
    def test_rejects_everything_else(self, bad):
        assert is_valid_bind_host(bad) is False


# ------------------------------------------------------------------ #
#  Write-time validation (PATCH /v1/config and `localm config`)       #
# ------------------------------------------------------------------ #

class TestValidateUpdate:
    @pytest.mark.parametrize("val, stored", [
        ("", ""),
        # None takes _validate_one's shared TEXT-widget early return (how every
        # text field clears); the read site treats None and "" identically.
        (None, None),
        ("0.0.0.0", "0.0.0.0"),
        (" 192.168.1.20 ", "192.168.1.20"), ("localhost", "localhost"),
    ])
    def test_bind_host_accepts_and_normalizes(self, val, stored):
        assert ss.validate_update({"bind_host": val})["bind_host"] == stored

    @pytest.mark.parametrize("bad", ["myhouse", "0.0.0.0:8642", "http://x", "a b",
                                     "fe80::1%eth0", "[::1]"])
    def test_bind_host_rejects_unbindable(self, bad):
        with pytest.raises(ValueError, match="bind_host"):
            ss.validate_update({"bind_host": bad})

    @pytest.mark.parametrize("val, stored", [
        ("::", "::"), ("::1", "::1"), (" 2001:db8::5 ", "2001:db8::5"),
        ("fe80::1", "fe80::1"),
    ])
    def test_bind_host_accepts_ipv6_literals(self, val, stored):
        """Write-time acceptance of IPv6.

        A regression that broke the port probe or the listening socket while
        leaving this validator open would put the Settings field back to
        accepting a value the server dies on."""
        assert ss.validate_update({"bind_host": val})["bind_host"] == stored

    def test_tls_paths_must_exist(self, tmp_path):
        real = tmp_path / "c.pem"
        real.write_text("x")
        assert ss.validate_update({"tls_cert": str(real)})["tls_cert"] == str(real)
        assert ss.validate_update({"tls_cert": ""})["tls_cert"] == ""
        with pytest.raises(ValueError, match="file not found"):
            ss.validate_update({"tls_key": str(tmp_path / "missing.pem")})

    def test_tls_enabled_is_a_bool(self):
        assert ss.validate_update({"tls_enabled": "off"})["tls_enabled"] is False
        assert ss.validate_update({"tls_enabled": True})["tls_enabled"] is True

    def test_all_four_are_owner_only_and_restart_scoped(self):
        by_key = {f.key: f for f in ss.CORE_FIELDS}
        for key in ("bind_host", "tls_enabled", "tls_cert", "tls_key"):
            assert by_key[key].admin_only is True, f"{key} widens network reach"
            assert by_key[key].applies == ss.Applies.RESTART

    def test_no_config_form_of_insecure_exists(self):
        """The unauthenticated-network override must stay terminal-only: no
        config key may exist whose name suggests it."""
        from localm.config import DEFAULT_CONFIG
        suspects = [k for k in DEFAULT_CONFIG if "insecure" in k.lower()]
        assert suspects == []


# ------------------------------------------------------------------ #
#  _resolve_bind_host (the read site that makes Settings+Restart work) #
# ------------------------------------------------------------------ #

class TestResolveBindHost:
    def test_explicit_cli_wins_over_config(self, cfg_home):
        _set(cfg_home, bind_host="0.0.0.0")
        assert _resolve_bind_host("127.0.0.1") == ("127.0.0.1", False)

    def test_config_applies_when_no_flag(self, cfg_home):
        _set(cfg_home, bind_host="0.0.0.0")
        assert _resolve_bind_host(None) == ("0.0.0.0", True)

    def test_unset_config_means_loopback(self, cfg_home):
        assert _resolve_bind_host(None) == ("127.0.0.1", False)

    def test_hand_edited_garbage_is_ignored_not_fatal(self, cfg_home):
        # Bypass save-time validation, so a hand-edited config.json reaches the
        # read site: it must not hand uvicorn an unbindable value.
        import json
        cfg_home.CONFIG_FILE.write_text(
            json.dumps({"bind_host": "not-an-address"}), encoding="utf-8")
        assert _resolve_bind_host(None) == ("127.0.0.1", False)


# ------------------------------------------------------------------ #
#  _bind_preflight_error (runtime bindability, not syntax)            #
# ------------------------------------------------------------------ #

class TestBindPreflight:
    def test_stale_interface_ip_is_reported(self):
        # Valid syntax, no such interface on this machine: returns the OS reason
        # rather than raising.
        err = _bind_preflight_error(STALE_IP)
        assert err is not None and isinstance(err, str)

    @pytest.mark.parametrize("ok", ["127.0.0.1", "0.0.0.0"])
    def test_bindable_hosts_pass(self, ok):
        assert _bind_preflight_error(ok) is None


# ------------------------------------------------------------------ #
#  _resolve_tls config reads                                          #
# ------------------------------------------------------------------ #

def _mint_pair(home_cfg, tmp_path):
    """A REAL loadable cert/key pair (localm's own CA machinery), copied to
    distinctive paths so a test can tell 'returned the config pair' apart from
    'returned the built-in pair'."""
    from localm import tls
    from localm.config import home_dir
    cert, key = tls.ensure_cert(home_dir())
    import shutil
    c2 = tmp_path / "custom-cert.pem"
    k2 = tmp_path / "custom-key.pem"
    shutil.copy(cert, c2)
    shutil.copy(key, k2)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(c2), str(k2))   # the fixture itself must be loadable
    return str(c2), str(k2)


class TestResolveTlsConfig:
    def test_tls_enabled_false_acts_as_no_tls(self, cfg_home):
        _set(cfg_home, tls_enabled=False)
        assert _resolve_tls("0.0.0.0", no_tls=False, tls_cert=None,
                            tls_key=None) == (None, None)

    def test_usable_config_pair_is_served(self, cfg_home, tmp_path):
        c2, k2 = _mint_pair(cfg_home, tmp_path)
        _set(cfg_home, tls_cert=c2, tls_key=k2)
        assert _resolve_tls("0.0.0.0", no_tls=False, tls_cert=None,
                            tls_key=None) == (c2, k2)

    def test_broken_config_pair_falls_back_to_builtin_tls(self, cfg_home, tmp_path):
        # A pair that exists but does not LOAD degrades to the built-in cert,
        # never cleartext and never an exception.
        bad_c = tmp_path / "bad.crt"
        bad_k = tmp_path / "bad.key"
        bad_c.write_text("not a cert")
        bad_k.write_text("not a key")
        _set(cfg_home, tls_cert=str(bad_c), tls_key=str(bad_k))
        cert, key = _resolve_tls("0.0.0.0", no_tls=False, tls_cert=None,
                                 tls_key=None)
        assert cert and key
        assert (cert, key) != (str(bad_c), str(bad_k))
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)   # the fallback pair actually loads

    def test_half_config_pair_falls_back_to_builtin_tls(self, cfg_home, tmp_path):
        c2, _k2 = _mint_pair(cfg_home, tmp_path)
        _set(cfg_home, tls_cert=c2)   # key left blank
        cert, key = _resolve_tls("0.0.0.0", no_tls=False, tls_cert=None,
                                 tls_key=None)
        assert cert and key and cert != c2

    def test_cli_pair_never_reads_config(self, cfg_home, tmp_path):
        # An explicit CLI pair is honoured verbatim even with tls_enabled=False
        # in config.
        _set(cfg_home, tls_enabled=False)
        c = tmp_path / "c.pem"
        k = tmp_path / "k.pem"
        c.write_text("cert")
        k.write_text("key")
        assert _resolve_tls("127.0.0.1", no_tls=False, tls_cert=str(c),
                            tls_key=str(k)) == (str(c), str(k))

    def test_loopback_stays_plain_regardless_of_config(self, cfg_home, tmp_path):
        c2, k2 = _mint_pair(cfg_home, tmp_path)
        _set(cfg_home, tls_cert=c2, tls_key=k2)
        assert _resolve_tls("127.0.0.1", no_tls=False, tls_cert=None,
                            tls_key=None) == (None, None)


# ------------------------------------------------------------------ #
#  The startup guard, through the real `localm gui` command           #
# ------------------------------------------------------------------ #

class _Sentinel(RuntimeError):
    """Raised by the pick_port double: everything under test (host resolution,
    the auth guard, TLS) runs BEFORE pick_port, so reaching this proves the
    guard let startup continue - without ever starting a real server."""


@pytest.fixture
def gui_probe(cfg_home, monkeypatch):
    """Drive the real `localm gui` main() up to (not including) the port pick.

    Returns (invoke, calls): calls["pick_port_host"] records the host the
    server would have picked its port for, calls["cert_hosts"] the SAN list a
    built-in-TLS provision was asked for (absent = TLS never provisioned).
    ensure_cert is doubled so no real crypto runs."""
    calls = {}

    def fake_pick_port(requested=None, host="127.0.0.1"):
        calls["pick_port_host"] = host
        raise _Sentinel("stop before any server starts")

    def fake_ensure_cert(home, hostnames=None, ips=None):
        calls["cert_hosts"] = list(hostnames or [])
        return "FAKE.crt", "FAKE.key"

    monkeypatch.setattr(cfg_home, "pick_port", fake_pick_port)
    import localm.tls as tls_mod
    monkeypatch.setattr(tls_mod, "ensure_cert", fake_ensure_cert)

    def invoke(args):
        from localm.plugins.gui.cli import main as gui_cmd
        return CliRunner().invoke(gui_cmd, args)

    return invoke, calls


class TestConfigBindGuard:
    def test_config_bind_without_key_falls_back_to_loopback_alive(
            self, cfg_home, gui_probe):
        """The fail-closed property: config says network, no key exists ->
        the server neither exits nor binds the network; it continues starting
        on loopback with the refusal printed."""
        invoke, calls = gui_probe
        _set(cfg_home, bind_host="0.0.0.0")
        r = invoke(["--no-model", "--no-browser"])
        assert isinstance(r.exception, _Sentinel), (
            f"startup stopped before the port pick: exit={r.exit_code} "
            f"output={r.output!r}")
        assert calls["pick_port_host"] == "127.0.0.1"
        assert "cert_hosts" not in calls, "TLS must not be provisioned for a refused bind"
        assert "Ignoring the configured bind address" in r.output
        assert r.exit_code != 2

    def test_config_bind_with_strong_key_binds_the_network(
            self, cfg_home, gui_probe, monkeypatch):
        invoke, calls = gui_probe
        _set(cfg_home, bind_host="0.0.0.0")
        monkeypatch.setenv("LOCALM_API_KEY", "a-strong-secret-123")
        r = invoke(["--no-model", "--no-browser"])
        assert isinstance(r.exception, _Sentinel), r.output
        assert calls["cert_hosts"][0] == "0.0.0.0"   # built-in TLS for the bind
        assert "Ignoring the configured bind address" not in r.output
        assert "Refusing to start" not in r.output

    def test_explicit_H_without_key_still_exits_2(self, cfg_home, gui_probe):
        """The CLI's historical fail-hard contract is unchanged: an operator
        typing -H past loopback with no key gets exit 2, not a fallback."""
        invoke, calls = gui_probe
        r = invoke(["--no-model", "--no-browser", "-H", "0.0.0.0"])
        assert r.exit_code == 2
        assert "Refusing to start" in r.output
        assert "pick_port_host" not in calls

    def test_explicit_loopback_H_beats_config_network_bind(
            self, cfg_home, gui_probe, monkeypatch):
        invoke, calls = gui_probe
        _set(cfg_home, bind_host="0.0.0.0")
        monkeypatch.setenv("LOCALM_API_KEY", "a-strong-secret-123")
        r = invoke(["--no-model", "--no-browser", "-H", "127.0.0.1"])
        assert isinstance(r.exception, _Sentinel), r.output
        assert calls["pick_port_host"] == "127.0.0.1"
        assert "cert_hosts" not in calls          # loopback: no TLS
        assert "Ignoring the configured bind address" not in r.output

    def test_stale_config_ip_falls_back_to_loopback_alive(
            self, cfg_home, gui_probe, monkeypatch):
        """A config address that is well-formed but no longer assigned (DHCP
        moved the machine) must not kill the server at the socket bind - the
        stale address is refused by a REAL preflight bind probe (nothing
        mocked at that layer; STALE_IP is genuinely unassigned) and startup
        continues on loopback. Without the preflight this dies with exit 3
        inside uvicorn, past every syntax check."""
        invoke, calls = gui_probe
        _set(cfg_home, bind_host=STALE_IP)
        monkeypatch.setenv("LOCALM_API_KEY", "a-strong-secret-123")
        r = invoke(["--no-model", "--no-browser"])
        assert isinstance(r.exception, _Sentinel), (
            f"startup stopped before the port pick: exit={r.exit_code} "
            f"output={r.output!r}")
        assert calls["pick_port_host"] == "127.0.0.1"
        assert "cert_hosts" not in calls, "TLS must not be provisioned for a refused bind"
        assert "cannot be bound on this machine" in r.output
        assert r.exit_code != 2

    def test_builtin_tls_failure_on_config_bind_falls_back_to_loopback(
            self, cfg_home, gui_probe, monkeypatch):
        """A broken crypto stack must not kill a config-driven bind (no
        terminal to see the exit) and must not serve the network in cleartext:
        the only remaining option is loopback, taken loudly."""
        invoke, calls = gui_probe
        _set(cfg_home, bind_host="0.0.0.0")
        monkeypatch.setenv("LOCALM_API_KEY", "a-strong-secret-123")
        import localm.tls as tls_mod

        def broken_ensure_cert(home, hostnames=None, ips=None):
            raise RuntimeError("crypto stack unavailable")

        monkeypatch.setattr(tls_mod, "ensure_cert", broken_ensure_cert)
        r = invoke(["--no-model", "--no-browser"])
        assert isinstance(r.exception, _Sentinel), r.output
        assert calls["pick_port_host"] == "127.0.0.1"
        assert "Could not set up built-in TLS" in r.output
        assert r.exit_code != 2
