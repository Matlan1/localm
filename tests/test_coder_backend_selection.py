# SPDX-License-Identifier: AGPL-3.0-or-later
"""A GUI coder session can choose WHICH model server answers it, the way the
terminal always could (--online / --anthropic / --url).

Choosing a non-local model is a TRUST BOUNDARY change, not a wiring change: the
user's prompts and whatever file contents the agent reads leave the machine. So
the properties pinned here are the ones that fail silently or dangerously, not
the happy path:

  1. Privacy mode REFUSES an off-machine model, and no session is created.
     localm already answers this question this way for memory and for the coder
     reviewer; a third answer would be drift.
  2. A REFUSAL IS A REFUSAL. The local model is never quietly substituted for
     the one that was chosen - that would be indistinguishable, from the
     client's side, from having honoured the choice.
  3. A loopback URL is NOT off-machine, so privacy mode must still allow it.
     Collapsing "custom URL" into "cloud" gets Ollama and LM Studio wrong.
  4. A GUI-supplied base URL passes netpolicy, including the fixed provider
     bases, so net_mode=off and a deny list still mean what they say.
  5. A scoped (non-owner) key cannot point the coder anywhere.
  6. The credential never comes back out.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from localm import remotegate
from localm.plugins.builtin.coder.plug import (
    CreateSessionRequest,
    _resolve_backend,
    _url_leaves_machine,
)


# --------------------------------------------------------------------------- #
#  The leaf gate                                                               #
# --------------------------------------------------------------------------- #

def test_privacy_refuses_off_machine_every_other_mode_allows_it():
    """Mirrors memory/gating.py's writes_allowed: privacy is off, log and full
    are on. Asserted against the RESOLVED mode, because a coder session's mode
    is not always the ambient one (an explicit mode on the request and a
    per-project .localcoder/config.toml both override it)."""
    assert remotegate.remote_allowed_for_mode("privacy") is False
    assert remotegate.remote_allowed_for_mode("log") is True
    assert remotegate.remote_allowed_for_mode("full") is True


def test_refusal_names_the_setting_that_enables_it():
    """A refusal that does not say what to change is a dead end. One wording is
    used across every call site, so this holds it to the same bar."""
    msg = remotegate.refusal_message("Off-machine models")
    # The WHOLE sentence, not fragments of it. A fragment assertion passes
    # happily on "Off-machine models is off in privacy mode"; the message is
    # user-facing prose, so the test has to read it as prose.
    assert msg == (
        "Off-machine models are off in privacy mode (nothing leaves this "
        "machine). Set mode/coder_mode to 'log' or 'full' to enable them.")
    # And the two properties that must survive any future rewording: it says
    # WHICH setting, and WHAT to set it to.
    assert "coder_mode" in msg
    assert "log" in msg and "full" in msg


# --------------------------------------------------------------------------- #
#  Loopback classification                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:11434/v1",
    "http://localhost:8080/v1",
    # The whole 127.0.0.0/8 block, which reviewer.py's literal set misses.
    "http://127.0.0.2:1234/v1",
    "http://[::1]:8080/v1",
])
def test_loopback_urls_stay_on_this_machine(url):
    assert _url_leaves_machine(url) is False


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1",
    "http://192.168.1.50:11434/v1",
    "http://169.254.169.254/latest/meta-data/",
    # A wildcard BIND address is not a destination, and a missing host is
    # malformed. Answering "on this machine" for either would hand the quiet
    # path to a string nobody validated.
    "http://0.0.0.0:8080/v1",
    "http:///v1",
    "not a url at all",
])
def test_anything_else_is_treated_as_leaving_the_machine(url):
    assert _url_leaves_machine(url) is True


# --------------------------------------------------------------------------- #
#  _resolve_backend                                                            #
# --------------------------------------------------------------------------- #

def _req(**kw):
    kw.setdefault("cwd", ".")
    return CreateSessionRequest(**kw)


def _resolve(req, *, restricted=False, session_mode="privacy"):
    return _resolve_backend(req, self_url="http://127.0.0.1:9/v1",
                            model_name="local-model", restricted=restricted,
                            session_mode=session_mode)


def _no_netpolicy(monkeypatch):
    """Neutralise the SSRF guard for tests that are not about it.

    Patched on ``localm.netpolicy`` itself, which is where the resolver's
    function-local import resolves it from - patching a name imported into some
    other module would leave the real one running and quietly make these tests
    dial DNS for api.openai.com.
    """
    import localm.netpolicy as np
    monkeypatch.setattr(np, "check_url", lambda url: None)


def test_default_is_this_localm_and_stays_on_the_machine():
    backend, info, notes = _resolve(_req())
    assert info == {"backend": "local", "leaves_machine": False,
                    "target": "this localm", "model": "local-model"}
    # The self-connection keeps grammar-constrained tool calls, which is the
    # capability every other option loses.
    assert backend.supports_grammar is True
    assert notes == []


def test_privacy_refuses_a_cloud_backend_and_says_what_to_change(monkeypatch):
    _no_netpolicy(monkeypatch)
    with pytest.raises(HTTPException) as e:
        _resolve(_req(backend="openai", backend_api_key="sk-test"),
                 session_mode="privacy")
    assert e.value.status_code == 403
    assert "privacy mode" in e.value.detail
    assert "coder_mode" in e.value.detail


def test_privacy_still_allows_a_loopback_url():
    """Ollama / LM Studio on this machine are LOCAL, so privacy mode must allow
    them: this is the single most likely use of the field.

    RUNS THE REAL netpolicy - no monkeypatch. check_url's public-address arm
    blocks loopback by default, and a fixture that removes the guard cannot fail
    on the guard being miscalibrated, which is the whole defect. If this test
    ever needs netpolicy stubbed out to pass, the calibration has regressed."""
    backend, info, notes = _resolve(
        _req(backend="url", backend_url="http://127.0.0.1:11434/v1",
             backend_model="qwen"),
        session_mode="privacy")
    assert info["leaves_machine"] is False
    assert info["target"] == "http://127.0.0.1:11434/v1"
    assert info["model"] == "qwen"
    # No "leaves this machine" note, because it does not.
    assert not any("leave this machine" in n for n in notes)
    # It is still not localm's own server, so grammar is off and SAID so.
    assert backend.supports_grammar is False
    assert any("Grammar-constrained" in n for n in notes)


def test_a_scoped_key_cannot_point_the_coder_anywhere():
    """An exfil channel for the project's source and a billing channel for
    someone else's account. Matches coder_reviewer's admin_only=True."""
    with pytest.raises(HTTPException) as e:
        _resolve(_req(backend="url", backend_url="http://127.0.0.1:11434/v1"),
                 restricted=True, session_mode="log")
    assert e.value.status_code == 403
    assert "owner key" in e.value.detail


def test_missing_provider_key_is_refused_now_not_at_first_message(monkeypatch):
    """Building a session that 401s on its first message makes a missing key
    look like the provider being down."""
    _no_netpolicy(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(HTTPException) as e:
        _resolve(_req(backend="openai"), session_mode="log")
    assert e.value.status_code == 400
    assert "OPENAI_API_KEY" in e.value.detail


def test_a_cloud_backend_reports_what_it_costs_the_user(monkeypatch):
    _no_netpolicy(monkeypatch)
    backend, info, notes = _resolve(
        _req(backend="anthropic", backend_api_key="sk-ant-test",
             backend_model="claude-opus-4-5"),
        session_mode="log")
    assert info["backend"] == "anthropic"
    assert info["leaves_machine"] is True
    assert info["model"] == "claude-opus-4-5"
    assert backend.anthropic is True
    assert backend.supports_grammar is False
    assert any("leave this machine" in n for n in notes)
    # The descriptor is what session.info() publishes to a browser, so it must
    # never carry the credential.
    assert "sk-ant-test" not in repr(info)


@pytest.mark.parametrize("req_kw,fragment", [
    ({"backend": "url"}, "backend_url"),
    ({"backend": "url", "backend_url": "ftp://x/v1"}, "http://"),
    ({"backend": "nonsense"}, "Unknown backend"),
])
def test_malformed_selections_are_refused(monkeypatch, req_kw, fragment):
    _no_netpolicy(monkeypatch)
    with pytest.raises(HTTPException) as e:
        _resolve(_req(**req_kw), session_mode="log")
    assert e.value.status_code == 400
    assert fragment in e.value.detail


def test_netpolicy_gates_even_the_fixed_provider_bases(monkeypatch):
    """net_mode=off means the user disabled network access. That has to hold for
    a provider WE picked just as much as for a URL they typed, or the setting is
    a suggestion. Asserted by spying on the real choke point rather than by
    trusting that the call is there."""
    seen = []

    import localm.netpolicy as np

    def _spy(url):
        seen.append(url)
        raise np.NetworkPolicyError("Network access is disabled (net_mode=off)")

    monkeypatch.setattr(np, "check_url", _spy)
    with pytest.raises(HTTPException) as e:
        _resolve(_req(backend="openai", backend_api_key="sk-test"),
                 session_mode="log")
    assert seen == ["https://api.openai.com/v1"]
    assert e.value.status_code == 403
    assert "net_mode=off" in e.value.detail


def test_a_gui_supplied_url_reaches_the_ssrf_guard(monkeypatch):
    """This backend posts with requests.post directly, so a base URL arriving
    from a web form must still be checked against link-local metadata addresses,
    internal hosts and private ranges."""
    seen = []
    import localm.netpolicy as np
    monkeypatch.setattr(np, "check_url", lambda url: seen.append(url))
    _resolve(_req(backend="url", backend_url="http://169.254.169.254/v1"),
             session_mode="log")
    assert seen == ["http://169.254.169.254/v1"]


# --------------------------------------------------------------------------- #
#  The route - the layer where the defect can actually come back               #
# --------------------------------------------------------------------------- #
# The unit tests above cannot see the ARRANGEMENT: whether create_session calls
# the resolver at all, whether it passes the session's own resolved mode, and
# whether the descriptor reaches the client, so at least one test stays at that
# site.

def _coder_app(tmp_path, monkeypatch, *, api_key="ownersecret"):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", api_key)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    proj = tmp_path / "proj"
    proj.mkdir()
    app.state.root_dir = str(proj)
    return app, proj, {"Authorization": "Bearer %s" % api_key}


def test_route_refuses_a_cloud_session_in_privacy_and_creates_nothing(
        tmp_path, monkeypatch):
    """The whole point of the gate, at the layer it has to hold: a refused
    session must NOT exist afterwards, and must not have been quietly downgraded
    to the local model. Privacy is the DEFAULT, so this request sets no mode."""
    app, proj, owner = _coder_app(tmp_path, monkeypatch)
    _no_netpolicy(monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "backend": "openai",
                              "backend_api_key": "sk-test"})
        assert r.status_code == 403, r.text
        assert "privacy mode" in r.json()["detail"]
        # Not created, not downgraded. A session in the list here would mean the
        # user got a LOCAL session while believing they had asked for OpenAI.
        listed = client.get("/api/coder/sessions", headers=owner).json()
        assert listed["sessions"] == []


def test_route_builds_the_chosen_backend_and_keeps_it_visible(
        tmp_path, monkeypatch):
    """The descriptor has to survive all the way to the client, because it is
    what keeps "this session is remote" on screen for the LIFE of the session
    rather than only in the setup hint."""
    app, proj, owner = _coder_app(tmp_path, monkeypatch)
    _no_netpolicy(monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log",
                              "backend": "openai",
                              "backend_api_key": "sk-secret-value",
                              "backend_model": "gpt-4o"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["backend_info"] == {
            "backend": "openai", "leaves_machine": True,
            "target": "https://api.openai.com/v1", "model": "gpt-4o"}
        assert any("leave this machine" in n for n in body["notes"])
        # The credential goes in and never comes back out - not in the create
        # response and not in the session listing a browser polls.
        assert "sk-secret-value" not in r.text
        listed = client.get("/api/coder/sessions", headers=owner)
        assert "sk-secret-value" not in listed.text
        assert listed.json()["sessions"][0]["backend_info"]["leaves_machine"] is True


def test_route_default_session_is_unchanged_and_local(tmp_path, monkeypatch):
    """An unchanged form posts the body it always did and gets the behaviour it
    always got. The feature must not move the default."""
    app, proj, owner = _coder_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj)})
        assert r.status_code == 200, r.text
        assert r.json()["backend_info"]["leaves_machine"] is False
        assert r.json()["backend_info"]["backend"] == "local"


def test_an_on_machine_url_is_not_put_through_the_destination_policy(monkeypatch):
    """check_url's public-address arm refuses loopback, so applying the
    DESTINATION policy to a local model server breaks the feature's main use
    case. Destination policy is for destinations that leave."""
    called = []
    import localm.netpolicy as np
    monkeypatch.setattr(np, "check_url", lambda url: called.append(url))
    _resolve(_req(backend="url", backend_url="http://127.0.0.1:11434/v1"),
             session_mode="log")
    assert called == []


def test_the_shape_guard_runs_even_for_an_on_machine_url(monkeypatch):
    """Skipping the destination policy must NOT skip the parser-differential
    guard. urlparse reads this host as 'evil.example' while an HTTP client
    terminates the userinfo at the backslash and dials 127.0.0.1, so a URL that
    classifies one way here can be dialled another - which is exactly why the
    shape guard runs BEFORE anything branches on the classification."""
    import localm.netpolicy as np
    monkeypatch.setattr(np, "check_url", lambda url: None)
    with pytest.raises(HTTPException) as e:
        _resolve(_req(backend="url",
                      backend_url="http://127.0.0.1\\@evil.example/v1"),
                 session_mode="log")
    assert e.value.status_code == 400
    assert "backslash" in e.value.detail
