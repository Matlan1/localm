# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI capability probes are INDEPENDENT.

`/api/capabilities` reports whether three optional features are configured: the
bug-report upload channel, the read-only issues view, and the update banner. A
single try block around two of them:

    try:
        from localm import issue_tracker, updater
        issues_avail = issue_tracker.available()
        update_avail = updater.available()
    except Exception:
        issues_avail = update_avail = False

`issue_tracker.available()` and `updater.available()` read INDEPENDENT endpoints,
so a raise in the first leaves `update_avail` False having never called
`updater.available()` at all, and a user silently stops being offered updates
because an unrelated issues probe broke.

Failing CLOSED is right (hiding a control that might not work beats offering one
that errors). Failing closed on BEHALF OF A FEATURE THAT WAS NEVER PROBED is not.
"""

from __future__ import annotations

import importlib

import pytest

sysroutes = importlib.import_module("localm.plugins.gui.routes.system")


def _boom():
    raise RuntimeError("probe exploded")


# --------------------------------------------------------------------------- #
#  The isolation itself                                                        #
# --------------------------------------------------------------------------- #

def test_a_failing_probe_reports_only_its_own_feature_unavailable():
    """The defect, stated directly: a broken issues probe must not withdraw the
    update banner."""
    assert sysroutes._probe_available("issues", _boom) is False
    assert sysroutes._probe_available("update", lambda: True) is True


def test_a_raising_probe_does_not_call_the_others():
    """Proves ISOLATION rather than just the returned values: the surviving
    probes must actually run, not merely happen to hold a truthy default."""
    calls = []

    def _tracked(name, value):
        def _fn():
            calls.append(name)
            return value
        return _fn

    assert sysroutes._probe_available("bug", _boom) is False
    assert sysroutes._probe_available("issues", _tracked("issues", True)) is True
    assert sysroutes._probe_available("update", _tracked("update", True)) is True

    assert calls == ["issues", "update"], calls


def test_every_probe_can_fail_independently():
    """Whichever one breaks, the other two are unaffected - not just the pair
    that happened to share a try block."""
    for broken in range(3):
        fns = [lambda: True, lambda: True, lambda: True]
        fns[broken] = _boom
        got = [sysroutes._probe_available(f"p{i}", f) for i, f in enumerate(fns)]
        assert got[broken] is False, got
        assert all(v for i, v in enumerate(got) if i != broken), (broken, got)


# --------------------------------------------------------------------------- #
#  Fail closed, but never silently                                             #
# --------------------------------------------------------------------------- #

def test_a_failing_probe_fails_closed():
    """Hiding a control that might not work beats offering one that errors."""
    assert sysroutes._probe_available("anything", _boom) is False


def test_a_failing_probe_is_recorded(caplog):
    """Fail-closed is correct; fail-closed in TOTAL silence is not. The reason
    must be discoverable, at debug rather than warning because this runs on every
    capability fetch and a broken probe would otherwise flood."""
    import logging
    with caplog.at_level(logging.DEBUG, logger="localm"):
        sysroutes._probe_available("issues view", _boom)

    assert any("issues view" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


def test_a_truthy_non_bool_is_normalised():
    """The route ships this straight to the GUI as JSON, so it must be a real
    bool rather than whatever the probe happened to return."""
    assert sysroutes._probe_available("x", lambda: "yes") is True
    assert sysroutes._probe_available("x", lambda: 0) is False


# --------------------------------------------------------------------------- #
#  The real probes are wired to the right functions                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fn,module,attr", [
    (sysroutes._bugreport_upload_available, "localm.bugreport", "upload_available"),
    (sysroutes._issues_available, "localm.issue_tracker", "available"),
    (sysroutes._update_available, "localm.updater", "available"),
])
def test_each_wrapper_calls_its_own_module(fn, module, attr, monkeypatch):
    """Guards against the wrappers being cross-wired, which would reintroduce
    exactly the bug this file exists for - one feature answering for another."""
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, attr, lambda: True)
    assert fn() is True
    monkeypatch.setattr(mod, attr, lambda: False)
    assert fn() is False


# --------------------------------------------------------------------------- #
#  The property at the ROUTE, not at the helper                                #
# --------------------------------------------------------------------------- #

@pytest.fixture
def caps_app(tmp_path, monkeypatch):
    """GUI stack on a throwaway home, no owner key, so /api/capabilities answers
    in open mode without needing a minted key."""
    from pathlib import Path

    from fastapi import FastAPI

    from localm.plugins.gui.web import attach_gui

    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "model-a")
    return app


def test_a_broken_issues_probe_does_not_withdraw_the_update_banner(
        caps_app, monkeypatch):
    """The user-visible regression, driven through the REAL route.

    The helper tests above cannot catch this on their own: they exercise
    _probe_available directly, so restoring the shared try block would not flip
    any of them. This one goes through /api/capabilities, so it fails if the two
    probes are ever bundled again."""
    from fastapi.testclient import TestClient

    from localm import issue_tracker, updater

    monkeypatch.setattr(issue_tracker, "available", _boom)
    monkeypatch.setattr(updater, "available", lambda: True)

    with TestClient(caps_app) as c:
        body = c.get("/api/capabilities").json()

    assert body["issues_available"] is False        # its own probe raised
    assert body["update_available"] is True, body   # untouched by the other's fault


def test_a_broken_bugreport_probe_leaves_the_other_two_alone(
        caps_app, monkeypatch):
    """Same property from the other end, so the isolation is not accidentally
    specific to one ordering."""
    from fastapi.testclient import TestClient

    from localm import bugreport, issue_tracker, updater

    monkeypatch.setattr(bugreport, "upload_available", _boom)
    monkeypatch.setattr(issue_tracker, "available", lambda: True)
    monkeypatch.setattr(updater, "available", lambda: True)

    with TestClient(caps_app) as c:
        body = c.get("/api/capabilities").json()

    assert body["bugreport_upload"] is False
    assert body["issues_available"] is True, body
    assert body["update_available"] is True, body
