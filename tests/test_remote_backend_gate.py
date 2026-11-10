# SPDX-License-Identifier: AGPL-3.0-or-later
"""The off-machine-model gate lives in the coder's HTTP backend constructors.

``make_openai_backend``, ``make_anthropic_backend`` and an ``HTTPBackend`` with
a non-loopback base URL ask ``remotegate`` and ``netpolicy.check_url`` before
the backend exists. The coder CLI's typed flags pass the "user typed it"
override; the ``LOCALCODER_URL`` environment variable does not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import click
import pytest
from click.core import ParameterSource

from localm import netpolicy, remotegate
from localm.audit import SessionMode
from localm.plugins.coder.backends.http import (
    HTTPBackend,
    make_anthropic_backend,
    make_openai_backend,
)
from localm.plugins.coder.cli import _main as cli_main

PKG = Path(remotegate.__file__).resolve().parent
REMOTE = "http://203.0.113.5/v1"


@pytest.fixture
def checked(monkeypatch):
    calls = []
    monkeypatch.setattr(netpolicy, "check_url", lambda url, **kw: calls.append(url))
    return calls


@pytest.mark.parametrize("make", [make_openai_backend, make_anthropic_backend])
def test_provider_constructors_refuse_in_privacy_mode(make, checked):
    with pytest.raises(remotegate.OffMachineModelRefused) as exc:
        make(gate_mode="privacy")
    assert "privacy mode" in str(exc.value)
    assert checked == []


@pytest.mark.parametrize("make", [make_openai_backend, make_anthropic_backend])
def test_provider_constructors_ask_the_network_policy_when_allowed(make, checked):
    backend = make(gate_mode="log")
    assert isinstance(backend, HTTPBackend)
    assert len(checked) == 1 and checked[0].startswith("https://api.")


def test_typed_override_passes_privacy_but_not_the_network_policy(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    with pytest.raises(netpolicy.NetworkPolicyError):
        make_openai_backend(gate_mode="privacy", user_typed_remote=True)


def test_typed_override_builds_when_policy_allows(checked):
    backend = make_openai_backend(gate_mode="privacy", user_typed_remote=True)
    assert isinstance(backend, HTTPBackend) and len(checked) == 1


def test_non_loopback_http_backend_is_gated(checked):
    with pytest.raises(remotegate.OffMachineModelRefused):
        HTTPBackend(REMOTE, "m", gate_mode="privacy")
    assert checked == []
    HTTPBackend(REMOTE, "m", gate_mode="full")
    assert checked == [REMOTE]


def test_loopback_backend_is_not_gated(checked):
    HTTPBackend("http://127.0.0.1:8642/v1", "m", gate_mode="privacy")
    HTTPBackend("http://localhost:8642/v1", "m", gate_mode="privacy")
    assert checked == []


def test_trusted_peer_skips_the_gate(checked):
    HTTPBackend(REMOTE, "m", gate_mode="privacy", trusted_peer=True)
    assert checked == []


def test_ambient_mode_is_used_without_gate_mode(monkeypatch, checked):
    monkeypatch.setattr(remotegate, "remote_allowed", lambda surface: False)
    with pytest.raises(remotegate.OffMachineModelRefused):
        HTTPBackend(REMOTE, "m")


def _build_url_backend(source, mode, tmp_path):
    ctx = click.Context(click.Command("coder"))
    ctx.set_parameter_source("url", source)
    with ctx:
        return cli_main._build_backend(
            None, REMOTE, "m", "k", False, None, True, False, tmp_path, False,
            session_mode=mode)


def test_cli_environment_url_is_refused_in_privacy_mode(tmp_path, checked):
    with pytest.raises(SystemExit) as exc:
        _build_url_backend(ParameterSource.ENVIRONMENT, SessionMode.PRIVACY, tmp_path)
    assert exc.value.code == 1
    assert checked == []


def test_cli_typed_url_proceeds_in_privacy_mode(tmp_path, checked):
    backend = _build_url_backend(ParameterSource.COMMANDLINE, SessionMode.PRIVACY,
                                 tmp_path)
    assert isinstance(backend, HTTPBackend) and checked == [REMOTE]


def test_cli_environment_url_proceeds_outside_privacy_mode(tmp_path, checked):
    backend = _build_url_backend(ParameterSource.ENVIRONMENT, SessionMode.LOG, tmp_path)
    assert isinstance(backend, HTTPBackend) and checked == [REMOTE]


def test_cli_network_policy_refusal_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    with pytest.raises(SystemExit) as exc:
        _build_url_backend(ParameterSource.COMMANDLINE, SessionMode.FULL, tmp_path)
    assert exc.value.code == 1


def test_reviewer_asks_remotegate(monkeypatch):
    from localm.plugins.coder import reviewer
    assert reviewer._is_privacy(SessionMode.PRIVACY) is True
    assert reviewer._is_privacy(SessionMode.FULL) is False
    monkeypatch.setattr(remotegate, "remote_allowed_for_mode",
                        lambda m: (_ for _ in ()).throw(RuntimeError("x")))
    assert reviewer._is_privacy(SessionMode.FULL) is True


_CONSTRUCTORS = {"HTTPBackend", "make_openai_backend", "make_anthropic_backend"}
_GATING_KEYWORDS = {"gate_mode", "trusted_peer", "localm_server"}


def test_every_backend_construction_outside_the_module_carries_the_gate():
    """A call outside backends/http.py must pass ``gate_mode`` (decided by its
    session), ``trusted_peer`` (an owner-registered peer) or
    ``localm_server=True`` (a connection to a localm on this machine)."""
    own = PKG / "plugins" / "coder" / "backends" / "http.py"
    bad = []
    seen = 0
    for path in PKG.rglob("*.py"):
        if path == own:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name not in _CONSTRUCTORS:
                continue
            seen += 1
            if not {k.arg for k in node.keywords} & _GATING_KEYWORDS:
                bad.append(f"{path.relative_to(PKG).as_posix()}:{node.lineno}")
    assert seen >= 10
    assert bad == []
