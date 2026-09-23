# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic control mutations for the two trust-boundary decision classes
that live in localm/inference/http_server.py, outside the [tool.mutmut]
only_mutate modules scripts/check_mutation_floors.py ratchets:

* an unsafe route exempted from the origin gate: every kernel route with a
  state-changing method must refuse a cross-origin request unless it is on
  the reviewed exempt list below, and the live ``_CROSS_ORIGIN_OK`` tuple must
  equal that list exactly;
* ``bind_host`` replaced by the peer address: the loopback-only surfaces
  (docs, /debug/stacks) must follow ``app.state.bind_host`` in BOTH
  directions - hidden on a network bind even for a 127.0.0.1 peer (what every
  peer looks like behind portmux), and served on a loopback bind even for a
  network peer.

The other four classes (a weakened scope check, a skipped SSRF redirect
re-validation, a deny-to-allow fallback, a path-confinement bypass) are
mutmut mutants of the only_mutate modules and are pinned as ``controls`` in
scripts/mutation_baseline.json.
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from localm.inference.http_server import create_app

_UNSAFE = ("POST", "PUT", "PATCH", "DELETE")

# The reviewed contents of _CROSS_ORIGIN_OK, in order. Exempting another route
# from the cross-origin/CSRF refusal is a change to this list, in the same
# diff, with its own reason.
_REVIEWED_CROSS_ORIGIN_OK = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/surfaces/gui",
    "/v1/instances/cooperate-unload",
)

_CROSS_ORIGIN = {"Origin": "http://localhost:9999"}


@pytest.fixture
def app(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return create_app(None)


def _live_cross_origin_ok(app) -> tuple:
    for mw in app.user_middleware:
        if mw.cls is not BaseHTTPMiddleware:
            continue
        fn = (getattr(mw, "kwargs", None) or {}).get("dispatch")
        if fn is None or fn.__name__ != "_origin_guard":
            continue
        freevars = fn.__code__.co_freevars
        assert "_CROSS_ORIGIN_OK" in freevars
        return fn.__closure__[freevars.index("_CROSS_ORIGIN_OK")].cell_contents
    raise AssertionError("_origin_guard not found in app.user_middleware")


def _concrete(path: str) -> str:
    """A request path for a route template: every ``{param}`` becomes ``x``."""
    out = []
    for seg in path.split("/"):
        out.append("x" if seg.startswith("{") and seg.endswith("}") else seg)
    return "/".join(out)


class TestOriginGateExemption:
    def test_live_exempt_tuple_equals_the_reviewed_list(self, app):
        assert _live_cross_origin_ok(app) == _REVIEWED_CROSS_ORIGIN_OK

    def test_every_unsafe_kernel_route_refuses_cross_origin_unless_reviewed(self, app):
        """The behavioural half: a route quietly added to _CROSS_ORIGIN_OK stops
        answering 403 here, whatever the tuple test above says."""
        routes = [r for r in app.routes if isinstance(r, APIRoute)]
        unsafe = [(m, r.path) for r in routes for m in sorted(r.methods) if m in _UNSAFE]
        assert len(unsafe) >= 20, unsafe
        offenders = []
        with TestClient(app) as c:
            for method, path in unsafe:
                if path in _REVIEWED_CROSS_ORIGIN_OK:
                    continue
                r = c.request(method, _concrete(path), headers=_CROSS_ORIGIN)
                detail = r.json().get("detail", "") if r.headers.get("content-type", "").startswith("application/json") else r.text
                if r.status_code != 403 or "cross-origin" not in str(detail).lower():
                    offenders.append(f"{method} {path} -> {r.status_code} {detail!r}"[:160])
        assert not offenders, "\n".join(offenders)

    def test_a_reviewed_inference_route_is_reachable_cross_origin(self, app):
        """Positive control: the walk above is not passing because everything
        403s - an exempt route gets past the origin guard."""
        with TestClient(app) as c:
            r = c.post("/v1/chat/completions", json={}, headers=_CROSS_ORIGIN)
        assert "cross-origin" not in r.text.lower()


class TestBindHostNotPeerDecidesLoopbackOnlySurfaces:
    LOOPBACK_PEER = ("127.0.0.1", 40001)
    NETWORK_PEER = ("203.0.113.5", 40002)

    def test_network_bind_hides_docs_and_debug_even_for_a_loopback_peer(self, app):
        """Behind portmux every request peer is 127.0.0.1; a decision keyed on
        the peer would serve these to the whole LAN."""
        with TestClient(app, client=self.LOOPBACK_PEER) as c:
            app.state.bind_host = "0.0.0.0"
            token = app.state.shell_token
            assert c.get("/docs").status_code == 404
            assert c.get("/openapi.json").status_code == 404
            assert c.get("/debug/stacks",
                         headers={"Authorization": f"Bearer {token}"}).status_code == 404

    def test_loopback_bind_serves_docs_and_debug_even_for_a_network_peer(self, app):
        """The other direction: a decision keyed on the peer would 404 here."""
        with TestClient(app, client=self.NETWORK_PEER) as c:
            app.state.bind_host = "127.0.0.1"
            token = app.state.shell_token
            assert c.get("/docs").status_code == 200
            assert c.get("/openapi.json").status_code == 200
            r = c.get("/debug/stacks", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 200, r.text

    def test_bind_host_unset_defaults_to_loopback(self, app):
        with TestClient(app, client=self.NETWORK_PEER) as c:
            assert not hasattr(app.state, "bind_host") or app.state.bind_host is None
            assert c.get("/docs").status_code == 200
