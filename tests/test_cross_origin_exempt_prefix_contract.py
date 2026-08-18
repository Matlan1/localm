# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract test for ``_CROSS_ORIGIN_OK`` in ``localm/inference/http_server.py``.

Every entry in that tuple must be a full route path, never a directory prefix
(it is matched with ``str.startswith()``), because a prefix silently exempts
every FUTURE route added under it from both the cross-origin/CSRF refusal and
the open-mode shell-token gate - without its author writing anything that
looks like a security decision.

Two things are pinned independently, and both are needed:

- The route walk below catches an unreviewed route landing under an exempt
  prefix.
- The tuple-contents assertion catches a regression back to directory-prefix
  matching directly, which the route walk alone cannot: the two real exempt
  routes satisfy a prefix match exactly as well as a full-path match, so a
  route walk against a reverted (prefixed) tuple still finds no offenders.

``_CROSS_ORIGIN_OK`` is local to ``create_app()``, not a module attribute, so
it is recovered here from the live ``_origin_guard`` middleware's closure -
the actual object controlling production behavior, not a re-implementation
of it.
"""

from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware

from localm.inference.http_server import create_app

# Reviewed set of (method, path) pairs allowed to sit under a _CROSS_ORIGIN_OK
# entry. Adding a route here is a deliberate security review, not a way to
# silence this test - see the reasoning at each entry's definition site in
# http_server.py and (for the last two) in _BESPOKE_GATED_ROUTES,
# tests/test_kernel_routes_scope_contract.py.
_REVIEWED_CROSS_ORIGIN_OK_ROUTES = {
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/surfaces/gui"),
    ("POST", "/v1/instances/cooperate-unload"),
}


def _live_cross_origin_ok(app) -> tuple:
    """Recover the actual ``_CROSS_ORIGIN_OK`` tuple ``_origin_guard`` closes
    over, from its live closure cell - the real deployed value, not a
    hardcoded guess at what it should be."""
    for mw in app.user_middleware:
        if mw.cls is not BaseHTTPMiddleware:
            continue
        fn = (getattr(mw, "kwargs", None) or {}).get("dispatch")
        if fn is None or fn.__name__ != "_origin_guard":
            continue
        freevars = fn.__code__.co_freevars
        assert "_CROSS_ORIGIN_OK" in freevars, (
            "_origin_guard no longer closes over _CROSS_ORIGIN_OK - this "
            "test needs updating to match the new implementation")
        idx = freevars.index("_CROSS_ORIGIN_OK")
        return fn.__closure__[idx].cell_contents
    raise AssertionError(
        "could not find _origin_guard in app.user_middleware - "
        "create_app()'s middleware registration changed shape; "
        "this test needs updating to match it")


def test_cross_origin_ok_routes_match_reviewed_allowlist():
    app = create_app(None)
    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    # Sanity floor, same reasoning as test_kernel_routes_scope_contract.py:
    # if create_app()'s shape changes so drastically that far fewer routes
    # are found, the walk below would trivially "pass" over nothing.
    assert len(api_routes) >= 30, (
        f"only {len(api_routes)} kernel APIRoute(s) found - create_app() "
        "may have changed shape; re-verify this test's route walk still "
        "reaches every kernel route before trusting a pass here")

    cross_origin_ok = _live_cross_origin_ok(app)

    offenders = []
    for route in api_routes:
        if not route.path.startswith(cross_origin_ok):
            continue
        for method in sorted(route.methods):
            if (method, route.path) not in _REVIEWED_CROSS_ORIGIN_OK_ROUTES:
                offenders.append(f"{method} {route.path}")

    assert not offenders, (
        "route(s) under a cross-origin-exempt prefix not on the reviewed "
        "allowlist: " + ", ".join(sorted(offenders)) + " - such a route "
        "silently inherits the cross-origin/CSRF refusal exemption and the "
        "open-mode shell-token exemption from _CROSS_ORIGIN_OK; review it "
        "and either give it its own credential check plus a listing in "
        "_REVIEWED_CROSS_ORIGIN_OK_ROUTES above (with a one-line reason), "
        "or leave it off _CROSS_ORIGIN_OK entirely")


def test_cross_origin_ok_tuple_entries_are_full_paths_not_prefixes():
    """Pins the production half of the fix directly. The route-walk test
    above alone cannot catch a regression back to directory-prefix matching:
    both real exempt routes satisfy a prefix match exactly as well as a
    full-path match, so it would report the same clean result either way."""
    app = create_app(None)
    cross_origin_ok = _live_cross_origin_ok(app)
    assert cross_origin_ok[-2:] == (
        "/v1/surfaces/gui", "/v1/instances/cooperate-unload"), (
        f"_CROSS_ORIGIN_OK's last two entries are {cross_origin_ok[-2:]!r}, "
        "expected the two full route paths ('/v1/surfaces/gui', "
        "'/v1/instances/cooperate-unload') - a directory-prefix entry here "
        "(e.g. '/v1/surfaces/') would silently exempt every future route "
        "added under it from the cross-origin/CSRF refusal and the "
        "open-mode shell-token gate")
