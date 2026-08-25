# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract test for ``_CROSS_ORIGIN_OK`` in ``localm/inference/http_server.py``."""

from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware

from localm.inference.http_server import create_app

# The two directories _CROSS_ORIGIN_OK used to match by prefix before this
# fix. Hardcoded (not read off the live tuple - see the module docstring for
# why) so any route appearing under either one is forced through review here,
# independent of whether the current _CROSS_ORIGIN_OK entry happens to match
# it.
_WATCHED_PREFIXES = ("/v1/surfaces/", "/v1/instances/")

# Reviewed set of (method, path) pairs allowed to sit under a watched
# prefix. Adding a route here is a deliberate security review, not a way to
# silence this test - see the reasoning at each entry's definition site in
# http_server.py and (for these two) in _BESPOKE_GATED_ROUTES,
# tests/test_kernel_routes_scope_contract.py.
_REVIEWED_WATCHED_ROUTES = {
    ("POST", "/v1/surfaces/gui"),
    ("POST", "/v1/instances/cooperate-unload"),
}


def _live_cross_origin_ok(app) -> tuple:
    """Recover the actual ``_CROSS_ORIGIN_OK`` tuple ``_origin_guard`` closes over, from its live closure cell - the real deployed value, not a hardcoded guess at what it should be."""
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


def test_every_route_under_a_watched_prefix_is_reviewed():
    app = create_app(None)
    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    # Sanity floor, same reasoning as test_kernel_routes_scope_contract.py:
    # if create_app()'s shape changes so drastically that far fewer routes
    # are found, the walk below would trivially "pass" over nothing.
    assert len(api_routes) >= 30, (
        f"only {len(api_routes)} kernel APIRoute(s) found - create_app() "
        "may have changed shape; re-verify this test's route walk still "
        "reaches every kernel route before trusting a pass here")

    offenders = []
    for route in api_routes:
        if not route.path.startswith(_WATCHED_PREFIXES):
            continue
        for method in sorted(route.methods):
            if (method, route.path) not in _REVIEWED_WATCHED_ROUTES:
                offenders.append(f"{method} {route.path}")

    assert not offenders, (
        "route(s) under a watched, formerly-prefix-exempted directory "
        "(/v1/surfaces/ or /v1/instances/) not on the reviewed allowlist: "
        + ", ".join(sorted(offenders)) + " - such a route was never "
        "individually reviewed for the cross-origin/CSRF refusal and the "
        "open-mode shell-token gate; confirm it does NOT need to be added "
        "to _CROSS_ORIGIN_OK in http_server.py, then add it to "
        "_REVIEWED_WATCHED_ROUTES above with a one-line reason")


def test_cross_origin_ok_tuple_entries_are_full_paths_not_prefixes():
    """Pins the production half of the fix directly."""
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
