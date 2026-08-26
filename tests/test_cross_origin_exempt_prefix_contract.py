# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract test for ``_CROSS_ORIGIN_OK`` in ``localm/inference/http_server.py``.

Every entry in that tuple must be a full route path, never a directory prefix
(it is matched with ``str.startswith()``), because a prefix silently exempts
every FUTURE route added under it from both the cross-origin/CSRF refusal and
the open-mode shell-token gate - without its author writing anything that
looks like a security decision.

Two things are pinned independently, and both are needed:

- The route walk below watches two DIRECTORIES (``/v1/surfaces/``,
  ``/v1/instances/``), hardcoded here rather than read off the live tuple.
  Reading it off the live (narrowed) tuple could never flag a new, unrelated
  route under either directory, because such a route no longer matches the
  narrowed tuple at all - correct production behavior, but blind to the
  "was this new route reviewed" question this walk exists to ask. Any route
  appearing under either directory, authenticated or not, exempt or not,
  fails this walk until someone looks at it.
- The tuple-contents assertion catches a regression back to directory-prefix
  matching directly, which the route walk alone cannot: the two real exempt
  routes satisfy a prefix match exactly as well as a full-path match, so a
  route walk against a prefixed tuple still finds no offenders.

``_CROSS_ORIGIN_OK`` is local to ``create_app()``, not a module attribute, so
the second check recovers it from the live ``_origin_guard`` middleware's
closure - the actual object controlling production behavior, not a
re-implementation of it.
"""

from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware

from localm.inference.http_server import create_app

# The two directories watched here. Hardcoded rather than read off the live
# tuple, so any route under either one is forced through review.
_WATCHED_PREFIXES = ("/v1/surfaces/", "/v1/instances/")

# Reviewed set of (method, path) pairs allowed to sit under a watched prefix.
_REVIEWED_WATCHED_ROUTES = {
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


def test_every_route_under_a_watched_prefix_is_reviewed():
    app = create_app(None)
    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    # Floor on the route count, so the walk below cannot pass over nothing.
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
