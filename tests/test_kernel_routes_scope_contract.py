# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kernel HTTP route scope-gating, enforced by enumeration.

Unlike plugin routes (structurally gated by ``mount_router``'s
``include_router`` call - a plugin cannot add an ungated route), kernel routes in
``localm/inference/http_server.py`` and ``localm/inference/routes/*.py`` carry
``Depends(require_scope(...))`` individually per handler, with no shared base
router or FastAPI-level default dependency. This walks the live app's routes and
asserts every entry is accounted for.
"""

import pytest
from fastapi.routing import APIRoute

from localm.inference.http_server import create_app

# Dependency callables that enforce at least a valid, checked API key before
# the handler body runs:
#   - require_scope(...)'s returned closure (dep), used via
#     ``dependencies=[Depends(require_scope(scopes.X))]`` at the route
#     decorator - the normal case, most kernel routes.
#   - _require_auth: the baseline "any valid key, no specific scope required"
#     gate used by the chat-completions family (chat is the baseline scope,
#     see localm/scopes.py).
#   - require_fs_host: layers a host-filesystem-access check on top of the
#     same base ``_enforce_request`` used by require_scope/_require_auth,
#     used by /debug/stacks.
# All three ultimately call ``_enforce_request``, so any of them satisfies
# "this route is gated."
_GATED_DEP_QUALNAMES = {
    "require_scope.<locals>.dep",
    "_require_auth",
    "require_fs_host",
}

# Genuinely unauthenticated by design, keyed by (method, path). Each must work
# before any credential exists (login), or is a public identity, discovery or
# download endpoint that carries no secret.
_PUBLIC_ROUTES = {
    ("GET", "/health"),                  # liveness probe, no secret data
    ("GET", "/whoami"),                  # instance identity handshake
    ("GET", "/localm-ca.crt"),           # CA certificate (not the key)
    ("GET", "/api/session"),             # "am I logged in?" state check
    ("POST", "/api/session"),            # login: exchanges a key for a cookie
    ("POST", "/api/session/logout"),     # logout: must work post-key-clear
    # api_landing=True only (localm serve): a bare redirect to /docs, no data.
    ("GET", "/"),
}

# Gated by a bespoke mechanism OTHER than a FastAPI Depends (the check runs
# inline in the handler body), so no recognizable dependency name shows up in
# route.dependant.dependencies. A new entry here needs an equivalent inline gate.
_BESPOKE_GATED_ROUTES = {
    # attach-token-or-ADMIN check; localm/inference/routes/system.py
    ("POST", "/v1/surfaces/gui"),
    # per-instance coordination_token check; localm/inference/routes/gpu.py
    ("POST", "/v1/instances/cooperate-unload"),
}


@pytest.mark.parametrize("api_landing", [False, True], ids=["gui-mode", "api-landing"])
def test_every_kernel_route_is_gated_or_explicitly_allowlisted(api_landing):
    """Walk the live app's routes; every one must be either scope/auth-gated
    via a recognized FastAPI dependency, or present on one of the two
    hardcoded, commented allowlists above. A route added with neither fails
    this test.

    Parametrized over both real app shapes: ``create_app(None)`` (GUI mode)
    and ``create_app(None, api_landing=True)`` (the ``localm serve`` API-only
    shape), which registers one extra inline route (``GET /``) the GUI-mode
    shape never exercises."""
    app = create_app(None, api_landing=api_landing)
    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
# Sanity floor: fail loudly if create_app() yields far fewer routes than this, so
# the walk below cannot pass over nothing.
    assert len(api_routes) >= 30, (
        f"only {len(api_routes)} kernel APIRoute(s) found - create_app() "
        "may have changed shape; re-verify this test's route walk still "
        "reaches every kernel route before trusting a pass here")

    offenders = []
    for route in api_routes:
        dep_names = {getattr(dep.call, "__qualname__", "")
                     for dep in route.dependant.dependencies}
        if dep_names & _GATED_DEP_QUALNAMES:
            continue
        for method in sorted(route.methods):
            if (method, route.path) in _PUBLIC_ROUTES:
                continue
            if (method, route.path) in _BESPOKE_GATED_ROUTES:
                continue
            offenders.append(f"{method} {route.path}")

    assert not offenders, (
        "kernel route(s) with no recognized scope/auth dependency and not "
        "on the public or bespoke-gated allowlist (LM-DA-016): "
        + ", ".join(sorted(offenders))
        + " - add dependencies=[Depends(require_scope(...))], or if it is "
        "genuinely public or gated by its own inline mechanism, add it to "
        "the matching allowlist in this file with a one-line reason")
