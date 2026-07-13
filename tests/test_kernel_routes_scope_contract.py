# SPDX-License-Identifier: AGPL-3.0-or-later
"""LM-DA-016 - kernel HTTP route scope-gating is enforced by convention plus a
PR-template checklist, not an automated route-enumeration test.

Unlike plugin routes (structurally gated by ``mount_router``'s
``include_router`` call - a plugin literally cannot add an ungated route),
kernel routes in ``localm/inference/http_server.py`` and
``localm/inference/routes/*.py`` carry ``Depends(require_scope(...))``
individually per handler, with no shared base router or FastAPI-level default
dependency. This mirrors the coder plugin's own default-deny contract test
for its tool registry (``test_scope_allowlist_is_default_deny`` in
``tests/test_coder_security_2026_07_01.py``): walk the real registry (here,
the live app's routes) and assert every entry is accounted for, so an
omission fails CI instead of relying on a manual checklist alone.
"""

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

# Genuinely unauthenticated by design, keyed by (method, path). Each is on
# this list because it must work before any credential exists (login) or is
# a deliberately public identity/discovery/download endpoint that carries no
# secret - see the matching route's own docstring in
# localm/inference/routes/*.py for the full reasoning.
_PUBLIC_ROUTES = {
    ("GET", "/health"),                  # liveness probe, no secret data
    ("GET", "/whoami"),                  # instance identity handshake
    ("GET", "/localm-ca.crt"),           # CA certificate (not the key)
    ("GET", "/api/session"),             # "am I logged in?" state check
    ("POST", "/api/session"),            # login: exchanges a key for a cookie
    ("POST", "/api/session/logout"),     # logout: must work post-key-clear
}

# Gated by a bespoke, documented mechanism OTHER than a FastAPI Depends (the
# check runs inline in the handler body), so no recognizable dependency name
# shows up in route.dependant.dependencies. Each is a deliberate, narrow
# exception reviewed in the 2026-07-13 design audit (LM-DA-016) - adding a
# new route here requires an equivalent inline gate, reviewed the same way.
_BESPOKE_GATED_ROUTES = {
    # attach-token-or-ADMIN check; localm/inference/routes/system.py
    ("POST", "/v1/surfaces/gui"),
    # per-instance coordination_token check; localm/inference/routes/gpu.py
    ("POST", "/v1/instances/cooperate-unload"),
}


def test_every_kernel_route_is_gated_or_explicitly_allowlisted():
    """Walk the live app's routes; every one must be either scope/auth-gated
    via a recognized FastAPI dependency, or present on one of the two
    hardcoded, commented allowlists above. A future route added with
    neither - the exact gap LM-DA-016 flagged - fails this test."""
    app = create_app(None)
    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    # Sanity floor: if create_app()'s shape changes so drastically that far
    # fewer routes are found, the walk below would trivially "pass" over
    # nothing - fail loudly instead of giving a false green.
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
