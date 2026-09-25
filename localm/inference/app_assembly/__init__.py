# SPDX-License-Identifier: AGPL-3.0-or-later
"""App-assembly components of ``localm.inference.http_server.create_app()``.

``create_app()`` is the boot order. Each module here holds steps of it: functions
that add exception handlers, middleware, routes or ``app.state`` fields to the app
they are given, in the order ``create_app()`` calls them:

    context      the AppContext handed to the route groups; the app.state fields
    errors       exception handlers
    diagnostics  the debug-mode request log and GET /debug/stacks
    security     CORS, the origin / shell-token gate, security headers, docs guard
    transport    body cap, disconnect signal, request progress (outermost)
    mounting     the api-mode landing, the route groups, the plugin engine (last)

None of these modules holds state. Process-lifetime state (the engine registry,
the running loop, the hang alarm, GPU coordination, the audit log) stays in
``http_server``, together with the functions that write it: ``_init_engine_state``,
``_init_session_audit`` and ``_make_lifespan``. A name ``http_server`` defines is
read here through that module (``_hs.<name>``), at the same moment the old
``create_app()`` closure read it, so a monkeypatch on
``localm.inference.http_server`` still reaches this code. ADR-0023 records the
seams.
"""
