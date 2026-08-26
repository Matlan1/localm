# SPDX-License-Identifier: AGPL-3.0-or-later
"""Route groups for the localm GUI surface.

Each module exposes ``register(app, ctx)`` which defines that group's routes on
the FastAPI ``app``. ``attach_gui`` (in ``localm.plugins.gui.web``) creates the
shared services (the coder ``SessionManager`` and the background ``JobManager``),
publishes them on ``app.state``, builds the shared ``ctx`` (the active-model
accessor, the model-switch callable, and the job manager), and calls each group's
``register``.

The page-shell ``/`` + ``/index.html`` handler and the catch-all static mount stay
in ``attach_gui`` itself: the shell carries conditional, per-request auth seeding
and the mount MUST be registered last so the API routes above always win.

Shared request models and pure helpers (multipart parsing, the uploads-dir
confinement, the shell-token HTML) live as module-level names in
``localm.plugins.gui.web`` and are imported here by name; the cross-module
auth/scope helpers stay in ``localm.inference.http_server`` and are imported from
there, exactly as ``web.py`` does.
"""
