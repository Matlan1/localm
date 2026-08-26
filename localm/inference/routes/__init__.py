# SPDX-License-Identifier: AGPL-3.0-or-later
"""Route groups for the localm inference server.

Each module exposes ``register(app, ctx)`` which defines that group's routes on
the FastAPI ``app``. ``create_app`` (in ``localm.inference.http_server``) builds
the app, installs middleware + the lifespan + the 500 backstop, constructs the
shared ``ctx`` (audit log / transcript / session mode), and calls each group's
``register``.

The shared engine state (``_engine``, ``_inference_sem``) lives as module globals
in ``http_server``; route modules read the live values via
``import localm.inference.http_server as _hs`` (so a model swap that reassigns the
global is seen here too). The cross-module auth/scope helpers also stay in
``http_server`` and are imported from there.
"""
