# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI model routes: registry list/load, VRAM estimate, pull/remove/alias, and
HuggingFace/CivitAI discovery.

The routes are registered in four groups by responsibility:

- ``inventory``: what is registered and resident (list, scan, roles,
  shortcuts, load, unload, embedder warm-up, VRAM estimate, GPUs);
- ``acquisition``: getting model files onto this machine (pull-token
  redemption, pull, media preflight, curated ComfyUI downloads);
- ``mutation``: changing a registration in place (remove, alias, rename,
  type, relocate);
- ``discovery``: searching HuggingFace and CivitAI.

``register`` is the facade ``attach_gui`` calls; each group's own ``register``
takes the typed :class:`ModelRouteContext` built here from the shared ``ctx``.
"""

from __future__ import annotations

from fastapi import FastAPI

from localm.plugins.gui.routes.models import (acquisition, discovery,
                                              inventory, mutation)
from localm.plugins.gui.routes.models._context import ModelRouteContext

__all__ = ["ModelRouteContext", "register"]


def register(app: FastAPI, ctx) -> None:
    """Mount every GUI model route on *app*, in the order the groups are
    listed in the module docstring."""
    context = ModelRouteContext.from_ctx(ctx)
    inventory.register(app, context)
    acquisition.register(app, context)
    mutation.register(app, context)
    discovery.register(app, context)
