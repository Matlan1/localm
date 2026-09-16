# SPDX-License-Identifier: AGPL-3.0-or-later
"""The typed context the GUI model route groups share, plus the registry
precondition every model-naming route applies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import HTTPException

if TYPE_CHECKING:
    from localm.plugins.gui.jobs import JobManager


@dataclass(frozen=True)
class ModelRouteContext:
    """The services ``attach_gui`` hands the model route groups.

    ``active_model`` returns the name of the model currently serving requests
    ("" when none is). ``switch_model(name, *, force=False)`` is the
    coordinator that swaps the active engine and returns its status dict (or
    None for a minimal callable). ``jobs`` is the background-job manager the
    pull, scan, remove and warm-up routes start work on.
    """

    active_model: Callable[[], str]
    switch_model: Callable[..., Awaitable[Any]]
    jobs: JobManager

    @classmethod
    def from_ctx(cls, ctx) -> ModelRouteContext:
        """Build from the ``ctx`` namespace ``attach_gui`` passes every route
        group (attributes ``active_model``, ``switch_model`` and ``jobs``)."""
        return cls(active_model=ctx.active_model, switch_model=ctx.switch_model,
                   jobs=ctx.jobs)


def _require_registered(model: str, registry: dict | None = None) -> dict:
    """Raise 404 unless *model* is in the registry. Returns the registry, so a
    caller that needs it afterward (model_alias) doesn't load it twice."""
    from localm.config import load_registry
    if registry is None:
        registry = load_registry()
    if model not in registry:
        raise HTTPException(404, f"Model not registered: {model}")
    return registry
