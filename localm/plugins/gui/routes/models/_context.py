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


def resident_engine(model: str, registry: dict):
    """The loaded engine in this process that runs the model file registered as
    *model*, under any of its names (a registered alias, or a name another
    process has since renamed), or None. A path that cannot be resolved counts
    as resident, because it cannot be ruled out. Does filesystem I/O; callers
    on the event loop use the executor."""
    from pathlib import Path

    import localm.inference.http_server as _hs
    from localm.model_manager import _entry_path, names_same_model
    model_path = _entry_path(registry.get(model))
    for key, engine in list(_hs._engines.items()):
        if not getattr(engine, "loaded", False):
            continue
        try:
            if names_same_model(key, model, registry):
                return engine
            engine_path = getattr(engine, "model_path", None)
            if engine_path is None or model_path is None:
                continue
            if Path(str(engine_path)).resolve() == Path(model_path).resolve():
                return engine
        except (OSError, ValueError):
            return engine
    return None


def _require_registered(model: str, registry: dict | None = None) -> dict:
    """Raise 404 unless *model* is in the registry. Returns the registry, so a
    caller that needs it afterward (model_alias) doesn't load it twice."""
    from localm.config import load_registry
    if registry is None:
        registry = load_registry()
    if model not in registry:
        raise HTTPException(404, f"Model not registered: {model}")
    return registry
