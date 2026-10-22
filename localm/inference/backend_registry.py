# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inference backends registered by plugins (``Host.register_inference_backend``).

Process-wide list of ``InferenceBackendDescriptor`` objects, asked in
registration order by ``engine.create_backend`` for a model path the built-in
GGUF and Hugging Face backends refuse. Holds no vendor code.

``_LOCK`` is a leaf lock: it guards only the list, and no descriptor callback
(``claims``/``create``) and no other lock is ever taken while it is held. See
test_claims_runs_without_the_registry_lock.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from localm.debuglog import logger

_LOCK = threading.Lock()
_DESCRIPTORS: list = []


def register(descriptor: Any) -> None:
    """Add *descriptor* at the end, or in the place of an earlier one with the
    same ``name`` from the same plugin. Raises ValueError for an empty name or a
    ``claims``/``create`` that is not callable."""
    if not getattr(descriptor, "name", ""):
        raise ValueError("an inference backend descriptor needs a name")
    if not callable(getattr(descriptor, "claims", None)):
        raise ValueError(f"inference backend {descriptor.name!r}: claims is not callable")
    if not callable(getattr(descriptor, "create", None)):
        raise ValueError(f"inference backend {descriptor.name!r}: create is not callable")
    key = (descriptor.name, descriptor.plugin_name)
    with _LOCK:
        for i, d in enumerate(_DESCRIPTORS):
            if (d.name, d.plugin_name) == key:
                _DESCRIPTORS[i] = descriptor
                return
        _DESCRIPTORS.append(descriptor)


def unregister_plugin(plugin_name: str) -> int:
    """Remove every descriptor *plugin_name* registered. Returns how many."""
    with _LOCK:
        before = len(_DESCRIPTORS)
        _DESCRIPTORS[:] = [d for d in _DESCRIPTORS if d.plugin_name != plugin_name]
        return before - len(_DESCRIPTORS)


def registered() -> list:
    """A snapshot of the registered descriptors, in registration order."""
    with _LOCK:
        return list(_DESCRIPTORS)


def claiming(model_path: str) -> Optional[Any]:
    """The first registered descriptor whose ``claims(model_path)`` is True, or
    None. A ``claims`` that raises is logged at WARNING and counts as not
    claiming."""
    for descriptor in registered():
        try:
            if descriptor.claims(str(model_path)):
                return descriptor
        except Exception as e:
            logger.warning("inference backend %r (plugin %r): claims() raised on %s "
                           "(%s: %s); treated as not claiming",
                           descriptor.name, descriptor.plugin_name, model_path,
                           type(e).__name__, e)
    return None


def builtin_refuses(model_path: str) -> bool:
    """True when neither built-in backend would load *model_path*: not a GGUF
    file, and not a Hugging Face folder the HF backend accepts (a folder in a
    format ``unsupported.hf_folder_refusal`` names is refused by it)."""
    from pathlib import Path

    from localm.inference.engine import _is_gguf, _is_hf_dir
    from localm.model_manager.unsupported import hf_folder_refusal
    if _is_gguf(model_path):
        return False
    if _is_hf_dir(model_path):
        return hf_folder_refusal(Path(model_path)) is not None
    return True


def _load_plugin_backends() -> bool:
    """Load the installed, enabled plugins that register an inference backend,
    once per process (``plugins.engine.load_inference_backend_plugins``). True
    when this call loaded any. A failure is logged at WARNING and reads False.

    Covers a process whose plugins are not loaded yet: an in-process ``localm
    run``, and a server building its startup model before it attaches plugins.
    Takes no engine lock, so it may run under ``engine._LOAD_LOCK``. See
    test_lazy_plugin_load_under_the_engine_load_lock."""
    try:
        from localm.plugins.engine import load_inference_backend_plugins
        return bool(load_inference_backend_plugins())
    except Exception as e:
        logger.warning("could not load the plugins that register inference "
                       "backends (%s: %s)", type(e).__name__, e)
        return False


def backend_for(model_path: str) -> Optional[Any]:
    """The descriptor that runs *model_path*, or None when a built-in backend
    loads it or no registered backend claims it. When the built-ins refuse it
    and nothing registered claims it, the enabled backend plugins are loaded
    once per process and asked again."""
    if not builtin_refuses(model_path):
        return None
    descriptor = claiming(model_path)
    if descriptor is None and _load_plugin_backends():
        descriptor = claiming(model_path)
    return descriptor
