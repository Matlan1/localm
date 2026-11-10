# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compatibility for a model's bundled code that predates the pinned transformers.

Ouro (``model_type`` ``ouro``) ships ``modeling_ouro.py`` written for transformers
4.55. On 5.x it fails three ways: the config has no ``pad_token_id``, the rope
table has no ``"default"`` entry, and ``create_causal_mask`` takes ``inputs_embeds``
and no ``cache_position``. The Thinking checkpoints add a fourth: their cache class
overrides ``get_mask_sizes`` for a ``cache_position`` tensor, and transformers 5.x
passes the query length as an int. :func:`for_model` returns a :class:`RemoteCodeCompat`
that closes those, for that ``model_type`` only and only while the owner has
turned ``hf_trust_remote_code`` on; it returns None otherwise.

Each fix is additive: a value, table entry or keyword the running transformers
already provides is left as it is, and the rope entry exists only while the model
is built. The shim runs in the HF worker process.
"""

import contextlib
import functools
import inspect
import json
import logging
import sys
import weakref
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_TYPES = frozenset({"ouro"})

_MASK_FUNCTIONS = ("create_causal_mask", "create_sliding_window_causal_mask")
_REMOVED_MASK_KEYWORDS = frozenset({"cache_position"})

_ADAPTED = weakref.WeakSet()


def _model_type(model_path: str) -> Optional[str]:
    """``model_type`` from *model_path*'s config.json, or None when the file is
    missing, unreadable or states none."""
    try:
        data = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    value = data.get("model_type") if isinstance(data, dict) else None
    return value if isinstance(value, str) else None


def default_rope_parameters(config, device=None, seq_len=None):
    """``(inv_freq, attention_factor)`` of unscaled rotary embeddings, the
    function transformers 4.x registered as ``ROPE_INIT_FUNCTIONS["default"]``."""
    import torch
    base = config.rope_theta
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    exponents = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
    return 1.0 / (base ** (exponents / dim)), 1.0


def adapt_mask_function(fn):
    """*fn* (a transformers mask function) wrapped to accept the keywords of the
    older API: ``input_embeds`` is passed on as ``inputs_embeds`` and a
    ``cache_position`` keyword *fn* does not take is dropped. Every other keyword
    is passed as given, so a mismatch still raises. *fn* itself when it needs
    neither change or is already such a wrapper."""
    if fn in _ADAPTED:
        return fn
    params = inspect.signature(fn).parameters
    renames = "inputs_embeds" in params and "input_embeds" not in params
    drops = "cache_position" not in params and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not renames and not drops:
        return fn

    @functools.wraps(fn)
    def adapted(*args, **kwargs):
        if renames and "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if drops:
            for name in _REMOVED_MASK_KEYWORDS:
                kwargs.pop(name, None)
        return fn(*args, **kwargs)

    _ADAPTED.add(adapted)
    return adapted


def adapt_cache_class(cls) -> bool:
    """Make *cls*'s own ``get_mask_sizes`` accept the query length as an int, the
    form transformers 5.x passes, when it is written for a ``cache_position``
    tensor: an int is handed over as an empty meta tensor of that length, which
    has the ``shape`` the method reads. True when it was adapted; a class without
    its own such method, or one already adapted, is left as it is."""
    method = cls.__dict__.get("get_mask_sizes")
    if method is None or method in _ADAPTED:
        return False
    names = list(inspect.signature(method).parameters)
    if len(names) < 2 or names[1] != "cache_position":
        return False

    @functools.wraps(method)
    def get_mask_sizes(self, cache_position, *args, **kwargs):
        if isinstance(cache_position, int):
            import torch
            cache_position = torch.empty(cache_position, device="meta")
        return method(self, cache_position, *args, **kwargs)

    _ADAPTED.add(get_mask_sizes)
    cls.get_mask_sizes = get_mask_sizes
    return True


class RemoteCodeCompat:
    """The fixes for one model directory, applied around its ``from_pretrained``."""

    def __init__(self, model_path: str, trust_remote_code: bool):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code

    def config(self, tr):
        """The model's config with ``pad_token_id`` set to None when the running
        transformers leaves it off; None when the config cannot be loaded (the
        load itself then reports the failure)."""
        try:
            config = tr.AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code)
        except Exception as exc:
            logger.warning("remote-code compat: could not read the config (%s: %s)",
                           type(exc).__name__, exc)
            return None
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = None
        return config

    @contextlib.contextmanager
    def loading(self):
        """While the model is built, ``ROPE_INIT_FUNCTIONS["default"]`` is
        :func:`default_rope_parameters` when transformers has no such entry. An
        entry added here is removed on exit, success or failure: a ``"default"``
        left in the table changes how transformers reads other models' configs."""
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        added = "default" not in ROPE_INIT_FUNCTIONS
        if added:
            ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters
        try:
            yield
        finally:
            if added:
                ROPE_INIT_FUNCTIONS.pop("default", None)

    def after_load(self, model) -> None:
        """Adapt the mask functions and the cache classes in the module that
        defines *model*; a module that is not found, or that has no mask function,
        is logged."""
        name = type(model).__module__
        module = sys.modules.get(name)
        found = [attr for attr in _MASK_FUNCTIONS if callable(getattr(module, attr, None))]
        if not found:
            logger.warning("remote-code compat: no mask function to adapt in module %s", name)
        for attr in found:
            setattr(module, attr, adapt_mask_function(getattr(module, attr)))
        if module is None:
            return
        for value in list(vars(module).values()):
            if inspect.isclass(value) and value.__module__ == name:
                adapt_cache_class(value)


def for_model(model_path: str, trust_remote_code: bool) -> Optional[RemoteCodeCompat]:
    """The compat for *model_path*, or None unless its ``model_type`` is listed and
    *trust_remote_code* is on."""
    if not trust_remote_code or _model_type(model_path) not in _MODEL_TYPES:
        return None
    return RemoteCodeCompat(model_path, trust_remote_code)
