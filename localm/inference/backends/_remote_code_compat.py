# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keeps a model's bundled code (``auto_map``) working with the installed transformers
when that code was written for an older release.

:func:`apply_for` makes ``transformers.masking_utils.create_causal_mask`` and
``create_sliding_window_causal_mask`` accept the keyword arguments the older
releases took (``input_embeds`` for ``inputs_embeds``; ``cache_position``, which
the current releases no longer use) when a model folder's code passes them. The
wrapped functions take every argument the originals take, so other models are
unaffected.

:func:`apply_to_model` stops ``generate`` from handing a default ``DynamicCache`` to
a loaded model whose class declares ``_is_stateful`` (it builds and returns its own
cache object), as older transformers releases did, and gives that model's cache classes
the methods the current mask functions call.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import logging
import re
import sys
from pathlib import Path

import torch

logger = logging.getLogger("localm")

_MASK_FUNCTIONS = ("create_causal_mask", "create_sliding_window_causal_mask")
_LEGACY_CALL = re.compile(r"\b(?:create_causal_mask|create_sliding_window_causal_mask)\s*\(")
_WRAPPED_MARK = "__localm_accepts_legacy_mask_arguments__"


def _accepting_legacy_arguments(fn):
    params = inspect.signature(fn).parameters

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "input_embeds" in kwargs and "input_embeds" not in params:
            kwargs.setdefault("inputs_embeds", kwargs.pop("input_embeds"))
        if "cache_position" in kwargs and "cache_position" not in params:
            kwargs.pop("cache_position")
        return fn(*args, **kwargs)

    setattr(wrapper, _WRAPPED_MARK, True)
    return wrapper


def _folder_calls_mask_functions(folder: Path) -> bool:
    for source in folder.glob("*.py"):
        try:
            if _LEGACY_CALL.search(source.read_text(encoding="utf-8", errors="replace")):
                return True
        except OSError:
            continue
    return False


def apply_for(model_path: str) -> bool:
    """Wrap transformers' mask-building functions to accept the older keyword names
    when the code bundled in the model folder *model_path* calls them. Returns True
    when a function was wrapped by this call; False when the folder does not call
    them, transformers has no such functions, or they are already wrapped."""
    folder = Path(model_path)
    if not folder.is_dir() or not _folder_calls_mask_functions(folder):
        return False
    try:
        import transformers.masking_utils as masking
    except ImportError:
        return False
    wrapped = False
    for name in _MASK_FUNCTIONS:
        fn = getattr(masking, name, None)
        if fn is None or getattr(fn, _WRAPPED_MARK, False):
            continue
        setattr(masking, name, _accepting_legacy_arguments(fn))
        wrapped = True
    if wrapped:
        logger.debug("hf load: mask functions now accept the keyword names older model code uses")
    return wrapped


def _adapt_cache_class(cache_cls) -> bool:
    """Give a cache class written for the older mask API the two things the current
    one calls: ``get_query_offset(layer_idx)`` and ``get_mask_sizes`` taking the
    query length as an int."""
    if hasattr(cache_cls, "get_query_offset"):
        return False
    original = cache_cls.get_mask_sizes

    def get_mask_sizes(self, query_length, layer_idx):
        if isinstance(query_length, int):
            query_length = torch.empty(query_length, device="meta")
        return original(self, query_length, layer_idx)

    cache_cls.get_mask_sizes = get_mask_sizes
    cache_cls.get_query_offset = lambda self, layer_idx=0: self.get_seq_length(layer_idx)
    return True


def _sdpa_without_flash_attn(model) -> bool:
    config = getattr(model, "config", None)
    if getattr(config, "_attn_implementation", None) != "flash_attention_2":
        return False
    if importlib.util.find_spec("flash_attn") is not None:
        return False
    config._attn_implementation = "sdpa"
    logger.debug("hf load: flash_attn is not installed; the model uses sdpa attention")
    return True


def apply_to_model(model) -> bool:
    """Adjust a loaded model whose bundled code was written for an older transformers.
    A model configured for ``flash_attention_2`` while ``flash_attn`` is not installed
    uses ``sdpa`` attention. A model whose class declares ``_is_stateful`` (it carries
    its own generation cache) is no longer handed a default ``DynamicCache`` by
    ``generate``, and its cache classes written for the older mask API are given the
    methods the current one calls. Returns True when anything was changed."""
    changed = _sdpa_without_flash_attn(model)
    cls = type(model)
    if not getattr(cls, "_is_stateful", False):
        return changed
    supports = getattr(cls, "_supports_default_dynamic_cache", None)
    if callable(supports) and supports():
        cls._supports_default_dynamic_cache = classmethod(lambda _cls: False)
        changed = True
    module = sys.modules.get(cls.__module__)
    for obj in list(vars(module).values()) if module is not None else []:
        if inspect.isclass(obj) and callable(getattr(obj, "get_mask_sizes", None)) \
                and callable(getattr(obj, "get_seq_length", None)):
            changed = _adapt_cache_class(obj) or changed
    if changed:
        logger.debug("hf load: %s keeps its own generation cache", cls.__name__)
    return changed
