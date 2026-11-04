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
import importlib
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
_BUNDLED_CODE_PACKAGE = "transformers_modules."
_ENCODE_NOTICE = "Calling super().encode with"
# Old module -> names that moved out of it -> the module that now holds each.
_MOVED_NAMES = {"transformers.utils.generic": {"OutputRecorder": "transformers.utils.output_capturing"}}


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


def _restore_moved_names(folder: Path) -> bool:
    """Make each name of :data:`_MOVED_NAMES` that the code of *folder* imports from its
    old module importable there again. True when a name was added."""
    restored = False
    for source in folder.glob("*.py"):
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for old_module, names in _MOVED_NAMES.items():
            imports = re.search(rf"from\s+{re.escape(old_module)}\s+import\s+(\([^)]*\)|[^\n]*)", text)
            if imports is None:
                continue
            for name, new_module in names.items():
                if re.search(rf"\b{name}\b", imports.group(1)) is None:
                    continue
                try:
                    old = importlib.import_module(old_module)
                    if not hasattr(old, name):
                        setattr(old, name, getattr(importlib.import_module(new_module), name))
                        restored = True
                except (ImportError, AttributeError) as e:
                    logger.debug("hf load: %s could not be restored in %s: %s", name, old_module, e)
    return restored


def apply_for(model_path: str) -> bool:
    """Prepare transformers for the code bundled in the model folder *model_path*: a name
    that code imports from where older releases kept it (:data:`_MOVED_NAMES`) is made
    importable there, and when that code calls the mask-building functions they are
    wrapped to accept the older keyword names. Returns True when this call changed
    anything; False for a folder whose code uses neither, or when transformers is
    already prepared."""
    folder = Path(model_path)
    if not folder.is_dir():
        return False
    wrapped = _restore_moved_names(folder)
    if not _folder_calls_mask_functions(folder):
        return wrapped
    try:
        import transformers.masking_utils as masking
    except ImportError:
        return wrapped
    for name in _MASK_FUNCTIONS:
        fn = getattr(masking, name, None)
        if fn is None or getattr(fn, _WRAPPED_MARK, False):
            continue
        setattr(masking, name, _accepting_legacy_arguments(fn))
        wrapped = True
    if wrapped:
        logger.debug("hf load: mask functions now accept the keyword names older model code uses")
    return wrapped


class _EncodeNoticeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_ENCODE_NOTICE)


def drop_encode_notice(tokenizer) -> None:
    """Drop the warning a bundled tokenizer's module logs on every encode with keyword
    arguments, which would land in the reply on a console. Other records of that module
    pass, and a tokenizer of transformers itself is left alone. See
    test_the_encode_notice_of_a_bundled_tokenizer_is_dropped_and_nothing_else."""
    module = type(tokenizer).__module__
    if module.startswith(_BUNDLED_CODE_PACKAGE):
        logging.getLogger(module).addFilter(_EncodeNoticeFilter())


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
    if config is None or getattr(config, "_attn_implementation", None) != "flash_attention_2":
        return False
    if importlib.util.find_spec("flash_attn") is not None:
        return False
    config._attn_implementation = "sdpa"
    logger.debug("hf load: flash_attn is not installed; the model uses sdpa attention")
    return True


def apply_to_model(model) -> bool:
    """Adjust a loaded model whose class comes from its folder's bundled code, written
    for an older transformers; a class of transformers itself is left alone. A model
    configured for ``flash_attention_2`` while ``flash_attn`` is not installed uses
    ``sdpa`` attention. A model whose class declares ``_is_stateful`` (it carries its own
    generation cache) is no longer handed a default ``DynamicCache`` by ``generate``, and
    its cache classes written for the older mask API are given the methods the current
    one calls. Returns True when anything was changed."""
    cls = type(model)
    if not cls.__module__.startswith(_BUNDLED_CODE_PACKAGE):
        return False
    changed = _sdpa_without_flash_attn(model)
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
