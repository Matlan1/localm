# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tiny on-disk cache of per-model metadata learned at load time.

A GGUF's true transformer layer count (``llama_model_n_layer``) is only knowable
AFTER a native load - there is no pre-load header parser in the tree. The auto
GPU-layer sizing (``GgufBackend._auto_gpu_layers``) and the GUI VRAM estimate
(``/api/vram-estimate`` -> ``sysstats.estimate_vram``) both want that count to
scale a partial offload precisely, so we remember it the first time a model
loads and reuse it on every later load and estimate.

This is STATIC MODEL METADATA (filename, size, mtime, layer count), the same
class of data as ``registry.json`` already keeps - NOT chat or session content -
so it is ordinary data-home state and is NOT gated by privacy mode. It lives at
``<LOCALM_HOME>/model_meta.json``. Writes are best-effort: a failure only means a
later load re-estimates the layer count (correct, just less precise), so it is
logged at debug level and never raised.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .debuglog import logger

_META_FILENAME = "model_meta.json"
# Cap on remembered models; oldest entries are dropped first.
_MAX_ENTRIES = 256


def _meta_path() -> Path:
    """Path to the metadata cache under the CURRENT data home (read at call time
    so a test that monkeypatches ``config.HOME_DIR`` / ``LOCALM_HOME`` sees it)."""
    from localm import config
    return config.HOME_DIR / _META_FILENAME


def _model_key(model_path: str) -> Optional[str]:
    """Stable identity for a model file: ``name:size:mtime``. Returns None when
    the file cannot be stat'd (then the caller simply skips the cache)."""
    try:
        p = Path(model_path)
        st = p.stat()
        return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return None


def _load_all() -> dict:
    """Read the whole cache dict. Missing file -> {} (benign, expected on first
    run). Present but unreadable/corrupt -> {} plus a debug line: a torn write
    must not crash a load, but per rule 5 the corrupt case is distinguished from
    the absent one and surfaced, not silently equated with 'empty'."""
    path = _meta_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("model_meta: ignoring unreadable %s (%s)", path.name, exc)
        return {}
    return data if isinstance(data, dict) else {}


def cached_n_layers(model_path: str) -> Optional[int]:
    """The remembered true layer count for *model_path*, or None when unknown
    (never loaded, cache absent/corrupt, or the file changed size/mtime since)."""
    key = _model_key(model_path)
    if key is None:
        return None
    val = _load_all().get(key)
    if isinstance(val, int) and val > 0:
        return val
    return None


def store_n_layers(model_path: str, n_layers: int) -> None:
    """Remember *n_layers* for *model_path*. Best-effort: any failure is logged at
    debug and never raised; the next load re-estimates."""
    if not isinstance(n_layers, int) or n_layers <= 0:
        return
    key = _model_key(model_path)
    if key is None:
        return
    try:
        data = _load_all()
        if data.get(key) == n_layers:
            return  # already current - avoid a needless rewrite
        # Refresh insertion order (move-to-end) then trim oldest over the cap.
        data.pop(key, None)
        data[key] = n_layers
        while len(data) > _MAX_ENTRIES:
            data.pop(next(iter(data)))
        path = _meta_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Per-process temp name, so two instances cannot share one temp file.
        tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)   # atomic: a torn write can never corrupt the cache
    except OSError as exc:
        logger.debug("model_meta: could not persist layer count for %s (%s)",
                     Path(model_path).name, exc)
