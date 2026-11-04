# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reads the file layout of a Kimi-Linear checkpoint: whether its routed experts can be read one at
a time. Imports neither torch nor transformers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("localm")

MODEL_TYPE = "kimi_linear"
_EXPERT_KEY = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w([123])\.weight$")


def _safetensors_header(path: Path) -> tuple[dict, int]:
    """``(header, offset of the data)`` of a ``.safetensors`` file."""
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(size)), 8 + size


def _weight_map(folder: Path) -> dict:
    """``{weight name: file name}`` of the checkpoint in *folder*; empty when it has no
    readable ``.safetensors`` weights."""
    index = folder / "model.safetensors.index.json"
    try:
        if index.is_file():
            return dict(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
        single = folder / "model.safetensors"
        if single.is_file():
            names = (n for n in _safetensors_header(single)[0] if n != "__metadata__")
            return dict.fromkeys(names, single.name)
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.debug("hf load: could not read the weight map of %s: %s", folder, e)
    return {}


def worth_using(model_path: str, available_bytes: int) -> bool:
    """True for a ``kimi_linear`` checkpoint in per-expert ``.safetensors`` form whose weights
    do not fit in *available_bytes*."""
    folder = Path(model_path)
    try:
        if json.loads((folder / "config.json").read_text(encoding="utf-8")).get("model_type") != MODEL_TYPE:
            return False
        total = sum(f.stat().st_size for f in folder.glob("*.safetensors"))
    except (OSError, ValueError, AttributeError):
        return False
    if total <= available_bytes:
        return False
    return any(_EXPERT_KEY.match(name) for name in _weight_map(folder))
