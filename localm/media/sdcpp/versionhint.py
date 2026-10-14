# SPDX-License-Identifier: AGPL-3.0-or-later
"""A tensor-name hint file that makes stable-diffusion.cpp recognise a UNet-only
SDXL, SD 1.x or SD 2.x model.

The file is passed as the context's ``model_path`` beside the real
``diffusion_model_path``; stable-diffusion.cpp reads only its tensor names and
shapes, never their values. Without it a UNet-only file fails to load with "get
sd version from file failed".

SDXL: four one-element tensors named like a second text encoder
(``conditioner.embedders.1``), ``middle_block.1``, ``output_blocks.3.1`` and
``input_blocks``. SD 1.x and SD 2.x: a ``middle_block.1`` tensor and an
unprefixed ``text_model.embeddings.token_embedding.weight`` whose last
dimension (768 or 1024) names the version.
See test_an_sdxl_unet_alone_gets_its_parts_and_the_version_hint and
test_an_sd1_or_sd2_unet_alone_gets_its_parts_and_the_version_hint.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

_SDXL = {
    "model.diffusion_model.input_blocks.localm_hint": (1,),
    "model.diffusion_model.middle_block.1.localm_hint": (1,),
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.localm_hint": (1,),
    "conditioner.embedders.1.localm_hint": (1,),
}
_TOKEN_EMBEDDING = "text_model.embeddings.token_embedding.weight"


def _clip_hint(width: int) -> dict:
    return {
        "model.diffusion_model.middle_block.1.localm_hint": (1,),
        _TOKEN_EMBEDDING: (1, width),
    }


HINT_TENSORS = {"sdxl": _SDXL, "sd1": _clip_hint(768), "sd2": _clip_hint(1024)}
HINT_FAMILIES = frozenset(HINT_TENSORS)
_FILE_NAMES = {"sdxl": "sdxl-version-hint.safetensors",
               "sd1": "sd1-version-hint.safetensors",
               "sd2": "sd2-version-hint.safetensors"}


def hint_bytes(family: str = "sdxl") -> bytes:
    """The safetensors file holding the hint tensors of *family* (``sdxl``,
    ``sd1`` or ``sd2``). Raises ``KeyError`` for another family."""
    header, offset = {}, 0
    for name, shape in HINT_TENSORS[family].items():
        size = 4
        for d in shape:
            size *= d
        header[name] = {"dtype": "F32", "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"\0" * offset


def hint_path(family: str) -> Path:
    """The hint file of *family* under the data directory, written when
    missing or different. Raises ``OSError`` when it cannot be written and
    ``KeyError`` for a family without a hint."""
    from localm.config import cache_dir
    path = cache_dir() / "sdcpp" / _FILE_NAMES[family]
    want = hint_bytes(family)
    try:
        if path.read_bytes() == want:
            return path
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(want)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return path
