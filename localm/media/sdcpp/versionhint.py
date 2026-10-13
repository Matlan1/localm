# SPDX-License-Identifier: AGPL-3.0-or-later
"""A one-file tensor-name hint that lets stable-diffusion.cpp recognise a
UNet-only SDXL model.

stable-diffusion.cpp calls a UNet an SDXL model only when some loaded tensor
is named like a second text encoder (``conditioner.embedders.1``), and takes a
diffusers-layout UNet for SD 1.x without a ``middle_block.1`` tensor. A
UNet-only file loaded with separate CLIP-L and CLIP-G files carries neither
name, so the load fails with "get sd version from file failed". The hint file
holds four one-element tensors with those names and is passed as the context's
``model_path`` beside the real ``diffusion_model_path``; stable-diffusion.cpp
never reads their values.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

HINT_TENSORS = (
    "model.diffusion_model.input_blocks.localm_hint",
    "model.diffusion_model.middle_block.1.localm_hint",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.localm_hint",
    "conditioner.embedders.1.localm_hint",
)


def hint_bytes() -> bytes:
    """The safetensors file holding ``HINT_TENSORS``."""
    header = {name: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
              for i, name in enumerate(HINT_TENSORS)}
    raw = json.dumps(header).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * len(HINT_TENSORS))


def sdxl_hint_path() -> Path:
    """The SDXL hint file under the data directory, written when missing or
    different. Raises ``OSError`` when it cannot be written."""
    from localm.config import home_dir
    path = home_dir() / "cache" / "sdcpp" / "sdxl-version-hint.safetensors"
    want = hint_bytes()
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
