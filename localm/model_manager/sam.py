# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recognise a Segment Anything (SAM / SAM 2) PyTorch checkpoint from its file."""

import zipfile
from pathlib import Path

from localm.debuglog import logger

_SUFFIXES = (".pth", ".pt")
_PICKLE_MAX_BYTES = 8 * 1024 * 1024
_SAM1_KEYS = (b"image_encoder.patch_embed", b"prompt_encoder", b"mask_decoder")
_SAM2_KEYS = (b"image_encoder.trunk", b"sam_prompt_encoder", b"sam_mask_decoder")


def is_sam_checkpoint(path: Path) -> bool:
    """True when *path* is a ``.pth`` / ``.pt`` zip checkpoint whose tensor names
    are the Segment Anything layout (image encoder, prompt encoder and mask
    decoder). Reads only the archive's ``data.pkl`` member (at most 8 MiB) as
    bytes and never unpickles it. False for any other file and for a file that
    cannot be read or decompressed (the failure is logged at debug level); never
    raises."""
    if path.suffix.lower() not in _SUFFIXES:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            member = next((i for i in zf.infolist() if i.filename.endswith("data.pkl")), None)
            if member is None or member.file_size > _PICKLE_MAX_BYTES:
                return False
            blob = zf.read(member)
    except Exception as e:
        logger.debug("SAM checkpoint probe could not read %s: %s: %s",
                     path, type(e).__name__, e)
        return False
    return all(k in blob for k in _SAM1_KEYS) or all(k in blob for k in _SAM2_KEYS)
