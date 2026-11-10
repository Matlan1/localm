# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small real image, audio and video files for tests whose fake generator must
write a file the media metadata strip accepts (a PNG, a FLAC, an MP4)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures" / "media"


def _chunk(ctype: bytes, data: bytes) -> bytes:
    return (len(data).to_bytes(4, "big") + ctype + data
            + zlib.crc32(ctype + data).to_bytes(4, "big"))


def png_bytes() -> bytes:
    """A structurally valid 1x1 PNG without metadata."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + _chunk(b"IEND", b""))


def flac_bytes() -> bytes:
    """A FLAC file with an embedded workflow comment, as ComfyUI writes it."""
    return (_FIXTURES / "prompt.flac").read_bytes()


def mp4_bytes() -> bytes:
    """An MP4 file with embedded workflow metadata, as ComfyUI writes it."""
    return (_FIXTURES / "prompt.mp4").read_bytes()
