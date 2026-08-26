# SPDX-License-Identifier: AGPL-3.0-or-later
"""Privacy regression tests for `_strip_png_metadata`.

ComfyUI PNGs embed the full prompt/workflow in tEXt/eXIf chunks, so localm strips
them before the saved copy is treated as clean. The function's own docstring
promises "A failure here must NOT be silent."

The gap these tests pin: a custom workflow save node can emit WebP/JPEG to the
hardcoded `.png` path. The stripper only knows PNG chunks, so for a non-PNG file
it skips the loop and returns "" (= success) with NO warning, while the caller
still prints "Image saved ...". In privacy mode the retained copy then still
carries the prompt/EXIF while the user is told the strip succeeded.

Covered:
  * a non-PNG (JPEG-like) blob returns a NON-EMPTY warning and is left untouched.
  * a real PNG carrying a tEXt "prompt" chunk still strips clean (returns "")
    and the secret bytes are gone from the file.
  * a missing file still returns a warning.
"""

from __future__ import annotations

import struct
import zlib

from localm.image_gen import comfy

PNG_SIG = b"\x89PNG\r\n\x1a\n"
_SECRET = b"prompt\x00super secret prompt do not leak"


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    """A single PNG chunk: len + type + data + real CRC32(type+data)."""
    return (len(data).to_bytes(4, "big") + ctype + data
            + zlib.crc32(ctype + data).to_bytes(4, "big"))


def _png_with_text_prompt() -> bytes:
    """A minimal, structurally valid 1x1 RGB PNG carrying a tEXt prompt chunk."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")  # 1 filter byte + one RGB pixel
    return (PNG_SIG
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"tEXt", _SECRET)
            + _png_chunk(b"IDAT", idat)
            + _png_chunk(b"IEND", b""))


def test_non_png_file_warns_and_is_left_untouched(tmp_path):
    """A JPEG/WebP written to the .png path cannot be stripped, so it must return
    a non-empty warning and the caller must stop implying a clean strip.
    """
    blob = b"\xff\xd8\xff\xe1" + b"Exif" + b"prompt: secret"  # JPEG/EXIF magic
    path = tmp_path / "generated.png"
    path.write_bytes(blob)

    warning = comfy._strip_png_metadata(path)

    assert warning, "non-PNG file must yield a non-empty warning, not silent success"
    assert "prompt" in warning.lower() or "exif" in warning.lower()
    # An unrecognised format is left byte-for-byte alone.
    assert path.read_bytes() == blob


def test_real_png_strips_text_chunk_clean(tmp_path):
    """A genuine PNG with a tEXt prompt chunk strips clean and returns ""."""
    path = tmp_path / "generated.png"
    path.write_bytes(_png_with_text_prompt())

    warning = comfy._strip_png_metadata(path)

    assert warning == "", f"clean PNG strip should return '', got {warning!r}"
    out = path.read_bytes()
    assert out.startswith(PNG_SIG)
    assert b"super secret prompt do not leak" not in out, "tEXt chunk not stripped"
    assert b"tEXt" not in out
    # Non-metadata chunks survive.
    assert b"IHDR" in out and b"IDAT" in out and b"IEND" in out


def test_missing_file_warns(tmp_path):
    """A missing output file still returns a warning (unchanged behavior)."""
    warning = comfy._strip_png_metadata(tmp_path / "does_not_exist.png")

    assert warning, "a missing file must return a warning, not silent success"
