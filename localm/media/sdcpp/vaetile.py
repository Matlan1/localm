# SPDX-License-Identifier: AGPL-3.0-or-later
"""VAE decode tiling for a GPU runtime that refuses tensors above a size.

A ggml backend that refuses an op (``ggml_backend_dev_supports_op``) hands it to
the CPU, so a full-frame decode whose largest tensor is over the device's limit
runs those ops on the CPU. ``decode_tile_size`` picks a tile small enough for the
decode to stay on the GPU.
"""

from __future__ import annotations

import math
from typing import Optional

TILE_STEP = 64


def kl_decode_bytes_per_pixel(decoder_ch: int) -> int:
    """Bytes per output pixel of the largest tensor in a 2D AutoencoderKL decode
    whose decoder base width is *decoder_ch*: the F32 im2col matrix of a 3x3
    conv over ``2 * decoder_ch`` channels at full resolution.
    See test_kl_decoder_peak_bytes_per_pixel."""
    return 2 * int(decoder_ch) * 3 * 3 * 4


def decode_tile_size(width: int, height: int, max_tensor_bytes: Optional[int],
                     decoder_ch: Optional[int]) -> Optional[int]:
    """The VAE decode tile side, in image pixels, for a *width* x *height* image
    decoded by a 2D AutoencoderKL of base width *decoder_ch* on a device whose
    ops accept tensors up to *max_tensor_bytes*: the largest multiple of
    ``TILE_STEP`` whose square fits, at least ``TILE_STEP``. None when the whole
    image fits (no tiling) or either value is not known."""
    if not max_tensor_bytes or max_tensor_bytes <= 0 or not decoder_ch or decoder_ch <= 0:
        return None
    bpp = kl_decode_bytes_per_pixel(decoder_ch)
    if int(width) * int(height) * bpp <= max_tensor_bytes:
        return None
    side = math.isqrt(max_tensor_bytes // bpp)
    return max(TILE_STEP, side // TILE_STEP * TILE_STEP)
