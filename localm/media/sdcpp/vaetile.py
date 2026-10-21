# SPDX-License-Identifier: AGPL-3.0-or-later
"""VAE decode tile size for a GPU device that refuses ops on tensors above a size
(``_binding.max_tensor_bytes``): ``decode_tile_size`` (2D AutoencoderKL) and
``wan_decode_tile_size`` (Wan VAE, images and video) pick a tile whose largest
decode tensor fits under the limit."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


class DecodeTooLarge(ValueError):
    """Even the smallest decode tile has a tensor larger than the device takes."""


@dataclass(frozen=True)
class WanVae:
    """The decoder shape of a Wan-family VAE (stable-diffusion.cpp
    ``WAN::WanVAE``): base width, per-stage width multipliers, which upsample
    stages also double the frame count, latent channels, patch size, and
    whether the export uses 2D convolutions."""
    dec_dim: int = 96
    dim_mult: tuple = (1, 2, 4, 4)
    temporal_up: tuple = (True, True, False)
    z_dim: int = 16
    patch: int = 1
    wan2_2: bool = False
    conv2d: bool = False

    @property
    def scale(self) -> int:
        """Image pixels per latent pixel on a side."""
        return (1 << (len(self.dim_mult) - 1)) * self.patch


def wan_vae(version: str, conv_type: Optional[str]) -> Optional[WanVae]:
    """The Wan VAE the library builds for model *version* (its
    ``sd_get_model_version_name``) when its load log named the VAE's
    convolution type *conv_type* (``"2D"`` or ``"3D"``). None when
    *conv_type* is not known (the model has no Wan VAE)."""
    if conv_type not in ("2D", "3D"):
        return None
    conv2d = conv_type == "2D"
    if version == "Wan 2.2 TI2V":
        return WanVae(dec_dim=256, z_dim=48, patch=2, wan2_2=True, conv2d=conv2d,
                      temporal_up=(False,) * 3 if conv2d else (True, True, False))
    if version == "Qwen Image 2.1":
        return WanVae(dec_dim=144, dim_mult=(1, 2, 4, 8, 8), temporal_up=(True, True, True, False),
                      z_dim=64, wan2_2=True, conv2d=conv2d)
    return WanVae(conv2d=conv2d, temporal_up=(False,) * 3 if conv2d else (True, True, False))


def _conv_bytes(cin: int, w: int, h: int, t: int, three_d: bool) -> int:
    """The largest tensor of a 3x3 (2D) or 3x3x3 causal (3D) convolution over
    *cin* channels on a *w* x *h* x *t* input: its F16 im2col matrix, or for 3D
    also its padded F32 input (two cached frames ahead)."""
    if not three_d:
        return cin * 9 * 2 * w * h * t
    return max(cin * 27 * 2 * w * h * t, cin * (t + 2) * (w + 2) * (h + 2) * 4)


def wan_decode_peak_bytes(vae: WanVae, width: int, height: int, frames: int = 1) -> int:
    """The largest tensor, in bytes, of a full-frame decode of *frames* frames
    of *width* x *height* by *vae*: the im2col matrix (F16) of each
    convolution, the padded input of each 3D convolution, each upsampled
    activation (F32), the mid-block attention scores (F32) and the decoded
    frames (F32, up to 4 channels). One latent frame is decoded at a time, so
    each decoder tensor holds the frames one latent frame becomes (up to 4).
    See test_wan_decode_peak_bytes_at_the_default_video_size."""
    lw = -(-int(width) // vae.scale)
    lh = -(-int(height) // vae.scale)
    frames = max(1, int(frames))
    video = frames > 1 and not vae.conv2d
    three_d = not vae.conv2d
    dims = [vae.dec_dim * vae.dim_mult[-1]] + [vae.dec_dim * m for m in reversed(vae.dim_mult)]
    stages = len(vae.dim_mult)
    peak = [_conv_bytes(vae.z_dim, lw, lh, 1, three_d), _conv_bytes(dims[0], lw, lh, 1, three_d),
            (lw * lh) ** 2 * 4]
    t = 1
    for i in range(stages):
        w, h = lw << i, lh << i
        cin = dims[i] // 2 if not vae.wan2_2 and i > 0 else dims[i]
        cout = dims[i + 1]
        peak.append(_conv_bytes(max(cin, cout), w, h, t, three_d))
        if i < stages - 1:
            t_up = t * 2 if video and vae.temporal_up[i] else t
            peak.append(cout * 3 * 2 * w * h * t)
            peak.append(cout * t_up * (2 * w) * (2 * h) * 4)
            peak.append(_conv_bytes(cout, 2 * w, 2 * h, t_up, False))
            t = t_up
    peak.append(_conv_bytes(dims[-1], lw << (stages - 1), lh << (stages - 1), t, three_d))
    peak.append(4 * frames * lw * lh * vae.scale * vae.scale * 4)
    return max(peak)


def wan_decode_tile_size(width: int, height: int, frames: int,
                         max_tensor_bytes: Optional[int], vae: Optional[WanVae]) -> Optional[int]:
    """The decode tile side, in image pixels, that keeps every tensor of a
    *width* x *height* decode of *frames* frames by *vae*
    (``wan_decode_peak_bytes``) at most *max_tensor_bytes*: the largest
    multiple of ``TILE_STEP``. None when the whole frame fits or either value is
    not known. Raises DecodeTooLarge when a ``TILE_STEP`` tile does not fit."""
    if not max_tensor_bytes or max_tensor_bytes <= 0 or vae is None:
        return None
    width, height = int(width), int(height)
    if wan_decode_peak_bytes(vae, width, height, frames) <= max_tensor_bytes:
        return None
    side = max(width, height) // TILE_STEP * TILE_STEP
    while side >= TILE_STEP:
        if wan_decode_peak_bytes(vae, min(side, width), min(side, height),
                                 frames) <= max_tensor_bytes:
            return side
        side -= TILE_STEP
    need = wan_decode_peak_bytes(vae, min(TILE_STEP, width), min(TILE_STEP, height), frames)
    raise DecodeTooLarge(
        f"the VAE decode needs a {need / 1024 ** 2:.0f} MB buffer even in "
        f"{TILE_STEP} px tiles and this GPU runtime takes buffers of up to "
        f"{max_tensor_bytes / 1024 ** 2:.0f} MB")
