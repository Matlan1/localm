# SPDX-License-Identifier: AGPL-3.0-or-later
"""The longest single GPU job of a video model's denoiser step: the
self-attention over every video token, which the runtime runs as ONE flash
attention op. ``video_attention`` gives a loaded model's attention shape,
``dispatch_seconds`` the time that op takes on a device of measured
throughput (``_binding.attention_rate``), and ``max_frames`` the longest clip
whose op stays within ``SAFE_DISPATCH_SECONDS``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ._binding import attention_flops

SAFE_DISPATCH_SECONDS = 1.0
DRIVER_RESET_SECONDS = 2
TEMPORAL_COMPRESSION = 4

_WAN_SIZES = (("-1.3B", 1536), ("-5B", 3072), ("-14B", 5120))
_WAN_LARGEST_DIM = 5120
_HUNYUAN_VAE_STRIDE = 16
_HUNYUAN_LARGEST_DIM = 3072
_HUNYUAN_TEXT_TOKENS = 512


@dataclass(frozen=True)
class VideoAttention:
    """A video model's self-attention: hidden size *dim* (heads times head
    size), image pixels per video token on a side (*stride*), and the text
    tokens joined to the video tokens (*text_tokens*)."""
    dim: int
    stride: int
    text_tokens: int = 0

    def tokens(self, width: int, height: int, frames: int) -> int:
        """Tokens of one self-attention over a *width* x *height* clip of
        *frames* frames."""
        latent_frames = (max(1, int(frames)) - 1) // TEMPORAL_COMPRESSION + 1
        per_frame = -(-int(width) // self.stride) * -(-int(height) // self.stride)
        return latent_frames * per_frame + self.text_tokens


def video_attention(version: str, desc: str = "", dit_hidden: Optional[int] = None,
                    dit_patch: Optional[int] = None) -> VideoAttention:
    """The self-attention of the video model the library loaded as *version*
    (its ``sd_get_model_version_name``) with description *desc* (a Wan load
    log line such as ``Wan2.1-T2V-1.3B``); *dit_hidden* and *dit_patch* are
    the HunyuanVideo hidden size and spatial patch its load log reported. A
    model whose size is not known gets the largest of its family; any other
    model gets the largest Wan shape with HunyuanVideo's text tokens."""
    if version.startswith("Wan"):
        dim = next((d for suffix, d in _WAN_SIZES if desc.endswith(suffix)), _WAN_LARGEST_DIM)
        return VideoAttention(dim=dim, stride=32 if version == "Wan 2.2 TI2V" else 16)
    if version == "Hunyuan Video":
        return VideoAttention(dim=int(dit_hidden or _HUNYUAN_LARGEST_DIM),
                              stride=_HUNYUAN_VAE_STRIDE * int(dit_patch or 1),
                              text_tokens=_HUNYUAN_TEXT_TOKENS)
    return VideoAttention(dim=_WAN_LARGEST_DIM, stride=16, text_tokens=_HUNYUAN_TEXT_TOKENS)


def dispatch_seconds(att: VideoAttention, width: int, height: int, frames: int,
                     rate: float) -> float:
    """Seconds one self-attention op over a *width* x *height* clip of
    *frames* frames takes on a device that runs *rate* attention floating
    point operations per second."""
    return attention_flops(att.tokens(width, height, frames), att.dim) / float(rate)


def max_frames(att: VideoAttention, width: int, height: int, rate: float,
               budget: float = SAFE_DISPATCH_SECONDS) -> int:
    """The most frames (``4k + 1``) of a *width* x *height* clip whose
    self-attention op takes at most *budget* seconds at *rate*; 0 when even one
    frame takes longer."""
    if dispatch_seconds(att, width, height, 1, rate) > budget:
        return 0
    tokens = math.isqrt(int(budget * rate / (4 * att.dim)))
    per_frame = att.tokens(width, height, 1) - att.text_tokens
    latent_frames = max(1, (tokens - att.text_tokens) // per_frame)
    while latent_frames > 1 and dispatch_seconds(
            att, width, height, (latent_frames - 1) * TEMPORAL_COMPRESSION + 1, rate) > budget:
        latent_frames -= 1
    return (latent_frames - 1) * TEMPORAL_COMPRESSION + 1
