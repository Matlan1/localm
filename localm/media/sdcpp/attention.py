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


@dataclass(frozen=True)
class ImageAttention:
    """An image model's largest attention: *heads* x *head_dim*, image pixels
    per query token on a side (*stride*), the text tokens joined to the image
    tokens (*text_tokens*), the tokens each reference image adds
    (*ref_extra*, beside its image tokens), whether reference images join the
    attention (*refs*), and how many images run in one graph (*batch*)."""
    heads: int
    head_dim: int
    stride: int
    text_tokens: int = 0
    batch: int = 1
    refs: bool = False
    ref_extra: int = 0

    def tokens(self, width: int, height: int, ref_count: int = 0) -> int:
        """Query tokens of the attention over a *width* x *height* image with
        *ref_count* reference images, each counted at the image's size."""
        image = -(-int(width) // self.stride) * -(-int(height) // self.stride)
        refs = int(ref_count) * (image + self.ref_extra) if self.refs else 0
        return image + self.text_tokens + refs

    def score_bytes(self, width: int, height: int, ref_count: int = 0) -> int:
        """Bytes of the F32 attention score matrix the attention builds
        without flash attention: heads x batch x tokens^2 x 4."""
        n = self.tokens(width, height, ref_count)
        return self.heads * self.batch * n * n * 4

    def flops(self, width: int, height: int, ref_count: int = 0) -> int:
        """Floating point operations of the attention as one flash attention op."""
        return attention_flops(self.tokens(width, height, ref_count),
                               self.heads * self.head_dim) * self.batch


_IMAGE_SHAPES = {
    "sd1": ImageAttention(8, 40, 8, batch=2),
    "sd2": ImageAttention(5, 64, 8, batch=2),
    "sdxl": ImageAttention(10, 64, 16, batch=2),
    "sd3": ImageAttention(38, 64, 16, text_tokens=308),
    "flux": ImageAttention(24, 128, 16, text_tokens=512, refs=True),
    "chroma": ImageAttention(24, 128, 16, text_tokens=512, refs=True),
    "ovis-image": ImageAttention(24, 128, 16, text_tokens=512, refs=True),
    "flux2": ImageAttention(48, 128, 16, text_tokens=512, refs=True),
    "qwen-image": ImageAttention(24, 128, 16, text_tokens=512, refs=True, ref_extra=400),
    "z-image": ImageAttention(30, 128, 16, text_tokens=512, refs=True),
    "anima": ImageAttention(16, 128, 16, refs=True),
    "ernie-image": ImageAttention(32, 128, 16, text_tokens=512),
}


def image_attention(family: Optional[str], hidden: Optional[int] = None,
                    heads: Optional[int] = None) -> Optional[ImageAttention]:
    """The largest attention of an image model of localm *family*; *hidden*
    and *heads* are the hidden size and head count its load log reported, which
    replace the family's largest size. None for a family with no known shape."""
    shape = _IMAGE_SHAPES.get(family or "")
    if shape is None:
        return None
    if hidden and heads and hidden % heads == 0:
        return ImageAttention(int(heads), int(hidden) // int(heads), shape.stride,
                              shape.text_tokens, shape.batch, shape.refs, shape.ref_extra)
    return shape


def largest_square(fits, step: int = 64, top: int = 4096) -> int:
    """The largest side, a multiple of *step* up to *top*, whose square image
    ``fits(side)``; 0 when not even *step* does."""
    best = 0
    for side in range(step, top + 1, step):
        if fits(side):
            best = side
        else:
            break
    return best
