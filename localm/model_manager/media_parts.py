# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hugging Face folders that are one part of an image, video or music pipeline.

A diffusers-style pipeline ships each part as its own folder with a
``config.json``: the diffusion transformer, the autoencoder, the vocoder and the
text encoder. :func:`hf_media_part` names the part from that config, so a part
registers with its real role instead of ``unknown``. The ACE-Step music parts
also carry the model they belong to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional

ACE_STEP = "ACE-Step"

_TRANSFORMER_SUFFIXES = ("Transformer2DModel", "Transformer3DModel", "TransformerModel")
_VOCODER_MARKERS = ("HiFiGAN", "Vocoder")

# config.json "architectures" of the encoder-only text and vision encoders that
# image, video and music pipelines load as separate parts.
_TEXT_ENCODER_ARCHITECTURES = frozenset({
    "T5EncoderModel", "UMT5EncoderModel", "CLIPTextModel", "CLIPTextModelWithProjection",
    "CLIPVisionModel", "CLIPVisionModelWithProjection",
})

# The audio autoencoder of the ACE-Step music model encodes stereo audio.
_ACE_STEP_AUDIO_CHANNELS = 2


class MediaPart(NamedTuple):
    """One pipeline part: its registry ``model_type``, the config class that
    names it, and the model it belongs to (None when the config does not say)."""
    model_type: str
    architecture: str
    part_of: Optional[str]


def _read_config(folder: Path) -> Optional[dict]:
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return config if isinstance(config, dict) else None


def _first_architecture(config: dict) -> Optional[str]:
    archs = config.get("architectures")
    if isinstance(archs, list):
        for arch in archs:
            if isinstance(arch, str) and arch:
                return arch
    return None


def hf_media_part(folder: Path) -> Optional[MediaPart]:
    """The pipeline part the Hugging Face folder *folder* holds, from its
    ``config.json``, or None when the folder is not a part this module knows or
    the config cannot be read.

    A diffusers config names its class in ``_class_name``: a ``*Transformer2DModel``,
    ``*Transformer3DModel`` or ``UNet*`` class is a diffusion model, an
    ``Autoencoder*`` class a VAE, a HiFiGAN or vocoder class a vocoder. A
    transformers config names an encoder-only text or vision encoder in
    ``architectures``. ACE-Step parts are recognised by the class names and the
    stereo-audio input only its autoencoder has; a text encoder carries no
    marker of its own, see :func:`companion_part_of`."""
    config = _read_config(folder)
    if config is None:
        return None
    cls = config.get("_class_name")
    if isinstance(cls, str) and cls:
        if cls == "ACEStepTransformer2DModel":
            return MediaPart("diffusion-unet", cls, ACE_STEP)
        if cls.endswith(_TRANSFORMER_SUFFIXES) or cls.startswith("UNet"):
            return MediaPart("diffusion-unet", cls, None)
        if any(marker in cls for marker in _VOCODER_MARKERS):
            return MediaPart("vocoder", cls, ACE_STEP if cls == "ADaMoSHiFiGANV1" else None)
        if cls.startswith("Autoencoder"):
            audio = config.get("in_channels") == _ACE_STEP_AUDIO_CHANNELS
            return MediaPart("vae", cls, ACE_STEP if audio else None)
    arch = _first_architecture(config)
    if arch in _TEXT_ENCODER_ARCHITECTURES:
        return MediaPart("text-encoder", arch, None)
    return None


def text_embedding_width(folder: Path) -> Optional[int]:
    """The text-embedding width an ACE-Step transformer in *folder* reads
    (``text_embedding_dim``), or None for any other folder."""
    config = _read_config(folder)
    if config is None or config.get("_class_name") != "ACEStepTransformer2DModel":
        return None
    width = config.get("text_embedding_dim")
    return width if isinstance(width, int) and not isinstance(width, bool) else None


def encoder_hidden_size(folder: Path) -> Optional[int]:
    """The ``d_model`` of the T5-family encoder in *folder*, or None."""
    config = _read_config(folder)
    if config is None or _first_architecture(config) not in ("T5EncoderModel", "UMT5EncoderModel"):
        return None
    width = config.get("d_model")
    return width if isinstance(width, int) and not isinstance(width, bool) else None


def companion_part_of(folder: Path, transformer_widths: frozenset) -> Optional[str]:
    """The model a T5-family text encoder folder belongs to, from the folders
    registered beside it: ACE-Step when its hidden size equals the text-embedding
    width of a registered ACE-Step transformer (*transformer_widths*), else None."""
    width = encoder_hidden_size(folder)
    return ACE_STEP if width is not None and width in transformer_widths else None
