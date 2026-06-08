"""Decode base64-encoded or URL-fetched media into PIL / numpy objects."""

from __future__ import annotations

import base64
import io
import re
from typing import Tuple

import numpy as np


def decode_image_url(url: str):
    """Return a PIL.Image.Image from a data-URI or http(s) URL."""
    from PIL import Image

    if url.startswith("data:"):
        # data:image/jpeg;base64,<bytes>
        match = re.match(r"data:[^;]+;base64,(.+)", url, re.DOTALL)
        if not match:
            raise ValueError(f"Malformed data URI: {url[:60]}")
        raw = base64.b64decode(match.group(1))
        return Image.open(io.BytesIO(raw)).convert("RGB")

    import requests
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def decode_audio(b64: str, fmt: str) -> Tuple[np.ndarray, int]:
    """Return (samples_float32, sample_rate) from a base64-encoded audio blob."""
    import soundfile as sf

    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    samples, sr = sf.read(buf, dtype="float32")
    return samples, sr


def pil_to_data_uri(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"
