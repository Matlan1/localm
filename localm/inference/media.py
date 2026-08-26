# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decode base64-encoded or URL-fetched media into PIL / numpy objects."""

from __future__ import annotations

import base64
import io
import re
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    # numpy is referenced ONLY by decode_audio's return annotation, and this
    # module has `from __future__ import annotations`, so that annotation is
    # never evaluated at runtime. A module-scope import would make the whole
    # module unimportable without numpy - including decode_image_url, which needs
    # only Pillow - and numpy is not a core dependency (it arrives transitively
    # via the voice extra).
    import numpy as np


# A vision image can be several MB; cap the network fetch generously but
# bounded.
_IMAGE_MAX_BYTES = 25_000_000


def decode_image_url(url: str):
    """Return a PIL.Image.Image from a data-URI or http(s) URL.

    A remote (http/https) URL is fetched through localm.netpolicy so a chat
    'image_url' content part cannot turn the server into an SSRF proxy: the
    private-address guard, net_mode, and net_allow/net_deny all apply, every
    redirect hop is re-validated, and the body is size-capped. (Chat is the
    baseline capability - any key can send an image_url - so this fetch must be
    policy-checked like every other model-triggered request.)
    """
    try:
        from PIL import Image
    except ImportError as e:
        # Pillow is a core dependency, so this is reachable only on a build where
        # it failed to install or was removed. Report THAT, rather than letting an
        # ImportError escape: on the GGUF path this call runs inside the worker
        # process, whose dispatch loop treats any escaping exception as a native
        # fault and kills the process, evicting the model.
        # ImageDecodeUnavailable is an UnsupportedInputError, so the worker
        # reports it per-request and keeps serving (_runner.py).
        from localm.inference.backends.base import ImageDecodeUnavailable
        raise ImageDecodeUnavailable(
            "Cannot decode the attached image: the Pillow imaging library is not "
            "installed in this localm environment. Install it into the same "
            "environment (uv pip install pillow) and try again."
        ) from e

    if url.startswith("data:"):
        # data:image/jpeg;base64,<bytes>
        match = re.match(r"data:[^;]+;base64,(.+)", url, re.DOTALL)
        if not match:
            raise ValueError(f"Malformed data URI: {url[:60]}")
        raw = base64.b64decode(match.group(1))
        return Image.open(io.BytesIO(raw)).convert("RGB")

    from localm.netpolicy import safe_fetch_bytes
    _final_url, _content_type, raw = safe_fetch_bytes(
        url, max_bytes=_IMAGE_MAX_BYTES)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def decode_audio(b64: str, fmt: str) -> Tuple[np.ndarray, int]:
    """Return (samples_float32, sample_rate) from a base64-encoded audio blob."""
    import soundfile as sf

    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    samples, sr = sf.read(buf, dtype="float32")
    return samples, sr
