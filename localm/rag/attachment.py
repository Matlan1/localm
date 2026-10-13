# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat or coder attachment (a base64 upload) converted to plain text,
entirely in memory. Nothing is written to disk."""

from __future__ import annotations

import base64
import os
from typing import Optional

from . import extract as _extract

MAX_B64_CHARS = 40_000_000
MAX_DECODED_BYTES = 30_000_000


class AttachmentError(Exception):
    """An attachment was refused. ``status`` is the HTTP status that fits: 413
    too large, 400 not valid base64, 422 no extractable text."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def refusal_text(message: str, filename: str) -> str:
    """*message* from an extraction refusal of *filename*, with the advice to
    load a vision model replaced for an image: attachment extraction never
    describes images."""
    if os.path.splitext(filename)[1].lower() in _extract._IMAGE_SUFFIXES \
            and "vision" in message:
        return (f"No extractable text in {filename}. Images are not converted to "
                "text here; attach them in Chat with a vision model.")
    return message


def extract_attachment(filename: str, content_b64: str,
                       max_chars: Optional[int] = None) -> dict:
    """``{"filename", "text", "chars", "truncated"}`` for the attachment
    *filename* whose bytes are *content_b64*. The whole text unless *max_chars*
    asks for an excerpt (clamped to 500..200000). Raises ``AttachmentError``.
    Blocking: run it off the event loop."""
    if len(content_b64) > MAX_B64_CHARS:
        raise AttachmentError(413, "Attachment too large (max 30 MB)")
    try:
        data = base64.b64decode(content_b64, validate=True)
    except Exception as exc:
        raise AttachmentError(400, "content_b64 is not valid base64") from exc
    if len(data) > MAX_DECODED_BYTES:
        raise AttachmentError(413, "Attachment too large (max 30 MB)")
    try:
        text = _extract.extract_bytes(data, filename)
    except _extract.ExtractError as exc:
        raise AttachmentError(422, refusal_text(str(exc), filename)) from exc
    if max_chars is None:
        return {"filename": filename, "text": text,
                "chars": len(text), "truncated": False}
    cap = max(500, min(max_chars, 200_000))
    return {"filename": filename, "text": text[:cap],
            "chars": len(text), "truncated": len(text) > cap}
