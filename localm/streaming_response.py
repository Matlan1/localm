# SPDX-License-Identifier: AGPL-3.0-or-later
"""The streaming response every localm route returns."""

from __future__ import annotations

import logging

from starlette.responses import StreamingResponse as _StarletteStreamingResponse
from starlette.types import Send

__all__ = ["StreamingResponse"]

logger = logging.getLogger("localm")


class StreamingResponse(_StarletteStreamingResponse):
    """``starlette.responses.StreamingResponse`` that closes its body iterator
    (``aclose``) whenever streaming stops: after the last chunk, on a client
    disconnect, and when a send raises (a credential refused mid-stream by
    ``CredentialRecheckMiddleware``). A body generator left suspended at a
    ``yield`` gets ``GeneratorExit`` there, so its ``finally`` runs before the
    response returns. A body iterator without ``aclose`` is left as it is.
    A failure while closing is logged at WARNING and does not replace the
    outcome of the stream."""

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    logger.warning("closing a streaming response's body failed",
                                   exc_info=True)
