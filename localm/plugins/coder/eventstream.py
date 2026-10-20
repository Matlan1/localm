# SPDX-License-Identifier: AGPL-3.0-or-later
"""JSON Lines output for unattended coder runs (``--output-format jsonl``)."""

from __future__ import annotations

import json
import threading
from typing import IO


class JsonlEventWriter:
    """Writes each record as one JSON object on one line and flushes it.

    Usable directly as an agent ``on_event`` sink: the agent's events are
    written unchanged (``{"type": ..., ...}``). Writes are serialised, since
    events also arrive from background sub-agent threads. Output is ASCII-only
    JSON, so no terminal encoding can corrupt a line; values that are not JSON
    types are written as their ``str()``."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    __call__ = write
