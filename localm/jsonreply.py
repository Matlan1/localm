"""Parsing JSON from a reply or a text so that hostile input fails like malformed input.

``json.loads`` raises ``RecursionError`` (a ``RuntimeError``, not a ``ValueError``)
for deeply nested input. These helpers raise ``ValueError`` for it instead, so a
caller that already handles an unparseable reply by catching ``ValueError`` handles
a nested one the same way.
"""
import json
from typing import Any

__all__ = ["loads", "response_json"]


def loads(text: "str | bytes | bytearray") -> Any:
    """``json.loads(text)``; raises ``ValueError`` when *text* is nested too
    deeply to parse."""
    try:
        return json.loads(text)
    except RecursionError as e:
        raise ValueError("JSON is nested too deeply to parse") from e


def response_json(resp: Any) -> Any:
    """``resp.json()`` for a ``requests`` response; raises ``ValueError`` when the
    body is not valid JSON or is nested too deeply to parse."""
    try:
        return resp.json()
    except RecursionError as e:
        raise ValueError("JSON is nested too deeply to parse") from e
