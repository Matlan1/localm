"""Refusal to read a store written by a newer build.

Stores that stamp a format version call :func:`check_stamp` on read, so a build
that does not understand the stamp stops and names the file instead of
reading it as the shape it knows and overwriting it on the next save.
"""

from __future__ import annotations

from pathlib import Path


class StoreFormatTooNew(RuntimeError):
    """A store file carries a format stamp newer than this build understands.

    Deliberately not a ``ValueError``/``OSError``: callers that treat those as
    "corrupt, skip it" must not swallow this. ``path`` is the file."""

    def __init__(self, path, stamp, understood, what: str = "store"):
        self.path = Path(path)
        self.stamp = stamp
        self.understood = understood
        super().__init__(
            f"{what} {self.path} has format version {stamp!r}, newer than the "
            f"{understood!r} this build understands; it was written by a newer "
            "localm. Update localm, or restore the data backup taken before "
            "the downgrade")


def check_stamp(stamp, understood: int, path, what: str = "store") -> None:
    """Raise :class:`StoreFormatTooNew` when *stamp* is an integer above
    *understood*. A missing stamp, a bool or a non-integer is not newer."""
    if isinstance(stamp, bool) or not isinstance(stamp, int):
        return
    if stamp > understood:
        raise StoreFormatTooNew(path, stamp, understood, what)
