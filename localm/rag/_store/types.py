# SPDX-License-Identifier: AGPL-3.0-or-later
"""Callback types taken by ``Collection``'s indexing and search methods, and the
small names every module of the store shares."""

from __future__ import annotations

from typing import Callable, Optional


ClassifyFn = Callable[[str], Optional[str]]


DescribeImageFn = Callable[[bytes, str], Optional[str]]


EmbedFn = Callable[[list[str]], list[list[float]]]


RerankFn = Callable[[str, list[str]], list[float]]


# Called with a human-readable message as the sole positional argument. A call
# site with an exact numerator/denominator additionally passes phase/done/total/
# unit as keywords; any sink such a site can reach must accept and ignore them
# (**_).
ProgressFn = Callable[..., None]


class EmbedTextRefused(RuntimeError):
    """Raised by an ``EmbedFn`` when the embedding model cannot take one of the
    texts. Other texts are unaffected."""


class EmbeddingLossRefused(ValueError):
    """A forced re-index would have removed the embeddings of a document it
    could not re-embed. Nothing was saved: the collection is as it was."""


def count_noun(n: int, noun: str) -> str:
    """``"1 chunk"`` / ``"2 chunks"``: the count with the noun in the right number."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"
