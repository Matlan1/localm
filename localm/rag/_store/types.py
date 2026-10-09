# SPDX-License-Identifier: AGPL-3.0-or-later
"""Callback types taken by ``Collection``'s indexing and search methods."""

from __future__ import annotations

from typing import Callable, Optional


ClassifyFn = Callable[[str], Optional[str]]


DescribeImageFn = Callable[[bytes, str], Optional[str]]


EmbedFn = Callable[[list[str]], list[list[float]]]


# Called with a human-readable message as the sole positional argument. A call
# site with an exact numerator/denominator additionally passes phase/done/total/
# unit as keywords; any sink such a site can reach must accept and ignore them
# (**_).
ProgressFn = Callable[..., None]
