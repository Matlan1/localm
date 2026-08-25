# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm plugin system."""

from __future__ import annotations


class Plugin:
    """Minimal interface every plugin should satisfy."""

    #: Short identifier used in the CLI (``localm <name>``).
    name: str = ""

    #: Human-readable description shown in ``localm --help``.
    description: str = ""

    #: pip extras key that enables this plugin (e.g. ``"coder"``).
    extras_key: str = ""
