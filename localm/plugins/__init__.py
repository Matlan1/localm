"""
localm plugin system.

A plugin is a Python package under ``localm.plugins`` that:
  - exposes a Click command (or group) registered as ``localm <plugin>``
  - is gated behind an optional extras group in pyproject.toml

Built-in plugins
----------------
coder  - offline AI coding agent (``pip install localm[coder]``)
         Invoked as: ``localm coder`` or the backward-compat alias ``localcoder``
"""

from __future__ import annotations


class Plugin:
    """
    Minimal interface every plugin should satisfy.

    Plugins are not required to subclass this - they only need to expose
    a Click command called ``cli_command``.  This base class serves as
    documentation and an optional convenience base.
    """

    #: Short identifier used in the CLI (``localm <name>``).
    name: str = ""

    #: Human-readable description shown in ``localm --help``.
    description: str = ""

    #: pip extras key that enables this plugin (e.g. ``"coder"``).
    extras_key: str = ""
