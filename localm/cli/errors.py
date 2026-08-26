# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared CLI domain-error-to-exit translation."""

from __future__ import annotations

import sys

from ._core import console


def run_or_die(fn, *args, missing_msg=None, **kwargs):
    """Call ``fn(*args, **kwargs)``; on ``KeyError`` (unknown name) print
    *missing_msg* (or a generic fallback) in red and exit(1); on ``ValueError``
    (a domain-rule violation) print the exception in red and exit(1). Returns
    ``fn``'s result on success.

    *missing_msg* and the ``ValueError`` text are both caller/exception-
    supplied and not restricted to a safe character class here (e.g.
    plugins.py builds ``missing_msg`` from a user-typed plugin name) - escape
    both, the same defense-in-depth this shared helper's other reporting
    functions use."""
    from rich.markup import escape

    try:
        return fn(*args, **kwargs)
    except KeyError:
        console.print(f"[red]{escape(missing_msg or 'Not found')}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        sys.exit(1)


def _note_env_override(message: str) -> None:
    """Print a note when ``LOCALM_API_KEY`` (auth.ENV_VAR) is set and non-blank.
    While set it overrides the on-disk key. *message* is the action-specific
    tail describing what that means for the command being run."""
    import os

    from localm import auth
    env_key = os.environ.get(auth.ENV_VAR)
    if env_key and env_key.strip():
        console.print(f"[yellow]Note:[/yellow] {auth.ENV_VAR} {message}")


def _run_probe_subprocess(code: str, prefix: str) -> dict | None:
    """Run *code* in a fresh subprocess, isolating a native-library-touching
    probe from the caller, and parse the one stdout line starting with *prefix*
    as JSON, e.g. ``"GPU_PROBE:{...}"``. Returns None on any failure (timeout,
    crash, no matching line). Delegates to ``localm.diagnostics``."""
    from localm.diagnostics import run_probe_subprocess

    return run_probe_subprocess(code, prefix)


def _report_add_paths_result(result: dict) -> None:
    """Print each ``Collection.add_paths()`` failure and exit(1) if anything
    failed."""
    from rich.markup import escape

    for f in result["failed"]:
        console.print(f"  [yellow]failed:[/yellow] {escape(f['path'])}: {escape(f['error'])}")
    if result["failed"]:
        sys.exit(1)
