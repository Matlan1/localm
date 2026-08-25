# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared CLI domain-error-to-exit translation.

Every CLI-surfaced feature independently wrote "run a domain call, catch
KeyError/ValueError, print red text, sys.exit(1)" (plugin commands, key
commands, doctor probes, rag commands) - part of the same duplication shape
``localm/inference/errors.py`` closes for HTTP routes (see
PATHFINDER-2026-07-11/03-unified-proposal.md section 1.5). ``run_or_die``
covers the exception-to-exit-code shape; ``_run_probe_subprocess`` and
``_report_add_paths_result`` are two structurally different shapes (a
subprocess probe, and a partial-failure result list) that do NOT get forced
into ``run_or_die``'s shape.
"""

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
    """Print a note when ``LOCALM_API_KEY`` (auth.ENV_VAR) is set, since it
    overrides the on-disk key while set. *message* is the action-specific tail
    describing what that means right now (CLI-4: three ``key_*`` commands
    repeated the same env-var check with near-identical, sometimes
    contextually different, wording)."""
    import os

    from localm import auth
    env_key = os.environ.get(auth.ENV_VAR)
    if env_key and env_key.strip():
        console.print(f"[yellow]Note:[/yellow] {auth.ENV_VAR} {message}")


def _run_probe_subprocess(code: str, prefix: str) -> dict | None:
    """Run *code* in a fresh subprocess (isolates a native-library-touching
    probe so a broken DLL/lib can never crash the caller) and parse the one
    stdout line starting with *prefix* as JSON, e.g. ``"GPU_PROBE:{...}"``.
    Returns None on any failure (timeout, crash, no matching line).

    The implementation moved to ``localm.diagnostics`` when the active doctor
    probes did (that module must not import click or rich, so it cannot import
    this one). Kept here as the CLI's name for it rather than deleted: the GPU
    device probe still calls it from doctor, and one copy of these ten lines is
    the whole point of the move."""
    from localm.diagnostics import run_probe_subprocess

    return run_probe_subprocess(code, prefix)


def _report_add_paths_result(result: dict) -> None:
    """Print each ``Collection.add_paths()`` failure and exit(1) if anything
    failed. Shared by ``rag add``/``rag repair``, whose summary lines differ
    but whose failure reporting is identical."""
    from rich.markup import escape

    for f in result["failed"]:
        console.print(f"  [yellow]failed:[/yellow] {escape(f['path'])}: {escape(f['error'])}")
    if result["failed"]:
        sys.exit(1)
