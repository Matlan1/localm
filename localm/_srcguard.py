# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refuse to run one localm source checkout while standing in another one.

THE TRAP THIS CLOSES, measured repeatedly on a box that carries several checkouts
(a main clone, plus git worktrees, plus a separate offload clone):

A console script (``localm``, ``localcoder``) and a script run by path both set
``sys.path[0]`` to the SCRIPT's directory, never the working directory. Only
``python -c`` and ``python -m`` insert cwd. So with an editable install, whose
``.pth`` names the single checkout it was installed from, this happens:

    cd <a different checkout>
    localm gui                  # serves the INSTALLED checkout's code, silently

Nothing announces it. The caller edits one tree and exercises another, then
records an observation about code they did not write. It fails in BOTH
directions, which is what makes it expensive:

  * a correct change looks broken, because the running code does not contain it;
  * an unapplied change looks verified, because the other tree already behaved
    that way. That one is silent, and can be believed for a long time.

WHAT COUNTS AS A SOURCE CHECKOUT: a directory holding both ``pyproject.toml``
and ``localm/__init__.py``. An ordinary installation lives in ``site-packages``,
which has no ``pyproject.toml``, so :func:`foreign_source` returns ``None`` for
every normal user and this module is inert. It can only ever speak when two
DEVELOPMENT checkouts are in play, which is exactly the trap and nothing else.

TWO STRENGTHS, deliberately different:

  * :func:`require_own_source` REFUSES and exits. Wired into the console-script
    and ``python -m`` entry points, where a wrong-source launch has no legitimate
    reading and the caller is about to spend real time on a false observation.
  * :func:`warn_if_foreign_source` only reports, on stderr. Wired into package
    import, because that is the ONLY point which also covers a throwaway repro
    script (the case that has cost the most time, since such a script is written
    precisely to decide whether a fix works). An import must never be able to
    take a process down, so this one never exits.

The escape hatch is deliberate, and deliberately awkward: set
``LOCALM_ALLOW_FOREIGN_SRC=1`` to run another checkout on purpose (an A/B
comparison against the installed tree is a real thing to want). It has to be an
explicit act, so it cannot be reached by accident.
"""

from __future__ import annotations

import os
import sys

#: Set to 1/true/yes/on to run a checkout other than the one cwd sits in.
ENV_ALLOW = "LOCALM_ALLOW_FOREIGN_SRC"

#: Depth bound on the walk from cwd toward the filesystem root. A checkout root
#: is never this deep, and a bound means a pathological path cannot hang an
#: import. The loop also stops when the parent stops changing, so this is only a
#: backstop.
_MAX_WALK = 64

_TRUE = ("1", "true", "yes", "on")


def _allowed() -> bool:
    return os.environ.get(ENV_ALLOW, "").strip().lower() in _TRUE


def _key(path: str) -> str:
    """Comparison form of a path. Windows paths are case-insensitive, and one
    directory reaches us spelled ``D:\\x``, ``D:/x`` or (under MSYS) ``/d/x``; a
    worktree may also sit behind a junction. normcase+realpath collapses all of
    that, so two spellings of one checkout never read as two different ones."""
    return os.path.normcase(os.path.realpath(path))


def _is_checkout_root(directory: str) -> bool:
    """True when `directory` is the root of a localm SOURCE checkout.

    Both markers are required. ``pyproject.toml`` alone matches any Python
    project the caller happens to stand in; ``localm/__init__.py`` alone matches
    an installed ``site-packages``. Only together do they mean "a localm tree
    that somebody could be editing"."""
    return (os.path.isfile(os.path.join(directory, "pyproject.toml"))
            and os.path.isfile(os.path.join(directory, "localm", "__init__.py")))


def _standing_root(start: str) -> str | None:
    """The nearest checkout root at or above `start`, or None.

    Walks upward, so running from a subdirectory of a checkout (``docs/``,
    ``tests/``, a nested tool directory) still counts as standing in it."""
    current = os.path.abspath(start)
    for _ in range(_MAX_WALK):
        if _is_checkout_root(current):
            return current
        parent = os.path.dirname(current)
        if parent == current:               # reached the filesystem root
            return None
        current = parent
    return None


def _running_root() -> str | None:
    """The checkout root the IMPORTED localm belongs to, or None when it is an
    ordinary installation rather than a source tree."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(package_dir)
    return root if _is_checkout_root(root) else None


def foreign_source(cwd: str | None = None) -> tuple[str, str] | None:
    """``(running_root, standing_root)`` when the imported localm belongs to a
    DIFFERENT source checkout than `cwd` sits in. ``None`` otherwise.

    ``None`` is returned, and no guard fires, whenever any of these hold, which
    between them cover every legitimate launch:

      * the escape hatch is set;
      * localm is an ordinary installation, not a source tree (every user);
      * cwd is not inside any checkout (a temp dir, a home directory, a service
        working directory);
      * cwd is inside the SAME checkout that is running, however it is spelled.
    """
    if _allowed():
        return None
    running = _running_root()
    if running is None:
        return None
    try:
        here = os.getcwd() if cwd is None else cwd
    except OSError:                         # cwd deleted out from under us
        return None
    standing = _standing_root(here)
    if standing is None:
        return None
    if _key(running) == _key(standing):
        return None
    return (running, standing)


def _show(path: str) -> str:
    return path.replace("\\", "/")


def _fix_command(standing: str) -> str:
    """The same command, corrected. Handed back verbatim so the caller re-runs it
    instead of reconstructing it and getting the quoting wrong."""
    parts = [sys.argv[0] or "localm"] + list(sys.argv[1:])
    rendered = " ".join('"' + p + '"' if " " in p else p for p in parts)
    return 'PYTHONPATH="' + _show(standing) + '" ' + _show(rendered)


def _report(running: str, standing: str) -> str:
    return (
        "localm is running code from a DIFFERENT source checkout than the one "
        "you are standing in.\n"
        "\n"
        "  running code from : " + _show(running) + "\n"
        "  your directory is : " + _show(standing) + "\n"
        "\n"
        "Console scripts, and scripts run by path, put the SCRIPT's directory on "
        "sys.path and never the current directory, so an editable install "
        "resolves to whichever checkout it was installed from. Anything you "
        "observe now describes that other checkout, not the tree you are "
        "editing.\n"
        "\n"
        "Re-run with the checkout you are in on PYTHONPATH:\n"
        "\n"
        "  " + _fix_command(standing) + "\n"
        "\n"
        "(PowerShell: set $env:PYTHONPATH=\"" + _show(standing) + "\" first.)\n"
        "\n"
        "To run the other checkout on purpose, set " + ENV_ALLOW + "=1."
    )


def require_own_source() -> None:
    """Exit when the running checkout is not the one cwd sits in.

    Called from the console-script and ``python -m`` entry points. Exits with a
    non-zero status so a shell script or CI step cannot carry on past it: the
    whole defect is that the wrong code runs to completion and looks fine."""
    found = foreign_source()
    if found is None:
        return
    sys.exit("Error: " + _report(*found))


def warn_if_foreign_source() -> None:
    """Report the mismatch on stderr and return. Never exits, never raises.

    Called at package import, the only point that also covers a throwaway script
    run by path. stderr and never stdout, because localm has child processes
    whose stdout is a machine-read protocol."""
    try:
        found = foreign_source()
        if found is None:
            return
        sys.stderr.write("WARNING: " + _report(*found) + "\n")
        sys.stderr.flush()
    except Exception:                                       # noqa: BLE001
        # A guard defect must never break an import, and there is nothing to
        # report to here: this runs before logging is configured, and the caller
        # may be a child process whose stdout is a protocol. The entry-point
        # guard still runs for any real launch, so a failure here degrades to
        # "no warning", never to a broken localm.
        pass
