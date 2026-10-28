# SPDX-License-Identifier: AGPL-3.0-or-later
"""Console output and the stdin helpers shared by ``localm setup-llama``.
"""

from __future__ import annotations

import json
import os
import sys

from localm.console import SafeConsole as Console
from localm.setup_spec import ASK_ENV, ASK_PREFIX

console = Console(highlight=False)


def _net_allowed() -> bool:
    """Whether the network policy lets setup-llama download: False when
    ``net_mode`` is ``off`` and ``net_allow_model_downloads`` does not exempt
    downloads. Reads the policy only."""
    from localm import netpolicy
    return not (netpolicy.network_mode() == "off"
                and not netpolicy.downloads_allowed_when_off())


def _require_network_allowed() -> None:
    """Raise :class:`localm.netpolicy.NetworkPolicyError` when :func:`_net_allowed`
    is False. Reads the policy only; performs no network or DNS access."""
    from localm import netpolicy
    if not _net_allowed():
        raise netpolicy.NetworkPolicyError(
            "setup-llama downloads the llama.cpp runtime, and network access is "
            "disabled (net_mode=off). Enable it with:  localm config net_mode ask  "
            "or provision from a local build:  localm setup-llama --from <build-dir>",
            off=True)


def _flush_stdin() -> None:
    """Discard any input the OS/terminal buffered while we were NOT actually
    waiting on it (e.g. a stray Enter pressed while a driver probe or a
    multi-hundred-MB download was running). Without this, that buffered
    keystroke is silently consumed the instant the NEXT ``click.confirm()``
    prompt appears - answering a question the user never actually read, rather
    than the one they meant to answer (or none at all). Call this immediately
    before every interactive prompt in the setup flow.

    Best-effort and silent on failure: a piped/non-tty stdin (tests, CI, a
    non-interactive install) has nothing to flush and isatty() already guards
    that; any other failure just leaves stray input in place - the pre-fix
    behaviour - which is not a regression, so there is nothing worth surfacing."""
    if not sys.stdin.isatty():
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        else:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass




def _asks_over_stdio() -> bool:
    """Whether questions go over stdio: ``LOCALM_SETUP_ASK=stdio``, set by the
    graphical installer, which shows each question in its window."""
    return os.environ.get(ASK_ENV, "").strip().lower() == "stdio"


def _interactive() -> bool:
    """Whether a question can be answered: stdin is a terminal, or questions go
    over stdio."""
    if _asks_over_stdio():
        return True
    return bool(sys.stdin) and sys.stdin.isatty()


def _ask_stdio(question: dict) -> str:
    """Write *question* as one ``ASK_PREFIX`` + JSON line to stdout and return
    the answer line read from stdin, stripped; empty at end of input."""
    sys.stdout.write(ASK_PREFIX + json.dumps(question) + "\n")
    sys.stdout.flush()
    line = sys.stdin.readline() if sys.stdin else ""
    return (line or "").strip()


def _confirm(text: str, default: bool) -> bool:
    """A yes/no question: in the window over stdio, else ``click.confirm``.
    Over stdio, an empty or unrecognised answer is *default*."""
    if not _asks_over_stdio():
        import click
        return click.confirm(text, default=default)
    answer = _ask_stdio({"kind": "confirm", "text": text.strip(),
                         "default": "y" if default else "n"}).lower()
    if answer[:1] in ("y", "n"):
        return answer[:1] == "y"
    return default


def _choose(text: str, choices: list, default: str) -> str:
    """A pick from *choices*, ``(answer, label)`` pairs: in the window over
    stdio, else ``click.prompt``. Returns an answer from *choices*; over stdio
    an empty or unknown answer is *default*."""
    keys = [k for k, _ in choices]
    if not _asks_over_stdio():
        import click
        return click.prompt(text, type=click.Choice(keys), default=default)
    answer = _ask_stdio({"kind": "choose", "text": text.strip(),
                         "choices": [list(c) for c in choices], "default": default})
    return answer if answer in keys else default
