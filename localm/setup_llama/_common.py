# SPDX-License-Identifier: AGPL-3.0-or-later
"""Console output and the stdin helper shared by ``localm setup-llama``.
"""

from __future__ import annotations

import sys

from localm.console import SafeConsole as Console

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
