# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Privacy gate for OFF-MACHINE models (the one place that decides whether a
surface's own content may be sent to a model that is not on this machine).

``remote_allowed(surface)`` is the single decision point callers use before
building a non-local backend. It is gated on the effective session mode and is
OFF in privacy mode, which is the default.

A refusal is a refusal: a caller must never silently substitute the local model
for the one the user picked, and never report success.

The coder's HTTP backend constructors call :func:`require_off_machine_allowed`
themselves, so a surface that builds a backend to a non-loopback address is
gated without having to remember to ask.
"""

from __future__ import annotations

from typing import Optional


class OffMachineModelRefused(PermissionError):
    """An off-machine model was refused by the session mode."""


def remote_allowed(surface: str) -> bool:
    """True when *surface* ("chat" | "coder" | "server") may talk to a model
    that is not on this machine. False in privacy mode (the default).

    Resolved fresh each call so a mid-run config/env change is honoured.
    """
    from localm.audit import SessionMode, effective_mode
    return effective_mode(surface) != SessionMode.PRIVACY


def remote_allowed_for_mode(session_mode: str) -> bool:
    """The same decision for a session whose mode is ALREADY RESOLVED.

    A coder session's mode is not always the ambient one: an explicit ``mode``
    on the create request, and a per-project ``.localcoder/config.toml``, both
    override it (see ``audit.effective_mode``'s precedence). Answers against the
    mode the caller has already established rather than re-deriving it from the
    ambient surface.
    """
    from localm.audit import SessionMode
    return session_mode != SessionMode.PRIVACY.value


def refusal_message(what: str) -> str:
    """The refusal a caller raises. One wording for every call site: it says
    what is off, why, and names the setting that turns it on.

    *what* is a PLURAL noun phrase ("Off-machine models"). The template supplies
    the verb and the pronoun.
    """
    return (f"{what} are off in privacy mode (nothing leaves this machine). "
            "Set mode/coder_mode to 'log' or 'full' to enable them.")


def require_off_machine_allowed(base_url: str, *, session_mode: Optional[str] = None,
                                surface: str = "coder",
                                user_typed: bool = False) -> None:
    """Refuse to build a backend for *base_url* when it is off this machine and
    the session may not use one.

    A loopback host returns silently. Otherwise: in privacy mode raises
    :class:`OffMachineModelRefused` unless *user_typed* (the user typed the flag
    that names this destination; an environment variable never counts), then
    runs ``netpolicy.check_url`` (raises ``NetworkPolicyError``).

    *session_mode* is the already-resolved mode of the session ("privacy" |
    "log" | "full"); None resolves the ambient mode of *surface*. A URL whose
    shape is refused raises ``NetworkPolicyError`` before it is classified.
    """
    from localm.bindhost import is_loopback_host
    from localm.netpolicy import check_url, check_url_shape
    host = check_url_shape(base_url)
    if is_loopback_host(host):
        return
    allowed = (remote_allowed_for_mode(session_mode) if session_mode is not None
               else remote_allowed(surface))
    if not allowed and not user_typed:
        raise OffMachineModelRefused(refusal_message("Off-machine models"))
    check_url(base_url)
