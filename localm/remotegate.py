# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Privacy gate for OFF-MACHINE models (the one place that decides whether a
surface's own content may be sent to a model that is not on this machine).

localm is offline-first. Choosing a cloud or remote model is not a wiring
change, it is a trust-boundary change: the user's prompts, and whatever file
contents the surface reads on their behalf, leave the machine. So it is gated on
the effective session mode, and it is OFF in privacy mode, which is the default.

``remote_allowed(surface)`` is the single decision point callers use before
building a non-local backend. A refusal must be a refusal: never silently
substitute the local model for the one the user picked (that is the
never-override-an-explicit-choice rule), and never report success (AGENTS.md
rule 5).

This mirrors ``localm/memory/gating.py`` deliberately, down to the leaf-module
shape, and for the same three reasons: callers share ONE decision point instead
of each re-deriving it, the module is trivially unit-testable, and the subsystems
that use it stay free of the audit import. localm already answers this exact
question this way in two other places, so a third answer would be drift rather
than nuance:

  localm/memory/gating.py        memory is fully off in privacy mode
  localm/plugins/coder/reviewer.py   a network reviewer is skipped in privacy mode

ONE DIFFERENCE FROM reviewer.py IS DELIBERATE AND IS NOT DRIFT. The reviewer is
an OPTIONAL EXTRA PASS, so when it is refused it falls back to reviewing with the
local model: the user still gets a review, just a different one. A session's LLM
backend is not optional - it IS the session - so falling back would hand the user
a different model than the one they chose, silently. Callers of this gate must
therefore REFUSE, not degrade.
"""

from __future__ import annotations


def remote_allowed(surface: str) -> bool:
    """True when *surface* ("chat" | "coder" | "server") may talk to a model
    that is not on this machine. False in privacy mode (the default).

    Resolved fresh each call so a mid-run config/env change is honoured.
    """
    from localm.audit import SessionMode, effective_mode
    return effective_mode(surface) != SessionMode.PRIVACY


def remote_allowed_for_mode(session_mode: str) -> bool:
    """The same decision for a session whose mode is ALREADY RESOLVED.

    A coder session's mode is not always the ambient one: an explicit
    ``mode`` on the create request, and a per-project
    ``.localcoder/config.toml``, both override it (see
    ``audit.effective_mode``'s precedence). Re-resolving from the ambient
    surface here would answer a question about a DIFFERENT session than the one
    being created, and would do it in the permissive direction whenever the
    ambient mode is looser than the session's own.

    Same decision, same rule, expressed against the mode the caller has already
    established rather than re-deriving it.
    """
    from localm.audit import SessionMode
    return session_mode != SessionMode.PRIVACY.value


def refusal_message(what: str) -> str:
    """The refusal a caller raises, worded like memory's.

    ONE wording for every call site, on purpose: memory's own plug records that
    letting refusal messages drift between sites was treated as a defect worth
    fixing, and picked the most helpful variant for all five. This says what is
    off, why, and NAMES THE SETTING that turns it on, because a refusal that
    does not tell the user what to change is a dead end.

    *what* is a PLURAL noun phrase ("Off-machine models"), matching memory's
    "Memory writes are off in privacy mode ...". The template owns the verb and
    the pronoun, so there is no per-caller agreement to get wrong - the first
    draft took a singular subject and shipped "Off-machine models is off" to a
    live server, which the unit test could not see because it asserted on
    fragments rather than on the whole sentence.
    """
    return (f"{what} are off in privacy mode (nothing leaves this machine). "
            "Set mode/coder_mode to 'log' or 'full' to enable them.")
