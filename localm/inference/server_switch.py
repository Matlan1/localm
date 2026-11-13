# SPDX-License-Identifier: AGPL-3.0-or-later
"""Make the localm server already running for a project answer with the model
a command names: reuse it when it serves that model, otherwise switch it
through the server's own ``POST /v1/models/load``.

Every command that meets a running localm server with a model of its own
(``localm run``, ``localm gui``, ``localm serve``, ``localm coder``) calls
:func:`serve_requested_model`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

SWITCH_TIMEOUT_S = (10, 900)


@dataclass(frozen=True)
class ServedModel:
    """What :func:`serve_requested_model` decided.

    name: the name to address the server's model by, or None when the server
        does not serve the requested model.
    note: when *name* is None, the sentence saying why and what happens
        instead; None otherwise.
    not_applied: the load options given that the server's model was not
        loaded with (it keeps its own settings); empty when *name* is None.
    switched: True when the server was switched to the model.
    """

    name: Optional[str]
    note: Optional[str] = None
    not_applied: tuple = ()
    switched: bool = False


def switch_server_model(base_url: str, token: Optional[str],
                        model: str) -> tuple[Optional[str], Optional[str]]:
    """Ask the localm server at *base_url* (its ``/v1`` root) to load *model*
    and make it the active model, through ``POST /v1/models/load``. *token* is
    sent as given as the bearer credential (no Authorization header when it is
    empty or None); the caller decides which credential its command uses.

    Returns ``(name, None)`` on success, where *name* is the name the server
    reports for the model (address requests by it), or ``(None, reason)`` when
    the server did not load it: unreachable, a non-2xx answer (its ``detail``
    is quoted), a reply that is not a JSON object, or a status other than
    ``loaded``, ``already_loaded`` or ``peer_routed``. Never raises for those."""
    import requests

    from localm import tls
    from localm.jsonreply import response_json
    url = f"{base_url.rstrip('/')}/models/load"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.post(url, json={"model": model}, headers=headers,
                             timeout=SWITCH_TIMEOUT_S, allow_redirects=False,
                             verify=tls.requests_verify(url))
    except requests.RequestException as e:
        return None, f"it could not be reached: {e}"
    try:
        data = response_json(resp)
    except ValueError:
        data = None
    if resp.status_code >= 300:
        detail = data.get("detail") if isinstance(data, dict) else None
        return None, f"HTTP {resp.status_code}" + (f": {detail}" if detail else "")
    if not isinstance(data, dict):
        return None, "its reply was not valid JSON"
    status = data.get("status")
    if status not in ("loaded", "already_loaded", "peer_routed"):
        return None, f"it answered with status {status!r}"
    name = data.get("model")
    return (name if isinstance(name, str) and name else model), None


def serve_requested_model(base_url: str, token: Optional[str], model: str, *,
                          load_options: Sequence[str] = (),
                          in_process: bool = False,
                          by_name: bool = True,
                          starting: bool = False,
                          say: Optional[Callable[[str], None]] = None) -> ServedModel:
    """Make the localm server at *base_url* (its ``/v1`` root) answer with
    *model*. *token* is the bearer credential of the calling command, sent as
    given on every request this makes (the probes and the switch).

    *load_options*: the load flags the user typed (``--ctx``, ...). The
    server's load route takes only a model name, so a switch never applies
    them.
    *in_process*: the caller can load *model* itself with those options
    (``localm run``). Then a switch that would drop them is not made, and a
    failed switch ends in loading the model in the caller's process; the
    returned note says so.
    *by_name*: the caller names *model* in every request, so a server whose
    model cannot be read (``remote_model_status`` answered ``unknown``) is
    used without a switch. With False (a GUI, which asks the server for its
    active model) such a server is switched.
    *starting*: the server was just started with *model*; one that reports no
    model yet is still loading it and is not switched.
    *say*: called with a plain-text line before a switch is requested.

    Returns a :class:`ServedModel`. Its ``name`` is the server's active model
    when that is *model* by name or by file (``served_model_is_requested``),
    *model* itself in the ``by_name``/``starting`` cases above, or the name
    the switch reports. ``name`` is None when the server is not switched: the
    switch failed, or *in_process* is set and *load_options* were given; the
    ``note`` then says why and what happens instead (*in_process*: the model
    loads in this process; otherwise the server keeps what it had)."""
    from .http_engine import remote_model_status, served_model_is_requested
    load_options = tuple(load_options)
    state, active = remote_model_status(base_url, token)
    if (state == "unknown" and by_name) or (starting and state == "empty"):
        return ServedModel(model, not_applied=load_options)
    if active and served_model_is_requested(base_url, token, active, model):
        return ServedModel(active, not_applied=load_options)
    if active:
        held, kept = f"serves {active}", f"it keeps serving {active}"
    elif state == "empty":
        held, kept = "has no model loaded", "it still has no model loaded"
    else:
        held, kept = ("did not say which model it serves",
                      "it keeps the model it had")
    flags = ", ".join(load_options)
    if load_options and in_process:
        return ServedModel(None, note=(
            f"The localm server here {held}; switching it to {model} would "
            f"load it with the server's own settings instead of {flags}, so "
            f"{model} is loaded in this process."))
    line = f"The localm server here {held}; switching it to {model}"
    if load_options:
        line += f" with its own settings ({flags} not applied)"
    if say is not None:
        say(line + "...")
    name, why = switch_server_model(base_url, token, model)
    if name is None:
        then = "loading it in this process" if in_process else kept
        return ServedModel(None, note=(
            f"The localm server here could not switch to {model} ({why}); "
            f"{then}."))
    return ServedModel(name, not_applied=load_options, switched=True)
