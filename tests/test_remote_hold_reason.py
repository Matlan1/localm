# SPDX-License-Identifier: AGPL-3.0-or-later
"""remote_hold_reason's own contract, independent of either caller.

Both localm rm and the MCP remove_model tool build a message around whatever
this function returns, and localm rm renders that message through
rich.console.Console - a real ANSI-interpreting terminal. rich.markup.escape()
only neutralizes Rich's own '[' markup syntax; it does nothing to a raw
control byte, including ESC (the ANSI escape introducer). A `reason` this
function receives from a discovered instance's `/v1/models/{id}/hold`
response is embedded into the returned string unescaped by anything upstream
of this function, so this is the one place that can make every caller safe
at once.
"""

from __future__ import annotations

import localm.selfclient as sc

ROW = {"scheme": "http", "host": "127.0.0.1", "port": 8123,
       "instance_id": "abc", "pid": 4242, "token": "t", "alive": True}


def _servers(monkeypatch, rows):
    from localm import instances
    monkeypatch.setattr(instances, "snapshot", lambda *a, **kw: rows)


def _reader(monkeypatch, result):
    def fake(scheme, port, model, token=None, bind_host=None):
        return result
    monkeypatch.setattr(sc, "read_model_file_hold", fake)


def test_strips_control_characters_from_a_remote_reported_reason(monkeypatch):
    _servers(monkeypatch, [dict(ROW)])
    hostile = "loaded\x1b[31;1mFAKE PROMPT\x1b[0m\x07and locked"
    _reader(monkeypatch, ("ok", {"held": True, "key": "served",
                                 "reason": hostile}))

    result = sc.remote_hold_reason("victim")

    assert result is not None
    assert "\x1b" not in result, result
    assert "\x07" not in result, result
    # The wording survives; only the control bytes are gone.
    assert "FAKE PROMPT" in result
    assert "and locked" in result


def test_a_non_string_reason_cannot_smuggle_control_characters(monkeypatch):
    """payload["reason"] comes from parsed JSON, so nothing upstream of this
    function guarantees it is a string. A list/dict wrapping a hostile string
    must not resurface it unescaped either."""
    _servers(monkeypatch, [dict(ROW)])
    _reader(monkeypatch, ("ok", {"held": True, "key": "served",
                                 "reason": ["\x1b[31mFAKE"]}))

    result = sc.remote_hold_reason("victim")

    assert result is not None
    assert "\x1b" not in result, result


def test_an_ordinary_reason_is_unaffected(monkeypatch):
    """The three real reasons this function has ever been asked to render
    contain no control characters, so the fix must be invisible to them."""
    _servers(monkeypatch, [dict(ROW)])
    _reader(monkeypatch, ("ok", {"held": True, "key": "victim",
                                 "reason": "its model file path could not "
                                           "be resolved"}))

    result = sc.remote_hold_reason("victim")

    assert result == (
        "the localm server at http://127.0.0.1:8123 has 'victim' loaded "
        "and its model file path could not be resolved, so it cannot be "
        "ruled out as holding this file")
