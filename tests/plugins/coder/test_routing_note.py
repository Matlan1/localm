# SPDX-License-Identifier: AGPL-3.0-or-later
"""A coder session says when the server kept its model because a model that
could have answered was skipped or failed to load.

The note comes from the server's ``X-Localm-Model-Routing`` header. The HTTP
backend keeps it in ``routing_note`` and hands each new one to
``on_routing_note``; the session shows it in its feed and reports it in
``info()``."""

from __future__ import annotations

import json
from pathlib import Path

from localm.plugins.coder.backends.http import HTTPBackend
from localm.plugins.coder.sessions import CoderSession

NOTE = ("kept plain (tool_use=absent); big was skipped because its last load "
        "failed at 14:08 (the native model-loading process crashed); it is tried "
        "again after 14:18 or when its load settings change")


def _headers(**fields):
    body = {"resolved": "plain", "requested": "plain", "routed": False,
            "pinned": False, "gaps": {"tool_use": "absent"},
            "unmet": ["tool_use"], **fields}
    return {"X-Localm-Model-Routing": json.dumps(body)}


def _backend():
    return HTTPBackend("http://127.0.0.1:1/v1", model="plain", api_key="k",
                       localm_server=True, model_pinned=False,
                       required_capabilities=("tool_use",))


class TestBackendNote:
    def test_a_kept_model_with_a_note_is_recorded_and_announced(self):
        be, seen = _backend(), []
        be.on_routing_note = seen.append
        be._note_routing(_headers(note=NOTE))
        assert be.routing_note == NOTE
        assert seen == [NOTE]

    def test_the_same_note_is_announced_once(self):
        be, seen = _backend(), []
        be.on_routing_note = seen.append
        for _ in range(3):
            be._note_routing(_headers(note=NOTE))
        assert seen == [NOTE]

    def test_a_different_note_is_announced_again(self):
        be, seen = _backend(), []
        be.on_routing_note = seen.append
        be._note_routing(_headers(note=NOTE))
        be._note_routing(_headers(note=NOTE + " (again)"))
        assert len(seen) == 2

    def test_a_reply_with_no_note_clears_it_and_a_returning_note_is_announced(self):
        be, seen = _backend(), []
        be.on_routing_note = seen.append
        be._note_routing(_headers(note=NOTE))
        be._note_routing({})
        assert be.routing_note is None
        be._note_routing(_headers(note=NOTE))
        assert seen == [NOTE, NOTE]

    def test_a_routed_reply_carries_no_note(self):
        be, seen = _backend(), []
        be.on_routing_note = seen.append
        be._note_routing(_headers(routed=True, resolved="big", note=NOTE))
        assert be.routing_note is None and seen == []

    def test_a_pinned_reply_carries_no_note(self):
        be = _backend()
        be._note_routing(_headers(pinned=True, note=NOTE))
        assert be.routing_note is None

    def test_a_header_that_is_not_json_is_ignored(self):
        be = _backend()
        be._note_routing({"X-Localm-Model-Routing": "{not json"})
        assert be.routing_note is None

    def test_a_response_without_headers_is_ignored(self):
        be = _backend()
        be._note_routing(None)
        assert be.routing_note is None

    def test_without_a_listener_the_note_is_still_recorded(self):
        be = _backend()
        be._note_routing(_headers(note=NOTE))
        assert be.routing_note == NOTE


class _Backend:
    model_id = "plain"
    native_tools = False
    routing_note = None
    on_routing_note = None

    def set_tools(self, defs):
        pass


def _session(tmp_path: Path, backend) -> CoderSession:
    return CoderSession(tmp_path, backend, mode="privacy", auto_verify=False)


class TestSessionShowsTheNote:
    def test_a_note_from_the_backend_reaches_the_session_feed(self, tmp_path):
        backend = _Backend()
        s = _session(tmp_path, backend)
        backend.on_routing_note(NOTE)
        assert {"type": "info", "text": NOTE} in s.history

    def test_info_reports_the_current_note(self, tmp_path):
        backend = _Backend()
        s = _session(tmp_path, backend)
        assert s.info()["routing_note"] is None
        backend.routing_note = NOTE
        assert s.info()["routing_note"] == NOTE

    def test_a_backend_that_reports_no_notes_is_left_alone(self, tmp_path):
        class Plain:
            model_id = "x"
            native_tools = False

            def set_tools(self, defs):
                pass

        backend = Plain()
        s = _session(tmp_path, backend)
        assert not hasattr(backend, "on_routing_note")
        assert s.info()["routing_note"] is None
