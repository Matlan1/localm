# SPDX-License-Identifier: AGPL-3.0-or-later
"""The browser tools are OFF unless the session was granted them.

The ADR's posture is that the capability grants eligibility and a separate
switch grants use, so the default for every session is off. Two independent
things enforce that and both are pinned here:

  1. the tools are removed from ``disabled_tools``' complement, so the model is
     never even OFFERED them;
  2. each tool function re-checks the flag, so a future refactor of (1) cannot
     silently reopen the browser.

A restricted session never browses, whatever the flag says.
"""

from pathlib import Path
from unittest.mock import patch

from localm.audit import SessionMode
from localm.plugins.coder.agent.constants import _BROWSER_TOOLS


class _Stub:
    model_id = "m"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def chat(self, messages, **kw):
        return "Done."

    def chat_stream(self, messages, **kw):
        yield "Done."


def _agent(cwd, **kw):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Stub(), cwd=cwd, self_verify=False, **kw)


class TestTheModelIsNotOfferedThem:
    def test_every_browser_tool_is_disabled_by_default(self, tmp_path):
        a = _agent(tmp_path)
        assert a.browser_enabled is False
        for name in _BROWSER_TOOLS:
            assert name in a.disabled_tools, name

    def test_granting_the_capability_leaves_them_enabled(self, tmp_path):
        a = _agent(tmp_path, browser_enabled=True)
        for name in _BROWSER_TOOLS:
            assert name not in a.disabled_tools, name

    def test_a_restricted_session_never_browses(self, tmp_path):
        a = _agent(tmp_path, restricted=True, browser_enabled=True)
        for name in _BROWSER_TOOLS:
            assert name in a.disabled_tools, name

    def test_they_are_absent_from_the_schema_the_model_sees(self, tmp_path):
        """The real path: a native-tools backend is handed the tool defs at
        construction, so read what it was actually given."""
        class _NativeStub(_Stub):
            native_tools = True

            def __init__(self):
                self.given = None

            def set_tools(self, defs):
                self.given = [d.get("function", {}).get("name") for d in defs]

        def build(**kw):
            from localm.plugins.coder.agent import Agent
            backend = _NativeStub()
            with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
                 patch("localm.plugins.coder.agent.make_audit_log"), \
                 patch("localm.plugins.coder.agent.load_memory", return_value=""):
                PM.build.return_value.file_count.return_value = 0
                PM.build.return_value.truncated = False
                Agent(backend, cwd=tmp_path, self_verify=False,
                      mode=SessionMode.LOG, auto_approve=True, **kw)
            return set(backend.given or [])

        names_off = build()
        names_on = build(browser_enabled=True)
        assert names_off, "the backend was handed no tools at all"
        assert not (names_off & _BROWSER_TOOLS), sorted(names_off & _BROWSER_TOOLS)
        assert _BROWSER_TOOLS <= names_on, sorted(_BROWSER_TOOLS - names_on)


class TestEachToolRechecks:
    """Defence in depth: the functions refuse on their own, so the narrowing
    above is not the only thing standing between a model and a browser."""

    def _sessions(self):
        class S:
            job_owner = "test-owner"
            browser_enabled = False
        return S()

    def test_every_tool_refuses_when_the_flag_is_off(self, tmp_path):
        from localm.plugins.coder.tools import browser as bt
        session = self._sessions()
        calls = [
            (bt.tool_browser_navigate, {"url": "https://example.com/"}),
            (bt.tool_browser_read, {}),
            (bt.tool_browser_click, {"selector": "a"}),
            (bt.tool_browser_fill, {"selector": "input", "value": "x"}),
            (bt.tool_browser_screenshot, {}),
            (bt.tool_browser_console, {}),
            (bt.tool_browser_network, {}),
            (bt.tool_browser_close, {}),
        ]
        for fn, kwargs in calls:
            res = fn(Path(tmp_path), _session=session, **kwargs)
            assert res.ok is False, fn.__name__
            assert "not enabled" in res.output, (fn.__name__, res.output)

    def test_navigate_does_not_start_a_browser_when_refused(self, tmp_path):
        from localm.plugins.coder.tools import browser as bt
        from localm.browser import session as bsession
        before = set(bsession.active_ids())
        res = bt.tool_browser_navigate(Path(tmp_path), url="https://example.com/",
                                       _session=self._sessions())
        assert res.ok is False
        assert set(bsession.active_ids()) == before, "a browser was started anyway"


class TestNotInTheRestrictedAllowlist:
    def test_no_browser_tool_is_safe_for_a_restricted_session(self):
        from localm.plugins.coder.tools.registry import SAFE_RESTRICTED_TOOLS
        assert not (_BROWSER_TOOLS & SAFE_RESTRICTED_TOOLS)


class TestPageContentIsUntrusted:
    def test_content_returning_tools_are_marked_untrusted(self):
        from localm.plugins.coder.provenance import is_untrusted_tool
        from localm.plugins.coder.tools.registry import TOOL_REGISTRY
        for name in ("browser_navigate", "browser_read", "browser_click",
                     "browser_fill", "browser_console", "browser_network"):
            assert is_untrusted_tool(name, TOOL_REGISTRY[name]), name
