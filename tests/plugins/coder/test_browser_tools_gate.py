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


class TestTheRouteDecidesFromScopeAndSetting:
    """Both must hold: the capability on the key, and the setting switched on."""

    def _helper(self):
        from localm.plugins.builtin.coder.plug import _browser_enabled_for
        return _browser_enabled_for

    def test_owner_still_needs_the_setting_on(self, monkeypatch):
        fn = self._helper()
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"browser_enabled": False})
        assert fn(True, set()) is False
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"browser_enabled": True})
        assert fn(True, set()) is True

    def test_a_key_without_the_capability_never_browses(self, monkeypatch):
        from localm import scopes as S
        fn = self._helper()
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"browser_enabled": True})
        assert fn(False, {S.CODER}) is False
        assert fn(False, {S.CODER_FULL}) is False, \
            "shell access must not imply browser access"
        assert fn(False, {S.BROWSER}) is True
        assert fn(False, {S.ADMIN}) is True

    def test_an_unreadable_config_refuses(self, monkeypatch):
        fn = self._helper()

        def boom():
            raise RuntimeError("config gone")
        monkeypatch.setattr("localm.config.load_config", boom)
        assert fn(True, set()) is False


class TestNoDeadSettings:
    def test_every_browser_setting_is_actually_read(self):
        """A setting nothing reads is a facade. Each of these has a consumer."""
        import inspect
        from localm.plugins.coder.tools import browser as bt
        from localm.plugins.builtin.coder import plug
        src = (inspect.getsource(bt) + inspect.getsource(plug._browser_enabled_for))
        for key in ("browser_enabled", "browser_headless", "browser_engine",
                    "browser_custom_domain_rules", "browser_allow", "browser_deny"):
            assert key in src, key + " is declared in the schema but read nowhere"


class TestASpawnedChildDoesNotInheritTheBrowser:
    """A child shares the parent's job_owner, which is what the browser registry
    is keyed on, so it could otherwise reach the parent's live browser. It
    cannot: the capability is not inherited, and the child re-applies its own
    gate on top of the disabled_tools it inherits.

    This pins the SAFE direction (a child gets less). Granting a child the
    browser is a deliberate decision, not something to fall into by adding one
    key to inherited_child_kwargs.
    """

    def test_the_capability_is_not_in_the_inherited_child_kwargs(self):
        import inspect
        from localm.plugins.coder.tools import agents
        src = inspect.getsource(agents.inherited_child_kwargs)
        assert "browser_enabled" not in src, (
            "a child would inherit the browser; if that is wanted it needs its "
            "own decision and this test updated deliberately")

    def test_a_child_of_a_browsing_parent_still_cannot_browse(self, tmp_path):
        parent = _agent(tmp_path, browser_enabled=True)
        for name in _BROWSER_TOOLS:
            assert name not in parent.disabled_tools, name
        child = _agent(tmp_path, parent=parent,
                       disabled_tools=parent.disabled_tools)
        assert child.browser_enabled is False
        for name in _BROWSER_TOOLS:
            assert name in child.disabled_tools, name

    def test_the_child_shares_the_parent_owner_key(self, tmp_path):
        """The premise of this class: without the gate they WOULD collide."""
        parent = _agent(tmp_path, browser_enabled=True)
        child = _agent(tmp_path, parent=parent)
        assert child.job_owner == parent.job_owner


class TestTheNetworkPolicyPromptCoversTheBrowser:
    """net_mode governs the browser tools that reach the network, the same way
    it governs fetch_url. Without this, 'ask' prompted before a one-page fetch
    and not before driving a whole browser."""

    def test_the_reaching_tools_are_network_tools(self):
        from localm.plugins.coder.agent.constants import _NETWORK_TOOLS
        for name in ("browser_navigate", "browser_click", "browser_fill"):
            assert name in _NETWORK_TOOLS, name

    def test_the_reading_tools_are_not(self):
        """They read state the session already has, so prompting for them would
        be noise rather than a decision."""
        from localm.plugins.coder.agent.constants import _NETWORK_TOOLS
        for name in ("browser_read", "browser_console", "browser_network",
                     "browser_screenshot", "browser_close"):
            assert name not in _NETWORK_TOOLS, name

    def test_net_mode_off_refuses_before_a_browser_is_launched(self, tmp_path,
                                                              monkeypatch):
        from localm.browser import session as bsession
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "off")
        a = _agent(tmp_path, browser_enabled=True)
        before = set(bsession.active_ids())
        from localm.plugins.coder.tools.registry import TOOL_REGISTRY  # noqa: F401
        from localm.plugins.coder.parser import ToolCall
        res = a._execute_tool(
            ToolCall(name="browser_navigate", args={"url": "https://example.com/"},
                     raw="", start=0, end=0), interactive=False)
        assert res.ok is False
        assert "net_mode=off" in res.output, res.output
        assert set(bsession.active_ids()) == before, "a browser was launched anyway"


class TestTheBrowserIsClosedWithTheSession:
    def test_close_for_owner_is_wired_into_session_teardown(self):
        """A browser outlives its session otherwise: a real Chromium plus its
        driver, holding that session's cookies and storage."""
        import inspect
        from localm.plugins.coder import sessions
        src = inspect.getsource(sessions.CoderSession.close)
        assert "close_for_owner" in src, (
            "nothing closes the browser when the coder session ends")
