# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deferred tools and search_tools.

The system prompt documents the core tools in full and lists the deferred ones
(registry.DEFERRED_TOOLS plus every MCP and plugin tool) by name and one line;
search_tools returns a deferred tool's full block on demand and activates it on
the session.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from localm.plugins.coder import prompts
from localm.plugins.coder.provenance import is_untrusted_tool
from localm.plugins.coder.tool_registration import register_foreign_tool
from localm.plugins.coder.tools import TOOL_REGISTRY
from localm.plugins.coder.tools.registry import (
    DEFERRED_TOOLS,
    SAFE_RESTRICTED_TOOLS,
    TOOL_SEARCH_TOOL,
)
from localm.plugins.coder.tools.toolsearch import MAX_MATCHES, tool_search_tools
from localm.textguard import untrusted_spans_of

CWD = Path("proj")
DEFAULT_MODEL = "qwen2.5-coder-7b"


def _session(disabled=frozenset()):
    activated: list[str] = []
    return SimpleNamespace(disabled_tools=frozenset(disabled),
                           _activate_tools=lambda names: activated.extend(names),
                           activated=activated)


def _register_mcp(name="mcp_srv_lookup", description="Look up a record. Extra text."):
    reg, warn = [], []
    register_foreign_tool(
        name, fn=lambda cwd, **a: None, description=description,
        params={"key": {"type": "string", "description": "record key", "required": True}},
        destructive=True, source_label="MCP", registered=reg, warnings=warn)
    assert warn == []
    return name


# --------------------------------------------------------------------------- #
#  Registry                                                                    #
# --------------------------------------------------------------------------- #

def test_every_deferred_name_is_a_registered_tool():
    assert DEFERRED_TOOLS <= set(TOOL_REGISTRY)


def test_search_tools_is_registered_and_not_deferred():
    assert TOOL_SEARCH_TOOL in TOOL_REGISTRY
    assert TOOL_SEARCH_TOOL not in prompts.deferred_tool_names()


def test_search_tools_is_not_on_the_restricted_allowlist():
    assert TOOL_SEARCH_TOOL not in SAFE_RESTRICTED_TOOLS


# --------------------------------------------------------------------------- #
#  System prompt                                                              #
# --------------------------------------------------------------------------- #

def test_default_prompt_lists_deferred_tools_by_one_line_only():
    sp = str(prompts.build_system_prompt(CWD, model_name=DEFAULT_MODEL))
    for name in DEFERRED_TOOLS:
        assert f"## {name} -" not in sp, name
        assert f"\n- {name} - " in sp, name
    assert "## read_file -" in sp
    assert "## search_tools -" in sp
    assert "OTHER TOOLS:" in sp


def test_deferred_one_liner_is_the_first_sentence_without_the_example():
    sp = str(prompts.build_system_prompt(CWD, model_name=DEFAULT_MODEL))
    line = next(ln for ln in sp.splitlines() if ln.startswith("- git_commit - "))
    assert line == "- git_commit - Stage files and create a git commit"
    assert '"name": "git_commit"' not in sp


def test_disabling_search_tools_documents_every_enabled_tool_in_full():
    sp = str(prompts.build_system_prompt(
        CWD, model_name=DEFAULT_MODEL, disabled_tools=frozenset({TOOL_SEARCH_TOOL})))
    for name in DEFERRED_TOOLS:
        assert f"## {name} -" in sp, name
    assert "OTHER TOOLS:" not in sp
    assert prompts.deferred_tool_names(frozenset({TOOL_SEARCH_TOOL})) == frozenset()


def test_disabled_deferred_tool_is_not_listed_at_all():
    sp = str(prompts.build_system_prompt(
        CWD, model_name=DEFAULT_MODEL, disabled_tools=frozenset({"git_push"})))
    assert "git_push" not in sp


def test_small_family_prompt_is_unchanged_by_deferral():
    small = "phi-3-mini"
    with_search = prompts.build_system_prompt(CWD, model_name=small)
    without = prompts.build_system_prompt(
        CWD, model_name=small, disabled_tools=frozenset({TOOL_SEARCH_TOOL}))
    assert "OTHER TOOLS:" not in with_search
    assert "search_tools" not in with_search
    for name in DEFERRED_TOOLS:
        assert f"\n{name}(" in with_search, name
    assert str(with_search) == str(without)


def test_native_prompt_has_no_full_blocks_or_examples():
    sp = str(prompts.build_system_prompt(CWD, model_name=DEFAULT_MODEL, native_tools=True))
    assert "\n## " not in sp
    assert '{"name": "read_file"' not in sp
    assert "\nread_file(path, [offset], [limit]) - " in sp
    for name in DEFERRED_TOOLS:
        assert f"\n{name}(" not in sp, name
        assert f"\n- {name} - " in sp, name


def test_prompt_is_deterministic_across_builds():
    a = prompts.build_system_prompt(CWD, model_name=DEFAULT_MODEL)
    b = prompts.build_system_prompt(CWD, model_name=DEFAULT_MODEL)
    assert str(a) == str(b)
    assert untrusted_spans_of(a) == untrusted_spans_of(b)


def test_foreign_tool_is_deferred_and_its_block_leaves_the_prompt():
    with patch.dict(TOOL_REGISTRY, {}, clear=False):
        name = _register_mcp()
        assert name in prompts.deferred_tool_names()
        sp = prompts.build_system_prompt(CWD, model_name=DEFAULT_MODEL)
        assert f"## {name}" not in str(sp)
        # Not repeated in OTHER TOOLS: foreign tools have their own section.
        assert f"- {name} - " not in str(sp)
        assert not untrusted_spans_of(sp)


def test_foreign_tool_keeps_its_full_untrusted_block_when_search_is_disabled():
    with patch.dict(TOOL_REGISTRY, {}, clear=False):
        name = _register_mcp()
        sp = prompts.build_system_prompt(
            CWD, model_name=DEFAULT_MODEL, disabled_tools=frozenset({TOOL_SEARCH_TOOL}))
        spans = untrusted_spans_of(sp)
        assert any(f"## {name}" in str(sp)[a:b] for a, b in spans)


# --------------------------------------------------------------------------- #
#  search_tools                                                               #
# --------------------------------------------------------------------------- #

def test_keyword_search_returns_full_block_and_activates():
    s = _session()
    r = tool_search_tools(CWD, query="git commit", _session=s)
    assert r.ok
    assert r.output.count("## git_commit - ") == 1
    assert '{"name": "git_commit", "args": {"message": "fix: handle empty config"}}' in r.output
    assert s.activated[0] == "git_commit"
    assert getattr(r, "untrusted_output", False) is False


def test_select_form_returns_exact_names_in_order_and_reports_unknown():
    s = _session()
    r = tool_search_tools(CWD, query="select: browser_read , rag_search,nope,browser_read",
                          _session=s)
    assert r.ok
    assert s.activated == ["browser_read", "rag_search"]
    assert r.output.index("## browser_read") < r.output.index("## rag_search")
    assert "Not available in this session: nope" in r.output


def test_keyword_search_caps_the_number_of_matches():
    s = _session()
    r = tool_search_tools(CWD, query="browser", _session=s)
    assert len(s.activated) == MAX_MATCHES
    assert all(n.startswith("browser_") for n in s.activated)
    assert r.output.count("\n## ") + r.output.startswith("## ") <= MAX_MATCHES


def test_disabled_tool_is_never_returned_or_activated():
    s = _session(disabled={"git_push"})
    r = tool_search_tools(CWD, query="select:git_push", _session=s)
    assert s.activated == []
    assert "## git_push" not in r.output
    r = tool_search_tools(CWD, query="git push remote branch", _session=s)
    assert "git_push" not in s.activated
    assert "## git_push" not in r.output


def test_no_match_lists_the_deferred_names_and_activates_nothing():
    s = _session()
    r = tool_search_tools(CWD, query="zzqqxx", _session=s)
    assert r.ok
    assert s.activated == []
    assert "No tool matches" in r.output
    assert "git_commit" in r.output and "## " not in r.output


def test_no_match_listing_with_a_foreign_name_is_fenced():
    with patch.dict(TOOL_REGISTRY, {}, clear=False):
        name = _register_mcp(name="mcp_srv_odd")
        r = tool_search_tools(CWD, query="zzqqxx", _session=_session())
        assert name in r.output
        assert r.untrusted_output is True
    r = tool_search_tools(CWD, query="zzqqxx", _session=_session())
    assert getattr(r, "untrusted_output", False) is False


def test_empty_query_and_missing_session_are_errors():
    assert not tool_search_tools(CWD, query="  ", _session=_session()).ok
    assert not tool_search_tools(CWD, query="git", _session=None).ok


def test_search_never_returns_itself():
    s = _session()
    r = tool_search_tools(CWD, query="select:search_tools", _session=s)
    assert s.activated == []
    assert "## search_tools" not in r.output


def test_foreign_match_marks_the_result_untrusted():
    with patch.dict(TOOL_REGISTRY, {}, clear=False):
        name = _register_mcp(name="plg_lookup_record")
        s = _session()
        r = tool_search_tools(CWD, query=f"select:{name}", _session=s)
        assert s.activated == [name]
        assert r.untrusted_output is True
        assert is_untrusted_tool(TOOL_SEARCH_TOOL, TOOL_REGISTRY[TOOL_SEARCH_TOOL], r)
    trusted = tool_search_tools(CWD, query="select:git_log", _session=_session())
    assert not is_untrusted_tool(TOOL_SEARCH_TOOL, TOOL_REGISTRY[TOOL_SEARCH_TOOL], trusted)


# --------------------------------------------------------------------------- #
#  Through a real Agent                                                       #
# --------------------------------------------------------------------------- #

def _agent(tmp_path, backend=None, **kwargs):
    from unittest.mock import MagicMock
    from localm.plugins.coder.agent import Agent
    if backend is None:
        backend = MagicMock()
        backend.model_id = "qwen2.5-coder-7b"
        backend.native_tools = False
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def _http_backend(native=True, local=False):
    from localm.plugins.coder.backends.http import HTTPBackend
    return HTTPBackend("http://127.0.0.1:9/v1", "qwen2.5-coder-7b",
                       native_tools=native, localm_server=local, verify=False)


def _call(name, **args):
    from localm.plugins.coder.parser import ToolCall
    return ToolCall(name=name, args=args, raw="", start=0, end=0)


def _declared(body):
    return [t["function"]["name"] for t in body.get("tools", [])]


def test_search_tools_through_the_agent_keeps_the_prompt_byte_identical(tmp_path):
    agent = _agent(tmp_path, auto_approve=True)
    before = agent._system_prompt
    first = agent._build_messages()[0]["content"]
    r = agent._execute_tool(_call("search_tools", query="select:git_commit"), False)
    assert r.ok and "## git_commit - " in r.output
    assert agent._activated_tools == {"git_commit"}
    assert agent._system_prompt == before
    agent._rebuild_system_prompt()
    assert agent._system_prompt == before
    assert agent._build_messages()[0]["content"] == first


def test_deferred_tool_runs_without_a_search_first(tmp_path):
    agent = _agent(tmp_path, auto_approve=True)
    r = agent._execute_tool(_call("read_env"), False)
    assert "Unknown tool" not in r.output
    assert "disabled" not in r.output


def test_restricted_session_has_no_search_tools_and_full_docs(tmp_path):
    agent = _agent(tmp_path, restricted=True)
    assert "search_tools" in agent.disabled_tools
    assert "OTHER TOOLS:" not in agent._system_prompt
    assert "## edit_notebook_cell - " in agent._system_prompt
    r = agent._execute_tool(_call("search_tools", query="git"), False)
    assert not r.ok and "disabled" in r.output


def test_native_requests_declare_core_plus_activated_tools(tmp_path):
    backend = _http_backend()
    agent = _agent(tmp_path, backend=backend, auto_approve=True)
    assert "\n## " not in agent._system_prompt
    body = backend._body([{"role": "user", "content": "x"}], stream=False,
                         **agent._llm_kwargs())
    names = _declared(body)
    assert "read_file" in names and "search_tools" in names
    assert "git_commit" not in names and "browser_read" not in names
    agent._execute_tool(_call("search_tools", query="select:git_commit"), False)
    body = backend._body([{"role": "user", "content": "x"}], stream=False,
                         **agent._llm_kwargs())
    assert "git_commit" in _declared(body)
    assert "git_push" not in _declared(body)


def test_small_family_native_declares_every_enabled_tool_but_search_tools(tmp_path):
    from localm.plugins.coder.backends.http import HTTPBackend
    backend = HTTPBackend("http://127.0.0.1:9/v1", "phi-3-mini", native_tools=True,
                          verify=False)
    agent = _agent(tmp_path, backend=backend)
    names = _declared(backend._body([{"role": "user", "content": "x"}], stream=False,
                                    **agent._llm_kwargs()))
    assert DEFERRED_TOOLS <= set(names) | agent.disabled_tools
    assert "git_commit" in names
    assert "search_tools" not in names
    assert "search_tools" not in agent._system_prompt


def test_a_child_sharing_the_backend_does_not_change_the_parents_declared_tools(tmp_path):
    backend = _http_backend()
    parent = _agent(tmp_path, backend=backend, auto_approve=True)
    parent._activate_tools(["git_commit"])
    _agent(tmp_path, backend=backend, disabled_tools=frozenset({"run_shell", "git_commit"}))
    body = backend._body([{"role": "user", "content": "x"}], stream=False,
                         **parent._llm_kwargs())
    assert "git_commit" in _declared(body)
    assert "run_shell" in _declared(body)


def test_localm_server_keeps_full_docs_and_sends_no_tools(tmp_path):
    backend = _http_backend(local=True)
    agent = _agent(tmp_path, backend=backend)
    assert "## read_file - " in agent._system_prompt
    body = backend._body([{"role": "user", "content": "x"}], stream=False,
                         **agent._llm_kwargs())
    assert "tools" not in body and "tool_defs" not in body


def test_clear_and_checkpoint_round_trip_the_activation(tmp_path):
    agent = _agent(tmp_path, auto_approve=True)
    agent._activate_tools(["git_commit", "rag_search"])
    data = {"messages": [], "activated_tools": ["git_commit", "rag_search", 7, "no_such_tool"]}
    fresh = _agent(tmp_path)
    fresh.resume_checkpoint(data)
    assert fresh._activated_tools == {"git_commit", "rag_search"}
    fresh.resume_checkpoint({"messages": [], "activated_tools": "git_commit"})
    assert fresh._activated_tools == frozenset()
    agent.reset()
    assert agent._activated_tools == frozenset()


def test_checkpoint_writes_the_activation(tmp_path):
    import json
    from localm.plugins.coder.audit import SessionMode
    agent = _agent(tmp_path, auto_approve=True)
    agent.mode = SessionMode.FULL
    agent._activate_tools(["rag_search", "git_commit"])
    written = {}
    with patch("localm.plugins.coder.agent.persistence.atomic_write",
               side_effect=lambda p, text: written.update(json.loads(text))):
        assert agent.save_checkpoint() is True
    assert written["activated_tools"] == ["git_commit", "rag_search"]


def test_skill_restriction_does_not_block_search_tools(tmp_path):
    agent = _agent(tmp_path)
    agent._active_skill_tools = frozenset({"read_file"})
    agent._active_skill_names = ["s"]
    with patch.object(type(agent), "_expire_active_skill_locked", lambda self: None):
        assert agent._skill_gate_denial("search_tools") is None
        assert agent._skill_gate_denial("git_commit") is not None


def test_foreign_tool_line_is_a_short_untrusted_summary():
    from localm.plugins.coder.agent.core import _FOREIGN_SUMMARY_MAX, _foreign_tool_docs
    long_desc = "Does a thing. " + "word " * 200
    with patch.dict(TOOL_REGISTRY, {}, clear=False):
        name = _register_mcp(description="x" * 400)
        other = _register_mcp(name="mcp_srv_two", description=long_desc)
        docs = _foreign_tool_docs("EXTERNAL\n", [name, other], TOOL_REGISTRY)
        lines = str(docs).splitlines()[1:]
        assert lines[0] == f"- {name}: " + "x" * (_FOREIGN_SUMMARY_MAX - 3) + "..."
        assert lines[1] == f"- {other}: Does a thing"
        assert [str(docs)[a:b] for a, b in untrusted_spans_of(docs)] == [
            ln[2:] for ln in lines]
