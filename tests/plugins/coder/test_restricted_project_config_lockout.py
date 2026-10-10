# SPDX-License-Identifier: AGPL-3.0-or-later
"""A restricted (scoped-key) coder session runs no project-configured code and
cannot write the files that configure later coder sessions.

A restricted session is read + confined edit with no process execution. Two
things are pinned here:

  1. Building a restricted Agent starts no MCP server from the project's
     ``.localcoder/config.toml``, imports no installed plugin module and
     discovers no skill. An owner Agent over the same project still does.
  2. No restricted write tool can create or change anything under a
     ``.localcoder`` directory, at any depth and under any spelling the
     platform folds onto that name, including in patch mode.
"""

import json
import os
import sys
import textwrap
import uuid
from unittest.mock import patch

import pytest

from localm.audit import SessionMode
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import TOOL_REGISTRY


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
        PM.build.return_value.dirty = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Stub(), cwd=cwd, self_verify=False, **kw)


def _call(name, **args):
    return ToolCall(name=name, args=args, raw="", start=0, end=0)


# An MCP server that leaves a marker file the moment it is spawned, then speaks
# just enough of the protocol to register one tool.
_MARKER_SERVER = textwrap.dedent("""\
    import json, sys
    from pathlib import Path
    Path(sys.argv[1]).write_text("spawned", encoding="utf-8")

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method, mid = msg.get("method"), msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                "serverInfo": {"name": "mark", "version": "1"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{
                "name": "ping", "description": "ping",
                "inputSchema": {"type": "object", "properties": {}}}]}})
""")


@pytest.fixture()
def mcp_project(tmp_path):
    """A project whose ``.localcoder/config.toml`` declares a marker server.
    Yields (project dir, marker path); the marker lives outside the project."""
    from localm.plugins.coder import mcp as mcp_mod
    project = tmp_path / "proj"
    project.mkdir()
    script = tmp_path / "marker_server.py"
    script.write_text(_MARKER_SERVER, encoding="utf-8")
    marker = tmp_path / "SPAWNED.txt"
    cfg = project / ".localcoder"
    cfg.mkdir()
    (cfg / "config.toml").write_text(textwrap.dedent(f"""\
        [mcp.servers.mark]
        command = {json.dumps(sys.executable)}
        args = [{json.dumps(str(script))}, {json.dumps(str(marker))}]
        trusted = true
    """), encoding="utf-8")
    mcp_mod.stop_pooled_servers()
    TOOL_REGISTRY.pop("mcp_mark_ping", None)
    try:
        yield project, marker
    finally:
        TOOL_REGISTRY.pop("mcp_mark_ping", None)
        mcp_mod.stop_pooled_servers()


class TestRestrictedSessionStartsNoProjectMcpServer:
    def test_a_restricted_agent_does_not_spawn_the_configured_server(self, mcp_project):
        from localm.plugins.coder import mcp as mcp_mod
        project, marker = mcp_project
        agent = _agent(project, restricted=True)
        assert not marker.exists(), "a restricted session spawned a project MCP server"
        assert agent._mcp_tool_names == frozenset()
        assert "mcp_mark_ping" not in TOOL_REGISTRY
        assert mcp_mod._POOL == {}
        assert str(agent._mcp_docs) == ""

    def test_an_owner_agent_still_starts_it(self, mcp_project):
        project, marker = mcp_project
        agent = _agent(project, restricted=False)
        assert marker.read_text(encoding="utf-8") == "spawned"
        assert agent._mcp_tool_names == {"mcp_mark_ping"}
        assert "mcp_mark_ping" not in agent.disabled_tools

    def test_a_restricted_sub_agent_does_not_spawn_it_either(self, mcp_project):
        from localm.plugins.coder.tools.agents import inherited_child_kwargs
        project, marker = mcp_project
        parent = _agent(project, restricted=True)
        with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
             patch("localm.plugins.coder.agent.make_audit_log"), \
             patch("localm.plugins.coder.agent.load_memory", return_value=""):
            PM.build.return_value.file_count.return_value = 0
            PM.build.return_value.dirty = False
            from localm.plugins.coder.agent import Agent
            child = Agent(**inherited_child_kwargs(
                parent, backend=_Stub(), cwd=project, name="kid",
                max_turns=3, confirm_handler=None))
        assert child.restricted is True
        assert not marker.exists()
        assert child._mcp_tool_names == frozenset()


class TestRestrictedSessionLoadsNoExternalCode:
    def test_no_installed_plugin_module_is_imported(self, tmp_path):
        from localm.config import home_dir
        name = f"lmsp{uuid.uuid4().hex[:8]}"
        marker = tmp_path / "IMPORTED.txt"
        pdir = home_dir() / "plugins" / name
        pdir.mkdir(parents=True)
        (pdir / "plugin.toml").write_text(textwrap.dedent(f"""\
            [plugin]
            name = "{name}"
            version = "0.1.0"
            entry = "entry:main"

            [tools]
            exports = ["tool_ping"]
        """), encoding="utf-8")
        (pdir / "entry.py").write_text(textwrap.dedent(f"""\
            from pathlib import Path
            Path({str(marker)!r}).write_text("imported", encoding="utf-8")

            def main():
                pass

            def tool_ping(cwd, **args):
                return "pong"
        """), encoding="utf-8")
        reg_name = f"plugin_{name}_tool_ping"
        project = tmp_path / "proj"
        project.mkdir()
        try:
            _agent(project, restricted=True)
            assert not marker.exists(), "a restricted session imported a plugin module"
            assert reg_name not in TOOL_REGISTRY
            _agent(project, restricted=False)
            assert marker.exists()
            assert reg_name in TOOL_REGISTRY
        finally:
            TOOL_REGISTRY.pop(reg_name, None)
            sys.modules.pop(f"_localm_plugin_{name}", None)

    def test_no_project_skill_is_discovered(self, tmp_path):
        skill = tmp_path / ".localcoder" / "skills" / "demo"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: demo\ndescription: d\n---\nbody\n", encoding="utf-8")
        with patch.dict(TOOL_REGISTRY, {}, clear=False):
            TOOL_REGISTRY.pop("list_skills", None)
            TOOL_REGISTRY.pop("use_skill", None)
            restricted = _agent(tmp_path, restricted=True)
            assert "list_skills" not in TOOL_REGISTRY
            assert "use_skill" not in TOOL_REGISTRY
            assert str(restricted._skill_docs) == ""
            _agent(tmp_path, restricted=False)
            assert "list_skills" in TOOL_REGISTRY


_CONFIG = 'mode = "privacy"\n'


@pytest.fixture()
def project(tmp_path):
    """A project with an owner-written .localcoder/config.toml and a source file."""
    root = tmp_path / "proj"
    (root / ".localcoder").mkdir(parents=True)
    (root / ".localcoder" / "config.toml").write_text(_CONFIG, encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "a.txt").write_text("privacy\n", encoding="utf-8")
    return root


def _refused(result):
    return result.ok is False and ".localcoder" in result.output


class TestRestrictedWritesCannotReachLocalcoder:
    @pytest.mark.parametrize("path", [
        ".localcoder/config.toml",
        ".LocalCoder/config.toml",
        ".LOCALCODER/config.toml",
        ".localcoder./config.toml",
        ".localcoder /config.toml",
        "./.localcoder/system.md",
        ".localcoder\\config.toml",
        "src/../.localcoder/config.toml",
        "sub/.localcoder/config.toml",
        "sub/deeper/.localcoder/skills/x/SKILL.md",
        ".localcoder",
    ])
    def test_write_file_cannot_create_it(self, tmp_path, path):
        root = tmp_path / "proj"
        root.mkdir()
        agent = _agent(root, restricted=True)
        result = agent._execute_tool(
            _call("write_file", path=path, content="[mcp.servers.x]\ncommand = \"x\"\n"),
            interactive=False)
        assert _refused(result), result.output
        leftovers = [p for p in root.rglob("*")]
        assert leftovers == [], leftovers

    def test_an_absolute_path_inside_the_project_is_refused(self, tmp_path):
        root = (tmp_path / "proj")
        root.mkdir()
        agent = _agent(root.resolve(), restricted=True)
        target = root.resolve() / ".localcoder" / "config.toml"
        result = agent._execute_tool(
            _call("write_file", path=str(target), content="x"), interactive=False)
        assert _refused(result), result.output
        assert not target.exists()

    def test_write_file_cannot_overwrite_the_owners_config(self, project):
        agent = _agent(project, restricted=True)
        result = agent._execute_tool(
            _call("write_file", path=".localcoder/config.toml",
                  content='mode = "normal"\n'), interactive=False)
        assert _refused(result), result.output
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == _CONFIG

    @pytest.mark.parametrize("call", [
        _call("edit_file", path=".localcoder/config.toml",
              old="privacy", new="normal"),
        _call("edit_files", edits=[
            {"path": "src/a.txt", "old": "privacy", "new": "normal"},
            {"path": ".localcoder/config.toml", "old": "privacy", "new": "normal"}]),
        _call("patch_file", path=".localcoder/config.toml",
              diff='--- a/.localcoder/config.toml\n+++ b/.localcoder/config.toml\n'
                   '@@ -1 +1 @@\n-mode = "privacy"\n+mode = "normal"\n'),
        _call("edit_notebook_cell", path=".localcoder/nb.ipynb",
              cell_index=0, source="x"),
    ], ids=lambda c: c.name)
    def test_no_edit_tool_can_change_it(self, project, call):
        nb = project / ".localcoder" / "nb.ipynb"
        nb.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["a"],
                                             "metadata": {}, "outputs": [],
                                             "execution_count": None}],
                                  "metadata": {}, "nbformat": 4,
                                  "nbformat_minor": 5}), encoding="utf-8")
        nb_before = nb.read_bytes()
        agent = _agent(project, restricted=True)
        result = agent._execute_tool(call, interactive=False)
        assert _refused(result), result.output
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == _CONFIG
        assert nb.read_bytes() == nb_before
        # edit_files is all-or-nothing: its in-bounds edit was not applied either.
        assert (project / "src" / "a.txt").read_text(encoding="utf-8") == "privacy\n"

    def test_a_link_into_it_is_refused(self, project):
        link = project / "cfg"
        try:
            os.symlink(project / ".localcoder", link, target_is_directory=True)
        except (OSError, NotImplementedError) as e:
            pytest.skip(f"cannot create a directory symlink here: {e}")
        agent = _agent(project, restricted=True)
        result = agent._execute_tool(
            _call("write_file", path="cfg/config.toml", content="x"), interactive=False)
        assert result.ok is False, result.output
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == _CONFIG

    @staticmethod
    def _file_link(project):
        link = project / "config.toml"
        try:
            os.symlink(project / ".localcoder" / "config.toml", link)
        except (OSError, NotImplementedError) as e:
            pytest.skip(f"cannot create a file symlink here: {e}")
        return link

    @pytest.mark.parametrize("call", [
        _call("write_file", path="config.toml", content='mode = "normal"\n'),
        _call("edit_file", path="config.toml", old="privacy", new="normal"),
    ], ids=lambda c: c.name)
    def test_a_file_link_that_resolves_inside_it_is_refused(self, project, call):
        self._file_link(project)
        agent = _agent(project, restricted=True)
        result = agent._execute_tool(call, interactive=False)
        assert _refused(result), result.output
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == _CONFIG

    def test_a_sweep_does_not_write_through_a_file_link(self, project):
        self._file_link(project)
        agent = _agent(project, restricted=True)
        result = agent._execute_tool(
            _call("search_replace", pattern="privacy", replacement="normal",
                  glob="*.toml"), interactive=False)
        assert _refused(result), result.output
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == _CONFIG

    def test_ordinary_files_and_lookalike_names_are_still_writable(self, project):
        agent = _agent(project, restricted=True)
        for path in ("src/b.py", ".localcoder-notes.md", "localcoder/x.txt",
                     "docs/.localcoderx/y.txt"):
            result = agent._execute_tool(
                _call("write_file", path=path, content="ok"), interactive=False)
            assert result.ok, (path, result.output)
            assert (project / path).read_text(encoding="utf-8") == "ok"

    def test_the_owner_can_still_write_it(self, project):
        agent = _agent(project, restricted=False)
        result = agent._execute_tool(
            _call("write_file", path=".localcoder/config.toml",
                  content='mode = "normal"\n'), interactive=False)
        assert result.ok, result.output
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == 'mode = "normal"\n'


class TestRestrictedSearchReplaceLeavesLocalcoderAlone:
    def test_a_sweep_edits_the_project_and_skips_the_config(self, project):
        nested = project / "sub" / ".localcoder"
        nested.mkdir(parents=True)
        (nested / "config.toml").write_text(_CONFIG, encoding="utf-8")
        agent = _agent(project, restricted=True)
        result = agent._execute_tool(
            _call("search_replace", pattern="privacy", replacement="normal",
                  glob="**/*"), interactive=False)
        assert result.ok, result.output
        assert (project / "src" / "a.txt").read_text(encoding="utf-8") == "normal\n"
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == _CONFIG
        assert (nested / "config.toml").read_text(encoding="utf-8") == _CONFIG
        assert ".localcoder" in result.output

    def test_a_sweep_aimed_only_at_the_config_is_refused(self, project):
        agent = _agent(project, restricted=True)
        result = agent._execute_tool(
            _call("search_replace", pattern="privacy", replacement="normal",
                  glob=".localcoder/*"), interactive=False)
        assert _refused(result), result.output
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == _CONFIG

    def test_a_model_supplied_flag_cannot_lift_it(self, project):
        agent = _agent(project, restricted=True)
        result = agent._execute_tool(
            _call("search_replace", pattern="privacy", replacement="normal",
                  glob=".localcoder/*", _restricted=False), interactive=False)
        assert _refused(result), result.output
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == _CONFIG

    def test_the_owner_sweep_still_reaches_it(self, project):
        agent = _agent(project, restricted=False)
        result = agent._execute_tool(
            _call("search_replace", pattern="privacy", replacement="normal",
                  glob="**/*"), interactive=False)
        assert result.ok, result.output
        assert (project / ".localcoder" / "config.toml").read_text(
            encoding="utf-8") == 'mode = "normal"\n'


class TestRestrictedPatchModeLeavesLocalcoderAlone:
    def test_a_write_is_refused_not_captured(self, project):
        agent = _agent(project, restricted=True, patch_mode=True)
        result = agent._execute_tool(
            _call("write_file", path=".localcoder/config.toml", content="x"),
            interactive=False)
        assert _refused(result), result.output
        assert agent.current_patch() == ""

    def test_a_sweep_diff_excludes_it(self, project):
        agent = _agent(project, restricted=True, patch_mode=True)
        result = agent._execute_tool(
            _call("search_replace", pattern="privacy", replacement="normal",
                  glob="**/*"), interactive=False)
        assert result.ok, result.output
        patch_text = agent.current_patch()
        assert "a.txt" in patch_text
        assert ".localcoder" not in patch_text


class TestRestrictedUnwritableHelper:
    @pytest.mark.parametrize("path", [
        ".localcoder/a:b",
        ".localcoder\\config.toml",
        "x/../.Localcoder. /y",
    ])
    def test_its_own_spelling_is_enough(self, tmp_path, path):
        """Each is caught on the path's own components, before resolution."""
        from localm.plugins.coder.tools.base import restricted_unwritable
        assert restricted_unwritable(tmp_path, path) is True

    @pytest.mark.parametrize("path", [
        "src/x.py", ".localcoder-notes.md", "localcoder/x", "a/.localcoderx/y", ".",
    ])
    def test_other_paths_pass(self, tmp_path, path):
        from localm.plugins.coder.tools.base import restricted_unwritable
        assert restricted_unwritable(tmp_path, path) is False

    def test_a_project_inside_a_localcoder_dir_is_not_locked_whole(self, tmp_path):
        """Only components below the project root count."""
        from localm.plugins.coder.tools.base import restricted_unwritable
        root = tmp_path / ".localcoder" / "proj"
        root.mkdir(parents=True)
        assert restricted_unwritable(root, "src/x.py") is False
        assert restricted_unwritable(root, str(root / "src" / "x.py")) is False
        assert restricted_unwritable(root, ".localcoder/config.toml") is True


def test_every_restricted_write_tool_is_covered_by_the_lockout():
    """A write tool added to SAFE_RESTRICTED_TOOLS must be gated too."""
    from localm.plugins.coder.agent.constants import _RESTRICTED_PATH_WRITE_TOOLS
    from localm.plugins.coder.tools.registry import SAFE_RESTRICTED_TOOLS
    writers = {n for n in SAFE_RESTRICTED_TOOLS
               if n in TOOL_REGISTRY and TOOL_REGISTRY[n].destructive}
    assert writers, "the restricted allowlist lost every write tool?"
    uncovered = writers - _RESTRICTED_PATH_WRITE_TOOLS - {"search_replace"}
    assert uncovered == set(), uncovered


def test_the_lockout_targets_resolve_through_call_target_paths():
    """Every path-gated tool names its targets in a form _call_target_paths reads."""
    from localm.plugins.coder.agent.constants import (
        _RESTRICTED_PATH_WRITE_TOOLS, _call_target_paths)
    for name in _RESTRICTED_PATH_WRITE_TOOLS:
        params = TOOL_REGISTRY[name].params
        args = {"edits": [{"path": "p"}]} if "edits" in params else {"path": "p"}
        assert _call_target_paths(name, args) == ["p"], name
