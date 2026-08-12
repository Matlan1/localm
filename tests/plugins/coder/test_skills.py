# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.coder.skills (the SKILL.md importer)."""

from pathlib import Path

import pytest

from localm.plugins.coder import skills as S
from localm.plugins.coder.tools import TOOL_REGISTRY


@pytest.fixture
def clean_registry():
    """Snapshot/restore the global TOOL_REGISTRY around tests that register tools."""
    snap = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(snap)


def _make_skill(root: Path, name: str, *, desc="A test skill.",
                body="Do the thing.", frontmatter=None, files=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = frontmatter if frontmatter is not None else f"name: {name}\ndescription: {desc}\n"
    (d / "SKILL.md").write_text(f"---\n{fm}---\n\n{body}\n", encoding="utf-8")
    for fn, content in (files or {}).items():
        (d / fn).write_text(content, encoding="utf-8")
    return d


# --- frontmatter parsing ---------------------------------------------------- #

def test_parse_frontmatter_flat():
    meta, body = S._parse_frontmatter("---\nname: x\ndescription: hi there\n---\n\nBODY\n")
    assert meta == {"name": "x", "description": "hi there"}
    assert body.strip() == "BODY"


def test_parse_frontmatter_absent():
    meta, body = S._parse_frontmatter("no frontmatter here")
    assert meta == {} and body == "no frontmatter here"


def test_parse_frontmatter_quotes_and_comments():
    meta, _ = S._parse_frontmatter(
        "---\nname: \"quoted\"\n# a comment\ndescription: 'single'\n---\nb")
    assert meta["name"] == "quoted" and meta["description"] == "single"


# --- discovery -------------------------------------------------------------- #

def test_discover_project_skill(tmp_path):
    _make_skill(tmp_path / ".localcoder" / "skills", "alpha", desc="Alpha skill")
    skills = S.discover_skills(tmp_path)
    assert [s.name for s in skills] == ["alpha"]
    assert skills[0].description == "Alpha skill"


def test_discover_global_and_project_override(tmp_path):
    # global root = <LOCALM_HOME>/skills (conftest pins LOCALM_HOME under tmp_path)
    from localm.config import home_dir
    _make_skill(home_dir() / "skills", "shared", desc="global one")
    _make_skill(home_dir() / "skills", "gonly", desc="global only")
    _make_skill(tmp_path / ".localcoder" / "skills", "shared", desc="project wins")
    skills = {s.name: s for s in S.discover_skills(tmp_path)}
    assert set(skills) == {"shared", "gonly"}
    assert skills["shared"].description == "project wins"     # project beats global


def test_discover_skips_dirs_without_skill_md(tmp_path):
    root = tmp_path / ".localcoder" / "skills"
    (root / "no-skill-md").mkdir(parents=True)
    _make_skill(root, "good")
    assert [s.name for s in S.discover_skills(tmp_path)] == ["good"]


def test_discover_parses_allowed_tools(tmp_path):
    _make_skill(tmp_path / ".localcoder" / "skills", "alpha",
                frontmatter="name: alpha\ndescription: d\nallowed-tools: read_file, run_shell\n")
    s = S.discover_skills(tmp_path)[0]
    assert s.allowed_tools == ["read_file", "run_shell"]


def test_discover_none(tmp_path):
    assert S.discover_skills(tmp_path) == []


# --- registration ----------------------------------------------------------- #

def test_register_when_skills_exist(tmp_path, clean_registry):
    _make_skill(tmp_path / ".localcoder" / "skills", "alpha")
    names, warns = S.register_skill_tools(tmp_path)
    assert names == ["list_skills", "use_skill"] and warns == []
    assert "list_skills" in TOOL_REGISTRY and "use_skill" in TOOL_REGISTRY
    # read-only: discovering/loading a skill never needs a destructive confirm
    assert TOOL_REGISTRY["list_skills"].destructive is False
    assert TOOL_REGISTRY["use_skill"].destructive is False


def test_register_noop_without_skills(tmp_path, clean_registry):
    names, warns = S.register_skill_tools(tmp_path)
    assert names == [] and "list_skills" not in TOOL_REGISTRY


# --- the tools -------------------------------------------------------------- #

def test_list_skills_tool(tmp_path):
    _make_skill(tmp_path / ".localcoder" / "skills", "alpha", desc="does alpha")
    _make_skill(tmp_path / ".localcoder" / "skills", "beta", desc="does beta")
    out = S.tool_list_skills(tmp_path)
    assert out.ok
    assert "alpha: does alpha" in out.output and "beta: does beta" in out.output


def test_use_skill_returns_body_and_folder(tmp_path):
    d = _make_skill(tmp_path / ".localcoder" / "skills", "alpha", body="STEP ONE")
    out = S.tool_use_skill(tmp_path, name="alpha")
    assert out.ok and "STEP ONE" in out.output and str(d.resolve()) in out.output


def test_use_skill_reads_bundled_file(tmp_path):
    _make_skill(tmp_path / ".localcoder" / "skills", "alpha", files={"ref.txt": "REFDATA"})
    out = S.tool_use_skill(tmp_path, name="alpha", file="ref.txt")
    assert out.ok and out.output == "REFDATA"


def test_use_skill_confines_to_folder(tmp_path):
    _make_skill(tmp_path / ".localcoder" / "skills", "alpha")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    out = S.tool_use_skill(tmp_path, name="alpha", file="../../../secret.txt")
    # Message comes from pathsafe.confined_under now (traversal-specific,
    # not the old hand-rolled "escapes" wording) - what matters is that it
    # is refused and the secret is never read, not the exact phrasing.
    assert (not out.ok) and "traversal" in out.output
    assert "nope" not in out.output


def test_use_skill_rejects_reserved_characters(tmp_path):
    """Same class #1068 fixed for model filenames: a colon opens an NTFS
    Alternate Data Stream rather than failing the read, so a naive
    'no separators' check would pass 'ref.txt:hidden' straight through. The
    OLD hand-rolled _confine_skill_file had no such check at all."""
    _make_skill(tmp_path / ".localcoder" / "skills", "alpha", files={"ref.txt": "REFDATA"})
    out = S.tool_use_skill(tmp_path, name="alpha", file="ref.txt:hidden")
    assert not out.ok


def test_use_skill_alias_leaf_does_not_read_a_different_file(tmp_path, monkeypatch):
    """An OS-level short-name alias resolving 'file' to a DIFFERENT, real
    sibling inside the same skill folder stays strictly under the folder -
    containment alone would not catch it. Deterministic simulation, no real
    8.3-enabled volume needed (same technique as test_pathsafe_confined_under.py)."""
    d = _make_skill(tmp_path / ".localcoder" / "skills", "alpha",
                    files={"LongReferenceFileName.md": "SECRET REFERENCE DATA"})
    victim = d / "LongReferenceFileName.md"
    alias = "LONGRE~1.MD"
    real_resolve = Path.resolve

    def fake_resolve(self, *a, **k):
        parts = list(self.parts)
        if alias in parts:
            parts[parts.index(alias)] = victim.name
            return real_resolve(Path(*parts), *a, **k)
        return real_resolve(self, *a, **k)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    out = S.tool_use_skill(tmp_path, name="alpha", file=alias)
    assert not out.ok


def test_use_skill_missing(tmp_path):
    out = S.tool_use_skill(tmp_path, name="ghost")
    assert (not out.ok) and "No such skill" in out.output


def test_use_skill_requires_name(tmp_path):
    out = S.tool_use_skill(tmp_path)
    assert (not out.ok) and "name" in out.output


# --- agent wiring (integration) --------------------------------------------- #

class _StubBackend:
    """Minimal backend: Agent.__init__ only reads these attrs (no inference)."""
    model_id = "stub-model"
    native_tools = False
    supports_grammar = False
    last_usage: dict = {}

    def chat(self, *a, **k):
        return ""

    def chat_stream(self, *a, **k):
        yield ""


def test_agent_registers_skill_tools_on_init(tmp_path, clean_registry):
    _make_skill(tmp_path / ".localcoder" / "skills", "alpha", desc="does alpha")
    from localm.plugins.coder.agent import Agent
    agent = Agent(_StubBackend(), cwd=tmp_path)
    assert "list_skills" in TOOL_REGISTRY and "use_skill" in TOOL_REGISTRY
    # the system prompt advertises the feature + auto-documents the tools
    assert "list_skills" in agent._system_prompt
    assert "AGENT SKILLS" in agent._system_prompt


# --- allowed-tools is HARD-ENFORCED (REC-SKILL-TOOLS) ------------------------ #
#
# Every test here drives the REAL dispatch path, agent._execute_tool(ToolCall(...)),
# not the tool functions directly: the gate lives in the dispatcher, so calling
# tool_use_skill() by hand would prove nothing about what a model can actually
# reach. Denials are asserted from OUTSIDE with a counting mock plus
# assert_not_called() - never a raising side_effect, because _execute_tool wraps
# tool_def.fn in `except Exception` and would absorb an AssertionError as an
# ordinary tool failure, passing the test in both directions.

def _tool_call(_tool: str, /, **args):
    # positional-only: use_skill's own argument is called "name" too
    from localm.plugins.coder.parser import ToolCall
    return ToolCall(name=_tool, args=args, raw="", start=0, end=0)


def _counting_tool(registry, name: str, *, destructive=False):
    """Replace *name* in the registry with a MagicMock-backed no-op and return it."""
    from unittest.mock import MagicMock
    from localm.plugins.coder.tools import ToolDef, ToolResult
    fn = MagicMock(return_value=ToolResult.success("ran", summary=name))
    registry[name] = ToolDef(name=name, fn=fn, description="x", params={},
                             destructive=destructive)
    return fn


@pytest.fixture
def skill_agent(tmp_path, clean_registry):
    """An Agent in *tmp_path* with two project skills:
    ``narrow`` (allowed-tools: read_file) and ``open`` (no allowed-tools)."""
    root = tmp_path / ".localcoder" / "skills"
    _make_skill(root, "narrow",
                frontmatter="name: narrow\ndescription: d\nallowed-tools: read_file\n")
    _make_skill(root, "open", frontmatter="name: open\ndescription: d\n")
    from localm.plugins.coder.agent import Agent
    agent = Agent(_StubBackend(), cwd=tmp_path, auto_approve=True)
    # A user request is what starts a turn; the restriction is retired by the NEXT
    # one, so every test needs to be inside one to be testing the real lifetime.
    agent._last_user_request = "do the thing"
    return agent


def _use(agent, name):
    res = agent._execute_tool(_tool_call("use_skill", name=name), interactive=False)
    assert res.ok, res.output
    return res


def test_skill_gate_denies_a_tool_outside_allowed_tools(skill_agent):
    """(a) POSITIVE: the declared subset is enforced, and the tool never runs."""
    write = _counting_tool(TOOL_REGISTRY, "write_file")
    _use(skill_agent, "narrow")
    res = skill_agent._execute_tool(
        _tool_call("write_file", path="x.txt", content="hi"), interactive=False)
    assert not res.ok
    assert "narrow" in res.output and "read_file" in res.output
    # The load-bearing assertion: refused, not merely reported as refused.
    write.assert_not_called()


def test_skill_gate_allows_a_declared_tool(skill_agent):
    """(b) NEGATIVE: read_file still works under the same active skill."""
    read = _counting_tool(TOOL_REGISTRY, "read_file")
    _use(skill_agent, "narrow")
    res = skill_agent._execute_tool(_tool_call("read_file", path="x.txt"),
                                    interactive=False)
    assert res.ok
    read.assert_called_once()


def test_skill_restriction_is_retired_by_the_next_user_request(skill_agent):
    """(c) LIFETIME: a new USER request ends it - and nothing the model can call does.

    Asserted on the ASSIGNMENT, not the value: the same request text repeated is
    still a new turn, which is exactly the case a string comparison would miss.
    """
    write = _counting_tool(TOOL_REGISTRY, "write_file")
    _use(skill_agent, "narrow")
    assert not skill_agent._execute_tool(
        _tool_call("write_file", path="x.txt"), interactive=False).ok
    write.assert_not_called()

    skill_agent._last_user_request = "do the thing"      # byte-identical on purpose
    res = skill_agent._execute_tool(_tool_call("write_file", path="x.txt"),
                                    interactive=False)
    assert res.ok
    write.assert_called_once()


def test_skill_gate_intersects_with_disabled_tools_never_widens(tmp_path, clean_registry):
    """(d) INTERSECTION: allowed by the skill AND disabled by the operator = DENIED.

    The direction that matters: a skill must never hand back capability the
    session forbids, or the restriction is a privilege escalation.
    """
    _make_skill(tmp_path / ".localcoder" / "skills", "narrow",
                frontmatter="name: narrow\ndescription: d\nallowed-tools: read_file\n")
    from localm.plugins.coder.agent import Agent
    agent = Agent(_StubBackend(), cwd=tmp_path, auto_approve=True,
                  disabled_tools=frozenset({"read_file"}))
    agent._last_user_request = "go"
    read = _counting_tool(TOOL_REGISTRY, "read_file")
    _use(agent, "narrow")
    res = agent._execute_tool(_tool_call("read_file", path="x.txt"), interactive=False)
    assert not res.ok and "disabled" in res.output
    read.assert_not_called()


def test_absent_allowed_tools_restricts_nothing(skill_agent):
    """A skill with no allowed-tools arms nothing - deliberate and backward
    compatible, since the field is optional and most skills omit it."""
    write = _counting_tool(TOOL_REGISTRY, "write_file")
    _use(skill_agent, "open")
    assert skill_agent._execute_tool(
        _tool_call("write_file", path="x.txt"), interactive=False).ok
    write.assert_called_once()


def test_a_second_skill_can_only_narrow_further(skill_agent):
    """Loading an UNRESTRICTED skill while a restricted one is active must not
    widen. This is the bypass a hostile SKILL.md would reach for first: its own
    body can tell the model to load another skill."""
    write = _counting_tool(TOOL_REGISTRY, "write_file")
    _use(skill_agent, "narrow")
    _use(skill_agent, "open")            # declares nothing, so contributes nothing
    res = skill_agent._execute_tool(_tool_call("write_file", path="x.txt"),
                                    interactive=False)
    assert not res.ok
    write.assert_not_called()


def test_two_restricted_skills_intersect(tmp_path, clean_registry):
    """Two declarations compose to their INTERSECTION, not their union: after
    loading both, only what BOTH declare survives."""
    root = tmp_path / ".localcoder" / "skills"
    _make_skill(root, "reader",
                frontmatter="name: reader\ndescription: d\nallowed-tools: read_file, grep\n")
    _make_skill(root, "grepper",
                frontmatter="name: grepper\ndescription: d\nallowed-tools: grep\n")
    from localm.plugins.coder.agent import Agent
    agent = Agent(_StubBackend(), cwd=tmp_path, auto_approve=True)
    agent._last_user_request = "go"
    read = _counting_tool(TOOL_REGISTRY, "read_file")
    grep = _counting_tool(TOOL_REGISTRY, "grep")
    _use(agent, "reader")
    _use(agent, "grepper")
    assert not agent._execute_tool(_tool_call("read_file", path="x"),
                                   interactive=False).ok
    read.assert_not_called()
    assert agent._execute_tool(_tool_call("grep", pattern="x"), interactive=False).ok
    grep.assert_called_once()


def test_use_skill_itself_is_never_gated(skill_agent):
    """The skill tools stay reachable under any restriction: the loaded header
    tells the model to read bundled files with use_skill(name, file=...), so
    gating them would break the documented workflow."""
    _use(skill_agent, "narrow")
    res = skill_agent._execute_tool(_tool_call("list_skills"), interactive=False)
    assert res.ok
    res = skill_agent._execute_tool(
        _tool_call("use_skill", name="narrow"), interactive=False)
    assert res.ok


def test_loaded_header_states_the_restriction_is_enforced(skill_agent):
    """The model is told the truth about what it is holding - and is NOT told to
    reach for run_shell when the declaration excludes it."""
    out = _use(skill_agent, "narrow").output
    assert "ENFORCED" in out and "read_file" in out
    assert "run_shell" not in out


def test_spawned_child_inherits_the_active_skill_restriction(skill_agent):
    """A skill declaring spawn_agent must not delegate its way out of its own
    declaration: the child is built narrowed too."""
    from localm.plugins.coder.tools.agents import inherited_child_kwargs
    _use(skill_agent, "narrow")
    kwargs = inherited_child_kwargs(
        backend=_StubBackend(), cwd=skill_agent.cwd, name="child",
        max_turns=3, parent=skill_agent, confirm_handler=None, role=None)
    assert kwargs["inherited_skill_tools"] == frozenset({"read_file"})

    from localm.plugins.coder.agent import Agent
    child = Agent(**kwargs)
    assert "write_file" in child.disabled_tools     # narrowed
    assert "read_file" not in child.disabled_tools  # but not beyond the declaration
    assert "use_skill" not in child.disabled_tools  # bundled files stay reachable
