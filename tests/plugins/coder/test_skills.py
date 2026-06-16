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
    assert (not out.ok) and "escapes" in out.output


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
