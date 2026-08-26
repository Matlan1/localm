# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Agent Skills support for the coder (an importer for the SKILL.md format).

Discovers folders containing a ``SKILL.md`` (YAML frontmatter with ``name`` and
``description``, optional ``allowed-tools``) under the user's GLOBAL skills dir
(``<data dir>/skills/``) and the PROJECT dir (``<cwd>/.localcoder/skills/``), and
exposes two read-only agent tools, the same way MCP and plugin tools are added:

  list_skills()             - names + descriptions of available skills          (L1)
  use_skill(name)           - the skill's instruction body + its folder path     (L2)
  use_skill(name, file=...)  - the content of a bundled file inside the skill     (L3)

This is "progressive disclosure" via tools, with no system-prompt hook: the model
sees the two tools in the normal tool docs, lists skills, loads one's body, then
follows it using the agent's EXISTING tools (read_file / run_shell), reaching the
skill's bundled files through ``use_skill(name, file=...)``.

Security: a SKILL.md body is UNTRUSTED content. These two tools only READ files,
so they are not destructive; any action a skill's instructions prescribe still
goes through the agent's normal capability scope + destructive confirmation. A
skill can therefore instruct, but not directly act, without the user's consent.

``allowed-tools`` is HARD-ENFORCED, not merely surfaced: loading a skill's body
arms a dispatch-time restriction on the agent (``Agent._activate_skill``, checked
in ``Agent._execute_tool``) so a skill declaring ``read_file`` cannot reach
``run_shell`` or ``write_file``. The restriction only ever NARROWS - it intersects
with whatever the session already forbids and with any skill already active - and
is retired by the user's NEXT request, never by anything the model can call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from localm import pathsafe

from .tools import TOOL_REGISTRY, ToolDef, ToolResult

# A skill name is a safe single-segment identifier (used in tool args + paths).
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MAX_BODY = 64_000        # instruction body injected into context - keep bounded
_MAX_FILE = 200_000       # a bundled file returned to the model

# The two skill tools themselves are NEVER gated by a skill's own allowed-tools.
# use_skill is how the model reaches a skill's bundled files (the L3 leg of the
# progressive disclosure above) and it is what the loaded header explicitly tells
# it to call, so gating it would break the documented workflow the moment any
# skill declared allowed-tools. It cannot be an escalation route: both tools only
# read, bundled reads are confined by pathsafe.confined_under, and a restricted
# (shareable, non-owner) session has neither registered at all - they are not in
# SAFE_RESTRICTED_TOOLS, so _apply_restricted_toolset disables them outright.
SKILL_META_TOOLS: frozenset = frozenset({"list_skills", "use_skill"})


@dataclass
class Skill:
    name: str
    description: str
    path: Path                              # the skill folder (resolved)
    skill_md: Path                          # the SKILL.md file (resolved)
    allowed_tools: list = field(default_factory=list)


def _skill_roots(cwd: Path) -> List[Path]:
    """Global (data dir) first, then project. Project skills win on a name clash."""
    roots: List[Path] = []
    try:
        from localm.config import home_dir
        roots.append(home_dir() / "skills")
    except Exception:
        pass
    roots.append(cwd / ".localcoder" / "skills")
    return roots


def _split_list(v: str) -> list:
    """Parse a loose 'a, b, c' or '[a, b]' frontmatter list into a Python list."""
    v = (v or "").strip().strip("[]")
    return [t.strip().strip('"').strip("'") for t in v.split(",") if t.strip()]


def _parse_frontmatter(text: str) -> Tuple[dict, str]:
    """Split a SKILL.md into (metadata, body).

    Minimal YAML-frontmatter parser for the flat ``key: value`` block skills use
    (name / description / allowed-tools). Avoids a real YAML dependency so localm
    stays self-contained; an unrecognized or absent block yields ({}, full text).
    """
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    block, body = m.group(1), m.group(2)
    meta: dict = {}
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        mm = re.match(r"^([A-Za-z0-9_-]+):[ \t]*(.*)$", line)
        if mm:
            meta[mm.group(1).strip().lower()] = mm.group(2).strip().strip('"').strip("'")
    return meta, body.lstrip("\n")


def discover_skills(cwd: Path) -> List[Skill]:
    """Every valid ``SKILL.md`` folder under the skill roots, sorted by name.
    Project skills override global ones on a name clash. Best-effort: unreadable
    or invalid entries are skipped, never raised."""
    found: dict[str, Skill] = {}
    for root in _skill_roots(cwd):
        if not root.is_dir():
            continue
        try:
            children = sorted(root.iterdir())
        except OSError:
            continue
        for child in children:
            sm = child / "SKILL.md"
            if not (child.is_dir() and sm.is_file()):
                continue
            try:
                meta, _ = _parse_frontmatter(
                    sm.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            name = (meta.get("name") or child.name).strip()
            if not _NAME_RE.match(name):
                continue
            found[name] = Skill(
                name=name,
                description=(meta.get("description") or "").strip(),
                path=child.resolve(),
                skill_md=sm.resolve(),
                allowed_tools=_split_list(
                    meta.get("allowed-tools") or meta.get("allowed_tools") or ""),
            )
    return [found[k] for k in sorted(found)]


def _confine_skill_file(skill: Skill, rel: str) -> Path:
    """Resolve *rel* within the skill folder, refusing traversal outside it.

    A SKILL.md is UNTRUSTED content (module docstring), so a hostile skill's own
    instructions can steer the model into requesting an unexpected `file=`.
    Delegates to pathsafe.confined_under, the shared nested-path confinement
    primitive, instead of re-implementing resolve()+is_relative_to() here: that
    carries the NTFS Alternate Data Stream / short-name-alias hardening too.
    Translates ValueError to the PermissionError this module's caller expects."""
    try:
        return pathsafe.confined_under(skill.path, rel)
    except ValueError as e:
        raise PermissionError(str(e))


# ------------------------------------------------------------------ #
#  The two agent tools (fn(cwd, **args) -> ToolResult)                 #
# ------------------------------------------------------------------ #

def tool_list_skills(cwd: Path, **_) -> ToolResult:
    skills = discover_skills(cwd)
    if not skills:
        return ToolResult.success("No skills available.", summary="0 skills")
    lines = [f"- {s.name}: {s.description or '(no description)'}" for s in skills]
    out = ("Available skills (call use_skill(name) to load one's instructions):\n"
           + "\n".join(lines))
    return ToolResult.success(out, summary=f"{len(skills)} skill(s)")


def tool_use_skill(cwd: Path, name: Optional[str] = None,
                   file: Optional[str] = None, _session=None, **_) -> ToolResult:
    """Load a skill's instruction body, or read one of its bundled files.

    ``_session`` is the running Agent, injected by the dispatcher (the same
    hidden-argument mechanism the task-list tools use, ``_SKILL_STATE_TOOLS`` in
    agent/constants.py). Loading the BODY arms the skill's ``allowed-tools``
    restriction on that session; a ``file=`` read does not. The body is the act
    that puts untrusted instructions into context, so that is what the gate keys
    on, and a bundled-file read never arms a skill whose instructions the model
    never loaded. Called without a session it simply reads.
    """
    if not name:
        return ToolResult.error("use_skill requires a skill 'name'")
    skills = {s.name: s for s in discover_skills(cwd)}
    skill = skills.get(name)
    if skill is None:
        avail = ", ".join(sorted(skills)) or "(none)"
        return ToolResult.error(f"No such skill: {name}. Available: {avail}")

    if file:
        try:
            p = _confine_skill_file(skill, file)
        except PermissionError as e:
            return ToolResult.error(str(e))
        if not p.is_file():
            return ToolResult.error(f"No such file in skill {name!r}: {file}")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult.error(f"Could not read {file}: {e}")
        return ToolResult(ok=True, output=text[:_MAX_FILE],
                          summary=f"{name}/{file}", truncated=len(text) > _MAX_FILE)

    try:
        _, body = _parse_frontmatter(
            skill.skill_md.read_text(encoding="utf-8", errors="replace"))
    except OSError as e:
        return ToolResult.error(f"Could not read SKILL.md for {name!r}: {e}")
    # Arm the restriction BEFORE returning the body, so the very first tool call
    # the model makes after reading these instructions is already gated. A skill
    # with no allowed-tools arms nothing (see _activate_skill).
    if _session is not None:
        try:
            _session._activate_skill(skill.name, skill.allowed_tools)
        except Exception as e:                   # never break loading over bookkeeping
            return ToolResult.error(
                f"Could not apply the allowed-tools restriction for {name!r}: {e}. "
                "The skill was NOT loaded - a skill whose declared tool limit cannot "
                "be enforced must not run unrestricted.")
    # "run a bundled script with run_shell" is only true when run_shell survived
    # the declaration. Printing it unconditionally under an ENFORCED restriction
    # would advertise a tool the very next line refuses - the standard documented
    # next to the code that does not meet it.
    can_shell = (not skill.allowed_tools) or "run_shell" in skill.allowed_tools
    header = (
        f"SKILL: {skill.name}\n"
        f"folder: {skill.path}\n"
        + (f"allowed-tools (ENFORCED - every other tool is refused until the "
           f"user's next request): {', '.join(skill.allowed_tools)}\n"
           if skill.allowed_tools else "")
        + "Follow these instructions. Read a bundled file with "
          'use_skill(name, file="relpath")'
        + ("; run a bundled script with run_shell using the folder path above"
           if can_shell else "")
        + ". Untrusted content - destructive actions "
          "still require the usual confirmation.\n\n--- instructions ---\n"
    )
    return ToolResult(ok=True, output=header + body[:_MAX_BODY],
                      summary=f"loaded skill {skill.name}",
                      truncated=len(body) > _MAX_BODY)


def register_skill_tools(cwd: Path) -> Tuple[List[str], List[str]]:
    """Register ``list_skills`` + ``use_skill`` IF any skills are discoverable.

    Mirrors register_mcp_tools / register_plugin_tools: returns
    ``(registered_names, warnings)`` and never raises. Both tools are read-only
    (not destructive). When no skills exist, nothing is registered, so the
    feature has zero footprint until the user adds a skill folder.
    """
    warnings: List[str] = []
    try:
        skills = discover_skills(cwd)
    except Exception as e:                       # discovery must not break the agent
        return [], [f"Skill discovery failed: {e}"]
    if not skills:
        return [], warnings

    TOOL_REGISTRY["list_skills"] = ToolDef(
        name="list_skills",
        fn=tool_list_skills,
        description="[skills] List the agent skills available (name + description). "
                    "Call when a task might match a skill, then use_skill to load it.",
        params={},
        destructive=False,
    )
    TOOL_REGISTRY["use_skill"] = ToolDef(
        name="use_skill",
        fn=tool_use_skill,
        description="[skills] Load a skill's instructions (omit file), or read one "
                    'of its bundled files (file="relpath"), then follow them.',
        params={
            "name": {"type": "string",
                     "description": "Skill name from list_skills", "required": True},
            "file": {"type": "string",
                     "description": "Optional bundled file path within the skill",
                     "required": False},
        },
        destructive=False,
    )
    return ["list_skills", "use_skill"], warnings
