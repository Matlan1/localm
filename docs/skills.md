# Agent skills (the `SKILL.md` importer)

The coder agent can discover and use **Agent Skills** - folders containing a
`SKILL.md` file (the same format used by Claude/Agent skills). A skill is a piece
of reusable, on-demand know-how: a markdown instruction body plus optional bundled
scripts and reference files. It is localm's first third-party plugin-ecosystem
importer (see [docs/plugin-interop.md](plugin-interop.md)).

## Where skills live

The agent looks in two places, project overriding global on a name clash:

- **Global** (reusable across projects): `<data dir>/skills/`
- **Project** (this repo only): `<project>/.localcoder/skills/`

Each skill is a sub-folder containing a `SKILL.md`:

```
<data dir>/skills/
  pdf-fill/
    SKILL.md
    fill.py
    template.json
```

Skills are something you provide; localm ships no built-in catalog. Write your
own (just a `SKILL.md` plus any helper files), or get one from a colleague or a
public repo. To install a skill, drop its folder into the global skills dir
(reusable everywhere) or a project's `.localcoder/skills/`; to share one, copy
the folder out. Review any skill from an unknown source first (see [Security](#security)).

## The `SKILL.md` format

YAML-style frontmatter (a flat `key: value` block) followed by the instruction
body. `name` and `description` are used; `allowed-tools` is parsed and shown to the
model:

```markdown
---
name: pdf-fill
description: Fill a PDF form from a JSON of field values.
allowed-tools: read_file, run_shell
---

To fill a PDF form:
1. Read `template.json` for the field names (use_skill with file="template.json").
2. Run `python <folder>/fill.py --in form.pdf --data values.json`.
3. ...
```

## How the agent uses skills

When at least one skill is present, the coder agent gains two read-only tools and
follows **progressive disclosure** - it only pulls in what a task needs:

- `list_skills()` - names + descriptions of the available skills.
- `use_skill(name)` - the skill's instruction body plus its folder path.
- `use_skill(name, file="relpath")` - the contents of a bundled file inside the
  skill folder (confined to that folder).

The agent runs bundled scripts with its normal `run_shell` using the folder path
the skill reports. With no skills present, the feature is inactive until you add one.

## Security

A `SKILL.md` body is **untrusted content**. `list_skills` and `use_skill` only
*read* files, so they never need confirmation - but anything a skill's instructions
prescribe (writing files, running shell commands) still goes through the agent's
normal capability scope and destructive-action confirmation. A skill can therefore
*instruct* the agent, but cannot *act* without your consent. Treat skills from
unknown sources like any other code you would run, and review them first.

`allowed-tools` is currently surfaced to the model but not hard-enforced; enforcing
it (restricting a skill to a tool subset) is planned future work.
