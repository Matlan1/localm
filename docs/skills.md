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
the folder out. To remove a skill, delete its folder - there is no separate
uninstall step and no CLI or GUI skill manager, since skills are file-based
only. Review any skill from an unknown source first (see [Security](#security)).

## The `SKILL.md` format

YAML-style frontmatter (a flat `key: value` block) followed by the instruction
body. `name` and `description` are used; `allowed-tools` is parsed and
**enforced** (see [Enforcement](#enforcement-of-allowed-tools) below):

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

`allowed-tools` is a comma-separated list (`a, b, c`, with or without a
surrounding `[...]`). It is optional; most skills omit it, and an absent or
empty list restricts nothing.

## How the agent uses skills

When at least one skill is present, the coder agent gains two tools and follows
**progressive disclosure** - it only pulls in what a task needs:

- `list_skills()` - names + descriptions of the available skills. Read-only,
  never gated.
- `use_skill(name)` - the skill's instruction body plus its folder path. If the
  skill declares `allowed-tools`, this call also **arms** the restriction (see
  below) before the body is returned.
- `use_skill(name, file="relpath")` - the contents of a bundled file inside the
  skill folder (confined to that folder). Reading a file does **not** arm the
  restriction; only loading the body does.

The agent runs bundled scripts with its normal `run_shell` using the folder path
the skill reports. With no skills present, the feature is inactive until you add one.

## Enforcement of `allowed-tools`

A skill's `allowed-tools` is **hard-enforced**, not merely shown to the model.
The moment `use_skill(name)` returns the skill's body, every tool call for the
rest of the current turn is checked against that list; anything not on it is
refused with an explanation naming the active skill and what it does allow.

What this means in practice:

- **Only narrowing, never widening.** The restriction is intersected with
  whatever tools the session already disallows for other reasons, and with any
  skill already active. Loading a second skill that declares its own
  `allowed-tools` shrinks the effective set further; loading one with no
  `allowed-tools` at all leaves the existing restriction untouched.
- **No release the model can call.** There is no tool or argument that lifts an
  active restriction early - not asking again, not loading an unrestricted
  skill, not any other trick reachable from inside a turn.
- **Expires on your next message.** The restriction is scoped to the current
  user request; sending a new message to the agent clears it (a fresh
  `use_skill` call re-arms it if the new turn loads a skill again).
- **`list_skills` and `use_skill` are always exempt**, restriction or not - the
  model must still be able to discover and read a skill's files even while
  confined to it.
- **A spawned sub-agent inherits the restriction.** If the coder can delegate to
  a child agent (`spawn_agent`), the child starts with the same narrowed tool
  set the parent had when it was spawned, so a skill cannot escape its own
  declared limit by delegating the disallowed action to a fresh agent.
- **Arming is ordered against the same model reply.** If a reply calls
  `use_skill` alongside other tool calls, `use_skill` runs by itself: anything
  requested before it in that reply finishes first, and nothing requested after
  it starts until the restriction is in force. A skill's declared limit cannot
  be outrun by a tool call batched alongside the load itself.

If the restriction cannot be applied for some reason, the skill is not loaded at
all and `use_skill` reports the failure - a skill whose declared limit cannot be
enforced must not run unrestricted.

## Security

A `SKILL.md` body is **untrusted content**. `list_skills` and `use_skill` only
*read* files, so they never need confirmation - but anything a skill's instructions
prescribe (writing files, running shell commands) still goes through the agent's
normal capability scope, `allowed-tools` enforcement, and destructive-action
confirmation. A skill can therefore *instruct* the agent, but cannot *act* without
your consent, and a skill that declares `allowed-tools` cannot reach outside that
list even with auto-approve on. Treat skills from unknown sources like any other
code you would run, and review them first.
