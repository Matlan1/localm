# Contributor and Agent Guide (localm)

This file is binding for every human and every AI agent (Claude, Codex, Cursor,
Gemini, or any other) that edits this repository. Read it fully before writing
any code, docs, comments, or commit messages. `CLAUDE.md` mirrors this file so
Claude Code loads it automatically; the rules are identical.

This repository can become PUBLIC. Treat everything you write as if a stranger
on the internet will read it on another machine.

## Non-negotiable rules

### 1. No absolute or machine-specific paths in defaults

- Never hardcode an absolute path as a default or fallback. That means no
  `D:\...`, no `C:\Users\...`, no `/home/...`, no `/opt/...`, no `/mnt/...`,
  and no path that only exists on one person's disk.
- Resolve paths relative to the code or the data directory: use
  `Path(__file__).resolve().parent`, the repo root, the configured data home,
  `%~dp0` in batch files, or an OS special-folder API. Never assume a specific
  file or folder exists on disk.
- Anything machine-specific (a data directory, a model directory, an external
  tool location, a GPU binary directory) is USER CONFIG. Prompt for it on first
  use, read it from the config file or an environment variable, or derive it
  from the install. If it is not configured, resolve to nothing and tell the
  user how to set it. Do not guess an absolute path.
- A path that works only on the author's machine is both a bug (it breaks for
  everyone else) and a disclosure (rule 2).

### 2. No local or personal disclosure in tracked files

Never commit, in code, docs, comments, config, test data, or compiled
artifacts:

- A local username used as a path component, for example `C:\Users\someone\...`.
- A personal email, real name, hostname, or machine name.
- An API key, token, password, private key, or connection string.
- A private project name or an absolute path to another project on disk.

Allowed and not a violation: the public GitHub organization name in URLs, the
LICENSE copyright line, and clearly illustrative placeholder paths in help text
or examples that a USER would set (for example `D:\path\to\thing` shown as "set
this to your install"). When in doubt, make the example generic.

Keep personal and local files out of git with `.gitignore`: `config.json`,
`registry.json`, `*_local.json` personal workflows, `.venv/`, the native
binaries under `runtime/localm_llama_runtime/lib/`, `home/`, and `.claude/`.
Also keep out the installed-plugins directory (`~/.localm/plugins/`, the
user's own enabled plugins) and any per-plugin local config that overrides a
tracked template: `tts.json` (overrides `tts.example.json`) and
`flux_workflow.json` (overrides `flux_workflow.example.json`). Commit the
`*.example.json` templates, never the personal workflow, voice, or model
choices they stand in for.

### 3. No em-dashes

Do not use the em-dash character (the long dash, Unicode U+2014) anywhere: not
in code comments, docstrings, documentation, README files, or commit messages,
and especially not in public-facing text. Use a comma, a period, a colon,
parentheses, or a spaced hyphen ( - ) instead. The en-dash (U+2013) is also out.
Plain ASCII hyphen-minus only.

### 4. Self-contained

The project depends only on its own uv-managed `.venv` and its own data
directory. It must not rely on sibling folders or anything else on disk that
will not exist on another machine. Native llama.cpp binaries are provisioned
into the venv with `localm setup-llama` and resolved from the
`localm-llama-runtime` wheel or from user config, never from a hardcoded
external folder.

## Protected local paths (NEVER DELETE)

Some directories are local-only working state that is deliberately gitignored.
Gitignored does NOT mean disposable: because git does not track them, a deletion
is NOT recoverable from history. Treat these as read/append-only unless the
maintainer explicitly tells you, in the current session, to remove something.

- `issues/` - the maintainer's working backlog and issue/bug report. It is
  gitignored (local, machine-specific notes), so it will never show in git and
  was already lost once to a "tidy stray files" pass. Never `rm` it, never
  `git clean -x`/`-X` it away, never flag it as a "stray" or "untracked" file to
  remove, and never overwrite `issues/issues.txt` wholesale. You may read it and
  append to it. If you believe anything under a protected path should be removed,
  STOP and ask the maintainer first.
- `qa/` - the by-hand test campaign: the exhaustive feature matrix
  (`qa/FEATURE-MATRIX-2026-06-18.md`), `qa/feature-results.json`,
  `qa/byhand_record.py` (the recorder), and the test plans. Also gitignored and
  local-only - treat it exactly like `issues/` (read/append, never delete or
  `git clean` it). Run `python qa/byhand_record.py status` to see coverage. (The
  maintainer's test-instance logs/bug reports live in `issues/testinstance_home/`,
  NOT here - despite the name, that is bug-report data, not test-campaign data.)

Other gitignored local state (do not delete without being asked): `home/`,
`config.json`, `registry.json`, the installed-plugins dir, and the personal
`*_workflow*.json` / `tts.json` overrides listed under rule 2.

### `issues/` belongs to the maintainer; agent files go in `dev-notes/` (BINDING)

`issues/` (the `ISSUES/` directory) holds ONLY files the maintainer put there by
hand: their backlog, bug reports, logs, and notes. It is their inbox, not agent
workspace. This rule is absolute and global for this repo:

- Every file an agent or contributor creates - worklogs, loop state, `*.md`
  plans, design docs, handoffs, status reports, scratch, audits, any generated
  artifact - lives in `dev-notes/` (or `qa/`, `scratch/`), NEVER in `issues/`.
  Do not write into `issues/` to "keep related things together".
- ONE narrow exception: a todo list or progress report that is specifically
  about working through the maintainer's `issues/` backlog MAY live in `issues/`
  while that work is actively in progress. The moment it stops being relevant
  (the issues are fixed, or the work is parked), MOVE it to `dev-notes/` or
  DELETE it. An agent-authored file must never linger in `issues/` after it has
  served its purpose. The default home is still `dev-notes/`; only reach for this
  exception when keeping the working list next to the issues genuinely helps.
- A file the maintainer created is NEVER deleted and NEVER moved, under any
  circumstance, for any reason, including "cleanup", "tidying", "it looked
  stray", or "it seemed obsolete". Read it and append to it; otherwise leave it
  exactly where it is. If you believe a maintainer-authored file should be moved
  or removed, STOP and ask the maintainer first. The self-cleaning exception
  above applies ONLY to files the agent itself authored, never to anything the
  maintainer wrote.

## How these rules are enforced

Run the hygiene check before you commit:

```
python scripts/check_hygiene.py
```

It scans tracked files for absolute or machine-specific paths, the em-dash
character, and personal identifiers, and exits non-zero on a violation. Wire it
as a pre-commit hook (`scripts/check_hygiene.py --install-hook`) so a commit
that breaks these rules is blocked, not merely discouraged.

If you discover a violation already in git history, do not only fix it forward.
A non-sensitive bad path can be fixed in a normal commit, but a genuine
disclosure in history (a secret, a personal email, or a real-user absolute path
baked into a committed file or binary) requires a history rewrite and a
force-push, and the maintainer must be told before that happens.

## Running a test instance for verification

Agents may launch a local instance of the full app to verify a change in the
real product, not only through the test suite. This is pre-approved; you do not
need to ask first. Examples: `localm gui`, `localm serve`, `localm run <model>`.

Keep it cheap and self-contained:

- Use a SMALL model. A tiny GGUF is enough to smoke-test the UI, model routing,
  token streaming, and the coder loop; do not download a large model just to
  verify a change. `localm gui --no-model` covers checks that need no model at
  all (the GUI opens model-less and you add or switch on the Models page).
- Prefer an isolated, throwaway data directory so the run never touches the
  user's real models, config, or chat history: set the `LOCALM_HOME` environment
  variable to a temporary path for the test instance.
- Bind to localhost only, and stop the instance once the check is done. Do not
  leave a server bound across turns.

## Git and PR workflow

This is a solo-maintained repo, and the maintainer has delegated the full change
cycle: branch, commit, push, open a PR, and MERGE it yourself once it is ready.
Do NOT wait for the maintainer to click merge.

"Ready" means all of:

- the work is verified: tests and `scripts/check_hygiene.py` pass, and where the
  change is observable in the running app it was checked there, not only in the
  suite;
- CI is green; and
- the PR is mergeable (no conflicts).

Then squash-merge and delete the branch.

Guardrails that still apply:

- Never merge a PR with failing CI, merge conflicts, or unfinished or unverified
  work. If it is not actually ready, say what is missing instead of merging.
- Do not stack PRs: land one, then branch the next off the updated `master`.
- Only on the maintainer's own repos.
- A history rewrite or force-push still requires telling the maintainer first
  (see the secret-hygiene section above).
