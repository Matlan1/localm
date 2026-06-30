# CLAUDE.md

Claude Code loads this file automatically. The binding rules for every agent and
contributor live in `AGENTS.md` and apply in full here. Read `AGENTS.md` before
editing anything.

The four rules, in short (see `AGENTS.md` for the full text and rationale):

1. No absolute or machine-specific paths in defaults. Project-relative or
   user-config only. Never assume a file or folder exists on disk.
2. No local or personal disclosure in tracked files: no usernames in paths, no
   personal email or hostname, no secrets, no private project paths. This repo
   can become public.
3. No em-dashes (U+2014) or en-dashes (U+2013) in any file you write. Use ASCII
   hyphen-minus, commas, periods, or parentheses.
4. Self-contained: depend only on this venv and this data dir, never a sibling
   folder on disk.
5. We do not hide problems. Root-cause, document, fix. Never silence a warning,
   swallow an error, or assume "it did not crash, so it is fine". Ignore a warning
   ONLY if it is proven harmless (cosmetic / a documented fallback) AND that proof
   is written at the site as a why-comment. A privacy or security step that fails
   must NEVER report success. Surface real failures (a debug/WARNING line, a
   returned warning) rather than muting them; but a note/log is the right altitude,
   not escalating a legitimate best-effort path into a hard failure. See AGENTS.md
   "We do not hide problems".

Enforce with `python scripts/check_hygiene.py` before committing.

PROTECTED LOCAL PATHS (never delete): `issues/` (the maintainer's gitignored,
local-only backlog/bug report) and `qa/` (the gitignored, local-only by-hand test
campaign: the feature matrix, results, recorder, and test plans). Neither is in
git, so deleting either is unrecoverable (`issues/` was lost once already). Never
`rm` them, `git clean -x` them, treat them as "stray/untracked" files to tidy, or
overwrite `issues/issues.txt` wholesale. Read and append only; ask before removing
anything under them. See AGENTS.md "Protected local paths".

ISSUES IS THE MAINTAINER'S, NOT AGENT WORKSPACE (BINDING): `issues/` holds ONLY
files the maintainer put there. Every agent-created file (worklogs, loop md,
plans, design docs, handoffs, status reports, scratch, any generated artifact)
goes in `dev-notes/` (or `qa/`, `scratch/`), NEVER in `issues/`. The one
exception: a todo list / progress report specifically for working through the
`issues/` backlog may sit in `issues/` while that work is active, but MUST be
moved to `dev-notes/` or deleted the moment it is no longer relevant - never let
an agent file linger there. A maintainer-created file is NEVER deleted or moved,
for any reason; read/append only, and ask first if you think one should go. See
AGENTS.md "`issues/` belongs to the maintainer".

Verifying changes: you are pre-approved to launch a local test instance of the
full app (`localm gui`, `localm serve`, `localm run`) to check a change in the
real product, no need to ask. Use a small model (a tiny GGUF, or
`localm gui --no-model`), prefer a throwaway `LOCALM_HOME` so it does not touch
the user's real data, and stop it when done. See `AGENTS.md` for the full text.

Merging PRs: this is a solo-maintained repo and you are pre-approved to MERGE
your own PRs once ready (verified + CI green + mergeable), squash-merging and
deleting the branch. Do not wait for the maintainer to merge. Never merge
failing, conflicted, or unverified work; do not stack PRs. See `AGENTS.md`
"Git and PR workflow".
