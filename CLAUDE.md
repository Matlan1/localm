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

Enforce with `python scripts/check_hygiene.py` before committing.

PROTECTED LOCAL PATHS (never delete): `issues/` is the maintainer's gitignored,
local-only backlog/bug report. It is NOT in git, so deleting it is unrecoverable
(it was lost once already). Never `rm` it, `git clean -x` it, treat it as a
"stray/untracked" file to tidy, or overwrite `issues/issues.txt` wholesale. Read
and append only; ask before removing anything under it. See AGENTS.md
"Protected local paths".

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
