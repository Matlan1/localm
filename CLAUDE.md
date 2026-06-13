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
