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

Verifying changes: you are pre-approved to launch a local test instance of the
full app (`localm gui`, `localm serve`, `localm run`) to check a change in the
real product, no need to ask. Use a small model (a tiny GGUF, or
`localm gui --no-model`), prefer a throwaway `LOCALM_HOME` so it does not touch
the user's real data, and stop it when done. See `AGENTS.md` for the full text.
