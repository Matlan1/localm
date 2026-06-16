## Summary

<!-- What does this change, and why? -->

## Checklist

- [ ] `python scripts/check_hygiene.py` passes (no em/en dashes, no personal
      disclosure, no machine-absolute paths in code) - see AGENTS.md
- [ ] `pytest -m "not integration"` passes
- [ ] `ruff check .` reviewed (no new lint regressions)
- [ ] Defaults are project-relative or user-config; no absolute or
      machine-specific paths (AGENTS.md rule 1)
- [ ] No personal disclosure in tracked files: usernames, email, hostnames,
      secrets, private paths (AGENTS.md rule 2)
- [ ] Docs updated if behaviour, CLI, or the plugin contract changed
