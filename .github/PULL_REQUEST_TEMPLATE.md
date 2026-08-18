## Summary

<!-- What does this change, and why? -->

## Checklist

- [ ] `python scripts/check_hygiene.py` passes (no em/en dashes, no personal
      disclosure, no machine-absolute paths in code) - see AGENTS.md
- [ ] `CHANGELOG.md` carries RELEASED changes only. localm serves it in-app behind
      the Changelog button, so nothing unreleased and nothing internal goes in it,
      and a change a user cannot see or try does not belong there at all. Note an
      unreleased user-visible change in the internal changelog instead
- [ ] `pytest -m "not integration"` passes
- [ ] `ruff check .` reviewed (no new lint regressions)
- [ ] Defaults are project-relative or user-config; no absolute or
      machine-specific paths (AGENTS.md rule 1)
- [ ] No personal disclosure in tracked files: usernames, email, hostnames,
      secrets, private paths (AGENTS.md rule 2)
- [ ] Docs updated if behaviour, CLI, or the plugin contract changed

## Feature correctness

<!-- Cross-cutting invariants not visible from a diff and not caught by CI.
     Delete rows that don't apply. Detail: docs/plugins.md "Before you ship a
     plugin". (Manifest conformance and client_entry serving are already
     enforced by tests/test_builtin_plugins_contract.py, so they're not listed
     here.) -->

- [ ] New HTTP routes are scope-gated (`host.mount_router` / `require_scope`),
      not mounted on the bare app
- [ ] Session-derived disk writes are gated on `effective_mode()` - privacy mode
      leaves no traces unless explicitly toggled (localm/audit.py)
- [ ] No personal model/encoder/workflow choices hardcoded or committed (tracked
      `*.example.json` + a gitignored override)
- [ ] A new or changed plugin enables and disables without a server restart
- [ ] The behaviour change is covered by a test that fails before the fix
