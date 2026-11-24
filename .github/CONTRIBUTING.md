# Contributing to localm

Bug reports, questions, documentation fixes and code are all welcome. By taking part you agree
to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting and asking

- **Bug, or a model that will not load:** open an [issue](https://github.com/Matlan1/localm/issues/new/choose).
- **Setup failed:** use the "Setup failed" issue form; only one box is required.
- **Feedback, setup or "does this work with my card" questions:** [Discussions](https://github.com/Matlan1/localm/discussions).
- **No GitHub account:** run `localm bug-report -m "..."`, or `report-issue.bat` / `report-issue.sh`
  in the localm folder when localm does not start.
- **Security vulnerability:** follow [SECURITY.md](../SECURITY.md). Do not open a public issue.

## Proposing a change

Small fixes can go straight to a pull request. For a new feature or a larger change, open an
issue or discussion first, so the approach can be agreed before you spend time on it. Writing a
plugin? See [docs/plugins.md](../docs/plugins.md), including its pre-ship checklist.

## Pull requests

[docs/development.md](../docs/development.md) covers setting up a checkout, running the checks and
the rules every change follows.

- Branch from `master` and keep the pull request to one change.
- Say what changed and, when it is not obvious, why. The template lists what else to check.
- Add tests that exercise real behavior. A bug fix needs a test that fails without the fix.
- Update the docs when behavior, the CLI or the plugin contract changes, and add a bullet under
  `## [Unreleased]` in [CHANGELOG.md](../CHANGELOG.md) for a change users will notice.
- New HTTP routes must be scope-gated, and anything a chat session writes to disk must respect
  privacy mode.
- Fix what CI reports before asking for review.

## License

Contributions are licensed under the **GNU Affero General Public License, version 3 or later**
(AGPL-3.0-or-later), like localm itself; see [LICENSE](../LICENSE). There is no separate
contributor license agreement.
