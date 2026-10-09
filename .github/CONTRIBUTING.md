# Contributing to localm

Thanks for helping. This page covers setup, checks, and what a good pull request
looks like. By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Where to go first

- **Bug or crash:** open an [issue](https://github.com/Matlan1/localm/issues/new/choose),
  or run `localm bug-report -m "..."` to attach a safe hardware and backend snapshot.
- **Model that will not load or is not recognised:** use the "Model support" issue form.
- **Setup or "does this work with my card" questions:** [Discussions](https://github.com/Matlan1/localm/discussions).
- **Security vulnerability:** follow [SECURITY.md](../SECURITY.md). Do not open a public issue.
- **Larger change or new feature:** open an issue or discussion first so the approach
  can be agreed before you spend time on it.

## Set up a development checkout

```bash
git clone https://github.com/Matlan1/localm.git
cd localm
bash setup.sh          # Linux / macOS
setup.bat              # Windows
```

Setup creates a private `.venv` in the clone. Add the contributor tooling (ruff, pytest,
xdist, coverage) to it:

```bash
uv pip install -p .venv -e ".[dev]"
```

localm provisions its own Python (3.12), so the Python on your PATH does not matter.
The other extras and the manual `uv` path are described in the [README](../README.md#install).

For changes to the web GUI, also run `npm ci` once; the JavaScript tests use it.

## Run the checks

Run the tests for the area you changed rather than the whole suite:

```bash
pytest tests/test_some_module.py -m "not integration"
pytest tests/test_some_module.py -k some_case -m "not integration"
```

`python scripts/affected_tests.py --why` lists the test files affected by your committed
changes. When a change touches a module most tests import, it reports the selection as too
wide for a targeted run; the pull request checks run those in full.

Before you push:

```bash
python scripts/check_hygiene.py     # repository hygiene rules, see below
ruff check .                        # lint
npm test                            # only if you changed localm/plugins/gui/static or tests-js
```

`python scripts/check_hygiene.py --install-hook` installs it as a pre-commit hook. A warning that
the release manifest checker is missing is expected in a public clone.

Tests should exercise real behavior, including edge and error cases, rather than mocking the
code under test. A bug fix needs a test that fails without the fix.

## Rules every change follows

`check_hygiene.py` enforces the first three:

1. **No em-dashes or en-dashes** (U+2014, U+2013) in any file. Use a hyphen, comma, colon,
   period, or parentheses.
2. **No absolute or machine-specific paths** in defaults or code. Paths are project-relative or
   come from user configuration. An obviously fake placeholder such as `/path/to/model.gguf`
   in documentation is fine.
3. **No personal disclosure** in tracked files: usernames, hostnames, personal emails, secrets,
   private paths. This repository is public.
4. **Do not hide problems.** Do not silence a warning or swallow an error unless it is
   harmless, and a privacy or security step that fails must never report success.
5. **Comments describe what the code does.** Put background and rationale in the pull request
   description, not in code comments.

## Pull requests

- Branch from `master` and keep the pull request to one change.
- Fill in the pull request template: a short summary of what changed and, when it is not
  obvious, why.
- Update the docs when behavior, the CLI, or the plugin contract changes.
- A user-visible change gets a bullet under `## [Unreleased]` in [CHANGELOG.md](../CHANGELOG.md).
  Internal changes (tests, tooling, refactors, CI) do not.
- New HTTP routes must be scope-gated, and anything written to disk from a chat session must respect
  privacy mode. The checklist in the template lists the other cross-cutting expectations.
- CI runs lint, hygiene, and the affected tests on every pull request. Fix what it reports before
  asking for review.

Writing a plugin? See [docs/plugins.md](../docs/plugins.md), including its pre-ship checklist.

## License

localm is licensed under the **GNU Affero General Public License, version 3 or later**
(AGPL-3.0-or-later); see [LICENSE](../LICENSE) and the License section of the
[README](../README.md#license). The repository does not contain a separate contributor license
agreement.
