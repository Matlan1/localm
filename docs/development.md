# Development

How to set up a checkout of localm and check a change. What to report where and what a
pull request needs is in [CONTRIBUTING.md](https://github.com/Matlan1/localm/blob/master/.github/CONTRIBUTING.md).

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
./.uv/uv pip install -p .venv -e ".[dev]"      # Linux / macOS
.uv\uv.exe pip install -p .venv -e ".[dev]"    # Windows
```

Those paths are where setup's default (Portable) answer puts uv; if you chose Shared, plain
`uv` works. Then activate the venv (`source .venv/bin/activate`, or `.venv\Scripts\activate`
on Windows) so `pytest`, `ruff` and `basedpyright` below resolve to it.

localm provisions its own Python (3.12), so the Python on your PATH does not matter.
The other extras and the manual `uv` path are described in the [README](https://github.com/Matlan1/localm/blob/master/README.md#install).

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
basedpyright                        # type check against the committed baseline
npm test                            # only if you changed localm/plugins/gui/static or tests-js
```

`basedpyright` fails on a type error that is not in `.github/basedpyright-baseline.json`, and on
a baseline that still lists errors your change fixed. When it rewrites the baseline, commit
the updated `.github/basedpyright-baseline.json` with your change.

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
