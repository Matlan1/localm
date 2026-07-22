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
Also keep out the installed-plugins directory (`<data dir>/plugins/`, the
user's own enabled plugins) and any per-plugin local config file that
overrides a tracked template, such as `flux_workflow.json` (overrides
`flux_workflow.example.json`). Commit the `*.example.json` templates, never
the personal workflow choices they stand in for. A plugin's other personal
settings, such as the tts plugin's voice/model choice, live inside the
already-gitignored `config.json` itself, under `config["plugins"][<name>]`,
not a separate file.

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

### 5. We do not hide problems (root-cause, document, fix)

When something goes wrong - a warning, an error, an unexpected value, a failed
step - the answer is to understand WHY, document it, and fix it. It is NEVER to
silence the message, swallow the exception, or assume "it did not crash, so it is
probably fine". That assumption is how a real fault ships disguised as working
software.

The canonical anti-pattern we will not repeat: the bundled llama.cpp runtime
printed `failed to find ggml_backend_init ...` to stderr, the model loaded anyway,
and the first instinct was to suppress those lines because "it loads fine". That
hid a real question (are the compute backends actually registered, and on which
builds?). The correct fix was to understand the cause and gate on
`ggml_backend_dev_count`, not to mute the symptom. "The car drives fine with two
of five wheel screws missing, so let us hide the warning light" is not engineering.

Why this matters:

- A hidden problem becomes an invisible failure the user hits LATER, with the
  real cause already thrown away, so they cannot diagnose it and neither can we.
- A silenced diagnostic that was actually masking a fault wastes the person's
  debugging time and burns trust in the tool.
- For a PRIVACY or SECURITY operation (stripping metadata, scrubbing history,
  redacting a path, writing an audit record, enforcing auth), a silent failure is
  the worst kind: the user is told a safety property held when it did not. A
  privacy/security step that fails must NEVER report success - fail safe and say so.

The only thing you may ignore is a warning PROVEN harmless - genuinely cosmetic,
or a deliberate documented fallback - and only when that proof is written AT THE
SITE as a one-line why-comment. "Proven" means you found the cause and can state
why it does not matter; it does not mean "it seemed fine in one run".

The right altitude (this is not a license to over-engineer):

- Best-effort cleanup that genuinely does not matter on failure (deleting a temp
  file, clearing a remote queue entry) is fine - but still say WHY in a comment,
  and prefer a debug-level log over total silence so a failure is discoverable.
- When a swallow could hide a REAL failure, the fix is usually to SURFACE it (a
  debug/WARNING line, a warning folded into the returned message, a recorded
  error) - NOT to silence it, and NOT to escalate a legitimate best-effort path
  into a hard failure that breaks working setups. A note or a log is almost always
  the right altitude; failing the whole operation usually is not.
- Distinguish the intended benign case from the unexpected one: a default that is
  safe when a file is simply absent may be unsafe when the file exists but is
  unreadable. Branch on it; do not collapse "missing" and "corrupt" into one
  silent path.

There is no automated linter for this (a blanket "no `except: pass`" rule would
flag 150+ legitimate sites and train people to ignore it). It is a code-review
discipline, periodically reinforced by a codebase honesty audit (see
`dev-notes/` for the latest). When you touch code that swallows something, leave
it better: either surface the real failure or document why it is safe.

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
`*_workflow*.json` override (e.g. `flux_workflow.json`) listed under rule 2.

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

The same command also runs the release-file manifest gate
(`scripts/check_manifest.py`, from `release-manifest.toml`): every tracked file
must be classified release-include (ships in a release build.zip / self-update)
or release-exclude (tracked but dev-only), nothing declared local-only may be
committed, and no manifest pattern may go stale. So adding a new top-level file
or directory means classifying it in `release-manifest.toml`, or the gate fails.
`scripts/build_release.py` assembles the build.zip from the same release-include
list (then `scripts/sign_release.py` signs it).

### The release changelog is append-only

`CHANGELOG.md` is the permanent public record of what shipped. Only the PUBLISHED,
versioned sections (`## [x.y.z]`) are frozen: once a version's entry is published it
stays, as written, as the record of what that version shipped. Do not "tidy",
condense, re-summarize, delete, or reword a published section - that is rewriting
history, and it is a hard no unless the maintainer explicitly asks for it (typo and
formatting corrections aside). New releases ADD their section at the top (newest
first), above the prior ones, never in place of them.

The `## [Unreleased]` section (and any intro text before the first version header)
is the IN-PROGRESS draft, not history yet: rewrite, reorder, expand, or trim it
freely as the pending release takes shape. It becomes frozen only when it is cut
into a version at release time.

This is enforced, not merely asked. The same `check_hygiene.py` pass diffs the
working `CHANGELOG.md` against the published-record baseline (the merge-base with
`origin/master`, else the last commit) and fails if any PUBLISHED entry line was
removed or rewritten - INCLUDING a published section's own `## [x.y.z]` version
header (its number and ship date) and any `### Added`-style subsection header
within it, not just its bullet entries. Only lines under `## [Unreleased]` (its own
header, and any intro text before the first version header) and the link-reference
definitions at the bottom are exempt (cutting a release legitimately renames the
`[Unreleased]` header to a version and updates the compare link). The release
tooling honors the same invariant: `scripts/make_release.py` / `build_release.py`
never generate or rewrite the changelog - it is hand-maintained and shipped verbatim.

That `[Unreleased]` exemption has one blind spot, so the same pass also WARNS (it
does not fail) about two things: a draft line that existed at the baseline and is
missing from your working copy (a DROP), and a draft bullet that appears more often
than it did at the baseline, and more than once (a DUPLICATE). When several branches
each add draft bullets, a sibling branch's bullet can vanish around a rebase with
every mechanical check still reporting clean, and a landed PR's entry is simply
gone from the release notes. The duplicate half catches the botched remedy: a drop
check run against the moving `origin/master` ref (rather than the merge-base)
flags every bullet merged after your branch point as yours to restore, and
restoring one that was never lost leaves it in twice.

The warnings list every affected line verbatim so you can tell your own edit from
a bullet you lost. Attribute a line before acting on it
(`git log -S "<line>" -- CHANGELOG.md`), restore only what you actually dropped,
delete only the EXTRA copy of a duplicate, and never reset the section to master
or hand-copy a bullet back in blind. Rewording a draft line in place is reported
too (matching is exact, so that a near-match heuristic can never suppress the real
case); that is the intended cost of a warning you can read and dismiss. Warnings
never change the exit code unless you ask: `python scripts/check_hygiene.py
--strict`, or `LOCALM_HYGIENE_STRICT=1`, turns every warning into a failure.

On the cause, because a wrong story sends people hunting the wrong thing: it is
NOT every rebase that replays an `[Unreleased]` insertion (that was falsified by
direct comparison). What is evidenced is a CONFLICTED rebase resolved
bulk-take-mine; resolve those additively, keeping both sides' bullets.

If you discover a violation already in git history, do not only fix it forward.
A non-sensitive bad path can be fixed in a normal commit, but a genuine
disclosure in history (a secret, a personal email, or a real-user absolute path
baked into a committed file or binary) requires a history rewrite and a
force-push, and the maintainer must be told before that happens.

## Understand what localm is before you decide something needs the maintainer

localm is an offline local-LLM inference and plugin engine: it DOWNLOADS and RUNS
local models itself. So an agent can verify almost anything end to end without the
maintainer. Before you write "only the maintainer can test/do this", check whether
localm's own capabilities already cover it:

- `localm pull owner/repo:model.gguf` downloads a GGUF from HuggingFace; `localm
  run` / `localm serve` / `localm gui` runs it; `localm gui --no-model` runs the
  app with no model at all.
- `localm setup-embeddings` installs the small on-device embedding model, so
  semantic memory and RAG can be exercised for real, not just their lexical
  fallback.
- The coder plugin has file, shell, search, and test tools and speaks MCP both
  ways. Chat is the only always-on plugin; everything else is one you install.

So "needs a model to test", "needs someone to run the app", or "needs the coder to
try it" are NOT maintainer-only: pull a tiny model (or use `--no-model`) in a
throwaway `LOCALM_HOME` and check it yourself (see the next section). Treat any
"maintainer-only" conclusion as a claim you have to justify, not a default.

Genuinely maintainer-only is a short list: deploying or holding an external secret
(a server or worker deploy, an update token), cutting a public release, and
verifying on hardware nobody here has (for example a specific NVIDIA, Intel, or
macOS box). Almost everything else, an agent can do by understanding the project
and running it.

The deeper point: verifying or building without actually understanding what localm
is produces false confidence (shipping broken work under a "done" label) or false
deferral (stalling on something the tool already does). Real understanding is the
precondition for the "we do not hide problems" and verify-before-done rules above,
not an optional extra.

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
- **Eat the dogfood (Live MCP verification)**: When modifying the server, the CLI, the coder, or the API endpoints, always verify that the MCP server runs correctly with those changes. Run the latest version + your unpublished changes using `npx mcporter` or an MCP client over stdio, and call tools (e.g. `list_models`, `chat`) to guarantee that any bugs or integration issues are caught live.

## Test-run cadence: full suite once, right before the PR

Run the FULL suite (`pytest -m "not integration"` + `npm test`) once, as the gate
immediately before opening or updating a PR - not after every small edit, and not
once per individual unit of a larger task. The full suite takes minutes; CI runs
it again on the PR anyway, so re-running it after every small change buys no
extra signal for real cost.

The Python suite runs in parallel via `pytest-xdist`: add `-n auto` to distribute
tests across worker processes (`pytest -m "not integration" -n auto`). Every test
already gets its own `LOCALM_HOME` via the autouse `tmp_path` fixture in
`tests/conftest.py`, and the few tests that bind a real socket use an OS-assigned
ephemeral port, so this is safe. Not forced on by default (no `addopts`), so a
plain `pytest -k foo -s` or a debugger session still works without worker
overhead or interleaved output. CI runs the full suite with `-n auto`.

While iterating within a task:

- Use a targeted/scoped check against just the touched area for fast feedback per
  step: `pytest -k <substr>`, the single changed test file, `ruff check <file>`.
- `python scripts/check_hygiene.py` is fast (no pytest) and fine to run often.
- When a task naturally decomposes into several small units, batch them into one
  PR rather than opening a PR (and burning a full-suite run) per unit. One PR
  gets ONE full-suite pass before it opens; CI re-confirms it.

This changes cadence, not the bar: the full suite is still mandatory, green, and
non-negotiable before a PR opens or merges (see the "Ready" checklist below and
`.github/PULL_REQUEST_TEMPLATE.md`).

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

Then squash-merge and delete the branch. This repo is usually worked in git
worktrees, with `master` checked out in one of them. Git refuses to check out a
branch that is already active in another worktree, so `gh pr merge
--delete-branch` and `git checkout master` both fail here with `fatal: 'master'
is already used by worktree ...` (the API merge still lands, but the local
cleanup errors and leaves the remote branch undeleted). This repo also has
`delete_branch_on_merge` enabled, so the remote branch is auto-deleted by the
merge; running `git push origin --delete` yourself then errors with `remote ref
does not exist`. Merge and clean up WITHOUT ever checking out master and WITHOUT
an unconditional remote delete, from the worktree that is on the PR branch:

```
gh pr merge <N> --squash    # squash-merge; GitHub auto-deletes the remote branch
git switch --detach         # step off the PR branch without checking out master
git branch -D <branch>      # delete the local branch (no worktree collision)
```

Confirm with `git ls-remote --heads origin <branch>` (empty means the remote
branch is gone) and `git branch --list <branch>` (empty means the local branch
is gone); verify the merge landed via `git fetch origin master && git log
origin/master -1`, not a local checkout. Only if `delete_branch_on_merge` is
ever turned off, delete the remote branch guarded so it never errors:
`git ls-remote --exit-code --heads origin <branch> >/dev/null 2>&1 && git push
origin --delete <branch>`.

Keep `master` in the main checkout, not in a worktree. The top-level clone stays
checked out on `master` at all times, and feature work happens in worktrees on
their own branches. Do not check `master` out into a worktree, and do not switch
the main clone off `master`: that is what pins `master` in a side worktree and
triggers the merge error above. If you find `master` checked out somewhere else,
free it (detach the clean squatting worktree, never one with uncommitted work)
and restore it in the main clone. Only relocate `master` for a compelling,
deliberate reason.

Guardrails that still apply:

- Never merge a PR with failing CI, merge conflicts, or unfinished or unverified
  work. If it is not actually ready, say what is missing instead of merging.
- Do not stack PRs: land one, then branch the next off the updated `master`.
- Only on the maintainer's own repos.
- A history rewrite or force-push still requires telling the maintainer first
  (see the secret-hygiene section above).
