# CodeQL residue tooling

Dev-only tooling for reasoning about CodeQL code-scanning alerts on this repo.
Not shipped (classified release-exclude in `release-manifest.toml`).

## Why counting open alerts does not work here

Three measured failures, all on this codebase:

1. **A correct fix routinely leaves its alert OPEN.** CodeQL does not model this
   repo's confinement patterns as barriers. Taint was traced straight through a
   `re.sub()` rewrite, through `resolve()` + `is_relative_to()` WITH a raise, and
   through a parent-equality re-check WITH a raise. So "the alert is still open"
   is not evidence the code is unfixed.
2. **Line shifts masquerade as churn.** A fix that moves lines makes CodeQL retire
   an alert and raise a new one. A raw `(rule, file, line)` diff reported
   **79 gone / 85 new** between two analyses where the fingerprint diff reported
   **10 / 16** - about 69 were pure noise.
3. **A good fix can ADD alerts.** A shared helper introduced to centralise
   confinement becomes a sink in its own right.

So alert COUNT is not a measure of progress in either direction.

## `residue_check.py`

Classifies alerts between two SARIF analyses by evidence rather than count.

    python residue_check.py --baseline <sarif> --current <sarif> \
        [--barriers b.json] [--repo-root .]

Keys alerts on `(ruleId, file, primaryLocationLineHash)` - GitHub's own
fingerprint - so a line move does not read as a change. Reports RESOLVED / MOVED
/ SURVIVED / ADDED, and for survivors reports whether every code flow passes
through a declared barrier.

Get a SARIF with:

    gh api repos/<owner>/<repo>/code-scanning/analyses/<id> \
        -H "Accept: application/sarif+json" > sarif.json

(The REST alerts API returns `code_flows` EMPTY; only the SARIF has them.)

**Known range.** The fingerprint hashes line CONTENT, so it dissolves pure moves
but NOT an edit to the sink line - that legitimately reads as ADDED. Cross-FILE
moves are detected separately by re-matching on `(ruleId, lineHash)`.

## `sweep_dismissals.py`

**Run after every merge.**

    python sweep_dismissals.py            # check
    python sweep_dismissals.py --restore  # re-dismiss what evaporated

A dismissal attaches to an alert NUMBER. When a later merge edits that alert's
sink line, CodeQL retires the alert and raises a new one **carrying no
dismissal** - so a previously-dismissed false positive silently reappears as
open. No test fails, no check reports it, CI stays green, and the queue grows
with no explanation.

This is not hypothetical: on 2026-07-29 a merge editing `cli/models.py:333` for
an unrelated correctness fix silently un-dismissed three alerts.

**A merged security fix can un-dismiss a false positive it had nothing to do
with.** After a renumbering the link between the old and new alert is NOT
recoverable from the API, so `dismissed_fingerprints.json` records each
dismissal's `(rule, lineHash, justification)` in advance. The script re-captures
on every run, because a record that goes stale protects nothing.

## How to read a dismissal comment

Dismissal comments on this repo's alerts cite the triage record, which is a
point-in-time working document kept in `dev-notes/` and **not in git**. The
substantive justifications are reproduced below so a dismissal can be
understood without it.

Every group below asserts **the alert is WRONG**, not that a real risk was
accepted. There is no "won't fix" tranche.

| Group | Why the alert is a false positive |
|---|---|
| jobs store | The job id is rewritten by `_ID_RE.sub` (`..` collapses to `job`), then `_confine()` (resolve + `is_relative_to`), then an additional `parent == results_root` assert. Three independent barriers. |
| coder session download | The caller's path must be a member of `session.changed_files()` AND pass resolved-path confinement to the session root. The resolved test also defeats symlinks. |
| tts library/wasm paths | Fields are `admin_only=True` AND pass `_tts_relative_asset`, which rejects `:`, leading `/` or `\`, backslashes, `..` and empty segments, then requires containment in the plugin asset root. |
| comfy launcher | All sinks are downstream of an EXACT-MATCH allowlist of resolved configured workdirs. Exact-set membership is stronger than path confinement. |
| install-external | Naming the source directory IS the endpoint's purpose. `PLUGINS_ADMIN` is privileged and already grants installing third-party code localm imports and runs, so constraining which directory restores no boundary. |
| coder project cwd | Reachable only by an owner or `coder:full` principal, which already holds `run_shell`/`write_file`, i.e. arbitrary code execution as the server user. |
| `localm rm` prompt | Three read-only stat calls whose only output is a `click.confirm` string on the local user's own terminal. Nothing read, written or deleted; `remove_model` re-derives its own gate. |
| LOCALM_HOME / LOCALM_DEBUG | Reads localm's OWN process environment. Only the principal that LAUNCHES the process can set it, which already implies control of the process at the same OS privilege. |
| release tooling argv | The operator's own argv for the release-signing key. No CI path reaches it, and a wrong key cannot forge a release - the build is re-verified against a pinned public key. |
| HIP_PATH | localm's own process environment, reaching only an `isdir()` probe. Never written from config or any downloaded artifact. |
| coder `$HISTFILE` | Comes from the shell that launched `localm coder`; anyone able to set it can already execute code as that user. Rewriting the history file at the shell's own location IS the privacy feature. |
| `/debug/stacks` | Returning stacks IS the feature and the residual flow is by design. The boundary-crossing half was FIXED: it was unauthenticated in default keyless mode, now requires a credential and 404s off loopback, and absolute paths in emitted frames went from 32/63 to 0/65. |

## When an alert may be called fixed-but-not-closed

All three, or it stays counted as open:

1. the fix is DEMONSTRATED (a test that goes red pre-fix), not argued;
2. every code flow is shown to pass through the barrier, **from the SARIF, per
   alert**, not per file;
3. the barrier is NAMED, with the reason CodeQL does not model it.

`residue_check.py` mechanises only (2), and it does **not** verify that a guard
DOMINATES its sink (runs first on every path with no branch around it). Nothing
in this tooling checks that; a human reading the diff does.

Condition 2 is **not applicable** - not merely unmet - for three shapes: an
INLINE guard in the sink's own function, a CHOKE POINT that cannot barrier
itself, and an EXTRACTED but NON-TRANSFORMING guard that returns no value (the
tainted value passes to it, not through it). Hence the design rule that is worth
more than the tooling: **the sink must consume the barrier's RETURN VALUE.**
Transform, do not merely validate.

## `dispositions.json` + `apply_dispositions.py`

The adjudication of every open alert, and the script that posts it.

    python apply_dispositions.py            # report only, changes nothing
    python apply_dispositions.py --apply    # post the dismissals

Keyed on `(file, enclosing function)`, parsed from **the commit the alert was
raised against** (`git show <sha>:<path>`), never from the working tree. Both
halves of that matter and both were learned by getting them wrong:

- an alert NUMBER and a LINE both move when a fix edits a sink line, so a
  dismissal keyed on either silently detaches from what it was about;
- a function name survives that, but only if it is read from the revision the
  line number belongs to. Resolving against whatever happens to be checked out
  misfiled 8 of 125 on this script's first run, every one of them plausibly.

An alert whose function is not in the table is **never** dismissed. It is
reported as UNADJUDICATED and the exit code is non-zero. A tool that closed
whatever it did not recognise would defeat the point, which is that every closed
alert had someone look at it.

Each group's text is the justification posted to the alert, so it has to be true
of every alert in that function. A function holding sinks of two kinds gets a
comment covering both rather than the tidier of the two labels. Two groups are
filed `won't fix` rather than `false positive` on purpose: for the
authorization-gated routes and the `LLAMA_CPP_LIB` loader the taint flow is
REAL, and only the exposure is absent. Calling those "barriered" would file a
genuine finding under a label that hides it.

## Why there is no models-as-data barrier pack

The obvious idea is to declare this repo's confinement helpers as barriers via
CodeQL's supported models-as-data extension, so correct code stops being
reported. It works, and it is not worth it here. Measured offline against
`9fde2813`:

| declared barrier | path-injection alerts |
|---|---|
| none (baseline) | 168 |
| `_entry_path` | 167 |
| + `_check_plugin_name`, `confined_name/_file/_under` | 163 |

**5 of 168**, and the 5 are among the safest in the set. The reason is a real
limitation worth writing down: a models-as-data model resolves through the API
graph, which reaches a function through an IMPORT. `registry.py:547` calls
`_entry_path` in its own module, so the graph never reaches it and the declared
barrier is simply not applied. This repo's guards are overwhelmingly same-module
calls or inline `if ...: raise`, and neither shape is expressible.

So the mechanism is sound and the codebase is the wrong shape for it. Recorded
because "add a sanitizer pack" is the first thing anyone proposes here, and the
answer is now measured rather than argued.

## This tooling is a specimen of its own third finding

`residue_check.py` raises `py/path-injection` alerts on itself, and the count
went **4 -> 6** across two rounds of trying to fix them. Both rounds found a real
defect (a path component from a data file rather than argv; and resolving before
validating, when `resolve()` performs filesystem I/O and a declared
UNC path would dial SMB). Both are fixed. **The alert count still rose**,
because each guard added is itself a path expression on tainted data and becomes
a new sink.

That is the third bullet at the top of this file, observed on the file that
states it. Checking containment requires constructing the path first, so the
construction is always the flagged sink and the guard is always after it - a
validate-and-raise, which CodeQL walks straight through.

**The lesson is the one this whole directory exists to make operational: do not
drive the alert count to zero. Drive the code to correct, then adjudicate what
remains with evidence.** A patch loop against the count will keep finding work
long after it has stopped finding vulnerabilities.
