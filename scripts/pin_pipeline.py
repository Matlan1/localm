#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detect, locally confirm, bump, and merge a newer pinned external build.

Phase 1 (this file): the llama.cpp pin (``setup_llama._PINNED_TAG``) only.
``--pin comfyui``/``--pin rocm`` are reserved for later phases and refuse for
now, rather than silently doing nothing.

WHAT THIS DOES, per run:
  1. Reads the currency check (scripts/check_llama_pin.py, imported directly for
     structured data - never text-scraped) for the newest upstream tag.
  2. Consults a persisted, gitignored state file so a candidate already recorded
     FAIL is never retried without a newer one appearing, and a recorded
     INCONCLUSIVE only retries after a cooldown.
  3. On an eligible candidate: runs scripts/confirm_llama_runtime.py for real
     (cpu + vulkan) from the main checkout - read-only, it downloads to a
     throwaway directory and never touches a tracked file - held under the
     existing GPU lease for the one shared GPU.
  4. On a PASS receipt, EVERYTHING that writes anything happens inside this
     script's OWN dedicated worktree, never the shared main checkout: bump
     (scripts/bump_llama_pin.py --write), a templated CHANGELOG bullet, the
     pin's own targeted tests (run against the WORKTREE's tree via an
     explicit PYTHONPATH - a bare subprocess call from a worktree silently
     imports the MAIN checkout's localm otherwise), then commit/push/PR/merge
     - polling real per-check conclusions, never mergeable/mergeStateStatus.
  5. On FAIL/INCONCLUSIVE, or any refusal after that point: no PR is opened,
     the state file records why, and a genuine FAIL is also appended to
     issues/issues.txt so a human or a later session actually sees it.

WHAT THIS NEVER DOES: touch the shared main checkout's branch or working tree
(worktree isolation is checked, not assumed, and nothing in the write path
ever runs with the main checkout as its target), retry a build already
proven bad, or merge without the same automated checks a human-authored PR
would need.

Exit codes: 0 nothing to do / merged successfully; 1 FAIL (bad build, refused
bump, failing tests, red CI); 2 INCONCLUSIVE (could not measure this run).

Usage:
    python scripts/pin_pipeline.py --pin llama
    python scripts/pin_pipeline.py --pin llama --dry-run   # confirm only, never bump/merge
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE_DIR = REPO / "dev-notes" / "pin-pipeline"
ISSUES_PATH = REPO / "issues" / "issues.txt"

# The shared, board-wide GPU-mutex tool this box already has - not part of
# this repo (it coordinates every project on the machine, not just localm).
# LOCALM_PIN_PIPELINE_GPU_LEASE overrides the guessed location; the guess is
# checked to exist before being trusted.
_GPU_LEASE_ENV = "LOCALM_PIN_PIPELINE_GPU_LEASE"
_GPU_LEASE_GUESS = REPO.parent / ".claude" / "gpu_lease.py"

# How long an INCONCLUSIVE verdict (network blip, could not measure) is
# trusted before the SAME candidate is worth trying again. A FAIL (the build
# is actually bad) has no cooldown at all - it is never retried without a
# newer candidate appearing.
INCONCLUSIVE_COOLDOWN_HOURS = 24

# How long to wait for a merged branch's CI checks before giving up for this
# run (leaving the PR open, never force-merging, never abandoning it).
CI_WAIT_TIMEOUT_SECONDS = 45 * 60
CI_POLL_INTERVAL_SECONDS = 20

REQUIRE_BACKENDS = ("cpu", "vulkan")

# gpu_lease.py's own EX_TEMPFAIL - the card was busy and the wait elapsed.
# This is NOT the confirm script's own verdict: it means confirm never ran at
# all, so the candidate must not be recorded as tried.
LEASE_BUSY_EXIT = 75


class PipelineError(Exception):
    """Something refused the pipeline; the message says why. Distinguished
    from a build-quality FAIL - this is an internal-consistency problem
    worth a human's attention regardless of build quality. Recorded as a
    permanent FAIL (never auto-retried without a newer upstream tag) and
    logged to issues.txt."""


class InfraError(PipelineError):
    """A PipelineError specifically about THIS RUN's git/network/gh-CLI
    plumbing (a fetch, a push, `gh pr create`/`merge`) - never about the
    confirmed candidate build's own quality, which already passed real
    hardware confirm before any of these steps run. Recorded as INCONCLUSIVE
    (cooldown retry, never logged to issues.txt) rather than FAIL."""


def gpu_lease_script() -> Path:
    """The GPU lease script's path, verified to exist. Raises PipelineError
    rather than running without lease coordination."""
    configured = os.environ.get(_GPU_LEASE_ENV)
    path = Path(configured) if configured else _GPU_LEASE_GUESS
    if not path.is_file():
        raise PipelineError(
            f"GPU lease script not found at {path} (set {_GPU_LEASE_ENV} to its real "
            "location). Refusing to run a GPU-touching backend without lease coordination.")
    return path


# --------------------------------------------------------------------------- #
#  Currency (imported directly for structured data, never text-scraped)      #
# --------------------------------------------------------------------------- #

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def newest_candidate(repo: Path = REPO) -> "tuple[str, str] | None":
    """(pinned tag, newest upstream tag) when upstream has something newer
    than the pin, else None (already current or upstream unreadable - both
    read the same here: nothing eligible to test this run)."""
    check_llama_pin = _load_module(repo / "scripts" / "check_llama_pin.py", "check_llama_pin")
    pin = check_llama_pin.pinned_tag()
    releases, err = check_llama_pin.upstream_releases()
    if err or not releases:
        print(f"currency: could not read upstream releases ({err or 'empty'}); nothing to do")
        return None
    pin_n = check_llama_pin._build_number(pin)
    newest = releases[0]["tag"]
    newest_n = check_llama_pin._build_number(newest)
    if pin_n is None or newest_n is None or newest_n <= pin_n:
        print(f"currency: {pin} is already current (upstream newest: {newest})")
        return None
    print(f"currency: {pin} -> candidate {newest}")
    return pin, newest


# --------------------------------------------------------------------------- #
#  State (gitignored, main-checkout-only - never retry a known-bad candidate) #
# --------------------------------------------------------------------------- #

def _state_path() -> Path:
    return STATE_DIR / "llama-state.json"


def load_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _parse_iso(value: str) -> "_dt.datetime | None":
    try:
        return _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def should_skip(state: dict, candidate: str, *, now: "_dt.datetime | None" = None) -> "str | None":
    """A reason to skip *candidate* this run, or None to proceed.

    A FAIL is never retried while it remains the newest candidate: only a
    NEWER tag appearing clears it. An INCONCLUSIVE is retried once the
    cooldown elapses - it is evidence we could not measure, never evidence
    the build is bad."""
    if state.get("last_tag_tried") != candidate:
        return None
    verdict = state.get("verdict")
    if verdict == "FAIL":
        return f"{candidate} already recorded FAIL; no newer candidate has appeared since"
    if verdict == "INCONCLUSIVE":
        tried_at = _parse_iso(state.get("timestamp", ""))
        now = now or _dt.datetime.now(_dt.timezone.utc)
        if tried_at is not None and (now - tried_at) < _dt.timedelta(hours=INCONCLUSIVE_COOLDOWN_HOURS):
            return (f"{candidate} was INCONCLUSIVE {tried_at.isoformat()}; "
                    f"cooldown ({INCONCLUSIVE_COOLDOWN_HOURS}h) has not elapsed")
        return None
    return None


# --------------------------------------------------------------------------- #
#  Confirm (read-only against the main checkout - downloads to a throwaway   #
#  dir, never writes a tracked file, so it never needs the dedicated worktree)#
# --------------------------------------------------------------------------- #

def run_under_gpu_lease(cmd: "list[str]", *, purpose: str, cwd: Path) -> int:
    """Run *cmd* holding the shared GPU lease for its whole duration. Returns
    the WRAPPED command's own exit code on success; LEASE_BUSY_EXIT if the
    lease itself could not be acquired (the command never ran)."""
    lease_cmd = [sys.executable, str(gpu_lease_script()), "run", "--purpose", purpose, "--", *cmd]
    return subprocess.run(lease_cmd, cwd=cwd).returncode


def run_confirm(candidate: str, receipt_path: Path) -> int:
    """scripts/confirm_llama_runtime.py --tag candidate --backend cpu --backend
    vulkan --receipt receipt_path, from the MAIN checkout (read-only: it
    downloads to a throwaway dir and never edits a tracked file), held under
    the GPU lease for the whole invocation - cpu does not need the GPU, but
    running it outside the lease buys nothing, and holding one lease for the
    combined run is simpler than splitting cpu and vulkan into separate
    confirm invocations with two receipts to merge. Returns
    confirm_llama_runtime's own exit code (0/1/2), or LEASE_BUSY_EXIT if the
    lease could not be acquired at all."""
    cmd = [sys.executable, str(REPO / "scripts" / "confirm_llama_runtime.py"),
           "--tag", candidate, "--backend", "cpu", "--backend", "vulkan",
           "--receipt", str(receipt_path)]
    return run_under_gpu_lease(cmd, purpose=f"pin_pipeline llama confirm {candidate}", cwd=REPO)


# --------------------------------------------------------------------------- #
#  Everything past this point runs INSIDE the dedicated worktree, never the  #
#  shared main checkout - each subprocess gets PYTHONPATH pointed at the     #
#  worktree so it imports THAT tree's localm, not the main checkout's.       #
#  See test_run_bump_uses_the_worktrees_own_copy_not_the_main_checkout.      #
# --------------------------------------------------------------------------- #

def _worktree_env(worktree: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree)
    return env


def run_bump(worktree: Path, candidate: str, receipt_path: Path, *, write: bool) -> "tuple[int, str]":
    """The WORKTREE's own copy of scripts/bump_llama_pin.py --tag candidate
    --receipt receipt_path [--write] - never the main checkout's copy, so the
    edit lands in the worktree's tree, ready to commit there."""
    cmd = [sys.executable, str(worktree / "scripts" / "bump_llama_pin.py"),
           "--tag", candidate, "--receipt", str(receipt_path)]
    if write:
        cmd.append("--write")
    proc = subprocess.run(cmd, cwd=worktree, env=_worktree_env(worktree),
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def run_targeted_tests(worktree: Path) -> "tuple[bool, str]":
    """The pin's own checklist test files (bump_llama_pin.py's checklist item
    5), run targeted - never the full suite - against the WORKTREE's tree."""
    test_files = [
        "tests/test_llama_pin_constant_and_currency.py",
        "tests/test_mtp_arch_allowlist.py",
        "tests/test_llama_runtime_version_pin.py",
        "tests/test_setup_llama_abi_walkback.py",
        "tests/test_setup_llama_backends.py",
        "tests/test_cuda_arch_line_selection.py",
        "tests/test_llamacpp_abi.py",
        "tests/test_check_llama_abi.py",
    ]
    cmd = [sys.executable, "-m", "pytest", *test_files, "-m", "not integration", "-q"]
    proc = subprocess.run(cmd, cwd=worktree, env=_worktree_env(worktree),
                          capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
#  CHANGELOG - a generic, mechanically-correct bullet (no LLM in this loop)   #
# --------------------------------------------------------------------------- #

_UNRELEASED_RE = re.compile(r"^## \[Unreleased\]\n(?P<body>.*?)(?=\n## \[)", re.S | re.M)


def changelog_bullet(old_tag: str, new_tag: str) -> str:
    """A generic bullet naming only the fact (old -> new), never upstream's
    own release notes."""
    return (f"- **The bundled llama.cpp runtime moved from {old_tag} to {new_tag}.** "
            "An existing install picks it up with `localm setup-llama --force`.\n")


def insert_changelog_bullet(text: str, bullet: str) -> str:
    """Insert *bullet* under [Unreleased]'s ### Changed, creating that
    subsection right after the [Unreleased] heading if it does not exist yet.
    Never touches a released section - anchored strictly inside the
    [Unreleased] block."""
    m = _UNRELEASED_RE.search(text)
    if not m:
        raise PipelineError("could not find an [Unreleased] section in CHANGELOG.md")
    body = m.group("body")
    cm = re.search(r"### Changed\n", body)
    if cm:
        new_body = body[:cm.end()] + bullet + body[cm.end():]
    else:
        new_body = "\n### Changed\n" + bullet + body
    return text[:m.start("body")] + new_body + text[m.end("body"):]


# --------------------------------------------------------------------------- #
#  issues.txt - a genuine FAIL is logged where it will actually be seen       #
# --------------------------------------------------------------------------- #

def append_fail_issue(candidate: str, reason: str, receipt_path: "Path | None",
                      issues_path: Path = ISSUES_PATH) -> None:
    """Append a new OPEN entry to issues/issues.txt for a genuine FAIL (never
    for INCONCLUSIVE - that is not evidence of anything). Anchored on the
    file's own intro paragraph, matching the file's title-text convention -
    never a line number, which drifts as the file grows."""
    if not issues_path.exists():
        return
    text = issues_path.read_text(encoding="utf-8")
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    title = f"NEW-PIN-PIPELINE-LLAMA-{candidate.upper()}-CONFIRM-FAILED"
    if title in text:
        return  # already logged for this exact candidate
    anchor = ("in exactly one. Resolved items are DELETED once merged - their history lives in the\n"
              "merged PR and in the dated verbatim backups in dev-notes/issues-backups/.\n\n")
    if anchor not in text:
        return
    receipt_note = f" (receipt: {receipt_path})" if receipt_path else ""
    entry = (
        f"{title} [OPEN - filed {today}] bug/setup - the automated pin pipeline confirmed "
        f"llama.cpp {candidate} FAILS to load and generate on this hardware\n"
        f"    scripts/pin_pipeline.py ran scripts/confirm_llama_runtime.py against upstream's\n"
        f"    {candidate} for real (cpu + vulkan) and it did not pass: {reason}{receipt_note}.\n"
        "    The pin was NOT advanced. This candidate will not be retried automatically unless\n"
        "    a newer upstream release appears - see dev-notes/pin-pipeline/llama-state.json.\n\n"
    )
    issues_path.write_text(text.replace(anchor, anchor + entry, 1), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Git / PR / merge - always from this script's OWN dedicated worktree       #
# --------------------------------------------------------------------------- #

def _run_git(args: "list[str]", cwd: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, **kwargs)


def ensure_pipeline_worktree(repo: Path = REPO) -> Path:
    """The dedicated worktree this script's own branch operations run from,
    never the shared main checkout. Raises PipelineError unless *repo*
    resolves as its own git toplevel and is currently on master."""
    toplevel = _run_git(["rev-parse", "--show-toplevel"], cwd=repo).stdout.strip()
    if Path(toplevel).resolve() != repo.resolve():
        raise PipelineError(
            f"scripts/pin_pipeline.py must run from the main checkout, not a worktree "
            f"(git toplevel is {toplevel!r}, expected {repo})")
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).stdout.strip()
    if branch != "master":
        raise PipelineError(f"main checkout is on {branch!r}, not master; refusing to "
                            "proceed until it is back on master")

    worktree_path = repo.parent / f"{repo.name}-pin-pipeline-worktree"
    existing = _run_git(["worktree", "list", "--porcelain"], cwd=repo).stdout
    if str(worktree_path) not in existing and str(worktree_path).replace("\\", "/") not in existing:
        _run_git(["fetch", "origin"], cwd=repo, check=False)
        result = _run_git(["worktree", "add", "--detach", str(worktree_path),
                          "origin/master"], cwd=repo)
        if result.returncode != 0:
            raise InfraError(f"could not create the pipeline worktree: {result.stderr}")
    return worktree_path


def prepare_bump_branch(worktree: Path, candidate: str) -> str:
    """Reset the pipeline worktree to a fresh branch off origin/master, ready
    for this candidate's bump commit. Raises InfraError if the fetch or the
    detach fails, rather than branching off whatever the worktree was on
    before. See test_prepare_bump_branch_refuses_when_fetch_fails."""
    branch = f"claude/pin-pipeline-llama-{candidate}"
    result = _run_git(["fetch", "origin"], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"git fetch origin failed: {result.stderr}")
    result = _run_git(["checkout", "--detach", "origin/master"], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"could not detach to origin/master: {result.stderr}")
    _run_git(["branch", "-D", branch], cwd=worktree)  # stale from a prior attempt; ok if absent
    result = _run_git(["checkout", "-b", branch], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"could not create branch {branch}: {result.stderr}")
    return branch


def commit_and_push(worktree: Path, branch: str, candidate: str, old_tag: str) -> None:
    commit_msg = (f"chore(llamacpp): advance the pinned build to {candidate}\n\n"
                  f"Automated: confirmed via scripts/confirm_llama_runtime.py on cpu+vulkan "
                  f"(real hardware, this machine) before advancing from {old_tag}.\n\n"
                  "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\n")
    _run_git(["add", "localm/setup_llama.py",
             "localm/inference/backends/llamacpp/_api.py", "CHANGELOG.md"], cwd=worktree)
    result = _run_git(["commit", "-m", commit_msg], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"commit failed: {result.stderr}")
    result = _run_git(["push", "-u", "origin", branch], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"push failed: {result.stderr}")


def open_pr(worktree: Path, branch: str, candidate: str, old_tag: str) -> int:
    body = (f"Automated: confirmed llama.cpp {candidate} loads and generates on cpu and "
            f"vulkan (real hardware) before advancing the pin from {old_tag}. See "
            "scripts/confirm_llama_runtime.py and scripts/bump_llama_pin.py for what was "
            "checked. The bundled-runtime CHANGELOG bullet is included in this diff.\n\n"
            "🤖 Generated with [Claude Code](https://claude.com/claude-code)\n")
    result = subprocess.run(
        ["gh", "pr", "create", "--title",
         f"chore(llamacpp): advance the pinned build to {candidate}",
         "--body", body, "--head", branch],
        cwd=worktree, capture_output=True, text=True)
    if result.returncode != 0:
        raise InfraError(f"gh pr create failed: {result.stderr}")
    m = re.search(r"/pull/(\d+)", result.stdout)
    if not m:
        raise InfraError(f"could not parse a PR number out of: {result.stdout!r}")
    return int(m.group(1))


def wait_for_ci(worktree: Path) -> str:
    """Poll per-check conclusions for the worktree's current HEAD (never
    mergeable/mergeStateStatus). Returns "GREEN", "RED", or "PENDING" (checks
    still running when the deadline passes; the caller leaves the PR open
    rather than force-merging)."""
    deadline = time.monotonic() + CI_WAIT_TIMEOUT_SECONDS
    head_sha = _run_git(["rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["gh", "api", f"repos/{{owner}}/{{repo}}/commits/{head_sha}/check-runs"],
            cwd=worktree, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                runs = json.loads(result.stdout).get("check_runs", [])
            except ValueError:
                runs = []
            if runs:
                non_skipped = [r for r in runs if r.get("conclusion") != "skipped"]
                if any(r.get("conclusion") == "failure" for r in runs):
                    return "RED"
                if non_skipped and all(r.get("status") == "completed" for r in non_skipped):
                    if all(r.get("conclusion") == "success" for r in non_skipped):
                        return "GREEN"
                    return "RED"
        time.sleep(CI_POLL_INTERVAL_SECONDS)
    return "PENDING"


def merge_pr(pr_number: int, worktree: Path, branch: str, candidate: str, old_tag: str) -> None:
    """Squash-merge, then detach and delete the LOCAL branch by hand rather
    than via `gh pr merge --delete-branch` (that flag's local cleanup
    switches to the default branch first, which this worktree is never on).
    See test_merge_pr_detaches_and_deletes_local_branch_without_delete_branch_flag.
    The remote branch is deleted automatically by the repo's own
    delete_branch_on_merge setting."""
    title = f"chore(llamacpp): advance the pinned build to {candidate}"
    body = (f"Automated: confirmed llama.cpp {candidate} loads and generates on cpu and "
            f"vulkan (real hardware) before advancing the pin from {old_tag}.\n\n"
            "🤖 Generated with [Claude Code](https://claude.com/claude-code)\n")
    result = subprocess.run(
        ["gh", "pr", "merge", str(pr_number), "--squash", "-t", title, "-b", body],
        cwd=worktree, capture_output=True, text=True)
    if result.returncode != 0:
        raise InfraError(f"gh pr merge failed: {result.stderr}")
    _run_git(["switch", "--detach"], cwd=worktree)
    _run_git(["branch", "-D", branch], cwd=worktree)


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #

def run_llama_pipeline(*, dry_run: bool) -> int:
    candidate_pair = newest_candidate()
    if candidate_pair is None:
        return 0
    old_tag, candidate = candidate_pair

    state = load_state()
    skip_reason = should_skip(state, candidate)
    if skip_reason:
        print(f"skip: {skip_reason}")
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = STATE_DIR / "receipts" / f"llama-{candidate}-{int(time.time())}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    rc = run_confirm(candidate, receipt_path)
    if rc == LEASE_BUSY_EXIT:
        print("GPU busy this run; not recording a verdict, will retry next run")
        return 2
    now_iso = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if rc == 1:
        reason = "confirm_llama_runtime.py reported FAIL"
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": reason})
        append_fail_issue(candidate, reason, receipt_path)
        print(f"FAIL: {reason}")
        return 1
    if rc == 2:
        save_state({"last_tag_tried": candidate, "verdict": "INCONCLUSIVE", "timestamp": now_iso,
                   "receipt_path": str(receipt_path)})
        print("INCONCLUSIVE: could not measure this run; will retry after cooldown")
        return 2
    print(f"PASS: {candidate} confirmed on {', '.join(REQUIRE_BACKENDS)}")

    if dry_run:
        print("--dry-run: stopping before bump/merge")
        return 0

    try:
        worktree = ensure_pipeline_worktree()
        branch = prepare_bump_branch(worktree, candidate)

        bump_rc, bump_out = run_bump(worktree, candidate, receipt_path, write=True)
        if bump_rc != 0:
            raise PipelineError(f"bump_llama_pin.py refused: {bump_out}")

        changelog_path = worktree / "CHANGELOG.md"
        text = changelog_path.read_text(encoding="utf-8")
        changelog_path.write_text(
            insert_changelog_bullet(text, changelog_bullet(old_tag, candidate)),
            encoding="utf-8")

        passed, test_out = run_targeted_tests(worktree)
        if not passed:
            raise PipelineError(f"targeted tests failed after the bump:\n{test_out[-2000:]}")

        commit_and_push(worktree, branch, candidate, old_tag)
        pr_number = open_pr(worktree, branch, candidate, old_tag)
        outcome = wait_for_ci(worktree)
        if outcome == "GREEN":
            merge_pr(pr_number, worktree, branch, candidate, old_tag)
            save_state({"last_tag_tried": candidate, "verdict": "PASS", "timestamp": now_iso,
                       "receipt_path": str(receipt_path), "merged_pr": pr_number})
            print(f"merged PR #{pr_number}: {old_tag} -> {candidate}")
            return 0
        reason = f"CI {outcome.lower()} on PR #{pr_number}; left open, not merged"
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": reason, "open_pr": pr_number})
        if outcome == "RED":
            append_fail_issue(candidate, reason, receipt_path)
        print(reason)
        return 1
    except PipelineError as e:
        if isinstance(e, InfraError):
            save_state({"last_tag_tried": candidate, "verdict": "INCONCLUSIVE", "timestamp": now_iso,
                       "receipt_path": str(receipt_path), "reason": str(e)})
            print(f"INCONCLUSIVE (infra): {e}")
            return 2
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": str(e)})
        append_fail_issue(candidate, str(e), receipt_path)
        print(f"FAIL: {e}")
        return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pin", required=True, choices=("llama", "comfyui", "rocm"))
    ap.add_argument("--dry-run", action="store_true",
                    help="confirm the candidate but never bump, commit, or merge")
    args = ap.parse_args(argv)

    if args.pin != "llama":
        print(f"--pin {args.pin} is not built yet (Phase 1 covers llama only)")
        return 1
    return run_llama_pipeline(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
