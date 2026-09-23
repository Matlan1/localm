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
import contextlib
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

# After this many CONSECUTIVE INCONCLUSIVE runs for the same candidate, log
# an issue even though the verdict itself stays INCONCLUSIVE (still
# cooldown-retried, never promoted to a permanent FAIL) - a command line or
# environment that is silently broken every single run must not loop
# invisibly forever with nobody ever finding out. See _record_inconclusive.
INCONCLUSIVE_ISSUE_THRESHOLD = 5

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


def newest_comfyui_candidate(repo: Path = REPO) -> "tuple[str, str, str] | None":
    """(pinned tag, candidate tag, candidate commit) when upstream has a
    newer release AND its commit could be resolved, else None (already
    current, upstream unreadable, or the tag->commit resolution failed - all
    read the same here: nothing eligible to test this run).

    Raises PipelineError for an internal-consistency problem worth a human's
    attention regardless of build quality: COMFYUI_PINNED_VERSION renamed/
    reformatted (check_comfyui_pin._pinned_version raises SystemExit for
    this - caught and re-raised as PipelineError so it is never mistaken for
    "nothing to do"), or the pinned tag itself no longer parses as a version
    (check_comfyui_pin's own "unparseable_pin" status)."""
    check_comfyui_pin = _load_module(repo / "scripts" / "check_comfyui_pin.py",
                                     "check_comfyui_pin")
    try:
        pin = check_comfyui_pin._pinned_version()
    except SystemExit as e:
        raise PipelineError(f"could not read the ComfyUI pin: {e}") from e
    releases = check_comfyui_pin._fetch_releases()
    if releases is None:
        print("currency: could not read upstream ComfyUI releases; nothing to do")
        return None
    result = check_comfyui_pin._compare(pin, releases)
    if result["status"] == "unparseable_pin":
        raise PipelineError(f"the pinned ComfyUI version {pin!r} does not parse as vX.Y[.Z]")
    if result["status"] in ("no_data", "current"):
        print(f"currency: ComfyUI {pin} is already current")
        return None
    candidate = result["latest"]
    commit = check_comfyui_pin.resolve_tag_commit(candidate)
    if commit is None:
        print(f"currency: could not resolve {candidate}'s commit; nothing to do this run")
        return None
    print(f"currency: ComfyUI {pin} -> candidate {candidate} ({commit[:12]})")
    return pin, candidate, commit


# --------------------------------------------------------------------------- #
#  State (gitignored, main-checkout-only - never retry a known-bad candidate) #
# --------------------------------------------------------------------------- #

def _state_path(pin: str = "llama") -> Path:
    return STATE_DIR / f"{pin}-state.json"


def load_state(*, pin: str = "llama") -> dict:
    path = _state_path(pin)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict, *, pin: str = "llama") -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(pin).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


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


def _comfyui_scratch_workdir() -> Path:
    """The ComfyUI confirm's dedicated scratch dir - never the llama one, and
    never a fresh tempfile.mkdtemp() per run: it must PERSIST across the
    separate provision/smoke subprocess invocations below (provision installs
    into it, smoke launches and tests that SAME install), and its cache/
    subdir surviving between pipeline RUNS is what avoids re-downloading
    several GB of torch/ComfyUI wheels every single time."""
    return REPO.parent / f"{REPO.name}-pin-pipeline-comfy-scratch"


def run_comfyui_confirm(tag: str, commit: str, receipt_path: Path) -> int:
    """scripts/confirm_comfyui_runtime.py, run as three SEPARATE subprocess
    invocations (not --phase all) so only the smoke phase - the one that
    actually launches a server and touches the GPU - runs under the shared
    lease. provision and the final teardown never hold it: an install can run
    well past the lease's own TTL and other sessions' wait budget (see
    confirm_comfyui_runtime.py's own module docstring). Teardown ALWAYS runs
    (a finally-equivalent, both a pre-run cleanup of any crashed prior
    attempt's leftovers and a post-run cleanup here), regardless of what
    provision/smoke returned. Returns the overall confirm exit code (0/1/2),
    or LEASE_BUSY_EXIT if the smoke phase's lease could not be acquired at
    all (provision still ran; its receipt is kept for the next attempt)."""
    workdir = _comfyui_scratch_workdir()
    script = REPO / "scripts" / "confirm_comfyui_runtime.py"

    def _run_phase(phase: str, *, require_gpu: bool = False) -> "tuple[int, str]":
        cmd = [sys.executable, str(script), "--tag", tag, "--commit", commit,
              "--workdir", str(workdir), "--receipt", str(receipt_path), "--phase", phase]
        if require_gpu:
            cmd.append("--require-gpu")
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        return proc.returncode, proc.stdout + proc.stderr

    teardown_rc, teardown_out = _run_phase("teardown")
    if teardown_rc != 0:
        print(f"INCONCLUSIVE: pre-run teardown of a stale scratch install failed:\n"
             f"{teardown_out[-2000:]}")
        return 2

    provision_rc, provision_out = _run_phase("provision", require_gpu=True)
    print(provision_out)
    if provision_rc != 0:
        _run_phase("teardown")
        return provision_rc

    try:
        smoke_cmd = [sys.executable, str(script), "--tag", tag, "--commit", commit,
                    "--workdir", str(workdir), "--receipt", str(receipt_path),
                    "--phase", "smoke"]
        smoke_rc = run_under_gpu_lease(
            smoke_cmd, purpose=f"pin_pipeline comfyui smoke {tag}", cwd=REPO)
    finally:
        teardown_rc, teardown_out = _run_phase("teardown")
        if teardown_rc != 0:
            print(f"WARNING: post-run teardown failed:\n{teardown_out[-2000:]}")
    return smoke_rc


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


def run_comfyui_bump(worktree: Path, tag: str, commit: str, receipt_path: Path,
                     *, write: bool) -> "tuple[int, str]":
    """The WORKTREE's own copy of scripts/bump_comfyui_pin.py --tag tag
    --commit commit --receipt receipt_path [--write] - never the main
    checkout's copy, matching run_bump()'s own reasoning."""
    cmd = [sys.executable, str(worktree / "scripts" / "bump_comfyui_pin.py"),
           "--tag", tag, "--commit", commit, "--receipt", str(receipt_path)]
    if write:
        cmd.append("--write")
    proc = subprocess.run(cmd, cwd=worktree, env=_worktree_env(worktree),
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def run_comfyui_targeted_tests(worktree: Path) -> "tuple[bool, str]":
    """The ComfyUI pin's own checklist test files (bump_comfyui_pin.py's
    checklist item 1), run targeted - never the full suite - against the
    WORKTREE's tree."""
    test_files = [
        "tests/test_check_comfyui_pin.py",
        "tests/test_bump_comfyui_pin.py",
        "tests/test_confirm_comfyui_runtime.py",
        "tests/test_media_placement.py",
        "tests/test_managed_comfy_s1.py",
        "tests/test_managed_comfy_s3.py",
        "tests/test_managed_comfy_s4.py",
        "tests/test_managed_comfy_s5_gui.py",
        "tests/test_comfy_cli_outcome_honesty.py",
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


def comfyui_changelog_bullet(old_tag: str, new_tag: str, *,
                             requirements_changed: bool = False) -> str:
    """The ComfyUI analog of changelog_bullet() - names the `--reinstall-
    requirements` flag whenever the receipt showed requirements.txt actually
    changed, since plain `localm comfy update` does not reinstall
    requirements by default (see bump_comfyui_pin.py's own checklist)."""
    if requirements_changed:
        pickup = ("`localm comfy update --reinstall-requirements` (this release's "
                 "Python requirements changed)")
    else:
        pickup = "`localm comfy update`"
    return (f"- **localm's managed ComfyUI now installs {new_tag}, up from {old_tag}.** "
            f"An existing install picks it up with {pickup}.\n")


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

# Matches the structural end of issues.txt's own intro paragraph (its first
# blank line) without spelling out that paragraph's actual prose as a literal
# string constant here - this file is tracked and public, and the prose
# itself names a gitignored path (see test_no_gitignored_path_leak.py).
_ISSUES_INTRO_RE = re.compile(r"an entry lives\nin exactly one\..*?\n\n", re.S)


def append_fail_issue(candidate: str, reason: str, receipt_path: "Path | None",
                      issues_path: Path = ISSUES_PATH, *, pin: str = "llama",
                      summary: "str | None" = None, kind: str = "CONFIRM-FAILED") -> bool:
    """Append a new OPEN entry to issues/issues.txt for a genuine FAIL (never
    for an ordinary INCONCLUSIVE - that is not evidence of anything; a
    repeated-INCONCLUSIVE streak past INCONCLUSIVE_ISSUE_THRESHOLD and an
    uncaught-exception INCONCLUSIVE are both deliberate exceptions, logged
    via _record_inconclusive). Anchored on the file's own intro paragraph,
    matching the file's title-text convention - never a line number, which
    drifts as the file grows. Returns True if the entry was written or
    already present. Returns False, printing a warning, if issues_path
    exists but the intro anchor no longer matches (NOT the same as
    issues_path simply not existing, which is the expected, silent case on
    CI or a fresh clone).

    *pin* names the title prefix (upper-cased; a dotted candidate like a
    ComfyUI tag is dash-safed for the title only - issue ids use dashes,
    never dots). *kind* names the title's final segment and is the dedup
    key alongside pin+candidate - keeping it at its default for every FAIL
    call site and giving _record_inconclusive's own STUCK-INCONCLUSIVE
    entries a DIFFERENT kind means the two can never silently deduplicate
    against each other: a genuine FAIL reported after a candidate already
    has a stuck-INCONCLUSIVE entry (or the reverse) still gets its own
    entry, rather than the second call finding the first's title already
    present and silently writing nothing. *summary* overrides the WHOLE
    default llama-specific descriptive block (what ran, on what, and how) -
    the default reproduces the exact original text so the llama call site
    needs no change; a different pin passes its own multi-line description
    ending in ": {reason}{receipt_note}." to match the same shape."""
    if not issues_path.exists():
        return False
    text = issues_path.read_text(encoding="utf-8")
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    title = (f"NEW-PIN-PIPELINE-{pin.upper()}-"
            f"{candidate.upper().replace('.', '-')}-{kind}")
    if title in text:
        return True  # already logged for this exact candidate
    m = _ISSUES_INTRO_RE.search(text)
    if not m:
        print(f"WARNING: issues.txt's intro anchor no longer matches; {title} was "
              "NOT logged there (the FAIL above and the state file are still recorded)")
        return False
    receipt_note = f" (receipt: {receipt_path})" if receipt_path else ""
    summary = summary or (
        f"llama.cpp {candidate} FAILS to load and generate on this hardware\n"
        f"    scripts/pin_pipeline.py ran scripts/confirm_llama_runtime.py against upstream's\n"
        f"    {candidate} for real (cpu + vulkan) and it did not pass: {reason}{receipt_note}.")
    entry = (
        f"{title} [OPEN - filed {today}] bug/setup - the automated pin pipeline confirmed "
        f"{summary}\n"
        "    The pin was NOT advanced. This candidate will not be retried automatically unless\n"
        "    a newer upstream release appears.\n\n"
    )
    issues_path.write_text(text[:m.end()] + entry + text[m.end():], encoding="utf-8")
    return True


def _record_inconclusive(candidate: str, receipt_path: "Path | None", *, pin: str = "llama",
                         reason: "str | None" = None, force_issue: bool = False,
                         issue_summary: "str | None" = None, **extra) -> int:
    """Save an INCONCLUSIVE verdict for *candidate* and track how many
    CONSECUTIVE runs have been INCONCLUSIVE for this exact candidate (any
    other verdict, or a different candidate becoming newest, resets the
    streak to 1). The verdict itself always stays INCONCLUSIVE - never
    promoted to FAIL - regardless of the streak.

    Once the streak first reaches INCONCLUSIVE_ISSUE_THRESHOLD, logs a
    STUCK-INCONCLUSIVE issue - a DIFFERENT kind than a genuine FAIL, so the
    two can never silently deduplicate against each other via
    append_fail_issue's own per-title dedup: a command line or environment
    that is silently broken every single run must not retry forever with
    nobody ever finding out.

    *force_issue* additionally logs on EVERY call regardless of streak,
    using the default CONFIRM-FAILED kind, for the unexpected-exception
    handlers which already log unconditionally on their own reasoning (an
    uncaught exception is always worth a human's attention - see
    test_run_llama_pipeline_unexpected_exception_is_inconclusive_and_logged).
    *issue_summary*, when given, overrides append_fail_issue's own default
    summary text. Returns the new streak count."""
    prior = load_state(pin=pin)
    streak = (prior.get("inconclusive_streak", 0) + 1
             if prior.get("last_tag_tried") == candidate and prior.get("verdict") == "INCONCLUSIVE"
             else 1)
    now_iso = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state = {"last_tag_tried": candidate, "verdict": "INCONCLUSIVE", "timestamp": now_iso,
             "receipt_path": str(receipt_path), "inconclusive_streak": streak}
    if reason is not None:
        state["reason"] = reason
    state.update(extra)
    save_state(state, pin=pin)
    if force_issue:
        append_fail_issue(candidate, reason or "unexpected error", receipt_path, pin=pin,
                          summary=issue_summary)
    elif streak == INCONCLUSIVE_ISSUE_THRESHOLD:
        append_fail_issue(
            candidate, reason or "repeated INCONCLUSIVE results", receipt_path, pin=pin,
            kind="STUCK-INCONCLUSIVE",
            summary=issue_summary or (
                f"{pin} pin pipeline: {candidate} has been INCONCLUSIVE for {streak} runs "
                f"in a row\n    scripts/pin_pipeline.py --pin {pin} could not reach a "
                f"verdict for {candidate} on\n    {streak} consecutive attempts. The verdict "
                f"stays INCONCLUSIVE (never promoted to\n    a build FAIL, still "
                f"cooldown-retried automatically), but this many repeats\n    in a row "
                f"usually means the command line or environment itself is broken\n    "
                f"rather than transient GPU contention - worth a human look. Latest "
                f"reason:\n    {reason or 'n/a'}."))
    return streak


# --------------------------------------------------------------------------- #
#  Git / PR / merge - always from this script's OWN dedicated worktree       #
# --------------------------------------------------------------------------- #

def _run_git(args: "list[str]", cwd: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, **kwargs)


def _verify_main_checkout(repo: Path) -> None:
    """Raise PipelineError unless *repo* resolves as its own git toplevel
    and is currently on master."""
    toplevel = _run_git(["rev-parse", "--show-toplevel"], cwd=repo).stdout.strip()
    if Path(toplevel).resolve() != repo.resolve():
        raise PipelineError(
            f"scripts/pin_pipeline.py must run from the main checkout, not a worktree "
            f"(git toplevel is {toplevel!r}, expected {repo})")
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).stdout.strip()
    if branch != "master":
        raise PipelineError(f"main checkout is on {branch!r}, not master; refusing to "
                            "proceed until it is back on master")


def sync_main_checkout(repo: Path = REPO) -> None:
    """Verify *repo* is the real main checkout on master, then fast-forward
    it to origin/master. newest_candidate() reads the pin straight off this
    checkout's own working tree, and nothing else in this script ever
    updates it (a merge happens from the dedicated worktree instead) - a
    stale main checkout would keep reading the pin's value from before the
    last successful merge, forever. See
    test_run_llama_pipeline_resyncs_the_main_checkout_before_reading_the_pin."""
    _verify_main_checkout(repo)
    result = _run_git(["fetch", "origin"], cwd=repo)
    if result.returncode != 0:
        raise InfraError(f"git fetch origin failed: {result.stderr}")
    result = _run_git(["merge", "--ff-only", "origin/master"], cwd=repo)
    if result.returncode != 0:
        raise InfraError(f"could not fast-forward the main checkout to origin/master: "
                         f"{result.stderr}")


def ensure_pipeline_worktree(repo: Path = REPO) -> Path:
    """The dedicated worktree this script's own branch operations run from,
    never the shared main checkout. Raises PipelineError unless *repo*
    resolves as its own git toplevel and is currently on master."""
    _verify_main_checkout(repo)
    worktree_path = repo.parent / f"{repo.name}-pin-pipeline-worktree"
    existing = _run_git(["worktree", "list", "--porcelain"], cwd=repo).stdout
    if str(worktree_path) not in existing and str(worktree_path).replace("\\", "/") not in existing:
        _run_git(["fetch", "origin"], cwd=repo, check=False)
        result = _run_git(["worktree", "add", "--detach", str(worktree_path),
                          "origin/master"], cwd=repo)
        if result.returncode != 0:
            raise InfraError(f"could not create the pipeline worktree: {result.stderr}")
    return worktree_path


def close_stale_pr(branch: str, worktree: Path) -> None:
    """Close (never merge) any PR already open for *branch*, and delete its
    remote ref - left behind by a prior run that crashed or timed out
    between pushing and merging. Left alone, the next attempt's push would
    be a non-fast-forward rejection against the stale remote branch, and
    `gh pr create` would refuse a second PR for the same head - both read as
    a fresh infra failure instead of the resumable situation this actually
    is. The caller must already have the worktree detached from *branch*
    before this runs. See test_prepare_bump_branch_closes_a_stale_pr_left_by_a_prior_run."""
    result = subprocess.run(
        ["gh", "pr", "list", "--head", branch, "--state", "open", "--json", "number"],
        cwd=worktree, capture_output=True, text=True)
    if result.returncode != 0:
        raise InfraError(f"gh pr list failed: {result.stderr}")
    try:
        prs = json.loads(result.stdout)
    except ValueError:
        raise InfraError(f"could not parse gh pr list output: {result.stdout!r}") from None
    for pr in prs:
        result = subprocess.run(
            ["gh", "pr", "close", str(pr["number"])], cwd=worktree, capture_output=True, text=True)
        if result.returncode != 0:
            raise InfraError(f"could not close stale PR #{pr['number']}: {result.stderr}")
    result = _run_git(["ls-remote", "--exit-code", "--heads", "origin", branch], cwd=worktree)
    if result.returncode == 0:
        result = _run_git(["push", "origin", "--delete", branch], cwd=worktree)
        if result.returncode != 0:
            raise InfraError(f"could not delete stale remote branch {branch}: {result.stderr}")


def prepare_bump_branch(worktree: Path, candidate: str, *, pin: str = "llama") -> str:
    """Reset the pipeline worktree to a fresh branch off origin/master, ready
    for this candidate's bump commit. Raises InfraError if the fetch or the
    detach fails, rather than branching off whatever the worktree was on
    before. See test_prepare_bump_branch_refuses_when_fetch_fails."""
    branch = f"claude/pin-pipeline-{pin}-{candidate}"
    result = _run_git(["fetch", "origin"], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"git fetch origin failed: {result.stderr}")
    result = _run_git(["checkout", "--detach", "origin/master"], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"could not detach to origin/master: {result.stderr}")
    close_stale_pr(branch, worktree)
    _run_git(["branch", "-D", branch], cwd=worktree)  # stale local branch; ok if absent
    result = _run_git(["checkout", "-b", branch], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"could not create branch {branch}: {result.stderr}")
    return branch


def commit_and_push(worktree: Path, branch: str, candidate: str, old_tag: str,
                    *, message: "str | None" = None) -> None:
    """Stages every TRACKED modification (`git add -u`), not a hardcoded file
    list. run_bump() always touches setup_llama.py/_api.py/CHANGELOG.md, but
    a bump can also require a manual follow-on fix to a safety-relevant
    constant elsewhere in setup_llama.py plus its own tests (see the
    b11118 cuda-13 13.3->13.4 toolkit rename) - a hardcoded list silently
    drops such a fix from the commit, so the PR would pass locally (the
    working tree has the fix) and then fail on CI (the commit does not).
    `git add -u` is safe here because prepare_bump_branch() always resets
    this dedicated worktree to a fresh origin/master checkout first, so
    nothing untracked or unrelated can be sitting in it. See
    test_commit_and_push_stages_every_tracked_modification_not_just_the_bump_files.

    *message* overrides the default llama-specific commit message (for a
    different pin's own wording); the default reproduces the exact original
    text so the llama call site needs no change."""
    commit_msg = message or (
        f"chore(llamacpp): advance the pinned build to {candidate}\n\n"
        f"Automated: confirmed via scripts/confirm_llama_runtime.py on cpu+vulkan "
        f"(real hardware, this machine) before advancing from {old_tag}.\n\n"
        "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\n")
    _run_git(["add", "-u"], cwd=worktree)
    result = _run_git(["commit", "-m", commit_msg], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"commit failed: {result.stderr}")
    result = _run_git(["push", "-u", "origin", branch], cwd=worktree)
    if result.returncode != 0:
        raise InfraError(f"push failed: {result.stderr}")


def open_pr(worktree: Path, branch: str, candidate: str, old_tag: str,
           *, title: "str | None" = None, body: "str | None" = None) -> int:
    """*title*/*body* override the default llama-specific text; the defaults
    reproduce the exact original text so the llama call site needs no
    change."""
    title = title or f"chore(llamacpp): advance the pinned build to {candidate}"
    body = body or (
        f"Automated: confirmed llama.cpp {candidate} loads and generates on cpu and "
        f"vulkan (real hardware) before advancing the pin from {old_tag}. See "
        "scripts/confirm_llama_runtime.py and scripts/bump_llama_pin.py for what was "
        "checked. The bundled-runtime CHANGELOG bullet is included in this diff.\n\n"
        "🤖 Generated with [Claude Code](https://claude.com/claude-code)\n")
    result = subprocess.run(
        ["gh", "pr", "create", "--title", title, "--body", body, "--head", branch],
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


def merge_pr(pr_number: int, worktree: Path, branch: str, candidate: str, old_tag: str,
            *, title: "str | None" = None, body: "str | None" = None) -> None:
    """Squash-merge, then detach and delete the LOCAL branch by hand rather
    than via `gh pr merge --delete-branch` (that flag's local cleanup
    switches to the default branch first, which this worktree is never on).
    See test_merge_pr_detaches_and_deletes_local_branch_without_delete_branch_flag.
    The remote branch is deleted automatically by the repo's own
    delete_branch_on_merge setting.

    *title*/*body* override the default llama-specific text, same convention
    as open_pr()."""
    title = title or f"chore(llamacpp): advance the pinned build to {candidate}"
    body = body or (
        f"Automated: confirmed llama.cpp {candidate} loads and generates on cpu and "
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
#  Cross-pin lock: llama and comfyui share ONE dedicated worktree and ONE     #
#  CHANGELOG.md. Without this, prepare_bump_branch()'s fresh                  #
#  `checkout --detach origin/master` for one pin can silently wipe another    #
#  pin's in-progress, uncommitted bump if both ever ran concurrently (two     #
#  separate scheduled tasks). Held for the WHOLE pipeline run, not just the   #
#  write steps - a candidate could be found eligible for both pins at once.   #
# --------------------------------------------------------------------------- #

_LOCK_OWNER_FILE = "owner.json"


def _pipeline_lock_path() -> Path:
    return STATE_DIR / "pipeline.lock"


class PipelineLockBusy(PipelineError):
    """Another pin_pipeline.py run already holds the cross-pin lock. Recorded
    as INCONCLUSIVE (cooldown retry) - this run never even started, so it is
    not evidence about any candidate's quality."""


@contextlib.contextmanager
def pipeline_lock():
    """Atomic-mkdir lock, reclaimed if its recorded owner pid is dead (same
    pattern as localm.media.managed_comfy._acquire_update_lock). Never
    waits - a second concurrent run raises PipelineLockBusy immediately
    rather than blocking, since a scheduled task should fail fast and retry
    on its own next schedule, not pile up waiting."""
    lock = _pipeline_lock_path()
    from localm.instances import pid_alive

    def _try_acquire() -> bool:
        """True if the lock was taken; False if it is held by a genuinely
        live owner (never raises for that case - only a real I/O problem
        raises)."""
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(str(lock))  # ATOMIC: creates or raises FileExistsError
            return True
        except FileExistsError:
            pass
        pid = None
        try:
            owner = json.loads((lock / _LOCK_OWNER_FILE).read_text(encoding="utf-8"))
            pid = owner.get("pid") if isinstance(owner, dict) else None
        except (OSError, ValueError):
            pass
        if isinstance(pid, int) and not pid_alive(pid):
            try:
                (lock / _LOCK_OWNER_FILE).unlink(missing_ok=True)
                lock.rmdir()
            except OSError as e:
                raise PipelineLockBusy(
                    f"a stale pipeline lock at {lock} (dead pid {pid}) could not be "
                    f"cleared: {e}") from e
            return False  # reclaimed; caller retries the mkdir once
        raise PipelineLockBusy(
            f"another pin_pipeline.py run already holds {lock}"
            + (f" (pid {pid})" if pid else " (owner unreadable)"))

    if not _try_acquire() and not _try_acquire():
        raise PipelineLockBusy(f"another pin_pipeline.py run already holds {lock}")
    try:
        (lock / _LOCK_OWNER_FILE).write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    except OSError:
        pass
    try:
        yield
    finally:
        try:
            (lock / _LOCK_OWNER_FILE).unlink(missing_ok=True)
            lock.rmdir()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #

def run_llama_pipeline(*, dry_run: bool) -> int:
    try:
        sync_main_checkout()
    except InfraError as e:
        print(f"INCONCLUSIVE (infra, before any candidate is known): {e}")
        return 2
    except PipelineError as e:
        print(f"FAIL (before any candidate is known): {e}")
        return 1

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
        _record_inconclusive(candidate, receipt_path)
        print("INCONCLUSIVE: could not measure this run; will retry after cooldown")
        return 2
    if rc != 0:
        reason = f"confirm_llama_runtime.py exited with unexpected code {rc}"
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": reason})
        append_fail_issue(candidate, reason, receipt_path)
        print(f"FAIL: {reason}")
        return 1
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
        if outcome == "PENDING":
            # Checks were still running when the wait window closed - not evidence
            # the build is bad, so this retries after a cooldown like any other
            # INCONCLUSIVE, and the next attempt's prepare_bump_branch() closes
            # this PR and its branch before starting fresh.
            _record_inconclusive(candidate, receipt_path, open_pr=pr_number)
            print(f"INCONCLUSIVE: CI still pending on PR #{pr_number} after the wait window; "
                 "left open, will retry")
            return 2
        reason = f"CI red on PR #{pr_number}; left open, not merged"
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": reason, "open_pr": pr_number})
        append_fail_issue(candidate, reason, receipt_path)
        print(reason)
        return 1
    except PipelineError as e:
        if isinstance(e, InfraError):
            _record_inconclusive(candidate, receipt_path, reason=str(e))
            print(f"INCONCLUSIVE (infra): {e}")
            return 2
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": str(e)})
        append_fail_issue(candidate, str(e), receipt_path)
        print(f"FAIL: {e}")
        return 1
    except Exception as e:
        # Anything past this point that is NOT a PipelineError is about this
        # run's own tooling (a subprocess whose executable could not be
        # resolved, an environment problem) rather than the candidate build's
        # quality, which already passed real hardware confirm before this
        # block runs - so INCONCLUSIVE (cooldown retry), never a permanent
        # FAIL. Unlike an ordinary INCONCLUSIVE this is always logged: an
        # uncaught exception is always worth a human's attention, whatever
        # the retry eventually does. See
        # test_run_llama_pipeline_unexpected_exception_is_inconclusive_and_logged.
        reason = f"unexpected error after a PASS confirm: {type(e).__name__}: {e}"
        _record_inconclusive(candidate, receipt_path, reason=reason, force_issue=True)
        print(f"INCONCLUSIVE (unexpected error): {reason}")
        return 2


def run_comfyui_pipeline(*, dry_run: bool) -> int:
    """The ComfyUI analog of run_llama_pipeline(), same overall shape and
    same state-machine outcomes, adapted for: a 3-part candidate (old tag,
    new tag, new commit) instead of 2; a 3-phase confirm
    (run_comfyui_confirm already owns provision/smoke/teardown and the GPU
    lease internally, unlike llama's single confirm invocation); and no
    _PIN_CONFIRMATION-vs-receipt cross-check (bump_comfyui_pin.py has none,
    since ComfyUI has no per-backend confirmation table to keep honest)."""
    try:
        sync_main_checkout()
    except InfraError as e:
        print(f"INCONCLUSIVE (infra, before any candidate is known): {e}")
        return 2
    except PipelineError as e:
        print(f"FAIL (before any candidate is known): {e}")
        return 1

    candidate_triple = newest_comfyui_candidate()
    if candidate_triple is None:
        return 0
    old_tag, candidate, commit = candidate_triple

    state = load_state(pin="comfyui")
    skip_reason = should_skip(state, candidate)
    if skip_reason:
        print(f"skip: {skip_reason}")
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = STATE_DIR / "receipts" / f"comfyui-{candidate}-{int(time.time())}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    rc = run_comfyui_confirm(candidate, commit, receipt_path)
    if rc == LEASE_BUSY_EXIT:
        print("GPU busy this run; not recording a verdict, will retry next run")
        return 2
    now_iso = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if rc == 1:
        reason = "confirm_comfyui_runtime.py reported FAIL"
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": reason}, pin="comfyui")
        append_fail_issue(
            candidate, reason, receipt_path, pin="comfyui",
            summary=f"ComfyUI {candidate} FAILS to install/run on this hardware\n"
                   f"    scripts/pin_pipeline.py ran scripts/confirm_comfyui_runtime.py against\n"
                   f"    upstream's {candidate} ({commit[:12]}) for real and it did not pass: "
                   f"{reason} (receipt: {receipt_path}).")
        print(f"FAIL: {reason}")
        return 1
    if rc == 2:
        _record_inconclusive(candidate, receipt_path, pin="comfyui")
        print("INCONCLUSIVE: could not measure this run; will retry after cooldown")
        return 2
    if rc != 0:
        reason = f"confirm_comfyui_runtime.py exited with unexpected code {rc}"
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": reason}, pin="comfyui")
        append_fail_issue(candidate, reason, receipt_path, pin="comfyui")
        print(f"FAIL: {reason}")
        return 1
    print(f"PASS: {candidate} ({commit[:12]}) confirmed")

    if dry_run:
        print("--dry-run: stopping before bump/merge")
        return 0

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        requirements_changed = bool((receipt.get("baseline") or {}).get("requirements_changed"))

        worktree = ensure_pipeline_worktree()
        branch = prepare_bump_branch(worktree, candidate, pin="comfyui")

        bump_rc, bump_out = run_comfyui_bump(worktree, candidate, commit, receipt_path,
                                             write=True)
        if bump_rc != 0:
            raise PipelineError(f"bump_comfyui_pin.py refused: {bump_out}")

        changelog_path = worktree / "CHANGELOG.md"
        text = changelog_path.read_text(encoding="utf-8")
        changelog_path.write_text(
            insert_changelog_bullet(
                text, comfyui_changelog_bullet(
                    old_tag, candidate, requirements_changed=requirements_changed)),
            encoding="utf-8")

        passed, test_out = run_comfyui_targeted_tests(worktree)
        if not passed:
            raise PipelineError(f"targeted tests failed after the bump:\n{test_out[-2000:]}")

        commit_msg = (f"chore(comfyui): advance the pinned build to {candidate}\n\n"
                     f"Automated: confirmed via scripts/confirm_comfyui_runtime.py "
                     f"(real ROCm hardware, this machine) before advancing from "
                     f"{old_tag}.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\n")
        commit_and_push(worktree, branch, candidate, old_tag, message=commit_msg)

        pr_body = (f"Automated: confirmed ComfyUI {candidate} ({commit[:12]}) installs and "
                  f"runs (real GPU kernel, real ROCm hardware) before advancing the pin "
                  f"from {old_tag}. See scripts/confirm_comfyui_runtime.py and "
                  f"scripts/bump_comfyui_pin.py for what was checked; this confirm is "
                  f"deliberately model-free (see the receipt's own not_covered list). "
                  f"The CHANGELOG bullet is included in this diff.\n\n"
                  f"🤖 Generated with [Claude Code](https://claude.com/claude-code)\n")
        pr_title = f"chore(comfyui): advance the pinned build to {candidate}"
        pr_number = open_pr(worktree, branch, candidate, old_tag, title=pr_title, body=pr_body)
        outcome = wait_for_ci(worktree)
        if outcome == "GREEN":
            merge_pr(pr_number, worktree, branch, candidate, old_tag,
                    title=pr_title, body=pr_body)
            save_state({"last_tag_tried": candidate, "verdict": "PASS", "timestamp": now_iso,
                       "receipt_path": str(receipt_path), "merged_pr": pr_number},
                      pin="comfyui")
            print(f"merged PR #{pr_number}: {old_tag} -> {candidate}")
            return 0
        if outcome == "PENDING":
            _record_inconclusive(candidate, receipt_path, pin="comfyui", open_pr=pr_number)
            print(f"INCONCLUSIVE: CI still pending on PR #{pr_number} after the wait window; "
                 "left open, will retry")
            return 2
        reason = f"CI red on PR #{pr_number}; left open, not merged"
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": reason, "open_pr": pr_number},
                  pin="comfyui")
        append_fail_issue(candidate, reason, receipt_path, pin="comfyui")
        print(reason)
        return 1
    except PipelineError as e:
        if isinstance(e, InfraError):
            _record_inconclusive(candidate, receipt_path, pin="comfyui", reason=str(e))
            print(f"INCONCLUSIVE (infra): {e}")
            return 2
        save_state({"last_tag_tried": candidate, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt_path), "reason": str(e)}, pin="comfyui")
        append_fail_issue(candidate, str(e), receipt_path, pin="comfyui")
        print(f"FAIL: {e}")
        return 1
    except Exception as e:
        # See run_llama_pipeline's identical handler for the full reasoning:
        # this is about THIS RUN's own tooling, never the candidate's quality
        # (already proven by a PASS confirm before this block runs), so it is
        # INCONCLUSIVE (cooldown retry) but always logged regardless.
        reason = f"unexpected error after a PASS confirm: {type(e).__name__}: {e}"
        _record_inconclusive(
            candidate, receipt_path, pin="comfyui", reason=reason, force_issue=True,
            issue_summary=f"ComfyUI {candidate}: unexpected pipeline error after a PASS "
                         f"confirm\n    scripts/pin_pipeline.py confirmed {candidate} for "
                         f"real but then hit an\n    unexpected error before it could "
                         f"bump/commit/merge: {reason}\n    (receipt: {receipt_path}).")
        print(f"INCONCLUSIVE (unexpected error): {reason}")
        return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pin", required=True, choices=("llama", "comfyui", "rocm"))
    ap.add_argument("--dry-run", action="store_true",
                    help="confirm the candidate but never bump, commit, or merge")
    args = ap.parse_args(argv)

    if args.pin == "rocm":
        print(f"--pin {args.pin} is not built yet (Phase 3)")
        return 1

    # Both pins share one dedicated worktree and one CHANGELOG.md - see
    # pipeline_lock()'s own docstring for why this must wrap the WHOLE run,
    # not just the write steps.
    try:
        with pipeline_lock():
            if args.pin == "llama":
                return run_llama_pipeline(dry_run=args.dry_run)
            return run_comfyui_pipeline(dry_run=args.dry_run)
    except PipelineLockBusy as e:
        print(f"INCONCLUSIVE: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
