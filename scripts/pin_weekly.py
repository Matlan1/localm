#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The weekly pin run: advance every runtime localm pins, and re-prove the current ones.

One invocation, per run:
  1. Reads every pin's currency (scripts/check_pins.py) for the report.
  2. For each runtime with an advancer: if upstream has a newer build, confirms it
     WORKS with localm on this machine (scripts/confirm_<x>_runtime.py), and on PASS
     bumps the pin, opens a PR, waits for CI and merges it. llama.cpp and ComfyUI
     delegate to scripts/pin_pipeline.py; the others run the same state machine here.
  3. For each runtime with a confirm script: re-confirms the build pinned TODAY
     (``--current``), so a regression in the driver, the OS or a dependency shows up
     even when upstream did not move.
  4. Writes a dated report under dev-notes/pin-pipeline/weekly/ and logs a genuine
     failure to issues/issues.txt.

A runtime whose confirm or bump script does not exist yet is reported as NOT BUILT,
never skipped silently and never counted as current.

Exit codes: 0 everything advanced or verified, nothing needs attention; 1 at least
one FAIL (a candidate that does not work, a current pin that no longer works, a
refused bump); 2 nothing failed but at least one result is INCONCLUSIVE, NOT BUILT or
a PR opened for review.

Usage (from the main checkout):
    python scripts/pin_weekly.py --dry-run     # confirm only, never bump, commit or merge
    python scripts/pin_weekly.py
    python scripts/pin_weekly.py --only koboldcpp,llama.cpp
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
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
NONE_NEWER = "NOTHING NEWER"
MERGED = "MERGED"
DRY_PASS = "PASS (dry run)"
REVIEW = "OPENED FOR REVIEW"
NOT_BUILT = "NOT BUILT"
SKIPPED = "SKIPPED"
NOT_RUN = "NOT RUN"
NOT_MEASURED = "NOT MEASURED"

CONFIRM_TIMEOUT_SECONDS = 3 * 3600
TEST_TIMEOUT_SECONDS = 30 * 60
LEASE_BUSY_EXIT = 75


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pp = _load("pin_pipeline", SCRIPTS / "pin_pipeline.py")
cp = _load("check_pins", SCRIPTS / "check_pins.py")

WEEKLY_DIR = pp.STATE_DIR / "weekly"


@dataclass
class Outcome:
    key: str
    title: str
    kind: str
    verdict: str
    detail: str = ""
    pinned: str = ""
    candidate: str = ""
    receipt: str = ""
    pr: int | None = None


@dataclass
class Advancer:
    key: str
    title: str
    confirm_script: str | None
    bump_script: str
    candidate: Callable[[], tuple[str, str] | None]
    tests: tuple[str, ...] = ()
    gpu: bool = True
    changelog: Callable[[str, str], str | None] = lambda old, new: None
    auto_merge: Callable[[str, str], bool] = lambda old, new: True
    delegate: Callable[[bool], int] | None = None
    verify_current_cmd: Callable[[Path, Path], list[str]] | None = None
    bump_args: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.confirm_script is None:
            self.auto_merge = lambda old, new: False


# --------------------------------------------------------------------------- #
#  Process helpers                                                            #
# --------------------------------------------------------------------------- #

def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        with contextlib.suppress(OSError):
            os.kill(pid, 9)


def run_cmd(cmd: list[str], *, cwd: Path, timeout: float, env: dict | None = None) -> tuple[int, str]:
    """Run *cmd*, returning (exit code, combined output). On timeout the whole
    process tree is killed and the exit code is 2 (INCONCLUSIVE)."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        out, _ = proc.communicate()
        return 2, (out or "") + f"\n(timed out after {int(timeout)}s; process tree killed)"
    return proc.returncode, out or ""


def scratch_dir(key: str) -> Path:
    return REPO.parent / f"{REPO.name}-pin-pipeline-{key}-scratch"


def _gpu_wrap(cmd: list[str], purpose: str) -> list[str]:
    return [sys.executable, str(pp.gpu_lease_script()), "run", "--purpose", purpose, "--", *cmd]


def _verdict_from_rc(rc: int) -> str:
    if rc == 0:
        return PASS
    if rc == 1:
        return FAIL
    return INCONCLUSIVE


def run_confirm(adv: Advancer, args: list[str], receipt: Path) -> tuple[str, str]:
    """Run the advancer's confirm script with *args*. Returns (verdict, detail);
    the verdict is NOT_RUN when the script never started (no GPU lease), which is
    no statement about the build."""
    script = REPO / adv.confirm_script
    work = scratch_dir(adv.key)
    work.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), *args, "--workdir", str(work), "--receipt", str(receipt)]
    if adv.gpu:
        try:
            cmd = _gpu_wrap(cmd, f"pin_weekly {adv.key} confirm")
        except pp.PipelineError as e:
            return NOT_RUN, str(e)
    rc, out = run_cmd(cmd, cwd=REPO, timeout=CONFIRM_TIMEOUT_SECONDS)
    if rc == LEASE_BUSY_EXIT:
        return NOT_RUN, "GPU busy this run (lease wait elapsed)"
    return _verdict_from_rc(rc), out[-1500:]


def _receipt_path(key: str, tag: str) -> Path:
    path = pp.STATE_DIR / "receipts" / f"{key}-{tag}-{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _receipt_summary(receipt: Path) -> str:
    try:
        data = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no readable receipt"
    checks = data.get("checks") if isinstance(data, dict) else None
    if not isinstance(checks, dict):
        return str(data.get("why", "")) if isinstance(data, dict) else ""
    parts = []
    for name, c in checks.items():
        status = c.get("status") if isinstance(c, dict) else c
        parts.append(f"{name}={status}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
#  Generic advance: detect -> confirm -> bump -> tests -> PR -> CI -> merge    #
# --------------------------------------------------------------------------- #

def advance(adv: Advancer, *, dry_run: bool) -> Outcome:
    """The state machine run_comfyui_pipeline implements, for any advancer with a
    confirm script and a bump script that follow dev-notes/pin-automation/CONTRACT.md."""
    out = Outcome(adv.key, adv.title, "advance", NONE_NEWER)
    for rel in (adv.confirm_script, adv.bump_script):
        if rel is not None and not (REPO / rel).is_file():
            out.verdict, out.detail = NOT_BUILT, f"{rel} does not exist"
            return out

    try:
        found = adv.candidate()
    except (pp.PipelineError, cp.FetchError, OSError, ValueError) as e:
        out.verdict, out.detail = INCONCLUSIVE, f"could not determine a candidate: {e}"
        return out
    if found is None:
        return out
    old, new = found
    out.pinned, out.candidate = old, new

    state = pp.load_state(pin=adv.key)
    skip = pp.should_skip(state, new)
    if skip is None and state.get("last_tag_tried") == new and state.get("verdict") == "REVIEW":
        skip = f"PR #{state.get('open_pr')} for {new} is already open for review"
    if skip:
        out.verdict, out.detail = SKIPPED, skip
        return out

    receipt = _receipt_path(adv.key, new)
    now_iso = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if adv.confirm_script is None:
        out.detail = "no runtime confirmation exists for this pin; the PR is opened for review"
        if dry_run:
            out.verdict = DRY_PASS
            return out
        return _bump_with_error_handling(adv, out, old, new, None, now_iso)
    out.receipt = str(receipt)
    verdict, detail = run_confirm(adv, ["--tag", new], receipt)
    if verdict == NOT_RUN:
        out.verdict, out.detail = INCONCLUSIVE, detail
        return out
    out.detail = _receipt_summary(receipt) or detail[-300:]
    if verdict == FAIL:
        reason = f"{adv.confirm_script} reported FAIL"
        pp.save_state({"last_tag_tried": new, "verdict": "FAIL", "timestamp": now_iso,
                       "receipt_path": str(receipt), "reason": reason}, pin=adv.key)
        pp.append_fail_issue(
            new, reason, receipt, pin=adv.key,
            summary=f"{adv.title} {new} does not work with localm on this hardware\n"
                    f"    scripts/pin_weekly.py ran {adv.confirm_script} against {new} for real "
                    f"and it did not pass ({out.detail}).")
        out.verdict = FAIL
        return out
    if verdict == INCONCLUSIVE:
        pp._record_inconclusive(new, receipt, pin=adv.key)
        out.verdict = INCONCLUSIVE
        out.detail = f"{out.detail}; {detail[-300:]}".strip("; ")
        return out
    if dry_run:
        out.verdict = DRY_PASS
        return out

    return _bump_with_error_handling(adv, out, old, new, receipt, now_iso)


def _bump_with_error_handling(adv: Advancer, out: Outcome, old: str, new: str,
                              receipt: Path | None, now_iso: str) -> Outcome:
    try:
        return _bump_and_merge(adv, out, old, new, receipt, now_iso)
    except pp.InfraError as e:
        pp._record_inconclusive(new, receipt, pin=adv.key, reason=str(e))
        out.verdict, out.detail = INCONCLUSIVE, f"infra: {e}"
    except pp.PipelineError as e:
        pp.save_state({"last_tag_tried": new, "verdict": "FAIL", "timestamp": now_iso,
                       "receipt_path": str(receipt or ""), "reason": str(e)}, pin=adv.key)
        pp.append_fail_issue(new, str(e), receipt, pin=adv.key)
        out.verdict, out.detail = FAIL, str(e)
    except Exception as e:  # noqa: BLE001 - a tooling crash after a PASS is logged, never a build FAIL
        reason = f"unexpected error after a PASS confirm: {type(e).__name__}: {e}"
        pp._record_inconclusive(new, receipt, pin=adv.key, reason=reason, force_issue=True,
                                issue_summary=f"{adv.title} {new}: unexpected pipeline error "
                                              f"after a PASS confirm: {reason}")
        out.verdict, out.detail = INCONCLUSIVE, reason
    return out


def close_superseded_prs(worktree: Path, key: str, keep_branch: str) -> list[int]:
    """Close every open pin-pipeline PR for *key* except the one on *keep_branch*, so a
    newer candidate replaces an older review-only PR instead of piling up beside it.
    Returns the closed PR numbers; raises InfraError when the PR list cannot be read."""
    prefix = f"claude/pin-pipeline-{key}-"
    proc = subprocess.run(["gh", "pr", "list", "--state", "open", "--json", "number,headRefName"],
                          cwd=worktree, capture_output=True, text=True)
    if proc.returncode != 0:
        raise pp.InfraError(f"gh pr list failed: {proc.stderr}")
    try:
        prs = json.loads(proc.stdout)
    except ValueError:
        raise pp.InfraError(f"could not parse gh pr list output: {proc.stdout!r}") from None
    closed = []
    for pr in prs:
        head = pr.get("headRefName", "")
        if head.startswith(prefix) and head != keep_branch:
            res = subprocess.run(
                ["gh", "pr", "close", str(pr["number"]), "--comment",
                 "Superseded by a newer upstream release."],
                cwd=worktree, capture_output=True, text=True)
            if res.returncode != 0:
                raise pp.InfraError(f"could not close superseded PR #{pr['number']}: {res.stderr}")
            closed.append(pr["number"])
    return closed


def _bump_and_merge(adv: Advancer, out: Outcome, old: str, new: str, receipt: Path | None,
                    now_iso: str) -> Outcome:
    worktree = pp.ensure_pipeline_worktree()
    close_superseded_prs(worktree, adv.key, f"claude/pin-pipeline-{adv.key}-{new}")
    branch = pp.prepare_bump_branch(worktree, new, pin=adv.key)

    bump_cmd = [sys.executable, str(worktree / adv.bump_script), *adv.bump_args, "--tag", new]
    if receipt is not None:
        bump_cmd += ["--receipt", str(receipt)]
    proc = subprocess.run(bump_cmd + ["--write"], cwd=worktree, env=pp._worktree_env(worktree),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise pp.PipelineError(f"{adv.bump_script} refused: {(proc.stdout + proc.stderr)[-800:]}")

    bullet = adv.changelog(old, new)
    if bullet:
        changelog = worktree / "CHANGELOG.md"
        changelog.write_text(
            pp.insert_changelog_bullet(changelog.read_text(encoding="utf-8"), bullet),
            encoding="utf-8")

    if adv.tests:
        rc, test_out = run_cmd(
            [sys.executable, "-m", "pytest", *adv.tests, "-m", "not integration", "-q"],
            cwd=worktree, timeout=TEST_TIMEOUT_SECONDS, env=pp._worktree_env(worktree))
        if rc != 0:
            raise pp.PipelineError(f"targeted tests failed after the bump:\n{test_out[-2000:]}")

    title = f"Advance the bundled {adv.title} to {new}"
    tested = ("The new build was installed and exercised on real hardware before the pin changed."
              if adv.confirm_script else
              "This build could not be exercised on this machine; review before merging.")
    body = (f"Moves the bundled {adv.title} from {old} to {new}. {tested}\n\n"
            "🤖 Generated with [Claude Code](https://claude.com/claude-code)\n")
    how = (f"confirmed with {adv.confirm_script}" if adv.confirm_script
           else "mechanical bump, opened for review")
    message = (f"chore({adv.key}): advance the pinned build to {new}\n\n"
               f"Automated: {how} before advancing from {old}.\n\n"
               "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\n")
    pp.commit_and_push(worktree, branch, new, old, message=message)
    pr = pp.open_pr(worktree, branch, new, old, title=title, body=body)
    out.pr = pr

    if not adv.auto_merge(old, new):
        pp.save_state({"last_tag_tried": new, "verdict": "REVIEW", "timestamp": now_iso,
                       "receipt_path": str(receipt or ""), "open_pr": pr}, pin=adv.key)
        out.verdict = REVIEW
        out.detail = f"PR #{pr} opened; this bump is not auto-merged"
        return out

    ci = pp.wait_for_ci(worktree)
    if ci == "GREEN":
        pp.merge_pr(pr, worktree, branch, new, old, title=title, body=body)
        pp.save_state({"last_tag_tried": new, "verdict": "PASS", "timestamp": now_iso,
                       "receipt_path": str(receipt or ""), "merged_pr": pr}, pin=adv.key)
        out.verdict, out.detail = MERGED, f"PR #{pr}: {old} -> {new}"
        return out
    if ci == "PENDING":
        pp._record_inconclusive(new, receipt, pin=adv.key, open_pr=pr)
        out.verdict, out.detail = INCONCLUSIVE, f"CI still pending on PR #{pr}"
        return out
    reason = f"CI red on PR #{pr}; left open, not merged"
    pp.save_state({"last_tag_tried": new, "verdict": "FAIL", "timestamp": now_iso,
                   "receipt_path": str(receipt or ""), "reason": reason, "open_pr": pr}, pin=adv.key)
    pp.append_fail_issue(new, reason, receipt, pin=adv.key)
    out.verdict, out.detail = FAIL, reason
    return out


def advance_delegated(adv: Advancer, *, dry_run: bool) -> Outcome:
    """llama.cpp and ComfyUI: pin_pipeline.py owns the whole flow and its state."""
    out = Outcome(adv.key, adv.title, "advance", NONE_NEWER)
    try:
        rc = adv.delegate(dry_run)
    except pp.PipelineError as e:
        out.verdict, out.detail = FAIL, str(e)
        return out
    state = pp.load_state(pin=adv.key)
    out.candidate = str(state.get("last_tag_tried", ""))
    out.receipt = str(state.get("receipt_path", ""))
    if rc == 1:
        out.verdict, out.detail = FAIL, str(state.get("reason", "see the pipeline output"))
    elif rc == 2:
        out.verdict, out.detail = INCONCLUSIVE, str(state.get("reason", "could not reach a verdict"))
    elif state.get("merged_pr") and state.get("verdict") == "PASS":
        out.verdict = MERGED
        out.pr = state.get("merged_pr")
    elif dry_run and state.get("verdict") == "PASS":
        out.verdict = DRY_PASS
    return out


# --------------------------------------------------------------------------- #
#  Current-pin verification                                                   #
# --------------------------------------------------------------------------- #

def verify_current(adv: Advancer) -> Outcome:
    """Re-confirm the build the repo pins today."""
    out = Outcome(adv.key, adv.title, "verify-current", INCONCLUSIVE)
    receipt = _receipt_path(adv.key, "current")
    out.receipt = str(receipt)
    if adv.verify_current_cmd is not None:
        work = scratch_dir(adv.key)
        work.mkdir(parents=True, exist_ok=True)
        try:
            cmd = adv.verify_current_cmd(work, receipt)
            if adv.gpu:
                cmd = _gpu_wrap(cmd, f"pin_weekly {adv.key} verify current")
        except pp.PipelineError as e:
            out.detail = str(e)
            return out
        rc, text = run_cmd(cmd, cwd=REPO, timeout=CONFIRM_TIMEOUT_SECONDS)
        if rc == LEASE_BUSY_EXIT:
            out.detail = "GPU busy this run (lease wait elapsed)"
            return out
        out.verdict = _verdict_from_rc(rc)
        out.detail = _receipt_summary(receipt) if receipt.exists() else text[-400:]
        return out
    if adv.confirm_script is None:
        out.verdict = NOT_MEASURED
        out.detail = "no runtime check exists for this pin on this machine"
        return out
    if not (REPO / adv.confirm_script).is_file():
        out.verdict, out.detail = NOT_BUILT, f"{adv.confirm_script} does not exist"
        return out
    out.verdict, out.detail = run_confirm(adv, ["--current"], receipt)
    if out.verdict == NOT_RUN:
        out.verdict = INCONCLUSIVE
    elif receipt.exists():
        out.detail = _receipt_summary(receipt)
    return out


def _report_current_failure(out: Outcome) -> None:
    pp.append_fail_issue(
        f"current-{int(time.time()) // 86400}", f"{out.title} (pinned build) {out.detail}",
        Path(out.receipt) if out.receipt else None, pin=out.key, kind="CURRENT-PIN-BROKEN",
        summary=f"the PINNED {out.title} build no longer works with localm on this machine\n"
                f"    scripts/pin_weekly.py re-confirmed the build the repo ships today and "
                f"it did not pass: {out.detail}")


# --------------------------------------------------------------------------- #
#  Candidate detection                                                        #
# --------------------------------------------------------------------------- #

def _github_candidate(repo: str, pin_file: str, pin_pattern: str, key) -> Callable[[], tuple[str, str] | None]:
    def find():
        pinned = cp._read_const(pin_file, pin_pattern)
        pin_key = key(pinned)
        if pin_key is None:
            raise cp.FetchError(f"cannot parse pinned version {pinned!r}")
        keyed = [(key(t), t) for t, _ in cp._github_releases(repo)]
        keyed = [(k, t) for k, t in keyed if k is not None and k > pin_key]
        if not keyed:
            return None
        return pinned, max(keyed)[1]
    return find


def _vendored_candidate(name: str) -> Callable[[], tuple[str, str] | None]:
    spec = next(v for v in cp._VENDORED if v[0] == name)
    _, rel, pattern, (source, ref), _tolerance = spec

    def find():
        pinned = cp._read_const(rel, pattern)
        pin_key = cp._semver_key(pinned)
        releases = cp._github_releases(ref) if source == "gh" else cp._npm_releases(ref)
        newer = [(cp._semver_key(t), t) for t, _ in releases]
        newer = [(k, t) for k, t in newer if k is not None and k > pin_key]
        if not newer:
            return None
        return pinned, re.sub(r"^v", "", max(newer)[1])
    return find


def _cuda_runtime_candidate() -> tuple[str, str] | None:
    text = cp._read_text("localm/setup_llama/cuda.py")
    m = cp._CUDA_RUNTIME_RE.search(text)
    if not m:
        raise cp.FetchError("_CUDA_RUNTIME_PIN not found in localm/setup_llama/cuda.py")
    pins = re.findall(r'"([a-z0-9._-]+)":\s*\(\s*"([^"]+)"', m.group(1))
    old, new = [], []
    for package, version in pins:
        latest, _ = cp._pypi_latest(package)
        old.append(f"{package}=={version}")
        new.append(f"{package}=={latest}")
    if old == new:
        return None
    return ",".join(old), ",".join(new)


def _gguf_node_candidate() -> tuple[str, str] | None:
    pinned = cp._read_const("localm/media/managed_comfy_fresh.py",
                            r'name="ComfyUI-GGUF",\s*repo="[^"]+",\s*commit="([0-9a-f]{40})"')
    head = cp._get_json("https://api.github.com/repos/city96/ComfyUI-GGUF/commits/main")
    try:
        head_sha = head["sha"]
    except (KeyError, TypeError) as e:
        raise cp.FetchError("unexpected commits response shape") from e
    return None if head_sha == pinned else (pinned, head_sha)


def _docker_base_candidate() -> tuple[str, str] | None:
    ref = cp._read_const("docker/Dockerfile",
                         r"^ARG UBUNTU_IMAGE=(ubuntu:[\d.]+@sha256:[0-9a-f]{64})$")
    image, _, pinned = ref.partition("@")
    repo, _, tag = image.partition(":")
    body = cp._get_json(f"https://hub.docker.com/v2/repositories/library/{repo}/tags/{tag}")
    try:
        current = body["digest"]
    except (KeyError, TypeError) as e:
        raise cp.FetchError("unexpected Docker Hub response shape") from e
    return None if current == pinned else (pinned, current)


def _rocm_candidate() -> tuple[str, str] | None:
    mod = _load("check_llama_rocm_pin", SCRIPTS / "check_llama_rocm_pin.py")
    pin = mod.pinned_tag()
    releases, err = mod.upstream_releases()
    if err or not releases:
        raise cp.FetchError(f"lemonade releases unreadable: {err or 'empty'}")
    pin_n = mod._build_number(pin)
    best = max(releases, key=lambda r: mod._build_number(r["tag"]) or -1)
    best_n = mod._build_number(best["tag"])
    if pin_n is None or best_n is None or best_n <= pin_n:
        return None
    return pin, best["tag"]


def _semver_major(tag: str) -> int | None:
    key = cp._semver_key(tag)
    return key[0] if key else None


def _same_major_only(old: str, new: str) -> bool:
    return _semver_major(old) is not None and _semver_major(old) == _semver_major(new)


def _llama_delegate(dry_run: bool) -> int:
    return pp.run_llama_pipeline(dry_run=dry_run)


def _comfyui_delegate(dry_run: bool) -> int:
    return pp.run_comfyui_pipeline(dry_run=dry_run)


def _llama_verify_cmd(work: Path, receipt: Path) -> list[str]:
    pinned = _load("check_llama_pin", SCRIPTS / "check_llama_pin.py").pinned_tag()
    return [sys.executable, str(SCRIPTS / "confirm_llama_runtime.py"), "--tag", pinned,
            "--backend", "cpu", "--backend", "vulkan", "--workdir", str(work),
            "--receipt", str(receipt)]


def _comfyui_verify_cmd(work: Path, receipt: Path) -> list[str]:
    tag = _load("check_comfyui_pin", SCRIPTS / "check_comfyui_pin.py")._pinned_version()
    commit = cp._read_const("localm/media/managed_comfy_fresh.py",
                            r'^COMFYUI_PINNED_COMMIT = "([0-9a-f]{40})"')
    return [sys.executable, str(SCRIPTS / "confirm_comfyui_runtime.py"), "--tag", tag,
            "--commit", commit, "--workdir", str(work), "--receipt", str(receipt),
            "--phase", "all", "--require-gpu"]


def _bullet(what: str, how: str) -> Callable[[str, str], str]:
    def make(old: str, new: str) -> str:
        return f"- **The bundled {what} moved from {old} to {new}.** {how}\n"
    return make


def build_advancers() -> list[Advancer]:
    return [
        Advancer("llama", "llama.cpp", "scripts/confirm_llama_runtime.py",
                 "scripts/bump_llama_pin.py", candidate=lambda: None,
                 delegate=_llama_delegate, verify_current_cmd=_llama_verify_cmd),
        Advancer("comfyui", "ComfyUI", "scripts/confirm_comfyui_runtime.py",
                 "scripts/bump_comfyui_pin.py", candidate=lambda: None,
                 delegate=_comfyui_delegate, verify_current_cmd=_comfyui_verify_cmd),
        Advancer("rocm", "AMD ROCm llama.cpp build", "scripts/confirm_rocm_runtime.py",
                 "scripts/bump_rocm_pin.py", candidate=_rocm_candidate,
                 tests=("tests/test_llama_rocm_pin_currency.py", "tests/test_bump_rocm_pin.py",
                        "tests/test_llama_pin_constant_and_currency.py"),
                 changelog=_bullet("AMD ROCm llama.cpp runtime",
                                   "An existing install picks it up with `localm setup-llama --force`.")),
        Advancer("koboldcpp", "koboldcpp", "scripts/confirm_koboldcpp_runtime.py",
                 "scripts/bump_koboldcpp_pin.py",
                 candidate=_github_candidate("LostRuins/koboldcpp", "localm/media/koboldcpp/pins.py",
                                             r'^TAG = "([^"]+)"', cp._semver_key),
                 tests=("tests/test_koboldcpp_runtime.py", "tests/test_koboldcpp_server.py",
                        "tests/test_bump_koboldcpp_pin.py"),
                 changelog=_bullet("koboldcpp runtime (native music generation)",
                                   "An existing install picks it up the next time music generation starts.")),
        Advancer("sdcpp", "stable-diffusion.cpp", "scripts/confirm_sdcpp_runtime.py",
                 "scripts/bump_sdcpp_pin.py",
                 candidate=_github_candidate("leejet/stable-diffusion.cpp", "localm/media/sdcpp/pins.py",
                                             r'^TAG = "([^"]+)"', cp._sdcpp_key),
                 tests=("tests/test_sdcpp_binding.py", "tests/test_sdcpp_runner.py",
                        "tests/test_sdcpp_runtime.py", "tests/test_bump_sdcpp_pin.py"),
                 changelog=_bullet("stable-diffusion.cpp runtime",
                                   "An existing install picks it up the next time image generation starts.")),
        Advancer("uv", "uv installer", "scripts/confirm_uv_runtime.py", "scripts/bump_uv_pin.py",
                 candidate=_github_candidate("astral-sh/uv", "setup.sh",
                                             r'^UV_INSTALLER_VERSION="([^"]+)"', cp._semver_key),
                 tests=("tests/test_uv_installer_pin.py", "tests/test_docker_image_files.py",
                        "tests/test_bump_uv_pin.py"), gpu=False),
        *_review_only_advancers(),
    ]


def _review_only_advancers() -> list[Advancer]:
    """Pins that cannot be exercised on this machine (or whose output ships to every
    browser): the bump is mechanical, CI runs on the PR, and a person merges it."""
    vendored = [
        Advancer(f"vendored-{name.lower().replace('.', '')}", f"vendored {name}", None,
                 "scripts/bump_vendored_js.py", candidate=_vendored_candidate(name),
                 bump_args=("--lib", name), gpu=False)
        for name in ("marked", "DOMPurify", "highlight.js", "KaTeX")
    ]
    return [
        *vendored,
        Advancer("cuda-runtime", "Linux CUDA runtime wheels", None,
                 "scripts/bump_cuda_runtime_pin.py", candidate=_cuda_runtime_candidate, gpu=False),
        Advancer("gguf-node", "ComfyUI-GGUF node", None, "scripts/bump_gguf_node_pin.py",
                 candidate=_gguf_node_candidate, gpu=False),
        Advancer("docker-base", "Docker base image", None, "scripts/bump_docker_base_pin.py",
                 candidate=_docker_base_candidate, gpu=False),
    ]


# --------------------------------------------------------------------------- #
#  Report                                                                     #
# --------------------------------------------------------------------------- #

def severity(outcomes: list[Outcome]) -> int:
    if any(o.verdict == FAIL for o in outcomes):
        return 1
    if any(o.verdict in (INCONCLUSIVE, NOT_BUILT, REVIEW) for o in outcomes):
        return 2
    return 0


def render_report(rows: list, outcomes: list[Outcome], started: _dt.datetime, dry_run: bool) -> str:
    lines = [f"# Weekly pin run {started.strftime('%Y-%m-%d %H:%M UTC')}"
             + (" (dry run)" if dry_run else ""), ""]
    lines += ["## Advancing", "", "| runtime | result | pinned | candidate | detail |", "|---|---|---|---|---|"]
    for o in (x for x in outcomes if x.kind == "advance"):
        lines.append(f"| {o.title} | {o.verdict} | {o.pinned or '-'} | {o.candidate or '-'} | "
                     f"{(o.detail or '-').replace('|', '/')[:240]} |")
    lines += ["", "## Pinned build still works with localm", "",
              "| runtime | result | detail |", "|---|---|---|"]
    for o in (x for x in outcomes if x.kind == "verify-current"):
        lines.append(f"| {o.title} | {o.verdict} | {(o.detail or '-').replace('|', '/')[:240]} |")
    lines += ["", "## Currency of every pin", "", cp.format_table(rows)]
    return "\n".join(lines) + "\n"


def write_report(text: str, outcomes: list[Outcome], started: _dt.datetime) -> Path:
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y-%m-%d")
    path = WEEKLY_DIR / f"{stamp}.md"
    path.write_text(text, encoding="utf-8")
    (WEEKLY_DIR / "latest.md").write_text(text, encoding="utf-8")
    (WEEKLY_DIR / f"{stamp}.json").write_text(
        json.dumps([asdict(o) for o in outcomes], indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #

def run_weekly(advancers: list[Advancer], *, dry_run: bool, verify: bool = True) -> tuple[int, Path]:
    started = _dt.datetime.now(_dt.UTC)
    rows = cp.run_checks(cp.build_registry(), started)
    outcomes: list[Outcome] = []
    for adv in advancers:
        out = (advance_delegated(adv, dry_run=dry_run) if adv.delegate
               else advance(adv, dry_run=dry_run))
        print(f"[advance] {adv.title}: {out.verdict} {out.detail[:160]}")
        outcomes.append(out)
    if verify:
        for adv in advancers:
            advanced = next((o for o in outcomes if o.key == adv.key and o.kind == "advance"), None)
            if advanced is not None and advanced.verdict == MERGED:
                outcomes.append(Outcome(adv.key, adv.title, "verify-current", PASS,
                                        "confirmed moments ago as the candidate that was just merged"))
                continue
            out = verify_current(adv)
            print(f"[verify-current] {adv.title}: {out.verdict} {out.detail[:160]}")
            if out.verdict == FAIL:
                _report_current_failure(out)
            outcomes.append(out)
    report = render_report(rows, outcomes, started, dry_run)
    path = write_report(report, outcomes, started)
    print(f"report: {path}")
    return severity(outcomes), path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="confirm candidates but never bump, commit or merge")
    ap.add_argument("--only", default="",
                    help="comma-separated runtime keys or titles (llama, comfyui, rocm, koboldcpp, sdcpp)")
    ap.add_argument("--skip-verify-current", action="store_true",
                    help="do not re-confirm the builds pinned today")
    args = ap.parse_args(argv)

    advancers = build_advancers()
    if args.only:
        wanted = {w.strip().lower() for w in args.only.split(",") if w.strip()}
        advancers = [a for a in advancers if a.key.lower() in wanted or a.title.lower() in wanted]
        if not advancers:
            print(f"no runtime matches --only {args.only!r}", file=sys.stderr)
            return 2

    try:
        with pp.pipeline_lock():
            pp.sync_main_checkout()
            code, _ = run_weekly(advancers, dry_run=args.dry_run,
                                 verify=not args.skip_verify_current)
            return code
    except pp.PipelineLockBusy as e:
        print(f"INCONCLUSIVE: {e}")
        return 2
    except pp.InfraError as e:
        print(f"INCONCLUSIVE (infra, before any runtime was examined): {e}")
        return 2
    except pp.PipelineError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
