#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Entry point for the Tier 2 GPU-split real-hardware gate (issues/issues.txt
GPU-SPLIT-TESTING). Runs on the MAINTAINER's machine (not the rented box):
optionally launches a Lambda Cloud instance (NVIDIA/vulkan run only - see
README.md for why Hot Aisle/amd-rocm is launch=manual), waits for it, uploads
and runs provision_remote.sh, runs the real-hardware pytest gate over ssh
inside a hard wall-clock timeout, and ALWAYS tears down - on success, on a
test failure, on a timeout, on a harness error, or on Ctrl-C.

Usage (see scripts/tier2_gpu_split/README.md for the full walkthrough):

    # Dry run - no ssh, no cloud API calls, no charges:
    python run_gate.py --backend vulkan --dry-run

    # NVIDIA/vulkan run, Lambda launches and terminates the instance:
    export LAMBDA_API_KEY=...
    python run_gate.py --backend vulkan --launch lambda \\
        --ssh-key-name my-lambda-key --local-ssh-key ~/.ssh/my-lambda-key \\
        --ref master

    # AMD/amd-rocm run - launch the Hot Aisle box by hand first (no verified
    # API for this harness to call - see README.md), then:
    python run_gate.py --backend amd-rocm --launch manual \\
        --ssh-host <ip> --ssh-user ubuntu --local-ssh-key ~/.ssh/hotaisle_key \\
        --ref master

Exit codes: 0 all tests passed, and teardown was either CONFIRMED (Lambda) or
not applicable for this launch mode (manual/Hot Aisle always prints its own
"MANUAL TEARDOWN REQUIRED" reminder regardless - check the log); 1 same, but
a test failed; 2 a harness/provisioning error before a pass/fail could be
determined; 3 teardown was attempted (Lambda only) and definitively FAILED to
confirm, OR an instance may have been created but never learned an id (a
possible untracked orphan) - a real billing risk, surfaced regardless of test
outcome; 130 the run was interrupted by Ctrl-C/SIGTERM (the conventional
128+SIGINT code) - teardown was attempted before exiting, read the log to see
whether it confirmed. Check the provider dashboard yourself immediately on
exit code 3 OR 130.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_selection import resolve_live_sizes  # noqa: E402 (needs sys.path set first)

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"
REPO_URL_DEFAULT = "https://github.com/Matlan1/localm.git"

# Instance specs and pricing as verified live 2026-07-29 (see README.md's
# sourcing notes) - prices drift; the script prints them and the "confirm
# before paying" reminder rather than trusting them blindly.
INSTANCE_SPECS = {
    "vulkan": {
        "provider": "lambda",
        "instance_type_name": "gpu_2x_a6000",
        "hourly_rate_usd": 1.60,
        "fallback_instance_type_name": "gpu_2x_a100",
        "fallback_hourly_rate_usd": 3.98,
    },
    "amd-rocm": {
        # Hot Aisle - launch=manual only, see the module docstring and README.md
        # for why: this harness has no confidently-verified API for it.
        "provider": "manual",
        "hourly_rate_usd": 5.98,
    },
}


class GateError(Exception):
    """A harness/provisioning failure before or during the run (exit 2),
    distinct from a TEST failure (exit 1) and a teardown-confirmation failure
    (exit 3, which outranks both - a billing risk beats a result report)."""


def log(msg: str) -> None:
    print(f"[run_gate] {msg}", flush=True)


# --------------------------------------------------------------------------- #
#  Lambda Cloud API (NVIDIA run only) - see README.md: constructed from
#  Lambda's long-stable, publicly documented REST shape, but NOT exercised
#  against a live account in this session (no key available). Fails loudly
#  (raises GateError with the raw response body) on any non-2xx response
#  rather than guessing on drift.
# --------------------------------------------------------------------------- #

def _bounded_timeout(remaining_s: float, default: int) -> int:
    """A sub-call's timeout must never exceed what is actually left of its
    caller's own deadline - a fixed literal here (e.g. a flat 30s HTTP
    timeout invoked from inside a loop whose OWN deadline is only 10s away)
    lets a single sub-call overshoot the caller's budget, which compounds
    into the run's overall --timeout-minutes ceiling silently not holding.
    Always >= 1 so a near-zero remaining budget still gets one last real
    attempt rather than a zero/negative timeout (which some subprocess/socket
    APIs treat as "no timeout" - the opposite of intended)."""
    return max(1, min(default, int(remaining_s)))


def _lambda_request(method: str, path: str, api_key: str, body=None,
                     timeout: int = 30) -> dict:
    url = f"{LAMBDA_API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization",
                    "Basic " + base64.b64encode(f"{api_key}:".encode()).decode())
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        raise GateError(f"Lambda API {method} {path} -> HTTP {e.code}: {raw}") from e
    except urllib.error.URLError as e:
        raise GateError(f"Lambda API {method} {path} -> {e}") from e


def lambda_pick_region(api_key: str, instance_type_name: str,
                        fallback_instance_type_name=None) -> tuple:
    resp = _lambda_request("GET", "/instance-types", api_key)
    data = resp.get("data", {})
    entry = data.get(instance_type_name)
    if entry and entry.get("regions_with_capacity_available"):
        return instance_type_name, entry["regions_with_capacity_available"][0]["name"]
    if fallback_instance_type_name:
        entry = data.get(fallback_instance_type_name)
        if entry and entry.get("regions_with_capacity_available"):
            log(f"WARNING: no capacity for {instance_type_name} anywhere right "
                f"now - falling back to {fallback_instance_type_name}")
            return (fallback_instance_type_name,
                    entry["regions_with_capacity_available"][0]["name"])
    raise GateError(
        f"no capacity available for {instance_type_name} or its fallback - "
        f"check https://cloud.lambda.ai/instances for current availability")


def lambda_launch(api_key: str, instance_type_name: str, region_name: str,
                   ssh_key_names: list, name: str) -> str:
    body = {
        "region_name": region_name,
        "instance_type_name": instance_type_name,
        "ssh_key_names": ssh_key_names,
        "name": name,
        "quantity": 1,
    }
    resp = _lambda_request("POST", "/instance-operations/launch", api_key, body)
    ids = resp.get("data", {}).get("instance_ids", [])
    if not ids:
        raise GateError(f"Lambda launch returned no instance_ids: {resp!r}")
    return ids[0]


def lambda_wait_for_active(api_key: str, instance_id: str, timeout_s: int) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        req_timeout = _bounded_timeout(deadline - time.monotonic(), default=30)
        resp = _lambda_request("GET", "/instances", api_key, timeout=req_timeout)
        for inst in resp.get("data", []):
            if inst.get("id") == instance_id:
                if inst.get("status") == "active":
                    return inst
                if inst.get("status") in ("terminated", "terminating"):
                    raise GateError(
                        f"instance {instance_id} entered status "
                        f"{inst['status']!r} while waiting to become active")
        time.sleep(_bounded_timeout(deadline - time.monotonic(), default=10))
    raise GateError(f"instance {instance_id} did not become active within {timeout_s}s")


def lambda_terminate(api_key: str, instance_id: str) -> None:
    _lambda_request("POST", "/instance-operations/terminate", api_key,
                     {"instance_ids": [instance_id]})


def lambda_confirm_terminated(api_key: str, instance_id: str, timeout_s: int) -> bool:
    """True once the instance no longer appears in the running list. This is
    the POSITIVE confirmation the cost ceiling depends on - termination is
    never reported as done just because the terminate call returned 2xx."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        req_timeout = _bounded_timeout(deadline - time.monotonic(), default=30)
        try:
            resp = _lambda_request("GET", "/instances", api_key, timeout=req_timeout)
        except GateError:
            time.sleep(_bounded_timeout(deadline - time.monotonic(), default=10))
            continue
        if not any(i.get("id") == instance_id for i in resp.get("data", [])):
            return True
        time.sleep(_bounded_timeout(deadline - time.monotonic(), default=10))
    return False


# --------------------------------------------------------------------------- #
#  ssh/scp orchestration                                                       #
# --------------------------------------------------------------------------- #

def ssh_run(host: str, user: str, key_path: str, command: str,
            timeout_s: int) -> subprocess.CompletedProcess:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=30", "-i", key_path, f"{user}@{host}", command]
    log(f"+ ssh {user}@{host} {command!r} (timeout {timeout_s}s)")
    return subprocess.run(cmd, timeout=timeout_s)


def scp_put(host: str, user: str, key_path: str, local_path: str, remote_path: str,
            timeout_s: int = 60) -> None:
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new", "-i", key_path,
           local_path, f"{user}@{host}:{remote_path}"]
    subprocess.run(cmd, check=True, timeout=timeout_s)


def wait_for_ssh(host: str, user: str, key_path: str, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        attempt_timeout = _bounded_timeout(deadline - time.monotonic(), default=20)
        try:
            r = ssh_run(host, user, key_path, "echo ready", timeout_s=attempt_timeout)
            if r.returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(_bounded_timeout(deadline - time.monotonic(), default=10))
    raise GateError(f"ssh to {user}@{host} not available within {timeout_s}s")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=sorted(INSTANCE_SPECS), required=True)
    p.add_argument("--ref", default="master", help="git branch/tag/commit to test")
    p.add_argument("--repo-url", default=REPO_URL_DEFAULT)
    p.add_argument("--timeout-minutes", type=int, default=60,
                    help="hard wall-clock budget; teardown always runs at or "
                         "before this regardless of test outcome")
    p.add_argument("--launch", choices=["lambda", "manual"], default="manual",
                    help="'lambda': this script calls Lambda Cloud's API to "
                         "launch+terminate (vulkan backend only - UNVERIFIED "
                         "against a live account, see README.md). 'manual': "
                         "you already launched the box (required for "
                         "amd-rocm/Hot Aisle) and pass --ssh-host")
    p.add_argument("--lambda-api-key-env", default="LAMBDA_API_KEY",
                    help="name of the env var holding the Lambda API key "
                         "(never pass the key itself as an argument)")
    p.add_argument("--lambda-instance-type", default=None)
    p.add_argument("--ssh-key-name", default=None,
                    help="SSH key name already registered with the provider (lambda launch)")
    p.add_argument("--local-ssh-key", default=None,
                    help="path to the private key file for ssh/scp (required "
                         "for any non-dry-run)")
    p.add_argument("--ssh-host", default=None, help="required when --launch manual")
    p.add_argument("--ssh-user", default="ubuntu")
    p.add_argument("--dry-run", action="store_true",
                    help="resolve real model sizes and render the exact remote "
                         "commands - no ssh, no cloud API calls, no charges")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    return p


def _remote_test_command(backend: str) -> str:
    return (f"cd localm-tier2 && LOCALM_TEST_REAL_MULTI_GPU=1 "
            f"LOCALM_TEST_TIER2_BACKEND={backend} "
            f".venv/bin/pytest -m real_multi_gpu_hardware "
            f"tests/test_gpu_split_real_hardware.py -v")


def _provision_command(backend: str, ref: str, repo_url: str) -> str:
    return (f"bash provision_remote.sh {shlex.quote(backend)} {shlex.quote(ref)} "
            f"{shlex.quote(repo_url)}")


def _dry_run(args: argparse.Namespace) -> int:
    log("DRY RUN - no ssh, no cloud API calls, no charges")
    log("resolving live Hugging Face candidate sizes for the combined-VRAM test...")
    sized = resolve_live_sizes()
    if not sized:
        log("  WARNING: resolved zero candidates - either a network problem, or "
            "every repo/file in model_selection.CANDIDATE_TABLE has drifted")
    for c in sized:
        log(f"  {c.repo_id}/{c.filename}: {c.size_bytes / 1024 ** 3:.1f} GB")

    log("would launch:")
    if args.launch == "lambda":
        spec = INSTANCE_SPECS[args.backend]
        log(f"  Lambda {args.lambda_instance_type or spec['instance_type_name']} "
            f"(fallback {spec.get('fallback_instance_type_name', 'none')}), "
            f"region auto-picked from live capacity")
    else:
        log(f"  manual target: {args.ssh_user}@{args.ssh_host or '<--ssh-host required>'}")
    log("would upload: provision_remote.sh")
    log(f"would run: {_provision_command(args.backend, args.ref, args.repo_url)}")
    log(f"would run: {_remote_test_command(args.backend)}")
    log(f"would enforce a {args.timeout_minutes}-minute wall-clock timeout, then "
        f"ALWAYS tear down (terminate + confirm, or print a manual-termination reminder)")
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.launch == "manual" and not args.ssh_host:
        log("ERROR: --launch manual requires --ssh-host")
        return 2
    if not args.local_ssh_key or not Path(args.local_ssh_key).is_file():
        log(f"ERROR: --local-ssh-key must point at a real private key file "
            f"(got {args.local_ssh_key!r})")
        return 2

    deadline = time.monotonic() + args.timeout_minutes * 60
    state = {"instance_id": None, "api_key": None, "host": args.ssh_host,
             "possible_orphan": False}
    teardown_state = {"in_progress": False, "done": False, "result": None}

    def remaining() -> int:
        return max(1, int(deadline - time.monotonic()))

    def teardown():
        """Returns True (confirmed torn down), False (confirmation failed, or
        a possible untracked orphan - exit code 3, a billing risk), or None
        (manual provider - not applicable, the harness cannot confirm it).

        Idempotent and reentry-safe: a signal arriving while a teardown is
        already in flight (e.g. the normal post-test path is mid-way through
        lambda_confirm_terminated's 300s poll) must not re-enter the Lambda
        terminate+confirm cycle a second time, and must not let a SIGINT/
        SIGTERM unwind through the ORIGINAL in-progress call - see
        handle_signal below, which never exits while a teardown it did not
        start is still running."""
        if teardown_state["done"]:
            return teardown_state["result"]
        if teardown_state["in_progress"]:
            return None  # another call already owns this - do not duplicate
        teardown_state["in_progress"] = True
        result = None
        try:
            if args.launch == "lambda" and state["instance_id"] is not None:
                log(f"terminating Lambda instance {state['instance_id']}...")
                try:
                    lambda_terminate(state["api_key"], state["instance_id"])
                    if lambda_confirm_terminated(state["api_key"], state["instance_id"],
                                                  timeout_s=300):
                        log(f"CONFIRMED terminated: {state['instance_id']}")
                        result = True
                    else:
                        result = False
                except Exception as e:  # noqa: BLE001 - teardown must never
                    # crash; every failure funnels into the warning below.
                    log(f"terminate call raised: {e}")
                    result = False
                if result is False:
                    log(f"!!! COULD NOT CONFIRM TERMINATION of "
                        f"{state['instance_id']} - CHECK "
                        f"https://cloud.lambda.ai/instances YOURSELF NOW, "
                        f"this may still be billing !!!")
            elif state["possible_orphan"]:
                # The launch request may have created a billable instance
                # even though no id was ever recorded (see the launch
                # try/except below) - there is nothing to call terminate()
                # on, so the only recourse is a by-NAME dashboard check.
                log("!!! NO INSTANCE ID WAS EVER RECORDED, but the launch "
                    "request may have created one anyway - CHECK "
                    "https://cloud.lambda.ai/instances FOR A NAME STARTING "
                    "WITH 'tier2-gpu-split-gate' AND TERMINATE IT "
                    "YOURSELF. !!!")
                result = False
            elif state["host"]:
                log(f"!!! MANUAL TEARDOWN REQUIRED: TERMINATE "
                    f"{args.ssh_user}@{state['host']} NOW. This harness has "
                    f"no verified API for this provider (see README.md) and "
                    f"cannot confirm termination for you. !!!")
                result = None
        finally:
            teardown_state["result"] = result
            teardown_state["done"] = True
            teardown_state["in_progress"] = False
        return result

    def handle_signal(signum, _frame):
        if teardown_state["in_progress"] or teardown_state["done"]:
            log(f"received signal {signum} - a teardown is already in "
                f"progress/done, letting it finish rather than re-entering "
                f"or exiting mid-call")
            return
        log(f"received signal {signum} ({signal.Signals(signum).name}) - "
            f"tearing down before exit")
        result = teardown()
        # 130 (128+SIGINT) is the conventional Unix code for a
        # signal-interrupted process - documented here and in README.md as a
        # 5th, deliberate exit code, distinct from exit 3's specific "Lambda
        # teardown definitively failed to confirm" meaning.
        os._exit(3 if result is False else 130)

    # Install signal handling as the FIRST thing this function does, right
    # after the state it closes over exists - minimizing the window where a
    # Ctrl-C still carries Python's default disposition (an uncaught
    # KeyboardInterrupt that would skip teardown() - and its manual-
    # termination reminder - entirely).
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    tests_passed = False
    try:
        if args.launch == "lambda":
            state["api_key"] = os.environ.get(args.lambda_api_key_env)
            if not state["api_key"]:
                raise GateError(f"environment variable {args.lambda_api_key_env} is not set")
            spec = INSTANCE_SPECS[args.backend]
            instance_type = args.lambda_instance_type or spec["instance_type_name"]
            log(f"picking a region with capacity for {instance_type}...")
            instance_type, region = lambda_pick_region(
                state["api_key"], instance_type, spec.get("fallback_instance_type_name"))
            log(f"launching {instance_type} in {region}...")
            launch_name = "tier2-gpu-split-gate"
            try:
                state["instance_id"] = lambda_launch(
                    state["api_key"], instance_type, region,
                    [args.ssh_key_name] if args.ssh_key_name else [],
                    name=launch_name)
            except Exception:
                # The launch POST may have created a billable instance
                # server-side even though we never learned its id (e.g. a
                # network read error/reset while parsing the response) -
                # flag it so teardown() gives an actionable by-NAME warning
                # instead of silently doing nothing (it has no id to
                # terminate otherwise).
                state["possible_orphan"] = True
                raise
            log(f"launched {state['instance_id']}, waiting to become active...")
            inst = lambda_wait_for_active(state["api_key"], state["instance_id"],
                                           timeout_s=remaining())
            state["host"] = inst.get("ip")
            if not state["host"]:
                raise GateError(f"active instance has no IP: {inst!r}")
            log(f"instance active at {state['host']}")

        if not state["host"]:
            raise GateError("no ssh target - nothing to connect to")

        log(f"waiting for ssh at {args.ssh_user}@{state['host']}...")
        wait_for_ssh(state["host"], args.ssh_user, args.local_ssh_key,
                     timeout_s=min(remaining(), 300))

        script_path = Path(__file__).resolve().parent / "provision_remote.sh"
        log("uploading provision_remote.sh...")
        scp_put(state["host"], args.ssh_user, args.local_ssh_key,
                str(script_path), "provision_remote.sh", timeout_s=min(remaining(), 60))

        log("provisioning the remote box...")
        r = ssh_run(state["host"], args.ssh_user, args.local_ssh_key,
                    _provision_command(args.backend, args.ref, args.repo_url),
                    timeout_s=remaining())
        if r.returncode != 0:
            raise GateError(f"provisioning failed (ssh exit {r.returncode})")

        log("running the real-hardware gate tests...")
        r = ssh_run(state["host"], args.ssh_user, args.local_ssh_key,
                    _remote_test_command(args.backend), timeout_s=remaining())
        tests_passed = (r.returncode == 0)
        if not tests_passed:
            log(f"gate tests FAILED (pytest exit {r.returncode})")
    except Exception as e:  # noqa: BLE001 - any failure here still must tear down
        log(f"HARNESS ERROR: {e}")
        confirmed = teardown()
        return 3 if confirmed is False else 2

    confirmed = teardown()
    if confirmed is False:
        return 3
    return 0 if tests_passed else 1


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.launch == "lambda" and args.backend != "vulkan":
        log("ERROR: --launch lambda is only wired for the vulkan/NVIDIA "
            "backend; amd-rocm (Hot Aisle) has no verified API in this "
            "harness - launch it manually and pass --launch manual --ssh-host. "
            "See README.md.")
        return 2

    spec = INSTANCE_SPECS[args.backend]
    rate = spec.get("hourly_rate_usd", 0.0)
    fallback_rate = spec.get("fallback_hourly_rate_usd")
    # The ceiling shown and confirmed here must be the WORST CASE this run can
    # actually incur, not just the primary instance's rate: lambda_pick_region()
    # (called later, inside _run()) silently substitutes the pricier fallback
    # instance type whenever the primary lacks capacity - which README.md
    # itself calls common for Lambda - with no re-confirmation when that
    # happens. A ceiling based only on the cheaper primary rate would then be
    # quietly wrong for the actual amount billed.
    worst_case_rate = max(rate, fallback_rate) if fallback_rate else rate
    ceiling = worst_case_rate * (args.timeout_minutes / 60)
    log(f"backend={args.backend}  timeout={args.timeout_minutes}min  "
        f"rate=${rate:.2f}/hr" + (
            f" (fallback ${fallback_rate:.2f}/hr may be substituted if the "
            f"primary instance type has no capacity)" if fallback_rate else ""
        ) + f"  COST CEILING=${ceiling:.2f} (worst case)")
    log("this ceiling holds only if teardown succeeds - the script always "
        "attempts it, but a confirmation failure is its own loud exit code "
        "(3), never silently folded into success. Confirm current pricing "
        "on the provider's own page before paying - see README.md.")

    if args.dry_run:
        return _dry_run(args)

    if not args.yes:
        resp = input(f"Proceed and incur up to ~${ceiling:.2f}? [y/N] ").strip().lower()
        if resp != "y":
            log("aborted (pass --yes to skip this prompt)")
            return 2

    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
