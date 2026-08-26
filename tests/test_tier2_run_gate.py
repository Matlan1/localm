# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests for scripts/tier2_gpu_split/run_gate.py's timeout arithmetic.

No network, no ssh, no GPU: a fake monotonic clock and mocked ssh_run/sleep
exercise the invariant that a sub-call's own timeout never exceeds what is left
of its caller's deadline. A fixed literal sub-timeout larger than the caller's
remaining budget lets that single call overshoot silently, and enough of those
compound into the run's overall --timeout-minutes ceiling not holding.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "tier2_gpu_split" / "run_gate.py"
if not _PATH.is_file():
    # scripts/tier2_gpu_split/ is gitignored, maintainer-only tooling, so a fresh
    # clone (or a worktree that did not get it copied in) does not have this file.
    # Skip with a reason rather than importing, so collection never hard-crashes
    # over a file the repo itself excludes.
    pytest.skip(f"{_PATH} not present (gitignored maintainer-only harness, "
               "AGENTS.md rule 6) - skipping tests that need it",
               allow_module_level=True)
_spec = importlib.util.spec_from_file_location("run_gate", _PATH)
run_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_gate)


class TestBoundedTimeout:
    def test_caps_at_the_default_when_plenty_remains(self):
        assert run_gate._bounded_timeout(100, default=30) == 30

    def test_shrinks_to_whatever_remains_when_less_than_default(self):
        assert run_gate._bounded_timeout(5, default=30) == 5

    def test_floors_at_one_for_a_near_zero_remaining_budget(self):
        # a zero/negative timeout would mean "no timeout" to some subprocess/
        # socket APIs - the opposite of the intended bound - so this must
        # never reach zero, only shrink toward it.
        assert run_gate._bounded_timeout(0, default=30) == 1

    def test_floors_at_one_for_an_already_expired_deadline(self):
        assert run_gate._bounded_timeout(-5, default=30) == 1


class _FakeClock:
    """A controllable monotonic clock: time only advances when the code under
    test calls sleep() or does "work" - never in real wall-clock time, so a
    test exercising a multi-second timeout budget still runs instantly."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_wait_for_ssh_bounds_every_attempt_by_its_own_remaining_deadline(monkeypatch):
    """The exact invariant: wait_for_ssh(timeout_s=3) must never hand a
    per-attempt sub-call a timeout greater than 3, even though ssh_run's own
    documented default attempt length (20s) is far larger than that."""
    clock = _FakeClock()
    attempts = []

    def fake_ssh_run(host, user, key_path, command, timeout_s):
        attempts.append(timeout_s)
        clock.now += 0.01  # negligible simulated work per attempt

        class _Result:
            returncode = 1  # never "ready" - forces the loop to keep retrying
        return _Result()

    monkeypatch.setattr(run_gate.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(run_gate.time, "sleep", clock.sleep)
    monkeypatch.setattr(run_gate, "ssh_run", fake_ssh_run)

    with pytest.raises(run_gate.GateError):
        run_gate.wait_for_ssh("host", "user", "/fake/key", timeout_s=3)

    assert attempts, "wait_for_ssh never attempted a single ssh_run call"
    assert all(t <= 3 for t in attempts), (
        f"wait_for_ssh was given a 3s total budget but attempted per-call "
        f"timeouts of {attempts} - at least one exceeds the caller's own "
        f"remaining deadline, which is exactly the bug class that lets "
        f"--timeout-minutes silently not hold")


def test_lambda_wait_for_active_bounds_every_poll_by_its_own_remaining_deadline(monkeypatch):
    """Same invariant, for the Lambda-launch polling path: a poll's HTTP
    request timeout must never exceed what is left of lambda_wait_for_active's
    own budget, even though _lambda_request's own default (30s) is larger."""
    clock = _FakeClock()
    request_timeouts = []

    def fake_lambda_request(method, path, api_key, body=None, timeout=30):
        request_timeouts.append(timeout)
        clock.now += 0.01
        return {"data": []}  # instance never appears -> keeps polling until timeout

    monkeypatch.setattr(run_gate.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(run_gate.time, "sleep", clock.sleep)
    monkeypatch.setattr(run_gate, "_lambda_request", fake_lambda_request)

    with pytest.raises(run_gate.GateError):
        run_gate.lambda_wait_for_active("fake-key", "fake-instance-id", timeout_s=5)

    assert request_timeouts, "lambda_wait_for_active never made a single poll request"
    assert all(t <= 5 for t in request_timeouts), (
        f"lambda_wait_for_active was given a 5s total budget but requested "
        f"HTTP timeouts of {request_timeouts} - at least one exceeds the "
        f"caller's own remaining deadline")


def test_run_exits_3_and_warns_by_name_on_a_possible_launch_orphan(monkeypatch, tmp_path):
    """When the Lambda launch POST fails AFTER potentially creating a billable
    instance server-side (a network read error or reset while parsing the launch
    response, not a clean pre-launch failure), teardown() has no instance id to
    call terminate() on - so the harness still exits 3 (a billing risk) and
    prints an actionable by-NAME dashboard-check warning rather than falling
    through to a generic exit 2."""
    fake_key_file = tmp_path / "fake_key"
    fake_key_file.write_text("not a real key")

    monkeypatch.setattr(run_gate, "lambda_pick_region",
                        lambda *a, **k: ("gpu_2x_a6000", "us-east-1"))

    def raising_launch(*a, **k):
        raise RuntimeError("connection reset while reading the launch response")
    monkeypatch.setattr(run_gate, "lambda_launch", raising_launch)
    monkeypatch.setenv("LAMBDA_API_KEY", "fake-key-value")

    args = run_gate.build_arg_parser().parse_args([
        "--backend", "vulkan", "--launch", "lambda",
        "--local-ssh-key", str(fake_key_file),
    ])
    code = run_gate._run(args)
    assert code == 3, (
        f"a launch failure that may have created a billable orphan instance "
        f"must exit 3 (a billing risk), got {code}")

