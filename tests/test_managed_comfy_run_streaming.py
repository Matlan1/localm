# SPDX-License-Identifier: AGPL-3.0-or-later
"""#953 (ADR-0004): managed_comfy_provision._run() streamed nothing to its
on_progress callback until the WHOLE subprocess exited, because it used
subprocess.run(capture_output=True) - a fully blocking call. This is the ONE
shared helper behind every git/pip/venv step in both provisioning pipelines
(fresh install and copy-from-existing) plus `comfy update`, so a real setup
whose slow steps are a multi-minute `pip install` or a full `git clone` read
as "a couple of lines, then silence for minutes, then a couple more" - the
exact symptom reported (5m39s, two log lines total).

Fix: Popen + a background reader thread that hands each line to on_progress
AS IT ARRIVES, with the timeout still enforced (and the child still killed on
expiry) from the calling thread via proc.wait(timeout=...).

These tests drive the REAL function against a REAL child process (no mocking
of subprocess) - a fake/mocked subprocess could not distinguish "streamed
live" from "batched at exit", which is exactly what pre-fix code would also
pass under a naive mock-based test.
"""

from __future__ import annotations

import sys
import time

import pytest

from localm.media import managed_comfy_provision as prov


def test_run_streams_lines_live_not_batched_at_process_exit(tmp_path):
    """THE assertion: the first line's callback must fire close to when the CHILD
    printed it, not close to when the child eventually exited ~1.5s later."""
    script = tmp_path / "slow_emitter.py"
    script.write_text(
        "import time\n"
        "print('line-one', flush=True)\n"
        "time.sleep(1.5)\n"
        "print('line-two', flush=True)\n",
        encoding="utf-8")

    received: list[tuple[str, float]] = []
    start = time.monotonic()

    def on_progress(line):
        if line in ("line-one", "line-two"):
            received.append((line, time.monotonic() - start))

    ok, out = prov._run([sys.executable, str(script)], on_progress=on_progress,
                        timeout=10)

    assert ok, out
    assert [r[0] for r in received] == ["line-one", "line-two"], received
    t_line_one, t_line_two = received[0][1], received[1][1]
    # line-one must arrive near-instantly after being printed, not batched with
    # line-two at process exit: the discriminator between live streaming and
    # batching.
    assert t_line_one < 0.5, (
        f"line-one's callback fired at {t_line_one:.2f}s, close to the child's "
        "~1.5s exit time rather than near-instantly after it was printed - "
        "progress is still batched at exit, not streamed live (#953)")
    assert t_line_two >= 1.0, received  # sanity: the sleep actually happened
    assert "line-one" in out and "line-two" in out, out


def test_run_preserves_ok_and_combined_output_contract(tmp_path):
    """The (ok, output) return contract every caller depends on is unchanged."""
    script = tmp_path / "emit_and_fail.py"
    script.write_text(
        "import sys\n"
        "print('stdout line')\n"
        "print('stderr line', file=sys.stderr)\n"
        "sys.exit(3)\n",
        encoding="utf-8")

    ok, out = prov._run([sys.executable, str(script)], timeout=10)

    assert ok is False               # non-zero exit -> ok is False
    assert "stdout line" in out
    assert "stderr line" in out      # stderr is still merged into the combined output


def test_run_timeout_kills_the_child_and_returns_clean_false(tmp_path):
    """Preserves the pre-existing contract: a timeout is a clean (False, reason),
    never a raise, and the call returns promptly (the child is actually killed,
    not merely abandoned to run out its full sleep)."""
    script = tmp_path / "hang.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    t0 = time.monotonic()
    ok, out = prov._run([sys.executable, str(script)], timeout=1)
    elapsed = time.monotonic() - t0

    assert ok is False
    assert out == "timed out after 1s", out
    assert elapsed < 10, (
        f"_run() took {elapsed:.1f}s to return after a 1s timeout - the child was "
        "not actually killed, only the wait was abandoned")


def test_run_missing_executable_is_a_clean_false_not_a_raise(tmp_path):
    ok, out = prov._run(["definitely-not-a-real-executable-xyz"], timeout=5)
    assert ok is False
    assert "not found" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
