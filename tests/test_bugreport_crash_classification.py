# SPDX-License-Identifier: AGPL-3.0-or-later
"""The crash report classifies HOW the previous run died from the evidence
actually collected, names the operation that was in flight, and points at the
evidence, rather than emitting a fixed "a native crash, an OS kill, or a
force-closed window" guess.

One signature it recognises: the previous run's log stops DEAD mid-word (e.g.
at "llama_co") inside "llama_context: constructing llama_context", with
faulthandler having captured no trace - a hard native crash during model load.
"""

import json

from localm import bugreport, instances


class TestClassifyPriorDeath:
    """Pure-function tests: no I/O, just the classification logic."""

    def test_native_trace_wins_and_names_the_fault(self):
        summary, reason = bugreport._classify_prior_death(
            native_trace="Windows fatal exception: access violation\n\n"
                         "Current thread 0x1 (most recent call first):\n"
                         "  File \"llama.py\", line 942 in _generate",
            hang_trace="",
            raw_tail_truncated=True,
            raw_tail_last_line="llama_co",
        )
        assert "native fault captured" in summary
        assert "access violation" in summary
        assert "captured trace below" in reason

    def test_truncated_native_op_line_without_a_trace(self):
        summary, reason = bugreport._classify_prior_death(
            native_trace="",
            hang_trace="",
            raw_tail_truncated=True,
            raw_tail_last_line="llama_context: constructing llama_context",
        )
        assert "model load/construction" in summary
        assert "native crash suspected" in summary
        assert "llama_context: constructing llama_context" in reason

    def test_truncated_line_that_is_not_a_native_op_does_not_claim_a_crash(self):
        """Only a truncation INSIDE a recognizable native-operation line counts
        as evidence of a native crash. Anything else (e.g. a cut-off HTTP access
        log line) falls through to the unknown-cause message."""
        summary, reason = bugreport._classify_prior_death(
            native_trace="",
            hang_trace="",
            raw_tail_truncated=True,
            raw_tail_last_line="DEBUG localm: GET /api/stats -> 200 (3 m",
        )
        assert "native crash suspected" not in summary
        assert "OS kill" in reason or "force-closed" in reason

    def test_hang_trace_when_nothing_else_present(self):
        summary, reason = bugreport._classify_prior_death(
            native_trace="", hang_trace="Thread 1: <stack>\n",
            raw_tail_truncated=False, raw_tail_last_line="",
        )
        assert "frozen" in summary.lower() or "unresponsive" in summary.lower()
        assert "hang watchdog" in reason

    def test_nothing_found_falls_back_to_the_honest_unknown(self):
        summary, reason = bugreport._classify_prior_death(
            native_trace="", hang_trace="",
            raw_tail_truncated=False, raw_tail_last_line="",
        )
        assert "crashed" in summary.lower()
        assert "OS kill" in reason and "force-closed" in reason

    def test_priority_order_native_trace_beats_truncation_beats_hang(self):
        """With all three signals present at once, a captured trace wins over a
        mid-operation truncation, which wins over a hang stack."""
        summary, _ = bugreport._classify_prior_death(
            native_trace="Fatal Python error: Segmentation fault\nthread info",
            hang_trace="some hang stack",
            raw_tail_truncated=True,
            raw_tail_last_line="llama_context: constructing llama_context",
        )
        assert "native fault captured" in summary

        summary2, _ = bugreport._classify_prior_death(
            native_trace="", hang_trace="some hang stack",
            raw_tail_truncated=True,
            raw_tail_last_line="ggml_backend_cuda_graph_compute: warmup",
        )
        assert "native crash suspected" in summary2


class TestRawTailTruncationSignal:
    def _log(self, home, pid, body: str, *, newline_terminated: bool):
        d = home / "logs"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"localm_2026-07-26_170900_{pid}.log"
        text = body if newline_terminated else body.rstrip("\n")
        p.write_bytes(text.encode("utf-8"))
        return p

    def test_no_trailing_newline_is_truncated(self, tmp_path):
        self._log(tmp_path, 111,
                  "llama_context: constructing llama_context\n"
                  "llama_co", newline_terminated=False)
        truncated, last_line = bugreport._raw_tail_truncation_signal(
            home=tmp_path, pid=111)
        assert truncated is True
        assert last_line == "llama_co"

    def test_trailing_newline_is_not_truncated(self, tmp_path):
        self._log(tmp_path, 222,
                  "DEBUG localm: clean shutdown requested\n",
                  newline_terminated=True)
        truncated, _ = bugreport._raw_tail_truncation_signal(home=tmp_path, pid=222)
        assert truncated is False

    def test_no_log_file_is_not_truncated(self, tmp_path):
        truncated, last_line = bugreport._raw_tail_truncation_signal(
            home=tmp_path, pid=999)
        assert truncated is False
        assert last_line == ""


def _write_marker(run_dir, instance_id, pid):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"server-crash.{instance_id}.marker").write_text(
        json.dumps({"pid": pid, "context": {}}), encoding="utf-8")


class TestEndToEndClassificationInReport:
    """check_and_report_prior_crash puts the classification into the filed
    report."""

    def test_mid_native_op_cutoff_reaches_the_filed_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
        home = tmp_path
        run = home / "run"
        _write_marker(run, "inst-crash", 7777)
        logs = home / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "localm_2026-07-26_170900_7777.log").write_bytes(
            b"llama_context: constructing llama_context\nllama_co")

        captured = {}
        monkeypatch.setattr(bugreport, "report_failure",
                            lambda **k: captured.update(k) or str(tmp_path / "r.md"))

        bugreport.check_and_report_prior_crash(home=str(home))

        assert "native crash suspected" in captured["summary"]
        assert "llama_co" in captured["reason"]

    def test_clean_looking_tail_with_no_evidence_stays_honest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
        home = tmp_path
        run = home / "run"
        _write_marker(run, "inst-clean", 8888)
        logs = home / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "localm_2026-07-26_170900_8888.log").write_bytes(
            b"DEBUG localm: GET /api/stats -> 200 (3 ms)\n")

        captured = {}
        monkeypatch.setattr(bugreport, "report_failure",
                            lambda **k: captured.update(k) or str(tmp_path / "r.md"))

        bugreport.check_and_report_prior_crash(home=str(home))

        assert "native crash suspected" not in captured["summary"]
        assert "OS kill" in captured["reason"]
