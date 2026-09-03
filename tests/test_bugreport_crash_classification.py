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


# The exact trace a HEALTHY standalone app-window start writes, reduced to its
# fault header. Measured 2026-09-03: a run that wrote this went on to load its
# window, serve for over an hour and exit cleanly - faulthandler's Windows
# handler logs an exception and returns EXCEPTION_CONTINUE_SEARCH, so the
# trace records that an exception happened, never that it ended the run.
_SURVIVED_COM_TRACE = (
    "Windows fatal exception: code 0x8001010d\n"
    "\n"
    "Current thread 0x000063b4 (most recent call first):\n"
    '  File "webview\\platforms\\winforms.py", line 808 in create\n'
)


class TestSurvivableFaultsAreNotCrashes:
    """A faulthandler trace holding only first-chance exceptions the process
    HANDLED must not be reported to the user as a crash.

    This shipped as a real false report: a run that was working fine produced
    "localm server crashed - native fault captured: Windows fatal exception:
    code 0x8001010d" plus a filed bug report, because any non-empty trace was
    taken as proof of a native crash. 0x8001010d is
    RPC_E_CANTCALLOUT_ININPUTSYNCCALL - an HRESULT raised and handled inside
    WebView2/.NET while the app window is created.
    """

    def test_a_survived_com_exception_is_not_reported_as_a_native_crash(self):
        summary, reason = bugreport._classify_prior_death(
            native_trace=_SURVIVED_COM_TRACE,
            hang_trace="",
            raw_tail_truncated=False,
            raw_tail_last_line="",
        )
        assert "native fault captured" not in summary
        assert "0x8001010d" not in summary
        # Falls through to the honest unknown rather than inventing a cause.
        assert "OS kill" in reason and "force-closed" in reason

    def test_a_survived_com_exception_does_not_mask_the_real_evidence(self):
        """The benign trace must not outrank the mid-operation cutoff that
        actually explains the death - the ordering bug the false positive hid."""
        summary, reason = bugreport._classify_prior_death(
            native_trace=_SURVIVED_COM_TRACE,
            hang_trace="",
            raw_tail_truncated=True,
            raw_tail_last_line="llama_context: constructing llama_context",
        )
        assert "model load/construction" in summary
        assert "llama_context: constructing llama_context" in reason

    def test_a_real_access_violation_is_still_reported(self):
        """Fires-control for the fix: the genuine fatal fault must survive it."""
        summary, _ = bugreport._classify_prior_death(
            native_trace="Windows fatal exception: access violation\n\n"
                         "Current thread 0x1 (most recent call first):\n"
                         '  File "llama.py", line 942 in _generate',
            hang_trace="", raw_tail_truncated=False, raw_tail_last_line="",
        )
        assert "native fault captured" in summary
        assert "access violation" in summary

    def test_a_fatal_fault_after_benign_noise_is_the_one_named(self):
        """A real crash preceded by survivable noise must be titled by the
        REAL fault, not by whichever line happened to come first."""
        summary, _ = bugreport._classify_prior_death(
            native_trace=_SURVIVED_COM_TRACE +
                         "\nWindows fatal exception: access violation\n",
            hang_trace="", raw_tail_truncated=False, raw_tail_last_line="",
        )
        assert "access violation" in summary
        assert "0x8001010d" not in summary

    def test_ntstatus_range_numeric_codes_are_still_fatal(self):
        """Only the software-exception ranges are excused. A bare NTSTATUS
        code faulthandler had no name for is a genuine fault."""
        summary, _ = bugreport._classify_prior_death(
            native_trace="Windows fatal exception: code 0xc0000005\n",
            hang_trace="", raw_tail_truncated=False, raw_tail_last_line="",
        )
        assert "native fault captured" in summary

    def test_posix_fatal_python_error_is_fatal(self):
        summary, _ = bugreport._classify_prior_death(
            native_trace="Fatal Python error: Segmentation fault\n\n"
                         "Current thread 0x1 (most recent call first):\n",
            hang_trace="", raw_tail_truncated=False, raw_tail_last_line="",
        )
        assert "native fault captured" in summary
        assert "Segmentation fault" in summary

    def test_clr_and_cpp_throw_codes_are_survivable(self):
        """The CLR's own exception code and the C++ throw code are raised by
        software and normally handled, exactly like the COM HRESULT range."""
        for code in ("0xe0434352", "0xe06d7363"):
            summary, _ = bugreport._classify_prior_death(
                native_trace=f"Windows fatal exception: code {code}\n",
                hang_trace="", raw_tail_truncated=False, raw_tail_last_line="",
            )
            assert "native fault captured" not in summary, code

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
