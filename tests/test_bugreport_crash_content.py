# SPDX-License-Identifier: AGPL-3.0-or-later
"""A recovered-crash report must contain something actionable.

faulthandler only writes a native trace on fault SIGNALS (SIGSEGV etc.); a
window-close or OS-kill leaves the trace file empty. The report renders the
native trace when present AND attaches the crashed run's own log tail (matched
by the pid in the log filename), home-path-scrubbed.
"""

import json
from pathlib import Path

import localm.bugreport as br


def _make_log(home: Path, pid: int, body: str) -> Path:
    d = home / "logs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"localm_2026-06-22_231715_{pid}.log"
    p.write_text(body, encoding="utf-8")
    return p


class TestRecentLogTail:
    def test_matches_the_crashed_pid(self, tmp_path):
        _make_log(tmp_path, 111, "old other-run line\n")
        _make_log(tmp_path, 222, "DEBUG n_ctx overflow then WinError 6\nlast line\n")
        tail = br._recent_log_tail(home=tmp_path, pid=222)
        assert "n_ctx overflow" in tail
        assert "other-run line" not in tail

    def test_no_logs_returns_empty(self, tmp_path):
        assert br._recent_log_tail(home=tmp_path, pid=999) == ""

    def test_scrubs_home_path(self, tmp_path):
        home_str = str(Path.home())
        _make_log(tmp_path, 333, f"loaded model from {home_str}\\models\\m.gguf\n")
        tail = br._recent_log_tail(home=tmp_path, pid=333)
        assert home_str not in tail
        assert "~" in tail or "<redacted>" in tail

    def test_raw_model_output_never_reaches_the_report_even_via_the_real_file(self, tmp_path):
        # Full pipeline: a real on-disk debug log (what llama.py's
        # logger.debug writes) must never surface in the tail returned to a
        # report, even though the reply text below contains the word error.
        _make_log(tmp_path, 444, (
            "2026-07-13 15:24:50,000 ERROR   localm: model load failed\n"
            "Traceback (most recent call last):\n"
            "RuntimeError: Native llama runtime failed to load\n"
            "2026-07-13 15:24:51,000 DEBUG   localm: raw model output:\n"
            "Sorry, there was an error in your code on line 12: "
            "IndexError: list index out of range\n"
        ))
        tail = br._recent_log_tail(home=tmp_path, pid=444)
        assert "IndexError: list index out of range" not in tail
        assert "there was an error in your code" not in tail
        assert "RuntimeError: Native llama runtime failed to load" in tail
        assert "debug record(s) withheld" in tail

    def test_truncated_tail_starting_mid_content_never_leaks(self, tmp_path, monkeypatch):
        # Shrink _recent_log_tail's own 2MB tail-truncation cap so a small test
        # file triggers it for real, with the content engineered so the
        # truncation point lands INSIDE a content record and no header survives
        # into the slice.
        monkeypatch.setattr(br, "_LOG_TAIL_READ_BYTES", 200)
        header = "2026-07-13 15:24:50,000 DEBUG   localm: raw model output:\n"
        secret = "the secret chat reply keeps going and going with password=hunter2\n" * 5
        body = header + secret
        assert len(body) > 200   # confirm this really does trigger truncation
        _make_log(tmp_path, 555, body)
        tail = br._recent_log_tail(home=tmp_path, pid=555)
        assert "hunter2" not in tail


class TestLogUnavailableIsNotSilence:
    """An empty digest can mean THREE unrelated things: no log file matched
    this run, the file was found but could not be READ, or the run genuinely
    logged nothing notable. The first two are failures to collect and must not
    render as silence."""

    def test_unreadable_log_is_distinguished_from_a_missing_one(self, tmp_path):
        # A REAL OSError out of a REAL filesystem state (a directory wearing the
        # log's name): read_text on a directory raises PermissionError on
        # Windows and IsADirectoryError on POSIX, and both are OSError.
        d = tmp_path / "logs"
        d.mkdir()
        (d / "localm_2026-06-22_231715_777.log").mkdir()
        tail, reason = br._recent_log_tail_result(home=tmp_path, pid=777)
        assert tail == ""
        assert "could not be read" in reason
        # ...and NOT the same answer as "there was no log at all" (case a).
        assert reason != br._LOG_UNAVAILABLE_NO_FILE

    def test_no_log_file_gets_its_own_message(self, tmp_path):
        tail, reason = br._recent_log_tail_result(home=tmp_path, pid=999)
        assert tail == ""
        assert reason == br._LOG_UNAVAILABLE_NO_FILE
        assert "could not be read" not in reason

    def test_a_log_that_was_read_is_never_reported_as_uncollected(self, tmp_path):
        # Case (c): the collection SUCCEEDED and the log is genuinely empty.
        _make_log(tmp_path, 888, "2026-07-13 15:24:50,000 INFO    localm: started\n")
        _, reason = br._recent_log_tail_result(home=tmp_path, pid=888)
        assert reason == ""

    def test_the_str_of_the_error_would_leak_the_account_name_and_is_not_used(self):
        # The privacy property needs an OSError whose filename is under the REAL
        # home dir; a tmp_path fixture cannot express it, since pytest's basetemp
        # is off the home tree.
        leaky = Path.home() / "logs" / "localm_2026-06-22_231715_777.log"
        exc = PermissionError(13, "Permission denied", str(leaky))
        # Confirm the naive formatting really does leak the path.
        assert Path.home().name in str(exc)
        reason = br._log_failure_reason(exc)
        assert "Permission denied" in reason      # it still says WHY
        assert str(leaky) not in reason
        assert Path.home().name not in reason

    def test_a_non_oserror_contributes_no_message_at_all(self):
        # repr() is safe for OSError only: its __repr__ drops the filename.
        leaky = Path.home() / "logs" / "localm_2026-06-22_231715_777.log"
        exc = ValueError(f"boom {leaky}")
        reason = br._log_failure_reason(exc)
        assert "ValueError" in reason
        assert str(leaky) not in reason
        assert Path.home().name not in reason

    def test_a_path_shaped_strerror_is_dropped_whole(self):
        # A leak is made UNREPRESENTABLE rather than trusting strerror never to
        # carry a path. Asserted against the instance's OWN class name, because
        # OSError auto-subclasses on errno (errno 13 constructs a
        # PermissionError).
        exc = OSError(13, "denied reading sub/dir")
        reason = br._log_failure_reason(exc)
        assert "sub/dir" not in reason
        assert "/" not in reason
        assert reason == type(exc).__name__   # nothing but the class name left

    def test_the_report_says_the_log_could_not_be_collected(self, tmp_path):
        # End to end through the REAL failure, into the REAL report text.
        d = tmp_path / "logs"
        d.mkdir()
        (d / "localm_2026-06-22_231715_777.log").mkdir()
        tail, reason = br._recent_log_tail_result(home=tmp_path, pid=777)
        text = br.build_report(
            "localm server crashed (recovered on the next start)",
            context={"recent_log_tail": tail, "log_unavailable": reason},
        )
        assert "## Recent log (tail)" in text
        assert "not collected" in text
        # The log's own absolute path must not reach the report.
        assert str(d) not in text
        assert str(tmp_path / "logs") not in text

    def test_a_real_tail_renders_the_log_and_never_the_notice(self):
        text = br.build_report(
            "crashed",
            context={"recent_log_tail": "DEBUG POST /api/models/load -> 200",
                     "log_unavailable": "must never render alongside a real tail"},
        )
        assert "POST /api/models/load" in text
        assert "not collected" not in text


class TestBuildReportRendersCrashDetail:
    def test_native_trace_and_log_tail_rendered(self):
        text = br.build_report(
            "localm server crashed (recovered on the next start)",
            reason="ended without a clean shutdown",
            error=None,
            context={"native_trace": "Current thread 0x1: SIGSEGV in ggml",
                     "recent_log_tail": "DEBUG POST /api/models/load -> 200"},
        )
        assert "## Native fault trace" in text
        assert "SIGSEGV in ggml" in text
        assert "## Recent log (tail)" in text
        assert "POST /api/models/load" in text

    def test_no_crash_sections_when_context_empty(self):
        text = br.build_report("something", reason="x", error=None, context={})
        assert "## Native fault trace" not in text
        assert "## Recent log (tail)" not in text

    def test_crash_sections_scrub_home(self):
        home_str = str(Path.home())
        text = br.build_report(
            "crash", error=None,
            context={"recent_log_tail": f"opened {home_str}\\run\\x"})
        assert home_str not in text


class TestCheckAndReportAttachesContent:
    def test_recovered_crash_report_includes_log_tail(self, tmp_path, monkeypatch):
        # Arrange: a crash marker for pid 4242 + that run's log, no native
        # trace. pid_alive is mocked so the pid-liveness check treats this
        # marker as a genuine crash.
        from localm import instances
        monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
        run = tmp_path / "run"
        run.mkdir(parents=True, exist_ok=True)
        (run / "server-crash.marker").write_text(
            json.dumps({"pid": 4242, "context": {"port": 5768}}), encoding="utf-8")
        _make_log(tmp_path, 4242, "DEBUG last activity before the hard kill\n")

        captured = {}

        def _fake_report(**kwargs):
            captured.update(kwargs)
            return tmp_path / "bug-x.md"

        monkeypatch.setattr(br, "report_failure", _fake_report)
        br.check_and_report_prior_crash(home=tmp_path)

        ctx = captured.get("context") or {}
        assert "recent_log_tail" in ctx
        assert "last activity before the hard kill" in ctx["recent_log_tail"]
        # And the marker was cleared so it does not re-report on the next start.
        assert not (run / "server-crash.marker").exists()
