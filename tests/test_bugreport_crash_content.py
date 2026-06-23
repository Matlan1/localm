# SPDX-License-Identifier: AGPL-3.0-or-later
"""BUG-1: a recovered-crash report must contain something actionable.

faulthandler only writes a native trace on fault SIGNALS (SIGSEGV etc.); a
window-close or OS-kill leaves the trace file empty, so the report had only a
generic "recovered on the next start" line ("dont seem to actually contain any
useful information"). build_report also never rendered the native_trace it WAS
given (only a 4-key allowlist reached the report). Now the report renders the
native trace when present AND attaches the crashed run's own log tail (matched by
the pid in the log filename), home-path-scrubbed.
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
        # Arrange: a crash marker for pid 4242 + that run's log, no native trace.
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
