# SPDX-License-Identifier: AGPL-3.0-or-later
"""#617 follow-up: the bug-report log tail must survive an arbitrary amount of
routine activity following the actual error, and must not bury a real error
report under repeated "all is well" polling lines.

localm._log_digest replaces the old blind last-N-lines cut with a digest that
keeps every WARNING+/traceback record from the whole run and collapses runs of
near-duplicate benign records (differing only in timestamp/numbers) to one
line + a repeat count.
"""

from __future__ import annotations

from localm import _log_digest as ld


def _polling_line(ts: str, ms: int, lag: float) -> str:
    return f"{ts},000 DEBUG   localm: GET /api/stats -> 200 ({ms} ms, loop_lag={lag}s)"


class TestParseRecords:
    def test_continuation_lines_attach_to_the_prior_record(self):
        text = (
            "2026-07-13 15:24:50,000 ERROR   localm: model load failed\n"
            "Traceback (most recent call last):\n"
            '  File "gguf.py", line 164, in load\n'
            "RuntimeError: Native llama runtime failed to load: [WinError 2]\n"
        )
        records = ld.parse_records(text)
        assert len(records) == 1
        assert records[0]["level"] == "ERROR"
        assert len(records[0]["lines"]) == 4

    def test_each_leveled_line_starts_a_new_record(self):
        text = (
            "2026-07-13 15:24:50,000 DEBUG   localm: a\n"
            "2026-07-13 15:24:51,000 DEBUG   localm: b\n"
        )
        records = ld.parse_records(text)
        assert len(records) == 2

    def test_unrecognized_leading_line_is_its_own_record(self):
        records = ld.parse_records("just some raw text\nmore raw text\n")
        assert len(records) == 1
        assert records[0]["level"] == ""


class TestIsErrorRecord:
    def test_warning_error_critical_are_errors(self):
        for level in ("WARNING", "ERROR", "CRITICAL"):
            rec = {"level": level, "logger": "localm", "lines": ["x"]}
            assert ld.is_error_record(rec)

    def test_debug_and_info_are_not_errors(self):
        for level in ("DEBUG", "INFO"):
            rec = {"level": level, "logger": "localm", "lines": ["x"]}
            assert not ld.is_error_record(rec)

    def test_raw_traceback_with_no_level_counts_as_an_error(self):
        rec = {"level": "", "logger": "", "lines": ["Traceback (most recent call last):"]}
        assert ld.is_error_record(rec)


class TestCollapseNearDuplicates:
    def test_short_runs_are_not_collapsed(self):
        text = "\n".join(
            _polling_line(f"2026-07-13 15:25:{20+i:02d}", 7, 0.2 + i / 100)
            for i in range(2)
        )
        digest = ld.build_digest(text)
        assert digest.count("GET /api/stats") == 2
        assert "repeated" not in digest

    def test_long_runs_of_near_duplicates_collapse(self):
        # The exact shape from issue #617: same request/status, different
        # timestamps and latency numbers every couple of seconds.
        text = "\n".join(
            _polling_line(f"2026-07-13 15:25:{20+i:02d}", 6 + i % 2, 0.2 + i / 100)
            for i in range(6)
        )
        digest = ld.build_digest(text)
        assert digest.count("GET /api/stats") == 1
        assert "repeated 6x" in digest

    def test_an_error_breaks_a_run_and_is_never_collapsed(self):
        lines = [_polling_line(f"2026-07-13 15:25:{20+i:02d}", 7, 0.2) for i in range(3)]
        lines.insert(2, "2026-07-13 15:25:22,500 ERROR   localm: something broke")
        digest = ld.build_digest("\n".join(lines))
        assert "something broke" in digest
        # The polling run was split by the error, so each side (2 + 1) is
        # below the collapse threshold and stays expanded, not silently lost.
        assert digest.count("GET /api/stats") == 3


class TestAllErrorsSurviveArbitraryTrailingNoise:
    def test_error_survives_a_huge_amount_of_later_polling(self):
        # Reproduces the #617 near-miss: the error would be pushed out of a
        # fixed-size tail by enough subsequent routine activity. Simulate far
        # more trailing noise than the old 120-line window ever allowed.
        error_block = (
            "2026-07-13 15:24:50,000 ERROR   localm: model load failed\n"
            "Traceback (most recent call last):\n"
            "RuntimeError: Native llama runtime failed to load: [WinError 2] "
            "The system cannot find the file specified.\n"
        )
        noise = "\n".join(
            _polling_line(f"2026-07-13 15:{30+i//60:02d}:{i%60:02d}", 7, 0.2)
            for i in range(2000)   # ~16x the old 120-line window
        )
        digest = ld.build_digest(error_block + noise)
        assert "RuntimeError: Native llama runtime failed to load" in digest
        assert "[WinError 2]" in digest

    def test_two_separate_errors_both_survive(self):
        text = (
            "2026-07-13 15:00:00,000 ERROR   localm: first failure\n"
            + "\n".join(_polling_line(f"2026-07-13 15:0{i}:00", 7, 0.2) for i in range(1, 6))
            + "\n2026-07-13 15:10:00,000 ERROR   localm: second failure\n"
            + "\n".join(_polling_line(f"2026-07-13 15:1{i}:00", 7, 0.2) for i in range(1, 6))
        )
        digest = ld.build_digest(text)
        assert "first failure" in digest
        assert "second failure" in digest


class TestBudgetFitting:
    def test_errors_are_kept_over_benign_context_when_over_budget(self):
        error_block = "2026-07-13 15:24:50,000 ERROR   localm: the actual failure\n"
        noise = "\n".join(
            _polling_line(f"2026-07-13 15:{30+i//60:02d}:{i%60:02d}", 7, 0.2)
            for i in range(50)
        )
        digest = ld.build_digest(error_block + noise, max_chars=200)
        assert "the actual failure" in digest

    def test_never_raises_on_garbage_input(self):
        assert ld.build_digest("") == ""
        assert ld.build_digest("\x00\x01 not a log at all \xff") != None  # noqa: E711


class TestFullPipelineMatchesIssue617Shape:
    def test_realistic_617_shaped_log(self):
        text = (
            "2026-07-13 15:24:50,123 ERROR   localm: model load failed\n"
            "Traceback (most recent call last):\n"
            '  File "engine.py", line 191, in load\n'
            "    self._backend.load()\n"
            '  File "gguf.py", line 164, in load\n'
            "    raise RuntimeError(\n"
            "RuntimeError: Native llama runtime failed to load: [WinError 2] "
            "The system cannot find the file specified.\n"
            "Provision or repair it with  localm setup-llama  (or set "
            "LLAMA_CPP_LIB to a working llama.dll).\n"
            + "\n".join(
                _polling_line(f"2026-07-13 15:25:{(22 + i * 2) % 60:02d}", 6 + i % 3, 0.2 + i / 50)
                for i in range(40)
            )
        )
        digest = ld.build_digest(text)
        assert "RuntimeError: Native llama runtime failed to load" in digest
        assert "repeated" in digest    # the polling spam collapsed
        # The collapsed digest is meaningfully smaller than the raw noise block.
        assert len(digest) < len(text)
