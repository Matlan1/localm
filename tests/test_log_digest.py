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

    def test_unleveled_continuation_with_native_crash_signal_counts_as_an_error(self):
        # Raw native (ggml/CUDA/HIP) stderr is appended with no leveled prefix
        # of its own (debuglog.py's dedup_native_stderr()/_write_debug()), so
        # it always lands as a CONTINUATION of whatever benign record precedes
        # it - here a routine DEBUG poll. Neither line contains the literal
        # Python traceback marker.
        rec = {"level": "DEBUG", "logger": "localm", "lines": [
            "2026-07-13 15:25:39,000 DEBUG   localm: GET /api/stats -> 200 (7 ms)",
            "CUDA error: operation not permitted when stream is capturing",
        ]}
        assert ld.is_error_record(rec)

    def test_benign_continuation_with_no_crash_signal_is_not_an_error(self):
        # A wrapped, genuinely benign continuation line must not be swept up
        # by the broadened signal scan just because it has a second line.
        rec = {"level": "DEBUG", "logger": "localm", "lines": [
            "2026-07-13 15:25:39,000 DEBUG   localm: multi-line startup banner",
            "    everything initialized without incident",
        ]}
        assert not ld.is_error_record(rec)

    def test_signal_in_the_leveled_header_line_itself_does_not_trigger(self):
        # The header line's own message came through a real "TIMESTAMP LEVEL
        # NAME:" prefix, so its content is already correctly judged by
        # rec["level"] - the broadened scan only applies to lines that never
        # passed that check (continuations, or a level=="" record).
        rec = {"level": "DEBUG", "logger": "localm", "lines": [
            "2026-07-13 15:25:39,000 DEBUG   localm: 0 errors in the last batch",
        ]}
        assert not ld.is_error_record(rec)


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


class TestNativeCrashContinuationSurvives:
    """A native (ggml/CUDA/HIP) crash written via debuglog.py's raw stderr
    append has no "TIMESTAMP LEVEL NAME:" prefix, so parse_records() always
    attaches it as a CONTINUATION of whatever record precedes it - almost
    always a routine DEBUG-level poll given how dense e.g. GET /api/stats
    logging is in a real log. Before this fix, such a record silently
    inherited the benign DEBUG level and was swept into collapse_records'
    near-duplicate collapsing, and even the one surviving instance of a
    collapsed run kept only its first line - so the crash text vanished
    from the digest entirely, with no "omitted" notice, violating the
    module's own "never drops an error record silently" guarantee. See the
    #928 bug report investigation."""

    def test_unformatted_cuda_crash_line_survives_dense_polling_noise(self):
        lines = [_polling_line(f"2026-07-13 15:25:{20+i:02d}", 7, 0.2 + i / 100)
                  for i in range(40)]
        lines.insert(20, "CUDA error: operation not permitted when stream is capturing")
        digest = ld.build_digest("\n".join(lines))
        assert "CUDA error: operation not permitted when stream is capturing" in digest
        # The polling run was split by the crash line, same as an explicit
        # ERROR record splits a run - the two halves (20 + 20) each still
        # collapse on their own.
        assert "repeated 20x" in digest

    def test_unrecognized_unleveled_continuation_still_not_collapsed_away(self):
        # Even a continuation that matches none of the known crash-signal
        # words must never be silently folded into a run of otherwise near-
        # duplicate polling records: record_template() hashes the WHOLE
        # record (continuation lines included), so this one record's uniquely
        # different continuation gives it a template that matches none of its
        # neighbors. With no run to join (run_len == 1, below
        # _MIN_RUN_TO_COLLAPSE), it is emitted expanded, not folded away.
        lines = [_polling_line(f"2026-07-13 15:25:{20+i:02d}", 7, 0.2 + i / 100)
                  for i in range(40)]
        lines.insert(20, "some totally novel native diagnostic line, no known keyword")
        digest = ld.build_digest("\n".join(lines))
        assert "some totally novel native diagnostic line, no known keyword" in digest

    def test_benign_multiline_near_duplicates_still_collapse_well_and_keep_content(self):
        # Real near-miss caught in review: an earlier draft of this fix simply
        # excluded every multi-line (continuation-carrying) record from
        # collapsing, to be safe. Measured against exactly this shape - dense
        # upstream "CUDA Graph id N reused" native spam (see
        # dedup_native_stderr's own docstring) attaching as a continuation of
        # routine polling records, alternating between two ids - that
        # exclusion defeated collapsing almost entirely: 200 such records
        # produced ~6000 chars and 55 uncollapsed repeats instead of one
        # collapsed line, reintroducing exactly the noise this module exists
        # to remove, on the very reports (a CUDA-crashing box) that most need
        # it removed. record_template() hashing the whole record fixes this
        # correctly: these 200 records ARE genuine near-duplicates end to end
        # (numbers masked), so they collapse - and the survivor keeps its own
        # full content, so the repeated native line is not simply lost either.
        lines = []
        for i in range(200):
            base = _polling_line(f"2026-07-13 15:{25 + i // 60:02d}:{i % 60:02d}", 7,
                                  0.2 + i / 1000)
            lines.append(base + f"\nCUDA Graph id {5 + i % 2} reused")
        digest = ld.build_digest("\n".join(lines))
        assert len(digest) < 500, f"benign near-duplicates failed to collapse ({len(digest)} chars)"
        assert "CUDA Graph id" in digest, "the collapsed survivor lost its own content"
        assert "repeated 200x" in digest

    def test_record_template_reflects_continuation_content_not_just_the_header(self):
        # Two records sharing an identical (masked) header but DIFFERENT
        # continuation content must not be treated as near-duplicates of each
        # other - that is what silently discarded a differing continuation
        # under the pre-fix, header-only template.
        header = "2026-07-13 15:25:20,000 DEBUG   localm: GET /api/stream -> 200 (7 ms)"
        rec_a = {"level": "DEBUG", "logger": "localm",
                  "lines": [header, "CUDA Graph id 5 reused"]}
        rec_b = {"level": "DEBUG", "logger": "localm",
                  "lines": [header, "CUDA Graph id 6 reused"]}
        rec_c = {"level": "DEBUG", "logger": "localm",
                  "lines": [header, "something completely unrelated"]}
        # Same masked shape (only the number differs) -> same template.
        assert ld.record_template(rec_a) == ld.record_template(rec_b)
        # Genuinely different continuation content -> different template.
        assert ld.record_template(rec_a) != ld.record_template(rec_c)

    def test_known_tradeoff_unrecognized_differing_numeric_continuations_still_collapse(self):
        # THIS TEST PASSES TODAY, asserting CURRENT, ACCEPTED behavior (see
        # record_template's docstring) - it is not a "should eventually pass
        # once X is added" placeholder. On POSIX, 137 and 139 are
        # 128 + SIGKILL and 128 + SIGSEGV, so the two unleveled continuation
        # lines this test collapses together stand for "the OOM killer took
        # the worker" and "the worker segfaulted" - two different faults, read
        # as one fault twice. That is the accepted trade-off: the alternative
        # (disabling number-masking for continuation lines) was measured too
        # and it breaks the CUDA-Graph-id collapsing this fix exists to keep
        # working (see the sibling test above).
        #
        # A RED here means someone changed the masking behavior, on purpose
        # or not. That is a trade-off decision to make again with full
        # knowledge of what it costs (see record_template's docstring) - NOT
        # a regression to chase back to green by construction alone.
        lines = []
        for i in range(6):
            base = _polling_line(f"2026-07-13 15:25:{20+i:02d}", 7, 0.2 + i / 100)
            code = 137 if i % 2 == 0 else 139
            lines.append(base + f"\nnative worker exit code {code}")
        digest = ld.build_digest("\n".join(lines))
        assert "repeated 6x" in digest
        # Exactly one of the two values survives (the module never drops the
        # kept instance's own real content) - which one is an implementation
        # detail (the last record in the run), not a guarantee to pin.
        assert ("137" in digest) != ("139" in digest), (
            "expected exactly one of the two masked-equal values to survive")

    def test_crash_line_survives_even_as_the_very_first_content_with_no_prior_record(self):
        # If the debug log file itself starts mid-crash (no leveled record
        # came before it at all), parse_records gives it level == "" and it
        # must still be recognized rather than only being checked when it is
        # a continuation of something else.
        text = ("CUDA error: operation not permitted when stream is capturing\n"
                + "\n".join(_polling_line(f"2026-07-13 15:25:{20+i:02d}", 7, 0.2)
                            for i in range(10)))
        digest = ld.build_digest(text)
        assert "CUDA error: operation not permitted when stream is capturing" in digest


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
