# SPDX-License-Identifier: AGPL-3.0-or-later
"""The bug-report log tail must survive an arbitrary amount of routine activity
following the actual error, and must not bury a real error report under
repeated "all is well" polling lines.

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
        # Raw native (ggml/CUDA/HIP) stderr is appended with no leveled prefix of
        # its own, so it lands as a CONTINUATION of whatever benign record precedes
        # it - here a routine DEBUG poll. Neither line contains the literal Python
        # traceback marker.
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
        # The header line's own message came through a real "TIMESTAMP LEVEL NAME:"
        # prefix, so its content is judged by rec["level"]; the broadened scan only
        # applies to lines that never passed that check (continuations, or a
        # level=="" record).
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
        # Same request/status, different timestamps and latency numbers every
        # couple of seconds.
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
        # The error would be pushed out of a fixed-size tail by enough subsequent
        # routine activity, so simulate a large amount of trailing noise.
        error_block = (
            "2026-07-13 15:24:50,000 ERROR   localm: model load failed\n"
            "Traceback (most recent call last):\n"
            "RuntimeError: Native llama runtime failed to load: [WinError 2] "
            "The system cannot find the file specified.\n"
        )
        noise = "\n".join(
            _polling_line(f"2026-07-13 15:{30+i//60:02d}:{i%60:02d}", 7, 0.2)
            for i in range(2000)   # well past any fixed tail window
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
    logging is in a real log. Such a record must not inherit the benign DEBUG
    level and be swept into collapse_records' near-duplicate collapsing, which
    keeps only the first line of the one surviving instance and would drop the
    crash text from the digest with no "omitted" notice."""

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
        # A continuation that matches none of the known crash-signal words is not
        # folded into a run of otherwise near-duplicate polling records:
        # record_template() hashes the WHOLE record (continuation lines included),
        # so this record's uniquely different continuation gives it a template that
        # matches none of its neighbors. With run_len == 1 it is emitted expanded.
        lines = [_polling_line(f"2026-07-13 15:25:{20+i:02d}", 7, 0.2 + i / 100)
                  for i in range(40)]
        lines.insert(20, "some totally novel native diagnostic line, no known keyword")
        digest = ld.build_digest("\n".join(lines))
        assert "some totally novel native diagnostic line, no known keyword" in digest

    def test_benign_multiline_near_duplicates_still_collapse_well_and_keep_content(self):
        # Dense upstream "CUDA Graph id N reused" native spam attaching as a
        # continuation of routine polling records, alternating between two ids.
        # record_template() hashes the whole record, so these 200 records are
        # genuine near-duplicates end to end (numbers masked) and collapse, and the
        # survivor keeps its own full content.
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
        # continuation content are not near-duplicates of each other.
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
        # On POSIX, 137 and 139 are 128 + SIGKILL and 128 + SIGSEGV, so the two
        # unleveled continuation lines collapsed together here stand for two
        # different faults read as one fault twice. Number-masking stays on for
        # continuation lines; the sibling test above depends on it.
        lines = []
        for i in range(6):
            base = _polling_line(f"2026-07-13 15:25:{20+i:02d}", 7, 0.2 + i / 100)
            code = 137 if i % 2 == 0 else 139
            lines.append(base + f"\nnative worker exit code {code}")
        digest = ld.build_digest("\n".join(lines))
        assert "repeated 6x" in digest
        # Exactly one of the two values survives; which one (the last record in the
        # run) is not pinned here.
        assert ("137" in digest) != ("139" in digest), (
            "expected exactly one of the two masked-equal values to survive")

    def test_crash_line_survives_even_as_the_very_first_content_with_no_prior_record(self):
        # If the debug log file itself starts mid-crash (no leveled record came
        # before it at all), parse_records gives it level == "" and it is still
        # recognized.
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


class TestContentNeverLeaks:
    """A bug report must never carry chat content. A content-bearing debug
    record (the raw model reply, a memory-embed content snippet, a web-tool
    query) that contains a signal word (error/exception/...) would otherwise be
    PROMOTED to ERROR status by is_error_record and kept verbatim, and
    prioritized over genuine errors when the digest is over budget. These
    records are dropped before that classification runs, whatever they
    contain."""

    def test_raw_model_output_withheld_even_when_it_contains_error_text(self):
        text = (
            "2026-07-13 15:24:50,000 ERROR   localm: model load failed\n"
            "Traceback (most recent call last):\n"
            "RuntimeError: Native llama runtime failed to load\n"
            "2026-07-13 15:24:51,000 DEBUG   localm: raw model output:\n"
            "Sorry, there was an error in your code on line 12: "
            "IndexError: list index out of range\n"
            "2026-07-13 15:24:52,000 INFO    localm: request served\n"
        )
        digest = ld.build_digest(text)
        assert "IndexError: list index out of range" not in digest
        assert "there was an error in your code" not in digest
        # The genuine, unrelated error survives.
        assert "RuntimeError: Native llama runtime failed to load" in digest
        # The redaction is disclosed, not silent. 2, not 1: the lone "request
        # served" line right after the content marker is ALSO withheld, because
        # _drop_content_records cannot yet trust it is not itself still part of the
        # content write.
        assert "2 debug record(s) withheld" in digest

    def test_embedded_header_lookalike_line_inside_the_reply_does_not_escape(self):
        # parse_records has no way to know a multi-line content write's true extent
        # (debuglog.py's writer adds no boundary marker). If the model's OWN reply
        # text contains a line shaped like localm's own log header, parse_records
        # splits the content write into two records there, and the SECOND fragment's
        # header matches none of the markers.
        text = (
            "2026-07-13 15:24:51,000 DEBUG   localm: raw model output:\n"
            "Sure, here is an example log line:\n"
            "2026-07-13 15:24:52,000 INFO    localm: my real secret content is "
            "CreditCard=4111111111111111 and password=hunter2\n"
            "that was the example\n"
        )
        digest = ld.build_digest(text)
        assert "hunter2" not in digest
        assert "CreditCard=4111111111111111" not in digest
        assert "debug record(s) withheld" in digest

    def test_embedded_header_lookalike_survives_only_after_genuine_resync(self):
        # Operational usefulness recovers once genuine traffic (3+ mutually
        # near-duplicate records, the same signal collapse_records already trusts)
        # resumes after a content write, even though the immediate next lines are
        # withheld.
        text = (
            "2026-07-13 15:24:50,000 DEBUG   localm: jobs web tool: web_search "
            "{'query': 'my private medical condition'}\n"
            + "\n".join(_polling_line(f"2026-07-13 15:24:{51+i:02d}", 7, 0.2)
                        for i in range(4))
        )
        digest = ld.build_digest(text)
        assert "my private medical condition" not in digest
        assert "GET /api/stats" in digest
        assert "debug record(s) withheld" in digest

    def test_truncated_tail_starting_mid_content_is_withheld_via_start_tainted(self):
        # bugreport.py's _recent_log_tail truncates a huge log file to its last N
        # bytes before ever calling build_digest, so the surviving text can start
        # mid-way through a content write with NO header at all (parse_records gives
        # it level=="" - the "file starts mid-record" branch). Without
        # start_tainted, that severed fragment is trusted immediately.
        text = ("my actual secret chat reply continues here with password=hunter2\n"
               "2026-07-13 15:24:53,000 INFO    localm: request served\n")
        assert "hunter2" in ld.build_digest(text)                       # untruncated: trusted
        assert "hunter2" not in ld.build_digest(text, start_tainted=True)  # truncated: withheld
        assert "debug record(s) withheld" in ld.build_digest(text, start_tainted=True)

    def test_memory_embed_content_snippet_is_withheld(self):
        text = (
            "2026-07-13 15:24:50,000 DEBUG   localm: memory embed_one failed "
            "for 'the user said their password is hunter2': ValueError: dim mismatch\n"
        )
        digest = ld.build_digest(text)
        assert "hunter2" not in digest
        assert "1 debug record(s) withheld" in digest

    def test_memory_embed_privacy_mode_sibling_message_is_not_withheld(self):
        # The privacy-mode sibling of the same log statement carries NO
        # content (length only) - it must survive; only the content-bearing
        # prefix is a withhold signal.
        text = (
            "2026-07-13 15:24:50,000 DEBUG   localm: memory embed_one failed "
            "(content withheld: privacy mode, 42 chars): ValueError: dim mismatch\n"
        )
        digest = ld.build_digest(text)
        assert "content withheld: privacy mode" in digest
        assert "debug record(s) withheld" not in digest

    def test_web_tool_args_withheld_and_bare_tool_name_alone_is_also_withheld(self):
        # A SINGLE trailing bare-tool-name record right after the marker is NOT
        # enough evidence to resynchronize - it is exactly as forgeable as the
        # marker line itself, so it is withheld too.
        text = (
            "2026-07-13 15:24:50,000 DEBUG   localm: jobs web tool: web_search "
            "{'query': 'my private medical condition'}\n"
            "2026-07-13 15:24:51,000 DEBUG   localm: jobs web tool: fetch_url\n"
        )
        digest = ld.build_digest(text)
        assert "my private medical condition" not in digest
        assert "jobs web tool: fetch_url" not in digest
        assert "2 debug record(s) withheld" in digest

    def test_web_tool_args_marker_distinguishes_content_from_name_only_directly(self):
        # is_content_record tells the two message shapes apart; the structural "{"
        # after the tool name is the signal.
        assert ld.is_content_record(
            {"level": "DEBUG", "logger": "localm", "lines": [
                "2026-07-13 15:24:50,000 DEBUG   localm: jobs web tool: web_search "
                "{'query': 'x'}"]})
        assert not ld.is_content_record(
            {"level": "DEBUG", "logger": "localm", "lines": [
                "2026-07-13 15:24:50,000 DEBUG   localm: jobs web tool: fetch_url"]})

    def test_content_record_dropped_before_it_could_survive_as_a_collapse_survivor(self):
        lines = [_polling_line(f"2026-07-13 15:25:{20+i:02d}", 7, 0.2) for i in range(5)]
        lines.append("2026-07-13 15:25:26,000 DEBUG   localm: raw model output:")
        lines.append("secret reply text")
        digest = ld.build_digest("\n".join(lines))
        assert "secret reply text" not in digest
        assert "repeated 5x" in digest

    def test_is_content_record_direct(self):
        assert ld.is_content_record(
            {"level": "DEBUG", "logger": "localm",
             "lines": ["2026-07-13 15:24:50,000 DEBUG   localm: raw model output:",
                       "hi"]})
        assert not ld.is_content_record(
            {"level": "DEBUG", "logger": "localm",
             "lines": ["2026-07-13 15:24:50,000 DEBUG   localm: GET /api/stats -> 200"]})
        assert not ld.is_content_record({"level": "", "logger": "", "lines": []})

    def test_content_notice_budget_is_reserved_not_squeezed_out_when_over_budget(self):
        # content_notice's reservation (len(notice) + 1) composes with the
        # pre-existing error/benign budget math in _fit_budget rather than being
        # squeezed out by it.
        error_block = "2026-07-13 15:24:50,000 ERROR   localm: the actual failure\n"
        content_block = ("2026-07-13 15:24:51,000 DEBUG   localm: raw model output:\n"
                         "this is the secret reply\n")
        noise = "\n".join(
            _polling_line(f"2026-07-13 15:{30+i//60:02d}:{i%60:02d}", 7, 0.2)
            for i in range(50)
        )
        max_chars = 200
        digest = ld.build_digest(error_block + content_block + noise, max_chars=max_chars)
        assert "the actual failure" in digest
        assert "secret reply" not in digest
        assert ld._content_withheld_notice(1) in digest
        # The reservation keeps the total within max_chars even with the notice
        # included; it is additive with the existing per-error 80-char reservation,
        # not competing with it for the same bytes.
        assert len(digest) <= max_chars


class TestNativeLineRunCollapsesWithinOneRecord:
    """A long run of unleveled native (ggml/CUDA/HIP) stderr has no
    "TIMESTAMP LEVEL NAME:" prefix of its own, so it always glues onto ONE
    record as continuation lines - never a run of multiple RECORDS for
    collapse_records' record-level collapse to fold."""

    def test_giant_run_of_near_duplicate_native_lines_collapses_within_one_record(self):
        # HUNDREDS of consecutive unleveled lines, not one native line per separate
        # leveled record. There is exactly ONE leveled header here, so parse_records
        # glues every one of these 300 lines onto that SAME record as continuation
        # lines.
        header = "2026-07-13 15:25:20,000 DEBUG   localm: GET /api/stream -> 200 (7 ms)"
        spam = [f"ggml_cuda: buffer pool alloc {1000+i} bytes" for i in range(300)]
        text = "\n".join([header] + spam)
        digest = ld.build_digest(text)
        assert digest.count("ggml_cuda: buffer pool alloc") == 1
        assert "repeated 300x" in digest
        assert len(digest) < len(text) / 10

    def test_native_run_with_no_leading_leveled_header_still_collapses(self):
        # The tail of a real captured log trimmed to the failure region: no leveled
        # record at all, just hundreds of raw native lines in a row (parse_records'
        # "file starts mid-record" branch glues them all into ONE level=="" record).
        spam = [f"ggml_cuda: buffer pool alloc {1000+i} bytes" for i in range(250)]
        digest = ld.build_digest("\n".join(spam))
        assert digest.count("ggml_cuda: buffer pool alloc") == 1
        assert "repeated 250x" in digest

    def test_realistic_captured_log_shape_hundreds_of_mixed_native_lines(self):
        # A handful of genuine leveled records, then HUNDREDS of consecutive raw
        # native lines alternating between two real ggml/CUDA message shapes with no
        # timestamp of their own. Both native message shapes must still be found
        # (the survivor keeps its own real content) and the digest must shrink by an
        # order of magnitude.
        lines = [
            "2026-07-13 15:24:10,000 INFO    localm: model load: gemma-3 on vulkan",
            "2026-07-13 15:24:11,000 DEBUG   localm: GET /api/stats -> 200 (7 ms, loop_lag=0.20s)",
        ]
        for i in range(400):
            lines.append(f"CUDA Graph id {5 + i % 2} reused")
        text = "\n".join(lines)
        digest = ld.build_digest(text)
        assert "model load: gemma-3 on vulkan" in digest
        assert "CUDA Graph id" in digest
        assert "repeated" in digest
        assert len(digest) < len(text) / 20

    def test_short_native_run_within_a_record_stays_expanded(self):
        header = "2026-07-13 15:25:20,000 DEBUG   localm: GET /api/stream -> 200 (7 ms)"
        spam = ["ggml_cuda: buffer pool alloc 1 bytes", "ggml_cuda: buffer pool alloc 2 bytes"]
        text = "\n".join([header] + spam)
        digest = ld.build_digest(text)
        assert digest.count("ggml_cuda: buffer pool alloc") == 2
        assert "repeated" not in digest

    def test_error_record_repeated_lines_now_also_collapse(self):
        """_collapse_line_runs applies to an ERROR-classified record too, so a
        long run of near-duplicate native (ggml/CUDA/HIP) stderr glued onto a
        WARNING/ERROR header collapses like the benign path.

        "Errors are kept verbatim" still holds: the header and one real
        instance of every distinct line survive, and only a genuinely-repeating
        run (same text once numbers are masked) folds to a repeat count - see
        test_error_record_with_distinct_lines_stays_fully_verbatim below for the
        case this must NOT touch."""
        header = "2026-07-13 15:25:20,000 WARNING localm: GPU probe degraded"
        # Differs only by a masked number, exactly like the benign fixtures
        # above - a genuine near-duplicate run, not distinct diagnostic content.
        spam = [f"CUDA error: op {i} not permitted while stream is capturing"
                for i in range(10)]
        text = "\n".join([header] + spam)
        digest = ld.build_digest(text)
        assert header in digest
        assert digest.count("CUDA error: op") == 1
        assert "repeated 10x" in digest

    def test_error_record_with_distinct_lines_stays_fully_verbatim(self):
        """The counter-case: a run this short of the 3-line minimum, or lines
        that genuinely differ (not just by a masked number), must NOT be
        folded - collapsing must never discard distinct diagnostic content
        from an error record, only compress genuine repetition."""
        header = "2026-07-13 15:24:50,123 ERROR   localm: model load failed"
        frames = [
            'File "engine.py", line 191, in load',
            "    self._backend.load()",
            'File "gguf.py", line 164, in load',
            "RuntimeError: out of memory",
        ]
        text = "\n".join([header] + frames)
        digest = ld.build_digest(text)
        for line in [header] + frames:
            assert line in digest
        assert "repeated" not in digest


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
