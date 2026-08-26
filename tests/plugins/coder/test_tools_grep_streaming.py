# SPDX-License-Identifier: AGPL-3.0-or-later
"""grep: streaming, pre-filter skips, and configurable caps.

grep streams line by line, skips noise directories, binaries and oversized files
before reading them, and takes its caps from call args or config.

These tests hold three properties:

- results are correct for the files it searches: line numbers, context windows,
  hit counts, and the exact "N more" remainders;
- every skip is REPORTED, so a search never covers less than it says;
- the caps are configurable, and 0 means no cap.
"""

import tracemalloc

import pytest

from localm.plugins.coder.tools import tool_grep


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "\n".join([
            "import os",           # 1
            "",                    # 2
            "def target(a):",      # 3
            "    return a",        # 4
            "",                    # 5
            "def other(b):",       # 6
            "    return target(b)",  # 7
        ]) + "\n", encoding="utf-8")
    return tmp_path


class TestResultsUnchanged:
    def test_finds_matches_with_line_numbers_and_marker(self, project):
        r = tool_grep(project, "def target")
        assert r.ok is True
        assert "1 match(es)" in r.summary
        assert "→    3: def target(a):" in r.output

    def test_context_window_matches_the_requested_size(self, project):
        r = tool_grep(project, "def other", context=2)
        # lines 4..8 around the hit on line 6 (file ends at 7)
        assert "     4:     return a" in r.output
        assert "     5: " in r.output
        assert "→    6: def other(b):" in r.output
        assert "     7:     return target(b)" in r.output

    def test_context_clamped_at_start_of_file(self, project):
        """A hit on line 1 has no lines before it - the window must not underflow."""
        r = tool_grep(project, "import os", context=3)
        assert "→    1: import os" in r.output
        assert "     0:" not in r.output

    def test_zero_context_shows_only_the_hit(self, project):
        """A context=0 window is ONE line. The rolling-window matcher closes a
        hit only when its trailing context arrives, so a hit needing zero
        trailing lines is closed at creation rather than picking up the next
        line."""
        r = tool_grep(project, "def target", context=0)
        assert "→    3: def target(a):" in r.output
        assert "return a" not in r.output

    def test_zero_context_with_several_hits_emits_one_line_each(self, project):
        (project / "many.txt").write_text("a\nhit\nb\nhit\nc\n", encoding="utf-8")
        r = tool_grep(project, "hit", path="many.txt", context=0)
        body = [ln for ln in r.output.splitlines() if ln.strip() and not ln.startswith("##")]
        assert body == ["→    2: hit", "→    4: hit"]

    def test_case_insensitive_and_regex_preserved(self, project):
        assert "2 match(es)" in tool_grep(project, r"DEF \w+\(").summary

    def test_no_match_message(self, project):
        r = tool_grep(project, "zzz_nothing_zzz")
        assert r.ok is True
        assert "No matches" in r.output
        assert "0 matches" == r.summary

    def test_invalid_regex_reports_the_error(self, project):
        r = tool_grep(project, "def target(")
        assert r.ok is False
        assert "Invalid regex" in r.output

    def test_single_file_target(self, project):
        r = tool_grep(project, "def target", path="src/main.py")
        assert "1 match(es)" in r.summary

    def test_multiple_hits_in_one_file_each_get_a_window(self, project):
        r = tool_grep(project, "target", context=0)
        assert "→    3:" in r.output and "→    7:" in r.output


class TestSkipsAreReported:
    def test_binary_file_is_skipped_and_reported(self, project):
        (project / "blob.dat").write_bytes(b"target\x00\x01binary target\n")
        r = tool_grep(project, "target")
        assert "blob.dat" not in r.output
        assert "1 binary" in r.output

    def test_extensionless_text_file_is_still_searched(self, project):
        """The binary check is a NUL sniff, not an extension allowlist, so
        Makefile and LICENSE are still searched."""
        (project / "Makefile").write_text("target: build\n", encoding="utf-8")
        r = tool_grep(project, "target: build")
        assert "Makefile" in r.output

    def test_oversized_file_is_skipped_and_reported_with_the_knob(self, project):
        (project / "big.log").write_text("target\n" * 5000, encoding="utf-8")
        r = tool_grep(project, "target", max_file_bytes=1024)
        assert "big.log" not in r.output
        assert "over the" in r.output and "size cap" in r.output
        assert "max_file_bytes=" in r.output

    def test_raising_the_size_cap_recovers_the_matches(self, project):
        (project / "big.log").write_text("target\n" * 5000, encoding="utf-8")
        r = tool_grep(project, "target", max_file_bytes=0)      # 0 = no cap
        assert "big.log" in r.output

    def test_noise_directories_are_skipped_and_reported(self, project):
        git = project / ".git" / "objects"
        git.mkdir(parents=True)
        (git / "obj").write_text("target\n", encoding="utf-8")
        nm = project / "node_modules" / "dep"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("target\n", encoding="utf-8")
        r = tool_grep(project, "target")
        assert "## .git" not in r.output
        assert "## node_modules" not in r.output
        assert "noise dir(s)" in r.output
        assert ".git" in r.output and "node_modules" in r.output   # named, not "..."

    def test_the_noise_note_counts_directories_not_entries(self, project):
        """The note counts directories, so a deep node_modules reports one
        skipped directory rather than thousands of entries."""
        nm = project / "node_modules" / "dep" / "lib"
        nm.mkdir(parents=True)
        for i in range(5):
            (nm / f"f{i}.js").write_text("target\n", encoding="utf-8")
        r = tool_grep(project, "target")
        # One directory named, not "8 files" (5 files + 3 dir entries).
        assert "noise dir(s) node_modules" in r.output
        assert "8" not in r.output.split("[not searched")[1]

    def test_a_noise_dir_named_in_the_glob_is_searched(self, project):
        """A directory named in the glob is searched even though it is in
        _SKIP_DIRS (which holds build/dist/target/venv)."""
        b = project / "build"
        b.mkdir()
        (b / "gen.py").write_text("needle here\n", encoding="utf-8")
        r = tool_grep(project, "needle", glob="build/**/*.py")
        assert "1 match(es)" in r.summary
        assert "gen.py" in r.output

    def test_the_note_says_how_to_search_a_skipped_dir(self, project):
        (project / "dist").mkdir()
        (project / "dist" / "out.js").write_text("target\n", encoding="utf-8")
        r = tool_grep(project, "target")
        assert "path=" in r.output and "glob=" in r.output

    def test_pointing_path_at_a_noise_dir_still_searches_it(self, project):
        """Pruning is by name BELOW the search root, so an explicit
        grep(path="node_modules") still searches it."""
        nm = project / "node_modules" / "dep"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("needle here\n", encoding="utf-8")
        r = tool_grep(project, "needle", path="node_modules")
        assert "1 match(es)" in r.summary

    def test_no_skip_note_when_nothing_was_skipped(self, project):
        r = tool_grep(project, "def target")
        assert "not searched" not in r.output

    def test_skips_are_reported_even_when_there_are_no_matches(self, project):
        (project / "blob.dat").write_bytes(b"\x00\x01\x02")
        r = tool_grep(project, "zzz_nothing_zzz")
        assert "No matches" in r.output
        assert "1 binary" in r.output

    def test_unreadable_file_is_still_reported(self, project, monkeypatch):
        """A file that cannot be read is named in the report."""
        (project / "src" / "locked.py").write_text("target\n", encoding="utf-8")
        from localm.plugins.coder.tools import files as files_mod
        real = files_mod._grep_file_hits

        def boom(fp, *a, **kw):
            if fp.name == "locked.py":
                raise OSError("permission denied")
            return real(fp, *a, **kw)

        monkeypatch.setattr(files_mod, "_grep_file_hits", boom)
        r = tool_grep(project, "target")
        assert "could not be read and were not searched" in r.output
        assert "locked.py" in r.output


class TestConfigurableCaps:
    def _many_hits(self, project, n=40):
        (project / "many.txt").write_text("hit\n" * n, encoding="utf-8")

    def test_per_file_cap_defaults_to_twenty_with_an_exact_remainder(self, project):
        self._many_hits(project, 40)
        r = tool_grep(project, "hit", context=0)
        assert "[... 20 more match(es) in many.txt not shown]" in r.output
        assert "40 match(es)" in r.summary, "the total must count past the cap"

    def test_per_file_cap_is_overridable_per_call(self, project):
        self._many_hits(project, 40)
        r = tool_grep(project, "hit", context=0, max_per_file=5)
        assert "[... 35 more match(es) in many.txt not shown]" in r.output

    def test_per_file_cap_zero_means_no_cap(self, project):
        self._many_hits(project, 40)
        r = tool_grep(project, "hit", context=0, max_per_file=0)
        assert "more match(es)" not in r.output
        assert r.output.count("→") == 40

    def test_per_file_cap_reads_the_config_setting(self, project, monkeypatch):
        self._many_hits(project, 40)
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config",
                            lambda: {"coder_grep_max_per_file": 7})
        r = tool_grep(project, "hit", context=0)
        assert "[... 33 more match(es) in many.txt not shown]" in r.output

    def test_size_cap_reads_the_config_setting(self, project, monkeypatch):
        (project / "big.log").write_text("hit\n" * 5000, encoding="utf-8")
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config",
                            lambda: {"coder_grep_max_file_bytes": 512})
        r = tool_grep(project, "hit", context=0)
        assert "big.log" not in r.output
        assert "over the" in r.output

    def test_output_cap_reports_files_not_searched(self, project):
        for i in range(6):
            (project / f"f{i}.txt").write_text("hit\n" * 5, encoding="utf-8")
        r = tool_grep(project, "hit", context=0, max_output_lines=4)
        assert "output cap reached" in r.output
        assert "more file(s) were NOT searched" in r.output

    def test_output_cap_is_overridable_upwards(self, project):
        for i in range(6):
            (project / f"f{i}.txt").write_text("hit\n" * 5, encoding="utf-8")
        r = tool_grep(project, "hit", context=0, max_output_lines=0)
        assert "output cap reached" not in r.output
        assert "30 match(es)" in r.summary

    def test_config_keys_are_registered_settings(self):
        """A cap read from config needs a registered default, or it cannot be
        set with `localm config ...`."""
        from localm.config import DEFAULT_CONFIG
        for key in ("coder_grep_max_per_file", "coder_grep_max_output_lines",
                    "coder_grep_max_file_bytes"):
            assert key in DEFAULT_CONFIG

    def test_broken_config_value_falls_back_instead_of_failing(self, project,
                                                               monkeypatch):
        self._many_hits(project, 40)
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config",
                            lambda: {"coder_grep_max_per_file": "not-a-number"})
        r = tool_grep(project, "hit", context=0)
        assert r.ok is True
        assert "[... 20 more match(es) in many.txt not shown]" in r.output

    def test_unreadable_config_falls_back_instead_of_failing(self, project,
                                                             monkeypatch):
        self._many_hits(project, 40)
        import localm.config as cfg

        def boom():
            raise OSError("config unreadable")

        monkeypatch.setattr(cfg, "load_config", boom)
        r = tool_grep(project, "hit", context=0)
        assert r.ok is True
        assert "[... 20 more match(es) in many.txt not shown]" in r.output


class TestStreaming:
    def test_a_large_file_is_not_materialised(self, project):
        """Peak allocation does not track file size: reading a 16 MB file whole
        would cost more than 16 MB as a list of str."""
        big = project / "big.txt"
        line = "x" * 99 + "\n"
        big.write_text(line * 160_000, encoding="utf-8")     # ~16 MB
        size = big.stat().st_size

        tracemalloc.start()
        try:
            r = tool_grep(project, "^zzz_nothing_zzz$", path="big.txt",
                          max_file_bytes=0)
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert r.ok is True
        assert peak < size / 4, (
            f"peak allocation {peak / 1e6:.1f} MB for a {size / 1e6:.1f} MB file - "
            "the file is being read whole, not streamed")

    def test_hit_windows_are_correct_in_a_streamed_file(self, project):
        """Context comes from a rolling window, so the surrounding line numbers
        are asserted exactly."""
        lines = [f"line{i}" for i in range(1, 101)]
        lines[49] = "needle"                                  # line 50
        (project / "long.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        r = tool_grep(project, "needle", path="long.txt", context=2)
        assert "    48: line48" in r.output
        assert "    49: line49" in r.output
        assert "→   50: needle" in r.output
        assert "    51: line51" in r.output
        assert "    52: line52" in r.output

    def test_overlapping_hits_each_keep_their_own_window(self, project):
        (project / "dense.txt").write_text(
            "a\nhit\nb\nhit\nc\n", encoding="utf-8")
        r = tool_grep(project, "hit", path="dense.txt", context=1)
        assert r.output.count("→") == 2
        assert "2 match(es)" in r.summary

    def test_trailing_context_at_eof_does_not_overrun(self, project):
        (project / "tail.txt").write_text("a\nb\nhit\n", encoding="utf-8")
        r = tool_grep(project, "hit", path="tail.txt", context=3)
        assert "→    3: hit" in r.output
        assert "     4:" not in r.output

    def test_file_without_trailing_newline_still_matches_last_line(self, project):
        (project / "nonl.txt").write_text("a\nhit", encoding="utf-8")
        r = tool_grep(project, "hit", path="nonl.txt", context=0)
        assert "→    2: hit" in r.output

    def test_crlf_line_endings_are_normalised(self, project):
        (project / "crlf.txt").write_bytes(b"a\r\nhit\r\nb\r\n")
        r = tool_grep(project, "^hit$", path="crlf.txt", context=0)
        assert "→    2: hit" in r.output

    def test_invalid_utf8_does_not_abort_the_search(self, project):
        (project / "mixed.txt").write_bytes("hit\n".encode() + b"\xff\xfe bad\n")
        r = tool_grep(project, "hit", path="mixed.txt", context=0)
        assert "1 match(es)" in r.summary
