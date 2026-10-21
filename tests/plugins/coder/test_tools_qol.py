# SPDX-License-Identifier: AGPL-3.0-or-later
"""
read_file offset/limit slicing, explicit truncation markers (grep, tree,
read_file), edit_file closest-match hints, read_env redaction and
edit_notebook_cell.
"""

import json
import os
from pathlib import Path

import pytest

from localm.plugins.coder.tools import (
    tool_edit_file,
    tool_edit_notebook_cell,
    tool_grep,
    tool_read_env,
    tool_read_file,
    tool_tree,
)


# ---------------------------------------------------------------------------
#  read_file offset / limit
# ---------------------------------------------------------------------------

class TestReadFileSlicing:
    @pytest.fixture
    def big(self, tmp_path):
        lines = "\n".join(f"line {i}" for i in range(1, 101)) + "\n"
        (tmp_path / "big.txt").write_text(lines, encoding="utf-8")
        return tmp_path

    def test_offset_and_limit_slice(self, big):
        r = tool_read_file(big, "big.txt", offset=40, limit=3)
        assert r.ok
        assert "line 40" in r.output and "line 42" in r.output
        assert "line 39" not in r.output and "line 43" not in r.output
        assert "<lines>40-42 of 100</lines>" in r.output

    def test_offset_without_limit_reads_to_end(self, big):
        r = tool_read_file(big, "big.txt", offset=99)
        assert r.ok and "line 99" in r.output and "line 100" in r.output

    def test_limit_without_offset_starts_at_top(self, big):
        r = tool_read_file(big, "big.txt", limit=2)
        assert r.ok and "line 1" in r.output and "line 3" not in r.output

    def test_offset_past_end_is_clear_error(self, big):
        r = tool_read_file(big, "big.txt", offset=9999)
        assert not r.ok and "past the end" in r.output

    def test_truncated_full_read_hints_at_slicing(self, tmp_path):
        (tmp_path / "huge.txt").write_text("x" * 20_000, encoding="utf-8")
        r = tool_read_file(tmp_path, "huge.txt")
        assert r.ok and r.truncated
        assert "re-read specific parts" in r.output

    def test_plain_read_unchanged(self, tmp_path):
        (tmp_path / "s.txt").write_text("hello\n", encoding="utf-8")
        r = tool_read_file(tmp_path, "s.txt")
        assert r.ok and "hello" in r.output and not r.truncated


class TestReadFileTruncationNamesTheMissingLines:
    """A read cut to its head and tail says exactly which lines did not
    arrive, so the next read can go straight to them instead of the model
    filling the gap in from memory."""

    @pytest.fixture
    def forty(self, tmp_path):
        # 1000 lines of exactly 40 characters: the 4000-character head and
        # tail _truncate keeps end and start exactly on line boundaries.
        text = "".join(f"line {i:04d}".ljust(39, ".") + "\n" for i in range(1, 1001))
        (tmp_path / "f.txt").write_text(text, encoding="utf-8")
        return tmp_path

    def test_full_read_names_the_missing_range(self, forty):
        r = tool_read_file(forty, "f.txt")
        assert r.truncated
        assert "line 0100" in r.output and "line 0901" in r.output
        assert "line 0101" not in r.output.split("[file truncated")[0]
        assert "lines 101-900 of 1000 are missing" in r.output
        assert "offset=101" in r.output
        assert "re-read specific parts" in r.output

    def test_a_cut_inside_a_line_counts_that_line_as_missing(self, tmp_path):
        text = "".join(f"line {i:04d}".ljust(47, ".") + "\n" for i in range(1, 1001))
        (tmp_path / "g.txt").write_text(text, encoding="utf-8")
        r = tool_read_file(tmp_path, "g.txt")
        # 48-character lines: the 4000-character head stops 16 characters into
        # line 84, and the tail (from character 44000) starts 32 characters
        # into line 917, so both of those lines are cut short.
        assert "lines 84-917 of 1000 are missing" in r.output

    def test_a_truncated_slice_names_file_line_numbers(self, forty):
        r = tool_read_file(forty, "f.txt", offset=200, limit=400)
        assert r.truncated
        assert "lines 300-499 of 1000 are missing" in r.output
        assert "offset=300" in r.output


# ---------------------------------------------------------------------------
#  grep truncation markers
# ---------------------------------------------------------------------------

class TestGrepMarkers:
    def test_per_file_cap_marker(self, tmp_path):
        (tmp_path / "many.txt").write_text(
            "\n".join("needle here" for _ in range(30)), encoding="utf-8")
        r = tool_grep(tmp_path, "needle", glob="many.txt", context=0)
        assert r.ok
        assert "more match(es) in" in r.output

    def test_file_cap_names_unsearched_count(self, tmp_path):
        # Enough matches in early files to blow the 300-line output cap,
        # with files left over afterwards.
        for i in range(8):
            (tmp_path / f"f{i}.txt").write_text(
                "\n".join("needle" for _ in range(25)), encoding="utf-8")
        r = tool_grep(tmp_path, "needle", glob="*.txt", context=2)
        assert r.ok
        assert "were NOT" in r.output and "narrow the search" in r.output


# ---------------------------------------------------------------------------
#  tree limit message
# ---------------------------------------------------------------------------

class TestTreeLimit:
    def test_limit_message_is_actionable(self, tmp_path):
        for i in range(12):
            (tmp_path / f"file{i:02d}.txt").write_text("x", encoding="utf-8")
        r = tool_tree(tmp_path, max_files=5)
        assert r.ok
        assert "file limit 5 reached" in r.output
        assert "raise max_files" in r.output


# ---------------------------------------------------------------------------
#  edit_file closest-match hint
# ---------------------------------------------------------------------------

class TestEditFileClosestMatch:
    def test_near_miss_shows_the_real_text(self, tmp_path):
        (tmp_path / "a.py").write_text(
            "def greet(name):\n    return f'hi {name}'\n", encoding="utf-8")
        # Indentation does not match the file: a near miss.
        r = tool_edit_file(tmp_path, "a.py",
                           old="def greet( name ):", new="def hello(name):")
        assert not r.ok
        assert "Closest match" in r.output
        assert "def greet(name):" in r.output     # the actual file text
        assert "whitespace" in r.output

    def test_totally_unrelated_text_has_no_snippet(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        r = tool_edit_file(tmp_path, "a.txt",
                           old="zzzzqqqq12345", new="x")
        assert not r.ok
        assert "Closest match" not in r.output


# ---------------------------------------------------------------------------
#  read_env redaction
# ---------------------------------------------------------------------------

class TestReadEnvRedaction:
    # (name, value, secret substring that must not appear). Every name here
    # is free of the secret-name markers, so only value inspection or the
    # extra name rules can catch it.
    _UNMARKED = [
        ("DATABASE_URL", "postgres://admin:S3cretPw@db.example.com:5432/app", "S3cretPw"),
        ("REDIS_URL", "redis://:hunter2@cache:6379/0", "hunter2"),
        ("PROVIDER", "sk-proj-" + "A" * 40, "A" * 40),
        ("GITHUB_PAT", "ghp_" + "B" * 36, "B" * 36),
        ("GH", "ghp_" + "C" * 36, "C" * 36),
        ("STRIPE_SK", "sk_live_" + "D" * 24, "D" * 24),
        ("BILLING", "sk_live_" + "E" * 24, "E" * 24),
        ("JWT", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlMTIz", "c2lnbmF0dXJlMTIz"),
        ("MASTER_PASSPHRASE", "correct horse battery staple", "horse battery"),
        ("WALLET_MNEMONIC", "abandon abandon ability able", "abandon ability"),
        ("WALLET_SEED", "zoo zoo zoo wrong", "zoo zoo"),
        ("RECOVERY_PHRASE", "legal winner thank year", "winner thank"),
        ("DISCORD_WEBHOOK", "https://discord.com/api/webhooks/12345/AbCdEfGh", "AbCdEfGh"),
        ("NOTIFY", "https://discord.com/api/webhooks/12345/ZyXwVuTs", "ZyXwVuTs"),
        ("SEARX", "https://searx.local/search?api_key=QQQsecretQQQ", "QQQsecretQQQ"),
        ("AWS_ID", "AKIA" + "F" * 16, "F" * 16),
    ]

    def test_unmarked_names_holding_secrets_are_redacted(self, tmp_path):
        body = "".join(f"{n}={v}\n" for n, v, _ in self._UNMARKED)
        (tmp_path / ".env").write_text(
            body + "API_KEY=sk-supersecret123\nDB_PASSWORD=hunter3\n"
                   "PLAIN_SETTING=visible-value\nPORT=8080\n",
            encoding="utf-8")
        r = tool_read_env(tmp_path, path=".env")
        assert r.ok
        for name, _value, secret in self._UNMARKED:
            assert secret not in r.output, name
            assert f"{name}=" in r.output, name
        assert "sk-supersecret123" not in r.output
        assert "hunter3" not in r.output
        assert "PLAIN_SETTING=visible-value" in r.output
        assert "PORT=8080" in r.output
        assert "db.example.com" in r.output     # only the credential goes
        assert r.summary == f"read_env ({len(self._UNMARKED) + 4} vars, " \
                            f"{len(self._UNMARKED) + 2} values redacted)"

    def test_process_environment_is_scrubbed(self, tmp_path, monkeypatch):
        home = str(Path.home())
        monkeypatch.setenv("ZZ_CONN", "mysql://root:Pw0rdZZ@10.0.0.5/db")
        monkeypatch.setenv("ZZ_WORKDIR", home + os.sep + "proj")
        monkeypatch.setenv("USERNAME", "zzaccountname")
        monkeypatch.setenv("COMPUTERNAME", "zzmachinename")
        monkeypatch.setenv("HOMEPATH", r"\Users\zzaccountname")
        monkeypatch.setenv("SUDO_USER", "zzaccountname")
        monkeypatch.setenv("ZZ_SHARE", r"\\fileserver\zzaccountname\docs")
        monkeypatch.setenv("ZZ_MAILBOX", "/var/mail/zzaccountname")
        monkeypatch.setenv("ZZ_NOTE", "zzaccountnamesuffix-not-a-segment")
        r = tool_read_env(tmp_path)
        assert r.ok
        assert "Pw0rdZZ" not in r.output
        assert "zzaccountname" not in r.output.replace("zzaccountnamesuffix", "")
        assert r"ZZ_SHARE=\\<host>\<user>\docs" in r.output
        assert "ZZ_MAILBOX=/var/mail/<user>" in r.output
        assert "ZZ_NOTE=zzaccountnamesuffix-not-a-segment" in r.output
        assert "zzmachinename" not in r.output
        assert "ZZ_WORKDIR=" in r.output
        assert home not in r.output
        assert f"ZZ_WORKDIR=~{os.sep}proj" in r.output

    def test_every_email_address_is_redacted(self, tmp_path, monkeypatch):
        """No address survives, the one the bug-report scrubber keeps
        (MAINTAINER_EMAIL) included, in a .env file and in the process
        environment alike."""
        from localm.bugreport._common import MAINTAINER_EMAIL
        (tmp_path / ".env").write_text(
            f"COMMIT_EMAIL={MAINTAINER_EMAIL}\nCONTACT=someone@example.org\n",
            encoding="utf-8")
        monkeypatch.setenv("ZZ_USER_EMAIL", MAINTAINER_EMAIL.upper())
        monkeypatch.setenv("ZZ_NOTIFY_TO", "Ops <" + MAINTAINER_EMAIL + ">")
        r = tool_read_env(tmp_path)
        assert r.ok
        assert MAINTAINER_EMAIL.split("@")[0].lower() not in r.output.lower()
        assert "someone@example.org" not in r.output
        assert "COMMIT_EMAIL=<redacted-email>" in r.output
        assert "ZZ_USER_EMAIL=<redacted-email>" in r.output
        assert "CONTACT=<redacted-email>" in r.output

    def test_summary_reports_zero_when_nothing_was_redacted(self, tmp_path):
        (tmp_path / "plain.env").write_text("PORT=8080\nMODE=dev\n", encoding="utf-8")
        r = tool_read_env(tmp_path, path="plain.env")
        assert r.ok
        assert r.summary == "read_env (2 vars, 0 values redacted)"
        assert "PORT=8080" in r.output and "MODE=dev" in r.output

    def test_explicit_missing_file_errors(self, tmp_path):
        r = tool_read_env(tmp_path, path="nope.env")
        assert not r.ok and "not found" in r.output


# ---------------------------------------------------------------------------
#  edit_notebook_cell
# ---------------------------------------------------------------------------

def _notebook(cells):
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


class TestEditNotebookCell:
    @pytest.fixture
    def nb_path(self, tmp_path):
        nb = _notebook([
            {"cell_type": "code", "source": ["import os\n"], "outputs": [],
             "metadata": {}, "execution_count": None},
            {"cell_type": "markdown", "source": ["# Title\n"], "metadata": {}},
        ])
        p = tmp_path / "n.ipynb"
        p.write_text(json.dumps(nb), encoding="utf-8")
        return p

    def test_edit_replaces_cell_source(self, nb_path, tmp_path):
        r = tool_edit_notebook_cell(tmp_path, "n.ipynb", 0,
                                    "import sys\nprint(sys.path)")
        assert r.ok
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["cells"][0]["source"] == ["import sys\n", "print(sys.path)"]
        # the other cell is untouched
        assert nb["cells"][1]["source"] == ["# Title\n"]

    def test_cell_type_override(self, nb_path, tmp_path):
        r = tool_edit_notebook_cell(tmp_path, "n.ipynb", 1, "plain text",
                                    cell_type="raw")
        assert r.ok
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["cells"][1]["cell_type"] == "raw"

    def test_out_of_range_index(self, nb_path, tmp_path):
        r = tool_edit_notebook_cell(tmp_path, "n.ipynb", 9, "x")
        assert not r.ok and "out of range" in r.output

    def test_not_a_notebook(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        r = tool_edit_notebook_cell(tmp_path, "a.py", 0, "x")
        assert not r.ok and "Not a notebook" in r.output



class TestLineCountIsWhatWcReports:
    """A final newline ends the last line; it does not start another."""

    def test_a_thousand_line_file_is_a_thousand_lines(self, tmp_path):
        (tmp_path / "t.txt").write_text("x\n" * 1000, encoding="utf-8")
        r = tool_read_file(tmp_path, "t.txt")
        assert "<lines>1000</lines>" in r.output
        assert "1000 lines" in r.summary

    def test_a_file_without_a_final_newline_counts_its_last_line(self, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\nc", encoding="utf-8")
        assert "<lines>3</lines>" in tool_read_file(tmp_path, "t.txt").output

    def test_an_offset_one_past_the_last_line_is_refused(self, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\n", encoding="utf-8")
        r = tool_read_file(tmp_path, "t.txt", offset=3)
        assert not r.ok
        assert "(2 lines)" in r.output

    def test_write_file_reports_the_lines_it_wrote(self, tmp_path):
        from localm.plugins.coder.tools import tool_write_file
        r = tool_write_file(tmp_path, "w.txt", "a\nb\n")
        assert r.ok
        assert "(2 lines)" in r.output


    def test_the_count_agrees_with_how_a_slice_splits_the_lines(self, tmp_path):
        (tmp_path / "t.txt").write_text("a\x0cb\n", encoding="utf-8")
        assert "<lines>2</lines>" in tool_read_file(tmp_path, "t.txt").output
        r = tool_read_file(tmp_path, "t.txt", offset=2)
        assert r.ok and "b" in r.output
