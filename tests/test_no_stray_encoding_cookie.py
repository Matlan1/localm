# SPDX-License-Identifier: AGPL-3.0-or-later
"""No Python source file may look like it declares a non-UTF-8 source encoding.

CodeQL's Python extractor applies the PEP 263 pattern to the first two lines
of a file whether or not they are comments, so prose such as
"speculative decoding: a second model" reads as ``coding: a`` and the file is
dropped from analysis with a parse error.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNED_DIRS = ("localm", "tests", "scripts")
COOKIE = re.compile(rb"coding[:=][ \t]*([-\w.]+)")
UTF8_NAMES = {b"utf-8", b"utf8", b"utf_8", b"u8"}


def _python_files():
    for name in SCANNED_DIRS:
        root = REPO_ROOT / name
        if root.is_dir():
            yield from sorted(root.rglob("*.py"))


def stray_cookie(first_two_lines: list):
    """The first match of the extractor's encoding pattern that does not name
    UTF-8, as bytes, else None."""
    for line in first_two_lines:
        for match in COOKIE.finditer(line):
            if match.group(1).lower() not in UTF8_NAMES:
                return match.group(0)
    return None


def test_the_pattern_reads_prose_as_a_cookie():
    prose = b'"""Draft-model speculative decoding: a second, smaller GGUF'
    assert stray_cookie([prose]) == b"coding: a"


@pytest.mark.parametrize("line", [
    b"# -*- coding: utf-8 -*-",
    b"# vim: set fileencoding=utf8 :",
    b'"""Speculative decoding. A second model drafts tokens.',
])
def test_a_utf8_cookie_or_plain_prose_is_accepted(line):
    assert stray_cookie([line]) is None


def test_no_python_file_declares_a_stray_encoding():
    scanned = 0
    offenders = []
    for path in _python_files():
        with open(path, "rb") as f:
            head = [f.readline(), f.readline()]
        scanned += 1
        hit = stray_cookie(head)
        if hit is not None:
            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {hit!r}")
    assert scanned > 500, f"scanned only {scanned} files"
    assert not offenders, "\n".join(offenders)
