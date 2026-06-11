"""Tests for localm.plugins.coder._patch (unified diff applier)."""

import pytest
from localm.plugins.coder._patch import apply_diff, PatchError


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def make_file(*lines):
    """Join lines with newlines, ending with a trailing newline."""
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  Basic cases
# ---------------------------------------------------------------------------

def test_simple_replace():
    original = make_file("a", "b", "c", "d", "e")
    diff = (
        "--- a/f\n"
        "+++ b/f\n"
        "@@ -2,3 +2,3 @@\n"
        " b\n"
        "-c\n"
        "+C\n"
        " d\n"
    )
    result = apply_diff(original, diff)
    assert result == make_file("a", "b", "C", "d", "e")


def test_pure_addition():
    original = make_file("a", "b", "c")
    diff = (
        "@@ -2,1 +2,2 @@\n"
        " b\n"
        "+b2\n"
    )
    result = apply_diff(original, diff)
    assert result == make_file("a", "b", "b2", "c")


def test_pure_deletion():
    original = make_file("a", "b", "c", "d")
    diff = (
        "@@ -2,2 +2,1 @@\n"
        " b\n"
        "-c\n"
    )
    result = apply_diff(original, diff)
    assert result == make_file("a", "b", "d")


def test_multi_hunk():
    original = make_file("a", "b", "c", "d", "e", "f", "g")
    diff = (
        "@@ -1,2 +1,2 @@\n"
        "-a\n"
        "+A\n"
        " b\n"
        "@@ -6,2 +6,2 @@\n"
        " f\n"
        "-g\n"
        "+G\n"
    )
    result = apply_diff(original, diff)
    assert result == make_file("A", "b", "c", "d", "e", "f", "G")


def test_no_file_headers():
    """Diff without --- +++ headers still works."""
    original = make_file("x", "y", "z")
    diff = (
        "@@ -2,1 +2,1 @@\n"
        "-y\n"
        "+Y\n"
    )
    result = apply_diff(original, diff)
    assert result == make_file("x", "Y", "z")


def test_file_headers_optional():
    """--- +++ headers are parsed but not required for correctness."""
    original = make_file("1", "2", "3")
    diff = (
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " 1\n"
        "-2\n"
        "+two\n"
        " 3\n"
    )
    result = apply_diff(original, diff)
    assert result == make_file("1", "two", "3")


def test_trailing_newline_preserved():
    original = "a\nb\nc\n"
    diff = "@@ -2,1 +2,1 @@\n-b\n+B\n"
    result = apply_diff(original, diff)
    assert result.endswith("\n")


def test_no_trailing_newline_preserved():
    original = "a\nb\nc"
    diff = "@@ -2,1 +2,1 @@\n-b\n+B\n"
    result = apply_diff(original, diff)
    # original had no trailing newline
    assert not result.endswith("\n\n")


def test_crlf_preserved():
    original = "a\r\nb\r\nc\r\n"
    diff = "@@ -2,1 +2,1 @@\n-b\n+B\n"
    result = apply_diff(original, diff)
    assert "\r\n" in result


# ---------------------------------------------------------------------------
#  Fuzzy matching
# ---------------------------------------------------------------------------

def test_fuzzy_line_number_off_by_one():
    """Hint line number in @@ is 1 off — applier should still find the context."""
    original = make_file("a", "b", "TARGET", "d", "e")
    diff = (
        "@@ -4,1 +4,1 @@\n"   # wrong: TARGET is at line 3
        "-TARGET\n"
        "+REPLACED\n"
    )
    result = apply_diff(original, diff)
    assert "REPLACED" in result
    assert "TARGET" not in result


def test_fuzzy_large_offset():
    """Hint is 15 lines off — still within the fuzz=20 window."""
    lines = [str(i) for i in range(1, 31)]  # "1" .. "30"
    original = "\n".join(lines) + "\n"
    target_line = "15"
    diff = (
        "@@ -1,1 +1,1 @@\n"   # wrong hint (off by 14)
        f"-{target_line}\n"
        "+fifteen\n"
    )
    result = apply_diff(original, diff)
    assert "fifteen" in result
    assert target_line not in result.split("\n")


# ---------------------------------------------------------------------------
#  Error cases
# ---------------------------------------------------------------------------

def test_no_hunks_raises():
    with pytest.raises(PatchError, match="No hunks"):
        apply_diff("hello\n", "--- a/f\n+++ b/f\n")


def test_context_mismatch_raises():
    original = make_file("a", "b", "c")
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " X\n"    # 'X' not in original
        "-b\n"
        "+B\n"
        " Y\n"
    )
    with pytest.raises(PatchError, match="could not be applied"):
        apply_diff(original, diff)


def test_no_newline_marker_ignored():
    """'\\ No newline at end of file' lines are silently tolerated."""
    original = make_file("a", "b")
    diff = (
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+A\n"
        "\\ No newline at end of file\n"
    )
    result = apply_diff(original, diff)
    assert result.startswith("A")


# ---------------------------------------------------------------------------
#  Realistic case — multi-hunk Python function edit
# ---------------------------------------------------------------------------

ORIGINAL_PY = """\
def greet(name):
    print("Hello,", name)
    return None


def farewell(name):
    print("Goodbye,", name)
    return None
"""

PATCHED_PY = """\
def greet(name: str) -> None:
    print("Hello,", name)
    return None


def farewell(name: str) -> None:
    print("Goodbye,", name)
    return None
"""

DIFF_PY = """\
--- a/hello.py
+++ b/hello.py
@@ -1,2 +1,2 @@
-def greet(name):
+def greet(name: str) -> None:
     print("Hello,", name)
@@ -6,2 +6,2 @@
-def farewell(name):
+def farewell(name: str) -> None:
     print("Goodbye,", name)
"""


def test_realistic_python_patch():
    result = apply_diff(ORIGINAL_PY, DIFF_PY)
    assert result == PATCHED_PY
