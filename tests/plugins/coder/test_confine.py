# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for _confine() path traversal protection and _verify_syntax().

_confine() is the security boundary for all file tools - it must reliably
reject any path that resolves outside cwd.
"""

import sys
import pytest

from localm.plugins.coder.tools import _confine, _verify_syntax


# ---------------------------------------------------------------------------
#  _confine - happy paths
# ---------------------------------------------------------------------------

class TestConfineAllowed:
    def test_simple_relative_path(self, tmp_path):
        result = _confine(tmp_path, "foo.py")
        assert result == (tmp_path / "foo.py").resolve()

    def test_nested_relative_path(self, tmp_path):
        result = _confine(tmp_path, "src/utils/helper.py")
        assert result == (tmp_path / "src" / "utils" / "helper.py").resolve()

    def test_dot_stays_in_cwd(self, tmp_path):
        result = _confine(tmp_path, ".")
        assert result == tmp_path.resolve()

    def test_absolute_path_inside_cwd(self, tmp_path):
        target = tmp_path / "data.txt"
        result = _confine(tmp_path, str(target))
        assert result == target.resolve()

    def test_non_existent_file_still_confined(self, tmp_path):
        # File doesn't have to exist - just must be within cwd
        result = _confine(tmp_path, "new_file_not_created_yet.py")
        assert result.parent == tmp_path.resolve()


# ---------------------------------------------------------------------------
#  _confine - rejected paths
# ---------------------------------------------------------------------------

class TestConfineRejected:
    def test_parent_traversal_two_dots(self, tmp_path):
        with pytest.raises(PermissionError, match="outside the working directory"):
            _confine(tmp_path, "../sibling.txt")

    def test_parent_traversal_many_dots(self, tmp_path):
        with pytest.raises(PermissionError):
            _confine(tmp_path, "../../etc/passwd")

    def test_absolute_path_outside_cwd(self, tmp_path):
        outside = tmp_path.parent / "other_project" / "secret.py"
        with pytest.raises(PermissionError):
            _confine(tmp_path, str(outside))

    def test_traversal_through_subdir(self, tmp_path):
        # Starts in cwd, climbs out via traversal
        with pytest.raises(PermissionError):
            _confine(tmp_path, "subdir/../../etc/shadow")

    def test_cwd_root_traversal(self, tmp_path):
        if sys.platform == "win32":
            outside = "C:\\Windows\\System32\\cmd.exe"
        else:
            outside = "/etc/passwd"
        with pytest.raises(PermissionError):
            _confine(tmp_path, outside)

    def test_error_message_contains_path(self, tmp_path):
        try:
            _confine(tmp_path, "../escape.txt")
        except PermissionError as e:
            assert "escape.txt" in str(e) or "outside" in str(e)


# ---------------------------------------------------------------------------
#  _verify_syntax - Python
# ---------------------------------------------------------------------------

class TestVerifySyntaxPython:
    def _path(self, tmp_path, name="test.py"):
        return tmp_path / name

    @pytest.mark.parametrize(
        "src",
        [
            "x = 1\nprint(x)\n",
            (
                "from typing import Optional\n\n"
                "def greet(name: Optional[str] = None) -> str:\n"
                "    return f'hello {name or \"world\"}'\n"
            ),
        ],
        ids=["simple", "complex_typed"],
    )
    def test_valid_python_returns_none(self, tmp_path, src):
        result = _verify_syntax(self._path(tmp_path), src)
        assert result is None

    def test_invalid_python_returns_error_string(self, tmp_path):
        result = _verify_syntax(self._path(tmp_path), "def foo(\n  # unclosed paren\n")
        assert result is not None
        assert "syntax" in result.lower() or "error" in result.lower()

    def test_syntax_error_string_mentions_file(self, tmp_path):
        p = self._path(tmp_path, "broken.py")
        result = _verify_syntax(p, "class bad syntax here !!!")
        assert result is not None
        assert "broken.py" in result or "syntax" in result.lower()

    def test_empty_file_is_valid(self, tmp_path):
        result = _verify_syntax(self._path(tmp_path), "")
        assert result is None


# ---------------------------------------------------------------------------
#  _verify_syntax - JSON
# ---------------------------------------------------------------------------

class TestVerifySyntaxJSON:
    def _path(self, tmp_path):
        return tmp_path / "data.json"

    def test_valid_json_returns_none(self, tmp_path):
        result = _verify_syntax(self._path(tmp_path), '{"key": "value", "n": 42}')
        assert result is None

    def test_invalid_json_returns_error(self, tmp_path):
        result = _verify_syntax(self._path(tmp_path), '{key: value}')
        assert result is not None
        assert "json" in result.lower()

    def test_json_array_valid(self, tmp_path):
        result = _verify_syntax(self._path(tmp_path), '[1, 2, 3]')
        assert result is None

    def test_trailing_comma_invalid(self, tmp_path):
        result = _verify_syntax(self._path(tmp_path), '{"a": 1,}')
        assert result is not None


# ---------------------------------------------------------------------------
#  _verify_syntax - TOML
# ---------------------------------------------------------------------------

class TestVerifySyntaxTOML:
    def _path(self, tmp_path):
        return tmp_path / "config.toml"

    def test_valid_toml_returns_none(self, tmp_path):
        result = _verify_syntax(self._path(tmp_path), '[section]\nkey = "value"\n')
        assert result is None

    def test_invalid_toml_returns_error(self, tmp_path):
        result = _verify_syntax(self._path(tmp_path), 'key = {bad toml')
        assert result is not None
        assert "toml" in result.lower()


# ---------------------------------------------------------------------------
#  _verify_syntax - unknown extension
# ---------------------------------------------------------------------------

class TestVerifySyntaxUnknown:
    def test_markdown_returns_none(self, tmp_path):
        result = _verify_syntax(tmp_path / "README.md", "# Title\n\nsome text\n")
        assert result is None

    def test_txt_returns_none(self, tmp_path):
        result = _verify_syntax(tmp_path / "notes.txt", "anything goes\n")
        assert result is None

    def test_yaml_not_checked_returns_none(self, tmp_path):
        result = _verify_syntax(tmp_path / "ci.yml", "invalid: yaml: :\n")
        assert result is None
