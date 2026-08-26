# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for _confine() path traversal protection and _verify_syntax().

_confine() is the security boundary for all file tools - it must reliably
reject any path that resolves outside cwd.
"""

import os
import tempfile
from pathlib import Path

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
        # The traversal syntax is the payload; the leaf is a neutral name.
        with pytest.raises(PermissionError):
            _confine(tmp_path, "../../up/two/levels.txt")

    def test_absolute_path_outside_cwd(self, tmp_path):
        outside = tmp_path.parent / "other_project" / "secret.py"
        with pytest.raises(PermissionError):
            _confine(tmp_path, str(outside))

    def test_traversal_through_subdir(self, tmp_path):
        # Starts in cwd, climbs out via traversal
        with pytest.raises(PermissionError):
            _confine(tmp_path, "subdir/../../up/out.txt")

    def test_absolute_path_to_an_existing_file_outside_cwd(self, tmp_path,
                                                           tmp_path_factory):
        """The refusal must not rest on the target merely being absent: a file
        that really exists outside cwd is rejected just the same.

        A disposable file the test owns, never a real OS path. _confine()
        resolve()s whatever it is handed - it has to, or a symlink would slip
        past - so a system target here would make the test suite itself open a
        real system file."""
        outside = tmp_path_factory.mktemp("outside_cwd") / "real.txt"
        outside.write_text("disposable\n", encoding="utf-8")
        with pytest.raises(PermissionError):
            _confine(tmp_path, str(outside))

    def test_error_message_contains_path(self, tmp_path):
        try:
            _confine(tmp_path, "../escape.txt")
        except PermissionError as e:
            assert "escape.txt" in str(e) or "outside" in str(e)


# ---------------------------------------------------------------------------
#  _confine - hardening migrated from pathsafe.confined_absolute_or_under
# ---------------------------------------------------------------------------

class TestConfineHardening:
    @pytest.mark.parametrize("raw", [r"\\192.0.2.1\share\x", "//192.0.2.1/share/x"])
    def test_unc_path_is_refused(self, tmp_path, raw):
        """Path(raw).is_absolute() is True for a UNC path, so with no UNC
        guard it reaches .resolve() unconditionally - the exact
        SMB-dial-and-hang danger reject_unsafe_path_string exists to prevent,
        on a sink this function shares with it."""
        with pytest.raises(PermissionError):
            _confine(tmp_path, raw)

    def test_unc_path_reaches_no_filesystem_call(self, tmp_path, monkeypatch):
        """ORDER, not verdict: the hostile UNC string itself must never reach
        .resolve() (that syscall is the SMB dial). _confine's own except
        branch DOES call cwd.resolve() to format the error message - that is
        the TRUSTED cwd, not attacker data, so it is excluded from the
        assertion rather than asserting zero resolve() calls of any kind."""
        seen = []
        real_resolve = Path.resolve
        monkeypatch.setattr(
            Path, "resolve",
            lambda self, *a, **k: (seen.append(str(self)), real_resolve(self, *a, **k))[1])
        with pytest.raises(PermissionError):
            _confine(tmp_path, r"\\192.0.2.1\share\x")
        assert not any("192.0.2.1" in s for s in seen), (
            f"the hostile UNC string was resolved: {seen}")

    @pytest.mark.parametrize("bad", [
        "somefile.exe:hidden.gguf", "src/somefile.exe:hidden.gguf",
    ])
    def test_reserved_characters_are_rejected(self, tmp_path, bad):
        """The NTFS Alternate Data Stream class: with no character check a
        colon stays confined (containment holds) while opening a hidden
        stream behind an apparently-empty sibling."""
        with pytest.raises(PermissionError):
            _confine(tmp_path, bad)

    def test_alias_leaf_is_rejected(self, tmp_path, monkeypatch):
        """An OS-level short-name alias resolving `path` to a DIFFERENT real
        sibling stays strictly inside cwd - containment alone would not
        catch it. Deterministic simulation."""
        victim = tmp_path / "LongModelNameThatIsVeryLong.py"
        victim.write_text("SECRET", encoding="utf-8")
        alias = "LONGMO~1.PY"
        real_resolve = Path.resolve

        def fake_resolve(self, *a, **k):
            if self.name == alias:
                return victim.resolve()
            return real_resolve(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", fake_resolve)
        with pytest.raises(PermissionError):
            _confine(tmp_path, alias)


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

    def test_leaves_no_compiled_artifact_in_system_temp(self, tmp_path):
        """_verify_syntax must not check Python syntax by writing the content
        to a real temp .py file and compiling it with py_compile: CPython's
        own import-cache write leaves a matching .pyc - this file's compiled
        CONTENT - behind in the system temp dir's __pycache__/, in every
        session mode, and nothing unlinks it. compile() the builtin parses to
        an in-memory code object and touches disk nowhere, so there is nothing
        left to gate by mode here - this proves the artifact class is gone
        rather than merely suppressed.

        Diffed against a before/after snapshot, not "the dir is empty",
        because this is a SHARED system temp dir other processes may also
        write .pyc files into concurrently on this box."""
        pycache = Path(tempfile.gettempdir()) / "__pycache__"
        before = set(pycache.glob("*.pyc")) if pycache.is_dir() else set()

        canary = f"localm_verify_syntax_canary_{os.getpid()}"
        src = f"def foo():\n    return '{canary}'\n"
        result = _verify_syntax(self._path(tmp_path, "leak_check.py"), src)
        assert result is None

        after = set(pycache.glob("*.pyc")) if pycache.is_dir() else set()
        for p in after - before:
            try:
                data = p.read_bytes()
            except OSError:
                continue
            assert canary.encode() not in data, f"compiled artifact leaked into {p}"


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
