# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tmp_path disk guard must actually FIRE, and must not false-positive.

conftest's `_no_giant_tmp_files` exists because `truncate()` is NOT sparse on
Windows/NTFS - it allocates the blocks for real (`fsutil file queryValidData` on
a 200 MB truncated file reports Valid Data Length = the full 200,000,000; a
sparse file reports 0). A test module that truncates a fake multi-GB GGUF to
drive size-reading code allocates hundreds of GB per suite run and fills the
disk.

A guard nobody proves can fire is worth nothing, so these drive it end to end
through real sub-pytest runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_REAL_CONFTEST = str(Path(__file__).parent / "conftest.py").replace("\\", "/")


def _with_real_conftest(pytester):
    """Point the sub-run at the REAL tests/conftest.py, so this exercises the
    shipped guard rather than a copy that could drift from it.

    Loaded by PATH under a distinct module name: a `from conftest import *`
    silently skips underscore-prefixed names, so the guard never arrives and the
    sub-run passes with nothing installed (the test then proves nothing while
    looking green); and naming it `conftest` explicitly hits a circular import,
    because the generated sub-conftest is ITSELF `conftest.py`."""
    pytester.makeconftest(
        "import importlib.util as _u, sys\n"
        "_s = _u.spec_from_file_location('_localm_real_conftest', r'"
        + _REAL_CONFTEST + "')\n"
        "_m = _u.module_from_spec(_s)\n"
        "sys.modules['_localm_real_conftest'] = _m\n"
        "_s.loader.exec_module(_m)\n"
        "_no_giant_tmp_files = _m._no_giant_tmp_files\n"
    )


class TestGuardFires:
    """The guard runs AFTER the test body (it can only measure what was left
    behind), so it reports as a teardown ERROR, not a test FAILURE - the body's own
    assertions really did pass. Either way the run is non-zero and the offender is
    named, which is what matters."""

    def test_a_truncated_giant_file_is_reported(self, pytester):
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            def test_writes_a_giant_file(tmp_path):
                with open(tmp_path / "model.gguf", "wb") as fh:
                    fh.truncate(200_000_000)     # 200 MB, real blocks on NTFS
                assert True                       # body passes; the guard must object
        """)
        result = pytester.runpytest("-q", "-p", "no:cacheprovider")
        result.assert_outcomes(passed=1, errors=1)
        assert result.ret != 0, "a violating run must not exit green"
        result.stdout.fnmatch_lines(["*over 100 MB left in tmp_path*"])

    def test_the_report_names_the_offending_file_and_the_fix(self, pytester):
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            def test_writes_a_giant_file(tmp_path):
                with open(tmp_path / "huge.bin", "wb") as fh:
                    fh.truncate(200_000_000)
        """)
        result = pytester.runpytest("-q", "-p", "no:cacheprovider")
        result.assert_outcomes(passed=1, errors=1)
        out = result.stdout.str()
        assert "huge.bin" in out, "must name the file"
        assert "_model_bytes" in out, "must point at the fix, not just complain"
        assert "not sparse" in out.lower(), "must say WHY"


class TestGuardDoesNotFalsePositive:
    def test_a_small_file_is_fine(self, pytester):
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            def test_small(tmp_path):
                (tmp_path / "small.bin").write_bytes(b"x" * 1024)
        """)
        pytester.runpytest("-q", "-p", "no:cacheprovider").assert_outcomes(passed=1)

    def test_a_FAKED_giant_size_is_fine(self, pytester):
        """THE regression that matters. A test module may monkeypatch Path.stat
        to report 15-40 GB for a tiny file - the correct way to drive a
        size-reading path. A guard that read Path.stat would see the fake and
        fail exactly those tests, so it must measure REAL bytes via os.stat."""
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            import os
            from pathlib import Path

            def test_fake_stat_is_not_real_bytes(tmp_path, monkeypatch):
                p = tmp_path / "model.gguf"
                p.touch()
                real_stat = Path.stat

                def fake_stat(self, *, follow_symlinks=True):
                    r = real_stat(self, follow_symlinks=follow_symlinks)
                    if self == p:
                        return os.stat_result(
                            (r.st_mode, r.st_ino, r.st_dev, r.st_nlink, r.st_uid,
                             r.st_gid, 40_000_000_000, r.st_atime, r.st_mtime,
                             r.st_ctime))
                    return r

                monkeypatch.setattr(Path, "stat", fake_stat)
                assert p.stat().st_size == 40_000_000_000   # the fake works
                assert os.stat(p).st_size == 0              # but disk is empty
        """)
        pytester.runpytest("-q", "-p", "no:cacheprovider").assert_outcomes(passed=1)

    def test_the_opt_out_marker_is_honoured(self, pytester):
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            import pytest

            @pytest.mark.allow_large_tmp_files
            def test_genuinely_needs_real_bytes(tmp_path):
                with open(tmp_path / "big.bin", "wb") as fh:
                    fh.truncate(200_000_000)
        """)
        pytester.runpytest("-q", "-p", "no:cacheprovider").assert_outcomes(passed=1)

    def test_an_integration_test_is_exempt(self, pytester):
        """Integration tests pull real models by design."""
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            import pytest

            @pytest.mark.integration
            def test_pulls_a_real_model(tmp_path):
                with open(tmp_path / "real.gguf", "wb") as fh:
                    fh.truncate(200_000_000)
        """)
        pytester.runpytest("-q", "-p", "no:cacheprovider").assert_outcomes(passed=1)


class TestGuardDoesNotLoopOnAJunction:
    """A guard that walks tmp_path must not re-introduce the junction DoS.

    A robustness-sweep fixture elsewhere deliberately builds BRANCHING
    self-referential junctions in its tmp_path. A Windows junction reports
    is_symlink() False, so a plain os.walk descends it forever - which is why
    localm/rag/store.py has _walk_files instead of rglob. A guard built on a
    naive os.walk hangs the suite on exactly that fixture, so this pins it.
    """

    @pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
    def test_a_self_referential_junction_does_not_hang_the_guard(self, pytester):
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            import subprocess

            def test_makes_a_junction_loop(tmp_path):
                d = tmp_path / "docs"
                d.mkdir()
                (d / "note.txt").write_bytes(b"x")
                # Branching self-referential junctions: a single one is depth-capped,
                # two make the walk explode combinatorially if it descends them.
                for jn in ("loop1", "loop2"):
                    r = subprocess.run(["cmd", "/c", "mklink", "/J",
                                        str(d / jn), str(d)],
                                       capture_output=True)
                    if r.returncode != 0:
                        import pytest as _p
                        _p.skip("could not create a junction on this box")
        """)
        # The assertion IS termination: a guard that descends loop1/loop2 forever
        # never returns, so reaching assert_outcomes at all is the proof.
        # Pytester.runpytest takes no timeout kwarg.
        result = pytester.runpytest("-q", "-p", "no:cacheprovider")
        result.assert_outcomes(passed=1)

    def test_a_symlink_loop_does_not_hang_the_guard(self, pytester):
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            import os

            def test_makes_a_symlink_loop(tmp_path):
                d = tmp_path / "docs"
                d.mkdir()
                (d / "note.txt").write_bytes(b"x")
                try:
                    os.symlink(d, d / "self", target_is_directory=True)
                except (OSError, NotImplementedError) as e:
                    import pytest as _p
                    _p.skip(f"cannot create a symlink here: {e}")
        """)
        result = pytester.runpytest("-q", "-p", "no:cacheprovider")
        result.assert_outcomes(passed=1)


class TestGuardIsDriveAgnostic:
    """localm must work installed anywhere, including C:. The guard is about how
    many bytes a test allocates, not where tmp_path happens to live, so it must
    never branch on a drive letter or path prefix - otherwise it would behave
    differently for a contributor whose temp dir is on another drive."""

    def _guard_code(self):
        """The guard's executable body, docstring EXCLUDED. Parsed rather than
        grepped: the docstring legitimately mentions drive letters while
        explaining that it ignores them, which a raw grep would flag as a
        violation."""
        import ast
        tree = ast.parse(Path(_REAL_CONFTEST).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_no_giant_tmp_files":
                body = list(node.body)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    body = body[1:]          # drop the docstring
                return [ast.dump(n) for n in body]
        raise AssertionError("_no_giant_tmp_files not found in conftest")

    def test_no_drive_letter_appears_in_the_guards_logic(self):
        dumped = " ".join(self._guard_code())
        for forbidden in ("C:", "D:", "/tmp", "USERPROFILE"):
            assert forbidden not in dumped, (
                f"the guard branches on a path/drive ({forbidden!r}); it must judge "
                "SIZE only, so it behaves identically on any install location")

    def test_the_guard_is_test_only_and_touches_no_product_code(self):
        """It lives in tests/conftest.py and is a pytest fixture, so it cannot
        affect a real install on any drive."""
        assert Path(_REAL_CONFTEST).parts[-2:] == ("tests", "conftest.py")
