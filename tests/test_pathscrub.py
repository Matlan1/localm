# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.pathscrub: the account-name/home-directory scrubber shared by every
privacy-sensitive text surface (bug reports, /debug/stacks, RAG error bodies).

Covers text that reaches the scrubber after being formatted with Python's
repr() (an exception's %r-rendered filename in a traceback, which is what
OSError/FileNotFoundError's own __str__ produces) or with json.dumps(): both
double every backslash in a Windows path.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pytest

from localm import pathscrub

WIN_HOME = r"C:\Users\victimuser"


def _assert_absent(haystack: str, needle: str) -> None:
    assert needle not in haystack, f"still discloses {needle!r} in {haystack!r}"


@pytest.mark.skipif(sys.platform != "win32",
                     reason="backslash-doubled escaping is a Windows path artifact")
class TestReprDoubledBackslashScrub:
    """The account name must not survive being formatted through repr() or
    json.dumps() before it reaches the scrubber - both escape a Windows
    path's backslashes by doubling them.

    An autouse fixture points Path.home() at WIN_HOME, so these cases run
    through the exact-prefix ("~") _sub_prefix path unless a test overrides
    it.
    """

    @pytest.fixture(autouse=True)
    def _home_is_win_home(self, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path(WIN_HOME)))

    def test_plain_path_still_scrubbed(self):
        """Guard the baseline the rest of this class assumes still holds."""
        text = WIN_HOME + r"\logs\a.txt"
        out = pathscrub.scrub_user_paths(text)
        _assert_absent(out, "victimuser")
        assert out.startswith("~"), out

    def test_repr_of_exact_home_path_is_scrubbed(self):
        """The doubled-backslash shape traceback.format_exception produces:
        OSError's __str__ renders its filename with %r."""
        text = repr(WIN_HOME + r"\model.gguf")
        out = pathscrub.scrub_user_paths(text)
        _assert_absent(out, "victimuser")
        assert "model.gguf" in out

    def test_repr_of_a_different_account_hits_the_backstop(self, monkeypatch):
        """Path.home() names a different account, so only the
        _USER_ROOT_PATTERN fallback can catch it, in the repr'd form as well
        as the plain one. Overrides the class fixture so home does not match."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path(r"C:\Users\nobodyhere")))
        plain = r"C:\Users\someoneelse\model.gguf"
        out_plain = pathscrub.scrub_user_paths(plain)
        _assert_absent(out_plain, "someoneelse")

        out_repr = pathscrub.scrub_user_paths(repr(plain))
        _assert_absent(out_repr, "someoneelse")
        assert "model.gguf" in out_repr

    def test_json_dumps_of_home_path_is_scrubbed(self):
        """json.dumps produces the same backslash-doubled shape."""
        text = json.dumps({"path": WIN_HOME + r"\a.gguf"})
        out = pathscrub.scrub_user_paths(text)
        _assert_absent(out, "victimuser")

    def test_real_filenotfound_traceback_is_scrubbed(self):
        """End to end: a real FileNotFoundError's own __str__, formatted the
        way bugreport._format_error formats it."""
        bad_path = WIN_HOME + r"\models\missing.gguf"
        try:
            raise FileNotFoundError(2, "No such file or directory", bad_path)
        except FileNotFoundError as exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        # Guard the premise: the traceback carries the account name.
        assert "victimuser" in tb, "test premise broken: OSError stopped repr-quoting its filename"
        out = pathscrub.scrub_user_paths(tb)
        _assert_absent(out, "victimuser")
        assert "missing.gguf" in out
