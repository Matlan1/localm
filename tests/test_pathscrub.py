# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.pathscrub: the account-name/home-directory scrubber shared by every
privacy-sensitive text surface (bug reports, /debug/stacks, RAG error bodies).

Regression coverage for the repr()/json.dumps() escaping gap: text that
reaches the scrubber after being formatted with Python's repr() (an
exception's %r-rendered filename in a traceback - which is exactly what
OSError/FileNotFoundError's own __str__ does) or json.dumps() has every
backslash doubled, and neither of _sub_prefix's separator variants nor the
_USER_ROOT_PATTERN backstop matched that doubled form. So the single most
common real-world trigger - a "file not found" traceback for a model path
under the user's home directory, landing in a bug report's "Error detail"
section - slipped past both layers of the "the username backstop runs
unconditionally" guarantee the module docstring promises.
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
    path's backslashes by doubling them, and that doubled form is exactly
    what a traceback's %r-formatted filename puts in front of the scrubber.

    Path.home() is monkeypatched to WIN_HOME for most of these: in real use
    the leaking path IS the user's own home (that is where models and logs
    live), so the exact-prefix ("~") code path is the one that matters most -
    a fake account name the test process is not actually running under would
    only ever exercise the _USER_ROOT_PATTERN backstop, silently skipping the
    _sub_prefix path this class exists to cover.
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
        """The shape traceback.format_exception actually produces: OSError's
        __str__ renders its filename with %r, so the traceback text a bug
        report ships carries this doubled-backslash form, never the plain
        one, whenever the failure is a missing/unreadable file."""
        text = repr(WIN_HOME + r"\model.gguf")
        out = pathscrub.scrub_user_paths(text)
        _assert_absent(out, "victimuser")
        assert "model.gguf" in out

    def test_repr_of_a_different_account_hits_the_backstop(self, monkeypatch):
        """Path.home() cannot match this (a different account), so only the
        _USER_ROOT_PATTERN fallback can catch it - and it must catch the
        repr'd form exactly as it already catches the plain one. Overrides
        the class fixture: this case is specifically about home NOT matching."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path(r"C:\Users\nobodyhere")))
        plain = r"C:\Users\someoneelse\model.gguf"
        out_plain = pathscrub.scrub_user_paths(plain)
        _assert_absent(out_plain, "someoneelse")

        out_repr = pathscrub.scrub_user_paths(repr(plain))
        _assert_absent(out_repr, "someoneelse")
        assert "model.gguf" in out_repr

    def test_json_dumps_of_home_path_is_scrubbed(self):
        """Same doubling mechanism, a different producer - a GUI-side
        payload or a config value serialized with json.dumps hits the
        identical backslash-doubled shape."""
        text = json.dumps({"path": WIN_HOME + r"\a.gguf"})
        out = pathscrub.scrub_user_paths(text)
        _assert_absent(out, "victimuser")

    def test_real_filenotfound_traceback_is_scrubbed(self):
        """End-to-end: the actual mechanism that puts a doubled-backslash
        path in front of the scrubber in production - not a hand-built
        repr() call, but a real FileNotFoundError's own __str__, formatted
        exactly the way bugreport._format_error formats it."""
        bad_path = WIN_HOME + r"\models\missing.gguf"
        try:
            raise FileNotFoundError(2, "No such file or directory", bad_path)
        except FileNotFoundError as exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        # Guard the premise: if this ever stops being true the test below
        # would pass for a reason that has nothing to do with the scrubber.
        assert "victimuser" in tb, "test premise broken: OSError stopped repr-quoting its filename"
        out = pathscrub.scrub_user_paths(tb)
        _assert_absent(out, "victimuser")
        assert "missing.gguf" in out
