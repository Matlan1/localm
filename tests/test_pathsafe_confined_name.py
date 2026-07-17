# SPDX-License-Identifier: AGPL-3.0-or-later
"""LM-PT-005: confined_name's docstring must not overclaim. It confines a name to
*base* (rejecting path separators, ``..``, and absolute/drive-relative names) but
it does NOT specially reject Windows reserved device names (con, nul, ...): those
pass as ordinary basenames, and confinement still holds because they resolve
directly inside *base*. The docstring is corrected to say so; behavior is
unchanged. (Blocking device names is deliberately NOT added - it would reject
legitimate names like a file literally called "con.txt"'s stem etc.)
"""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException

from localm.pathsafe import confined_name


def test_docstring_does_not_claim_device_name_rejection():
    # Normalise whitespace so wrapped phrases match regardless of line breaks.
    doc = " ".join((confined_name.__doc__ or "").lower().split())
    # The docstring used to end its "which also rejects ..." list with
    # "con/device names" - a false claim (device names are NOT rejected). That
    # overclaim tail must be gone...
    assert "/device names" not in doc
    # ...and the corrected docstring must positively document the real behavior.
    assert "not specially rejected" in doc


@pytest.mark.parametrize("name", ["con", "nul", "com1", "lpt1"])
def test_device_names_are_accepted_and_confined(tmp_path, name):
    # The real behavior the corrected docstring documents: a device-like name is
    # a legal basename, resolved directly inside base, never escaping it.
    resolved = confined_name(tmp_path, name)
    assert resolved.parent == tmp_path.resolve()
    assert resolved.name == name


@pytest.mark.parametrize("bad", ["..", ".", "", "a/b"])
def test_confinement_still_rejects_escapes(tmp_path, bad):
    # Guard the property that actually matters and is genuinely enforced. These four
    # mean the same thing on every platform; the backslash case does NOT, so it is
    # asserted per-platform below rather than sitting in this list.
    with pytest.raises(HTTPException):
        confined_name(tmp_path, bad)


# "a\b" used to live in the list above, which is why it failed on ubuntu CI: a
# backslash is a path SEPARATOR on Windows but an ordinary, legal filename character
# on POSIX, and confined_name() delegates to pathlib, which is platform-dependent by
# design. So the SAME input is two segments (an escape) on Windows and one plain
# basename on POSIX. Measured on 3.12: PureWindowsPath(r"a\b").name == "b", while
# PurePosixPath(r"a\b").name == r"a\b". Assert each platform's real behavior instead
# of dropping the case, so both stay covered and neither expectation is a guess.
@pytest.mark.skipif(os.name != "nt", reason="a backslash only separates paths on Windows")
def test_backslash_is_rejected_as_an_escape_on_windows(tmp_path):
    with pytest.raises(HTTPException):
        confined_name(tmp_path, "a\\b")


@pytest.mark.skipif(os.name == "nt", reason="a backslash is a legal filename character on POSIX")
def test_backslash_is_an_ordinary_basename_on_posix(tmp_path):
    # Not an escape here, so it is accepted - but confinement must still hold, which
    # is the property this file exists to guard.
    resolved = confined_name(tmp_path, "a\\b")
    assert resolved.parent == tmp_path.resolve()
    assert resolved.name == "a\\b"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
