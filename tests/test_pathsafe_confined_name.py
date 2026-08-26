# SPDX-License-Identifier: AGPL-3.0-or-later
"""confined_name's contract, and that its docstring states it without
overclaiming. It confines a name to *base* (rejecting path separators, ``..``,
and absolute/drive-relative names) but does NOT specially reject Windows
reserved device names (con, nul, ...): those pass as ordinary basenames, and
confinement still holds because they resolve directly inside *base*.
"""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException

from localm.pathsafe import confined_name


def test_docstring_does_not_claim_device_name_rejection():
    # Normalise whitespace so wrapped phrases match regardless of line breaks.
    doc = " ".join((confined_name.__doc__ or "").lower().split())
    # The docstring's rejection list must not mention con/device names: device
    # names are not rejected.
    assert "/device names" not in doc
    # ...and it positively documents the real behavior.
    assert "not specially rejected" in doc


@pytest.mark.parametrize("name", ["con", "nul", "com1", "lpt1"])
def test_device_names_are_accepted_and_confined(tmp_path, name):
    # A device-like name is a legal basename, resolved directly inside base,
    # never escaping it.
    resolved = confined_name(tmp_path, name)
    assert resolved.parent == tmp_path.resolve()
    assert resolved.name == name


@pytest.mark.parametrize("bad", ["..", ".", "", "a/b"])
def test_confinement_still_rejects_escapes(tmp_path, bad):
    # These four mean the same thing on every platform; the backslash case does
    # NOT, so it is asserted per-platform below rather than sitting in this list.
    with pytest.raises(HTTPException):
        confined_name(tmp_path, bad)


# A backslash is a path SEPARATOR on Windows but an ordinary, legal filename
# character on POSIX, and confined_name() delegates to pathlib, which is
# platform-dependent. So the SAME input is two segments (an escape) on Windows
# and one plain basename on POSIX; each platform is asserted separately.
@pytest.mark.skipif(os.name != "nt", reason="a backslash only separates paths on Windows")
def test_backslash_is_rejected_as_an_escape_on_windows(tmp_path):
    with pytest.raises(HTTPException):
        confined_name(tmp_path, "a\\b")


@pytest.mark.skipif(os.name == "nt", reason="a backslash is a legal filename character on POSIX")
def test_backslash_is_an_ordinary_basename_on_posix(tmp_path):
    # Not an escape here, so it is accepted - but confinement must still hold.
    resolved = confined_name(tmp_path, "a\\b")
    assert resolved.parent == tmp_path.resolve()
    assert resolved.name == "a\\b"


# --------------------------------------------------------------------------- #
#  NTFS Alternate Data Stream rejection                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [
    "somefile.exe:hidden.gguf",
    "note.txt:evil",
    "n<o>.txt", 'n"o.txt', "p|q", "p?q", "p*q",
    "ev\x00il.txt",
])
def test_reserved_characters_are_rejected(tmp_path, bad):
    """':' is the character with a live consequence: on NTFS it does not fail
    file creation, it opens an Alternate Data Stream, so
    'somefile.exe:hidden.gguf' passes a containment check (the write lands
    inside base) while writing invisibly behind an apparently-empty sibling
    file. confined_name rejects it with a 400."""
    with pytest.raises(HTTPException) as ei:
        confined_name(tmp_path, bad)
    assert ei.value.status_code == 400


def test_reserved_characters_are_rejected_before_any_write(tmp_path):
    """The rejection happens before the caller ever gets a path back to write
    through: nothing lands on disk."""
    with pytest.raises(HTTPException):
        confined_name(tmp_path, "somefile.exe:hidden.gguf")
    assert list(tmp_path.iterdir()) == [], (
        "a rejected ADS-shaped name must never reach a real write")


def test_single_letter_drive_shaped_name_is_still_rejected(tmp_path):
    """A single-letter-drive-shaped name ('a:b.txt') is refused end to end,
    whichever check inside confined_name catches it: on Windows,
    name != Path(name).name strips the drive before the character check runs,
    while the character check is what catches a multi-character ADS name like
    'somefile.exe:hidden.gguf', which has no drive-letter form to strip."""
    with pytest.raises(HTTPException):
        confined_name(tmp_path, "a:b.txt")


@pytest.mark.parametrize("good", ["a.txt", "model.gguf", "under_score-dash.bin",
                                  "spaced name.txt", "unicode-café.txt"])
def test_ordinary_names_are_unaffected(tmp_path, good):
    """The reserved-character check leaves ordinary names accepted."""
    resolved = confined_name(tmp_path, good)
    assert resolved.name == good


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
