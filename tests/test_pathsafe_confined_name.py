# SPDX-License-Identifier: AGPL-3.0-or-later
"""confined_name's contract: it confines a name to *base* (rejecting path
separators, ``..``, and absolute/drive-relative names), rejects reserved
characters, and on Windows rejects reserved device names (con, nul.txt, com1, ...).
"""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException

from localm import pathsafe
from localm.pathsafe import confined_name


@pytest.mark.parametrize("name", [
    "con", "nul", "NUL", "com1", "lpt9", "aux.txt", "nul.tar.gz", "prn ", "conin$",
    "conout$.log", "com¹", "Lpt³.bin",
])
def test_device_names_are_rejected_before_any_write(tmp_path, name, monkeypatch):
    monkeypatch.setattr(pathsafe, "_refuses_device_names", lambda: True)
    with pytest.raises(HTTPException) as ei:
        confined_name(tmp_path, name)
    assert ei.value.status_code == 400
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["console.txt", "nully", "com10", "lpt0", "my-nul.txt",
                                  "auxiliary.bin", "conin"])
def test_names_that_only_look_like_devices_are_accepted(tmp_path, name, monkeypatch):
    monkeypatch.setattr(pathsafe, "_refuses_device_names", lambda: True)
    resolved = confined_name(tmp_path, name)
    assert resolved.parent == tmp_path.resolve()
    assert resolved.name == name


def test_device_names_are_refused_exactly_on_windows():
    assert pathsafe._refuses_device_names() is (os.name == "nt")


@pytest.mark.skipif(os.name == "nt", reason="a device name is an ordinary file name off Windows")
@pytest.mark.parametrize("name", ["aux.png", "con.json", "nul.mp3"])
def test_an_existing_device_named_file_stays_reachable_off_windows(tmp_path, name):
    (tmp_path / name).write_bytes(b"x")
    assert pathsafe.confined_file(tmp_path, name, "file").read_bytes() == b"x"


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
