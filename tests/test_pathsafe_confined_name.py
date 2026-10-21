# SPDX-License-Identifier: AGPL-3.0-or-later
"""confined_name's contract: it confines a name to *base* (rejecting path
separators, ``..``, and absolute/drive-relative names), rejects reserved
characters, and on Windows rejects reserved device names (con, nul.txt, com1, ...).
"""

from __future__ import annotations

import os
import sys
import types

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


def _refusal(call, *args):
    with pytest.raises(HTTPException) as exc:
        call(*args)
    return exc.value


class TestRefusesDeviceNames:
    def test_only_the_nt_platform_refuses_device_names(self, monkeypatch):
        for name, expected in [("nt", True), ("posix", False), ("NT", False),
                               ("XXntXX", False), ("", False)]:
            monkeypatch.setattr(pathsafe, "os", types.SimpleNamespace(name=name))
            assert pathsafe._refuses_device_names() is expected, name


class TestIsWindowsDeviceName:
    @pytest.mark.parametrize("name", [
        "nul", "NUL", "Nul.txt", "com1.tar.gz", "LPT9", "conin$", "CONOUT$",
        "nul ", "nul   .txt", "COM¹", "aux."])
    def test_a_reserved_stem_is_a_device_name(self, name):
        assert pathsafe.is_windows_device_name(name) is True

    @pytest.mark.parametrize("name", [
        "NULX", "nulX", "COM1X", "con\t", "nul\n", "xnul", "nul1", "COM0",
        "com10", "", ".", "a.nul", "nulx.txt"])
    def test_a_name_that_only_resembles_a_reserved_stem_is_not(self, name):
        assert pathsafe.is_windows_device_name(name) is False

    def test_only_trailing_spaces_are_removed_before_the_comparison(self):
        assert pathsafe.is_windows_device_name(" nul") is False
        assert pathsafe.is_windows_device_name("nul ") is False


class TestCheckPlainName:
    @pytest.mark.parametrize("name", ["", ".", "..", "a/b", "/abs", "../x", "a/"])
    def test_a_non_basename_is_refused_with_400_and_the_standard_message(self, name):
        err = _refusal(pathsafe.check_plain_name, name)
        assert (err.status_code, err.detail) == (400, "Invalid file name")

    @pytest.mark.parametrize("name", ["a:b", "a|b", "a?b", "a*b", 'a"b', "a<b",
                                      "a>b", "a\x00b", "a\x1fb", "a\nb"])
    def test_a_reserved_character_is_refused_with_400_and_the_standard_message(self, name):
        err = _refusal(pathsafe.check_plain_name, name)
        assert (err.status_code, err.detail) == (400, "Invalid file name")

    def test_a_device_name_is_refused_only_where_the_platform_refuses_them(
            self, monkeypatch):
        monkeypatch.setattr(pathsafe, "_refuses_device_names", lambda: True)
        err = _refusal(pathsafe.check_plain_name, "nul.txt")
        assert (err.status_code, err.detail) == (400, "Invalid file name")
        monkeypatch.setattr(pathsafe, "_refuses_device_names", lambda: False)
        pathsafe.check_plain_name("nul.txt")

    @pytest.mark.parametrize("name", ["a", "a.txt", ".hidden", "a b", "...", "a..b"])
    def test_an_ordinary_name_passes(self, name):
        assert pathsafe.check_plain_name(name) is None

    def test_a_name_that_is_not_its_own_basename_is_refused_even_if_not_empty(self):
        assert _refusal(pathsafe.check_plain_name, "dir/file").detail == "Invalid file name"


class TestConfinedNameRefusals:
    def test_the_lexical_refusal_message_is_exact(self, tmp_path):
        err = _refusal(pathsafe.confined_name, tmp_path, "a/b")
        assert (err.status_code, err.detail) == (400, "Invalid file name")

    def test_a_dot_name_is_refused_with_the_exact_message(self, tmp_path):
        for name in ["", ".", ".."]:
            err = _refusal(pathsafe.confined_name, tmp_path, name)
            assert (err.status_code, err.detail) == (400, "Invalid file name"), name

    def test_a_name_the_filesystem_cannot_resolve_is_refused_with_the_exact_message(
            self, tmp_path, monkeypatch):
        def boom(self, *a, **k):
            raise OSError("cannot resolve")
        monkeypatch.setattr(type(tmp_path), "resolve", boom)
        err = _refusal(pathsafe.confined_name, tmp_path, "a.txt")
        assert (err.status_code, err.detail) == (400, "Invalid file name")
        assert isinstance(err.__cause__, OSError)

    def test_a_value_error_from_resolution_is_refused_with_the_exact_message(
            self, tmp_path, monkeypatch):
        def boom(self, *a, **k):
            raise ValueError("embedded null")
        monkeypatch.setattr(type(tmp_path), "resolve", boom)
        err = _refusal(pathsafe.confined_name, tmp_path, "a.txt")
        assert (err.status_code, err.detail) == (400, "Invalid file name")
        assert isinstance(err.__cause__, ValueError)

    def test_a_name_that_resolves_to_a_different_name_is_refused_with_the_exact_message(
            self, tmp_path):
        target = tmp_path / "real.txt"
        target.write_text("x")
        link = tmp_path / "alias.txt"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlinks are not available")
        err = _refusal(pathsafe.confined_name, tmp_path, "alias.txt")
        assert (err.status_code, err.detail) == (400, "Invalid file name")

    def test_a_symlinked_directory_that_leaves_the_base_is_refused(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        base = tmp_path / "base"
        base.mkdir()
        try:
            (base / "jump").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not available")
        err = _refusal(pathsafe.confined_name, base, "jump")
        assert (err.status_code, err.detail) == (400, "Invalid file name")

    def test_a_plain_name_resolves_inside_the_base(self, tmp_path):
        assert pathsafe.confined_name(tmp_path, "a.txt") == tmp_path.resolve() / "a.txt"


def _message(call, *args, **kwargs):
    with pytest.raises(ValueError) as exc:
        call(*args, **kwargs)
    return str(exc.value)


class TestConfinedPartsMessages:
    @pytest.mark.parametrize("relpath", ["", "   ", None])
    def test_an_empty_path_says_so(self, relpath):
        assert _message(pathsafe._confined_parts, relpath) == "empty path"

    def test_an_absolute_path_is_named(self):
        assert _message(pathsafe._confined_parts, "/etc/x") == (
            "absolute path not allowed: '/etc/x'")

    def test_a_drive_qualified_path_is_named(self):
        assert _message(pathsafe._confined_parts, "Q:evil") == (
            "drive-qualified path not allowed: 'Q:evil'")

    @pytest.mark.parametrize("relpath", ["./.", ".", "././"])
    def test_a_path_that_collapses_to_nothing_says_so(self, relpath):
        assert _message(pathsafe._confined_parts, relpath) == (
            f"path resolves to no name: {relpath!r}")

    def test_a_traversal_component_is_named(self):
        assert _message(pathsafe._confined_parts, "a/../b") == (
            "traversal component not allowed: 'a/../b'")

    @pytest.mark.parametrize("relpath", ["dir/Q:", "dir/a:b", "x/y/Z:"])
    def test_a_drive_qualified_component_is_named_before_the_reserved_character(
            self, relpath):
        assert _message(pathsafe._confined_parts, relpath) == (
            f"drive-qualified path component not allowed: {relpath!r}")

    @pytest.mark.parametrize("relpath", ["dir/abc:d", "dir/a|b", "dir/a?b"])
    def test_a_reserved_character_in_a_component_is_named(self, relpath):
        assert _message(pathsafe._confined_parts, relpath) == (
            f"reserved character in path component not allowed: {relpath!r}")

    def test_a_clean_path_is_split_into_its_components(self):
        assert pathsafe._confined_parts("a//b/./c\\d") == ["a", "b", "c", "d"]


class TestConfinedUnderMessages:
    def test_a_path_that_leaves_the_base_names_the_base_and_the_path(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        try:
            (base / "jump").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not available")
        assert _message(pathsafe.confined_under, base, "jump/file") == (
            f"path escapes {base.resolve()}: 'jump/file'")


class TestConfinedAbsoluteOrUnder:
    def test_an_empty_or_dot_path_is_invalid(self, tmp_path):
        for raw in ["", "  ", ".", "..", None]:
            assert _message(pathsafe.confined_absolute_or_under, tmp_path, raw) == (
                f"invalid path: {raw!r}"), raw

    @pytest.mark.parametrize("raw", [
        "\\\\host\\share\\x", "//host/share/x", "\\\\?\\Q:\\x"])
    def test_a_unc_or_device_path_is_named(self, tmp_path, raw):
        assert _message(pathsafe.confined_absolute_or_under, tmp_path, raw) == (
            f"UNC and device paths are not allowed: {raw!r}")

    def test_a_reserved_character_in_a_relative_component_is_named(self, tmp_path):
        assert _message(pathsafe.confined_absolute_or_under, tmp_path, "a/b|c") == (
            "reserved character in path component not allowed: 'a/b|c'")

    def test_the_first_component_after_the_anchor_is_checked_for_reserved_characters(
            self, tmp_path):
        raw = str(tmp_path.anchor) + "a|b" + "/x"
        assert _message(pathsafe.confined_absolute_or_under, tmp_path, raw) == (
            f"reserved character in path component not allowed: {raw!r}")

    def test_an_absolute_path_outside_the_base_names_the_base_and_the_path(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        outside = str(tmp_path / "other" / "f.txt")
        assert _message(pathsafe.confined_absolute_or_under, base, outside) == (
            f"path escapes {base.resolve()}: {outside!r}")

    def test_the_base_itself_is_outside_the_base(self, tmp_path):
        assert _message(pathsafe.confined_absolute_or_under, tmp_path, str(tmp_path)) == (
            f"path escapes {tmp_path.resolve()}: {str(tmp_path)!r}")

    def test_a_path_that_is_only_an_anchor_resolves_to_no_name(self, tmp_path):
        anchor = tmp_path.anchor
        assert _message(pathsafe.confined_absolute_or_under, tmp_path, anchor) == (
            f"path resolves to no name: {anchor!r}")

    def test_an_absolute_path_inside_the_base_is_accepted(self, tmp_path):
        target = tmp_path / "sub" / "f.txt"
        assert pathsafe.confined_absolute_or_under(tmp_path, str(target)) == (
            tmp_path.resolve() / "sub" / "f.txt")


class TestRejectUnsafePathString:
    def test_a_backslash_unc_path_is_refused_with_the_exact_message(self):
        assert _message(pathsafe.reject_unsafe_path_string, "\\\\host\\share") == (
            "UNC and device paths are not allowed")
        assert _message(pathsafe.reject_unsafe_path_string, "\\/host/share") == (
            "UNC and device paths are not allowed")

    def test_a_slash_unc_path_is_refused_only_where_the_platform_is_windows(
            self, monkeypatch):
        monkeypatch.setattr(pathsafe, "os", types.SimpleNamespace(name="nt"))
        assert _message(pathsafe.reject_unsafe_path_string, "//host/share") == (
            "UNC and device paths are not allowed")
        monkeypatch.setattr(pathsafe, "os", types.SimpleNamespace(name="posix"))
        pathsafe.reject_unsafe_path_string("//host/share")

    @pytest.mark.parametrize("name", ["NT", "XXntXX"])
    def test_other_platform_names_do_not_refuse_a_slash_path(self, monkeypatch, name):
        monkeypatch.setattr(pathsafe, "os", types.SimpleNamespace(name=name))
        pathsafe.reject_unsafe_path_string("//host/share")

    def test_the_unc_judgement_receives_the_path(self, monkeypatch):
        seen = []
        monkeypatch.setattr(pathsafe, "os", types.SimpleNamespace(name="nt"))
        monkeypatch.setattr(pathsafe, "is_unc_or_device_path",
                            lambda s: seen.append(s) or False)
        pathsafe.reject_unsafe_path_string("Q:\\ok")
        assert seen == ["Q:\\ok"]

    def test_network_drives_are_refused_only_when_asked(self, monkeypatch):
        seen = []
        monkeypatch.setattr(pathsafe, "is_mapped_network_drive",
                            lambda s: seen.append(s) or True)
        pathsafe.reject_unsafe_path_string("Z:\\x")
        assert seen == []
        assert _message(pathsafe.reject_unsafe_path_string, "Z:\\x",
                        reject_network_drives=True) == "network drives are not allowed"
        assert seen == ["Z:\\x"]

    def test_a_relative_path_is_refused_when_absolute_is_required(self):
        assert _message(pathsafe.reject_unsafe_path_string, "rel/x",
                        require_absolute=True) == "path must be absolute"
        assert _message(pathsafe.reject_unsafe_path_string, None,
                        require_absolute=True) == "path must be absolute"
        pathsafe.reject_unsafe_path_string("rel/x")


class TestIsMappedNetworkDrive:
    @staticmethod
    def _fake_windows(monkeypatch, drive_type, calls):
        monkeypatch.setattr(pathsafe, "os", types.SimpleNamespace(name="nt"))

        def get_drive_type(root):
            calls.append(root)
            return drive_type

        monkeypatch.setitem(sys.modules, "ctypes", types.SimpleNamespace(
            windll=types.SimpleNamespace(
                kernel32=types.SimpleNamespace(GetDriveTypeW=get_drive_type))))

    def test_a_remote_drive_letter_is_reported_on_windows(self, monkeypatch):
        calls = []
        self._fake_windows(monkeypatch, 4, calls)
        assert pathsafe.is_mapped_network_drive("Z:\\dir\\f") is True
        assert calls == ["Z:\\"]

    def test_a_local_drive_letter_is_not_reported(self, monkeypatch):
        calls = []
        self._fake_windows(monkeypatch, 3, calls)
        assert pathsafe.is_mapped_network_drive("Q:\\f") is False

    def test_a_path_without_a_drive_letter_makes_no_win32_call(self, monkeypatch):
        calls = []
        self._fake_windows(monkeypatch, 4, calls)
        assert pathsafe.is_mapped_network_drive("rel\\f") is False
        assert pathsafe.is_mapped_network_drive("\\\\host\\share") is False
        assert calls == []

    @pytest.mark.parametrize("name", ["posix", "NT", "XXntXX"])
    def test_other_platforms_never_report_a_mapped_drive(self, monkeypatch, name):
        calls = []
        self._fake_windows(monkeypatch, 4, calls)
        monkeypatch.setattr(pathsafe, "os", types.SimpleNamespace(name=name))
        assert pathsafe.is_mapped_network_drive("Z:\\f") is False
        assert calls == []
