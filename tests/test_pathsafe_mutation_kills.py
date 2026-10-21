# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pins the exact refusals of localm.pathsafe's lexical name checks: which
names are refused, with which status and message, and what counts as a Windows
device name."""

import types

import pytest
from fastapi import HTTPException

from localm import pathsafe


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
