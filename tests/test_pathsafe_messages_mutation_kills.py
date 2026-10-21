# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pins the exact refusal text and the platform and flag branches of the
localm.pathsafe path validators that the other pathsafe tests only exercise by
exception type."""

import sys
import types

import pytest

from localm import pathsafe


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
