# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the shared, non-HTTP path primitives in localm.pathsafe.

These are consumed across the codebase (media containment, the model puller, the
GUI admin routes), so they get their own truth-table tests rather than being
covered only through a caller. A security primitive with four divergent copies is
worse than any single unfixed bug, so there is exactly one implementation and this
file pins its contract.

Companion to test_pathsafe_confined_name.py, which covers the flat-basename,
HTTPException-raising variant.
"""

from __future__ import annotations

import os

import pytest

from localm.pathsafe import (confined_under, is_unc_or_device_path,
                             reject_unsafe_path_string)


# --------------------------------------------------------------------------- #
#  confined_under - nested-permitting, ValueError-raising confinement           #
# --------------------------------------------------------------------------- #

class TestConfinedUnder:

    @pytest.fixture
    def base(self, tmp_path):
        b = tmp_path / "root"
        (b / "nest" / "deep").mkdir(parents=True)
        return b

    @pytest.mark.parametrize("bad", [
        "../../victim.txt",          # traversal
        "..",
        "nest/../../victim.txt",     # traversal after a legitimate segment
        "/nonexistent/target",       # absolute POSIX
        "//host/share/x",            # UNC forward-slash
        "\\\\host\\share",           # UNC backslash
        "Q:/nonexistent/target",     # drive-qualified absolute (Q: is not mounted)
        "Q:evil",                    # drive-RELATIVE: pathlib lets this replace
                                     # the base on Windows, and is_absolute() is
                                     # False for it, so an is_absolute check alone
                                     # would miss it
        "",                          # empty
        ".",                         # collapses to base
        "   ",                       # whitespace only
        "./",
    ])
    def test_rejects_escape_and_collapse(self, base, bad):
        with pytest.raises(ValueError):
            confined_under(base, bad)

    @pytest.mark.parametrize("good", [
        "a.png",
        "nest/a.png",
        "nest/deep/b.png",
        "./a.png",
    ])
    def test_permits_nesting(self, base, good):
        """NESTING IS THE POINT. ComfyUI's `subfolder` and a HuggingFace
        `rfilename` both legitimately nest, so a flat-basename rule (confined_name)
        is the wrong primitive - only ESCAPE may be refused, never depth."""
        out = confined_under(base, good)
        assert out.relative_to(base.resolve())

    def test_result_is_strictly_below_base(self, base):
        out = confined_under(base, "nest/a.png")
        assert base.resolve() in out.parents

    def test_drive_relative_is_rejected_on_every_platform(self, base):
        """Judged by WINDOWS rules regardless of host: a remote-supplied name must
        be judged by what it would mean on the worst platform, not the running
        one. On Linux `Q:evil` is a legal filename, and treating it as one here
        would leave the Windows hole untested until it reached a Windows box."""
        with pytest.raises(ValueError):
            confined_under(base, "Q:evil")

    def test_symlink_out_of_base_is_rejected(self, base, tmp_path):
        """Lexical checks are not sufficient: a symlink INSIDE base pointing out of
        it turns a well-formed name into an escape, which is why the check is on
        the RESOLVED location."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "victim.txt").write_text("x", encoding="utf-8")
        try:
            (base / "link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as e:
            pytest.skip(f"symlink creation not permitted here: {e}")
        with pytest.raises(ValueError):
            confined_under(base, "link/victim.txt")

    def test_backslash_is_a_separator_on_every_platform(self, base):
        """Matches _apply_update._unsafe_member. A POSIX filename may legally
        contain a backslash, but splitting on it can only ever confine FURTHER,
        and one uniform rule beats a platform-conditional one."""
        with pytest.raises(ValueError):
            confined_under(base, "nest\\..\\..\\victim.txt")

    def test_does_not_require_the_target_to_exist(self, base):
        """Callers check existence themselves (a delete target may already be
        gone); confinement must not depend on it."""
        assert confined_under(base, "nest/not-created-yet.png")


# --------------------------------------------------------------------------- #
#  is_unc_or_device_path - the syntax PREDICATE (platform-independent)         #
# --------------------------------------------------------------------------- #

class TestIsUncOrDevicePath:

    @pytest.mark.parametrize("raw", [
        r"\\192.0.2.1\share",
        r"\\.\PhysicalDrive0",
        r"\\?\Q:\dir",
        "//192.0.2.1/share",
    ])
    def test_true_on_every_platform(self, raw):
        """NOT gated on os.name: this answers what the string MEANS, not where it
        is evaluated. Consumers handling remote-supplied values refuse on this
        unconditionally."""
        assert is_unc_or_device_path(raw)

    @pytest.mark.parametrize("raw", [
        "/nonexistent/x", "relative/x", "a.png", "", "Q:/x", r"Q:\x",
    ])
    def test_false_for_ordinary_paths(self, raw):
        assert not is_unc_or_device_path(raw)


# --------------------------------------------------------------------------- #
#  reject_unsafe_path_string - the POLICY for a path the USER named            #
# --------------------------------------------------------------------------- #

class TestRejectUnsafePathString:

    @pytest.mark.parametrize("raw", [
        r"\\192.0.2.1\share",
        r"\\.\PhysicalDrive0",
        r"\\?\Q:\dir",
    ])
    def test_backslash_unc_rejected_on_every_platform(self, raw):
        with pytest.raises(ValueError):
            reject_unsafe_path_string(raw)

    def test_forward_slash_unc_is_platform_split_by_design(self):
        """``//x`` is UNC on Windows but a LEGAL local prefix on POSIX (equivalent
        to ``/``). This function's input is a path the user themselves named - a
        folder picker, a configured directory - so refusing it on POSIX would break
        a legitimate local folder for no security gain. Remote-supplied values must
        use is_unc_or_device_path instead, which has no such split."""
        if os.name == "nt":
            with pytest.raises(ValueError):
                reject_unsafe_path_string("//192.0.2.1/share")
        else:
            reject_unsafe_path_string("//192.0.2.1/share")   # must NOT raise

    def test_require_absolute(self, tmp_path):
        with pytest.raises(ValueError):
            reject_unsafe_path_string("relative/x", require_absolute=True)
        with pytest.raises(ValueError):
            reject_unsafe_path_string("", require_absolute=True)
        reject_unsafe_path_string(str(tmp_path), require_absolute=True)

    def test_ordinary_paths_pass(self, tmp_path):
        reject_unsafe_path_string(str(tmp_path))
        reject_unsafe_path_string("~/Documents")       # expanded by the caller
        reject_unsafe_path_string("relative/x")        # allowed without the flag

    def test_makes_no_filesystem_call(self, monkeypatch):
        """The entire point: the syscall IS the vulnerability, so the check must
        run before one can happen. If this ever grows a stat, the UNC dial it
        exists to prevent moves back in front of it."""
        from pathlib import Path
        called = []
        for meth in ("resolve", "is_dir", "exists", "stat"):
            monkeypatch.setattr(
                Path, meth,
                lambda self, *a, _m=meth, **kw: called.append(_m))
        with pytest.raises(ValueError):
            reject_unsafe_path_string(r"\\192.0.2.1\share")
        reject_unsafe_path_string(r"Q:\ordinary")
        assert called == []
