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

import ntpath
import os
from pathlib import PureWindowsPath

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

    @pytest.mark.parametrize("bad", [
        "a/C:evil", "a/b/C:evil", "a/D:evil", "a/b/Q:z",
    ])
    def test_drive_on_a_NESTED_component_is_rejected(self, base, bad):
        """REGRESSION. The original check tested position 1 of the WHOLE string,
        so it only saw a drive on the FIRST component and ``a/C:evil`` passed.

        That is not a harmless miss. pathlib joins a drive-relative component
        against a SAME-DRIVE base by silently DROPPING the drive:
        base.joinpath("a", "C:evil") -> base/a/evil (measured on 3.12). The result
        stays strictly under base, so the resolved-containment check cannot see
        it - it is not an escape, it is a SILENT RENAME. At the ComfyUI delete
        call site that means unlink()ing a real file that is NOT the one ComfyUI
        named, while reporting containment succeeded.

        It also only reproduces when base is on the same drive as the injected
        letter, so it would present as "works on my D: install, deletes the wrong
        file on a C: one". Found by the WS2 lane."""
        with pytest.raises(ValueError):
            confined_under(base, bad)

    def test_nesting_never_silently_renames(self, base):
        """The property the bug above violated: whatever the caller named as the
        final component is what comes back. Asserting only 'stayed under base'
        would have passed the silent-rename bug."""
        for good in ["a.png", "nest/a.png", "nest/deep/b.png"]:
            want = good.rsplit("/", 1)[-1]
            assert confined_under(base, good).name == want

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
        # REGRESSION: mixed separators. Windows treats "\" and "/"
        # interchangeably in the UNC prefix, so all four of these are UNC to the
        # OS - PureWindowsPath(...).drive reports "\\host\share" for every one.
        # The original predicate tested startswith("\\\\") or startswith("//")
        # and returned False for the two mixed spellings: a live bypass in
        # exactly the position this predicate guards, since its documented use is
        # REMOTE-supplied values. Found by the WS7 lane, confirmed by WS2.
        "\\/192.0.2.1\\share",
        "/\\192.0.2.1/share",
        "\\/.\\PhysicalDrive0",
        "/\\?/Q:/dir",
        # Device and extended-length namespaces, caught by the splitdrive
        # backstop rather than the prefix table.
        "//./pipe/x",
        r"\\?\UNC\host\share",
    ])
    def test_true_on_every_platform(self, raw):
        """NOT gated on os.name: this answers what the string MEANS, not where it
        is evaluated. Consumers handling remote-supplied values refuse on this
        unconditionally.

        Cross-checked against Windows' own parser rather than only against the
        implementation, so the test cannot drift with the code it guards."""
        assert is_unc_or_device_path(raw)
        assert PureWindowsPath(raw).drive, (
            "corpus error: Windows does not consider this a drive/UNC root")

    @pytest.mark.parametrize("raw", [
        "/nonexistent/x", "relative/x", "a.png", "", "   ", "Q:/x", r"Q:\x",
        "Q://models/x.gguf",     # drive + DOUBLED slash: the drive-vs-scheme edge
        "Q:x",                   # drive-RELATIVE - not UNC (confined_under still
                                 # rejects it as a component; different question)
        "/usr/local/share/x.gguf",   # the false-positive trap: a POSIX absolute
                                     # path containing the word "share". Any rule
                                     # matching on "share" or on a single leading
                                     # slash fails here.
        "./rel/x.gguf",
        "~/models/x.gguf",       # tilde, unexpanded
        "bge-small-en-v1.5",     # a model key, not a path at all
        "my-registered-model",
    ])
    def test_false_for_ordinary_paths(self, raw):
        """Corpus contributed by the WS7 lane, which consumes this predicate for
        remote-supplied embedding specs."""
        assert not is_unc_or_device_path(raw)

    @pytest.mark.parametrize("raw", [
        "  \\\\host\\share\\x",     # leading spaces
        "\t\\\\host\\share\\x",     # leading tab
        "  //host/share/x",
        "\n\\\\host\\share",
    ])
    def test_whitespace_prefixed_is_NOT_unc_and_must_not_be_stripped(self, raw):
        """A padded UNC-looking string is NOT UNC, and this must stay False.

        Counter-intuitive, so it is pinned with the mechanism: Windows does not
        strip leading whitespace to reveal a UNC prefix. ``ntpath.normpath`` on
        "  \\\\\\\\host\\\\share\\\\x" yields "  \\\\host\\\\share\\\\x" - the DOUBLED separator
        collapses to a single one and the spaces are kept - so ``abspath`` resolves
        it RELATIVE to the process cwd, under a directory literally named "  ".
        There is no share, no dial, and nothing to guard against.

        This exists because a proposed alternative implementation normalised with
        ``.strip()`` before testing the prefix. That would return True here and
        REFUSE a path Windows treats as an ordinary relative one - a false
        positive in a security predicate, which is how a guard starts breaking
        legitimate input and gets weakened later to compensate.

        Both authoritative Windows parsers agree with us: PureWindowsPath(...).drive
        and ntpath.splitdrive both return "" for every string here."""
        assert not is_unc_or_device_path(raw)
        assert not PureWindowsPath(raw).drive, "corpus error: Windows sees a drive"
        assert not ntpath.splitdrive(ntpath.normpath(raw))[0], (
            "corpus error: normalization exposed a drive/UNC root")

    @pytest.mark.parametrize("raw", [
        "http://e/x", "https://e/x", "file:///nonexistent/x", "smb://h/s",
    ])
    def test_url_schemes_are_deliberately_NOT_judged_here(self, raw):
        """BOUNDARY, asserted so nobody "fixes" it by adding scheme handling.

        pathsafe answers "given that this IS a path, is it confined / is it UNC".
        Whether a string is a path at all or a URL is CALLER policy: the rules
        differ per call site (an embedding spec, a model ref, a media source), and
        a scheme check smuggled in here would be a second concern inside a
        security primitive that the next caller would either bend or fork.

        The WS7 lane composes its own scheme check locally, using
        ``^[A-Za-z][A-Za-z0-9+.-]+://`` - note the ``+`` rather than ``*``, which
        makes a single-letter drive unmatchable so ``C://models/x.gguf`` stays a
        path (verified: PureWindowsPath('C://models/x.gguf').drive == 'C:')."""
        assert not is_unc_or_device_path(raw)


# --------------------------------------------------------------------------- #
#  reject_unsafe_path_string - the POLICY for a path the USER named            #
# --------------------------------------------------------------------------- #

class TestRejectUnsafePathString:

    @pytest.mark.parametrize("raw", [
        r"\\192.0.2.1\share",
        r"\\.\PhysicalDrive0",
        r"\\?\Q:\dir",
        "\\/192.0.2.1\\share",     # REGRESSION: mixed separator, backslash-led
    ])
    def test_backslash_unc_rejected_on_every_platform(self, raw):
        """Backslash-led forms are refused everywhere: a leading "\\\\" or "\\/" is
        not a meaningful local-path prefix on POSIX either, so nothing legitimate
        is lost by refusing them there."""
        with pytest.raises(ValueError):
            reject_unsafe_path_string(raw)

    @pytest.mark.parametrize("raw", ["//192.0.2.1/share", "/\\192.0.2.1/share"])
    def test_slash_led_unc_rejected_on_windows(self, raw):
        """REGRESSION. These reach the filesystem as UNC on Windows and dial SMB,
        so on nt they must be refused - including the mixed "/\\" spelling, which
        an earlier revision let through. On POSIX they are ordinary local paths
        and must NOT be refused, which is why this is platform-split rather than
        unconditional."""
        if os.name == "nt":
            with pytest.raises(ValueError):
                reject_unsafe_path_string(raw)
        else:
            reject_unsafe_path_string(raw)   # must NOT raise

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
