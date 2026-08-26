# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the shared, non-HTTP path primitives in localm.pathsafe.

These are consumed across the codebase (media containment, the model puller, the
GUI admin routes), so they get their own truth-table tests rather than being
covered only through a caller. There is exactly one implementation and this file
pins its contract.

Companion to test_pathsafe_confined_name.py, which covers the flat-basename,
HTTPException-raising variant.
"""

from __future__ import annotations

import ntpath
import os
from pathlib import Path, PureWindowsPath

import pytest

from localm.pathsafe import (confined_absolute_or_under, confined_under,
                             is_mapped_network_drive, is_unc_or_device_path,
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
        """REGRESSION. A check that tests position 1 of the WHOLE string only
        sees a drive on the FIRST component, so ``a/C:evil`` passes it.

        That is not a harmless miss. pathlib joins a drive-relative component
        against a SAME-DRIVE base by silently DROPPING the drive:
        base.joinpath("a", "C:evil") -> base/a/evil. The result stays strictly
        under base, so the resolved-containment check cannot see it - it is not
        an escape, it is a SILENT RENAME. At the ComfyUI delete call site that
        means unlink()ing a real file that is NOT the one ComfyUI named, while
        reporting containment succeeded.

        It also only reproduces when base is on the same drive as the injected
        letter, so it presents as "works on my D: install, deletes the wrong file
        on a C: one"."""
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
        """Lexical checks are not sufficient: a symlink INSIDE base pointing out
        of it turns a well-formed name into an escape, so the check is on the
        RESOLVED location."""
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

    @pytest.mark.parametrize("bad", [
        "../../victim.txt", "..", "/nonexistent/target", "//host/share/x",
        "\\\\host\\share", "\\/host\\share\\x", "/\\host/share/x",
        "Q:/nonexistent/target", "Q:evil", "a/Q:evil", "a/b/Q:evil",
        "", ".", "   ", "./", "nest\\..\\..\\victim.txt",
        "somefile.exe:hidden.gguf", "nest/somefile.exe:hidden.gguf",
    ])
    def test_a_rejected_path_reaches_NO_filesystem_call(self, base, bad,
                                                        monkeypatch):
        """ORDER, not verdict: a rejected input must be refused BEFORE any
        syscall, not after one.

        A correct check paid for too late is worthless: a caller that calls
        Path(raw).expanduser().resolve() FIRST and consults its allowlist after
        does refuse a UNC input, but only once the SMB dial has already happened.

        confined_under does every lexical rejection before it touches
        joinpath().resolve(), and this pins that ordering so a future
        "simplification" that resolves first cannot pass by still returning the
        right answer."""
        seen = []
        for meth in ("resolve", "is_dir", "is_file", "exists", "stat",
                     "iterdir", "expanduser", "lstat", "glob"):
            real = getattr(Path, meth)
            monkeypatch.setattr(
                Path, meth,
                lambda self, *a, _m=meth, _r=real, **kw: (
                    seen.append(_m), _r(self, *a, **kw))[1])

        with pytest.raises(ValueError):
            confined_under(base, bad)
        assert seen == [], (
            f"{bad!r} was rejected, but only AFTER {seen} - on a UNC path that "
            f"is the SMB dial, so the refusal came too late to matter")

    def test_does_not_require_the_target_to_exist(self, base):
        """Callers check existence themselves (a delete target may already be
        gone); confinement must not depend on it."""
        assert confined_under(base, "nest/not-created-yet.png")

    @pytest.mark.parametrize("bad", [
        "somefile.exe:hidden.gguf",
        "nest/somefile.exe:hidden.gguf",
        "nest/deep/n<o>.txt",
        "p|q.png",
        "ev\x00il.txt",
    ])
    def test_reserved_characters_are_rejected(self, base, bad):
        """NTFS Alternate Data Stream syntax must be rejected, the same class
        model_manager/gguf.py's _safe_models_filename and confined_name reject.
        Without this check confined_under(base, "sub/somefile.exe:hidden.gguf")
        is accepted and a write through the returned path succeeds, landing an
        invisible stream behind a visible, apparently-empty sibling."""
        with pytest.raises(ValueError):
            confined_under(base, bad)

    def test_reserved_characters_are_rejected_before_any_write(self, base):
        with pytest.raises(ValueError):
            confined_under(base, "somefile.exe:hidden.gguf")
        assert not (base / "somefile.exe").exists(), (
            "a rejected ADS-shaped name must never reach a real write")

    @pytest.mark.parametrize("good", ["a.png", "nest/a.png", "café.png"])
    def test_ordinary_names_are_unaffected(self, base, good):
        out = confined_under(base, good)
        assert out.name == good.rsplit("/", 1)[-1]

    # ----------------------------------------------------------------------- #
    #  Short-name alias substitution                                          #
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _mock_alias_resolve(monkeypatch, alias_name, real_name):
        """Simulate an NTFS 8.3 short name: whenever a path being resolved has
        *alias_name* as one of its COMPONENTS - leaf or intermediate - replace
        that component with *real_name* before resolving for real.

        Path.resolve() is ONE call on the whole joined path (there is no
        per-component resolve() a monkeypatch could see individually), so a mock
        keyed only on the final component's name cannot model an alias in an
        earlier (subfolder) position - it has to inspect .parts. A resolved
        object's .name is the LEAF ("output.png"), not the aliased subfolder
        segment. Deterministic, no real 8.3-enabled volume needed."""
        real_resolve = Path.resolve

        def fake_resolve(self, *a, **k):
            parts = list(self.parts)
            if alias_name in parts:
                parts[parts.index(alias_name)] = real_name
                return real_resolve(Path(*parts), *a, **k)
            return real_resolve(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", fake_resolve)

    def test_alias_leaf_component_is_rejected(self, base, monkeypatch):
        victim = base / "LongModelNameThatIsVeryLong.gguf"
        victim.write_text("victim", encoding="utf-8")
        alias = "LONGMO~1.GGU"
        self._mock_alias_resolve(monkeypatch, alias, victim.name)

        with pytest.raises(ValueError):
            confined_under(base, alias)

    def test_alias_intermediate_component_is_rejected(self, base, monkeypatch):
        """The leaf isn't the only place an alias can hide: `subfolder` is
        ComfyUI's own nesting feature, and a short name there resolves the
        same way a filename does. Containment alone (strictly under base)
        would not catch it - the aliased sibling directory is still under
        base, just not the one requested."""
        real_dir = base / "LongSubfolderName"
        real_dir.mkdir()
        (real_dir / "output.png").write_text("victim", encoding="utf-8")
        alias = "LONGSU~1"
        self._mock_alias_resolve(monkeypatch, alias, real_dir.name)

        with pytest.raises(ValueError):
            confined_under(base, f"{alias}/output.png")

    def test_alias_rejection_matches_confined_name_wording(self, base, monkeypatch):
        """Not load-bearing on exact text, just confirms the new branch is what
        actually fires (not some other rejection) so this test cannot pass for
        the wrong reason."""
        victim = base / "LongModelNameThatIsVeryLong.gguf"
        victim.write_text("victim", encoding="utf-8")
        alias = "LONGMO~1.GGU"
        self._mock_alias_resolve(monkeypatch, alias, victim.name)

        with pytest.raises(ValueError, match="short-name alias"):
            confined_under(base, alias)


# --------------------------------------------------------------------------- #
#  confined_absolute_or_under - like confined_under, but an absolute *raw*     #
#  landing inside base is accepted rather than rejected as an escape.          #
# --------------------------------------------------------------------------- #

class TestConfinedAbsoluteOrUnder:

    @pytest.fixture
    def base(self, tmp_path):
        b = tmp_path / "root"
        (b / "nest").mkdir(parents=True)
        return b

    def test_relative_nested_is_permitted(self, base):
        out = confined_absolute_or_under(base, "nest/a.png")
        assert out == (base / "nest" / "a.png").resolve()

    def test_absolute_path_inside_base_is_accepted(self, base):
        target = str(base / "nest" / "a.png")
        out = confined_absolute_or_under(base, target)
        assert out == (base / "nest" / "a.png").resolve()

    def test_absolute_path_outside_base_is_refused(self, base, tmp_path):
        outside = tmp_path / "elsewhere" / "victim.txt"
        with pytest.raises(ValueError):
            confined_absolute_or_under(base, str(outside))

    @pytest.mark.parametrize("bad", [
        "../../victim.txt", "..", ".", "", "   ",
    ])
    def test_rejects_escape_and_collapse(self, base, bad):
        with pytest.raises(ValueError):
            confined_absolute_or_under(base, bad)

    def test_result_never_equals_base_itself(self, base):
        with pytest.raises(ValueError):
            confined_absolute_or_under(base, str(base))

    @pytest.mark.parametrize("raw", [r"\\192.0.2.1\share\x", "//192.0.2.1/share/x"])
    def test_unc_is_refused(self, base, raw):
        with pytest.raises(ValueError):
            confined_absolute_or_under(base, raw)

    def test_unc_reaches_no_filesystem_call(self, base, monkeypatch):
        """Same ordering discipline as confined_under's own test: the SMB dial
        is the vulnerability, so the refusal must come before resolve()."""
        seen = []
        real_resolve = Path.resolve
        monkeypatch.setattr(
            Path, "resolve",
            lambda self, *a, **k: (seen.append(1), real_resolve(self, *a, **k))[1])
        with pytest.raises(ValueError):
            confined_absolute_or_under(base, r"\\192.0.2.1\share\x")
        assert seen == []

    @pytest.mark.parametrize("bad", [
        "somefile.exe:hidden.gguf",
        "nest/somefile.exe:hidden.gguf",
        "n<o>.txt", "p|q.png", "ev\x00il.txt",
    ])
    def test_reserved_characters_are_rejected(self, base, bad):
        with pytest.raises(ValueError):
            confined_absolute_or_under(base, bad)

    def test_reserved_character_in_absolute_form_is_also_rejected(self, base):
        with pytest.raises(ValueError):
            confined_absolute_or_under(base, str(base / "note.txt:hidden"))

    @pytest.mark.parametrize("good", ["a.png", "nest/a.png"])
    def test_ordinary_names_are_unaffected(self, base, good):
        out = confined_absolute_or_under(base, good)
        assert out.name == good.rsplit("/", 1)[-1]

    @staticmethod
    def _mock_alias_resolve(monkeypatch, alias_name, real_name):
        real_resolve = Path.resolve

        def fake_resolve(self, *a, **k):
            parts = list(self.parts)
            if alias_name in parts:
                parts[parts.index(alias_name)] = real_name
                return real_resolve(Path(*parts), *a, **k)
            return real_resolve(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", fake_resolve)

    def test_alias_leaf_relative_is_rejected(self, base, monkeypatch):
        victim = base / "LongModelNameThatIsVeryLong.gguf"
        victim.write_text("victim", encoding="utf-8")
        alias = "LONGMO~1.GGU"
        self._mock_alias_resolve(monkeypatch, alias, victim.name)

        with pytest.raises(ValueError, match="short-name alias"):
            confined_absolute_or_under(base, alias)

    def test_alias_leaf_absolute_is_rejected(self, base, monkeypatch):
        """The same substitution, but *raw* is the ABSOLUTE form - the branch
        confined_under does not have at all."""
        victim = base / "LongModelNameThatIsVeryLong.gguf"
        victim.write_text("victim", encoding="utf-8")
        alias = "LONGMO~1.GGU"
        self._mock_alias_resolve(monkeypatch, alias, victim.name)

        with pytest.raises(ValueError, match="short-name alias"):
            confined_absolute_or_under(base, str(base / alias))

    def test_alias_intermediate_component_is_rejected(self, base, monkeypatch):
        real_dir = base / "LongSubfolderName"
        real_dir.mkdir()
        (real_dir / "output.png").write_text("victim", encoding="utf-8")
        alias = "LONGSU~1"
        self._mock_alias_resolve(monkeypatch, alias, real_dir.name)

        with pytest.raises(ValueError):
            confined_absolute_or_under(base, f"{alias}/output.png")


# --------------------------------------------------------------------------- #
#  is_unc_or_device_path - the syntax PREDICATE (platform-independent)         #
# --------------------------------------------------------------------------- #

class TestIsUncOrDevicePath:

    @pytest.mark.parametrize("raw", [
        r"\\192.0.2.1\share",
        r"\\.\PhysicalDrive0",
        r"\\?\Q:\dir",
        "//192.0.2.1/share",
        # Mixed separators. Windows treats a backslash and a forward slash
        # interchangeably in the UNC prefix, so all four of these are UNC to the
        # OS: PureWindowsPath(...).drive reports the same for every one.
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

        Windows does not strip leading whitespace to reveal a UNC prefix.
        ``ntpath.normpath`` on a space-padded double-backslash path collapses the
        DOUBLED separator to a single one and keeps the spaces, so ``abspath``
        resolves it RELATIVE to the process cwd, under a directory literally
        named "  ". There is no share, no dial, and nothing to guard against.

        An implementation that normalised with ``.strip()`` before testing the
        prefix would return True here and REFUSE a path Windows treats as an
        ordinary relative one - a false positive in a security predicate.

        Both authoritative Windows parsers agree: PureWindowsPath(...).drive
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
        a scheme check inside a security primitive would be a second concern the
        next caller would either bend or fork.

        A caller composing its own scheme check locally uses
        ``^[A-Za-z][A-Za-z0-9+.-]+://`` - note the ``+`` rather than ``*``, which
        makes a single-letter drive unmatchable so ``C://models/x.gguf`` stays a
        path (PureWindowsPath('C://models/x.gguf').drive == 'C:')."""
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
        so on nt they must be refused - including the mixed spelling that combines
        a forward slash with a backslash. On POSIX they are ordinary local paths
        and must NOT be refused, so this is platform-split, not unconditional."""
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


# --------------------------------------------------------------------------- #
#  is_mapped_network_drive - whether a drive letter is actually MAPPED to a   #
#  network share. is_unc_or_device_path returns False for the local-drive     #
#  form "Z:", which is a different, syntax-only question.                     #
# --------------------------------------------------------------------------- #

class TestIsMappedNetworkDrive:

    @pytest.mark.parametrize("raw", [
        r"\\192.0.2.1\share",   # UNC - not a drive letter at all
        "relative/x",
        "",
        "   ",
        "/usr/local/share/x.gguf",
    ])
    def test_false_without_any_win32_call(self, raw, monkeypatch):
        """No drive-letter prefix (or a non-Windows platform) must short-
        circuit before ever touching ctypes - proven by making the Win32 call
        itself explode if reached."""
        import ctypes

        def boom(root):
            raise AssertionError(f"GetDriveTypeW must not be called for {raw!r}")

        if os.name == "nt":
            monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", boom)
        assert not is_mapped_network_drive(raw)

    def test_false_on_non_windows(self, monkeypatch):
        """A stale os.name check must be provably load-bearing, not merely
        coincidental with this machine's real drive table. On a Windows box,
        prime GetDriveTypeW to report REMOTE BEFORE patching os.name to
        "posix" - if the `if os.name != "nt": return False` guard were ever
        removed, the function would fall through to that primed mock and
        return True, failing this assertion. Without priming it, "Z:"
        probably does not exist on the test box, so a guard-less version
        would coincidentally return False anyway (GetDriveTypeW answering
        DRIVE_NO_ROOT_DIR) and this test would prove nothing."""
        if os.name == "nt":
            import ctypes
            monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW",
                                lambda root: 4)   # DRIVE_REMOTE
        monkeypatch.setattr(os, "name", "posix")
        assert not is_mapped_network_drive("Z:\\shared\\docs")

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_true_when_getdrivetype_reports_remote(self, monkeypatch):
        import ctypes
        seen = []

        def fake(root):
            seen.append(root)
            return 4   # DRIVE_REMOTE

        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", fake)
        assert is_mapped_network_drive(r"Z:\shared\docs\a.gguf")
        # Queried the DRIVE ROOT, not the full path - GetDriveTypeW's contract.
        assert seen == ["Z:\\"]

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_drive_relative_spelling_still_queries_the_drive(self, monkeypatch):
        """"Q:x" (drive-relative, no separator) genuinely names drive Q -
        ntpath.splitdrive("Q:x") == ("Q:", "x"), matching
        is_unc_or_device_path's own test corpus comment for the same string.
        So this MUST reach the Win32 call, unlike a string with no drive
        prefix at all (test_false_without_any_win32_call above)."""
        import ctypes
        seen = []
        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW",
                            lambda root: seen.append(root) or 4)
        assert is_mapped_network_drive("Q:x")
        assert seen == ["Q:\\"]

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    @pytest.mark.parametrize("code", [0, 1, 2, 3, 5, 6])   # every non-REMOTE code
    def test_false_for_every_non_remote_drive_type(self, monkeypatch, code):
        import ctypes
        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW",
                            lambda root: code)
        assert not is_mapped_network_drive(r"C:\Users")

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_never_raises_when_win32_call_fails(self, monkeypatch):
        import ctypes

        def boom(root):
            raise OSError("no such drive")

        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", boom)
        assert is_mapped_network_drive(r"Q:\gone") is False

    @pytest.mark.skipif(os.name != "nt", reason="asserts against THIS machine's "
                        "real drive table")
    def test_real_local_drive_is_not_reported_remote(self, tmp_path):
        """Smoke test against the actual dev box, no mocking: tmp_path's own
        drive is a real local disk, never a network share."""
        drive = str(tmp_path)[:2]
        assert drive[1] == ":", f"corpus error: {tmp_path} has no drive letter"
        assert not is_mapped_network_drive(str(tmp_path))

    def test_classification_gap_is_real(self, monkeypatch):
        """is_unc_or_device_path's own docstring says "Z:" is the ordinary
        local-drive form, and it must keep saying so - that predicate's contract
        does not change. is_mapped_network_drive is what tells a REMOTE "Z:"
        apart from a local one, which is_unc_or_device_path was never designed to
        answer."""
        raw = r"Z:\shared\docs"
        assert not is_unc_or_device_path(raw)
        if os.name == "nt":
            import ctypes
            monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW",
                                lambda root: 4)
            assert is_mapped_network_drive(raw)


# --------------------------------------------------------------------------- #
#  reject_unsafe_path_string(reject_network_drives=...) - opt-in gate         #
# --------------------------------------------------------------------------- #

class TestRejectUnsafePathStringNetworkDrives:

    def test_ordinary_drive_allowed_regardless_of_the_flag(self, tmp_path):
        """Control, matches TestRejectUnsafePathString.test_makes_no_filesystem_
        call's existing pin byte-for-byte: an ordinary drive letter must never
        be rejected merely because the new keyword exists."""
        reject_unsafe_path_string(r"Q:\ordinary")
        reject_unsafe_path_string(r"Q:\ordinary", reject_network_drives=True)

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_network_drive_rejected_only_when_opted_in(self, monkeypatch):
        import ctypes
        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW",
                            lambda root: 4)   # DRIVE_REMOTE
        # Default (every pre-existing caller, unchanged): must NOT raise.
        reject_unsafe_path_string(r"Z:\shared")
        # Opted in: must raise.
        with pytest.raises(ValueError):
            reject_unsafe_path_string(r"Z:\shared", reject_network_drives=True)

    def test_real_local_drive_never_rejected_even_when_opted_in(self, tmp_path):
        """Control against this box's real (non-network) drive: opting in
        must not turn into a blanket rejection of every drive letter."""
        reject_unsafe_path_string(str(tmp_path), reject_network_drives=True)
