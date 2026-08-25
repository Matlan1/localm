# SPDX-License-Identifier: AGPL-3.0-or-later
"""Provisioning must refuse BEFORE deleting when the runtime is in use, and
`doctor` must fail an install whose BLAS kernel data is missing.

Both come from one real incident: a `setup-llama --force` on a machine that had
the runtime loaded left the install with 14 of 24 libraries and 0 of 995 rocBLAS
kernels, while every existing check still reported it healthy.
"""

import contextlib
import ctypes
import shutil
import sys
from pathlib import Path

import pytest

from localm import setup_llama


# --------------------------------------------------------------------------- #
#  _clear_target reports what it could not remove
# --------------------------------------------------------------------------- #

def test_clear_target_returns_empty_list_when_everything_was_removed(tmp_path):
    (tmp_path / "llama.dll").write_bytes(b"\x00")
    (tmp_path / "ggml-base.dll").write_bytes(b"\x00")
    assert setup_llama._clear_target(tmp_path) == []
    assert not (tmp_path / "llama.dll").exists()


def test_clear_target_preserves_the_git_sentinels(tmp_path):
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    (tmp_path / "llama.dll").write_bytes(b"\x00")
    assert setup_llama._clear_target(tmp_path) == []
    assert (tmp_path / ".gitignore").exists(), "the tracked sentinel must survive"


def test_clear_target_reports_a_file_it_could_not_remove(tmp_path, monkeypatch):
    """The regression this whole change exists for.

    Asserted from OUTSIDE via the return value, not by raising from a patched
    unlink: `_clear_target` catches OSError by design, so an exception-based
    assertion would be swallowed by the code under test and the test would pass
    in both directions (diff-review-discipline item 13)."""
    (tmp_path / "llama.dll").write_bytes(b"\x00")
    (tmp_path / "ggml-base.dll").write_bytes(b"\x00")

    real_unlink = Path.unlink

    def refuse_one(self, *a, **kw):
        if self.name == "llama.dll":
            raise PermissionError(13, "in use")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", refuse_one)
    left = setup_llama._clear_target(tmp_path)

    assert [p.name for p in left] == ["llama.dll"]
    assert not (tmp_path / "ggml-base.dll").exists(), "the removable one still goes"


# --------------------------------------------------------------------------- #
#  the pre-flight refuses BEFORE deleting anything
# --------------------------------------------------------------------------- #

def test_clear_target_or_refuse_deletes_nothing_when_a_file_is_in_use(tmp_path, monkeypatch):
    """The point of the pre-flight: not merely an honest error, but an INTACT
    install. Reporting after the fact would still leave a half-cleared dir."""
    (tmp_path / "llama.dll").write_bytes(b"\x00")
    (tmp_path / "ggml-base.dll").write_bytes(b"\x00")
    (tmp_path / "rocblas.dll").write_bytes(b"\x00")

    monkeypatch.setattr(setup_llama, "_files_in_use",
                        lambda t: [t / "llama.dll"])

    with pytest.raises(setup_llama.RuntimeInUseError) as ei:
        setup_llama._clear_target_or_refuse(tmp_path)

    assert ei.value.partial is False
    assert [p.name for p in ei.value.locked] == ["llama.dll"]
    # every file still present: nothing was deleted
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "ggml-base.dll", "llama.dll", "rocblas.dll"]


def test_clear_target_or_refuse_flags_the_probe_to_unlink_race_as_partial(tmp_path, monkeypatch):
    """If a file is locked AFTER the probe passes, files are already gone. That
    cannot be prevented here, but it must never be silent, and it must not claim
    the install is intact."""
    (tmp_path / "llama.dll").write_bytes(b"\x00")
    (tmp_path / "ggml-base.dll").write_bytes(b"\x00")

    monkeypatch.setattr(setup_llama, "_files_in_use", lambda t: [])
    real_unlink = Path.unlink

    def refuse_one(self, *a, **kw):
        if self.name == "llama.dll":
            raise PermissionError(13, "in use")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", refuse_one)

    with pytest.raises(setup_llama.RuntimeInUseError) as ei:
        setup_llama._clear_target_or_refuse(tmp_path)

    assert ei.value.partial is True, "a half-cleared dir must not report as intact"


def test_exit_runtime_in_use_exits_non_zero(capsys):
    """The self-updater runs `setup-llama` as a post-swap command and ROLLS THE
    WHOLE UPDATE BACK on a non-zero exit (updater.py `_rollback_or_raise`). That
    is the wanted behaviour here and it is why this contract is pinned: a runtime
    that could not be re-provisioned beside freshly swapped code is exactly the
    ABI mismatch the isolation exists to avoid, so completing the update would be
    worse than undoing it. Before this change the same situation could silently
    fall back to a different backend and report success."""
    with pytest.raises(SystemExit) as ei:
        setup_llama._exit_runtime_in_use(
            setup_llama.RuntimeInUseError([Path("llama.dll")], partial=False))
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "in use" in out
    assert "left untouched" in out, "an intact install must say so"


def test_exit_runtime_in_use_does_not_claim_intact_after_a_partial_clear(capsys):
    with pytest.raises(SystemExit):
        setup_llama._exit_runtime_in_use(
            setup_llama.RuntimeInUseError([Path("llama.dll")], partial=True))
    out = capsys.readouterr().out
    assert "left untouched" not in out, "it was NOT untouched; saying so would be a lie"
    assert "incomplete" in out


def test_clear_target_or_refuse_is_a_no_op_signal_on_a_clean_dir(tmp_path):
    (tmp_path / "llama.dll").write_bytes(b"\x00")
    setup_llama._clear_target_or_refuse(tmp_path)          # must not raise
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
#  the lock probe itself
# --------------------------------------------------------------------------- #

def test_files_in_use_finds_nothing_in_an_ordinary_directory(tmp_path):
    (tmp_path / "llama.dll").write_bytes(b"\x00")
    (tmp_path / "ggml-base.dll").write_bytes(b"\x00")
    assert setup_llama._files_in_use(tmp_path) == []


def test_files_in_use_ignores_the_preserved_sentinels(tmp_path):
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    assert setup_llama._files_in_use(tmp_path) == []


@pytest.mark.skipif(sys.platform != "win32",
                    reason="only Windows denies write access to a loaded library; "
                           "POSIX unlinks an open file happily, so there is no "
                           "half-state to detect and the probe correctly finds none")
def test_files_in_use_detects_a_genuinely_loaded_library(tmp_path):
    """The probe against a REAL loaded DLL, not a mock of one.

    Uses the machine's own provisioned runtime as the specimen (copied out, never
    touched in place). Skips rather than asserts when no runtime is provisioned,
    since that is an environment fact and not a defect."""
    from localm.config import find_binary_dir
    src_dir = find_binary_dir()
    if not src_dir:
        pytest.skip("no llama runtime provisioned on this machine")
    src = next((p for p in Path(src_dir).iterdir()
                if p.name.lower().startswith("ggml-base")), None)
    if src is None:
        pytest.skip("no ggml-base library to use as a specimen")

    spec = tmp_path / src.name
    shutil.copy2(src, spec)
    assert setup_llama._files_in_use(tmp_path) == [], "not loaded yet"

    lib = ctypes.CDLL(str(spec))                      # noqa: F841  (keep it loaded)
    assert [p.name for p in setup_llama._files_in_use(tmp_path)] == [spec.name]


# --------------------------------------------------------------------------- #
#  BLAS kernel data
# --------------------------------------------------------------------------- #

def _install(d: Path, libs, kernels=None):
    d.mkdir(parents=True, exist_ok=True)
    for name in libs:
        (d / name).write_bytes(b"\x00" * 16)
    for sub, n in (kernels or {}).items():
        s = d / sub
        s.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (s / f"kernel{i}.dat").write_bytes(b"\x00")
    return d


def test_blas_ok_when_rocblas_has_its_kernels(tmp_path):
    d = _install(tmp_path / "i", ["llama.dll", "rocblas.dll"], {"rocblas/library": 3})
    assert setup_llama.blas_kernel_problems(d) == []


def test_blas_flags_an_empty_kernel_directory(tmp_path):
    """The exact state a provision interrupted by a locked file leaves behind."""
    d = _install(tmp_path / "i", ["llama.dll", "rocblas.dll"], {"rocblas/library": 0})
    problems = setup_llama.blas_kernel_problems(d)
    assert len(problems) == 1 and "empty" in problems[0]


def test_blas_flags_a_missing_kernel_directory(tmp_path):
    """The original defect: the library was copied and its data was dropped."""
    d = _install(tmp_path / "i", ["llama.dll", "rocblas.dll"])
    problems = setup_llama.blas_kernel_problems(d)
    assert len(problems) == 1 and "missing entirely" in problems[0]


@pytest.mark.parametrize("libs", [
    ["llama.dll", "ggml-vulkan.dll"],
    ["llama.dll", "ggml-cuda.dll", "cudart64_12.dll"],
    ["libllama.dylib", "libggml-metal.dylib"],
    ["llama.dll"],
])
def test_blas_never_flags_a_backend_that_has_no_rocblas(tmp_path, libs):
    """Backend-awareness without a platform test or a backend marker: an install
    that does not ship the vendor library cannot be missing its data."""
    d = _install(tmp_path / "i", libs)
    assert setup_llama.blas_kernel_problems(d) == []


def test_blas_matches_the_posix_library_naming(tmp_path):
    """librocblas.so.4 must match as readily as rocblas.dll, or the check is
    silently Windows-only on a project that ships Linux ROCm archives too."""
    d = _install(tmp_path / "i", ["libllama.so", "librocblas.so.4"])
    assert len(setup_llama.blas_kernel_problems(d)) == 1


def test_blas_does_not_require_hipblaslt_kernels(tmp_path):
    """MEASURED on the shipped b1307 gfx103X archive: libhipblaslt is installed
    with NO hipblaslt/ directory at all, and that install is healthy. Requiring
    one would fire on every normal AMD install, which is how a check gets
    ignored."""
    d = _install(tmp_path / "i",
                 ["llama.dll", "rocblas.dll", "libhipblaslt.dll"],
                 {"rocblas/library": 3})
    assert setup_llama.blas_kernel_problems(d) == []


def test_blas_reports_an_unreadable_install_rather_than_calling_it_clean(tmp_path, monkeypatch):
    d = _install(tmp_path / "i", ["llama.dll", "rocblas.dll"], {"rocblas/library": 1})

    def boom(*a, **kw):
        raise OSError("permission denied")

    monkeypatch.setattr(setup_llama, "_has_vendor_library", boom)
    problems = setup_llama.blas_kernel_problems(d)
    assert problems and "could not inspect" in problems[0]


def test_blas_ignores_a_path_that_is_not_a_directory(tmp_path):
    f = tmp_path / "not-a-dir"
    f.write_bytes(b"\x00")
    assert setup_llama.blas_kernel_problems(f) == []


# --------------------------------------------------------------------------- #
#  provisioning notice wording
# --------------------------------------------------------------------------- #

class _StoppedBeforeProvisioning(Exception):
    """Raised by the patched provisioning lock so a test observes the notice
    text `main()` actually printed without letting it reach the network or
    the disk - see _invoke_provision_notice."""


def _invoke_provision_notice(monkeypatch, tmp_path, capsys, *, have, have_build, want):
    """Drive the REAL `main()` "already provisioned" notice logic up to, but
    not through, the point where it would mutate the target or touch the
    network, and return what it printed.

    have/have_build fake the marker `_provisioned_backend`/`_provisioned_build`
    would read back; want is the requested --backend. The interactive confirm
    gate (`want == "auto" or have == want`) is always satisfied (a fake tty
    plus a patched click.confirm), since both cases it guards are needed to
    reach every branch of the notice below it."""
    monkeypatch.setattr(setup_llama, "_repo_runtime_lib", lambda: tmp_path)
    monkeypatch.setattr(setup_llama, "_provisioned_backend", lambda target: have)
    monkeypatch.setattr(setup_llama, "_provisioned_build", lambda target: have_build)
    monkeypatch.setattr(setup_llama.click, "confirm", lambda *a, **k: True)

    class _FakeTTYStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(setup_llama.sys, "stdin", _FakeTTYStdin())
    (tmp_path / setup_llama._lib_name()).write_bytes(b"\x00")

    @contextlib.contextmanager
    def _refuse_to_provision(target):
        raise _StoppedBeforeProvisioning()
        yield  # pragma: no cover - the raise above always fires first

    monkeypatch.setattr(setup_llama, "_provisioning_lock", _refuse_to_provision)

    with pytest.raises(_StoppedBeforeProvisioning):
        setup_llama.main.callback(
            from_dir=None, backend=want, url=None, sha256=None,
            force=False, tag=None, rollback=False, assume_yes=False,
        )
    return " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("have,have_build,want,expect,forbid", [
    ("amd-rocm", None,     "amd-rocm", "Re-downloading the amd-rocm build (", "with amd-rocm"),
    ("amd-rocm", None,     "auto",     "with the auto-detected backend",    "with auto."),
    ("vulkan",   None,     "cuda",     "Replacing vulkan build with cuda", None),
    (None,       None,     "cuda",     "Replacing unrecorded build with cuda", None),
])
def test_provision_notice_reads_sensibly(monkeypatch, tmp_path, capsys,
                                         have, have_build, want, expect, forbid):
    """`--force` on an already-provisioned box used to announce "Replacing
    amd-rocm build with amd-rocm" (it is a re-download) or "...with auto" (auto
    is not a backend, it is how one gets picked). Drives the real main()
    branch instead of a hand-written copy of it."""
    out = _invoke_provision_notice(monkeypatch, tmp_path, capsys,
                                   have=have, have_build=have_build, want=want)
    assert expect in out
    if forbid:
        assert forbid not in out


def test_redownload_names_the_build_it_is_fetching(monkeypatch, tmp_path, capsys):
    """"Re-downloading the amd-rocm build" alone reads as a no-op; the case this
    came from was a real build upgrade. Drives the real main() branch with a
    recorded build tag that differs from the pinned one, so it must announce
    the upgrade rather than a bare re-download."""
    old_tag = f"not-{setup_llama._ROCM_TAG}"
    out = _invoke_provision_notice(monkeypatch, tmp_path, capsys,
                                   have="amd-rocm", have_build=old_tag, want="amd-rocm")
    assert f"Upgrading the amd-rocm build: {old_tag} -> {setup_llama._ROCM_TAG}." in out
