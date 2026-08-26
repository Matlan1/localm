# SPDX-License-Identifier: AGPL-3.0-or-later
"""Provisioning must refuse BEFORE deleting when the runtime is in use, and
`doctor` must fail an install whose BLAS kernel data is missing.

A `setup-llama --force` on a machine with the runtime loaded can otherwise leave
a partially cleared install that every other check still reports as healthy.
"""

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
    """Asserted from OUTSIDE via the return value, not by raising from a patched
    unlink: `_clear_target` catches OSError by design, so an exception-based
    assertion would be swallowed by the code under test."""
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
    """The pre-flight leaves an INTACT install: reporting after the fact would
    still leave a half-cleared dir."""
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
    WHOLE UPDATE BACK on a non-zero exit (updater.py `_rollback_or_raise`). A
    runtime that could not be re-provisioned beside freshly swapped code is the
    ABI mismatch the isolation exists to avoid, so the update is undone."""
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
    """The library is present and its kernel data directory is not."""
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
    """A healthy AMD install ships libhipblaslt with NO hipblaslt/ directory at
    all, so the check must not require one."""
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

@pytest.mark.parametrize("have,want,expect,forbid", [
    ("amd-rocm", "amd-rocm", "Re-downloading the amd-rocm build (", "with amd-rocm"),
    ("amd-rocm", "auto",     "with the auto-detected backend",    "with auto."),
    ("vulkan",   "cuda",     "Replacing vulkan build with cuda", None),
    (None,       "cuda",     "Replacing unrecorded build with cuda", None),
])
def test_provision_notice_reads_sensibly(have, want, expect, forbid, capsys):
    """`--force` on an already-provisioned box must not announce "Replacing
    amd-rocm build with amd-rocm" (it is a re-download) or "...with auto" (auto
    is not a backend, it is how one gets picked)."""
    from localm.setup_llama import console
    if not have:
        console.print(f"[yellow]Replacing unrecorded build with {want}.[/yellow]")
    elif want == "auto":
        console.print(f"[yellow]Replacing {have} build with the "
                      f"auto-detected backend.[/yellow]")
    elif have == want:
        tag = f" ({setup_llama._ROCM_TAG})" if want == "amd-rocm" else ""
        console.print(f"[yellow]Re-downloading the {have} build{tag}.[/yellow]")
    else:
        console.print(f"[yellow]Replacing {have} build with {want}.[/yellow]")
    out = " ".join(capsys.readouterr().out.split())
    assert expect in out
    if forbid:
        assert forbid not in out


def test_redownload_names_the_build_it_is_fetching(capsys):
    """"Re-downloading the amd-rocm build" alone reads as a no-op, so the notice
    names the target build. The OLD version cannot be named (the marker records
    the backend only), and only amd-rocm has a pinned tag constant; the upstream
    backends resolve theirs with a network call, which a print statement does not
    get to make."""
    from localm.setup_llama import console, _ROCM_TAG
    console.print(f"[yellow]Re-downloading the amd-rocm build ({_ROCM_TAG}).[/yellow]")
    out = " ".join(capsys.readouterr().out.split())
    assert _ROCM_TAG in out and _ROCM_TAG.startswith("b")
