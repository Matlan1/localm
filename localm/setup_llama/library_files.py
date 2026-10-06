# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which files of an extracted build are installed into the runtime lib dir:
the llama/ggml libraries, the ROCm BLAS kernel data, and the license text.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import localm.setup_llama as _sl

def _is_wanted(f: Path) -> bool:
    """Whether to copy *f*: the loadable library, its ggml deps, and the runtime
    libraries - matched by platform-appropriate naming (incl. versioned .so.N).

    LIBRARIES ONLY, NEVER EXECUTABLES. localm loads the native runtime in-process
    through ctypes and never shells out to a bundled binary, so the upstream
    archives' ~49 command-line tools (llama-cli, llama-server, llama-bench,
    ggml-rpc-server, ...) were dead weight in every Windows install.

    Verified before removing them, rather than inferred from their names:
      * no subprocess call anywhere in localm reaches the runtime binary dir -
        the only executables it ever runs are nvidia-smi, sys.executable and
        uv/pip;
      * no documented workflow in docs/ or README tells a user to run one;
      * nothing in setup-llama or doctor invokes one (both isolate via a plain
        python subprocess).

    THE DECIDING EVIDENCE IS THE PLATFORM ASYMMETRY: the darwin and Linux
    branches below have ALWAYS matched libraries only (.dylib / .so), so those
    archives' extensionless `llama-cli` and friends were never copied and those
    installs have never carried a single bundled executable. Windows was the
    lone outlier. Dropping .exe makes it agree with the platforms that already
    demonstrate the product does not need them.

    Libraries are kept WHOLESALE - a .dll may be an OS-resolved link dependency
    of ggml-hip/llama rather than something localm opens by name
    (amd_comgr, rocblas, hipblaslt, rocsolver, origami, rocm_kpack all are), so
    proving one unused would need a link-graph walk. Unproven means keep: a
    retained stray file costs disk, a removed dependency costs a broken install
    on hardware nobody here can test.

    Incidentally removes ggml-rpc-server.exe, which carries a critical
    unauthenticated-RCE advisory in its own component (CVE-2026-34159, fixed in
    the build we ship). localm never ran it, so this is not a vulnerability fix -
    but an unnecessary network daemon has no business in the install directory of
    an offline-first app.
    """
    n = f.name.lower()
    if sys.platform == "win32":
        return n.endswith(".dll")
    if sys.platform == "darwin":
        return n.endswith(".dylib")
    return ".so" in n          # libfoo.so and libfoo.so.1


# rocBLAS and hipBLASLt (ROCm's vendor BLAS libraries) resolve their GPU-arch-
# specific GEMM kernels ("Tensile" library) at RUNTIME from a "<name>/library/"
# data directory sitting next to their DLL - the kernels are NOT linked into the
# DLL itself. That data is pure .dat/.hsaco/.co files, so _is_wanted() (by
# design: it must not copy the source tree's .py/.md/etc) never matches them,
# and _copy_binaries' flat `target / f.name` copy would lose their required
# subdirectory layout even if it did. The result: every ROCm/HIP provision
# (amd-rocm auto-detect, or --from/--url pointed at the identical archive)
# silently shipped rocblas.dll/hipblaslt.dll with NO kernel data at all. Nothing
# failed at provision time - ggml's own hand-written HIP kernels cover ordinary
# chat decode - so this went undetected until a workload that dispatches a GEMM
# through Tensile (the embedder's non-causal batch encode) hit it: rocBLAS
# fails to init its Tensile host and hard-crashes the native process outright
# (uncatchable from Python - the whole reason the embedder load/embed calls run
# in an isolated child, see inference/embedder.py). The lemonade-sdk gfx103X
# archives ship a complete rocblas/library/ including gfx1030 kernels; the data
# was always in the archive, just dropped on the way in.
#
# The current gfx103X archive carries no hipblaslt/library/ at all, and that is
# not a regression to chase: the hipblaslt data an earlier archive carried held
# gfx1100 kernels only, so this target never had usable data. Both names stay
# listed because the same code provisions the gfx110X/gfx120X archives too, and
# a missing directory is already a no-op here.
_BLAS_LIBRARY_DIRS = ("rocblas", "hipblaslt")


# Of those, the ones whose kernel data is genuinely REQUIRED by an install that
# ships the matching vendor library. Only rocblas:
#   * rocblas WITHOUT its Tensile data hard-crashes the native process on the
#     first GEMM dispatched through it (the embedder's batch encode) - the
#     uncatchable crash documented above.
#   * hipblaslt is present as a library and has NO kernel directory at all on
#     the shipped gfx103X archive, and that install is healthy. The hipblaslt
#     data an earlier archive carried held gfx1100 kernels only, so this target
#     never had usable data to lose.
# So requiring a hipblaslt directory would fire on every healthy gfx103X install:
# a check that cries wolf on the normal case is worse than no check, because it
# trains people to ignore it. If a gfx110X/gfx120X user ever reports a hipblaslt
# Tensile crash, add it here with that evidence.
_BLAS_DIRS_REQUIRING_KERNELS = ("rocblas",)


def _has_vendor_library(target: Path, name: str) -> bool:
    """True when *target* holds the shared library for BLAS vendor *name*.

    Matches both naming conventions because the archives use both, on the same
    platform: the b1307 Windows build ships `rocblas.dll` AND `libhipblaslt.dll`.
    Covers `.so` version suffixes (librocblas.so.4) the same way _is_wanted does."""
    for f in target.iterdir():
        if not f.is_file():
            continue
        stem = f.name.lower()
        if stem.startswith("lib"):
            stem = stem[3:]
        if stem.startswith(name + "."):
            return True
    return False


def blas_kernel_problems(target: Path) -> "list[str]":
    """Human-readable problems with the BLAS kernel data in a provisioned runtime.

    Empty list means nothing to report, INCLUDING for every non-ROCm backend: the
    check is keyed on whether the install actually ships the vendor library, so a
    vulkan / cuda / cpu / metal install has nothing to match and is silently fine.
    No platform test and no backend marker is consulted - the marker is written
    last during a provision, so a half-finished install can be missing it exactly
    when this check matters most.

    Scope: this catches "the library is installed but its runtime
    kernel data is not", which is the SILENT failure - provisioning succeeds, chat
    works, and the crash arrives later on the first Tensile GEMM. It does not try
    to catch a missing library, because that one already fails loudly at load."""
    problems: "list[str]" = []
    try:
        if not target.is_dir():
            return problems
        for name in _BLAS_DIRS_REQUIRING_KERNELS:
            if not _sl._has_vendor_library(target, name):
                continue
            d = target / name
            if not d.is_dir():
                problems.append(f"{name} is installed but its {name}/ kernel "
                                f"directory is missing entirely")
                continue
            n = sum(1 for p in d.rglob("*") if p.is_file())
            if n == 0:
                problems.append(f"{name} is installed but its {name}/ kernel "
                                f"directory is empty")
    except OSError as e:
        # Cannot read the install: say so rather than returning "no problems",
        # which is what an unreadable directory would otherwise look like.
        problems.append(f"could not inspect BLAS kernel data: {e}")
    return problems


def _copy_blas_library_dirs(src_dir: Path, target: Path) -> int:
    """Copy any of ``_BLAS_LIBRARY_DIRS`` found under *src_dir* into *target*,
    preserving their internal directory structure (unlike _copy_binaries' flat
    DLL copy - rocBLAS/hipBLASLt resolve this data by RELATIVE PATH, not by
    file name, so flattening it would be as useless as dropping it). Searches
    one level of nesting too, in case an archive wraps its contents in a single
    top-level folder. Returns the number of files copied."""
    n = 0
    for name in _BLAS_LIBRARY_DIRS:
        src = src_dir / name
        if not src.is_dir():
            nested = list(src_dir.glob(f"*/{name}"))
            src = nested[0] if nested else None
        if not src or not src.is_dir():
            continue
        dest_root = target / name
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            out = dest_root / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            n += 1
    return n


# Fallback notice written next to the bundled binaries when the upstream archive
# ships no LICENSE file, so a release never redistributes the MIT-licensed
# llama.cpp/ggml binaries without their license text (MIT requires it).
_LLAMA_CPP_MIT_NOTICE = """MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


_LICENSE_NAME_PREFIXES = ("license", "licence", "copying", "notice")


def _copy_license_files(src_dir: Path, target: Path) -> int:
    """Copy upstream license/notice files from *src_dir* into *target* so the MIT
    text travels with the binaries. Falls back to a bundled llama.cpp/ggml MIT
    notice when the archive ships none. Returns the number of files written."""
    found = [f for f in sorted(src_dir.rglob("*"))
             if _safe_is_file(f)
             and any(f.name.lower().startswith(k) for k in _LICENSE_NAME_PREFIXES)]
    if found:
        written = 0
        for i, f in enumerate(found):
            dest = target / ("LICENSE.llama-cpp" if i == 0
                             else f"LICENSE.llama-cpp.{i}")
            try:
                shutil.copy2(f, dest)
                written += 1
            except OSError:
                pass
        if written:
            return written
    (target / "LICENSE.llama-cpp").write_text(_LLAMA_CPP_MIT_NOTICE, encoding="utf-8")
    return 1


def _safe_is_file(f: Path) -> bool:
    try:
        return f.is_file()
    except OSError:
        return False


def _copy_binaries(src_dir: Path, target: Path) -> int:
    """Copy the llama/ggml/runtime libraries from *src_dir* (recursively) into
    *target*. Returns the number of files copied."""
    n = 0
    for f in src_dir.rglob("*"):
        if f.is_file() and _is_wanted(f):
            shutil.copy2(f, target / f.name)
            n += 1
    # rocBLAS/hipBLASLt Tensile kernel data (see _BLAS_LIBRARY_DIRS) - a no-op
    # on every non-ROCm backend, since src_dir then has no rocblas/hipblaslt dir.
    n += _copy_blas_library_dirs(src_dir, target)
    # MIT requires the license to accompany the binaries; capture it (or a
    # bundled fallback) alongside them whenever we actually placed binaries.
    if n:
        _copy_license_files(src_dir, target)
    return n
