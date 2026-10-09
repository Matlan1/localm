# SPDX-License-Identifier: AGPL-3.0-or-later
"""The SIMD CPU backend installed over the amd-rocm build.

The amd-rocm (lemonade-sdk) build ships a ggml-cpu.dll compiled with no x86
SIMD, so every matmul that runs on the CPU (Mixture-of-Experts weights kept in
system RAM, layers left on the CPU, CPU-only loads) runs scalar code. Upstream's
Windows CPU archive from the same llama.cpp commit (``_ROCM_CPU_TAG``) ships one
ggml-cpu variant per x86 microarchitecture. The variant with the highest
``ggml_backend_score()`` on this CPU (each probed in its own subprocess, see
:func:`localm.cpu_backend_select._probe_score`) replaces ggml-cpu.dll, together
with the OpenMP runtime the variants link against.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from localm.debuglog import logger
from localm.setup_llama._common import console
from localm.setup_llama.download import _extract_archive, _validate_archive
from localm.setup_llama.pins import (_PINNED_FALLBACK_SHA256, _ROCM_BUILD,
                                     _ROCM_CPU_ASSET, _ROCM_CPU_TAG, _ROCM_TAG,
                                     _UPSTREAM_REPO)
import localm.setup_llama as _sl

# Written into the runtime lib dir once the overlay is in place:
# {"tag": <upstream tag>, "variant": <variant file name>}. Cleared with every
# other provisioned file on a re-provision.
CPU_OVERLAY_MARKER = ".localm-cpu-overlay"

_CPU_DLL = "ggml-cpu.dll"
_OPENMP_DLL = "libomp140.x86_64.dll"
_BACKUP_SUFFIX = ".amd-rocm"


def rocm_cpu_overlay_url() -> str:
    """Download URL of the pinned upstream CPU archive (``_ROCM_CPU_ASSET``)."""
    return (f"https://github.com/{_UPSTREAM_REPO}/releases/download/"
            f"{_ROCM_CPU_TAG}/{_ROCM_CPU_ASSET}")


def installed_cpu_overlay(target: Path) -> Optional[dict]:
    """The overlay marker in *target* (``{"tag", "variant"}``), or None when
    the amd-rocm build there still runs its own CPU backend. Never raises."""
    try:
        data = json.loads((target / CPU_OVERLAY_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("variant"):
        return None
    return data


def rocm_build(target: Path) -> str:
    """The amd-rocm build identity to record for *target*: ``_ROCM_BUILD``
    when the SIMD CPU backend from ``_ROCM_CPU_TAG`` is installed there, else
    ``_ROCM_TAG``."""
    overlay = installed_cpu_overlay(target)
    if overlay is not None and overlay.get("tag") == _ROCM_CPU_TAG:
        return _ROCM_BUILD
    return _ROCM_TAG


def _best_variant(variants: list[Path]) -> tuple[Optional[Path], dict]:
    """The variant with the highest positive ``ggml_backend_score()`` on this
    CPU, and every variant's score (None when its probe failed)."""
    from localm.cpu_backend_select import _probe_score
    scores = {v.name: _probe_score(v, v.parent) for v in variants}
    usable = [(v, scores[v.name]) for v in variants
              if scores[v.name] is not None and scores[v.name] > 0]
    if not usable:
        return None, scores
    return max(usable, key=lambda pair: pair[1])[0], scores


def install_rocm_simd_cpu(target: Path) -> Optional[str]:
    """Replace the amd-rocm build's ggml-cpu.dll in *target* with the best SIMD
    variant from ``_ROCM_CPU_ASSET`` and prove the runtime still loads
    (``_native_loads_ok``). Returns the installed variant's file name, or None
    when the overlay was not installed: the download, checksum, scoring or load
    check failed, or *target* has no ggml-cpu.dll. On None, *target* holds the
    amd-rocm build's own CPU backend and a yellow warning names why.

    Never raises."""
    original = target / _CPU_DLL
    backup = target / (_CPU_DLL + _BACKUP_SUFFIX)
    omp = target / _OPENMP_DLL
    try:
        if not original.is_file():
            _warn("this amd-rocm build has no ggml-cpu.dll to replace")
            return None
        with tempfile.TemporaryDirectory() as tmp:
            arc = Path(tmp) / _ROCM_CPU_ASSET
            dl = _sl._download(rocm_cpu_overlay_url(), arc)
            _validate_archive(arc, expected_sha256=_PINNED_FALLBACK_SHA256[_ROCM_CPU_ASSET],
                              dl=dl)
            ex = Path(tmp) / "x"
            _extract_archive(arc, ex)
            variants = sorted(p for p in ex.rglob("ggml-cpu-*.dll") if p.is_file())
            if not variants:
                _warn(f"{_ROCM_CPU_ASSET} contains no ggml-cpu variants")
                return None
            winner, scores = _best_variant(variants)
            logger.info("amd-rocm CPU backend: variant scores %s", scores)
            if winner is None:
                _warn("no CPU backend variant reports support for this CPU")
                return None
            os.replace(original, backup)
            shutil.copy2(winner, original)
            openmp = winner.parent / _OPENMP_DLL
            if openmp.is_file():
                shutil.copy2(openmp, omp)
        loaded, detail = _sl._native_loads_ok()
        if not loaded:
            _restore(original, backup, omp)
            _warn(f"the runtime did not load with it ({detail})")
            return None
        (target / CPU_OVERLAY_MARKER).write_text(
            json.dumps({"tag": _ROCM_CPU_TAG, "variant": winner.name}), encoding="utf-8")
        backup.unlink()
    except Exception as e:  # noqa: BLE001 - reported below, provisioning continues
        try:
            _restore(original, backup, omp)
        except OSError as restore_error:
            logger.warning("amd-rocm CPU backend: could not restore %s: %s",
                           original, restore_error)
        _warn(f"{type(e).__name__}: {e}")
        return None
    console.print(f"[dim]CPU backend:[/dim] {winner.stem} from llama.cpp "
                  f"{_ROCM_CPU_TAG} (SIMD), replacing the amd-rocm build's own")
    return winner.name


def _restore(original: Path, backup: Path, omp: Path) -> None:
    """Put the amd-rocm build's own ggml-cpu.dll back from *backup* (when one
    was made) and remove the copied OpenMP runtime and the overlay marker."""
    if not backup.is_file():
        return
    os.replace(backup, original)
    if omp.is_file():
        omp.unlink()
    marker = original.parent / CPU_OVERLAY_MARKER
    if marker.is_file():
        marker.unlink()


def _warn(why: str) -> None:
    logger.warning("amd-rocm CPU backend overlay not installed: %s", why)
    console.print(
        f"[yellow]Warning: the SIMD CPU backend for the amd-rocm build was not "
        f"installed ({why}). Work that runs on the CPU (MoE experts in system "
        f"RAM, layers left on the CPU) will be several times slower. Retry with "
        f"localm setup-llama --backend amd-rocm --force.[/yellow]")
