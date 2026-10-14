# SPDX-License-Identifier: AGPL-3.0-or-later
"""Matrix-multiply kernel tuning for HF decoding on ROCm.

One-token decoding is dominated by matrix-vector products (a 1 x K input times
each linear layer's weight). On ROCm the BLAS default for some of those shapes
and dtypes runs far below the card's memory bandwidth. ``tune_decode_gemms``
turns on PyTorch's TunableOp, times the available rocBLAS solutions once for
every distinct linear-layer shape of a loaded model at batch size 1, and keeps
TunableOp on with tuning off so later decoding uses the fastest solution found.
Results go to a CSV under the cache directory and are read back on the next
load, so a shape is tuned once per card and software version (TunableOp rejects
a file written by another torch, ROCm or rocBLAS version).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from localm.debuglog import logger

# Environment variable TunableOp reads to include hipBLASLt solutions.
HIPBLASLT_ENV = "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"
# Milliseconds TunableOp may spend timing the candidates of one GEMM shape.
MAX_TUNING_MS_PER_GEMM = 30
# Name of the results directory under the cache directory.
RESULTS_DIR_NAME = "rocm-gemm-tuning"


def decode_gemm_weights(torch: Any, model: Any) -> list[Any]:
    """One weight per distinct ``(in_features, out_features, dtype, device)``
    among the ``torch.nn.Linear`` layers of *model* whose weight is a
    floating-point tensor on a GPU, in first-seen order. Quantized,
    CPU-resident and meta weights are skipped."""
    seen: list[tuple[int, int, Any, Any]] = []
    weights: list[Any] = []
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2 or not weight.is_floating_point():
            continue
        if weight.device.type != "cuda":
            continue
        out_features, in_features = weight.shape
        key = (int(in_features), int(out_features), weight.dtype, weight.device)
        if key not in seen:
            seen.append(key)
            weights.append(weight)
    return weights


def tune_decode_gemms(torch: Any, model: Any, results_dir: Path) -> Optional[str]:
    """On a ROCm build of torch, enable TunableOp with rocBLAS solutions (and
    hipBLASLt only when ``PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED`` is already set
    to enable it), read earlier results from *results_dir*, run one batch-1
    linear product per weight from ``decode_gemm_weights`` with tuning on, then
    turn tuning off and leave TunableOp enabled.

    Returns a one-line summary, or None when nothing was done: not a ROCm
    build, no GPU-resident linear layer, or a failure (logged at debug level,
    with TunableOp turned off again). Never raises. Must run in the process
    that decodes, before its first GEMM if hipBLASLt is to stay excluded."""
    if not getattr(getattr(torch, "version", None), "hip", None):
        return None
    tunable = getattr(getattr(torch, "cuda", None), "tunable", None)
    if tunable is None:
        return None
    try:
        weights = decode_gemm_weights(torch, model)
    except Exception as exc:
        logger.debug("gemm tuning: could not list linear layers (%s)", type(exc).__name__)
        return None
    if not weights:
        return None
    os.environ.setdefault(HIPBLASLT_ENV, "0")
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        tunable.set_filename(str(results_dir / "tunableop_results.csv"),
                             insert_device_ordinal=True)
        tunable.set_max_tuning_duration(MAX_TUNING_MS_PER_GEMM)
        tunable.enable(True)
        tunable.tuning_enable(True)
        started = time.monotonic()
        with torch.no_grad():
            for weight in weights:
                x = torch.zeros((1, weight.shape[1]), dtype=weight.dtype, device=weight.device)
                torch.nn.functional.linear(x, weight)
                del x
        torch.cuda.synchronize()
        tunable.tuning_enable(False)
        elapsed = time.monotonic() - started
    except Exception as exc:
        logger.debug("gemm tuning failed (%s)", type(exc).__name__)
        for switch in (tunable.tuning_enable, tunable.enable):
            try:
                switch(False)
            except Exception as off_exc:
                logger.debug("gemm tuning: %s(False) failed (%s)",
                             getattr(switch, "__name__", "switch"), type(off_exc).__name__)
        return None
    summary = f"tuned {len(weights)} GEMM shapes in {elapsed:.1f}s"
    logger.debug("gemm tuning: %s (%s=%s)", summary, HIPBLASLT_ENV, os.environ.get(HIPBLASLT_ENV))
    return summary
