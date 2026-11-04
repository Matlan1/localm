# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which Kimi Delta Attention kernels transformers' own Kimi-Linear classes use.

Those classes use the kernels of an installed ``fla`` (fla-core) and otherwise their own PyTorch
ones, deciding when their module is imported. ``fla`` imports fine where its Triton kernels cannot
run (a CPU, an AMD GPU), so :func:`choose_kernels` runs them first on the load's device and hides the
package from that import when they fail.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys

logger = logging.getLogger("localm")

_PYTORCH = "linear attention kernels: PyTorch, transformers' own"
_FALLBACK_ADVICE = "is falling back to its reference PyTorch implementation"


class _DropFallbackAdvice(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _FALLBACK_ADVICE not in record.getMessage()


def _hide_fla() -> None:
    for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
        del sys.modules[name]
    sys.modules["fla"] = None  # type: ignore[assignment]


def _run_kernels(torch, device: str) -> None:
    """Call the installed kernels on *device* the way transformers does; raises when they
    cannot run or give a result of the wrong shape."""
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    length, heads, dim = 70, 2, 8
    q, k, v = (torch.randn(1, length, heads, dim, device=device) for _ in range(3))
    g = -torch.rand(1, length, heads, dim, device=device)
    beta = torch.rand(1, length, heads, device=device)
    out, state = chunk_kda(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True,
                           use_qk_l2norm_in_kernel=True)
    step, _ = fused_recurrent_kda(q[:, :1], k[:, :1], v[:, :1], g=g[:, :1], beta=beta[:, :1], initial_state=state,
                                  output_final_state=True, use_qk_l2norm_in_kernel=True)
    if out.shape != v.shape or step.shape != v[:, :1].shape:
        raise ValueError(f"unexpected shapes {tuple(out.shape)}, {tuple(step.shape)}")


def choose_kernels(torch, device: str) -> str:
    """Decide the kernels for a load on *device* before transformers' classes are imported and
    return one line for the load output naming them. Never raises. The advice transformers logs
    when it falls back to its own kernels is dropped; the line says the same, and the kernels agree
    with fla's. See test_the_pytorch_kernels_agree_with_fla_at_the_models_sizes."""
    logging.getLogger("transformers.integrations.hub_kernels").addFilter(_DropFallbackAdvice())
    try:
        if importlib.util.find_spec("fla") is None:
            return _PYTORCH
    except (ImportError, ValueError):
        return _PYTORCH
    try:
        _run_kernels(torch, device)
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        logger.warning("hf load: the installed fla-core cannot run on %s (%s); using PyTorch kernels", device, reason)
        _hide_fla()
        return f"{_PYTORCH}; the installed fla-core cannot run on {device} ({reason})"
    version = getattr(sys.modules.get("fla"), "__version__", "?")
    return f"linear attention kernels: fla-core {version} (Triton)"
