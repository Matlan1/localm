# SPDX-License-Identifier: AGPL-3.0-or-later
"""PyTorch implementations of the parts of the ``fla`` (flash-linear-attention)
package that Kimi-Linear's bundled code imports.

The real package needs Triton kernels, which have no build for every platform
localm runs on (no Triton for AMD GPUs on Windows, and ``fla`` imports
``torch.distributed`` names that a Windows PyTorch lacks). This module provides
the same names with the same call signatures and results, computed with plain
PyTorch operations on whatever device the tensors are on, so such a model loads
and generates there.

:func:`provide_for` registers the modules ``fla``, ``fla.modules``,
``fla.ops``, ``fla.ops.kda``, ``fla.ops.kda.gate``, ``fla.ops.utils``,
``fla.ops.utils.index`` and ``fla.utils`` in ``sys.modules`` when a model
folder's bundled code imports ``fla`` and the real package cannot be imported.
A working ``fla`` install is always preferred.

Covered: the forward (inference) computation of Kimi Delta Attention in chunked
form (:func:`chunk_kda`, any length) and recurrent form (:func:`fused_recurrent_kda`),
with ``initial_state``, ``output_final_state``, ``use_qk_l2norm_in_kernel`` and
variable-length batches (``cu_seqlens``); the KDA gate; the gated RMS norm; and
the short causal convolution with a carried state. Not covered: backward passes
and training.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("localm")

CHUNK_SIZE = 64
_L2_EPS = 1e-6
STAND_IN_MARK = "__localm_stand_in__"


def prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Number of non-padding tokens per row of the ``[B, T]`` 0/1 *mask*, as int32."""
    return mask.sum(dim=-1, dtype=torch.int32)


def prepare_cu_seqlens_from_mask(mask: torch.Tensor,
                                 out_dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Cumulative sequence lengths ``[B + 1]`` of the ``[B, T]`` 0/1 *mask*."""
    return F.pad(prepare_lens_from_mask(mask).cumsum(0, dtype=out_dtype), (1, 0))


def tensor_cache(fn):
    """Decorator ``fla`` uses to memoise a function of tensors; here the function
    is returned unchanged."""
    return fn


def fused_kda_gate(g: torch.Tensor, A: torch.Tensor, head_k_dim: int,
                   g_bias: Optional[torch.Tensor] = None, beta: float = 1.0,
                   threshold: float = 20.0) -> torch.Tensor:
    """KDA decay gate: ``-exp(A) * softplus(g + g_bias)`` per head.

    *g* has shape ``[..., num_heads * head_k_dim]``, *A* holds ``num_heads``
    values (any shape), *g_bias* has shape ``[num_heads * head_k_dim]``. Returns a
    float32 tensor of shape ``[..., num_heads, head_k_dim]``."""
    A = A.reshape(-1)
    if g_bias is not None:
        g = g + g_bias
    g = g.reshape(*g.shape[:-1], -1, head_k_dim)
    return -A.float().exp().unsqueeze(-1) * F.softplus(g.float(), beta, threshold)


class FusedRMSNormGated(nn.Module):
    """RMS norm with a learned scale and an output gate: ``rmsnorm(x) * w * act(g)``.

    *activation* is ``swish``, ``silu`` or ``sigmoid``. The state dict holds one
    ``weight`` of shape ``[hidden_size]``."""

    def __init__(self, hidden_size: int, elementwise_affine: bool = True, eps: float = 1e-5,
                 activation: str = "swish", device=None, dtype=None):
        super().__init__()
        if activation not in ("swish", "silu", "sigmoid"):
            raise ValueError(f"Unsupported activation: {activation}")
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        gf = g.float()
        gate = torch.sigmoid(gf) if self.activation == "sigmoid" else gf * torch.sigmoid(gf)
        return (y * gate).to(x.dtype)


class ShortConvolution(nn.Conv1d):
    """Depthwise causal 1-D convolution over the time axis of ``[B, T, D]`` input,
    optionally followed by ``silu``. The state dict holds ``weight`` of shape
    ``[D, 1, kernel_size]``.

    ``forward`` returns ``(y, final_state)``. *cache* and *final_state* have shape
    ``[N, D, kernel_size]`` and hold the last ``kernel_size`` inputs of each
    sequence; *cu_seqlens* ``[N + 1]`` makes ``[1, T, D]`` input a batch of ``N``
    concatenated sequences, each starting from its own cache."""

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False,
                 activation: Optional[str] = "silu", backend: Optional[str] = "triton",
                 device=None, dtype=None, **kwargs):
        super().__init__(in_channels=hidden_size, out_channels=hidden_size,
                         kernel_size=kernel_size, groups=hidden_size, bias=bias,
                         padding=kernel_size - 1, device=device, dtype=dtype)
        if activation is not None and activation not in ("silu", "swish"):
            raise ValueError(f"Activation `{activation}` not supported.")
        self.hidden_size = hidden_size
        self.activation = activation
        self.backend = backend

    def _convolve(self, x: torch.Tensor, cache: Optional[torch.Tensor]):
        width = self.kernel_size[0]
        xt = x.transpose(1, 2)
        if cache is None:
            xin = F.pad(xt, (width - 1, 0))
        else:
            xin = torch.cat([cache[..., cache.shape[-1] - (width - 1):].to(xt.dtype), xt], dim=-1) \
                if width > 1 else xt
        y = F.conv1d(xin, self.weight, self.bias, groups=self.hidden_size)
        if self.activation is not None:
            y = F.silu(y)
        final = xin[..., -width:]
        if final.shape[-1] < width:
            final = F.pad(final, (width - final.shape[-1], 0))
        return y.transpose(1, 2), final

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None,  # pyright: ignore[reportIncompatibleMethodOverride]
                mask: Optional[torch.Tensor] = None, cache: Optional[torch.Tensor] = None,
                output_final_state: bool = False,
                cu_seqlens: Optional[torch.Tensor] = None, **kwargs):
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        if cu_seqlens is None:
            y, final = self._convolve(x, cache)
        else:
            bounds = cu_seqlens.tolist()
            outs, finals = [], []
            for i in range(len(bounds) - 1):
                start, end = bounds[i], bounds[i + 1]
                state = None if cache is None else cache[i:i + 1]
                if end > start:
                    part, final_i = self._convolve(x[:, start:end], state)
                else:
                    part = x[:, start:end]
                    final_i = state if state is not None else x.new_zeros(
                        1, self.hidden_size, self.kernel_size[0])
                outs.append(part)
                finals.append(final_i)
            y = torch.cat(outs, dim=1)
            final = torch.cat(finals, dim=0)
        if residual is not None:
            y = y + residual
        return y, (final if output_final_state else None)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + _L2_EPS)


def _recurrent(q, k, v, g, beta, state):
    out = torch.empty_like(v)
    for t in range(q.shape[1]):
        state = state * g[:, t].exp().unsqueeze(-1)
        k_t = k[:, t]
        error = v[:, t] - (k_t.unsqueeze(-1) * state).sum(-2)
        state = state + (beta[:, t].unsqueeze(-1) * k_t).unsqueeze(-1) * error.unsqueeze(-2)
        out[:, t] = (q[:, t].unsqueeze(-1) * state).sum(-2)
    return out, state


def _unit_lower_solve(unit_lower: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve_triangular(unit_lower, rhs, upper=False, unitriangular=True)
    except (RuntimeError, NotImplementedError):
        n = unit_lower.shape[-1]
        x = rhs.clone()
        for i in range(1, n):
            x[..., i, :] = x[..., i, :] - (unit_lower[..., i, :i, None] * x[..., :i, :]).sum(-2)
        return x


def _chunked(q, k, v, g, beta, state, chunk_size: int = CHUNK_SIZE):
    batch, length, heads, dim_k = q.shape
    pad = (-length) % chunk_size
    if pad:
        q, k, v, g = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v, g))
        beta = F.pad(beta, (0, 0, 0, pad))
    size = chunk_size
    device = q.device
    eye = torch.eye(size, device=device, dtype=q.dtype)
    lower = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    strict = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=-1)
    outs = []
    for start in range(0, length + pad, size):
        sl = slice(start, start + size)
        qc, kc, vc = (t[:, sl].transpose(1, 2) for t in (q, k, v))
        bc = beta[:, sl].transpose(1, 2)
        gc = g[:, sl].transpose(1, 2).cumsum(-2)
        diff = (gc.unsqueeze(3) - gc.unsqueeze(2)).masked_fill(~lower[..., None], float("-inf"))
        decay = diff.exp()
        k_pair = (decay * kc.unsqueeze(3) * kc.unsqueeze(2)).sum(-1)
        low = k_pair.masked_fill(~strict, 0.0) * bc.unsqueeze(-1)
        a = _unit_lower_solve(eye + low, torch.diag_embed(bc))
        q_pair = (decay * qc.unsqueeze(3) * kc.unsqueeze(2)).sum(-1)
        w = a @ (gc.exp() * kc)
        u = a @ vc
        v_new = u - w @ state
        out_c = (qc * gc.exp()) @ state + q_pair @ v_new
        g_last = gc[:, :, -1]
        state = state * g_last.exp().unsqueeze(-1) + \
            ((g_last.unsqueeze(2) - gc).exp() * kc).transpose(-1, -2) @ v_new
        outs.append(out_c.transpose(1, 2))
    return torch.cat(outs, dim=1)[:, :length], state


def _kda(core, q, k, v, g, beta, scale, initial_state, output_final_state,
         use_qk_l2norm_in_kernel, cu_seqlens):
    dtype = v.dtype
    heads_qk, heads_v, dim_k, dim_v = q.shape[2], v.shape[2], q.shape[3], v.shape[3]
    if scale is None:
        scale = dim_k ** -0.5
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    groups = heads_v // heads_qk
    q = q.repeat_interleave(groups, dim=2) * scale
    k = k.repeat_interleave(groups, dim=2)

    def start_state(rows: int, first: int):
        if initial_state is None:
            return q.new_zeros(rows, heads_v, dim_k, dim_v)
        return initial_state[first:first + rows].float().clone()

    if cu_seqlens is None:
        out, state = core(q, k, v, g, beta, start_state(q.shape[0], 0))
        return out.to(dtype), (state if output_final_state else None)
    bounds = cu_seqlens.tolist()
    outs, states = [], []
    for i in range(len(bounds) - 1):
        sl = slice(bounds[i], bounds[i + 1])
        if bounds[i + 1] > bounds[i]:
            out_i, state_i = core(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
                                  start_state(1, i))
        else:
            out_i, state_i = v[:, sl], start_state(1, i)
        outs.append(out_i)
        states.append(state_i)
    return torch.cat(outs, dim=1).to(dtype), \
        (torch.cat(states, dim=0) if output_final_state else None)


def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
    """Kimi Delta Attention over whole sequences in chunks of :data:`CHUNK_SIZE`.

    *q*, *k* are ``[B, T, H, K]``, *v* is ``[B, T, HV, V]`` (``HV`` a multiple of
    ``H``), *g* the log-space decay ``[B, T, HV, K]``, *beta* ``[B, T, HV]``.
    *initial_state* is ``[N, HV, K, V]``. Returns ``(o, final_state)`` with ``o``
    shaped like *v* and ``final_state`` None unless *output_final_state*."""
    return _kda(_chunked, q, k, v, g, beta, scale, initial_state, output_final_state,
                use_qk_l2norm_in_kernel, cu_seqlens)


def fused_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None,
                        output_final_state=False, use_qk_l2norm_in_kernel=False,
                        cu_seqlens=None, **kwargs):
    """Kimi Delta Attention one token at a time; same arguments and results as
    :func:`chunk_kda`."""
    return _kda(_recurrent, q, k, v, g, beta, scale, initial_state, output_final_state,
                use_qk_l2norm_in_kernel, cu_seqlens)


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    setattr(mod, STAND_IN_MARK, True)
    return mod


def _install_modules() -> None:
    pkg = _module("fla")
    pkg.__path__ = []
    tree = {
        "fla.modules": _module("fla.modules", FusedRMSNormGated=FusedRMSNormGated,
                               ShortConvolution=ShortConvolution),
        "fla.ops": _module("fla.ops"),
        "fla.ops.kda": _module("fla.ops.kda", chunk_kda=chunk_kda,
                               fused_recurrent_kda=fused_recurrent_kda),
        "fla.ops.kda.gate": _module("fla.ops.kda.gate", fused_kda_gate=fused_kda_gate),
        "fla.ops.utils": _module("fla.ops.utils"),
        "fla.ops.utils.index": _module(
            "fla.ops.utils.index", prepare_lens_from_mask=prepare_lens_from_mask,
            prepare_cu_seqlens_from_mask=prepare_cu_seqlens_from_mask),
        "fla.utils": _module("fla.utils", tensor_cache=tensor_cache),
    }
    sys.modules["fla"] = pkg
    for name, mod in tree.items():
        mod.__path__ = []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        setattr(sys.modules[parent], child, mod)


def _real_fla_imports() -> bool:
    if sys.modules.get("fla") is not None and not getattr(sys.modules["fla"], STAND_IN_MARK, False):
        return True
    if importlib.util.find_spec("fla") is None:
        return False
    try:
        for name in ("fla.modules", "fla.ops.kda", "fla.ops.kda.gate", "fla.ops.utils.index",
                     "fla.utils"):
            importlib.import_module(name)
    except Exception as e:
        logger.debug("hf load: the installed fla package does not import (%s: %s)",
                     type(e).__name__, e)
        for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
            del sys.modules[name]
        return False
    return True


def provide_for(model_path: str) -> bool:
    """Register the PyTorch stand-ins for ``fla`` when the bundled code of the
    model folder *model_path* imports it and the installed ``fla`` is absent or
    does not import. Returns True when the stand-ins were registered."""
    from localm.model_manager.missing_packages import folder_third_party_imports
    try:
        needs_fla = "fla" in folder_third_party_imports(Path(model_path), guarded=True)
    except (OSError, ValueError, RecursionError):
        return False
    if not needs_fla or _real_fla_imports():
        return False
    _install_modules()
    logger.debug("hf load: fla is not usable here; using localm's PyTorch implementations")
    return True
