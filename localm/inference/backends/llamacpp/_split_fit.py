# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-device fit for llama.cpp's implicit multi-GPU layer split.

With no ``tensor_split``, llama.cpp spreads a model's layers over every GPU in
proportion to each device's free memory, and puts the output layer on the
device that receives the LAST share. The context it creates then reserves the
logits buffer (``n_vocab * n_outputs`` floats) on that same device. This
module predicts that placement from the model's tensor sizes and each device's
free reading, charges every device what it would hold, and, when a device
cannot hold its charge, picks an explicit split that leaves it out.

Pure functions only: no probing, no native calls.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Bytes per logit: llama.cpp's logits tensor is f32.
_LOGIT_BYTES = 4


def _f32(x: float) -> float:
    """*x* rounded to an IEEE-754 single, the precision llama.cpp's split
    arithmetic runs in."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def layer_devices(shares: Sequence[float], n_layer_all: int,
                  n_gpu_layers: int) -> "tuple[list, Optional[int]]":
    """``(per-layer positions, output-layer position)`` for a layer split over
    devices weighted by *shares*, or ``None`` for a layer left on the CPU.

    Positions index *shares*. Mirrors llama.cpp's ``load_tensors`` (tag
    b11118): cumulative shares normalised in single precision, layer ``il``
    offloaded when ``il >= n_layer_all + 1 - n_gpu_layers``, and placed on the
    first device whose cumulative share is greater than
    ``(il - i_gpu_start) / act_gpu_layers``; the output layer is layer index
    ``n_layer_all``. A zero share gives a device no layers. Answers all-CPU
    when *shares* is empty or sums to zero."""
    n = len(shares)
    cum: List[float] = []
    total = _f32(0.0)
    for s in shares:
        total = _f32(total + _f32(float(s)))
        cum.append(total)
    if n == 0 or total <= 0:
        return [None] * n_layer_all, None
    cum = [_f32(c / total) for c in cum]
    i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)
    act = min(n_gpu_layers, n_layer_all + 1)

    def _dev(il: int) -> Optional[int]:
        if il < i_gpu_start or (il - i_gpu_start) >= act:
            return None
        v = _f32(_f32(float(il - i_gpu_start)) / float(act))
        for pos, c in enumerate(cum):
            if c > v:
                return pos
        return n - 1

    return [_dev(il) for il in range(n_layer_all)], _dev(n_layer_all)


@dataclass
class DeviceCharge:
    """What one device would hold under a split. Sizes in bytes; ``layers``
    counts the repeating layers placed on it."""
    index: int
    free: int
    share: float
    layers: int = 0
    weights: int = 0
    output: int = 0
    kv: int = 0
    logits: int = 0
    reserve: int = 0
    holds_output: bool = False

    @property
    def need(self) -> int:
        return self.weights + self.output + self.kv + self.logits + self.reserve

    @property
    def fits(self) -> bool:
        return self.share <= 0 or self.need <= self.free


@dataclass
class SplitFitPlan:
    """The outcome of :func:`plan_split`.

    ``tensor_split`` is ``None`` when llama.cpp's default split already fits
    every device, or when no split over two or more devices fits; otherwise it
    maps each device index to keep to its share. ``excluded`` lists the device
    indices left out. ``default`` is the charge of every device under the
    default split; ``chosen`` the charge under ``tensor_split`` (empty when
    ``tensor_split`` is ``None``)."""
    tensor_split: Optional[Dict[int, float]]
    excluded: List[int]
    default: List[DeviceCharge]
    chosen: List[DeviceCharge] = field(default_factory=list)

    @property
    def default_fits(self) -> bool:
        return all(c.fits for c in self.default)


def charge_devices(devices: Sequence[dict], shares: Sequence[float], *,
                   layer_bytes: Sequence[int], output_bytes: int,
                   layer_kv_bytes: Sequence[int], n_gpu_layers: int,
                   logits_bytes: int, reserve_bytes: int) -> List[DeviceCharge]:
    """Charge each device in *devices* (``{"index", "free"}``, in split order)
    for what a layer split weighted by *shares* places on it: its layers'
    weights and KV cache, the output layer's weights and the logits buffer on
    the device holding the output layer, and *reserve_bytes* per device with a
    non-zero share. *layer_kv_bytes* is each layer's KV cache, parallel to
    *layer_bytes*."""
    n_layer_all = len(layer_bytes)
    positions, out_pos = layer_devices(shares, n_layer_all, n_gpu_layers)
    charges = [DeviceCharge(index=int(d["index"]), free=int(d["free"]),
                            share=float(s))
               for d, s in zip(devices, shares)]
    for il, pos in enumerate(positions):
        if pos is None:
            continue
        charges[pos].layers += 1
        charges[pos].weights += int(layer_bytes[il])
        charges[pos].kv += int(layer_kv_bytes[il])
    if out_pos is not None:
        charges[out_pos].output = int(output_bytes)
        charges[out_pos].logits += int(logits_bytes)
        charges[out_pos].holds_output = True
    for c in charges:
        if c.share > 0:
            c.reserve = int(reserve_bytes)
    return charges


def logits_buffer_bytes(n_vocab: int, n_ctx: int, *, max_batch: int = 2048,
                        contexts: int = 1) -> int:
    """Bytes of the logits buffer llama.cpp reserves for one context whose
    ``n_batch`` and ``n_ubatch`` are ``min(n_ctx, max_batch)``: its worst-case
    graph outputs one row of ``n_vocab`` floats per token of the batch.
    *contexts* counts contexts sharing the output layer (a speculative draft
    context adds one)."""
    n_outputs = max(1, min(int(n_ctx), int(max_batch)))
    return int(n_vocab) * n_outputs * _LOGIT_BYTES * max(1, int(contexts))


def plan_split(devices: Sequence[dict], *, layer_bytes: Sequence[int],
               layer_kv_bytes: Sequence[int], output_bytes: int, n_gpu_layers: int,
               logits_bytes: int, reserve_bytes: int) -> SplitFitPlan:
    """Decide whether llama.cpp's default split fits *devices*
    (``[{"index", "free"}, ...]`` in the runtime's device order) and, when it
    does not, which devices to leave out.

    The default split weights devices by free memory. While some device with a
    non-zero share cannot hold its charge, the one with the least free memory
    among them gets a zero share and the rest are re-weighted by free memory.
    A plan keeps at least two devices; when none fits, ``tensor_split`` is
    ``None`` and only ``default`` reports the shortfall."""
    kw = dict(layer_bytes=layer_bytes, output_bytes=output_bytes,
              n_gpu_layers=n_gpu_layers, layer_kv_bytes=layer_kv_bytes,
              logits_bytes=logits_bytes, reserve_bytes=reserve_bytes)
    frees = [max(0, int(d["free"])) for d in devices]
    default = charge_devices(devices, frees, **kw)
    if all(c.fits for c in default):
        return SplitFitPlan(tensor_split=None, excluded=[], default=default)
    shares: List[float] = [float(f) for f in frees]
    charges = default
    while True:
        over = [pos for pos, c in enumerate(charges) if not c.fits]
        if not over:
            break
        victim = min(over, key=lambda pos: (frees[pos], pos))
        shares[victim] = 0.0
        if sum(1 for s in shares if s > 0) < 2:
            return SplitFitPlan(tensor_split=None, excluded=[], default=default)
        total = sum(shares)
        shares = [s / total for s in shares]
        charges = charge_devices(devices, shares, **kw)
    kept = {int(d["index"]): s for d, s in zip(devices, shares) if s > 0}
    excluded = [int(d["index"]) for d, s in zip(devices, shares) if s <= 0]
    return SplitFitPlan(tensor_split=kept, excluded=excluded, default=default,
                        chosen=charges)
