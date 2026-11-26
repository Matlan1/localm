# SPDX-License-Identifier: AGPL-3.0-or-later
"""Placement of one GGUF model over two or more GPUs.

llama.cpp gives the GPUs the last ``n_gpu_layers`` of the model's
``n_layer + 1`` units (the extra unit is the output layer) and hands each
device of ``llama_model_params.devices`` a contiguous run of them, in that
order, sized by ``tensor_split``. A layer's KV cache sits on the device that
holds the layer, and the output layer (with the logits of a batch) on the last
device that gets any unit. Every device in the list gets a backend, a compute
buffer and a driver context, whether or not it holds a layer.

:func:`plan_placement` chooses that list: which devices, in which order, how
many units each, how many layers stay on the CPU and how many blocks keep their
routed experts in system RAM. Its :class:`Placement` is what the worker passes
to llama.cpp: the devices in order and their unit counts as ``tensor_split``,
which llama.cpp turns back into exactly those counts.

Pure functions only: no probing, no native calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class PlanDevice:
    """One GPU a load may use.

    ``slot`` is the device's position in llama.cpp's own device list,
    ``index`` the GPU number the user sees (``localm gpus``, the GUI),
    ``device_id`` its PCI bus id when the runtime reports one. ``free`` and
    ``total`` are bytes. ``ratio`` is a pinned split ratio, None for an
    automatic share."""
    slot: int
    index: int
    name: str
    free: int
    total: int
    device_id: Optional[str] = None
    ratio: Optional[float] = None


@dataclass(frozen=True)
class ModelUnits:
    """What each unit of a model costs on the GPU that holds it.

    ``layer_bytes[il]`` is layer ``il``'s weights on the GPU with no routed
    experts in system RAM, ``expert_bytes[il]`` the part of them that is routed
    experts (moved to system RAM by ``n_cpu_moe``), ``layer_kv[il]`` its KV
    cache and recurrent state for one token window of ``n_ctx`` tokens.
    ``kv_per_token[il]`` is the KV bytes one more token of context adds to the
    layer. ``output_bytes`` is the output layer's weights and ``logits_bytes``
    the logits of one batch, both on the output device. ``reserve_bytes`` is
    charged to every device that is used (driver context and compute buffer)
    and ``spread_bytes`` (a draft model split the same way) is shared among
    them by the units they hold."""
    layer_bytes: Sequence[int]
    expert_bytes: Sequence[int]
    layer_kv: Sequence[int]
    kv_per_token: Sequence[int]
    output_bytes: int
    logits_bytes: int
    reserve_bytes: int
    spread_bytes: int = 0

    @property
    def n_layer(self) -> int:
        return len(self.layer_bytes)


@dataclass
class DeviceLoad:
    """What one device holds under a placement. Sizes in bytes."""
    device: PlanDevice
    layers: list[int] = field(default_factory=list)
    output: bool = False
    weights: int = 0
    kv: int = 0
    output_bytes: int = 0
    logits: int = 0
    reserve: int = 0
    kv_per_token: int = 0

    @property
    def units(self) -> int:
        """The units this device holds: its layers plus the output layer."""
        return len(self.layers) + (1 if self.output else 0)

    @property
    def need(self) -> int:
        return self.weights + self.kv + self.output_bytes + self.logits + self.reserve

    @property
    def fits(self) -> bool:
        return self.need <= self.device.free


@dataclass
class Placement:
    """One placement of a model over GPUs.

    ``loads`` are the devices that hold something, in the order llama.cpp
    receives them (the output device last). ``n_gpu_layers`` is the value for
    llama.cpp (the number of units on GPUs, ``n_layer + 1`` for every one),
    ``n_cpu_moe`` how many leading blocks keep their routed experts in system
    RAM, ``fits`` whether every device holds its charge."""
    loads: list[DeviceLoad]
    n_gpu_layers: int
    n_cpu_moe: int
    n_layer: int
    fits: bool

    @property
    def full_offload(self) -> bool:
        return self.n_gpu_layers >= self.n_layer + 1

    @property
    def tensor_split(self) -> list[int]:
        """Units per device, in ``loads`` order."""
        return [d.units for d in self.loads]

    @property
    def need(self) -> int:
        return sum(d.need for d in self.loads)

    def max_ctx(self, n_ctx: int, reserve_each: int = 0) -> Optional[int]:
        """The largest context window every device still holds: each device's
        free bytes after its charge at *n_ctx* tokens and *reserve_each*,
        divided by the KV bytes a token adds to its layers, plus *n_ctx*. None
        when no device holds a layer with a KV cache."""
        best = None
        for d in self.loads:
            if d.kv_per_token <= 0:
                continue
            spare = d.device.free - d.need - int(reserve_each)
            tokens = n_ctx + max(0, spare) // d.kv_per_token
            best = tokens if best is None else min(best, tokens)
        return best


def _gpu_units(units: ModelUnits, n_cpu_moe: int, n_gpu_layers: int) -> list:
    """``(layer index or None for the output, bytes, kv, kv_per_token)`` for
    each unit llama.cpp puts on a GPU, in order."""
    n = units.n_layer
    act = max(0, min(int(n_gpu_layers), n + 1))
    start = n + 1 - act
    out = []
    for il in range(start, n + 1):
        if il == n:
            out.append((None, units.output_bytes + units.logits_bytes, 0, 0))
            continue
        weights = int(units.layer_bytes[il])
        if il < n_cpu_moe:
            weights -= int(units.expert_bytes[il])
        out.append((il, max(0, weights), int(units.layer_kv[il]),
                    int(units.kv_per_token[il])))
    return out


def _order(devices: Sequence[PlanDevice]) -> list[PlanDevice]:
    """Devices in the order they receive units: the order of *devices*. The
    output layer goes on the last one that receives any unit."""
    return list(devices)


def _charge(ordered: Sequence[PlanDevice], counts: Sequence[int], gpu_units: list,
            units: ModelUnits) -> list[DeviceLoad]:
    """The :class:`DeviceLoad` of each device in *ordered* holding *counts*
    consecutive units of *gpu_units*. Devices with no unit are left out."""
    loads = []
    pos = 0
    for dev, count in zip(ordered, counts, strict=True):
        if count <= 0:
            continue
        load = DeviceLoad(device=dev, reserve=int(units.reserve_bytes))
        for il, weights, kv, per_token in gpu_units[pos:pos + count]:
            if il is None:
                load.output = True
                load.output_bytes = int(units.output_bytes)
                load.logits = int(units.logits_bytes)
            else:
                load.layers.append(il)
                load.weights += weights
                load.kv += kv
                load.kv_per_token += per_token
        pos += count
        loads.append(load)
    if units.spread_bytes > 0 and loads:
        total = sum(ld.units for ld in loads) or 1
        for ld in loads:
            ld.reserve += -(-int(units.spread_bytes) * ld.units // total)
    return loads


def _rooms(ordered: Sequence[PlanDevice], units: ModelUnits) -> list[int]:
    """Bytes each device can give to units: free memory less the reserve and
    its share of *spread_bytes* (charged as if every device were used)."""
    spread = -(-int(units.spread_bytes) // max(1, len(ordered)))
    return [d.free - int(units.reserve_bytes) - spread for d in ordered]


def _pack(gpu_units: list, rooms: Sequence[int], fill: float) -> Optional[list[int]]:
    """Unit counts that give each device, in order, the next units while they
    fit within *fill* of its room; None when units are left over."""
    counts = []
    pos = 0
    for room in rooms:
        cap = room * fill
        used = 0
        start = pos
        while pos < len(gpu_units):
            cost = gpu_units[pos][1] + gpu_units[pos][2]
            if used + cost > cap:
                break
            used += cost
            pos += 1
        counts.append(pos - start)
    return counts if pos == len(gpu_units) else None


def _pack_ratios(gpu_units: list, ordered: Sequence[PlanDevice]) -> list[int]:
    """Unit counts proportional to the devices' pinned ratios (largest
    remainder), every unit placed."""
    total = sum(max(0.0, float(d.ratio or 0.0)) for d in ordered) or 1.0
    n = len(gpu_units)
    exact = [n * max(0.0, float(d.ratio or 0.0)) / total for d in ordered]
    counts = [int(x) for x in exact]
    rest = n - sum(counts)
    for i in sorted(range(len(exact)), key=lambda i: exact[i] - counts[i],
                    reverse=True)[:rest]:
        counts[i] += 1
    return counts


def _fit(devices: Sequence[PlanDevice], units: ModelUnits, n_cpu_moe: int,
         n_gpu_layers: int) -> Optional[Placement]:
    """The most even placement of the last *n_gpu_layers* units over
    *devices* with every device within its free memory, or None when there is
    none. With pinned ratios the counts follow the ratios and the placement is
    returned whether it fits or not."""
    ordered = _order(devices)
    gpu_units = _gpu_units(units, n_cpu_moe, n_gpu_layers)
    if not gpu_units:
        return Placement([], 0, n_cpu_moe, units.n_layer, True)
    if all(d.ratio is not None for d in ordered):
        counts = _pack_ratios(gpu_units, ordered)
        loads = _charge(ordered, counts, gpu_units, units)
        return Placement(loads, len(gpu_units), n_cpu_moe, units.n_layer,
                         all(ld.fits for ld in loads))
    rooms = _rooms(ordered, units)
    if any(r > 0 for r in rooms) and _pack(gpu_units, rooms, 1.0) is not None:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if _pack(gpu_units, rooms, mid) is None:
                lo = mid
            else:
                hi = mid
        counts = _pack(gpu_units, rooms, hi)
        if counts is None:
            return None
        loads = _charge(ordered, counts, gpu_units, units)
        if all(ld.fits for ld in loads):
            return Placement(loads, len(gpu_units), n_cpu_moe, units.n_layer, True)
    return None


def plan_placement(devices: Sequence[PlanDevice], units: ModelUnits, *,
                   n_gpu_layers: Optional[int] = None,
                   n_cpu_moe: Optional[int] = None) -> Placement:
    """Place a model over *devices* (two or more), in their order: the first
    device gets the first units placed on a GPU.

    *n_gpu_layers* None places as many units as fit, else exactly that many
    (``n_layer + 1`` or more means every unit); *n_cpu_moe* None chooses how
    many leading blocks keep their routed experts in system RAM, else exactly
    that many. Automatic choices try, in order: every unit on the GPUs; every
    unit with the routed experts of the fewest leading blocks in system RAM;
    the most units that fit (with every block's experts in system RAM when the
    model has any). Devices that hold nothing are left out.

    When not even one unit fits, the automatic choice is every layer on the
    CPU (no devices, ``n_gpu_layers`` 0). The returned placement has ``fits``
    False only for a fixed *n_gpu_layers* that does not fit, or pinned ratios
    that overfill a device; it then holds the requested units (spread by free
    memory, or by the ratios), so the caller can report what each device
    would need."""
    n = units.n_layer
    all_units = n + 1
    experts = [il for il in range(n) if units.expert_bytes[il] > 0]
    moe_steps = [0] + [il + 1 for il in experts] if n_cpu_moe is None else [int(n_cpu_moe)]
    if n_gpu_layers is not None:
        want = min(max(0, int(n_gpu_layers)), all_units)
        for moe in moe_steps:
            plan = _fit(devices, units, moe, want)
            if plan is not None and (plan.fits or moe == moe_steps[-1]):
                return plan
        return _spread(devices, units, moe_steps[-1], want)
    for moe in moe_steps:
        plan = _fit(devices, units, moe, all_units)
        if plan is not None and plan.fits:
            return plan
    moe = moe_steps[-1]
    lo, hi = 0, all_units
    best = None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        plan = _fit(devices, units, moe, mid)
        if plan is not None and plan.fits:
            best, lo = plan, mid
        else:
            hi = mid - 1
    if best is not None:
        return best
    return Placement([], 0, 0, n, True)


def _spread(devices: Sequence[PlanDevice], units: ModelUnits, n_cpu_moe: int,
            n_gpu_layers: int) -> Placement:
    """The units spread over every device by free memory, fit or not."""
    ordered = _order(devices)
    gpu_units = _gpu_units(units, n_cpu_moe, n_gpu_layers)
    pinned = [PlanDevice(d.slot, d.index, d.name, d.free, d.total, d.device_id,
                         ratio=float(max(1, d.free))) for d in ordered]
    counts = _pack_ratios(gpu_units, pinned)
    loads = _charge(ordered, counts, gpu_units, units)
    return Placement(loads, len(gpu_units), n_cpu_moe, units.n_layer,
                     all(ld.fits for ld in loads))
