# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run a Kimi-Linear checkpoint larger than the memory at hand on transformers' own classes.

transformers stacks the routed experts of a layer into one tensor, about 3.4 GB per layer on the
48B model, so an automatic device map moves that whole tensor for every token even though a token
uses 8 of the layer's 256 experts. :func:`prepare` replaces the expert module with one that holds no
weights; everything else of the model goes on the device, and :class:`ExpertStore` reads each expert
from the checkpoint's own ``.safetensors`` files the first time it is routed to, and keeps the most
recently used ones on the device up to a byte budget.
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from localm.inference.backends._hf_expert_files import _EXPERT_KEY, _safetensors_header, _weight_map, worth_using

__all__ = ["ExpertStore", "OnDemandExperts", "_EXPERT_KEY", "_weight_map", "attach", "free_bytes", "prepare", "worth_using"]

logger = logging.getLogger("localm")

_HEADROOM_BYTES = int(1.5e9)


_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}


class ExpertStore:
    """The experts of a checkpoint, read on demand and kept on *device* in least recently used
    order up to *capacity* bytes. Not thread-safe."""

    def __init__(self, folder: Path, weight_map: dict, device, dtype, capacity: int):
        self._folder = folder
        self._weight_map = weight_map
        self.device = device
        self.dtype = dtype
        self.capacity = capacity
        self.held_bytes = 0
        self.reads = 0
        self.hits = 0
        self._files: dict = {}
        self._held: collections.OrderedDict = collections.OrderedDict()

    def _read(self, layer: int, expert: int, part: int) -> torch.Tensor:
        """One weight, read with plain file reads: mapping every shard of a large checkpoint
        needs more commit charge than Windows may have (os error 1455). Raises ValueError for a
        weight map that names a file outside the folder and OSError for a file shorter than its
        header says."""
        name = f"model.layers.{layer}.block_sparse_moe.experts.{expert}.w{part}.weight"
        file_name = self._weight_map[name]
        if Path(file_name).name != file_name or "\\" in file_name:
            raise ValueError(f"the weight map names a file outside the model folder: {file_name!r}")
        opened = self._files.get(file_name)
        if opened is None:
            path = self._folder / file_name
            opened = (open(path, "rb"), *_safetensors_header(path))
            self._files[file_name] = opened
        handle, header, data_start = opened
        entry = header[name]
        start, end = entry["data_offsets"]
        handle.seek(data_start + start)
        data = bytearray(end - start)
        if handle.readinto(data) != len(data):
            raise OSError(f"{file_name} is shorter than its header says; the download may be incomplete")
        return torch.frombuffer(data, dtype=_DTYPES[entry["dtype"]]).reshape(entry["shape"])

    def expert(self, layer: int, expert: int) -> tuple:
        """``(gate_up, down)`` of the expert, on the device: the gate and up weights
        concatenated on their first axis, and the down weight."""
        key = (layer, expert)
        held = self._held.get(key)
        if held is not None:
            self.hits += 1
            self._held.move_to_end(key)
            return held
        self.reads += 1
        gate_up = torch.cat([self._read(layer, expert, 1), self._read(layer, expert, 3)], dim=0)
        pair = (gate_up.to(self.device, self.dtype), self._read(layer, expert, 2).to(self.device, self.dtype))
        size = sum(t.numel() * t.element_size() for t in pair)
        while self._held and self.held_bytes + size > self.capacity:
            _, gone = self._held.popitem(last=False)
            self.held_bytes -= sum(t.numel() * t.element_size() for t in gone)
        if size <= self.capacity:
            self._held[key] = pair
            self.held_bytes += size
        return pair

    def close(self) -> None:
        """Let go of the file handles and of the experts on the device."""
        for opened in self._files.values():
            opened[0].close()
        self._files.clear()
        self._held.clear()
        self.held_bytes = 0


class OnDemandExperts(nn.Module):
    """``KimiLinearExperts`` with no weights of its own: :func:`attach` gives it the store."""

    def __init__(self, config):
        super().__init__()
        from transformers.activations import ACT2FN
        self.num_experts = config.num_local_experts
        self.act_fn = ACT2FN[config.hidden_act]
        self.store: Optional[ExpertStore] = None
        self.layer = -1
        # What transformers' weight initialisation touches on this class; holds nothing.
        self.gate_up_proj = torch.empty(0)
        self.down_proj = torch.empty(0)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        store = self.store
        if store is None:
            raise RuntimeError("the expert weights were not attached to this model")
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert in hit:
            expert = int(expert[0])
            if expert == self.num_experts:
                continue
            position, token = torch.where(mask[expert])
            gate_up, down = store.expert(self.layer, expert)
            gate, up = nn.functional.linear(hidden_states[token], gate_up).chunk(2, dim=-1)
            out = nn.functional.linear(self.act_fn(gate) * up, down) * top_k_weights[token, position, None]
            final.index_add_(0, token, out.to(final.dtype))
        return final


def prepare() -> None:
    """Make transformers' Kimi-Linear build :class:`OnDemandExperts`. Call before ``from_pretrained``."""
    from transformers.models.kimi_linear import modeling_kimi_linear as modeling
    modeling.KimiLinearExperts = OnDemandExperts


def attach(model, model_path: str, device, available_bytes: int) -> ExpertStore:
    """Give every :class:`OnDemandExperts` of *model* one store, which keeps up to
    *available_bytes* of experts on *device*. Returns the store."""
    folder = Path(model_path)
    dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
    store = ExpertStore(folder, _weight_map(folder), device, dtype, available_bytes)
    for layer, block in enumerate(model.model.layers):
        experts = getattr(getattr(block, "mlp", None), "experts", None)
        if isinstance(experts, OnDemandExperts):
            experts.store = store
            experts.layer = layer
    return store


def free_bytes(device: str, torch_module=torch, gpu: int = 0) -> int:
    """Bytes the store may keep on *device*: free video memory of GPU *gpu* less a margin for the
    generation's own tensors, or half the available RAM on the CPU."""
    if device == "cuda":
        return max(0, int(torch_module.cuda.mem_get_info(gpu)[0]) - _HEADROOM_BYTES)
    import psutil
    return int(psutil.virtual_memory().available) // 2
