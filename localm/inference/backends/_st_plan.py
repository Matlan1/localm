# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stages a sentence-transformers folder applies after its transformer.

A sentence-transformers model is a transformer (the folder's own config.json and
weights) followed by the modules listed in ``modules.json``: ``Pooling`` turns the
token vectors into one vector, ``Dense`` projects it, ``Normalize`` scales it to
unit length. ``load_plan`` reads that list and each stage's folder; ``apply_plan``
runs the stages on the transformer's last hidden states.

Only the stages named above are run. A folder listing any other module, or a Dense
stage whose weights are not in a ``model.safetensors`` file, raises
:class:`StUnsupportedError` naming it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_MAX_SEQ_LENGTH = 512

_LEGACY_POOLING_FLAGS = (
    ("pooling_mode_cls_token", "cls"),
    ("pooling_mode_max_tokens", "max"),
    ("pooling_mode_mean_tokens", "mean"),
    ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len_tokens"),
    ("pooling_mode_weightedmean_tokens", "weightedmean"),
    ("pooling_mode_lasttoken", "lasttoken"),
)
_POOLING_MODES = frozenset(mode for _, mode in _LEGACY_POOLING_FLAGS)
_ACTIVATIONS = {"Tanh": "tanh", "Identity": "identity", "ReLU": "relu", "GELU": "gelu",
                "Sigmoid": "sigmoid"}


class StUnsupportedError(RuntimeError):
    """A sentence-transformers folder uses a module this loader does not run."""


@dataclass(frozen=True)
class Stage:
    """One post-transformer stage: ``kind`` is ``pooling``, ``dense`` or ``normalize``."""
    kind: str
    pooling_modes: tuple = ()
    weight_file: Optional[Path] = None
    activation: str = "identity"


@dataclass(frozen=True)
class StPlan:
    stages: tuple
    max_seq_length: int


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None


def is_st_folder(model_path: str) -> bool:
    """Whether *model_path* holds a sentence-transformers ``modules.json``."""
    return (Path(model_path) / "modules.json").is_file()


def _pooling_modes(config: dict) -> tuple:
    mode = config.get("pooling_mode")
    if isinstance(mode, str):
        modes = (mode,)
    elif isinstance(mode, list):
        modes = tuple(m for m in mode if isinstance(m, str))
    else:
        modes = tuple(name for flag, name in _LEGACY_POOLING_FLAGS if config.get(flag))
    unknown = [m for m in modes if m not in _POOLING_MODES]
    if unknown or not modes:
        raise StUnsupportedError(
            "this sentence-transformers folder uses the pooling mode "
            f"{(unknown or ['(none)'])[0]!r}, which localm does not run")
    return modes


def _activation(config: dict, stage: str) -> str:
    name = str(config.get("activation_function") or "torch.nn.modules.linear.Identity")
    short = name.rsplit(".", 1)[-1]
    if short not in _ACTIVATIONS:
        raise StUnsupportedError(
            f"the Dense stage {stage!r} of this sentence-transformers folder uses the "
            f"activation {name!r}, which localm does not run")
    return _ACTIVATIONS[short]


def load_plan(model_path: str) -> Optional[StPlan]:
    """The stages of the sentence-transformers folder *model_path*, or None when it
    has no ``modules.json``. Raises :class:`StUnsupportedError` for a module or
    weights file localm cannot run."""
    folder = Path(model_path)
    modules = _read_json(folder / "modules.json")
    if modules is None:
        return None
    if not isinstance(modules, list):
        raise StUnsupportedError("modules.json of this sentence-transformers folder is not a list")
    stages = []
    for entry in sorted((m for m in modules if isinstance(m, dict)),
                        key=lambda m: m.get("idx", 0) if isinstance(m.get("idx"), int) else 0):
        kind = str(entry.get("type") or "").rsplit(".", 1)[-1]
        sub = str(entry.get("path") or "")
        if kind == "Transformer":
            if sub:
                raise StUnsupportedError(
                    f"the transformer of this sentence-transformers folder is in the "
                    f"subfolder {sub!r}; localm loads one at the top of the folder")
            continue
        if kind == "Pooling":
            config = _read_json(folder / sub / "config.json")
            if not isinstance(config, dict):
                raise StUnsupportedError(f"{sub}/config.json of the Pooling stage is missing")
            stages.append(Stage("pooling", pooling_modes=_pooling_modes(config)))
        elif kind == "Normalize":
            stages.append(Stage("normalize"))
        elif kind == "Dense":
            config = _read_json(folder / sub / "config.json")
            if not isinstance(config, dict):
                raise StUnsupportedError(f"{sub}/config.json of the Dense stage is missing")
            weights = folder / sub / "model.safetensors"
            if not weights.is_file():
                raise StUnsupportedError(
                    f"the Dense stage {sub!r} of this sentence-transformers folder keeps its "
                    "weights in a pickle file (pytorch_model.bin), which localm does not load; "
                    "use a folder with model.safetensors in it")
            stages.append(Stage("dense", weight_file=weights,
                                activation=_activation(config, sub)))
        else:
            raise StUnsupportedError(
                f"this sentence-transformers folder uses the module {kind or '(unnamed)'!r}, "
                "which localm does not run")
    if not any(s.kind == "pooling" for s in stages):
        raise StUnsupportedError("this sentence-transformers folder has no Pooling stage")
    bert = _read_json(folder / "sentence_bert_config.json")
    length = bert.get("max_seq_length") if isinstance(bert, dict) else None
    if not (isinstance(length, int) and not isinstance(length, bool) and length > 0):
        length = DEFAULT_MAX_SEQ_LENGTH
    return StPlan(tuple(stages), length)


def _pool(torch, hidden, mask, modes: tuple):
    """The pooled vector of *hidden* (1, seq, dim) under attention *mask* (1, seq)."""
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    out = []
    count = mask_f.sum(dim=1).clamp(min=1e-9)
    summed = (hidden * mask_f).sum(dim=1)
    for mode in modes:
        if mode == "cls":
            first = mask.to(torch.int).argmax(dim=1)
            out.append(hidden[torch.arange(hidden.shape[0]), first])
        elif mode == "max":
            out.append(hidden.masked_fill(mask_f == 0, float("-inf")).max(dim=1).values)
        elif mode == "mean":
            out.append(summed / count)
        elif mode == "mean_sqrt_len_tokens":
            out.append(summed / torch.sqrt(count))
        elif mode == "weightedmean":
            weights = torch.arange(1, hidden.shape[1] + 1, device=hidden.device,
                                   dtype=hidden.dtype).view(1, -1, 1)
            weighted = mask_f * weights
            out.append((hidden * weighted).sum(dim=1) / weighted.sum(dim=1).clamp(min=1e-9))
        elif mode == "lasttoken":
            values, indices = mask.flip(1).max(1)
            indices = torch.where(values == 0, hidden.shape[1] - 1, indices)
            last = hidden.shape[1] - indices - 1
            out.append((hidden * mask_f)[torch.arange(hidden.shape[0]), last])
    return torch.cat(out, dim=-1)


def _dense(torch, vector, stage: Stage, cache: dict):
    from safetensors.torch import load_file
    weight_file = stage.weight_file
    if weight_file is None:
        raise StUnsupportedError("a Dense stage has no weights file")
    key = str(weight_file)
    if key not in cache:
        cache[key] = load_file(key)
    tensors = cache[key]
    weight = tensors.get("linear.weight")
    if weight is None:
        raise StUnsupportedError(
            f"{weight_file.parent.name}/model.safetensors has no 'linear.weight' tensor")
    out = torch.nn.functional.linear(
        vector, weight.to(vector.dtype).to(vector.device),
        tensors["linear.bias"].to(vector.dtype).to(vector.device) if "linear.bias" in tensors else None)
    return {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid,
            "gelu": torch.nn.functional.gelu, "identity": lambda v: v}[stage.activation](out)


def apply_plan(torch, plan: StPlan, hidden, mask, cache: dict):
    """The sentence vector (1, dim) *plan* gives for the transformer's last hidden
    states *hidden* (1, seq, dim) under the attention *mask* (1, seq). *cache*
    holds the Dense weights already read from disk, keyed by file."""
    vector = None
    for stage in plan.stages:
        if stage.kind == "pooling":
            vector = _pool(torch, hidden, mask, stage.pooling_modes)
        elif vector is None:
            raise StUnsupportedError(
                "this sentence-transformers folder applies a stage before its Pooling stage")
        elif stage.kind == "dense":
            vector = _dense(torch, vector, stage, cache)
        elif stage.kind == "normalize":
            vector = torch.nn.functional.normalize(vector, p=2.0, dim=1)
    if vector is None:
        raise StUnsupportedError("this sentence-transformers folder has no Pooling stage")
    return vector
