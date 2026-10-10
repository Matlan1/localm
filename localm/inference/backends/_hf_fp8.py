# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load planning for FP8 checkpoints (``quantization_config.quant_method ==
"fp8"``) in the HF worker.

An FP8 checkpoint runs natively only on an NVIDIA CUDA GPU with compute
capability 8.9 or newer, with triton installed and the
``kernels-community/finegrained-fp8`` Hub kernel available. Everywhere else
its weights are expanded to bf16 at load time
(``FineGrainedFP8Config(dequantize=True)``), about 2 bytes per parameter.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from localm.debuglog import logger

FP8_QUANT_METHOD = "fp8"
NATIVE_FP8_MIN_CAPABILITY = (8, 9)
BF16_BYTES = 2
SAFETENSORS_MAX_HEADER = 100 * 1024 * 1024

_INT_ITEMSIZE = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
                 "U32": 4, "I32": 4, "U64": 8, "I64": 8}


@dataclass
class Fp8Plan:
    """How an FP8 checkpoint is loaded.

    ``native``: True to keep FP8 weights and use the Hub kernel, False to
    expand them to bf16. ``reason``: why the weights are expanded (empty when
    native). ``offline``: the native path may only use a kernel already in the
    local cache. ``expanded_bytes``: the bf16 size of the weights, None when
    it could not be read.
    """
    native: bool
    reason: str = ""
    offline: bool = False
    expanded_bytes: Optional[int] = None


def quant_method(model_path: str) -> Optional[str]:
    """``quantization_config.quant_method`` from the model's config.json,
    lowercased, or None when there is none or the file cannot be read."""
    cfg_path = Path(model_path) / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as e:
        logger.debug("hf fp8: could not read %s: %s", cfg_path, e)
        return None
    qc = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    method = qc.get("quant_method") if isinstance(qc, dict) else None
    return method.strip().lower() if isinstance(method, str) else None


def is_fp8_checkpoint(model_path: str) -> bool:
    return quant_method(model_path) == FP8_QUANT_METHOD


def cuda_device_ids(torch, device_map_kwargs: dict) -> list:
    """The CUDA device indices a load with *device_map_kwargs* (see
    ``_hf_worker._cuda_device_map``) can place weights on."""
    dm = device_map_kwargs.get("device_map")
    if isinstance(dm, dict):
        return [i for i in dm.values() if isinstance(i, int)]
    max_memory = device_map_kwargs.get("max_memory")
    if isinstance(max_memory, dict):
        return sorted(i for i in max_memory if isinstance(i, int))
    try:
        return list(range(int(torch.cuda.device_count())))
    except Exception as e:
        logger.debug("hf fp8: could not count CUDA devices: %s", e)
        return []


def _native_check_ids(torch, device_map_kwargs: dict) -> list:
    """The devices in the map plus torch's current CUDA device, which is the
    one transformers' FP8 quantizer checks."""
    ids = cuda_device_ids(torch, device_map_kwargs)
    try:
        current = int(torch.cuda.current_device())
    except Exception as e:
        logger.debug("hf fp8: could not read the current CUDA device: %s", e)
        return ids
    return sorted(set(ids) | {current})


def native_blocker(torch, device: str, device_ids: Iterable[int]) -> Optional[str]:
    """None when *device* is an NVIDIA CUDA GPU whose every device in
    *device_ids* has compute capability >= 8.9, else the reason it is not.
    A ROCm build (``torch.version.hip`` set) is never native, whatever
    capability it reports."""
    if device != "cuda":
        where = {"cpu": "the CPU", "xpu": "an Intel XPU GPU"}.get(
            device, f"the {device} device")
        return f"the model runs on {where}"
    version = getattr(torch, "version", None)
    if getattr(version, "hip", None):
        return "the GPU is an AMD ROCm device"
    if not getattr(version, "cuda", None):
        return "torch is not a CUDA build"
    ids = list(device_ids)
    if not ids:
        return "no CUDA device could be read"
    for idx in ids:
        try:
            cap = tuple(int(x) for x in torch.cuda.get_device_capability(idx))
        except Exception as e:
            return f"the compute capability of GPU {idx} could not be read ({e})"
        if cap < NATIVE_FP8_MIN_CAPABILITY:
            return f"GPU {idx} has compute capability {cap[0]}.{cap[1]}"
    return None


def triton_blocker() -> Optional[str]:
    """None when ``import triton`` succeeds, else the reason."""
    try:
        import triton  # noqa: F401
    except Exception as e:
        return f"triton is not available ({type(e).__name__}: {e})"
    return None


def _safetensors_header(path: Path) -> dict:
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("file shorter than its header length")
        n = struct.unpack("<Q", raw)[0]
        if n > SAFETENSORS_MAX_HEADER or 8 + n > size:
            raise ValueError(f"header length {n} out of range")
        header = json.loads(f.read(n))
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return header


def weight_files(model_path: str) -> list:
    """The safetensors files transformers loads from *model_path*: the shards
    named in ``model.safetensors.index.json`` when it exists, else
    ``model.safetensors``. Other ``*.safetensors`` files in the directory are
    not weights of this load. Empty when neither exists, the index cannot be
    read, or it names a file outside the directory."""
    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            names = sorted({str(v) for v in weight_map.values()})
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as e:
            logger.debug("hf fp8: could not read %s: %s", index, e)
            return []
        base = root.resolve()
        files = []
        for name in names:
            path = (root / name).resolve()
            if not path.is_relative_to(base):
                logger.debug("hf fp8: %s names %r outside the model directory", index, name)
                return []
            files.append(path)
        return files
    single = root / "model.safetensors"
    return [single] if single.is_file() else []


def expanded_bf16_bytes(model_path: str) -> Optional[int]:
    """Bytes the model's weights (:func:`weight_files`) take once every
    floating-point tensor is bf16: element count x 2 for floating dtypes, the
    stored item size for integer and bool dtypes. None when there are no
    weight files or one cannot be parsed."""
    files = weight_files(model_path)
    if not files:
        return None
    total = 0
    for path in files:
        try:
            header = _safetensors_header(path)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                numel = math.prod(int(d) for d in info["shape"])
                total += numel * _INT_ITEMSIZE.get(str(info["dtype"]), BF16_BYTES)
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.debug("hf fp8: could not size %s: %s", path, e)
            return None
    return total


def _xpu_memory(torch) -> Optional[int]:
    """Free memory of XPU device 0, or its total memory on parts without a
    free-memory query, or None."""
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return None
    try:
        return int(xpu.mem_get_info(0)[0])
    except Exception as e:
        logger.debug("hf fp8: xpu mem_get_info failed: %s", e)
    try:
        return int(xpu.get_device_properties(0).total_memory)
    except Exception as e:
        logger.debug("hf fp8: could not read XPU memory: %s", e)
        return None


def memory_budget(torch, device: str, device_map_kwargs: dict) -> Optional[int]:
    """Bytes a load on *device* with *device_map_kwargs* can place weights in:
    available system RAM for a CPU load; the smaller of RAM and XPU memory for
    an XPU load (it loads on the CPU, then moves to the XPU); RAM plus free VRAM
    of the usable GPUs for a CUDA load that may overflow to CPU. A CUDA load
    pinned to one device (``device_map={"": idx}``) counts only that device's
    free VRAM. None when RAM or VRAM cannot be read."""
    try:
        import psutil
        ram = int(psutil.virtual_memory().available)
    except Exception as e:
        logger.debug("hf fp8: could not read available RAM: %s", e)
        return None
    if device == "xpu":
        xpu = _xpu_memory(torch)
        return ram if xpu is None else min(ram, xpu)
    if device != "cuda":
        return ram
    max_memory = device_map_kwargs.get("max_memory")
    if isinstance(max_memory, dict):
        return sum(int(v) for v in max_memory.values())
    ids = cuda_device_ids(torch, device_map_kwargs)
    try:
        vram = sum(int(torch.cuda.mem_get_info(i)[0]) for i in ids)
    except Exception as e:
        logger.debug("hf fp8: could not read free VRAM: %s", e)
        return None
    if isinstance(device_map_kwargs.get("device_map"), dict):
        return vram
    return ram + vram


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


REASON_MAX_CHARS = 240


def one_line(text: str, limit: int = REASON_MAX_CHARS) -> str:
    """*text* with every whitespace run collapsed to one space, cut to *limit*
    characters with a trailing "..." when longer."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit - 3].rstrip() + "..."


def too_big_message(model_path: str, plan: Fp8Plan, budget: int) -> str:
    return (
        f"'{Path(model_path).name}' is an FP8 model and native FP8 is unavailable "
        f"here ({one_line(plan.reason)}), so its weights are expanded to bf16 (about "
        f"2 bytes per parameter): it needs about {_gb(plan.expanded_bytes or 0)}, but "
        f"only about {_gb(budget)} of memory is available. Native FP8 needs an NVIDIA "
        "GPU with compute capability 8.9 or newer, triton and the finegrained-fp8 "
        "kernel. A GGUF quantization of this model needs far less memory.")


def describe(plan: Fp8Plan) -> str:
    """One line for the load output."""
    if plan.native:
        source = " from the local cache" if plan.offline else ""
        return f"FP8 weights run natively (finegrained-fp8 kernel{source})"
    size = (f", about {_gb(plan.expanded_bytes)}"
            if plan.expanded_bytes else "")
    return (f"FP8 weights expanded to bf16 (about 2 bytes per parameter{size}); "
            f"native FP8 is unavailable: {one_line(plan.reason)}")


def plan_load(model_path: str, torch, device: str, device_map_kwargs: dict, *,
              hub_fetch_refusal=None, triton_check=None) -> Optional[Fp8Plan]:
    """The load plan for *model_path*, or None when it is not an FP8 checkpoint.

    Native only when :func:`native_blocker` (over the devices in the map and
    torch's current CUDA device) and *triton_check* (default
    :func:`triton_blocker`) both return None. *hub_fetch_refusal* is a callable
    returning None when the network policy allows fetching the kernel from the
    Hub; when it returns a reason, the native plan is ``offline`` (a locally
    cached kernel only).
    """
    if not is_fp8_checkpoint(model_path):
        return None
    blocker = native_blocker(torch, device, _native_check_ids(torch, device_map_kwargs)
                             if device == "cuda" else [])
    if blocker is None:
        blocker = (triton_check or triton_blocker)()
    if blocker is not None:
        return expanded_plan(model_path, blocker)
    refusal = hub_fetch_refusal() if hub_fetch_refusal is not None else None
    if refusal:
        logger.debug("hf fp8: kernel fetch refused by the network policy (%s); "
                     "trying the local cache only", refusal)
    return Fp8Plan(native=True, offline=bool(refusal))


def expanded_plan(model_path: str, reason: str) -> Fp8Plan:
    """A bf16-expansion plan for *model_path* with *reason*."""
    return Fp8Plan(native=False, reason=reason,
                   expanded_bytes=expanded_bf16_bytes(model_path))
