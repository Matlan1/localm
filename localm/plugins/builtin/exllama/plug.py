# SPDX-License-Identifier: AGPL-3.0-or-later
"""ExLlama plugin: registers an inference backend for Hugging Face folders
quantized as EXL2 or EXL3 (``config.json`` ``quantization_config.quant_method``).

The backend (``backend.py``) runs the model in an isolated child process
(``worker.py``) on an NVIDIA GPU through ExLlamaV3 (EXL3) or ExLlamaV2 (EXL2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from localm.plugins.contract import InferenceBackendDescriptor

FORMATS = {"exl2": "EXL2", "exl3": "EXL3"}


def quant_format(model_path: str) -> Optional[str]:
    """``"exl2"`` or ``"exl3"`` when *model_path* is a folder whose
    ``config.json`` names that ``quantization_config.quant_method`` (any case)
    and does not flag ``load_in_4bit`` / ``load_in_8bit``; None for anything
    else, including an unreadable or malformed ``config.json``. Never raises."""
    try:
        folder = Path(model_path)
        if not folder.is_dir():
            return None
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(config, dict):
        return None
    qconf = config.get("quantization_config")
    if not isinstance(qconf, dict):
        return None
    if qconf.get("load_in_4bit") or qconf.get("load_in_8bit"):
        return None
    method = qconf.get("quant_method")
    if not isinstance(method, str):
        return None
    method = method.lower()
    return method if method in FORMATS else None


def claims(model_path: str) -> bool:
    """True for an EXL2 or EXL3 model folder (see :func:`quant_format`)."""
    return quant_format(model_path) is not None


def create(model_path: str, **opts):
    """An unloaded :class:`backend.ExllamaBackend` for *model_path*."""
    from .backend import ExllamaBackend
    return ExllamaBackend(model_path, device=opts.get("device"), n_ctx=opts.get("n_ctx"),
                          fmt=quant_format(model_path))


def refusal(model_path: str) -> Optional[str]:
    """The sentence a load of *model_path* would be refused with on this
    machine, or None (see :func:`backend.load_refusal`)."""
    fmt = quant_format(model_path)
    if fmt is None:
        return None
    from .backend import load_refusal
    return load_refusal(fmt)


DESCRIPTOR = InferenceBackendDescriptor(
    name="exllama", claims=claims, create=create, label="ExLlama (EXL2 / EXL3)",
    refusal=refusal)


def register(host) -> None:
    host.register_inference_backend(DESCRIPTOR)


def unregister() -> None:
    """Nothing to release: the host removes the backend descriptor."""
