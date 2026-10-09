# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recognise model files and folders localm cannot run, and say why.

Every function here is a classifier that returns a one-sentence explanation
(format, why localm cannot run it, nearest runnable alternative) or ``None``.
``None`` means "no verdict": a file localm can run, or one that matches no known
signature, never gets a sentence. Pure stdlib; reads at most a few bytes or one
small JSON file and never raises.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

GGUF_MAX_VERSION = 3
_GGUF_MAX_PLAUSIBLE_VERSION = 255
GGUF_MIN_VERSION = 2
_GGUF_BE_COUNT_LIMIT = 1_000_000

GGUF_IMATRIX_REFUSAL = (
    "This GGUF is an importance matrix (general.type 'imatrix') produced by "
    "llama-imatrix to help quantise a model, not a model itself. Point localm at "
    "the model's own .gguf file instead."
)

_ALTERNATIVE = "Use a GGUF of the same model, or the original Hugging Face weights."


def gguf_version_supported(version: int) -> bool:
    """True for the GGUF format versions the bundled llama.cpp loads (2 and 3)."""
    return GGUF_MIN_VERSION <= version <= GGUF_MAX_VERSION


def gguf_header_refusal(head: bytes) -> Optional[str]:
    """The reason a file starting with *head* (its first 8 to 24 bytes) is a GGUF
    localm cannot load, or None when the magic is absent, the header is too short
    to judge, or the version is one that loads.

    Mirrors llama.cpp's own checks: a version whose low 16 bits are zero is a
    byte-swapped (big-endian) file, version 1 is no longer supported, and a
    version from 4 to 255 is newer than the loader understands. Any other value
    gets no verdict: a magic followed by zero bytes or text is also what a
    placeholder or a damaged copy looks like, and the loader reports it
    itself."""
    if len(head) < 8 or head[:4] != b"GGUF":
        return None
    (version,) = struct.unpack_from("<I", head, 4)
    if version == 0 or gguf_version_supported(version):
        return None
    if version & 0xFFFF == 0:
        (swapped,) = struct.unpack_from(">I", head, 4)
        if swapped in (1, 2, 3) and _big_endian_counts_plausible(head):
            return ("This GGUF was written for a big-endian machine (for example "
                    "IBM s390x), so its bytes are swapped for this CPU and it "
                    "cannot be loaded here. Use the standard little-endian GGUF "
                    "of the same model.")
    if version == 1:
        return ("This is a GGUF version 1 file, an early format llama.cpp no longer "
                "loads. Use a current GGUF (version 3) of the same model.")
    if version > _GGUF_MAX_PLAUSIBLE_VERSION:
        return None
    return (f"This GGUF declares version {version}, which this build of llama.cpp "
            f"does not understand (it loads versions {GGUF_MIN_VERSION} to "
            f"{GGUF_MAX_VERSION}). Use a GGUF of version {GGUF_MAX_VERSION}, or "
            "update localm if the file is newer.")


def _big_endian_counts_plausible(head: bytes) -> bool:
    """Whether the tensor and key-value counts of a header read big-endian are
    small, as in a real byte-swapped file. True when the header is too short to
    hold the counts."""
    if len(head) < 24:
        return True
    tensors, kvs = struct.unpack_from(">QQ", head, 8)
    return tensors <= _GGUF_BE_COUNT_LIMIT and kvs <= _GGUF_BE_COUNT_LIMIT


def gguf_file_refusal(path: Path) -> Optional[str]:
    """``gguf_header_refusal`` for the first bytes of *path*. None when the file
    cannot be read."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    return gguf_header_refusal(head)


# First four bytes (the little-endian bytes of the old uint32 magics) of the
# pre-GGUF llama.cpp / ggml file formats.
_LEGACY_GGML_MAGICS = {
    b"tjgg": ("GGJT", "legacy llama.cpp"),
    b"fmgg": ("GGMF", "legacy llama.cpp"),
    b"lmgg": ("GGML", "legacy ggml"),
}


def legacy_ggml_refusal(path: Path) -> Optional[str]:
    """The reason a lone ``.bin`` file with a pre-GGUF magic cannot be run, or
    None for any other file, including a ``.bin`` beside a ``config.json`` (a
    shard of a HuggingFace folder). The bare ``ggml`` magic is shared by early
    llama.cpp models and whisper.cpp models, so its sentence names both."""
    if path.suffix.lower() != ".bin" or not path.is_file():
        return None
    if (path.parent / "config.json").exists():
        return None
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
    except OSError:
        return None
    kind = _LEGACY_GGML_MAGICS.get(magic)
    if kind is None:
        return None
    fmt, who = kind
    if fmt == "GGML":
        return ("This is an old-format GGML file (the pre-GGUF format used by early "
                "llama.cpp and by whisper.cpp). localm loads GGUF, not GGML. For a "
                "language model, use a GGUF of the same model; a whisper.cpp speech "
                "model cannot be run from this file.")
    return (f"This is a {who} model in the old {fmt} format, from before GGUF. "
            "localm loads GGUF only. Use a GGUF of the same model.")


def _read_config(folder: Path) -> Optional[dict]:
    try:
        data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def mlx_quantized_refusal(config: dict) -> Optional[str]:
    """The reason a HuggingFace folder with this ``config.json`` is an
    MLX-quantized model (Apple's MLX packs weights its own way), or None.

    The signature is a top-level ``quantization`` object holding integer
    ``group_size`` and ``bits`` with no ``quant_method`` in any
    ``quantization_config`` beside it."""
    quant = config.get("quantization")
    if not (isinstance(quant, dict) and _is_int(quant.get("group_size"))
            and _is_int(quant.get("bits"))):
        return None
    qconf = config.get("quantization_config")
    if isinstance(qconf, dict) and qconf.get("quant_method"):
        return None
    return ("This is an MLX-quantized model (mlx-community format), which runs only "
            "in Apple's MLX and cannot be loaded here. Use the GGUF of the same "
            "model, or the original non-MLX weights.")


_UNSUPPORTED_QUANT_METHODS = {
    "exl2": ("EXL2", "ExLlamaV2"),
    "exl3": ("EXL3", "ExLlamaV3"),
}


def unsupported_quant_method_refusal(config: dict) -> Optional[str]:
    """The reason a HuggingFace folder with this ``config.json`` uses a
    quantization only another runtime can load, or None. Named methods only: any
    other ``quant_method`` is left to the loader."""
    qconf = config.get("quantization_config")
    if not isinstance(qconf, dict):
        return None
    method = qconf.get("quant_method")
    named = _UNSUPPORTED_QUANT_METHODS.get(method.lower()) if isinstance(method, str) else None
    if named is None:
        return None
    label, runtime = named
    return (f"This model is quantized with {label}, a format only {runtime} can run, "
            f"so localm cannot load it. {_ALTERNATIVE}")


def mistral_native_refusal(folder: Path) -> Optional[str]:
    """The reason a folder in Mistral's own layout (``params.json`` plus
    ``consolidated*.safetensors`` and no ``config.json``) cannot be loaded, or
    None. A folder that also holds ``config.json`` is a HuggingFace folder and
    gets no sentence."""
    try:
        if not folder.is_dir() or (folder / "config.json").exists():
            return None
        if not (folder / "params.json").is_file():
            return None
        if next(folder.glob("consolidated*.safetensors"), None) is None:
            return None
    except OSError:
        return None
    return ("This is a Mistral-native model folder (params.json plus "
            "consolidated*.safetensors, no config.json). localm loads GGUF files and "
            "Hugging Face-format folders (config.json plus model-*.safetensors), not "
            "this layout. Use the model's GGUF or its Hugging Face-format files.")


def openvino_refusal(folder: Path) -> Optional[str]:
    """The reason an OpenVINO export cannot be loaded (``openvino_model.xml`` at the
    top of *folder* and no PyTorch or safetensors weights beside it), or None."""
    try:
        if not (folder / "openvino_model.xml").is_file():
            return None
        for pattern in ("*.safetensors", "pytorch_model*", "*.pt", "*.pth"):
            if next(folder.glob(pattern), None) is not None:
                return None
    except OSError:
        return None
    return _OPENVINO_SENTENCE


def hf_folder_refusal(folder: Path) -> Optional[str]:
    """The reason the HuggingFace-style *folder* cannot be loaded (MLX-quantized,
    quantized for a runtime localm lacks, or an OpenVINO export), or None."""
    openvino = openvino_refusal(folder)
    if openvino is not None:
        return openvino
    config = _read_config(folder)
    if config is None:
        return None
    return mlx_quantized_refusal(config) or unsupported_quant_method_refusal(config)


# Extension -> sentence for a file pointed at directly.
_FILE_SUFFIX_EXPLANATIONS = {
    ".onnx": ("An .onnx file is an ONNX model, which needs an ONNX runtime; localm "
              "runs GGUF files and Hugging Face-format folders. " + _ALTERNATIVE),
    ".engine": ("An .engine file is a TensorRT engine, built for one GPU model and "
                "TensorRT version and runnable only by TensorRT. " + _ALTERNATIVE),
    ".plan": ("A .plan file is a TensorRT engine, built for one GPU model and "
              "TensorRT version and runnable only by TensorRT. " + _ALTERNATIVE),
    ".litertlm": ("A .litertlm file is a LiteRT-LM bundle for Google's on-device "
                  "runtime, which localm does not include. " + _ALTERNATIVE),
    ".task": ("A .task file is a MediaPipe / LiteRT bundle for Google's on-device "
              "runtime, which localm does not include. " + _ALTERNATIVE),
    ".nemo": ("A .nemo file is an NVIDIA NeMo checkpoint archive that loads only in "
              "NeMo. " + _ALTERNATIVE),
}

_OPENVINO_SENTENCE = (
    "This folder is an OpenVINO IR model, which runs only in the OpenVINO runtime. "
    + _ALTERNATIVE)

# Marker file inside a folder -> sentence for a folder pointed at directly.
_FOLDER_MARKER_EXPLANATIONS = (
    ("mlc-chat-config.json",
     "This folder is an MLC LLM compiled model, which runs only in the MLC LLM "
     "runtime. " + _ALTERNATIVE),
    ("openvino_model.xml", _OPENVINO_SENTENCE),
)


def explain_unsupported_model(path: Path) -> Optional[str]:
    """One sentence naming the format of *path*, why localm cannot run it, and the
    nearest runnable alternative; None for a model localm can run or for anything
    unrecognised.

    Covers what a user points ``localm add`` straight at: a lone legacy GGML
    ``.bin``, ``.onnx`` / TensorRT / LiteRT / NeMo files, and folders in MLC,
    OpenVINO, Mistral-native, MLX-quantized or EXL2/EXL3 layouts."""
    try:
        if path.is_file():
            legacy = legacy_ggml_refusal(path)
            if legacy is not None:
                return legacy
            return _FILE_SUFFIX_EXPLANATIONS.get(path.suffix.lower())
        if path.is_dir():
            for marker, sentence in _FOLDER_MARKER_EXPLANATIONS:
                if (path / marker).is_file():
                    return sentence
            return mistral_native_refusal(path) or hf_folder_refusal(path)
    except OSError:
        return None
    return None
