#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model library census: every model in a folder, the role its own metadata
implies, and the role localm assigns it, side by side.

For each top-level entry of ``--models-dir`` (a model file, or a folder holding
one model) the census:

  1. reads the entry's metadata with its own readers, never localm's: the GGUF
     header (key/value pairs and tensor names), the safetensors JSON header, the
     string constants of a PyTorch ``.pth`` pickle (read with ``pickletools``,
     never unpickled), and a folder's ``config.json`` / ``model_index.json``;
  2. derives the IMPLIED kind from that metadata alone (never from the file name)
     and the localm model types that kind may be registered as;
  3. registers the entry in place into a throwaway LOCALM_HOME with the real
     ``localm add`` (never ``--store copy``/``move``), reads the registry and the
     running GUI server's ``GET /api/models``, and compares.

A row passes when localm registered the entry, its ``/api/models`` row carries
the same ``model_type`` as the registry, that type is one the implied kind
allows, and the architecture localm recorded matches the GGUF header's
``general.architecture``. Every other outcome fails the row: localm type
``unknown``, a type the metadata does not allow, an entry ``localm add`` did not
register, a row missing from ``/api/models``, or metadata the census cannot
classify ("undetermined", which needs a new rule here before the census can
pass).

Exit codes: 0 every row passes; 1 at least one row fails; 2 the census could
not run (no such folder, an empty folder, ``localm add`` or the server could not
start, ``/api/models`` unreadable) or left its throwaway home or a server
process behind. 2 is never reported as 0.

Every localm child (``localm add``, ``localm key generate``, the server, the
identity probe) runs with the throwaway home as its working directory, so all of
them import the localm that ``--python`` resolves (with ``PYTHONPATH`` when set);
the ``localm:`` line of the report names it.

The models folder is only read. The throwaway home is created under
``--work-dir`` and deleted at the end unless ``--keep`` is given; its
``config.json`` carries ``"update_url": ""`` and ``"bugreport_upload_url": ""``.

Usage:
    python scripts/model_census.py --models-dir <folder> --work-dir <scratch dir>
    python scripts/model_census.py --models-dir <folder> --work-dir <dir> --json out.json
    python scripts/model_census.py --models-dir <folder> --metadata-only
"""

from __future__ import annotations

import argparse
import json
import os
import pickletools
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

MODEL_FILE_SUFFIXES = (".gguf", ".safetensors", ".pth", ".pt", ".ckpt", ".bin", ".onnx",
                       ".sft", ".llamafile")
PARTIAL_SUFFIXES = (".part", ".partial", ".incomplete", ".tmp", ".download")

GGUF_MAGIC = b"GGUF"
GGUF_MAX_KV = 100_000
GGUF_MAX_TENSORS = 1_000_000
GGUF_MAX_STRING = 64 * 1024 * 1024
GGUF_MAX_ARRAY = 50_000_000
SAFETENSORS_MAX_HEADER = 100 * 1024 * 1024
PICKLE_MAX_BYTES = 64 * 1024 * 1024

# GGUF value type ids (GGUF specification): uint8, int8, uint16, int16, uint32,
# int32, float32, bool, string, array, uint64, int64, float64.
_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING = 8
_GGUF_ARRAY = 9

# llama.cpp ``enum llama_pooling_type``: RANK is the reranker head.
POOLING_RANK = 4

# ggml tensor type id -> (elements per block, bytes per block).
GGML_TYPE_SIZES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
    9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
    21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8),
    28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17),
}
GGUF_DEFAULT_ALIGNMENT = 32


class MetadataError(Exception):
    """The file's metadata could not be read (truncated, corrupt, or not the
    format its suffix claims)."""


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

def _read_exact(fh, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise MetadataError("file ends inside its header")
    return data


def _gguf_string(fh) -> str:
    (n,) = struct.unpack("<Q", _read_exact(fh, 8))
    if n > GGUF_MAX_STRING:
        raise MetadataError(f"GGUF string of {n} bytes")
    return _read_exact(fh, n).decode("utf-8", "replace")


def _gguf_value(fh, vtype: int, keep_array: bool):
    if vtype in _GGUF_SCALAR:
        fmt = _GGUF_SCALAR[vtype]
        return struct.unpack(fmt, _read_exact(fh, struct.calcsize(fmt)))[0]
    if vtype == _GGUF_STRING:
        return _gguf_string(fh)
    if vtype == _GGUF_ARRAY:
        (etype,) = struct.unpack("<I", _read_exact(fh, 4))
        (count,) = struct.unpack("<Q", _read_exact(fh, 8))
        if count > GGUF_MAX_ARRAY:
            raise MetadataError(f"GGUF array of {count} items")
        if etype in _GGUF_SCALAR and not keep_array:
            fh.seek(struct.calcsize(_GGUF_SCALAR[etype]) * count, os.SEEK_CUR)
            return {"array_of": etype, "count": count}
        items = [_gguf_value(fh, etype, False) for _ in range(count)]
        return items if keep_array else {"array_of": etype, "count": count}
    raise MetadataError(f"unknown GGUF value type {vtype}")


def read_gguf(path: Path) -> dict:
    """``{"kv": {key: value}, "tensors": [name, ...]}`` from a GGUF header.
    Array values are summarised as ``{"array_of", "count"}`` except the
    ``general.*`` arrays. Raises MetadataError on anything that is not a
    readable GGUF, including a file shorter than the tensor data its header
    declares (for a tensor type missing from ``GGML_TYPE_SIZES`` only its start
    offset is checked)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if _read_exact(fh, 4) != GGUF_MAGIC:
                raise MetadataError("no GGUF magic")
            (version,) = struct.unpack("<I", _read_exact(fh, 4))
            if version < 2:
                raise MetadataError(f"GGUF version {version} is not supported")
            n_tensors, n_kv = struct.unpack("<QQ", _read_exact(fh, 16))
            if n_kv > GGUF_MAX_KV or n_tensors > GGUF_MAX_TENSORS:
                raise MetadataError("implausible GGUF header counts")
            kv: dict[str, Any] = {}
            for _ in range(n_kv):
                key = _gguf_string(fh)
                (vtype,) = struct.unpack("<I", _read_exact(fh, 4))
                keep = vtype == _GGUF_ARRAY and key.startswith("general.")
                kv[key] = _gguf_value(fh, vtype, keep)
            tensors = []
            data_end = 0
            for _ in range(n_tensors):
                name = _gguf_string(fh)
                (n_dims,) = struct.unpack("<I", _read_exact(fh, 4))
                if n_dims > 8:
                    raise MetadataError(f"tensor {name!r} has {n_dims} dimensions")
                dims = struct.unpack(f"<{n_dims}Q", _read_exact(fh, 8 * n_dims))
                ttype, offset = struct.unpack("<IQ", _read_exact(fh, 12))
                n_el = 1
                for d in dims:
                    n_el *= d
                block = GGML_TYPE_SIZES.get(ttype)
                nbytes = n_el // block[0] * block[1] if block else 0
                data_end = max(data_end, offset + nbytes)
                tensors.append(name)
            align = kv.get("general.alignment")
            align = align if isinstance(align, int) and align > 0 else GGUF_DEFAULT_ALIGNMENT
            data_start = (fh.tell() + align - 1) // align * align
    except OSError as e:
        raise MetadataError(f"cannot read: {e}") from e
    if n_tensors and size < data_start + data_end:
        raise MetadataError(f"file is {size} bytes, its header declares tensor data up to"
                            f" byte {data_start + data_end} (truncated)")
    return {"version": version, "kv": kv, "tensors": tensors}


def read_safetensors(path: Path) -> dict:
    """``{"metadata": {...}, "tensors": [name, ...]}`` from a safetensors header."""
    try:
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", _read_exact(fh, 8))
            if n > SAFETENSORS_MAX_HEADER:
                raise MetadataError(f"safetensors header of {n} bytes")
            raw = _read_exact(fh, n)
    except OSError as e:
        raise MetadataError(f"cannot read: {e}") from e
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise MetadataError(f"safetensors header is not JSON: {e}") from e
    if not isinstance(header, dict):
        raise MetadataError("safetensors header is not a JSON object")
    meta = header.pop("__metadata__", None)
    return {"metadata": meta if isinstance(meta, dict) else {}, "tensors": sorted(header)}


def read_torch_pickle_strings(path: Path) -> list[str]:
    """The string constants of a PyTorch checkpoint's pickle (``data.pkl``
    inside the zip format, or a legacy bare pickle), read opcode by opcode with
    ``pickletools.genops``. Nothing is unpickled and no code runs."""
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                pkl = [n for n in zf.namelist() if n.endswith("data.pkl")]
                if not pkl:
                    raise MetadataError("PyTorch zip without data.pkl")
                info = zf.getinfo(pkl[0])
                if info.file_size > PICKLE_MAX_BYTES:
                    raise MetadataError("data.pkl is too large to scan")
                data = zf.read(pkl[0])
        else:
            with open(path, "rb") as fh:
                data = fh.read(PICKLE_MAX_BYTES)
    except (OSError, zipfile.BadZipFile) as e:
        raise MetadataError(f"cannot read: {e}") from e
    out = []
    try:
        for op, arg, _pos in pickletools.genops(data):
            if isinstance(arg, str) and op.name in ("SHORT_BINUNICODE", "BINUNICODE",
                                                    "BINUNICODE8", "UNICODE"):
                out.append(arg)
    except Exception as e:
        if not out:
            raise MetadataError(f"not a readable pickle: {e}") from e
    return out


def read_json_file(path: Path) -> Optional[dict]:
    """The JSON object in *path*, or None when it is absent. Raises
    MetadataError when it exists and is not a JSON object."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise MetadataError(f"{path.name} is not readable JSON: {e}") from e
    if not isinstance(data, dict):
        raise MetadataError(f"{path.name} is not a JSON object")
    return data


# --------------------------------------------------------------------------
# Implied kind from metadata
# --------------------------------------------------------------------------

# The localm model types each implied kind may be registered as.
KIND_TYPES: dict[str, tuple[str, ...]] = {
    "llm": ("llm",),
    "embedding": ("embedding",),
    "reranker": ("embedding", "reranker"),
    "projector": ("mmproj",),
    "text-encoder": ("text-encoder",),
    "diffusion": ("diffusion-unet",),
    "vae": ("vae",),
    "vocoder": ("vocoder",),
    "lora": ("lora",),
    "tts": ("tts",),
    "upscaler": ("upscaler",),
    "controlnet": ("controlnet",),
    "textual-inversion": ("textual-inversion",),
    "undetermined": (),
}

_ENCODER_TEXT_ARCHES = frozenset({
    "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "modern-bert",
    "neo-bert", "gemma-embedding", "roberta", "xlm-roberta", "eurobert",
})
_TTS_ARCHES = frozenset({"wavtokenizer-dec", "outetts", "orpheus", "kokoro"})
_DIFFUSION_TENSOR_PREFIXES = (
    "model.diffusion_model.", "double_blocks.", "single_blocks.", "transformer_blocks.",
    "joint_blocks.", "input_blocks.", "img_in.", "diffusion_model.", "x_embedder.",
    "patch_embedding.", "noise_refiner.",
)
_VAE_TENSOR_PREFIXES = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.",
                        "first_stage_model.", "conv1.", "conv2.", "vae.")
_TEXT_ENCODER_TENSOR_PREFIXES = ("encoder.block.", "text_model.encoder.", "enc.blk.",
                                 "cond_stage_model.", "text_encoders.")
_LORA_MARKERS = ("lora_", "_lora", ".lora.", "lora.", "hada_w", "lokr_w")
_CONTROLNET_MARKERS = ("control_model.", "input_hint_block.", "controlnet_")
_TI_MARKERS = ("emb_params", "string_to_param", "clip_l", "clip_g")


@dataclass
class Implied:
    kind: str
    evidence: str
    architecture: Optional[str] = None
    allowed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.allowed:
            self.allowed = KIND_TYPES.get(self.kind, ())


def _gguf_kind(meta: dict) -> Implied:
    kv = meta["kv"]
    tensors = meta["tensors"]
    arch = kv.get("general.architecture")
    arch = arch if isinstance(arch, str) and arch else None
    gtype = kv.get("general.type")
    tags = kv.get("general.tags") if isinstance(kv.get("general.tags"), list) else []

    def out(kind: str, why: str) -> Implied:
        return Implied(kind, why, arch)

    if gtype == "mmproj" or arch == "clip" or any(k.startswith("clip.has_") for k in kv):
        modes = [k[len("clip.has_"):-len("_encoder")] for k, v in kv.items()
                 if k.startswith("clip.has_") and k.endswith("_encoder") and v is True]
        return out("projector", "GGUF clip projector (" + ", ".join(modes or ["no encoder flag"]) + ")")
    if gtype == "adapter" or "adapter.type" in kv:
        return out("lora", f"GGUF adapter (adapter.type={kv.get('adapter.type')!r})")
    pooling = kv.get(f"{arch}.pooling_type") if arch else None
    if isinstance(pooling, int) and pooling == POOLING_RANK:
        return out("reranker", f"GGUF {arch}.pooling_type={pooling} (rank)")
    if isinstance(pooling, int) and pooling > 0:
        return out("embedding", f"GGUF {arch}.pooling_type={pooling}")
    if arch == "t5encoder":
        return out("text-encoder", "GGUF t5encoder with no pooling head (diffusion text encoder)")
    if arch in _ENCODER_TEXT_ARCHES:
        return out("embedding", f"GGUF encoder-only text architecture {arch}")
    if arch and (arch in _TTS_ARCHES or "tts" in arch or "text-to-speech" in tags):
        return out("tts", f"GGUF speech-synthesis architecture {arch}")
    if any(t.startswith(_DIFFUSION_TENSOR_PREFIXES) for t in tensors):
        return out("diffusion", f"GGUF diffusion model tensors (architecture {arch})")
    if tensors and all(t.startswith(_VAE_TENSOR_PREFIXES) for t in tensors):
        return out("vae", "GGUF autoencoder tensors only")
    if arch and f"{arch}.block_count" in kv and "tokenizer.ggml.model" in kv:
        return out("llm", f"GGUF decoder {arch} with a tokenizer")
    return out("undetermined", f"GGUF architecture {arch!r} matches no census rule")


def _tensor_kind(tensors: list[str], meta: dict, label: str) -> Implied:
    spec = str(meta.get("modelspec.architecture", "")).lower()
    if spec.endswith("/lora"):
        return Implied("lora", f"{label} modelspec.architecture={spec!r}")
    if spec.endswith("vae"):
        return Implied("vae", f"{label} modelspec.architecture={spec!r}")
    if any(any(m in t.lower() for m in _LORA_MARKERS) for t in tensors):
        return Implied("lora", f"{label} LoRA tensors")
    if any(t.startswith(_CONTROLNET_MARKERS) for t in tensors):
        return Implied("controlnet", f"{label} ControlNet tensors")
    if any(t.startswith(_DIFFUSION_TENSOR_PREFIXES) for t in tensors) or (
            any(t.startswith("blocks.") for t in tensors)
            and any(t.startswith(("patch_embedding.", "time_embedding.")) for t in tensors)):
        return Implied("diffusion", f"{label} diffusion model tensors")
    if tensors and all(t.startswith(_VAE_TENSOR_PREFIXES) for t in tensors) and \
            any(t.startswith("decoder.") for t in tensors):
        return Implied("vae", f"{label} autoencoder tensors only (encoder/decoder)")
    if any(t.startswith(_TEXT_ENCODER_TENSOR_PREFIXES) for t in tensors) and \
            not any(t.startswith("lm_head.") for t in tensors):
        return Implied("text-encoder", f"{label} text encoder tensors")
    if any(t.startswith("conv_first.") for t in tensors) and \
            any(".rdb" in t or t.startswith("body.") for t in tensors):
        return Implied("upscaler", f"{label} ESRGAN/RRDB upscaler tensors")
    if 0 < len(tensors) <= 8 and any(any(m in t for m in _TI_MARKERS) for t in tensors):
        return Implied("textual-inversion", f"{label} embedding vectors only")
    return Implied("undetermined", f"{label} tensors match no census rule")


def _config_kind(folder: Path) -> Implied:
    index = read_json_file(folder / "model_index.json")
    if index is not None:
        cls = index.get("_class_name", "?")
        return Implied("diffusion", f"diffusers pipeline model_index.json ({cls})")
    if (folder / "adapter_config.json").is_file():
        return Implied("lora", "adapter_config.json (PEFT adapter)")
    cfg = read_json_file(folder / "config.json")
    if cfg is None:
        return Implied("undetermined", "folder without config.json or model_index.json")
    cls = str(cfg.get("_class_name") or "")
    if cls:
        low = cls.lower()
        if "vocoder" in low or "hifigan" in low or "bigvgan" in low:
            return Implied("vocoder", f"diffusers component {cls}")
        if low.startswith("autoencoder") or "vae" in low:
            return Implied("vae", f"diffusers component {cls}")
        if "controlnet" in low:
            return Implied("controlnet", f"diffusers component {cls}")
        if "transformer" in low or "unet" in low or low.endswith("ditmodel"):
            return Implied("diffusion", f"diffusers component {cls}")
        return Implied("undetermined", f"diffusers component {cls} matches no census rule")
    archs = [a for a in cfg.get("architectures") or [] if isinstance(a, str)]
    mtype = str(cfg.get("model_type") or "").lower()
    sentence = (folder / "modules.json").is_file() or \
        (folder / "config_sentence_transformers.json").is_file()
    for a in archs:
        if "tts" in a.lower() or "tts" in mtype:
            return Implied("tts", f"transformers {a}")
        if a.endswith("EncoderModel") and mtype in ("t5", "umt5", "mt5"):
            return Implied("text-encoder", f"transformers {a} (model_type {mtype})")
        if a.startswith(("CLIPTextModel", "T5EncoderModel")):
            return Implied("text-encoder", f"transformers {a}")
        if a.endswith(("ForCausalLM", "LMHeadModel", "ForConditionalGeneration")):
            return Implied("llm", f"transformers {a}")
        labels = cfg.get("id2label")
        if a.endswith("ForSequenceClassification") and (
                sentence or (isinstance(labels, dict) and len(labels) == 1)):
            return Implied("reranker", f"transformers {a} with a single relevance score")
        if sentence or (a.endswith("Model") and mtype in (
                "bert", "xlm-roberta", "roberta", "nomic_bert", "modernbert", "mpnet")):
            return Implied("embedding", f"transformers {a}"
                           + (" with sentence-transformers files" if sentence else ""))
    return Implied("undetermined", f"config.json architectures={archs} model_type={mtype!r}"
                   " match no census rule")


def implied_kind(path: Path) -> Implied:
    """The kind *path*'s metadata implies. A metadata read failure becomes kind
    ``undetermined`` with the reason as evidence."""
    try:
        if path.is_dir():
            return _config_kind(path)
        suffix = path.suffix.lower()
        if suffix == ".gguf":
            return _gguf_kind(read_gguf(path))
        if suffix in (".safetensors", ".sft"):
            meta = read_safetensors(path)
            return _tensor_kind(meta["tensors"], meta["metadata"], "safetensors")
        if suffix in (".pth", ".pt", ".ckpt", ".bin"):
            strings = read_torch_pickle_strings(path)
            return _tensor_kind([s for s in strings if "." in s and " " not in s], {},
                                "PyTorch pickle")
        return Implied("undetermined", f"no metadata reader for {suffix} files")
    except MetadataError as e:
        return Implied("undetermined", f"metadata unreadable: {e}")


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


@dataclass
class Entry:
    path: Path
    parts: list[Path] = field(default_factory=list)


def enumerate_models(root: Path) -> tuple[list[Entry], list[tuple[Path, str]]]:
    """``(entries, skipped)``. A folder holding a ``config.json`` or
    ``model_index.json`` is one entry; any other folder is searched. Split GGUF
    parts collapse into their first part. Dot-entries, partial downloads and
    files without a model file suffix are skipped with the reason."""
    entries: list[Entry] = []
    skipped: list[tuple[Path, str]] = []
    splits: dict[tuple[Path, str], Entry] = {}

    def walk(folder: Path) -> None:
        for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
            if p.name.startswith("."):
                skipped.append((p, "hidden"))
                continue
            if p.is_dir():
                if (p / "config.json").is_file() or (p / "model_index.json").is_file():
                    entries.append(Entry(p))
                else:
                    walk(p)
                continue
            low = p.name.lower()
            if low.endswith(PARTIAL_SUFFIXES):
                skipped.append((p, "partial download"))
                continue
            if not low.endswith(MODEL_FILE_SUFFIXES):
                skipped.append((p, f"not a model file suffix ({p.suffix or 'none'})"))
                continue
            m = _SPLIT_RE.search(p.name)
            if m:
                key = (p.parent, p.name[:m.start()].lower())
                ent = splits.get(key)
                if ent is None:
                    ent = splits[key] = Entry(p)
                    entries.append(ent)
                ent.parts.append(p)
                if int(m.group(1)) == 1:
                    ent.path = p
                continue
            entries.append(Entry(p))

    walk(root)
    return entries, skipped


# --------------------------------------------------------------------------
# localm side
# --------------------------------------------------------------------------

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def _norm(p) -> str:
    return os.path.normcase(os.path.realpath(str(p)))


def _localm_env(home: Path) -> dict:
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home)
    env["LOCALM_CRASH_WATCHDOG"] = "off"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(cmd: list[str], home: Path, timeout: float) -> tuple[int, str]:
    proc = subprocess.run(cmd, env=_localm_env(home), cwd=str(home), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          creationflags=_NO_WINDOW, stdin=subprocess.DEVNULL)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def make_home(work_dir: Path) -> Path:
    """A fresh throwaway LOCALM_HOME under *work_dir* whose config.json turns
    off the update check and bug-report uploads."""
    home = Path(tempfile.mkdtemp(prefix="census-home-", dir=work_dir))
    (home / "config.json").write_text(
        json.dumps({"update_url": "", "bugreport_upload_url": ""}), encoding="utf-8")
    return home


def load_registry_file(home: Path) -> dict:
    """The name -> entry dict of *home*'s registry.json ({} when absent).
    Raises ValueError when the file is not a JSON object."""
    path = home / "registry.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("registry.json is not a JSON object")
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def register_all(python: str, root: Path, entries: list[Entry], home: Path,
                 timeout: float) -> dict[str, str]:
    """Register *root* in place with one ``localm add``, then every entry that
    call did not register with its own ``localm add``. Returns, keyed by
    normalised path, the output of each per-entry call that still registered
    nothing."""
    _run([python, "-m", "localm", "add", str(root), "--no-hash", "--on-duplicate", "skip"],
         home, timeout)
    registered = {_norm(e.get("path", "")) for e in load_registry_file(home).values()}
    refused: dict[str, str] = {}
    for ent in entries:
        if _norm(ent.path) in registered:
            continue
        code, out = _run([python, "-m", "localm", "add", str(ent.path), "--no-hash",
                          "--on-duplicate", "skip"], home, timeout)
        registered = {_norm(e.get("path", "")) for e in load_registry_file(home).values()}
        if _norm(ent.path) not in registered:
            refused[_norm(ent.path)] = f"exit {code}: {' '.join(out.split())[-300:]}"
    return refused


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, key: Optional[str] = None, timeout: float = 10.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _descendants(pid: int) -> dict:
    """``{pid: psutil.Process}`` of *pid*'s live descendants ({} when psutil is
    missing or *pid* is gone)."""
    try:
        import psutil
        return {c.pid: c for c in psutil.Process(pid).children(recursive=True)}
    except Exception:
        return {}


def _kill_tree(proc: subprocess.Popen, seen: dict) -> list[int]:
    """Stop *proc* and its process tree, plus every process in *seen*
    (descendants recorded while it ran) that is still the same process (psutil
    compares its creation time, so a reused PID is left alone). Returns the
    PIDs still running afterwards."""
    seen = {**seen, **_descendants(proc.pid)}
    if proc.poll() is None:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           timeout=60, creationflags=_NO_WINDOW)
        else:
            import signal
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
        proc.wait(timeout=60)
    survivors = []
    for child in seen.values():
        try:
            if child.is_running():
                child.kill()
                child.wait(timeout=15)
        except Exception:
            pass
        try:
            if child.is_running():
                survivors.append(child.pid)
        except Exception:
            pass
    if proc.poll() is None:
        survivors.append(proc.pid)
    return survivors


def fetch_api_models(python: str, home: Path, timeout: float, info: dict) -> list[dict]:
    """``GET /api/models`` rows from a ``localm gui --no-model --no-browser
    --isolated`` server started on a free loopback port with *home*, sent with
    an owner key made in *home* by ``localm key generate``. The server is
    stopped by its own PID tree before this returns. Raises RuntimeError when no
    key could be made, or the server does not answer with a model list within
    *timeout* seconds, or keeps refusing the key. PIDs of the server tree still
    running after the stop are listed in ``info["leftover_pids"]``."""
    code, out = _run([python, "-m", "localm", "key", "generate"], home, 120)
    key_path = home / "auth.key"
    if code != 0 or not key_path.is_file():
        raise RuntimeError(f"localm key generate failed (exit {code}): {' '.join(out.split())[-300:]}")
    key = key_path.read_text(encoding="utf-8").strip()
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = home / "census-gui.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [python, "-m", "localm", "gui", "--no-model", "--no-browser", "--isolated", "--new",
             "--host", "127.0.0.1", "--port", str(port)],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=_localm_env(home), cwd=str(home), creationflags=_NO_WINDOW | _NEW_GROUP,
            start_new_session=sys.platform != "win32")
        seen: dict = {}
        try:
            deadline = time.monotonic() + timeout
            last = "no answer"
            refusals = 0
            while time.monotonic() < deadline:
                seen.update(_descendants(proc.pid))
                if proc.poll() is not None:
                    raise RuntimeError(f"the GUI server exited with code {proc.returncode}"
                                       f" before answering (log: {log_path})")
                try:
                    status, body = _get(f"{base}/api/models", key)
                    if status in (401, 403):
                        refusals += 1
                        if refusals >= 10:
                            raise RuntimeError(f"/api/models refused the owner key (HTTP {status})")
                    if status == 200:
                        data = json.loads(body)
                        rows = data.get("models") if isinstance(data, dict) else None
                        if not isinstance(rows, list):
                            raise RuntimeError(f"/api/models is not a model list: {body[:200]!r}")
                        return [r for r in rows if isinstance(r, dict)]
                    last = f"HTTP {status}"
                except (OSError, ValueError) as e:
                    last = str(e)
                time.sleep(1.0)
            raise RuntimeError(f"/api/models did not answer in {timeout:.0f}s ({last})")
        finally:
            leftover = _kill_tree(proc, seen)
            if leftover:
                info["leftover_pids"] = leftover


def localm_identity(python: str, home: Path) -> str:
    """``"<version> <path of the localm package>"`` as *python* imports it."""
    code, out = _run([python, "-c", "import localm; print(getattr(localm, '__version__', '?'),"
                      " localm.__file__)"], home, 120)
    return out.strip() if code == 0 else f"unknown (exit {code}: {out.strip()[-200:]})"


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

@dataclass
class Row:
    entry: str
    implied: Implied
    name: str = ""
    registry_type: str = ""
    registry_arch: Optional[str] = None
    api_type: Optional[str] = None
    verdict: str = "PASS"
    problems: list[str] = field(default_factory=list)


def judge(rel: str, implied: Implied, name: Optional[str], entry: Optional[dict],
          api_row: Optional[dict], refusal: Optional[str]) -> Row:
    """One census row. ``verdict`` is PASS or the code of the first failed
    check (UNDETERMINED, NOT-REGISTERED, UNKNOWN, MISMATCH, NOT-LISTED); every
    failed check is listed in ``problems``."""
    row = Row(rel, implied, name or "")
    fails: list[tuple[str, str]] = []
    if implied.kind == "undetermined":
        fails.append(("UNDETERMINED", implied.evidence))
    if entry is None:
        fails.append(("NOT-REGISTERED", f"localm add registered nothing ({refusal or 'no output'})"))
    else:
        rtype = str(entry.get("model_type") or "")
        row.registry_type = rtype or "(none)"
        row.registry_arch = entry.get("architecture")
        if rtype in ("", "unknown"):
            fails.append(("UNKNOWN", f"localm type {rtype or '(none)'!r}"))
        elif implied.allowed and rtype not in implied.allowed:
            fails.append(("MISMATCH", f"localm type {rtype!r}, metadata implies {implied.kind}"
                                      f" ({' or '.join(implied.allowed)})"))
        if rtype == "embedding" and implied.kind == "reranker" and not entry.get("reranker"):
            fails.append(("MISMATCH", "a reranker registered without the reranker flag"))
        if rtype == "embedding" and implied.kind == "embedding" and entry.get("reranker"):
            fails.append(("MISMATCH", "an embedding model registered with the reranker flag"))
        if implied.architecture and row.registry_arch and \
                str(row.registry_arch) != implied.architecture:
            fails.append(("MISMATCH", f"localm architecture {row.registry_arch!r}, GGUF header"
                                      f" {implied.architecture!r}"))
        if api_row is None:
            fails.append(("NOT-LISTED", "registered but absent from GET /api/models"))
        else:
            row.api_type = str(api_row.get("model_type"))
            if row.api_type != rtype:
                fails.append(("MISMATCH", f"/api/models type {row.api_type!r} differs from the"
                                          f" registry's {rtype!r}"))
    if fails:
        row.verdict = fails[0][0]
        row.problems = [f"{code}: {text}" for code, text in fails]
    return row


def census(root: Path, python: str, work_dir: Optional[Path], timeout: float,
           metadata_only: bool, keep: bool, info: Optional[dict] = None
           ) -> tuple[list[Row], dict]:
    """Run the census over *root*. Returns ``(rows, info)``; *info* (a fresh
    dict when None) is filled as the run goes, so a caller still holds the
    throwaway home's path and deletion outcome after an exception. Raises
    RuntimeError when it cannot run."""
    info = {} if info is None else info
    if not root.is_dir():
        raise RuntimeError(f"no such folder: {root}")
    entries, skipped = enumerate_models(root)
    if not entries:
        raise RuntimeError(f"no model files or folders in {root}")
    info.update({"models_dir": str(root), "entries": len(entries),
                 "skipped": [f"{p.name} ({why})" for p, why in skipped]})
    implied = {_norm(e.path): implied_kind(e.path) for e in entries}

    def rel(p: Path) -> str:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            return str(p)

    if metadata_only:
        return [Row(rel(e.path), implied[_norm(e.path)],
                    verdict="UNDETERMINED" if implied[_norm(e.path)].kind == "undetermined" else "-")
                for e in entries], info

    if work_dir is None:
        raise RuntimeError("--work-dir is required unless --metadata-only")
    work_dir.mkdir(parents=True, exist_ok=True)
    home = make_home(work_dir)
    info["home"] = str(home)
    try:
        info["localm"] = localm_identity(python, home)
        refused = register_all(python, root, entries, home, timeout)
        registry = load_registry_file(home)
        api_rows = {str(r.get("name")): r for r in fetch_api_models(python, home, timeout, info)}
        by_path: dict[str, list[str]] = {}
        for name, entry in registry.items():
            by_path.setdefault(_norm(entry.get("path", "")), []).append(name)
        rows = []
        seen = set()
        for e in entries:
            key = _norm(e.path)
            for name in by_path.get(key) or [None]:
                rows.append(judge(rel(e.path), implied[key], name,
                                  registry.get(name) if name else None,
                                  api_rows.get(name) if name else None, refused.get(key)))
                seen.add(name)
        for name, entry in sorted(registry.items()):
            if name in seen:
                continue
            p = Path(str(entry.get("path", "")))
            imp = implied_kind(p) if p.exists() else \
                Implied("undetermined", "registered path does not exist")
            rows.append(judge(rel(p), imp, name, entry, api_rows.get(name), None))
        return rows, info
    finally:
        if not keep:
            shutil.rmtree(home, ignore_errors=True)
            info["home_deleted"] = not home.exists()


def print_table(rows: list[Row]) -> None:
    head = ("VERDICT", "ENTRY", "IMPLIED", "LOCALM TYPE", "ARCH", "EVIDENCE / PROBLEMS")
    table = []
    for r in rows:
        label = r.entry
        if r.name and r.name != Path(r.entry).stem and r.name != r.entry:
            label += f" [{r.name}]"
        table.append((r.verdict, label, r.implied.kind, r.registry_type or "-",
                      str(r.registry_arch or r.implied.architecture or "-"),
                      "; ".join(r.problems) if r.problems else r.implied.evidence))
    widths = [max(len(x[i]) for x in [head, *table]) for i in range(5)]
    for line in [head, *table]:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(line[:5])) + "  " + line[5])


def rows_json(rows: list[Row], info: dict) -> dict:
    return {"info": info, "rows": [
        {"entry": r.entry, "name": r.name, "implied": r.implied.kind,
         "allowed": list(r.implied.allowed), "evidence": r.implied.evidence,
         "architecture": r.implied.architecture, "registry_type": r.registry_type,
         "registry_architecture": r.registry_arch, "api_type": r.api_type,
         "verdict": r.verdict, "problems": r.problems} for r in rows]}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", required=True, type=Path, help="folder to census (read only)")
    ap.add_argument("--work-dir", type=Path,
                    help="scratch folder for the throwaway LOCALM_HOME (required unless --metadata-only)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter that runs `python -m localm` (default: this one)")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="seconds allowed for each localm add and for the server to answer")
    ap.add_argument("--metadata-only", action="store_true",
                    help="print the implied kinds without registering anything")
    ap.add_argument("--keep", action="store_true", help="keep the throwaway home")
    ap.add_argument("--json", type=Path, help="also write the rows as JSON to this file")
    args = ap.parse_args(argv)
    info: dict[str, Any] = {}
    try:
        rows, info = census(args.models_dir, args.python, args.work_dir, args.timeout,
                            args.metadata_only, args.keep, info)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as e:
        print(f"model census could not run: {e}")
        _print_leftovers(info)
        return 2
    print(f"model census of {info['models_dir']}: {info['entries']} entries")
    if info.get("localm"):
        print(f"localm: {info['localm']}")
    for s in info["skipped"]:
        print(f"skipped: {s}")
    print_table(rows)
    failing = [r for r in rows if r.verdict not in ("PASS", "-")]
    if args.json:
        args.json.write_text(json.dumps(rows_json(rows, info), indent=2), encoding="utf-8")
    if args.metadata_only:
        print(f"metadata only, nothing registered: {len(failing)} undetermined")
    else:
        print(f"{len(rows) - len(failing)} pass, {len(failing)} fail")
    if _print_leftovers(info):
        return 2
    return 1 if failing else 0


def _print_leftovers(info: dict) -> bool:
    """Print what the run failed to clean up; True when anything is left."""
    left = False
    if info.get("home") and info.get("home_deleted") is False:
        print(f"model census left its throwaway home behind: {info['home']}")
        left = True
    if info.get("leftover_pids"):
        print(f"model census could not stop server process(es): {info['leftover_pids']}")
        left = True
    return left


if __name__ == "__main__":
    raise SystemExit(main())
