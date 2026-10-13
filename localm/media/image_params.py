# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generation parameters inside a PNG: writing them (opt-in) and reading them
back from localm's own images and from foreign ones.

Written: an A1111-style ``parameters`` text chunk (so other tools can read it)
and a ``localm`` chunk holding the full record as JSON.

Read: ``localm`` and ``parameters`` chunks, ComfyUI's ``prompt`` (API graph)
and ``workflow`` chunks, and a localm ``.json`` sidecar. Every reader is
bounded and never raises on hostile input; ``ImageParamsError`` carries a
user-facing reason.
"""

from __future__ import annotations

import json
import re
import struct
import zlib
from typing import Any, Optional

MAX_IMPORT_BYTES = 64 * 1024 * 1024
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_CHUNKS = 4096

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_RECORD_DROP = ("input_image", "elapsed_seconds", "runtime")


class ImageParamsError(ValueError):
    """The bytes are not something parameters can be read from."""


# ------------------------------------------------------------ PNG chunks ---

def _inflate(data: bytes) -> bytes:
    d = zlib.decompressobj()
    out = d.decompress(data, MAX_TEXT_BYTES + 1)
    if len(out) > MAX_TEXT_BYTES or d.unconsumed_tail:
        raise ValueError("a compressed text chunk is larger than the supported limit")
    return out


def read_png_text(data: bytes) -> dict[str, str]:
    """``{keyword: text}`` for every tEXt, zTXt and iTXt chunk of the PNG in
    *data* (a repeated keyword keeps its first value). Raises
    ``ImageParamsError`` when *data* is not a PNG or its chunk list is
    malformed or oversize."""
    if len(data) > MAX_IMPORT_BYTES:
        raise ImageParamsError("the file is too large to read parameters from")
    if not data.startswith(PNG_SIGNATURE):
        raise ImageParamsError("not a PNG image")
    out: dict[str, str] = {}
    pos = len(PNG_SIGNATURE)
    for _ in range(MAX_CHUNKS):
        if pos + 8 > len(data):
            break
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        body_start = pos + 8
        body_end = body_start + length
        if body_end + 4 > len(data):
            raise ImageParamsError("the PNG is truncated")
        if ctype == b"IEND":
            break
        if ctype in (b"tEXt", b"zTXt", b"iTXt"):
            try:
                key, text = _decode_text_chunk(ctype, data[body_start:body_end])
            except (ValueError, zlib.error, UnicodeDecodeError):
                key, text = None, None
            if key and key not in out and text is not None:
                out[key] = text
        pos = body_end + 4
    return out


def _decode_text_chunk(ctype: bytes, body: bytes) -> tuple[str, str]:
    key_raw, sep, rest = body.partition(b"\x00")
    if not sep:
        raise ValueError("no keyword terminator")
    key = key_raw.decode("latin-1")
    if ctype == b"tEXt":
        return key, rest.decode("latin-1")
    if ctype == b"zTXt":
        if not rest or rest[0] != 0:
            raise ValueError("unknown zTXt compression method")
        return key, _inflate(rest[1:]).decode("latin-1")
    if len(rest) < 2:
        raise ValueError("short iTXt chunk")
    compressed, method = rest[0], rest[1]
    _lang, sep, rest = rest[2:].partition(b"\x00")
    if not sep:
        raise ValueError("bad iTXt language tag")
    _trans, sep, text = rest.partition(b"\x00")
    if not sep:
        raise ValueError("bad iTXt translated keyword")
    if compressed:
        if method != 0:
            raise ValueError("unknown iTXt compression method")
        text = _inflate(text)
    return key, text.decode("utf-8")


# ------------------------------------------------------------ writing -----

def embedded_record(record: dict) -> dict:
    """*record* (the sidecar dict) without what would leak the machine or is
    not a generation parameter: the init-image path, timings and runtime."""
    return {k: v for k, v in record.items()
            if v is not None and k not in _RECORD_DROP}


def _a1111_value(value: Any) -> str:
    return str(value).replace("\n", " ")


def a1111_text(record: dict) -> str:
    """The A1111 ``parameters`` text for *record*: prompt, ``Negative prompt:``
    and the comma-separated settings line."""
    rec = embedded_record(record)
    lines = [str(rec.get("prompt") or "")]
    if rec.get("negative_prompt"):
        lines.append("Negative prompt: " + str(rec["negative_prompt"]))
    sampler = rec.get("sample_method") or rec.get("sampler")
    pairs = [
        ("Steps", rec.get("steps")),
        ("Sampler", sampler),
        ("Schedule type", rec.get("scheduler")),
        ("CFG scale", rec.get("cfg")),
        ("Distilled CFG Scale", rec.get("guidance")),
        ("Seed", rec.get("seed")),
        ("Size", f"{rec['width']}x{rec['height']}" if rec.get("width") and rec.get("height") else None),
        ("Model", rec.get("model")),
        ("Denoising strength", rec.get("denoise")),
    ]
    loras = rec.get("loras") or []
    if loras:
        pairs.append(("Lora", ", ".join(f"{x.get('name')}:{x.get('strength')}"
                                          for x in loras if isinstance(x, dict))))
    settings = ", ".join(f"{k}: {_a1111_value(v)}" for k, v in pairs if v not in (None, ""))
    if settings:
        lines.append(settings)
    return "\n".join(lines)


def png_text_entries(record: dict) -> list[tuple[str, str]]:
    """The ``(keyword, text)`` chunks to embed for *record*."""
    rec = embedded_record(record)
    return [("parameters", a1111_text(rec)),
            ("localm", json.dumps(rec, ensure_ascii=False, separators=(",", ":")))]


# ------------------------------------------------------------ reading -----

_NEG = "Negative prompt:"
_PAIR_RE = re.compile(r'\s*([\w][\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
_LORA_TAG_RE = re.compile(r"<lora:([^:>]+)(?::([-+]?\d*\.?\d+))?[^>]*>")


def _num(value: Any, cast) -> Optional[Any]:
    try:
        return cast(value)
    except (TypeError, ValueError, OverflowError):
        return None


def parse_a1111(text: str) -> dict:
    """Parameters from an A1111 ``parameters`` string. Raises
    ``ImageParamsError`` when it has no settings line."""
    lines = text.replace("\r\n", "\n").split("\n")
    settings_at = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith("Steps:"):
            settings_at = i
            break
    if settings_at is None:
        raise ImageParamsError("the parameters text has no settings line")
    head = lines[:settings_at]
    neg_at = next((i for i, ln in enumerate(head) if ln.startswith(_NEG)), None)
    if neg_at is None:
        prompt, negative = "\n".join(head), ""
    else:
        prompt = "\n".join(head[:neg_at])
        negative = "\n".join([head[neg_at][len(_NEG):].lstrip()] + head[neg_at + 1:])
    pairs: dict[str, str] = {}
    for m in _PAIR_RE.finditer(lines[settings_at]):
        val = m.group(2).strip()
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        pairs[m.group(1).strip()] = val
    loras = []
    for m in _LORA_TAG_RE.finditer(prompt):
        loras.append({"name": m.group(1), "strength": _num(m.group(2), float) or 1.0})
    prompt = _LORA_TAG_RE.sub("", prompt).strip()
    out: dict[str, Any] = {"prompt": prompt, "negative_prompt": negative.strip()}
    out["steps"] = _num(pairs.get("Steps"), int)
    out["seed"] = _num(pairs.get("Seed"), int)
    out["cfg"] = _num(pairs.get("CFG scale"), float)
    out["guidance"] = _num(pairs.get("Distilled CFG Scale"), float)
    out["denoise"] = _num(pairs.get("Denoising strength"), float)
    size = re.fullmatch(r"(\d+)x(\d+)", pairs.get("Size", ""))
    if size:
        out["width"], out["height"] = int(size.group(1)), int(size.group(2))
    out["model"] = pairs.get("Model") or None
    out["sampler"] = pairs.get("Sampler") or None
    out["scheduler"] = pairs.get("Schedule type") or None
    if loras:
        out["loras"] = loras
    return {k: v for k, v in out.items() if v is not None}


_SAMPLER_NODES = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced")
_MODEL_KEYS = ("ckpt_name", "unet_name", "model_name")


def _node(graph: dict, link: Any) -> Optional[dict]:
    if isinstance(link, list) and link and isinstance(link[0], (str, int)):
        node = graph.get(str(link[0]))
        return node if isinstance(node, dict) else None
    return None


def _follow_text(graph: dict, link: Any) -> tuple[Optional[str], Optional[float]]:
    """The CLIPTextEncode text reached from conditioning *link* (through
    guidance and conditioning wrappers) and a FluxGuidance value passed on the
    way. Bounded and cycle-safe."""
    guidance = None
    seen: set = set()
    for _ in range(12):
        node = _node(graph, link)
        node_id = str(link[0]) if isinstance(link, list) and link else None
        if node is None or node_id in seen:
            return None, guidance
        seen.add(node_id)
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        ctype = str(node.get("class_type") or "")
        if ctype == "FluxGuidance" and guidance is None:
            guidance = _num(inputs.get("guidance"), float)
        for key in ("text", "text_g", "prompt"):
            if isinstance(inputs.get(key), str):
                return inputs[key], guidance
        link = inputs.get("conditioning")
        if link is None:
            return None, guidance
    return None, guidance


def parse_comfy_prompt(graph: Any) -> dict:
    """Parameters from a ComfyUI API-format ``prompt`` graph (a dict of node
    id to ``{class_type, inputs}``): sampler settings, the prompts it
    conditions on, latent size, the main model and LoRAs. Raises
    ``ImageParamsError`` when no sampler node is found."""
    if not isinstance(graph, dict):
        raise ImageParamsError("the ComfyUI prompt is not a node graph")
    sampler = None
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") in _SAMPLER_NODES:
            sampler = node
            break
    if sampler is None:
        raise ImageParamsError("the ComfyUI graph has no sampler node")
    inputs = sampler.get("inputs") if isinstance(sampler.get("inputs"), dict) else {}
    out: dict[str, Any] = {}
    positive, guidance = _follow_text(graph, inputs.get("positive"))
    negative, _ = _follow_text(graph, inputs.get("negative"))
    out["prompt"] = positive or ""
    out["negative_prompt"] = negative or ""
    out["guidance"] = guidance
    seed = inputs.get("seed", inputs.get("noise_seed"))
    out["seed"] = _num(seed, int) if not isinstance(seed, list) else None
    out["steps"] = _num(inputs.get("steps"), int) if not isinstance(inputs.get("steps"), list) else None
    out["cfg"] = _num(inputs.get("cfg"), float) if not isinstance(inputs.get("cfg"), list) else None
    out["denoise"] = _num(inputs.get("denoise"), float) if not isinstance(inputs.get("denoise"), list) else None
    out["sampler"] = inputs.get("sampler_name") if isinstance(inputs.get("sampler_name"), str) else None
    out["scheduler"] = inputs.get("scheduler") if isinstance(inputs.get("scheduler"), str) else None
    latent = _node(graph, inputs.get("latent_image"))
    if latent and isinstance(latent.get("inputs"), dict):
        out["width"] = _num(latent["inputs"].get("width"), int)
        out["height"] = _num(latent["inputs"].get("height"), int)
    loras = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        ctype = str(node.get("class_type") or "")
        if ctype in ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader",
                     "unCLIPCheckpointLoader") and not out.get("model"):
            for key in _MODEL_KEYS:
                if isinstance(ins.get(key), str):
                    out["model"] = ins[key]
                    break
        elif ctype in ("LoraLoader", "LoraLoaderModelOnly") and isinstance(ins.get("lora_name"), str):
            loras.append({"name": ins["lora_name"],
                          "strength": _num(ins.get("strength_model"), float) or 1.0})
    if loras:
        out["loras"] = loras
    return {k: v for k, v in out.items() if v is not None}


_MAX_STR = 32000
_STR_KEYS = ("prompt", "negative_prompt", "model", "sampler", "scheduler", "sample_method")
_INT_KEYS = ("seed", "steps", "width", "height")
_FLOAT_KEYS = ("cfg", "guidance", "denoise")


def _str(value: Any) -> Optional[str]:
    return value[:_MAX_STR] if isinstance(value, str) else None


def _from_localm(blob: Any) -> dict:
    """The known keys of a localm record with their types checked; anything
    else (unknown keys, wrong types, local paths) is dropped."""
    if not isinstance(blob, dict):
        raise ImageParamsError("the localm parameters are not an object")
    out: dict[str, Any] = {}
    for key in _STR_KEYS:
        if _str(blob.get(key)) is not None:
            out[key] = _str(blob[key])
    for key in _INT_KEYS:
        if isinstance(blob.get(key), (int, float)) and not isinstance(blob.get(key), bool):
            val = _num(blob[key], int)
            if val is not None:
                out[key] = val
    for key in _FLOAT_KEYS:
        if isinstance(blob.get(key), (int, float)) and not isinstance(blob.get(key), bool):
            val = _num(blob[key], float)
            if val is not None and val == val and abs(val) != float("inf"):
                out[key] = val
    comps = blob.get("components")
    if isinstance(comps, dict):
        out["components"] = {str(k)[:64]: v[:512] for k, v in comps.items() if isinstance(v, str)}
    loras = blob.get("loras")
    if isinstance(loras, list):
        clean = []
        for item in loras[:16]:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                strength = item.get("strength", 1.0)
                strength = float(strength) if isinstance(strength, (int, float))                     and not isinstance(strength, bool) else 1.0
                clean.append({"name": item["name"][:512], "strength": strength})
        if clean:
            out["loras"] = clean
    return out


def params_from_bytes(data: bytes) -> dict:
    """``{"source", "params", "notes"}`` for an uploaded image or sidecar.

    *source* is ``localm``, ``a1111``, ``comfyui`` or ``sidecar``. *params*
    uses the sidecar's key names (``prompt``, ``negative_prompt``, ``seed``,
    ``steps``, ``cfg``, ``guidance``, ``denoise``, ``width``, ``height``,
    ``model``, ``components``, ``loras``, plus ``sampler``/``scheduler`` when
    present). Raises ``ImageParamsError`` with a reason when nothing is found."""
    if len(data) > MAX_IMPORT_BYTES:
        raise ImageParamsError("the file is too large to read parameters from")
    head = data.lstrip()[:1]
    if head == b"{" and not data.startswith(PNG_SIGNATURE):
        try:
            blob = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, RecursionError) as e:
            raise ImageParamsError(f"the file is not valid JSON: {e}") from e
        return {"source": "sidecar", "params": _from_localm(blob), "notes": []}
    text = read_png_text(data)
    notes: list[str] = []
    if "localm" in text:
        try:
            return {"source": "localm",
                    "params": _from_localm(json.loads(text["localm"])), "notes": notes}
        except (ValueError, RecursionError, ImageParamsError):
            notes.append("The localm parameters in this image could not be read.")
    if "parameters" in text:
        try:
            return {"source": "a1111", "params": _from_localm(parse_a1111(text["parameters"])),
                    "notes": notes}
        except ImageParamsError as e:
            notes.append(str(e) + ".")
    if "prompt" in text:
        try:
            graph = json.loads(text["prompt"])
            return {"source": "comfyui", "params": _from_localm(parse_comfy_prompt(graph)),
                    "notes": notes}
        except (ValueError, RecursionError) as e:
            notes.append(f"The ComfyUI prompt is not valid JSON: {e}.")
        except ImageParamsError as e:
            notes.append(str(e) + ".")
    detail = (" " + " ".join(notes)) if notes else ""
    raise ImageParamsError("this image carries no generation parameters." + detail)
