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
import struct
import zlib
from typing import Any, Optional

MAX_IMPORT_BYTES = 64 * 1024 * 1024
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_TOTAL_TEXT_BYTES = 8 * 1024 * 1024
MAX_CHUNKS = 4096
MAX_PARAMETERS_CHARS = 1024 * 1024
WANTED_KEYWORDS = frozenset({"localm", "parameters", "prompt"})

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_RECORD_DROP = ("input_image", "elapsed_seconds", "runtime")


class ImageParamsError(ValueError):
    """The bytes are not something parameters can be read from."""


# ------------------------------------------------------------ PNG chunks ---

def _inflate(data: bytes, limit: int = MAX_TEXT_BYTES) -> bytes:
    d = zlib.decompressobj()
    out = d.decompress(data, limit + 1)
    if len(out) > limit or d.unconsumed_tail:
        raise ValueError("a compressed text chunk is larger than the supported limit")
    return out


def read_png_text(data: bytes, keys=None) -> dict[str, str]:
    """``{keyword: text}`` for the tEXt, zTXt and iTXt chunks of the PNG in
    *data* (a repeated keyword keeps its first value). With *keys*, only those
    keywords are decoded; every other chunk is skipped unread. The decoded text
    of all chunks together is capped at ``MAX_TOTAL_TEXT_BYTES``: a chunk that
    would pass the cap is skipped. Raises ``ImageParamsError`` when *data* is
    not a PNG or its chunk list is malformed or oversize."""
    if len(data) > MAX_IMPORT_BYTES:
        raise ImageParamsError("the file is too large to read parameters from")
    if not data.startswith(PNG_SIGNATURE):
        raise ImageParamsError("not a PNG image")
    out: dict[str, str] = {}
    total = 0
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
            body = data[body_start:body_end]
            key_raw = body.partition(b"\x00")[0].decode("latin-1")
            if key_raw and key_raw not in out and (keys is None or key_raw in keys):
                try:
                    key, text = _decode_text_chunk(ctype, body, MAX_TOTAL_TEXT_BYTES - total)
                except (ValueError, zlib.error, UnicodeDecodeError):
                    key, text = None, None
                if key and text is not None:
                    total += len(text)
                    out[key] = text
        pos = body_end + 4
    return out


def _decode_text_chunk(ctype: bytes, body: bytes, budget: int) -> tuple[str, str]:
    key_raw, sep, rest = body.partition(b"\x00")
    if not sep:
        raise ValueError("no keyword terminator")
    key = key_raw.decode("latin-1")
    if ctype == b"tEXt":
        if len(rest) > budget:
            raise ValueError("text chunk over the remaining budget")
        return key, rest.decode("latin-1")
    if ctype == b"zTXt":
        if not rest or rest[0] != 0:
            raise ValueError("unknown zTXt compression method")
        return key, _inflate(rest[1:], min(MAX_TEXT_BYTES, budget)).decode("latin-1")
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
        text = _inflate(text, min(MAX_TEXT_BYTES, budget))
    elif len(text) > budget:
        raise ValueError("text chunk over the remaining budget")
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
_KEY_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ -/")
_LORA_OPEN = "<lora:"


def _num(value: Any, cast) -> Optional[Any]:
    """*value* as a finite *cast* number, or None when it is not one."""
    if isinstance(value, bool):
        return None
    try:
        out = cast(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if cast is float and (out != out or out in (float("inf"), float("-inf"))):
        return None
    if cast is int and abs(out) >= 2 ** 63:
        return None
    return out


def _split_settings(line: str) -> dict:
    """The ``Key: value`` pairs of an A1111 settings line. A value is either a
    double-quoted string (backslash escapes kept) or the text up to the next
    comma. One left-to-right pass, no backtracking."""
    pairs: dict[str, str] = {}
    n = len(line)
    i = 0
    while i < n:
        while i < n and line[i] in " \t,":
            i += 1
        start = i
        while i < n and line[i] in _KEY_CHARS:
            i += 1
        if i < n and line[i] == ":" and i > start:
            key = line[start:i].strip()
            i += 1
            while i < n and line[i] in " \t":
                i += 1
            if i < n and line[i] == '"':
                j = i + 1
                while j < n and line[j] != '"':
                    j += 2 if line[j] == "\\" else 1
                pairs[key] = line[i + 1:min(j, n)]
                i = j + 1
            else:
                j = line.find(",", i)
                j = n if j < 0 else j
                pairs[key] = line[i:j].strip()
                i = j
        else:
            j = line.find(",", i)
            i = n if j < 0 else j + 1
    return pairs


def _strip_lora_tags(prompt: str) -> tuple[str, list]:
    """*prompt* without its ``<lora:name:weight>`` tags, and the tags as
    ``[{"name", "strength"}]``. A tag with no closing ``>`` is left alone."""
    loras = []
    out = []
    pos = 0
    while True:
        start = prompt.find(_LORA_OPEN, pos)
        if start < 0:
            break
        end = prompt.find(">", start)
        if end < 0:
            break
        parts = prompt[start + len(_LORA_OPEN):end].split(":")
        name = parts[0].strip()
        if name:
            strength = _num(parts[1], float) if len(parts) > 1 else None
            loras.append({"name": name, "strength": 1.0 if strength is None else strength})
        out.append(prompt[pos:start])
        pos = end + 1
    out.append(prompt[pos:])
    return "".join(out).strip(), loras


def parse_a1111(text: str) -> dict:
    """Parameters from an A1111 ``parameters`` string. Raises
    ``ImageParamsError`` when it has no settings line or is over
    ``MAX_PARAMETERS_CHARS``."""
    if len(text) > MAX_PARAMETERS_CHARS:
        raise ImageParamsError("the parameters text is too long")
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
    pairs = _split_settings(lines[settings_at])
    prompt, loras = _strip_lora_tags(prompt)
    out: dict[str, Any] = {"prompt": prompt, "negative_prompt": negative.strip()}
    out["steps"] = _num(pairs.get("Steps"), int)
    out["seed"] = _num(pairs.get("Seed"), int)
    out["cfg"] = _num(pairs.get("CFG scale"), float)
    out["guidance"] = _num(pairs.get("Distilled CFG Scale"), float)
    out["denoise"] = _num(pairs.get("Denoising strength"), float)
    width, sep, height = pairs.get("Size", "").partition("x")
    if sep and width.isdigit() and height.isdigit() and len(width) < 7 and len(height) < 7:
        out["width"], out["height"] = int(width), int(height)
    out["model"] = pairs.get("Model") or None
    out["sampler"] = pairs.get("Sampler") or None
    out["scheduler"] = pairs.get("Schedule type") or None
    if loras:
        out["loras"] = loras
    return {k: v for k, v in out.items() if v is not None}


_SAMPLER_NODES = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced")
_MODEL_KEYS = ("ckpt_name", "unet_name", "model_name")
_MAX_HOPS = 12


def _node(graph: dict, link: Any) -> Optional[dict]:
    if isinstance(link, list) and link and isinstance(link[0], (str, int)):
        node = graph.get(str(link[0]))
        return node if isinstance(node, dict) else None
    return None


def _inputs(node: Optional[dict]) -> dict:
    if node is not None and isinstance(node.get("inputs"), dict):
        return node["inputs"]
    return {}


def _follow(graph: dict, link: Any, keys: tuple) -> tuple[Optional[dict], Optional[float]]:
    """Walk from *link* through nodes that pass a conditioning-like input on
    (the first of *keys* a node has as a link) to the first node holding text.
    Returns that node's inputs and a FluxGuidance value met on the way.
    Bounded and cycle-safe."""
    guidance = None
    seen: set = set()
    for _ in range(_MAX_HOPS):
        node = _node(graph, link)
        node_id = str(link[0]) if isinstance(link, list) and link else None
        if node is None or node_id in seen:
            return None, guidance
        seen.add(node_id)
        ins = _inputs(node)
        if str(node.get("class_type") or "") == "FluxGuidance" and guidance is None:
            guidance = _num(ins.get("guidance"), float)
        if any(isinstance(ins.get(k), str) for k in ("text", "text_g", "prompt")):
            return ins, guidance
        link = next((ins[k] for k in keys if isinstance(ins.get(k), list)), None)
        if link is None:
            return None, guidance
    return None, guidance


def _text_of(ins: Optional[dict]) -> Optional[str]:
    if ins is None:
        return None
    for key in ("text", "text_g", "prompt"):
        if isinstance(ins.get(key), str):
            return ins[key]
    return None


def _scalar(ins: dict, *names: str) -> Any:
    """The first of *names* in *ins* that holds a plain value (a link is a
    list and is not one)."""
    for name in names:
        value = ins.get(name)
        if value is not None and not isinstance(value, list):
            return value
    return None


def parse_comfy_prompt(graph: Any) -> dict:
    """Parameters from a ComfyUI API-format ``prompt`` graph (a dict of node
    id to ``{class_type, inputs}``): sampler settings (KSampler, or the custom
    sampler nodes through their noise, guider, sampler and sigmas inputs), the
    prompts it conditions on, latent size, the main model and LoRAs. A value
    the graph does not hold is left out. Raises ``ImageParamsError`` when no
    sampler node is found."""
    if not isinstance(graph, dict):
        raise ImageParamsError("the ComfyUI prompt is not a node graph")
    sampler = None
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") in _SAMPLER_NODES:
            sampler = node
            break
    if sampler is None:
        raise ImageParamsError("the ComfyUI graph has no sampler node")
    ins = _inputs(sampler)
    out: dict[str, Any] = {}
    positive_link, negative_link = ins.get("positive"), ins.get("negative")
    cfg = _scalar(ins, "cfg")
    guider = _node(graph, ins.get("guider"))
    if guider is not None:
        gins = _inputs(guider)
        positive_link = gins.get("positive", gins.get("conditioning"))
        negative_link = gins.get("negative")
        cfg = _scalar(gins, "cfg") if cfg is None else cfg
    positive, guidance = _follow(graph, positive_link, ("conditioning", "conditioning_1", "positive"))
    negative, _ = _follow(graph, negative_link, ("conditioning", "conditioning_1", "negative"))
    if _text_of(positive) is not None:
        out["prompt"] = _text_of(positive)
    if _text_of(negative) is not None:
        out["negative_prompt"] = _text_of(negative)
    out["guidance"] = guidance
    noise = _inputs(_node(graph, ins.get("noise")))
    seed = _scalar(ins, "seed", "noise_seed")
    out["seed"] = _num(seed if seed is not None else _scalar(noise, "noise_seed", "seed"), int)
    sigmas = _inputs(_node(graph, ins.get("sigmas")))
    steps = _scalar(ins, "steps")
    out["steps"] = _num(steps if steps is not None else _scalar(sigmas, "steps"), int)
    out["cfg"] = _num(cfg, float)
    denoise = _scalar(ins, "denoise")
    out["denoise"] = _num(denoise if denoise is not None else _scalar(sigmas, "denoise"), float)
    sampler_name = _scalar(ins, "sampler_name")
    if sampler_name is None:
        sampler_name = _scalar(_inputs(_node(graph, ins.get("sampler"))), "sampler_name")
    scheduler = _scalar(ins, "scheduler")
    if scheduler is None:
        scheduler = _scalar(sigmas, "scheduler")
    out["sampler"] = sampler_name if isinstance(sampler_name, str) else None
    out["scheduler"] = scheduler if isinstance(scheduler, str) else None
    latent = _inputs(_node(graph, ins.get("latent_image")))
    out["width"] = _num(_scalar(latent, "width"), int)
    out["height"] = _num(_scalar(latent, "height"), int)
    loras = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        nins = _inputs(node)
        ctype = str(node.get("class_type") or "")
        if ctype in ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader",
                     "unCLIPCheckpointLoader") and not out.get("model"):
            for key in _MODEL_KEYS:
                if isinstance(nins.get(key), str):
                    out["model"] = nins[key]
                    break
        elif ctype in ("LoraLoader", "LoraLoaderModelOnly") and isinstance(nins.get("lora_name"), str):
            strength = _num(_scalar(nins, "strength_model"), float)
            loras.append({"name": nins["lora_name"], "strength": 1.0 if strength is None else strength})
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
    """The known keys of a record with their types checked; anything else
    (unknown keys, wrong types, non-finite or out-of-range numbers) is
    dropped."""
    if not isinstance(blob, dict):
        raise ImageParamsError("the localm parameters are not an object")
    out: dict[str, Any] = {}
    for key in _STR_KEYS:
        if _str(blob.get(key)) is not None:
            out[key] = _str(blob[key])
    for key in _INT_KEYS:
        val = _num(blob.get(key), int) if isinstance(blob.get(key), (int, float)) else None
        if val is not None:
            out[key] = val
    for key in _FLOAT_KEYS:
        val = _num(blob.get(key), float) if isinstance(blob.get(key), (int, float)) else None
        if val is not None:
            out[key] = val
    comps = blob.get("components")
    if isinstance(comps, dict):
        out["components"] = {str(k)[:64]: v[:512] for k, v in comps.items() if isinstance(v, str)}
    loras = blob.get("loras")
    if isinstance(loras, list):
        clean = []
        for item in loras[:16]:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                strength = item.get("strength")
                strength = _num(strength, float) if isinstance(strength, (int, float)) else None
                clean.append({"name": item["name"][:512],
                              "strength": 1.0 if strength is None else strength})
        if clean:
            out["loras"] = clean
    return out


def params_from_bytes(data: bytes) -> dict:
    """``{"source", "params", "notes"}`` for an uploaded image or sidecar.

    *source* is ``localm``, ``a1111``, ``comfyui`` or ``sidecar``. *params*
    uses the sidecar's key names (``prompt``, ``negative_prompt``, ``seed``,
    ``steps``, ``cfg``, ``guidance``, ``denoise``, ``width``, ``height``,
    ``model``, ``components``, ``loras``, plus ``sampler``/``scheduler`` when
    present); a key the file does not hold is absent. Raises
    ``ImageParamsError`` with a reason when nothing is found."""
    if len(data) > MAX_IMPORT_BYTES:
        raise ImageParamsError("the file is too large to read parameters from")
    head = data.lstrip()[:1]
    if head == b"{" and not data.startswith(PNG_SIGNATURE):
        try:
            blob = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, RecursionError) as e:
            raise ImageParamsError(f"the file is not valid JSON: {e}") from e
        return {"source": "sidecar", "params": _from_localm(blob), "notes": []}
    text = read_png_text(data, WANTED_KEYWORDS)
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
            params = _from_localm(parse_comfy_prompt(graph))
            if "prompt" not in params:
                notes.append("No prompt text could be found in the ComfyUI graph.")
            return {"source": "comfyui", "params": params, "notes": notes}
        except (ValueError, RecursionError) as e:
            notes.append(f"The ComfyUI prompt is not valid JSON: {e}.")
        except ImageParamsError as e:
            notes.append(str(e) + ".")
    detail = (" " + " ".join(notes)) if notes else ""
    raise ImageParamsError("this image carries no generation parameters." + detail)
