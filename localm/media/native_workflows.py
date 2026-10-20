# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native workflows: saved presets for the native image, video and music
backends.

A native workflow names the models of a job (the main model, component slots
and LoRAs with strengths), each by registered model name and optionally with a
public download source, plus generation parameters, a name and a description.
User workflows are JSON files under ``home_dir()/workflows/<media>/native/``;
built-in starter workflows (ids ``builtin:<slug>``) come from this module and
cannot be renamed or deleted.

``resolve`` turns a workflow into what a generate request sends
(``native_models``, ``loras``) and lists every model that is not registered
with the download the page can offer for it.
"""

from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path
from typing import Optional

FORMAT = "localm-native-workflow"
VERSION = 1
MEDIA_TYPES = ("image", "video", "music")
BUILTIN_PREFIX = "builtin:"

MAX_NAME = 100
MAX_DESCRIPTION = 2000
MAX_TEXT = 8000
MAX_FILE_BYTES = 512 * 1024
MAX_WORKFLOWS = 500
MAX_MODEL_NAME = 200

SOURCE_MODEL_TYPES = ("diffusion-unet", "text-encoder", "vae", "lora", "llm", "unknown",
                      "embedding", "upscaler")

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_PULL_NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")

_TEXT = ("str", MAX_TEXT)


def _num(kind: str, lo: float, hi: float) -> tuple:
    return (kind, lo, hi)


def is_number(value) -> bool:
    """True for an int or float that is finite as a float (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


_COMMON = {"steps": _num("int", 1, 150), "cfg": _num("float", 0.0, 30.0),
           "sample_method": ("sampler",), "scheduler": ("scheduler",)}

PARAMS = {
    "image": dict(_COMMON, prompt=_TEXT, negative_prompt=_TEXT,
                  guidance=_num("float", 0.0, 100.0), width=_num("int", 64, 2048),
                  height=_num("int", 64, 2048), clip_skip=_num("int", 1, 12)),
    "video": dict(_COMMON, prompt=_TEXT, negative_prompt=_TEXT,
                  width=_num("int", 64, 2048), height=_num("int", 64, 2048),
                  seconds=_num("float", 0.1, 20.0), fps=_num("int", 1, 60),
                  flow_shift=_num("float", 0.0, 100.0)),
    "music": {"tags": _TEXT, "lyrics": _TEXT, "duration_seconds": _num("float", 1.0, 3600.0),
              "steps": _num("int", 1, 150), "cfg": _num("float", 0.0, 30.0),
              "shift": _num("float", 0.0, 100.0)},
}

_media_locks: dict[str, threading.Lock] = {}
_media_locks_guard = threading.Lock()


def lock_for(media: str) -> threading.Lock:
    """The lock that serialises every change to *media*'s native workflows."""
    with _media_locks_guard:
        lock = _media_locks.get(media)
        if lock is None:
            lock = _media_locks[media] = threading.Lock()
        return lock


class WorkflowNotFound(LookupError):
    """No workflow has the requested id."""


def slot_keys(media: str) -> tuple:
    """The model slots a *media* workflow may name."""
    from localm.media.native_models import media_keys
    return media_keys(media)


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

def _text(value, what: str, limit: int, *, required: bool = False,
          strip: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(f"{what} must be text.")
    if strip:
        value = value.strip()
    if len(value) > limit:
        raise ValueError(f"{what} is longer than {limit} characters.")
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in value):
        raise ValueError(f"{what} contains a control character.")
    if required and not value.strip():
        raise ValueError(f"{what} is empty.")
    return value


def check_model_name(value, what: str) -> str:
    """*value* as a registered model name a workflow may carry: text of 1 to
    ``MAX_MODEL_NAME`` characters without a control character or a path
    separator. Raises ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what}: the model name is empty.")
    value = value.strip()
    if (len(value) > MAX_MODEL_NAME or value in (".", "..")
            or any(ord(ch) < 32 or ch in "/\\" for ch in value)):
        raise ValueError(f"{what}: '{value[:60]}' is not a usable model name.")
    return value


def _file_name(value, what: str) -> str:
    from localm.model_manager.comfy_resolve import split_model_name
    split = split_model_name(value) if isinstance(value, str) else None
    if split is None or split[0]:
        raise ValueError(f"{what}: the file name must be a plain file name.")
    return split[1]


def check_source(value, what: str) -> Optional[dict]:
    """*value* as a download source ``{repo, file, sha256, size_bytes,
    model_type}`` (None stays None): a HuggingFace ``owner/name`` repository
    and a ``.safetensors``/``.sft``/``.gguf`` file in it, a lowercase hex
    sha256 or None, a positive size or None, a model type from
    ``SOURCE_MODEL_TYPES``. Raises ``ValueError``."""
    from localm.model_manager.comfy_resolve import is_safe_hf_file
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{what}: the source must be an object.")
    unknown = set(value) - {"repo", "file", "sha256", "size_bytes", "model_type"}
    if unknown:
        raise ValueError(f"{what}: unknown source field '{sorted(unknown)[0]}'.")
    repo, file = value.get("repo"), value.get("file")
    if not is_safe_hf_file(repo, file):
        raise ValueError(f"{what}: the source must name a HuggingFace repository and a "
                         ".safetensors, .sft or .gguf file in it.")
    sha = value.get("sha256")
    if sha is not None:
        if not isinstance(sha, str) or not _SHA256_RE.match(sha.lower()):
            raise ValueError(f"{what}: sha256 must be 64 hexadecimal characters.")
        sha = sha.lower()
    size = value.get("size_bytes")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
        raise ValueError(f"{what}: size_bytes must be a positive whole number.")
    mtype = value.get("model_type") or "unknown"
    if mtype not in SOURCE_MODEL_TYPES:
        raise ValueError(f"{what}: unknown model_type '{mtype}'.")
    return {"repo": repo, "file": file, "sha256": sha, "size_bytes": size, "model_type": mtype}


def _comfy_ref(value, what: str) -> Optional[dict]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{what}: 'comfy' must be an object.")
    out = {}
    for key in ("class_type", "input_name"):
        v = value.get(key)
        if not isinstance(v, str) or not v or len(v) > 255:
            raise ValueError(f"{what}: comfy.{key} must be a short text.")
        out[key] = v
    return out


def _model_ref(value, what: str, *, lora: bool = False) -> dict:
    if isinstance(value, str):
        value = {"name": value}
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be an object.")
    allowed = {"name", "file", "source", "comfy"} | ({"strength"} if lora else set())
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{what}: unknown field '{sorted(unknown)[0]}'.")
    out = {"name": check_model_name(value.get("name"), what)}
    source = check_source(value.get("source"), what)
    file = value.get("file")
    if file is not None:
        out["file"] = _file_name(file, what)
    elif source is not None:
        out["file"] = source["file"].rsplit("/", 1)[-1]
    if source is not None:
        out["source"] = source
    comfy = _comfy_ref(value.get("comfy"), what)
    if comfy is not None:
        out["comfy"] = comfy
    if lora:
        strength = value.get("strength", 1.0)
        if not is_number(strength) or not -4.0 <= strength <= 4.0:
            raise ValueError(f"{what}: strength must be a number from -4 to 4.")
        out["strength"] = float(strength)
    return out


def _param(media: str, key: str, value):
    from localm.media.sdcpp import sampling
    spec = PARAMS[media].get(key)
    if spec is None:
        raise ValueError(f"Unknown {media} workflow parameter '{key}'.")
    kind = spec[0]
    if kind == "str":
        return _text(value, f"Parameter '{key}'", spec[1])
    if kind in ("sampler", "scheduler"):
        if not isinstance(value, str):
            raise ValueError(f"Parameter '{key}' must be text.")
        check = sampling.check_sample_method if kind == "sampler" else sampling.check_scheduler
        return check(value) if value else None
    if not is_number(value):
        raise ValueError(f"Parameter '{key}' must be a number.")
    if kind == "int":
        if value != int(value):
            raise ValueError(f"Parameter '{key}' must be a whole number.")
        value = int(value)
    else:
        value = float(value)
    if not spec[1] <= value <= spec[2]:
        raise ValueError(f"Parameter '{key}' must be between {spec[1]:g} and {spec[2]:g}.")
    return value


def parse_workflow(data, media: str) -> dict:
    """*data* checked as a native *media* workflow and normalised:
    ``{"format", "version", "media", "name", "description", "models",
    "loras", "params"}``. A parameter set to None or ``""`` is dropped.
    Raises ``ValueError`` with a user-facing reason."""
    from localm.media.native_models import MAX_LORAS
    if media not in MEDIA_TYPES:
        raise ValueError(f"Unknown media '{media}'.")
    if not isinstance(data, dict):
        raise ValueError("A native workflow must be a JSON object.")
    if data.get("format") != FORMAT:
        raise ValueError(f"Not a localm native workflow (format must be '{FORMAT}').")
    if data.get("version") != VERSION:
        raise ValueError(f"Unsupported native workflow version {data.get('version')!r}.")
    if data.get("media") not in (None, media):
        raise ValueError(f"This is a {data.get('media')} workflow, not a {media} one.")
    unknown = set(data) - {"format", "version", "media", "name", "description", "models",
                           "loras", "params", "id", "builtin"}
    if unknown:
        raise ValueError(f"Unknown workflow field '{sorted(unknown)[0]}'.")
    models = data.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError("'models' must be an object.")
    keys = slot_keys(media)
    out_models = {}
    for slot, ref in models.items():
        if slot not in keys:
            raise ValueError(f"Unknown {media} model slot '{slot}'. Allowed: {', '.join(keys)}.")
        if ref in (None, ""):
            continue
        out_models[slot] = _model_ref(ref, f"Model '{slot}'")
    loras = data.get("loras") or []
    if not isinstance(loras, list):
        raise ValueError("'loras' must be a list.")
    if loras and media == "music":
        raise ValueError("Music workflows take no LoRAs.")
    if len(loras) > MAX_LORAS:
        raise ValueError(f"At most {MAX_LORAS} LoRAs per workflow.")
    out_loras = [_model_ref(x, f"LoRA {i + 1}", lora=True) for i, x in enumerate(loras)]
    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("'params' must be an object.")
    out_params = {}
    for key, value in params.items():
        if value is None or value == "":
            if key not in PARAMS[media]:
                raise ValueError(f"Unknown {media} workflow parameter '{key}'.")
            continue
        value = _param(media, key, value)
        if value is not None:
            out_params[key] = value
    return {"format": FORMAT, "version": VERSION, "media": media,
            "name": _text(data.get("name"), "The workflow name", MAX_NAME, required=True, strip=True),
            "description": _text(data.get("description"), "The description", MAX_DESCRIPTION),
            "models": out_models, "loras": out_loras, "params": out_params}


def load_json_text(text: str):
    """*text* parsed as JSON. Raises ``ValueError`` for text over
    ``MAX_FILE_BYTES``, invalid JSON or nesting too deep."""
    if len(text.encode("utf-8", "surrogatepass")) > MAX_FILE_BYTES:
        raise ValueError(f"The file is larger than {MAX_FILE_BYTES // 1024} KB.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Not valid JSON: {e}") from e
    except RecursionError as e:
        raise ValueError("Not valid JSON: nested too deeply.") from e


# ---------------------------------------------------------------------------
#  Built-in starter workflows
# ---------------------------------------------------------------------------

FLUX_SCHNELL_Q4 = {"repo": "city96/FLUX.1-schnell-gguf", "file": "flux1-schnell-Q4_K_S.gguf",
                   "sha256": "4fd16477b3a5296d0cf722c4b92a9fd7f30d09ac7495826e4465d8de9c9fd973",
                   "size_bytes": 6_783_943_712, "model_type": "diffusion-unet"}
FLUX_DEV_Q4 = {"repo": "city96/FLUX.1-dev-gguf", "file": "flux1-dev-Q4_K_S.gguf",
               "sha256": "75bb19459b5240c9f373b5af527584af15b675867fa142efadf7478d6abbf62b",
               "size_bytes": 6_805_988_640, "model_type": "diffusion-unet"}


def _ref(source: dict, name: str = "") -> dict:
    file = source["file"].rsplit("/", 1)[-1]
    return {"name": name or Path(file).stem, "file": file, "source": dict(source)}


def _download_source(d) -> dict:
    return {"repo": d.repo, "file": d.file, "sha256": d.sha256, "size_bytes": d.size_bytes,
            "model_type": d.model_type}


def _flux_models(main: dict) -> dict:
    from localm.media.sdcpp import pairing
    out = {"model": _ref(main)}
    for slot in ("clip_l", "t5xxl", "vae"):
        d = pairing.suggested_download(slot, "flux")
        out[slot] = _ref(_download_source(d), d.name)
    return out


def _builtin_data() -> list[tuple[str, dict]]:
    from localm.media.koboldcpp import models as ace
    from localm.plugins.builtin.image.backends.native import RECOMMENDED_MODELS
    from localm.plugins.builtin.video.backends.native import RECOMMENDED_VIDEO_MODELS
    turbo = RECOMMENDED_MODELS[0]
    wan = RECOMMENDED_VIDEO_MODELS[0]
    wan_models = {p.role: _ref({"repo": p.repo, "file": p.file, "sha256": p.sha256,
                                "size_bytes": p.size_bytes, "model_type": p.model_type})
                  for p in wan.parts}
    ace_models = {c: _ref({"repo": ace.DEFAULT_REPO, "file": ace.DEFAULT_FILES[c],
                           "sha256": ace.DEFAULT_SHA256[c], "size_bytes": ace.DEFAULT_SIZES[c],
                           "model_type": ace.REGISTRY_TYPES[c]})
                  for c in ace.COMPONENTS}
    return [
        ("image", {"slug": "sd-turbo", "name": "SD-Turbo (fast, 512 px)",
                   "description": "Stable Diffusion Turbo in 2 steps at CFG 1. Small and quick; "
                                  "a good first test.",
                   "models": {"model": _ref({"repo": turbo.repo, "file": turbo.file,
                                             "sha256": turbo.sha256,
                                             "size_bytes": turbo.size_bytes,
                                             "model_type": "diffusion-unet"}, turbo.name)},
                   "params": {"steps": turbo.steps, "cfg": turbo.cfg_scale,
                              "width": turbo.width, "height": turbo.height}}),
        ("image", {"slug": "flux1-schnell-q4", "name": "FLUX.1 schnell (Q4_K_S)",
                   "description": "FLUX.1 schnell GGUF with CLIP-L, T5-XXL fp8 and the FLUX VAE. "
                                  "4 steps at CFG 1.",
                   "models": _flux_models(FLUX_SCHNELL_Q4),
                   "params": {"steps": 4, "cfg": 1.0, "width": 1024, "height": 1024}}),
        ("image", {"slug": "flux1-dev-q4", "name": "FLUX.1 dev (Q4_K_S)",
                   "description": "FLUX.1 dev GGUF with CLIP-L, T5-XXL fp8 and the FLUX VAE. "
                                  "20 steps, guidance 3.5.",
                   "models": _flux_models(FLUX_DEV_Q4),
                   "params": {"steps": 20, "cfg": 1.0, "guidance": 3.5,
                              "width": 1024, "height": 1024}}),
        ("video", {"slug": "wan2.1-t2v-1.3b", "name": "Wan 2.1 text to video 1.3B",
                   "description": "Wan 2.1 T2V 1.3B with the UMT5 encoder and the Wan 2.1 VAE.",
                   "models": wan_models,
                   "params": {"steps": wan.steps, "cfg": wan.cfg_scale,
                              "flow_shift": wan.flow_shift, "width": wan.width,
                              "height": wan.height, "negative_prompt": wan.negative_prompt}}),
        ("music", {"slug": "ace-step-1.5-turbo", "name": "ACE-Step 1.5 turbo (default set)",
                   "description": "The default ACE-Step 1.5 GGUF parts: turbo diffusion model, "
                                  "text encoder, VAE and planner.",
                   "models": ace_models, "params": {}}),
    ]


def builtin_workflows(media: str) -> list[dict]:
    """The built-in starter workflows for *media*, each normalised
    (``parse_workflow``) with ``id`` and ``builtin`` set."""
    out = []
    for m, raw in _builtin_data():
        if m != media:
            continue
        slug = raw.pop("slug")
        wf = parse_workflow(dict(raw, format=FORMAT, version=VERSION, media=m), m)
        out.append(dict(wf, id=BUILTIN_PREFIX + slug, builtin=True))
    return out


# ---------------------------------------------------------------------------
#  Storage
# ---------------------------------------------------------------------------

def native_dir(media: str) -> Path:
    """The directory *media*'s user workflows live in (created on first save)."""
    from localm.config import home_dir
    return home_dir() / "workflows" / media / "native"


def _path_for(media: str, wid: str) -> Path:
    from fastapi import HTTPException

    from localm.pathsafe import confined_name
    if not isinstance(wid, str) or not _ID_RE.match(wid):
        raise WorkflowNotFound(wid)
    try:
        return confined_name(native_dir(media), f"{wid}.json")
    except HTTPException as e:
        raise WorkflowNotFound(wid) from e


def _read_file(media: str, path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        wf = parse_workflow(load_json_text(text), media)
    except (TypeError, AttributeError, KeyError, OverflowError, RecursionError) as e:
        raise ValueError(f"The stored workflow is not valid: {e}") from e
    return dict(wf, id=path.stem, builtin=False)


def list_workflows(media: str) -> list[dict]:
    """Every workflow for *media*: the built-ins first, then the user's, by
    name. Each item is ``{"id", "name", "description", "builtin"}``; a user
    file that cannot be read is listed with ``"error"`` set to the reason."""
    items = [{"id": w["id"], "name": w["name"], "description": w["description"],
              "builtin": True} for w in builtin_workflows(media)]
    d = native_dir(media)
    user = []
    if d.is_dir():
        for p in d.glob("*.json"):
            if not _ID_RE.match(p.stem):
                continue
            try:
                wf = _read_file(media, p)
                user.append({"id": wf["id"], "name": wf["name"],
                             "description": wf["description"], "builtin": False})
            except FileNotFoundError:
                continue
            except (OSError, UnicodeDecodeError, ValueError) as e:
                user.append({"id": p.stem, "name": p.stem, "description": "",
                             "builtin": False, "error": str(e) or type(e).__name__})
    user.sort(key=lambda w: (w["name"].lower(), w["id"]))
    return items + user


def get_workflow(media: str, wid: str) -> dict:
    """The workflow *wid* of *media* (normalised, with ``id`` and
    ``builtin``). Raises ``WorkflowNotFound``, or ``ValueError`` when the
    stored file is not a valid workflow."""
    if isinstance(wid, str) and wid.startswith(BUILTIN_PREFIX):
        for wf in builtin_workflows(media):
            if wf["id"] == wid:
                return wf
        raise WorkflowNotFound(wid)
    path = _path_for(media, wid)
    try:
        return _read_file(media, path)
    except FileNotFoundError as e:
        raise WorkflowNotFound(wid) from e
    except UnicodeDecodeError as e:
        raise ValueError(f"The stored workflow is not UTF-8 text: {e}") from e


def export_data(wf: dict) -> dict:
    """*wf* as the JSON a user downloads (no ``id``/``builtin``)."""
    return {k: wf[k] for k in ("format", "version", "media", "name", "description",
                               "models", "loras", "params")}


def _new_id(media: str, name: str) -> str:
    from localm.pathsafe import is_windows_device_name
    stem = _STEM_RE.sub("-", name).strip("-._")[:60].strip("-._") or "workflow"
    if not stem[0].isalnum():
        stem = "w" + stem
    if is_windows_device_name(stem):
        stem += "-workflow"
    d = native_dir(media)
    candidate, i = stem, 1
    while (d / f"{candidate}.json").exists():
        candidate = f"{stem}-{i}"
        i += 1
    return candidate


def _write(media: str, wid: str, wf: dict) -> None:
    from localm.storekit import atomic_write
    path = _path_for(media, wid)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(export_data(wf), indent=2, ensure_ascii=False))


def attach_known_sources(wf: dict, reg: dict) -> dict:
    """*wf* with each model that has no source given the file name of its
    registry entry and, when localm pins that file (``known_sources``), its
    download source. A pinned source is attached only when the entry records
    no sha256 or the same one."""
    known = None
    refs = [(r, False) for r in wf["models"].values()] + [(r, True) for r in wf["loras"]]
    for ref, is_lora in refs:
        if ref.get("source"):
            continue
        entry = reg.get(ref["name"])
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get("path") or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename:
            continue
        ref.setdefault("file", filename)
        if known is None:
            known = known_sources()
        pin = known.get(filename)
        sha = str(entry.get("sha256") or "").lower()
        if pin and (not sha or sha == pin.get("sha256")):
            ref["source"] = dict(pin, model_type="lora" if is_lora else pin["model_type"])
    return wf


def save_workflow(media: str, data, *, wid: Optional[str] = None,
                  reg: Optional[dict] = None) -> str:
    """Validate *data* and store it. With *wid* naming an existing user
    workflow, that file is replaced; otherwise a new id is made from the name.
    With *reg*, models without a source get one (``attach_known_sources``).
    Returns the id. Caller must hold ``lock_for(media)``. Raises
    ``ValueError`` (invalid, built-in target, too many workflows) or
    ``WorkflowNotFound``."""
    wf = parse_workflow(data, media)
    if reg is not None:
        wf = parse_workflow(attach_known_sources(wf, reg), media)
    if wid is not None:
        if wid.startswith(BUILTIN_PREFIX):
            raise ValueError("A built-in workflow cannot be changed; save it under a new name.")
        if not _path_for(media, wid).is_file():
            raise WorkflowNotFound(wid)
    else:
        d = native_dir(media)
        if d.is_dir() and sum(1 for _ in d.glob("*.json")) >= MAX_WORKFLOWS:
            raise ValueError(f"There are already {MAX_WORKFLOWS} {media} workflows.")
        wid = _new_id(media, wf["name"])
    _write(media, wid, wf)
    return wid


def rename_workflow(media: str, wid: str, name: str) -> None:
    """Set the name of user workflow *wid* (its id stays). Caller must hold
    ``lock_for(media)``. Raises ``ValueError`` or ``WorkflowNotFound``."""
    if isinstance(wid, str) and wid.startswith(BUILTIN_PREFIX):
        raise ValueError("A built-in workflow cannot be renamed.")
    wf = get_workflow(media, wid)
    wf["name"] = _text(name, "The workflow name", MAX_NAME, required=True, strip=True)
    _write(media, wid, wf)


def delete_workflow(media: str, wid: str) -> None:
    """Delete user workflow *wid*. Caller must hold ``lock_for(media)``.
    Raises ``ValueError`` for a built-in or ``WorkflowNotFound``."""
    if isinstance(wid, str) and wid.startswith(BUILTIN_PREFIX):
        raise ValueError("A built-in workflow cannot be deleted.")
    path = _path_for(media, wid)
    try:
        path.unlink()
    except FileNotFoundError as e:
        raise WorkflowNotFound(wid) from e


def import_workflow(media: str, data, *, name: Optional[str] = None) -> tuple[str, dict]:
    """Store *data* as a new *media* workflow: a localm native workflow as is,
    a ComfyUI workflow (API or UI format) converted by
    ``comfy_import.convert``. *name* replaces the stored name when given.
    Returns ``(id, report)``; the report is ``comfy_import``'s for a ComfyUI
    workflow and empty lists otherwise. Caller must hold ``lock_for(media)``.
    Raises ``ValueError``."""
    from localm.media import comfy_import
    report = {"kind": "native", "mapped": [], "implicit": [], "unsupported": [], "notes": []}
    if isinstance(data, dict) and data.get("format") == FORMAT:
        wf = parse_workflow(data, media)
    elif comfy_import.is_comfy_workflow(data):
        try:
            wf, report = comfy_import.convert(data, media)
        except (TypeError, AttributeError, KeyError, OverflowError, RecursionError) as e:
            raise ValueError(f"The ComfyUI workflow could not be read: {e}") from e
    else:
        raise ValueError("Not a localm native workflow or a ComfyUI workflow.")
    if name:
        wf["name"] = name
    return save_workflow(media, wf), report


# ---------------------------------------------------------------------------
#  Resolution
# ---------------------------------------------------------------------------

def _entry_file(entry) -> Optional[Path]:
    from localm.model_manager.registry import _entry_path
    from localm.pathsafe import is_unc_or_device_path
    raw = _entry_path(entry)
    if not raw or is_unc_or_device_path(raw):
        return None
    try:
        p = Path(raw)
        return p if p.is_file() else None
    except OSError:
        return None


def _registered_by_file(reg: dict, filename: str, want_lora: bool) -> Optional[str]:
    for name, entry in reg.items():
        if not isinstance(entry, dict):
            continue
        if want_lora and entry.get("model_type") != "lora":
            continue
        raw = str(entry.get("path") or "").replace("\\", "/").rsplit("/", 1)[-1]
        if raw == filename and _entry_file(entry) is not None:
            return str(name)
    return None


def known_sources() -> dict:
    """``{file name: source}`` for every public file localm already pins a
    download (with sha256) for: the curated ComfyUI table, the component
    downloads, the recommended native models and the built-in workflows."""
    from localm.model_manager.registry import COMFY_MODEL_SOURCES
    from localm.media.sdcpp import pairing
    out: dict = {}
    for wf_media in MEDIA_TYPES:
        for wf in builtin_workflows(wf_media):
            for ref in list(wf["models"].values()) + wf["loras"]:
                if ref.get("source"):
                    out.setdefault(ref["file"], ref["source"])
    for d in pairing.all_downloads():
        out.setdefault(d.filename, _download_source(d))
    for filename, src in COMFY_MODEL_SOURCES.items():
        repo, path = src.spec.split(":", 1)
        out.setdefault(filename, {"repo": repo, "file": path, "sha256": src.sha256,
                                  "size_bytes": src.size_bytes, "model_type": src.model_type})
    return out


def _is_pinned(source: dict, known: dict) -> bool:
    pin = known.get(source["file"].rsplit("/", 1)[-1])
    return bool(pin and pin["repo"] == source["repo"] and pin["file"] == source["file"]
                and source.get("sha256") and pin.get("sha256") == source["sha256"])


def _missing_entry(slot: str, ref: dict, reg: dict, known: dict, taken: set) -> dict:
    from localm.model_manager.comfy_resolve import search_refusal
    filename = ref.get("file") or ref["name"]
    out = {"slot": slot, "name": ref["name"], "filename": filename, "native": True,
           "searchable": False, "source": None}
    src = ref.get("source")
    pull_name = (ref["name"] if _PULL_NAME_RE.match(ref["name"]) and ref["name"] not in reg
                 and ref["name"] not in taken else None)
    if pull_name:
        taken.add(pull_name)
    if src is not None:
        out["source"] = {"repo": src["repo"], "file": src["file"],
                         "spec": f"{src['repo']}:{src['file']}", "sha256": src["sha256"],
                         "size_bytes": src["size_bytes"], "model_type": src["model_type"],
                         "name": pull_name,
                         "origin": "curated" if _is_pinned(src, known) else "workflow"}
        return out
    comfy = ref.get("comfy")
    if comfy and ref.get("file"):
        refusal = search_refusal(ref["file"], comfy["class_type"], comfy["input_name"])
        out.update(class_type=comfy["class_type"], input_name=comfy["input_name"],
                   searchable=refusal is None, source_name=pull_name,
                   reason=refusal.reason if refusal else "",
                   detail=refusal.detail if refusal else "")
    return out


def resolve(wf: dict, media: str, reg: Optional[dict] = None) -> dict:
    """What workflow *wf* sends with a *media* generate request, and what is
    missing: ``{"native_models", "loras", "params", "missing", "notes"}``.

    A model resolves to its registered name when that entry's file exists
    and has the workflow's file name (when the workflow names one), else to
    the registered entry whose file has the workflow's file name (a LoRA only
    to a ``lora`` entry). Every other model is listed in
    ``missing`` as ``{"slot", "name", "filename", "native", "searchable",
    "source"}``: ``source`` is the download (``spec``, ``sha256``,
    ``model_type``, the ``name`` to register it under or None, ``origin``
    ``"curated"`` when it matches a file localm pins, else ``"workflow"``);
    without one, an imported ComfyUI model carries ``class_type``,
    ``input_name`` and ``source_name`` (the pull name), and ``searchable``
    says whether a HuggingFace search may look for it. A missing model's slot is left out of ``native_models``
    (picked automatically) and a missing LoRA is left out of ``loras``."""
    if reg is None:
        from localm.config import load_registry
        reg = load_registry()
    known = known_sources()
    native_models, loras, missing, notes = {}, [], [], []
    taken: set = set()

    def _one(slot: str, ref: dict, want_lora: bool) -> Optional[str]:
        entry = reg.get(ref["name"])
        path = _entry_file(entry) if isinstance(entry, dict) else None
        if path is not None:
            if want_lora and entry.get("model_type") != "lora":
                notes.append(f"'{ref['name']}' is registered, but not as a LoRA.")
            elif ref.get("file") and path.name != ref["file"]:
                notes.append(f"{slot}: the registered '{ref['name']}' is {path.name}, "
                             f"not the workflow's {ref['file']}.")
            else:
                return ref["name"]
        if ref.get("file"):
            hit = _registered_by_file(reg, ref["file"], want_lora)
            if hit is not None:
                if hit != ref["name"]:
                    notes.append(f"{slot}: using the registered model '{hit}' "
                                 f"({ref['file']}).")
                return hit
        missing.append(_missing_entry(slot, ref, reg, known, taken))
        return None

    for slot, ref in wf["models"].items():
        name = _one(slot, ref, False)
        if name is not None:
            native_models[slot] = name
    for i, ref in enumerate(wf["loras"]):
        name = _one(f"lora {i + 1}", ref, True)
        if name is not None:
            loras.append({"name": name, "strength": ref.get("strength", 1.0)})
    return {"native_models": native_models, "loras": loras, "params": dict(wf["params"]),
            "missing": missing, "notes": notes}


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------

MAX_IMPORT_BYTES = 8 * 1024 * 1024
_ROUTE_TIMEOUT_S = 60.0


def make_router(media: str):
    """``/api/<media>/native-workflows``: list (GET), get (GET ``/{id}``),
    save (POST ``{workflow, id?}``), rename (POST ``/{id}/rename {name}``),
    delete (DELETE ``/{id}``), import (POST ``/import {data, name?}``) and
    resolve (POST ``/{id}/resolve``). A bad workflow is a 400, an unknown id
    a 404."""
    from fastapi import APIRouter, HTTPException

    from localm.inference._threadpool_timeout import (
        ThreadCallTimeout, run_in_threadpool_bounded,
    )

    router = APIRouter()
    base = f"/api/{media}/native-workflows"

    async def _run(fn, what: str):
        try:
            return await run_in_threadpool_bounded(fn, timeout=_ROUTE_TIMEOUT_S)
        except WorkflowNotFound as e:
            raise HTTPException(404, f"No such {media} workflow: {e}") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"{what} timed out: {e}") from e

    @router.get(base)
    async def _list():
        def _do():
            with lock_for(media):
                return {"workflows": list_workflows(media)}
        return await _run(_do, f"Listing the {media} workflows")

    @router.get(base + "/{wid}")
    async def _get(wid: str):
        def _do():
            wf = get_workflow(media, wid)
            return {"id": wf["id"], "builtin": wf["builtin"], "workflow": export_data(wf)}
        return await _run(_do, "Reading the workflow")

    @router.post(base)
    async def _save(body: dict):
        wf = body.get("workflow")
        if not isinstance(wf, dict):
            raise HTTPException(400, "missing 'workflow' object")
        wid = body.get("id")
        if wid is not None and not isinstance(wid, str):
            raise HTTPException(400, "'id' must be text")

        def _do():
            from localm.config import load_registry
            with lock_for(media):
                saved = save_workflow(media, wf, wid=wid, reg=load_registry())
                return {"id": saved, "workflows": list_workflows(media)}
        return await _run(_do, "Saving the workflow")

    @router.post(base + "/import")
    async def _import(body: dict):
        data = body.get("data")
        name = body.get("name")
        if name is not None and not isinstance(name, str):
            raise HTTPException(400, "'name' must be text")
        try:
            size = len(json.dumps(data))
        except (RecursionError, TypeError, ValueError) as e:
            raise HTTPException(400, "The workflow is not usable JSON.") from e
        if size > MAX_IMPORT_BYTES:
            raise HTTPException(413, f"The workflow is larger than "
                                     f"{MAX_IMPORT_BYTES // (1024 * 1024)} MB.")

        def _do():
            with lock_for(media):
                checked = _text(name, "The workflow name", MAX_NAME, strip=True) if name else None
                saved, report = import_workflow(media, data, name=checked)
                return {"id": saved, "report": report, "workflows": list_workflows(media)}
        return await _run(_do, "Importing the workflow")

    @router.post(base + "/{wid}/rename")
    async def _rename(wid: str, body: dict):
        def _do():
            with lock_for(media):
                rename_workflow(media, wid, body.get("name"))
                return {"id": wid, "workflows": list_workflows(media)}
        return await _run(_do, "Renaming the workflow")

    @router.delete(base + "/{wid}")
    async def _delete(wid: str):
        def _do():
            with lock_for(media):
                delete_workflow(media, wid)
                return {"workflows": list_workflows(media)}
        return await _run(_do, "Deleting the workflow")

    @router.post(base + "/{wid}/resolve")
    async def _resolve(wid: str):
        def _do():
            wf = get_workflow(media, wid)
            return dict(resolve(wf, media), id=wf["id"], name=wf["name"])
        return await _run(_do, "Checking the workflow's models")

    return router
