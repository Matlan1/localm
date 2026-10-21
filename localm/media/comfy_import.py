# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convert a ComfyUI workflow (API format or UI/graph format) into a native
workflow (``native_workflows``).

The loader, sampler, latent, guidance, prompt and shift nodes map onto the
native workflow's models and parameters. Output and plumbing nodes the native
backend does by itself are listed as ``implicit``; every other node is listed
as ``unsupported``, and every value that could not be kept is a ``note``, so
nothing is dropped without saying so.

Model files get their download source from the registry (a registered file of
the same name), the files localm pins (``native_workflows.known_sources``), the
curated ComfyUI table and an earlier HuggingFace lookup still cached; a file
with none keeps its ComfyUI node and input name, so the page can offer a
search for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

MAX_NODES = 2000

WIDGETS = {
    "UNETLoader": ("unet_name", "weight_dtype"),
    "UnetLoaderGGUF": ("unet_name",),
    "CheckpointLoaderSimple": ("ckpt_name",),
    "DualCLIPLoader": ("clip_name1", "clip_name2", "type", "device"),
    "TripleCLIPLoader": ("clip_name1", "clip_name2", "clip_name3"),
    "CLIPLoader": ("clip_name", "type", "device"),
    "CLIPVisionLoader": ("clip_name",),
    "VAELoader": ("vae_name",),
    "LoraLoader": ("lora_name", "strength_model", "strength_clip"),
    "LoraLoaderModelOnly": ("lora_name", "strength_model"),
    "KSampler": ("seed", "control_after_generate", "steps", "cfg", "sampler_name",
                 "scheduler", "denoise"),
    "KSamplerAdvanced": ("add_noise", "noise_seed", "control_after_generate", "steps", "cfg",
                         "sampler_name", "scheduler", "start_at_step", "end_at_step",
                         "return_with_leftover_noise"),
    "EmptyLatentImage": ("width", "height", "batch_size"),
    "EmptySD3LatentImage": ("width", "height", "batch_size"),
    "EmptyHunyuanLatentVideo": ("width", "height", "length", "batch_size"),
    "FluxGuidance": ("guidance",),
    "CLIPTextEncode": ("text",),
    "ModelSamplingSD3": ("shift",),
    "EmptyAceStepLatentAudio": ("seconds", "batch_size"),
    "TextEncodeAceStepAudio": ("tags", "lyrics", "lyrics_strength"),
    "TextEncodeAceStepAudio15": ("tags", "lyrics", "seed", "control_after_generate", "bpm",
                                 "duration", "timesignature", "language", "keyscale",
                                 "generate_audio_codes", "cfg_scale", "temperature", "top_p",
                                 "top_k", "min_p"),
    "CreateVideo": ("fps",),
    "SaveAnimatedWEBP": ("filename_prefix", "fps", "lossless", "quality", "method"),
    "SaveWEBM": ("filename_prefix", "codec", "fps", "crf"),
    "PrimitiveNode": ("value",),
}

_SEED_CONTROLS = ("fixed", "increment", "decrement", "randomize")

IMPLICIT = frozenset({
    "VAEDecode", "VAEDecodeTiled", "SaveImage", "PreviewImage", "SaveAnimatedWEBP",
    "SaveAnimatedPNG", "SaveWEBM", "SaveVideo", "CreateVideo", "SaveAudio", "SaveAudioMP3",
    "SaveAudioOpus", "PreviewAudio", "VAEDecodeAudio", "Note", "MarkdownNote", "Reroute",
    "PrimitiveNode", "ConditioningZeroOut",
})

MODEL_LOADERS = ("UNETLoader", "UnetLoaderGGUF", "CheckpointLoaderSimple")
CLIP_LOADERS = {"DualCLIPLoader": ("clip_name1", "clip_name2"),
                "TripleCLIPLoader": ("clip_name1", "clip_name2", "clip_name3"),
                "CLIPLoader": ("clip_name",)}
SAMPLERS = ("KSampler", "KSamplerAdvanced")
LATENTS = ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyHunyuanLatentVideo")

SAMPLER_NAMES = {
    "euler": "euler", "euler_ancestral": "euler_a", "heun": "heun", "dpm_2": "dpm2",
    "dpmpp_2s_ancestral": "dpm++2s_a", "dpmpp_2m": "dpm++2m", "dpmpp_2m_sde": "dpm++2m_sde",
    "ipndm": "ipndm", "ipndm_v": "ipndm_v", "lcm": "lcm", "lms": "lms",
    "res_multistep": "res_multistep", "er_sde": "er_sde", "euler_cfg_pp": "euler_cfg_pp",
    "euler_ancestral_cfg_pp": "euler_a_cfg_pp",
}
SCHEDULER_NAMES = {
    "normal": "discrete", "karras": "karras", "exponential": "exponential",
    "sgm_uniform": "sgm_uniform", "simple": "simple", "beta": "beta",
    "kl_optimal": "kl_optimal",
}

_LLM_WORDS = ("qwen", "llama", "mistral", "gemma", "llava", "glm", "phi")
_LLM_TYPES = ("qwen_image", "lumina2", "hunyuan_image", "flux2", "ovis", "omnigen2",
              "hidream", "longcat_image")


class _Link:
    __slots__ = ("node", "slot")

    def __init__(self, node: str, slot: int):
        self.node, self.slot = node, slot


def is_api_format(data) -> bool:
    """A ComfyUI API-format workflow: a dict of node id -> node with a
    ``class_type``."""
    from localm.media_workflows import is_workflow_json
    return is_workflow_json(data)


def is_ui_format(data) -> bool:
    """A ComfyUI UI-format (graph) workflow: ``nodes`` (each with a ``type``)
    and ``links`` lists."""
    return (isinstance(data, dict) and isinstance(data.get("nodes"), list)
            and isinstance(data.get("links", []), list)
            and any(isinstance(n, dict) and isinstance(n.get("type"), str)
                    for n in data["nodes"]))


def is_comfy_workflow(data) -> bool:
    return is_api_format(data) or is_ui_format(data)


def _key(value):
    """*value* when it can identify a node or link (an int or a str), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    return value


def _api_nodes(data: dict, notes: list) -> dict:
    nodes = {}
    for nid, node in data.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            continue
        raw_inputs = node.get("inputs") or {}
        if not isinstance(raw_inputs, dict):
            notes.append(f"{node['class_type']} #{nid}: its inputs are not readable.")
            raw_inputs = {}
        inputs = {}
        for name, value in raw_inputs.items():
            if (isinstance(value, list) and len(value) == 2
                    and isinstance(value[0], (str, int)) and not isinstance(value[0], bool)
                    and isinstance(value[1], int)):
                inputs[name] = _Link(str(value[0]), value[1])
            else:
                inputs[name] = value
        title = (node.get("_meta") or {}).get("title") if isinstance(node.get("_meta"), dict) \
            else None
        nodes[str(nid)] = {"class_type": node["class_type"], "inputs": inputs, "title": title}
    return nodes


def _widget_names(class_type: str, values: list) -> list:
    names = list(WIDGETS.get(class_type, ()))
    if "control_after_generate" in names:
        i = names.index("control_after_generate")
        if not (i < len(values) and values[i] in _SEED_CONTROLS):
            names.pop(i)
    return names


def _ui_nodes(data: dict, notes: list) -> dict:
    links = {}
    for link in data.get("links") or []:
        if isinstance(link, list) and len(link) >= 5:
            lid, origin, slot = _key(link[0]), _key(link[1]), link[2]
        elif isinstance(link, dict):
            lid, origin, slot = _key(link.get("id")), _key(link.get("origin_id")), \
                link.get("origin_slot")
        else:
            continue
        if lid is None or origin is None:
            continue
        links[lid] = _Link(str(origin), slot if isinstance(slot, int)
                           and not isinstance(slot, bool) else 0)
    nodes = {}
    skipped = []
    for node in data["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("type"), str):
            continue
        nid = str(node.get("id"))
        if node.get("mode") in (2, 4):
            skipped.append(f"{node['type']} #{nid}")
            continue
        inputs = {}
        values = node.get("widgets_values")
        if isinstance(values, dict):
            inputs.update(values)
        elif isinstance(values, list):
            for name, value in zip(_widget_names(node["type"], values), values, strict=False):
                inputs[name] = value
        raw_inputs = node.get("inputs")
        node_inputs = raw_inputs if isinstance(raw_inputs, list) else []
        for inp in node_inputs:
            if (isinstance(inp, dict) and _key(inp.get("link")) is not None
                    and isinstance(inp.get("name"), str)):
                link = links.get(inp["link"])
                if link is not None:
                    inputs[inp["name"]] = link
        nodes[nid] = {"class_type": node["type"], "inputs": inputs, "title": node.get("title")}
    if skipped:
        notes.append("Muted or bypassed nodes were left out: " + ", ".join(skipped) + ".")
    definitions = data.get("definitions")
    if isinstance(definitions, dict) and definitions.get("subgraphs"):
        notes.append("The workflow contains subgraphs; their nodes were not read.")
    return nodes


def graph_nodes(data, notes: Optional[list] = None) -> dict:
    """``{node id: {"class_type", "inputs", "title"}}`` for an API- or
    UI-format workflow; a linked input is a link object. Raises
    ``ValueError`` for anything else or more than ``MAX_NODES`` nodes."""
    notes = [] if notes is None else notes
    if is_api_format(data):
        nodes = _api_nodes(data, notes)
    elif is_ui_format(data):
        nodes = _ui_nodes(data, notes)
    else:
        raise ValueError("Not a ComfyUI workflow (API format or UI format).")
    if len(nodes) > MAX_NODES:
        raise ValueError(f"The workflow has more than {MAX_NODES} nodes.")
    return nodes


class _Graph:
    def __init__(self, nodes: dict):
        self.nodes = nodes

    def value(self, nid: str, name: str, notes: list):
        """The literal value of input *name* of node *nid*: a widget value, or
        the value of a ``PrimitiveNode`` it is linked to; None otherwise."""
        v = self.nodes[nid]["inputs"].get(name)
        if isinstance(v, _Link):
            src = self.nodes.get(v.node)
            if src and src["class_type"] == "PrimitiveNode" and "value" in src["inputs"]:
                return src["inputs"]["value"]
            if src is not None:
                notes.append(f"{self.label(nid)}: '{name}' comes from {self.label(v.node)} "
                             "and was not read.")
            return None
        return v

    def label(self, nid: Optional[str]) -> str:
        n = self.nodes.get(nid) if nid is not None else None
        return f"{n['class_type']} #{nid}" if n else f"#{nid}"

    def upstream(self, nid: str, names: tuple, stop: tuple) -> Optional[str]:
        """The first node of a class in *stop* reached from *nid* by following
        inputs named in *names*, or None."""
        seen = set()
        cur = nid
        while cur is not None and cur not in seen and cur in self.nodes:
            seen.add(cur)
            node = self.nodes[cur]
            if node["class_type"] in stop and cur != nid:
                return cur
            nxt = None
            for name in names:
                v = node["inputs"].get(name)
                if isinstance(v, _Link):
                    nxt = v.node
                    break
            cur = nxt
        return None

    def of(self, classes) -> list:
        return sorted((nid for nid, n in self.nodes.items() if n["class_type"] in classes),
                      key=lambda x: (len(x), x))


def _clip_slot(filename: str, clip_type) -> Optional[str]:
    n = filename.lower().replace("\\", "/").rsplit("/", 1)[-1]
    if "umt5" in n or "t5" in n:
        return "t5xxl"
    if any(w in n for w in ("clip_g", "clip-g", "clipg")):
        return "clip_g"
    if any(w in n for w in ("clip_l", "clip-l", "clipl")):
        return "clip_l"
    if any(w in n for w in _LLM_WORDS):
        return "llm"
    if clip_type == "wan":
        return "t5xxl"
    if clip_type in _LLM_TYPES:
        return "llm"
    if clip_type == "stable_diffusion":
        return "clip_l"
    return None


def _model_ref(value, class_type: str, input_name: str, *, lora: bool,
               reg: dict, known: dict, notes: list) -> Optional[dict]:
    from localm.media import native_workflows as nw
    from localm.model_manager.comfy_resolve import (
        cached_comfy_download, curated_comfy_download, split_model_name,
    )
    split = split_model_name(value) if isinstance(value, str) else None
    if not split or not split[1]:
        notes.append(f"{class_type}: '{value}' is not a usable model file name.")
        return None
    filename = split[1]
    hit = nw._registered_by_file(reg, filename, lora)
    name = hit or Path(filename).stem
    try:
        name = nw.check_model_name(name, class_type)
    except ValueError as e:
        notes.append(str(e))
        return None
    ref: dict = {"name": name, "file": filename}
    source = known.get(filename)
    if source is None:
        dl = curated_comfy_download(value) or cached_comfy_download(value, class_type,
                                                                    input_name)
        if dl is not None:
            source = {"repo": dl.repo, "file": dl.path, "sha256": dl.sha256,
                      "size_bytes": dl.size_bytes, "model_type": dl.model_type}
    if source is not None:
        source = dict(source)
        if lora:
            source["model_type"] = "lora"
        try:
            ref["source"] = nw.check_source(source, class_type)
        except ValueError as e:
            notes.append(str(e))
    if "source" not in ref:
        ref["comfy"] = {"class_type": class_type, "input_name": input_name}
    return ref


def _number(value):
    from localm.media.native_workflows import is_number
    return value if is_number(value) else None


def convert(data, media: str, reg: Optional[dict] = None) -> tuple[dict, dict]:
    """``(workflow, report)`` for ComfyUI workflow *data* imported as a native
    *media* workflow. The workflow is normalised by
    ``native_workflows.parse_workflow``; the report is ``{"kind": "comfy-api"
    | "comfy-ui", "mapped": [{"node", "class_type", "to"}], "implicit":
    [{"node", "class_type"}], "unsupported": [{"node", "class_type",
    "title"}], "notes": [str]}``. Raises ``ValueError`` when *data* is not a
    ComfyUI workflow or maps to nothing."""
    from localm.media import native_workflows as nw
    if reg is None:
        from localm.config import load_registry
        reg = load_registry()
    notes: list = []
    kind = "comfy-api" if is_api_format(data) else "comfy-ui"
    g = _Graph(graph_nodes(data, notes))
    known = nw.known_sources()
    slots = nw.slot_keys(media)
    models: dict = {}
    loras: list = []
    params: dict = {}
    mapped, used = [], set()

    def _map(nid: str, to: str) -> None:
        used.add(nid)
        mapped.append({"node": nid, "class_type": g.nodes[nid]["class_type"], "to": to})

    def _set_param(key: str, value, nid: str) -> None:
        if value is None:
            return
        try:
            params[key] = nw._param(media, key, value)
        except ValueError as e:
            notes.append(f"{g.label(nid)}: {e}")

    def _put(slot: str, ref: Optional[dict], nid: str) -> None:
        if ref is None:
            used.add(nid)
            return
        if slot not in slots:
            notes.append(f"{g.label(nid)}: a {slot} file has no {media} slot; "
                         f"'{ref['file']}' was not kept.")
            used.add(nid)
            return
        if slot in models:
            notes.append(f"{g.label(nid)}: the {slot} slot is already '{models[slot]['file']}'; "
                         f"'{ref['file']}' was not kept.")
            used.add(nid)
            return
        models[slot] = ref
        _map(nid, f"{slot}: {ref['file']}")

    samplers = g.of(SAMPLERS)
    sampler = samplers[0] if samplers else None
    if len(samplers) > 1:
        used.update(samplers[1:])
        notes.append(f"The workflow has {len(samplers)} samplers; the settings of "
                     f"{g.label(sampler)} were used.")

    if media == "music":
        for nid in g.of(MODEL_LOADERS + tuple(CLIP_LOADERS) + ("VAELoader",)):
            used.add(nid)
            notes.append(f"{g.label(nid)}: the native music backend runs ACE-Step 1.5 GGUF "
                         "parts, so the ComfyUI model files are not used; the default parts "
                         "are picked automatically.")
    else:
        loaders = g.of(MODEL_LOADERS)
        main = (g.upstream(sampler, ("model",), MODEL_LOADERS) if sampler else None) or (
            loaders[0] if loaders else None)
        for nid in loaders:
            node = g.nodes[nid]
            field = "ckpt_name" if node["class_type"] == "CheckpointLoaderSimple" else "unet_name"
            if nid != main:
                used.add(nid)
                notes.append(f"{g.label(nid)}: a second model loader; only "
                             f"{g.label(main)} was kept.")
                continue
            ref = _model_ref(g.value(nid, field, notes), node["class_type"], field, lora=False,
                             reg=reg, known=known, notes=notes)
            _put("model", ref, nid)
            dtype = node["inputs"].get("weight_dtype")
            if isinstance(dtype, str) and dtype != "default":
                notes.append(f"{g.label(nid)}: weight_dtype '{dtype}' does not apply natively.")
        for nid in g.of(tuple(CLIP_LOADERS)):
            node = g.nodes[nid]
            clip_type = node["inputs"].get("type")
            for field in CLIP_LOADERS[node["class_type"]]:
                value = g.value(nid, field, notes)
                if not isinstance(value, str):
                    continue
                slot = _clip_slot(value, clip_type)
                if slot is None:
                    notes.append(f"{g.label(nid)}: cannot tell which text encoder slot "
                                 f"'{value}' fills; it was not kept.")
                    used.add(nid)
                    continue
                _put(slot, _model_ref(value, node["class_type"], field, lora=False, reg=reg,
                                      known=known, notes=notes), nid)
        for nid in g.of(("VAELoader",)):
            _put("vae", _model_ref(g.value(nid, "vae_name", notes), "VAELoader", "vae_name",
                                   lora=False, reg=reg, known=known, notes=notes), nid)
        for nid in g.of(("CLIPVisionLoader",)):
            _put("clip_vision", _model_ref(g.value(nid, "clip_name", notes), "CLIPVisionLoader",
                                           "clip_name", lora=False, reg=reg, known=known,
                                           notes=notes), nid)
        for nid in g.of(("LoraLoader", "LoraLoaderModelOnly")):
            node = g.nodes[nid]
            ref = _model_ref(g.value(nid, "lora_name", notes), node["class_type"], "lora_name",
                             lora=True, reg=reg, known=known, notes=notes)
            used.add(nid)
            if ref is None:
                continue
            sm = _number(g.value(nid, "strength_model", notes))
            sm = 1.0 if sm is None else float(sm)
            sc = _number(node["inputs"].get("strength_clip"))
            if sc is not None and abs(float(sc) - sm) > 1e-6:
                notes.append(f"{g.label(nid)}: one strength applies natively; "
                             f"strength_model {sm:g} was kept, strength_clip {sc:g} was not.")
            if not -4.0 <= sm <= 4.0:
                notes.append(f"{g.label(nid)}: strength {sm:g} is outside -4 to 4 and was "
                             "clamped.")
                sm = max(-4.0, min(4.0, sm))
            if len(loras) >= 8:
                notes.append(f"{g.label(nid)}: more than 8 LoRAs; '{ref['file']}' was not kept.")
                continue
            ref["strength"] = sm
            loras.append(ref)
            mapped.append({"node": nid, "class_type": node["class_type"],
                           "to": f"LoRA: {ref['file']} ({sm:g})"})

    if sampler is not None:
        node = g.nodes[sampler]
        _set_param("steps", _number(g.value(sampler, "steps", notes)), sampler)
        _set_param("cfg", _number(g.value(sampler, "cfg", notes)), sampler)
        sname = g.value(sampler, "sampler_name", notes)
        sched = g.value(sampler, "scheduler", notes)
        if media == "music":
            if sname not in (None, "euler") or sched not in (None, "simple", "normal"):
                notes.append(f"{g.label(sampler)}: the native music backend has no sampler or "
                             "scheduler choice.")
        else:
            if isinstance(sname, str):
                if sname in SAMPLER_NAMES:
                    params["sample_method"] = SAMPLER_NAMES[sname]
                else:
                    notes.append(f"{g.label(sampler)}: sampler '{sname}' has no native "
                                 "equivalent; the model's default is used.")
            if isinstance(sched, str):
                if sched in SCHEDULER_NAMES:
                    params["scheduler"] = SCHEDULER_NAMES[sched]
                else:
                    notes.append(f"{g.label(sampler)}: scheduler '{sched}' has no native "
                                 "equivalent; the model's default is used.")
        denoise = _number(node["inputs"].get("denoise"))
        if denoise is not None and denoise < 0.999:
            notes.append(f"{g.label(sampler)}: denoise {denoise:g} (image to image) was not kept.")
        if node["class_type"] == "KSamplerAdvanced":
            start = _number(node["inputs"].get("start_at_step"))
            end = _number(node["inputs"].get("end_at_step"))
            steps = params.get("steps")
            if (start or 0) > 0 or (end is not None and steps is not None and end < steps):
                notes.append(f"{g.label(sampler)}: start/end steps {start}-{end} were not kept.")
            for field, default in (("add_noise", "enable"), ("return_with_leftover_noise", "disable")):
                value = node["inputs"].get(field)
                if isinstance(value, str) and value != default:
                    notes.append(f"{g.label(sampler)}: {field} '{value}' was not kept.")
        notes.append("The seed was not kept; each run picks one unless the page sets it.")
        _map(sampler, "steps, CFG, sampler, scheduler")
        for side, key in (("positive", "prompt"), ("negative", "negative_prompt")):
            if media == "music":
                break
            v = node["inputs"].get(side)
            if not isinstance(v, _Link):
                continue
            src = v.node if g.nodes.get(v.node, {}).get("class_type") in (
                "CLIPTextEncode", "ConditioningZeroOut") else g.upstream(
                    v.node, ("conditioning", "conditioning_1", "positive", "negative"),
                    ("CLIPTextEncode", "ConditioningZeroOut"))
            if src is None:
                continue
            used.add(src)
            if g.nodes[src]["class_type"] == "ConditioningZeroOut":
                continue
            text = g.value(src, "text", notes)
            if isinstance(text, str):
                _set_param(key, text, src)
                _map(src, key.replace("_", " "))
        guide = g.upstream(sampler, ("positive",), ("FluxGuidance",))
        guides = g.of(("FluxGuidance",))
        guide = guide or (guides[0] if guides else None)
        if guide is not None:
            if media == "image":
                _set_param("guidance", _number(g.value(guide, "guidance", notes)), guide)
                _map(guide, "guidance")
            else:
                used.add(guide)
                notes.append(f"{g.label(guide)}: guidance does not apply to {media}.")

    for nid in g.of(("CLIPTextEncode",)):
        if nid not in used:
            used.add(nid)
            notes.append(f"{g.label(nid)}: a prompt not connected to the sampler was not kept.")

    shifts = g.of(("ModelSamplingSD3",))
    for nid in shifts[1:]:
        used.add(nid)
        notes.append(f"{g.label(nid)}: a second sampling shift was not used.")
    for nid in shifts[:1]:
        shift = _number(g.value(nid, "shift", notes))
        key = {"video": "flow_shift", "music": "shift"}.get(media)
        if key is None:
            used.add(nid)
            notes.append(f"{g.label(nid)}: the sampling shift does not apply to native images.")
            continue
        _set_param(key, shift, nid)
        _map(nid, key.replace("_", " "))

    latents = g.of(LATENTS)
    if latents:
        lat = (g.upstream(sampler, ("latent_image",), LATENTS) if sampler else None) or latents[0]
        if media == "music":
            used.add(lat)
            notes.append(f"{g.label(lat)}: an image or video latent does not apply to music.")
        else:
            _set_param("width", _number(g.value(lat, "width", notes)), lat)
            _set_param("height", _number(g.value(lat, "height", notes)), lat)
            batch = _number(g.nodes[lat]["inputs"].get("batch_size"))
            if batch is not None and batch > 1:
                notes.append(f"{g.label(lat)}: batch size {batch:g} was not kept.")
            _map(lat, "width, height")
            length = _number(g.nodes[lat]["inputs"].get("length"))
            if media == "video" and length is not None:
                fps = None
                for nid in g.of(("CreateVideo", "SaveAnimatedWEBP", "SaveWEBM")):
                    fps = _number(g.value(nid, "fps", notes))
                    if fps:
                        break
                if fps:
                    fps_i = max(1, min(60, int(round(fps))))
                    _set_param("fps", fps_i, lat)
                    _set_param("seconds", round(length / fps_i, 2), lat)
                else:
                    notes.append(f"{g.label(lat)}: {length:g} frames, but no frame rate was "
                                 "found; the length was not kept.")
            for nid in latents:
                if nid != lat:
                    used.add(nid)
                    notes.append(f"{g.label(nid)}: a second latent was not used.")

    if media == "music":
        audio_latents = g.of(("EmptyAceStepLatentAudio",))
        for nid in audio_latents[:1]:
            _set_param("duration_seconds", _number(g.value(nid, "seconds", notes)), nid)
            _map(nid, "duration")
        encoders = g.of(("TextEncodeAceStepAudio", "TextEncodeAceStepAudio15"))
        for nid in audio_latents[1:] + encoders[1:]:
            used.add(nid)
            notes.append(f"{g.label(nid)}: a second one was not used.")
        for nid in encoders[:1]:
            for field, key in (("tags", "tags"), ("lyrics", "lyrics")):
                v = g.value(nid, field, notes)
                if isinstance(v, str):
                    _set_param(key, v, nid)
            if g.nodes[nid]["class_type"] == "TextEncodeAceStepAudio15":
                duration = _number(g.value(nid, "duration", notes))
                if "duration_seconds" not in params:
                    _set_param("duration_seconds", duration, nid)
                elif duration is not None and abs(duration - params["duration_seconds"]) > 1e-6:
                    notes.append(f"{g.label(nid)}: duration {duration:g} differs from the "
                                 f"latent's {params['duration_seconds']:g}; the latent's was kept.")
            ls = _number(g.nodes[nid]["inputs"].get("lyrics_strength"))
            if ls is not None and abs(ls - 1.0) > 1e-6:
                notes.append(f"{g.label(nid)}: lyrics strength {ls:g} has no native setting.")
            _map(nid, "tags, lyrics")

    implicit, unsupported = [], []
    for nid in sorted(g.nodes, key=lambda x: (len(x), x)):
        if nid in used:
            continue
        node = g.nodes[nid]
        if node["class_type"] in IMPLICIT:
            implicit.append({"node": nid, "class_type": node["class_type"]})
        else:
            unsupported.append({"node": nid, "class_type": node["class_type"],
                                "title": node.get("title") or ""})
    if not mapped:
        raise ValueError("No node of this ComfyUI workflow maps to a native setting.")
    title = None
    if isinstance(data, dict):
        extra = data.get("extra")
        raw_title = extra.get("title") if isinstance(extra, dict) else None
        title = raw_title if isinstance(raw_title, str) else None
    name = "".join(ch for ch in (title or "") if ord(ch) >= 32).strip()[:nw.MAX_NAME]
    name = name or "Imported ComfyUI workflow"
    desc = f"Imported from a ComfyUI workflow ({len(g.nodes)} nodes)."
    if unsupported:
        desc += " Nodes without a native equivalent: " + ", ".join(
            sorted({u["class_type"] for u in unsupported})) + "."
    wf = nw.parse_workflow({"format": nw.FORMAT, "version": nw.VERSION, "media": media,
                            "name": name, "description": desc[:nw.MAX_DESCRIPTION],
                            "models": models, "loras": loras, "params": params}, media)
    return wf, {"kind": kind, "mapped": mapped, "implicit": implicit,
                "unsupported": unsupported, "notes": notes}
