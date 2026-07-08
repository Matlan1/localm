# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Shared ComfyUI client - generic plumbing for image / video / music generation.

One ComfyUI server, one set of helpers. This module holds everything that is
NOT specific to a single medium's workflow: role-based node resolution, model
preflight against ``/object_info``, the VRAM handoff, server reachability and
launch, and the queue / poll / download transport shared by every generator.

The image, video, and music modules import from here and keep only their own
workflow shaping (which nodes to inject, which output keys to read).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
#  Role-based node resolution (resolve nodes by class_type + graph edges)
# ---------------------------------------------------------------------------
#
#  A ComfyUI workflow is a dict of {node_id: {"class_type": str, "inputs": {...}}}.
#  An input value is either a literal or a LINK [source_node_id, output_index].
#  Hardcoding node ids (workflow["8"]["inputs"]["seed"]) breaks the moment a user
#  exports their own graph from ComfyUI, because the ids are arbitrary. Resolving
#  by ROLE - find the sampler by class_type, then follow its positive / negative /
#  latent edges to their source nodes - works on any graph that wires those roles,
#  so a local override no longer has to preserve the template's exact ids (I3).

# Sampler node classes that carry the seed / steps / cfg knobs and the
# positive / negative / latent_image edges we trace the other roles from.
_SAMPLER_CLASSES = (
    "KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
)


def _is_link(value) -> bool:
    """True when an input value is a ComfyUI link: ``[source_node_id, output_index]``."""
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], (str, int)) and not isinstance(value[0], bool)
            and isinstance(value[1], int) and not isinstance(value[1], bool))


def _link_source_id(node: dict, input_name: str) -> Optional[str]:
    """The source node id an input is wired to, or None when it is a literal."""
    if not isinstance(node, dict):
        return None
    value = node.get("inputs", {}).get(input_name)
    return str(value[0]) if _is_link(value) else None


def find_node_by_class(workflow: dict, *class_types: str):
    """First ``(id, node)`` whose class_type is one of *class_types*, else
    ``(None, None)``. Dict insertion order is preserved, so the first matching node
    in the graph wins."""
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") in class_types:
            return str(nid), node
    return None, None


def find_nodes_by_class(workflow: dict, *class_types: str) -> list:
    """All ``(id, node)`` pairs whose class_type is one of *class_types*."""
    return [(str(nid), node) for nid, node in workflow.items()
            if isinstance(node, dict) and node.get("class_type") in class_types]


def resolve_sampler_roles(workflow: dict) -> dict:
    """Resolve ``{sampler, positive, negative, latent}`` to ``(node_id, node)`` by
    following the sampler's input edges, so node ids can be anything.

    A role is ``(None, None)`` when the sampler or that edge is absent. ``positive``
    / ``negative`` point at whatever conditioning node feeds them (a CLIPTextEncode,
    a TextEncodeAceStepAudio, a ConditioningZeroOut, ...); the caller injects the
    field that node actually has rather than assuming a text box."""
    roles = {"sampler": (None, None), "positive": (None, None),
             "negative": (None, None), "latent": (None, None)}
    sid, sampler = find_node_by_class(workflow, *_SAMPLER_CLASSES)
    if sampler is None:
        return roles
    roles["sampler"] = (sid, sampler)
    for role, input_name in (("positive", "positive"),
                             ("negative", "negative"),
                             ("latent", "latent_image")):
        src = _link_source_id(sampler, input_name)
        if src is not None and src in workflow:
            roles[role] = (src, workflow[src])
    return roles


def set_seed_on(node: dict, seed: int) -> None:
    """Set whichever seed field a sampler node actually has (KSampler uses ``seed``;
    a RandomNoise / KSamplerAdvanced uses ``noise_seed``). Never invents a field the
    node does not declare, which ComfyUI would reject."""
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return
    if "seed" in inputs:
        inputs["seed"] = seed
    elif "noise_seed" in inputs:
        inputs["noise_seed"] = seed


def set_seed_on_all(workflow: dict, seed: int) -> int:
    """Set the seed on EVERY sampler/noise node in the graph and return how many were
    set. A multi-sampler graph (a two-stage Wan high/low-noise split, an SDXL refiner)
    must get the seed on each stage, or a later stage stays on the template's fixed
    seed - making "random" output partially deterministic. Only touches nodes that
    actually declare a seed field (set_seed_on is a no-op otherwise)."""
    n = 0
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in _SAMPLER_CLASSES or \
                node.get("class_type") == "RandomNoise":
            inputs = node.get("inputs")
            if isinstance(inputs, dict) and ("seed" in inputs or "noise_seed" in inputs):
                set_seed_on(node, seed)
                n += 1
    return n


def next_node_id(workflow: dict) -> str:
    """A node id not already used by *workflow* (max numeric id + 1, else a stable
    name). Injected helper nodes (e.g. a LoadImage for img2img/img2video) use this
    instead of a hardcoded id, which could clobber a node in a user's own graph."""
    numeric = [int(k) for k in workflow if str(k).isdigit()]
    if numeric:
        return str(max(numeric) + 1)
    base, i = "localm_node_", 0
    while f"{base}{i}" in workflow:
        i += 1
    return f"{base}{i}"


# ---------------------------------------------------------------------------
#  Pre-submit model validation (/object_info) - fail BEFORE the LLM unload
# ---------------------------------------------------------------------------
#
#  Before unloading the chat model (an expensive VRAM handoff) we ask ComfyUI which
#  model files each loader can actually see (GET /object_info) and confirm the
#  workflow's filenames exist. A missing file is named EXACTLY, and where the user
#  has the same model in a different precision/quant we auto-substitute the single
#  unambiguous variant. This turns a late "rejected (HTTP 400) after the unload"
#  into an early, specific error pointing at the Workflow panel (I3 / MEDIA-1).
#
#  Best-effort by design: when /object_info cannot be read it does NOT block - the
#  submit-time validation still applies. We never escalate a best-effort early
#  check into a hard failure that breaks an otherwise-working setup (AGENTS rule 5).

# Extensions that mark a combo's options as model files (so a value not in the list
# is a missing-model error we can name/fix), as opposed to an enum like sampler_name.
_MODEL_FILE_EXTS = (".safetensors", ".ckpt", ".gguf", ".pt", ".pth", ".bin",
                    ".sft", ".onnx", ".pte")


def comfy_object_info(api_url: str, timeout: float = 10.0) -> Optional[dict]:
    """ComfyUI's full ``/object_info`` map ``{class_type: spec}``, or None when it
    cannot be fetched or parsed (so the caller treats preflight as best-effort)."""
    try:
        with urllib.request.urlopen(f"{api_url}/object_info", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _combo_options(spec: dict, input_name: str) -> Optional[list]:
    """The list of literal choices for a combo input of an /object_info node spec,
    or None when the input is not a combo.

    Shape: ``spec["input"]["required"|"optional"][name] == [choices, meta?]`` where
    ``choices`` is a list of strings for a dropdown. A non-combo input's first element
    is a type-name string ("INT", "MODEL", ...), not a list."""
    if not isinstance(spec, dict):
        return None
    io = spec.get("input")
    if not isinstance(io, dict):
        return None
    for section in ("required", "optional"):
        sec = io.get(section)
        if isinstance(sec, dict) and input_name in sec:
            entry = sec[input_name]
            if isinstance(entry, list) and entry and isinstance(entry[0], list):
                return [o for o in entry[0] if isinstance(o, str)]
    return None


def _looks_like_model_files(options: list) -> bool:
    """True when a combo's options look like model files (a mismatch is then a
    missing-model error we can name), not an enum like sampler_name / scheduler."""
    if not options:
        return False
    hits = sum(1 for o in options if o.lower().endswith(_MODEL_FILE_EXTS))
    return hits >= max(1, len(options) // 2)


def _normalize_model_base(name: str) -> str:
    """A precision/quant-insensitive key for a model filename, so e.g.
    ``wan_5B_fp16.safetensors`` and ``wan_5B_fp8_scaled.safetensors`` share a base
    and one can stand in for the other. Drops the extension and ONLY unambiguous
    precision / quant markers (fp16/fp8/bf16/e4m3fn/scaled/qN...), keeping everything
    else.

    Crucially it does NOT drop bare digits or lone single letters: those are version
    / variant discriminators (``wan2.1`` vs ``wan2.2``, ``model_s`` vs ``model_m``,
    ``vae_1`` vs ``vae_2``) and collapsing them would merge genuinely different models
    into one base and trigger a WRONG auto-substitution - the opposite of "a single
    unambiguous precision variant"."""
    import re as _re
    n = name.lower().rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    for ext in _MODEL_FILE_EXTS:
        if n.endswith(ext):
            n = n[: -len(ext)]
            break
    drop = _re.compile(
        r"^(fp8|fp16|fp32|f8|f16|f32|bf16|e\dm\d[a-z0-9]*|scaled|default|q\d+)$")
    kept = [t for t in _re.split(r"[^a-z0-9]+", n) if t and not drop.match(t)]
    return "".join(kept)


def _pick_variant(missing_name: str, options: list) -> Optional[str]:
    """The single unambiguous precision/quant variant of *missing_name* among
    *options*, or None when there are zero or more than one candidates (we never
    guess between several)."""
    base = _normalize_model_base(missing_name)
    if not base:
        return None
    cands = [o for o in options
             if o != missing_name and _normalize_model_base(o) == base]
    return cands[0] if len(cands) == 1 else None


def _format_missing(missing: list) -> str:
    lines = ["ComfyUI is missing model files this workflow needs:"]
    for cls, field, name, options in missing:
        shown = ", ".join(options[:8]) + (", ..." if len(options) > 8 else "")
        avail = f" Available {field}: {shown}." if options else ""
        lines.append(
            f"  - '{name}' (the {field} for the {cls} node) is not installed.{avail}")
    lines.append(
        "Install the file(s) into ComfyUI's models folder, or pick a workflow whose "
        "models you have on the Workflow panel (Settings -> Media). The chat model was "
        "NOT unloaded - fix this and run again.")
    return "\n".join(lines)


def preflight_models(workflow: dict, api_url: str, *, on_progress=None) -> tuple[bool, str]:
    """Validate every loader's model file against ComfyUI ``/object_info`` BEFORE the
    caller unloads the chat model.

    Mutates *workflow* in place to substitute the single unambiguous precision/quant
    variant for a missing file. Returns ``(ok, message)``: ``ok=False`` with a
    specific, Workflow-panel-pointing error when a required model is missing and no
    one variant fits; ``ok=True`` (empty message) otherwise. Best-effort: returns
    ``(True, "")`` when /object_info is unavailable (defer to submit-time validation)."""
    info = comfy_object_info(api_url)
    if not info:
        return True, ""        # cannot validate -> defer to submit-time validation
    missing: list = []
    subs: list = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        spec = info.get(node.get("class_type"))
        inputs = node.get("inputs")
        if not isinstance(spec, dict) or not isinstance(inputs, dict):
            continue           # unknown class / no inputs here -> can't validate; skip
        for input_name, value in list(inputs.items()):
            if not isinstance(value, str):
                continue       # links and numbers are not model-file names
            options = _combo_options(spec, input_name)
            if options is None or not _looks_like_model_files(options):
                continue
            if value in options:
                continue       # the file is present - good
            variant = _pick_variant(value, options)
            if variant is not None:
                inputs[input_name] = variant
                subs.append((node.get("class_type"), input_name, value, variant))
            else:
                missing.append((node.get("class_type"), input_name, value, options))
    if on_progress:
        for cls, field, old, new in subs:
            try:
                on_progress(
                    f"Model '{old}' is not installed; substituting '{new}' "
                    f"(same model, different precision) for the {cls} node.")
            except Exception:
                pass
    if missing:
        return False, _format_missing(missing)
    return True, ""


# ---------------------------------------------------------------------------
#  VRAM management
# ---------------------------------------------------------------------------

def _localm_unload(localm_url: Optional[str] = None) -> Optional[dict]:
    """
    Ask a localm server to release its model from GPU memory.

    Reads LOCALM_URL from the environment if *localm_url* is not given, and
    authenticates with the LOCALM_API_KEY bearer token when one is set. The
    ``/v1/models/unload`` endpoint requires the models-write scope, so an
    UNAUTHENTICATED POST is rejected with 401 and the chat model stays resident
    in VRAM - the media model then loads on top of it, exceeds total VRAM and
    hangs the GPU driver (the AMD TDR the user hit). For the same reason the
    built-in TLS cert of a loopback ``https`` self-call must be trusted, exactly
    as the media-job model reload does (``localm.tls.requests_verify``); plain
    ``urllib`` would reject the self-signed cert and silently skip the unload.

    Silent no-op when the URL is unset. Returns the server's JSON result
    (``status`` / ``vram_freed`` / ``vram_before_bytes`` / ``vram_after_bytes``)
    on success, or None on any failure - never blocks generation if localm is
    not in the picture.
    """
    url = (localm_url or os.environ.get("LOCALM_URL", "")).rstrip("/")
    if not url:
        return None
    try:
        import requests as _rq

        from localm import tls as _tls
        headers = {}
        key = os.environ.get("LOCALM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        # Unload waits for any in-flight generation to finish AND for VRAM to be
        # actually reclaimed before it returns (it must not free the context
        # mid-decode), so give it time.
        resp = _rq.post(f"{url}/models/unload", headers=headers, timeout=180,
                        verify=_tls.requests_verify(url))
        if not resp.ok:
            return None
        try:
            return resp.json()
        except Exception:
            return {"status": "unloaded"}
    except Exception:
        return None


_COMFY_LOOPBACK_DEFAULT = "http://127.0.0.1:8188"


def _host_is_link_local(host: str) -> bool:
    """True if *host* is (or resolves to) a link-local / cloud-metadata address
    (169.254.0.0/16, fe80::/10). A ComfyUI never lives there; loopback / LAN /
    public do, and are allowed."""
    import ipaddress
    import socket
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_link_local
    except ValueError:
        pass
    try:
        for info in socket.getaddrinfo(host, None):
            try:
                if ipaddress.ip_address(info[4][0]).is_link_local:
                    return True
            except ValueError:
                continue
    except (socket.gaierror, OSError):
        return False
    return False


def sanitize_comfy_url(url: str) -> str:
    """Return *url* unless its host is link-local / cloud-metadata (or the guard
    itself cannot validate it), in which case warn and fall back to the loopback
    default. Defense-in-depth so an ADMIN-set comfy_api_url cannot turn the comfy
    control calls (free / interrupt / stop) into an SSRF probe of cloud metadata
    (CHK-COMFY-APIURL). Loopback + LAN + public are allowed - a real ComfyUI runs
    on any of those."""
    try:
        if _host_is_link_local(urllib.parse.urlparse(url).hostname or ""):
            from localm.debuglog import logger
            logger.warning(
                "comfy_api_url %r targets a link-local/metadata address; ignoring "
                "it and using the loopback default (CHK-COMFY-APIURL)", url)
            return _COMFY_LOOPBACK_DEFAULT
    except Exception as e:
        # FAIL CLOSED: if the guard itself cannot parse or verify the URL (e.g.
        # urlparse raising "Invalid IPv6 URL"), refusing is the only honest
        # outcome - returning the URL unchecked would silently approve exactly
        # what this guard exists to refuse (AGENTS.md rule 5). A URL the guard
        # cannot parse is not a working ComfyUI endpoint anyway.
        from localm.debuglog import logger
        logger.warning(
            "comfy_api_url %r could not be validated (%s); ignoring it and "
            "using the loopback default (CHK-COMFY-APIURL)", url, e)
        return _COMFY_LOOPBACK_DEFAULT
    return url


def default_api_url() -> str:
    """ComfyUI base URL: FLUX_API_URL env override, then a localm-MANAGED
    instance when one is installed and selected (coexistence, decision 6), then
    the ``comfy_api_url`` config key, else the ComfyUI default port. A link-local
    / cloud-metadata target is refused and falls back to loopback
    (CHK-COMFY-APIURL).

    The managed-ComfyUI hook is the ONLY managed touch in this module and is
    confined to TARGET RESOLUTION (never the launch/spawn path): it returns None
    - byte-identical to before - until a managed instance actually exists on
    disk, so nothing changes for a user who has not opted in."""
    env = os.environ.get("FLUX_API_URL")
    if env:
        return sanitize_comfy_url(env.rstrip("/"))
    try:
        from localm.media.managed_comfy import managed_comfy_api_url_if_active
        managed = managed_comfy_api_url_if_active()
        if managed:
            return managed
    except Exception:
        pass
    try:
        from localm.config import load_config
        cfg_url = load_config().get("comfy_api_url")
        if cfg_url:
            return sanitize_comfy_url(str(cfg_url).rstrip("/"))
    except Exception:
        pass
    return _COMFY_LOOPBACK_DEFAULT


def free_comfy_vram(api_url: Optional[str] = None) -> bool:
    """
    Ask ComfyUI to unload its models and free VRAM (POST /free).

    Returns True when ComfyUI accepted the request. Used after generation
    so the chat model can be reloaded immediately instead of spilling into
    system RAM next to a resident FLUX. Both an HTTP error (older ComfyUI
    builds without /free) and a network error return False, but each is now
    logged at debug level so --debug-discoverable can tell the two apart;
    callers then leave the LLM reload lazy.
    """
    url = (api_url or default_api_url()).rstrip("/")
    try:
        body = json.dumps({"unload_models": True, "free_memory": True}).encode()
        req = urllib.request.Request(
            f"{url}/free", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30):
            return True
    except urllib.error.HTTPError as e:
        # Distinguish the documented older-build case (no /free endpoint) from a
        # real network failure below, so a missing endpoint is not mistaken for
        # ComfyUI being unreachable.
        from localm.debuglog import logger
        logger.debug("free_comfy_vram: ComfyUI /free returned HTTP %s (%s) at %s; "
                     "likely an older build without /free", e.code, e.reason, url)
        return False
    except (urllib.error.URLError, Exception) as e:
        # A real network/connection failure reaching ComfyUI, not a missing endpoint.
        from localm.debuglog import logger
        logger.debug("free_comfy_vram: could not reach ComfyUI /free at %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _comfy_alive(api_url: str, timeout: float = 3.0) -> bool:
    """Quick reachability probe so callers can fail fast with a clear error."""
    try:
        with urllib.request.urlopen(f"{api_url}/system_stats", timeout=timeout):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  Readiness cache - ComfyUI does not need re-checking on every task
#
#  Before this, ensure_comfy() pinged /system_stats on EVERY call, and every
#  media submission called it twice back-to-back (once at the route-handler
#  layer via ensure_available, once again at the top of the generator
#  function) - plain redundant network round-trips, on top of the GUI's own
#  5-second poll (settings.js, removed separately). ComfyUI does not appear
#  or disappear on its own between requests, so once it has been confirmed
#  reachable for a given api_url in this process's lifetime, later calls
#  trust that instead of re-pinging: check on app start, on the Settings/
#  Media page being opened, before the FIRST task submission, and on an
#  explicit status request are enough. mark_comfy_dead() (called from
#  stop_comfy(), and internally when ensure_comfy() can no longer reach it)
#  clears the entry so the next check is real again.
# ---------------------------------------------------------------------------

_confirmed_alive: set[str] = set()


def mark_comfy_alive(api_url: str) -> None:
    _confirmed_alive.add(api_url.rstrip("/"))


def mark_comfy_dead(api_url: str) -> None:
    _confirmed_alive.discard(api_url.rstrip("/"))


def is_comfy_confirmed(api_url: Optional[str] = None) -> bool:
    """True when *api_url* has already been confirmed reachable this process
    lifetime and nothing has invalidated that since (see module docstring)."""
    return (api_url or default_api_url()).rstrip("/") in _confirmed_alive


def warm_comfy_status_async(api_url: Optional[str] = None) -> None:
    """Fire-and-forget readiness check for the "on app start" trigger: primes
    the cache without blocking plugin registration and without attempting to
    launch ComfyUI (an app boot should not decide FOR the user that this
    session needs ComfyUI running - only the on-demand triggers do that)."""
    import threading

    url = (api_url or default_api_url()).rstrip("/")

    def _check() -> None:
        if _comfy_alive(url):
            mark_comfy_alive(url)

    threading.Thread(target=_check, name="comfy-status-warm", daemon=True).start()


def history_execution_error(entry: dict) -> Optional[str]:
    """Return a human-readable ComfyUI execution error from a ``/history`` entry,
    or None when the job did not error.

    When a node crashes mid-render (a missing model, an ACE-Step/ComfyUI version
    mismatch, an OOM) ComfyUI still records the prompt in ``/history`` but with
    ``status.status_str == "error"`` and an ``("execution_error", {...})`` message
    carrying the node type and exception text. The poll loops otherwise only look
    for an output artifact and, finding none, blame a generic "no output" - so the
    real cause was hidden and the user was told to read the ComfyUI console (issue
    I2). Surfacing it here turns that into the actual reason."""
    status = entry.get("status") or {}
    for m in (status.get("messages") or []):
        if isinstance(m, (list, tuple)) and len(m) >= 2 and m[0] == "execution_error":
            info = m[1] if isinstance(m[1], dict) else {}
            node = info.get("node_type") or info.get("node_id")
            exc = (info.get("exception_message")
                   or info.get("exception_type") or "").strip()
            detail = " ".join(p for p in (exc, f"(node {node})" if node else "") if p)
            if detail:
                return detail
    if status.get("status_str") == "error":
        return "ComfyUI reported an execution error (no detail in /history)."
    return None


# ---------------------------------------------------------------------------
#  Reactive ComfyUI __func__ regression detection + offer (MEDIA-1)
# ---------------------------------------------------------------------------
#
#  A ComfyUI CORE regression in comfy_api/internal/__init__.py's
#  make_locked_method_func does `getattr(type_obj, func).__func__`, assuming a
#  node's FUNCTION is a bound method. A node whose FUNCTION resolves to a plain
#  function (the core audio VAEDecodeAudio, used by native ACE-Step) has no
#  `.__func__` -> AttributeError. Refs: Comfy-Org/ComfyUI #12116,
#  patientx/ComfyUI-Zluda #424. localm only submits a workflow + polls, so this is
#  purely upstream. We do NOT assume ComfyUI is broken: only when a REAL generation
#  hits this exact error do we offer a localm-side, in-memory shim (see comfy_shim/).

def is_known_comfy_func_regression(detail) -> bool:
    """True when a ComfyUI execution-error detail is the known upstream
    make_locked_method_func __func__ regression (Comfy-Org/ComfyUI #12116,
    patientx/ComfyUI-Zluda #424). ONLY this specific error should trigger the
    reactive offer; every other execution error is unrelated and behaves as before."""
    if not detail:
        return False
    text = str(detail)
    return ("has no attribute '__func__'" in text
            or "make_locked_method_func" in text)


def comfy_exec_error_message(payload, api_url: Optional[str] = None) -> str:
    """Build the message for a ComfyUI POLL_EXEC_ERROR. For an unrelated error this
    is the plain wording every generator used before. For the known __func__
    regression it is a richer, actionable message: what it is, that the fix is
    localm-side and in-memory (writes NOTHING into the user's ComfyUI install and
    self-expires once ComfyUI is fixed), and how to apply it."""
    detail = "" if payload is None else str(payload)
    if not is_known_comfy_func_regression(detail):
        return f"ComfyUI execution failed: {detail}"
    localm_started = spawned_pid(api_url) is not None
    apply_hint = (
        "localm can restart the ComfyUI it launched and retry with the fix."
        if localm_started else
        "Close your ComfyUI first (localm never touches a ComfyUI it did not "
        "start), then let localm launch a fixed one (needs comfy_workdir set).")
    return (
        "ComfyUI execution failed: " + detail + "\n"
        "This is a known ComfyUI core regression (Comfy-Org/ComfyUI #12116, "
        "patientx/ComfyUI-Zluda #424): a node whose function is a plain function "
        "(the native ACE-Step audio decode) trips an internal __func__ access.\n"
        "localm can apply an in-memory, localm-side compatibility shim - it patches "
        "ONLY a ComfyUI that localm itself starts (via a PYTHONPATH env var), writes "
        "NOTHING into your ComfyUI install, and self-expires once ComfyUI ships its "
        "own fix. " + apply_hint + "\n"
        "To stop being asked, turn it on: localm config comfy_func_shim on")


# ---------------------------------------------------------------------------
#  S5 BUG-REOFFER: the __func__ regression re-offers managed ComfyUI ONCE
# ---------------------------------------------------------------------------
#
#  The T1 shim (above) fixes THIS run in memory. As a durable answer, decision 1
#  + 8 of the managed-ComfyUI design let the __func__ crash ALSO offer localm's
#  own managed, patched ComfyUI (`localm comfy setup`) - the fix-for-good - but at
#  most ONCE (a persisted flag), never when a managed instance already exists (it
#  is moot: localm routes to the patched managed one, decision 6), and never for an
#  unrelated error. This is a pure decision + message; the CLI offer point
#  (cli/media.py) presents it alongside the shim offer.

_MANAGED_SETUP_OFFERED_KEY = "comfy_managed_setup_offered"


def should_offer_managed_comfy_setup(detail, cfg: Optional[dict] = None) -> bool:
    """True when the ONE-TIME durable-fix offer (set up localm's own managed,
    patched ComfyUI) should be surfaced for a media error.

    Gated on all three (design decisions 1 + 8): the error is the known upstream
    ``__func__`` regression, NO managed ComfyUI is installed yet (else the offer is
    moot - localm already routes to the patched managed instance, decision 6), and
    localm has not already made this offer (the persisted
    ``comfy_managed_setup_offered`` flag). Any one false -> no offer, so it never
    nags. A managed-install check that itself errors fails SAFE toward not offering
    (never surface something the user may already have)."""
    if not is_known_comfy_func_regression(detail):
        return False
    try:
        from localm.media.managed_comfy import is_managed_comfy_installed
        if is_managed_comfy_installed():
            return False
    except Exception:
        return False
    if cfg is None:
        try:
            from localm.config import load_config
            cfg = load_config()
        except Exception:
            cfg = {}
    try:
        return not bool(cfg.get(_MANAGED_SETUP_OFFERED_KEY))
    except AttributeError:
        return False


def managed_comfy_setup_offer_message() -> str:
    """The one-time durable-fix offer text: set up localm's OWN managed, patched
    ComfyUI so the recurring upstream regression cannot recur. Points at the
    explicit opt-in command (`localm comfy setup`); the user's own ComfyUI is left
    untouched. Framed as the fix-for-good next to the shim's fix-this-run."""
    return (
        "This is a recurring upstream ComfyUI bug. For a durable fix - not just a "
        "patch for this run - localm can set up its OWN managed, patched ComfyUI so "
        "it cannot recur; your own ComfyUI is left untouched. To set it up:\n"
        "  localm comfy setup")


def mark_managed_comfy_setup_offered() -> None:
    """Persist ``comfy_managed_setup_offered=True`` so the durable-fix offer is made
    at most ONCE (it never nags). Written directly (like the shim's
    ``comfy_func_shim``), not through the settings form. Best-effort: a failure to
    persist must not break the media error path (worst case the offer reappears next
    time - annoying, not harmful), so it is logged at debug, never muted blind."""
    try:
        from localm.config import load_config, save_config
        cfg = load_config()
        cfg[_MANAGED_SETUP_OFFERED_KEY] = True
        save_config(cfg)
    except Exception:
        from localm.debuglog import logger
        logger.debug("could not persist %s (the managed-ComfyUI offer may show "
                     "again next time)", _MANAGED_SETUP_OFFERED_KEY, exc_info=True)


def _derive_workdir_from_cmd(launch_cmd: str) -> Optional[str]:
    """The folder of the launcher script, so a .bat / .sh that references paths
    relative to its own location (the ComfyUI + ZLUDA convention, e.g. a copied
    ``launch-comfyui.bat`` next to ``python_embeded`` / ``venv``) runs from the
    right place even when ``comfy_workdir`` was not set. Best-effort: returns the
    parent of the first token that is an existing file, else None."""
    import shlex
    try:
        tokens = shlex.split(launch_cmd, posix=(os.name != "nt"))
    except ValueError:
        tokens = launch_cmd.split()
    for tok in tokens:
        tok = tok.strip().strip('"').strip("'")
        if not tok:
            continue
        try:
            p = Path(tok)
        except Exception:
            break
        if p.is_file():
            return str(p.parent)
        break  # only the first token is the program/script being launched
    return None


# Common ComfyUI launcher scripts, in priority order. A user's own
# launch-comfyui.* (the convention from the ComfyUI + ZLUDA community, and what
# this repo's own issue reporter hand-made) is preferred over the stock launcher,
# so localm uses the setup the user already has instead of imposing its own.
_LAUNCHER_NAMES_WIN = (
    "launch-comfyui.bat", "comfyui.bat", "run_nvidia_gpu.bat",
    "run_amd_gpu.bat", "run_cpu.bat", "run.bat",
)
_LAUNCHER_NAMES_POSIX = (
    "launch-comfyui.sh", "comfyui.sh", "run_nvidia_gpu.sh",
    "run_amd_gpu.sh", "run_cpu.sh", "run.sh",
)


def _venv_python(folder: Path) -> Optional[Path]:
    """A ComfyUI install's own venv interpreter, if present (venv/ or .venv/)."""
    if os.name == "nt":
        cands = [folder / "venv" / "Scripts" / "python.exe",
                 folder / ".venv" / "Scripts" / "python.exe"]
    else:
        cands = [folder / "venv" / "bin" / "python",
                 folder / ".venv" / "bin" / "python"]
    for c in cands:
        if c.is_file():
            return c
    return None


def discover_launch_cmd(folder: Path) -> Optional[str]:
    """Build a launch command for an existing ComfyUI install in *folder*.

    Prefers a launcher script the user already has (their own launch-comfyui.*,
    else the stock comfyui.* / run.*), falling back to running ``main.py`` with
    the install's own venv. Returns an absolute, quoted command string (run with
    *folder* as the working dir) or None when nothing recognizable is found.
    Absolute paths keep it cwd-independent and cross-platform (no PATH lookup,
    no bare-name resolution differences between cmd.exe and a POSIX shell)."""
    try:
        if not folder.is_dir():
            return None
    except OSError:
        return None
    names = _LAUNCHER_NAMES_WIN if os.name == "nt" else _LAUNCHER_NAMES_POSIX
    for name in names:
        p = folder / name
        if p.is_file():
            return f'"{p}"'
    py = _venv_python(folder)
    main = folder / "main.py"
    if py and main.is_file():
        return f'"{py}" "{main}"'
    return None


def _amd_rocm_launch_env() -> Optional[dict]:
    """Child env for the ComfyUI launch with ROCm's bin on PATH, or None to inherit.

    ZLUDA's ``cublas64_11.dll`` / ``cusparse64_11.dll`` shims load rocBLAS / rocSPARSE
    from ``%HIP_PATH%\\bin`` at ``import torch`` time. A localm process spawned from a
    context that has only HIP_PATH set (not ROCm\\bin on PATH) would otherwise make a
    ZLUDA ComfyUI die importing torch (WinError 126 on cublas64_11.dll). No-op off
    Windows, without HIP_PATH, when the bin is missing, or when it is already on PATH.
    """
    import sys as _sys
    if _sys.platform != "win32":
        return None
    hip = os.environ.get("HIP_PATH")
    if not hip:
        return None
    rocm_bin = os.path.join(hip, "bin")
    if not os.path.isdir(rocm_bin):
        return None
    cur = os.environ.get("PATH", "")
    if any(rocm_bin.lower() == p.strip().lower() for p in cur.split(os.pathsep) if p):
        return None
    env = dict(os.environ)
    env["PATH"] = rocm_bin + os.pathsep + cur
    return env


# ---------------------------------------------------------------------------
#  Reactive __func__ shim: apply ONLY to a ComfyUI localm spawns (MEDIA-1)
# ---------------------------------------------------------------------------
#
#  The fix is a localm-owned sitecustomize.py in comfy_shim/. localm never writes it
#  into the user's ComfyUI install: it only adds the shim DIRECTORY to the PYTHONPATH
#  of a ComfyUI process localm ITSELF spawns, so the interpreter auto-imports it and
#  patches the regression in memory. Default = off (no shim on PYTHONPATH). It turns
#  on only per the reactive offer: `enable_func_shim_once()` for this process, or the
#  persistent `comfy_func_shim` config for every future localm-spawned ComfyUI. If
#  localm did not spawn ComfyUI, the shim is simply absent.

# Process-local one-shot: "apply once" for this run without persisting a preference.
_func_shim_once = False


def comfy_shim_dir() -> Path:
    """The localm-owned directory holding the ComfyUI __func__ compatibility
    sitecustomize.py. Always inside the localm package, never in a ComfyUI folder."""
    return Path(__file__).resolve().parent / "comfy_shim"


def enable_func_shim_once() -> None:
    """Arrange for the NEXT ComfyUI that localm spawns to get the shim on its child
    PYTHONPATH, for this process only (does not persist a preference)."""
    global _func_shim_once
    _func_shim_once = True


def func_shim_enabled(cfg: Optional[dict] = None) -> bool:
    """Whether a ComfyUI localm spawns should get the shim on its child PYTHONPATH:
    the process one-shot, or the persistent ``comfy_func_shim`` config. Default off."""
    if _func_shim_once:
        return True
    if cfg is None:
        try:
            from localm.config import load_config
            cfg = load_config()
        except Exception:
            cfg = {}
    try:
        return bool(cfg.get("comfy_func_shim"))
    except AttributeError:
        return False


def comfy_child_env(cfg: Optional[dict] = None) -> Optional[dict]:
    """The environment for a ComfyUI process localm SPAWNS. Starts from the AMD/ROCm
    launch env (or the inherited env) and, ONLY when the shim is enabled, PREPENDS the
    localm-owned shim dir to PYTHONPATH (preserving any pre-existing PYTHONPATH). When
    the shim is off this returns exactly what the launch used before (None to inherit,
    or the AMD env), so a normal run is untouched. Never writes anything to disk."""
    base = _amd_rocm_launch_env()
    if not func_shim_enabled(cfg):
        return base
    env = base if base is not None else dict(os.environ)
    shim = str(comfy_shim_dir())
    prev = env.get("PYTHONPATH", "")
    # Avoid a duplicate entry if we are re-spawning a ComfyUI that already had it.
    if prev.split(os.pathsep)[:1] == [shim]:
        env["PYTHONPATH"] = prev
    else:
        env["PYTHONPATH"] = shim + (os.pathsep + prev if prev else "")
    return env


# NEW-STOPCOMFY: the ComfyUI processes localm itself launched, keyed by api_url,
# so we can terminate/restart the one WE spawned (and never a ComfyUI the user
# started themselves - we only have handles to our own). Guarded by a lock because
# a launch and a stop can race across request threads.
import threading as _threading

_spawned_procs: dict = {}
_spawned_lock = _threading.Lock()


def _remember_spawned(api_url: str, proc) -> None:
    with _spawned_lock:
        _spawned_procs[api_url] = proc


def _take_spawned(api_url: str):
    """Pop and return the proc localm launched for *api_url*, or None."""
    with _spawned_lock:
        return _spawned_procs.pop(api_url, None)


def spawned_pid(api_url: Optional[str] = None) -> Optional[int]:
    """The PID of the ComfyUI localm launched for *api_url* if it is still ours
    and running, else None (localm did not launch it, or it has exited)."""
    api_url = (api_url or default_api_url()).rstrip("/")
    with _spawned_lock:
        proc = _spawned_procs.get(api_url)
    if proc is None:
        return None
    try:
        return proc.pid if proc.poll() is None else None
    except Exception:
        return None


def _kill_process_tree(proc) -> None:
    """Terminate *proc* AND its children. On Windows the launcher we spawn is a
    `cmd /S /c "<bat>"` whose real ComfyUI (python) is a CHILD, so terminating the
    cmd alone would orphan it - use taskkill /T. On POSIX the child was started in
    its own session (start_new_session), so signal the whole process group."""
    import os as _os
    import signal as _signal
    import subprocess as _sp
    import sys as _sys
    pid = getattr(proc, "pid", None)
    if not pid:
        return
    if _sys.platform == "win32":
        try:
            _sp.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=15)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return
    # POSIX: signal the process group, then wait, then hard-kill the group.
    try:
        pgid = _os.getpgid(pid)
    except Exception:
        pgid = None
    try:
        if pgid is not None:
            _os.killpg(pgid, _signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
        return
    except Exception:
        pass
    try:
        if pgid is not None:
            _os.killpg(pgid, _signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def stop_comfy(api_url: Optional[str] = None) -> tuple[bool, str]:
    """Stop ComfyUI (NEW-STOPCOMFY). Always aborts the in-flight render + clears
    the queue and frees VRAM first (graceful). Then, IF localm launched this
    ComfyUI, terminates the process tree we spawned; if localm did NOT launch it,
    the process is left alone (we only kill our own) and the caller is told so."""
    api_url = (api_url or default_api_url()).rstrip("/")
    mark_comfy_dead(api_url)   # about to stop it either way - the next check must be real
    interrupt_comfy(api_url)                       # abort render + clear queue
    try:
        free_comfy_vram(api_url)
    except Exception:
        pass
    proc = _take_spawned(api_url)
    if proc is None:
        if _comfy_alive(api_url, timeout=2.0):
            mark_comfy_alive(api_url)   # localm did not launch it and did not stop it
            return True, ("Aborted the in-flight render and cleared the queue. "
                          "localm did not launch this ComfyUI, so its process was "
                          "left running - stop it where you started it.")
        return True, "ComfyUI is not running."
    try:
        if proc.poll() is None:
            _kill_process_tree(proc)
    except Exception as e:
        return False, f"Could not stop the ComfyUI localm launched: {e}"
    return True, "Stopped the ComfyUI that localm launched."


def restart_comfy(api_url: Optional[str] = None, on_progress=None,
                  wait_seconds: Optional[int] = None,
                  launch_cmd: Optional[str] = None,
                  workdir: Optional[str] = None) -> tuple[bool, str]:
    """Stop the ComfyUI localm launched (if any), then launch a fresh one
    (NEW-STOPCOMFY). Only meaningful when localm has a launch command configured."""
    stop_comfy(api_url)
    return ensure_comfy(api_url=api_url, on_progress=on_progress,
                        wait_seconds=wait_seconds, launch_cmd=launch_cmd,
                        workdir=workdir)


def ensure_comfy(api_url: Optional[str] = None, on_progress=None,
                 wait_seconds: Optional[int] = None,
                 launch_cmd: Optional[str] = None,
                 workdir: Optional[str] = None) -> tuple[bool, str]:
    """
    Make sure ComfyUI is reachable, launching it when configured.

    Used by every generator (image, music, video) from any caller - GUI,
    CLI, or the coder's generate_image tool. When ComfyUI is down and a launch
    command is configured, the command is started (optionally in *workdir*) and
    polled until the API answers.

    *launch_cmd* / *workdir* let a caller pass per-plugin config; when not given
    they fall back to the global ``comfy_launch_cmd`` / ``comfy_workdir`` config
    keys (kept for callers not yet migrated to per-plugin config).

    Returns (ok, message); the message explains what to configure when
    nothing could be launched.
    """
    import shlex
    import subprocess
    import sys as _sys
    import time as _t
    from localm.config import load_config

    def _say(text: str) -> None:
        if on_progress:
            try:
                on_progress(text)
            except Exception:
                pass

    api_url = (api_url or default_api_url()).rstrip("/")
    if is_comfy_confirmed(api_url):
        return True, "ComfyUI is running."
    if _comfy_alive(api_url):
        mark_comfy_alive(api_url)
        return True, "ComfyUI is running."
    mark_comfy_dead(api_url)

    cfg = load_config()

    # Resolve the ComfyUI folder (working dir) FIRST: explicit arg, then config.
    # It anchors both launcher discovery and the cwd a relative launcher name
    # runs from, so a bare "launch-comfyui.bat" works once the folder is known.
    if workdir is None:
        workdir = cfg.get("comfy_workdir")

    # Resolve the launch command: explicit arg, then config, then - when the
    # ComfyUI folder is known - auto-discover a launcher inside it (the user's
    # own launch-comfyui.bat, else the stock comfyui.bat / run.bat). This is the
    # "work with the install the user already has" path: pointing localm at the
    # ComfyUI folder is enough; naming a script is optional.
    if not launch_cmd:
        launch_cmd = cfg.get("comfy_launch_cmd")
    discovered = False
    if not launch_cmd and workdir:
        found = discover_launch_cmd(Path(workdir))
        if found:
            launch_cmd, discovered = found, True
    if not launch_cmd:
        return False, (
            f"ComfyUI is not reachable at {api_url}.\n"
            "Point localm at your ComfyUI install so it can start it for you:\n"
            '  localm config comfy_workdir "<path-to-your-ComfyUI-folder>"\n'
            "localm then runs the launcher it finds there (your own "
            "launch-comfyui.bat, or the stock comfyui.bat / run.bat).\n"
            "To name a specific launcher instead of auto-detecting:\n"
            '  localm config comfy_launch_cmd "<path>\\launch-comfyui.bat"\n'
            "Or start ComfyUI yourself first (default http://127.0.0.1:8188), or "
            "set the FLUX_API_URL environment variable if it runs elsewhere."
        )

    # A ZLUDA / ROCm cold start compiles GPU kernels and can take minutes, so
    # honour the configurable timeout when the caller did not pin one.
    if wait_seconds is None:
        try:
            wait_seconds = int(cfg.get("comfy_launch_timeout") or 300)
        except (TypeError, ValueError):
            wait_seconds = 300
    wait_seconds = max(30, wait_seconds)

    # MEDIA-2: optionally start ComfyUI headless. Off by default (keep the
    # current behavior); when comfy_disable_auto_launch is set, append ComfyUI's
    # --disable-auto-launch so it does not pop open its own web page (localm has
    # its own GUI). Shared by image/music/video. The stock run_*.bat / comfyui.*
    # and a bare "python main.py" forward extra args to main.py; a launcher that
    # drops args simply ignores the flag (no error), so this stays non-breaking.
    if cfg.get("comfy_disable_auto_launch") and \
            "--disable-auto-launch" not in launch_cmd:
        launch_cmd = launch_cmd + " --disable-auto-launch"

    _say(f"ComfyUI not running - launching: {launch_cmd}")
    if discovered:
        _say(f"Found a ComfyUI launcher in {workdir}")
    # The command is the user's own config value (their launcher script).
    # On Windows pass `cmd /S /c "<line>"` as a single string: /S strips the
    # outer quotes and runs the line verbatim, so quoted executable paths
    # survive (a `["cmd", "/c", line]` list gets re-quoted by subprocess and
    # mangles them). POSIX uses shlex.
    if not workdir:
        # No configured ComfyUI folder: fall back to the launcher file's own
        # folder so a .bat/.sh that uses paths relative to itself (ComfyUI +
        # ZLUDA) still works when the user gave an absolute launcher path.
        workdir = _derive_workdir_from_cmd(launch_cmd)
        if workdir:
            _say(f"Running the launcher from {workdir}")
    workdir = workdir or None
    argv = shlex.split(launch_cmd, posix=(_sys.platform != "win32"))
    if _sys.platform == "win32":
        argv = [a.strip('"\'') for a in argv]
        if argv and (argv[0].lower().endswith(".bat") or argv[0].lower().endswith(".cmd")):
            argv = ["cmd", "/d", "/c"] + argv
    # Redirect the launcher's own stdout+stderr to a log file instead of
    # discarding them, so a ComfyUI that fails to start leaves its reason on
    # disk for the user (and for --debug-discoverable). Best-effort: fall back
    # to DEVNULL if the log cannot be opened, so launching still proceeds.
    from localm.config import home_dir
    launch_log_path = home_dir() / "comfy-launch.log"
    try:
        launch_out = open(launch_log_path, "w", encoding="utf-8", errors="replace")
    except OSError:
        launch_out = subprocess.DEVNULL
        launch_log_path = None
    # NEW-STOPCOMFY: start the launcher in its OWN process group/session so the
    # whole ComfyUI tree (the launcher + the python it spawns) can later be
    # terminated together (see stop_comfy / _kill_process_tree). Harmless to the
    # normal run; only changes signal grouping.
    _popen_kw: dict = {}
    if _sys.platform == "win32":
        _popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        _popen_kw["start_new_session"] = True
    try:
        proc = subprocess.Popen(argv, cwd=workdir,
                         env=comfy_child_env(cfg),
                         stdout=launch_out,
                         stderr=subprocess.STDOUT,
                         **_popen_kw)
        _remember_spawned(api_url, proc)   # retain the handle so Stop can reach it
        _t.sleep(0.5)
        if proc.poll() is not None and proc.returncode != 0:
            _take_spawned(api_url)         # it died; drop the dead handle
            return False, f"ComfyUI launcher exited immediately with code {proc.returncode}"
    except Exception as e:
        return False, f"Could not launch ComfyUI ({launch_cmd}): {e}"
    finally:
        # Close the PARENT's copy of the log handle on every path: the child
        # inherited its own dup'd descriptor, so leaving this open would leak a
        # file handle per launch (and on Windows hold a write lock the next
        # launch's re-open would contend with). DEVNULL is a sentinel, not a file.
        if hasattr(launch_out, "close"):
            launch_out.close()

    deadline = _t.monotonic() + wait_seconds
    last_said = 0.0
    while _t.monotonic() < deadline:
        if _comfy_alive(api_url):
            mark_comfy_alive(api_url)
            return True, "ComfyUI is up."
        elapsed = wait_seconds - (deadline - _t.monotonic())
        if elapsed - last_said >= 15:
            _say(f"Waiting for ComfyUI… ({int(elapsed)}s)")
            last_said = elapsed
        _t.sleep(2)
    # Point the user at the captured launcher log (when we managed to open one):
    # a ComfyUI that died on startup wrote its own error there, not to the window.
    log_hint = (f" The launcher's own output was captured to {launch_log_path} - "
                "check it for the reason it failed to start." if launch_log_path else "")
    return False, (
        f"ComfyUI did not come up within {wait_seconds // 60} minutes - "
        "check the launcher window for errors. If it is just a slow first "
        "(ZLUDA / ROCm) start, raise the limit: "
        "localm config comfy_launch_timeout 600"
        f"{log_hint}"
    )


def comfy_http_error_detail(e: "urllib.error.HTTPError") -> str:
    """
    Human-readable detail from a ComfyUI /prompt error response.

    A 400 from /prompt means the workflow failed validation - not a
    connectivity problem. The response body is JSON naming the failing
    node and why (a model file missing from ComfyUI's models directory
    is the usual cause). Shared by image, music, and video generation.
    """
    try:
        body = json.loads(e.read().decode("utf-8", "replace"))
    except Exception:
        return f"HTTP {e.code}: {e.reason}"
    lines = []
    err = body.get("error") or {}
    if err.get("message"):
        msg = err["message"]
        if err.get("details"):
            msg += f" - {err['details']}"
        lines.append(msg)
    for node_id, info in (body.get("node_errors") or {}).items():
        cls = info.get("class_type") or f"node {node_id}"
        for ne in info.get("errors", []):
            msg = ne.get("message", "")
            if ne.get("details"):
                msg += f" ({ne['details']})"
            lines.append(f"{cls}: {msg}")
    return "\n".join(lines) or f"HTTP {e.code}: {e.reason}"


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) from a PNG or JPEG without any external libs."""
    try:
        with open(path, "rb") as f:
            data = f.read(32)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if data[:2] == b"\xff\xd8":
            # JPEG - scan for SOF0/SOF2 markers
            full = path.read_bytes()
            i = 2
            while i < len(full) - 8:
                if full[i] != 0xFF:
                    break
                marker = full[i + 1]
                length = int.from_bytes(full[i + 2:i + 4], "big")
                if marker in (0xC0, 0xC2):
                    h = int.from_bytes(full[i + 5:i + 7], "big")
                    w = int.from_bytes(full[i + 7:i + 9], "big")
                    return w, h
                i += 2 + length
    except Exception:
        pass
    return 1024, 1024


def _upload_image(image_path: Path, api_url: str) -> str:
    """
    Upload a local image to ComfyUI via POST /upload/image.

    Returns the filename ComfyUI assigned (used in the LoadImage node).
    Raises on failure.
    """
    boundary = "LocalcoderUploadBoundary"
    img_bytes = image_path.read_bytes()
    content_type = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{api_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
    name = result.get("name")
    if not name:
        raise RuntimeError(f"ComfyUI upload returned no filename: {result}")
    return name


# ---------------------------------------------------------------------------
#  Output containment
# ---------------------------------------------------------------------------
#
#  Default: localm LEAVES ComfyUI's own copies alone. A user may run ComfyUI for
#  its own gallery and legitimately want the generated files and the /history
#  entry, so deleting them is NOT the default. Containment is opt-in (the
#  comfy_delete_outputs config / a per-plugin delete_outputs), and privacy mode
#  forces it on (no trace anywhere). When enabled, after the artifact has been
#  fetched into localm's own location we clear ComfyUI's /history entry AND delete
#  its on-disk copy of the output plus any img2img source we uploaded into input/.
#  Deleting files needs ComfyUI's output/ dir; when it cannot be resolved we
#  return a loud warning rather than silently leaving a copy the user asked to
#  remove. Shared by image, music, and video generation.

def _comfy_output_root(comfy_output_dir: Optional[str] = None) -> Optional[Path]:
    """ComfyUI's output/ directory, or None when it cannot be resolved.

    Order: explicit arg, COMFY_OUTPUT_DIR env, the ``comfy_output_dir`` config
    key, then a derived ``<comfy_workdir>/output`` when that exists."""
    cand = comfy_output_dir or os.environ.get("COMFY_OUTPUT_DIR")
    if not cand:
        try:
            from localm.config import load_config
            cfg = load_config()
            cand = cfg.get("comfy_output_dir")
            if not cand:
                wd = cfg.get("comfy_workdir")
                if wd:
                    derived = Path(wd) / "output"
                    if derived.is_dir():
                        return derived
        except Exception:
            return None
    return Path(cand) if cand else None


def clear_comfy_history(api_url: str, prompt_id: str) -> bool:
    """Remove this job from ComfyUI's /history (the Queue/History panel and
    gallery views read it). POST /history {"delete": [prompt_id]}. Best-effort;
    returns True when ComfyUI accepted the request."""
    if not prompt_id:
        return False
    try:
        body = json.dumps({"delete": [prompt_id]}).encode()
        req = urllib.request.Request(
            f"{api_url}/history", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception:
        return False


def contain_comfy_artifacts(
    api_url: str,
    prompt_id: str,
    info: dict,
    *,
    comfy_output_dir: Optional[str] = None,
    uploaded_input: Optional[str] = None,
    delete_outputs: bool = False,
) -> str:
    """Optionally remove ComfyUI's own copies of a generation.

    By DEFAULT (delete_outputs=False) this is a no-op: ComfyUI keeps its /history
    entry and its on-disk output, because a user may run ComfyUI for its own
    gallery and want them. When the user opts in (or privacy mode forces no-trace),
    it clears the history entry and deletes ComfyUI's duplicate output plus any
    img2img source we uploaded into input/. Returns a WARNING string when
    containment was requested but a copy could NOT be removed (so the user is told
    a copy remains), or "" otherwise."""
    if not delete_outputs:
        return ""   # keep ComfyUI's history + on-disk copy by default

    clear_comfy_history(api_url, prompt_id)
    root = _comfy_output_root(comfy_output_dir)

    warnings: list = []
    # Remove the uploaded img2img source from ComfyUI's input/ dir (sibling of
    # output/). Surface a failure (do not silence): it is still a stray copy of
    # the user's input that they asked to contain.
    if uploaded_input and root is not None:
        try:
            inp = root.parent / "input" / uploaded_input
            if inp.exists():
                inp.unlink()
        except OSError as e:
            warnings.append(
                f"a copy of your input image remains in ComfyUI's input folder ({e})")

    # Delete ComfyUI's on-disk copy of the output. type "temp" is auto-purged
    # by ComfyUI, so only a real "output" artifact needs removing.
    if (info.get("type") or "output") == "output":
        if root is None:
            warnings.append(
                "a copy of this file remains in ComfyUI's output folder; localm "
                "cleared the ComfyUI history entry but cannot delete the file "
                "until you set the ComfyUI output dir (Settings -> Media, or: "
                "localm config comfy_output_dir <path>)")
        else:
            try:
                copy = root / info.get("subfolder", "") / info.get("filename", "")
                if copy.exists():
                    copy.unlink()
            except OSError as e:
                warnings.append(f"could not delete ComfyUI's copy of the output ({e})")

    return ("WARNING: " + "; ".join(warnings) + ".") if warnings else ""


def interrupt_comfy(api_url: str) -> bool:
    """Abort ComfyUI's currently running prompt and clear its queue (POST
    ``/interrupt`` + POST ``/queue {"clear": true}``).

    Best-effort, used to honour a user's Stop so a long media gen actually stops
    instead of running to completion. Shared by image, music, and video. Returns
    True when the interrupt was accepted."""
    ok = False
    try:
        req = urllib.request.Request(f"{api_url}/interrupt", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10):
            ok = True
    except Exception:
        pass
    try:
        body = json.dumps({"clear": True}).encode()
        req = urllib.request.Request(
            f"{api_url}/queue", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass
    return ok


def _with_warning(message: str, warning: str) -> str:
    """Append a containment warning to a success message when present."""
    return f"{message}\n{warning}" if warning else message


# ---------------------------------------------------------------------------
#  Shared transport: queue -> poll -> download
# ---------------------------------------------------------------------------
#
#  The three generators (image / video / music) share the exact same ComfyUI
#  transport: POST /prompt to queue, poll /history until the job finishes (with
#  the same cancel handling, the same execution-error check, the same
#  last-poll-error tracking on timeout), then GET /view to fetch the artifact.
#  Only the medium-specific bits differ (which output keys to read, the wording
#  of each error, how progress is shown), so those stay in each generator and the
#  shared loop below takes them as parameters. Behavior, retry cadence, timeouts,
#  and output selection are unchanged from the per-module originals.

# Sentinel "kind" values returned by comfy_submit_prompt so the caller can build
# its own medium-specific error text without this helper hardcoding any.
SUBMIT_OK = "ok"
SUBMIT_NO_ID = "no_prompt_id"
SUBMIT_HTTP_ERROR = "http_error"
SUBMIT_URL_ERROR = "url_error"
SUBMIT_ERROR = "error"


def comfy_submit_prompt(api_url: str, workflow: dict, *, timeout: float = 10.0):
    """Queue *workflow* in ComfyUI (POST /prompt) and return ``(kind, value)``.

    ``kind`` is one of the ``SUBMIT_*`` sentinels:
      - ``SUBMIT_OK``        -> value is the prompt_id (str)
      - ``SUBMIT_NO_ID``     -> value is None (accepted but no prompt_id returned)
      - ``SUBMIT_HTTP_ERROR``-> value is the ``urllib.error.HTTPError``
      - ``SUBMIT_URL_ERROR`` -> value is the ``urllib.error.URLError``
      - ``SUBMIT_ERROR``     -> value is the ``Exception``

    The caller maps each kind to its own (unchanged) message text. This mirrors
    the per-module try/except exactly: an HTTPError, a URLError, then a catch-all
    Exception, in that order."""
    try:
        req = urllib.request.Request(
            f"{api_url}/prompt",
            data=json.dumps({"prompt": workflow}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            prompt_id = json.loads(response.read().decode("utf-8")).get("prompt_id")
        if not prompt_id:
            return SUBMIT_NO_ID, None
        return SUBMIT_OK, prompt_id
    except urllib.error.HTTPError as e:
        return SUBMIT_HTTP_ERROR, e
    except urllib.error.URLError as e:
        return SUBMIT_URL_ERROR, e
    except Exception as e:
        return SUBMIT_ERROR, e


# Poll outcomes (the "status" element of comfy_poll_until_done's return).
POLL_CANCELLED = "cancelled"
POLL_EXEC_ERROR = "exec_error"
POLL_FINISHED = "finished"
POLL_TIMEOUT = "timeout"


def comfy_poll_until_done(
    api_url: str,
    prompt_id: str,
    *,
    max_poll_seconds: float,
    cancel_check=None,
    on_tick=None,
    history_timeout: float = 5.0,
    sleep_seconds: float = 2.0,
):
    """Poll ComfyUI ``/history/<prompt_id>`` until the job finishes, is cancelled,
    errors, or times out. Returns ``(status, value)``:

      - ``POLL_CANCELLED``  -> value None. ``cancel_check()`` returned truthy; this
        helper has already issued ``interrupt_comfy`` + ``clear_comfy_history``.
      - ``POLL_EXEC_ERROR`` -> value is the ``history_execution_error`` detail (str).
      - ``POLL_FINISHED``   -> value is the finished ``history[prompt_id]`` entry
        (dict); the caller reads its ``"outputs"`` for the medium's artifact.
      - ``POLL_TIMEOUT``    -> value is the last poll Exception (or None if none).

    ``on_tick(elapsed_seconds)`` is called once per iteration BEFORE the /history
    request (``elapsed_seconds`` is ``time.time() - start_time``), so the caller can
    drive its own progress UI. Retry cadence, the in-loop exception swallow + retry,
    and the timeout-with-last-error semantics match the per-module originals."""
    import time
    start_time = time.time()
    last_poll_err: Optional[Exception] = None
    while time.time() - start_time < max_poll_seconds:
        if cancel_check and cancel_check():
            interrupt_comfy(api_url)
            clear_comfy_history(api_url, prompt_id)
            return POLL_CANCELLED, None
        if on_tick is not None:
            on_tick(time.time() - start_time)
        try:
            hist_req = urllib.request.Request(f"{api_url}/history/{prompt_id}")
            with urllib.request.urlopen(hist_req, timeout=history_timeout) as response:
                history = json.loads(response.read().decode("utf-8"))
            if prompt_id in history:
                entry = history[prompt_id]
                err = history_execution_error(entry)
                if err:
                    return POLL_EXEC_ERROR, err
                return POLL_FINISHED, entry
        except Exception as e:
            # Keep retrying within the loop (ComfyUI may just be busy), but
            # remember the last failure so a crashed/unreachable ComfyUI is
            # distinguishable from a merely slow one if we time out below.
            last_poll_err = e
        time.sleep(sleep_seconds)
    return POLL_TIMEOUT, last_poll_err


def select_output_info(entry: dict, output_keys) -> Optional[dict]:
    """The first artifact info dict in a finished ``/history`` entry's outputs,
    scanning each node's output for any of *output_keys* (a non-empty list under
    that key yields its first element). Returns None when no node carries one.

    ``output_keys`` is the medium's save-node output keys, e.g. ``("images",)``
    for image, ``("videos", "gifs", "images")`` for video, ``("audio",)`` for
    music. The first key (in the given order) that a node has a non-empty list
    for wins, matching the per-module poll loops."""
    for node_output in entry.get("outputs", {}).values():
        for key in output_keys:
            if node_output.get(key):
                return node_output[key][0]
    return None


def comfy_fetch_output(api_url: str, info: dict, output_path: Path, *,
                       timeout: float) -> None:
    """Download a finished artifact from ComfyUI (GET /view?filename&subfolder&type)
    and write the bytes to *output_path* (creating parent dirs). Builds the query
    from *info*'s ``filename`` / ``subfolder`` / ``type`` (``type`` defaults to
    "output"). Raises on any failure - the caller wraps it in its own error text."""
    params = urllib.parse.urlencode({
        "filename": info.get("filename"),
        "subfolder": info.get("subfolder", ""),
        "type": info.get("type", "output"),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{api_url}/view?{params}", timeout=timeout) as response:
        output_path.write_bytes(response.read())
