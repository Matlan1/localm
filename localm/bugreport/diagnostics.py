# SPDX-License-Identifier: AGPL-3.0-or-later
"""The environment and app-state snapshot a report carries: platform, GPUs,
native runtime, loaded model, session mode, plugins, the allowlisted config
subset, dependency versions, and the in-memory recent-activity log.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Optional

from localm.bugreport.scrub import (
    _scrub_emails, _scrub_home, _scrub_query_and_header_secrets, _scrub_url_creds,
)


def _localm_version() -> str:
    # Live VERSION-file read, falling back to installed metadata.
    try:
        from localm._version import read_version
        return read_version()
    except Exception:
        return "unknown"


def _runtime_libs() -> tuple:
    """(binary_dir or None, sorted list of native library file NAMES present).
    Names only - never full paths beyond the resolved dir, never contents."""
    try:
        from localm.inference.backends.llamacpp._loader import (
            lib_filename, runtime_binary_dir)
        d = runtime_binary_dir()
        if d is None:
            return None, [], ""
        # Shared libraries only; the bundled llama-*.exe tools are skipped.
        names = sorted(
            f.name for f in Path(d).iterdir()
            if f.is_file() and (f.suffix.lower() in (".dll", ".so", ".dylib")
                                or ".so." in f.name)
        )
        return d, names, lib_filename()
    except Exception:
        return None, [], ""


# Shown for an environment field whose probe ran but returned nothing.
_NOT_DETECTED = "not detected"


def _gpu_inventory() -> list:
    """Every GPU in this process's last completed ``discover.list_gpus``
    reading, one line each: index, name, and free / total memory, numbered as
    torch numbers them (nvidia-smi's own order for a line marked "via
    nvidia-smi"). That is not llama.cpp's device numbering on a Vulkan or
    SYCL build, nor when an integrated GPU sits beside a discrete one. Never probes: a
    process that has not read its GPUs yet gets one ``not measured`` line, and
    an empty reading one ``not detected`` line. Never raises."""
    try:
        from localm.discover import last_gpu_reading
        gpus = last_gpu_reading()
    except Exception as e:
        return [f"not measured ({type(e).__name__})"]
    if gpus is None:
        return ["not measured in this process"]
    if not gpus:
        return [_NOT_DETECTED]
    gb = 1024 ** 3
    lines = []
    for g in gpus:
        if not isinstance(g, dict):
            continue
        free, total = g.get("free"), g.get("total")
        if isinstance(free, int) and isinstance(total, int):
            mem = f"{free / gb:.1f} of {total / gb:.1f} GB free"
        elif isinstance(total, int):
            mem = f"{total / gb:.1f} GB total, free {_NOT_DETECTED}"
        else:
            mem = f"memory {_NOT_DETECTED}"
        src = f" via {g['source']}" if g.get("source") else ""
        lines.append(f"{g.get('index')}: {g.get('name') or _NOT_DETECTED} "
                     f"({mem}{src})")
    return lines


def collect_diagnostics(context: Optional[dict] = None) -> dict:
    """Gather a safe, useful environment snapshot. Never raises."""
    context = dict(context or {})
    diag: dict = {}
    try:
        diag["localm_version"] = _localm_version()
        diag["python"] = sys.version.split()[0]
        diag["platform"] = platform.platform()
        diag["machine"] = platform.machine()
    except Exception:
        # Omit these basic fields on failure; the rest of the snapshot is
        # gathered by the independently guarded blocks below.
        pass

    try:
        from localm import hwdetect
        det = hwdetect.detect()
        diag["gpu_vendors"] = det.vendors or []
        # recommended_install_backend, NOT det.recommended: the latter is the
        # blanket "vulkan whenever any GPU is present" value and can contradict
        # what the installer would actually provision.
        diag["recommended_backend"] = hwdetect.recommended_install_backend(det)
        diag["detect_source"] = det.source
    except Exception:
        diag["gpu_vendors"] = []

    # NVIDIA driver / CUDA capability, when an NVIDIA GPU is involved.
    try:
        from localm.setup_llama import nvidia_preflight
        if "nvidia" in diag.get("gpu_vendors", []) or context.get("backend") == "cuda":
            nv = nvidia_preflight()
            if nv.present:
                diag["nvidia_gpu"] = nv.gpu_name or _NOT_DETECTED
                diag["nvidia_driver"] = nv.driver_version or _NOT_DETECTED
                diag["nvidia_cuda_capability"] = nv.cuda_capability or _NOT_DETECTED
                diag["nvidia_compute_capability"] = nv.compute_capability or _NOT_DETECTED
                diag["nvidia_cuda_line"] = nv.cuda_line
                diag["nvidia_gpus"] = [
                    f"{g['index']}: {g['name'] or _NOT_DETECTED} "
                    f"({g['free_mib'] / 1024:.1f} of {g['total_mib'] / 1024:.1f} GB free)"
                    for g in (getattr(nv, "gpus", None) or [])
                ] or [_NOT_DETECTED]
    except Exception:
        # Omit the NVIDIA fields on failure (no nvidia-smi, a driver hiccup).
        pass

    diag["gpus"] = _gpu_inventory()

    try:
        res = _runtime_libs()
        binary_dir = res[0]
        names = res[1]
        diag["native_runtime_provisioned"] = binary_dir is not None
        diag["native_libs"] = names
    except Exception:
        # Omit the native library names when the runtime dir is missing or
        # unreadable.
        pass

    # WHICH llama.cpp build is installed, and whether the user has pinned one.
    # Separately guarded from the library listing above so one failing does not
    # cost the other. Recorded at provision time by setup_llama, so this is a
    # lookup, not a probe.
    try:
        from localm import setup_llama
        build = setup_llama.installed_build()
        backend = setup_llama.installed_backend()
        if backend:
            diag["native_runtime_backend"] = backend
        # "not recorded" rather than an omitted field, so an install with no
        # recorded tag stays distinguishable from a failed collection.
        diag["native_runtime_build"] = build or "not recorded"
        pin = setup_llama.pinned_tag()
        if pin:
            diag["native_runtime_pin"] = pin
    except Exception:
        # Omit these fields when the marker/config is unreadable or setup_llama
        # cannot import on a broken install.
        pass

    # Caller-supplied context (chosen backend, the operation, etc.).
    for k in ("operation", "backend", "requested_backend", "with_cudart"):
        if k in context:
            diag[k] = context[k]

    # App / runtime state. Each collector is independently guarded and returns
    # empty on failure, so a missing piece never costs the rest of the report.
    diag["loaded_model"] = _loaded_model_info()
    diag.update(_runtime_state())          # session_mode, debug_mode
    diag["enabled_plugins"] = _enabled_plugins()
    diag["config_subset"] = _safe_config_subset()
    diag["config_unreadable"] = _config_unreadable()
    diag["dependencies"] = _dependency_versions()
    return diag


# Config keys SAFE to echo into a report and useful for debugging a setup. An
# allowlist (never the whole config) so a future secret-bearing key cannot be
# auto-leaked; path-like values are home-scrubbed and URL-like values
# credential-scrubbed before they are rendered. The API key lives in auth.key,
# never config.json.
_SAFE_CONFIG_KEYS = (
    "binary_dir", "n_ctx", "n_ctx_max", "n_ctx_grow", "ctx_auto", "n_gpu_layers",
    "n_gpu_layers_auto", "n_cpu_moe", "use_mmap", "mtp_enabled", "mtp_draft_tokens",
    "spec_source", "spec_draft_tokens", "spec_draft_model", "main_gpu_index",
    "gpu_split_indices", "gpu_split_ratios", "max_tokens",
    "model_swap_policy", "idle_unload_seconds", "reload_llm_after_imagine",
    "port", "require_auth", "cors_origins", "mode", "chat_mode", "coder_mode",
    "net_mode", "comfy_launch_cmd", "comfy_workdir", "comfy_api_url",
    "net_search_url", "coder_reviewer", "coder_review", "voice_stt_model",
)


# Distributions whose versions actually shape a localm failure. Optional / absent
# packages are simply skipped, so this stays correct across install profiles.
_REPORTED_DISTS = (
    "localm", "localm-llama-runtime", "llama-cpp-python", "torch", "transformers",
    "huggingface-hub", "fastapi", "uvicorn", "ctranslate2", "faster-whisper", "av",
)


def _loaded_model_info() -> dict:
    """Snapshot of the in-process inference engine (model, backend, loaded?, ctx).

    Empty when no server engine is loaded in THIS process (e.g. a CLI
    ``bug-report``) or it cannot be read. Never raises."""
    try:
        from localm.inference import http_server as hs
        eng = getattr(hs, "_engine", None)
        if eng is None:
            return {}
        info: dict = {
            "model": getattr(eng, "display_name", "") or "(unnamed)",
            "loaded": bool(getattr(eng, "loaded", False)),
        }
        backend = getattr(eng, "_backend", None)
        if backend is not None:
            info["backend"] = type(backend).__name__
        ctx = getattr(eng, "effective_ctx_max", None)
        if ctx:
            info["effective_ctx_max"] = ctx
        return info
    except Exception:
        return {}


def _runtime_state() -> dict:
    """Session mode + debug flag. These explain WHY a report may carry little
    history: privacy mode writes nothing, and a non-debug run keeps no log file.
    Never raises."""
    state: dict = {}
    try:
        from localm.audit import effective_mode
        state["session_mode"] = effective_mode("server").value
    except Exception:
        # Omit the session mode when it cannot be read. This is a status
        # readout, NOT the privacy-mode gate that protects session writes.
        pass
    try:
        from localm.debuglog import debug_enabled
        state["debug_mode"] = bool(debug_enabled())
    except Exception:
        # Omit the debug flag when it cannot be read.
        pass
    return state


def _enabled_plugins() -> list:
    """Names of enabled engine plugins, from config. Never raises."""
    try:
        from localm.config import load_config
        pl = load_config().get("plugins_enabled", [])
        return [str(p) for p in pl] if isinstance(pl, list) else []
    except Exception:
        return []


def _safe_config_subset() -> dict:
    """An allowlisted, scrubbed slice of the user config: enough to debug a setup
    problem (context sizing, ports, media URLs, privacy mode) without any secret.
    Never raises."""
    try:
        from localm.config import load_config
        cfg = load_config()
    except Exception:
        return {}
    out: dict = {}
    for key in _SAFE_CONFIG_KEYS:
        if key not in cfg:
            continue
        val = cfg[key]
        if isinstance(val, str) and val:
            # comfy_api_url / net_search_url / coder_reviewer routinely carry a
            # credential as a query parameter (?api_key=...), not only as
            # user:pass@ - see _scrub_query_and_header_secrets.
            val = _scrub_config_text(val)
        elif isinstance(val, (list, tuple)):
            val = [_scrub_config_text(v) if isinstance(v, str) else v for v in val]
        out[key] = val
    return out


def _scrub_config_text(val: str) -> str:
    """Home paths, URL credentials, credential-named parameters and email
    addresses removed from one config string."""
    return _scrub_emails(
        _scrub_query_and_header_secrets(_scrub_url_creds(_scrub_home(val))))


def _config_unreadable() -> bool:
    """True only when config.json (or its .bak) EXISTS and could not be parsed,
    in which case _safe_config_subset() is showing built-in defaults rather than
    the user's real settings - see config.load_config_checked. Independent of
    _safe_config_subset (which stays on load_config(), where the signal this
    needs is discarded), so a corrupt config does not render identically to no
    config at all. Never raises."""
    try:
        from localm.config import load_config_checked
        return not load_config_checked()[1]
    except Exception:
        # Could not even determine read_ok (e.g. a broken install failing the
        # import): say nothing rather than assert a corruption we never checked.
        return False


def _dependency_versions() -> dict:
    """Versions of the packages that shape a localm failure (native runtime, HF
    stack, server stack). Absent packages are omitted. Never raises."""
    out: dict = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return out
    for dist in _REPORTED_DISTS:
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return out


def _ring_activity() -> list:
    """The in-memory recent-activity buffer, or [] if unavailable. Never raises."""
    try:
        from localm.debuglog import recent_activity
        lines = recent_activity()
        try:
            from localm.config import home_dir
            pre_log = home_dir() / "logs" / "pre_restart.log"
            if pre_log.exists():
                lines = pre_log.read_text(encoding="utf-8").splitlines() + ["--- RESTART ---"] + lines
                try:
                    pre_log.unlink(missing_ok=True)
                except OSError as e:
                    # Privacy cleanup: the plaintext pre-restart log is deleted
                    # after being folded into the report. If the delete fails
                    # the file PERSISTS on disk, so the failure is logged rather
                    # than swallowed and the leftover stays discoverable.
                    from localm.debuglog import logger
                    logger.warning(
                        "bugreport: could not delete %s after reading it (%s); "
                        "the plaintext log remains on disk", pre_log, e)
        except Exception:
            # A missing or unreadable pre-restart log leaves the in-memory
            # activity on its own.
            pass
        return lines
    except Exception:
        return []
