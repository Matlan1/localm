# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hub kernel and network gate for the HF worker process.

Blocks the ``kernels`` package, through which transformers downloads and
imports code from the Hugging Face Hub (``lazy_load_kernel``,
``integrations.hub_kernels.get_kernel`` and direct ``from kernels import
get_kernel`` calls in quantizers, attention fallbacks and model files).
``sys.modules["kernels"] = None`` makes ``importlib.util.find_spec("kernels")``
return None and ``import kernels`` raise ``ModuleNotFoundError``, so
transformers treats the package as not installed for the rest of the process.

``close_hub_gate`` runs first thing in the worker, before anything imports
transformers or huggingface_hub. ``open_hub_kernels`` reverses the kernels
half for a load that is allowed to use a Hub kernel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from localm.debuglog import logger

KERNELS_MODULE = "kernels"
HF_ENDPOINT_DEFAULT = "https://huggingface.co"

# transformers quantization methods (other than "fp8", see _hf_fp8) whose
# native path loads a kernel from the Hub through the kernels package.
HUB_KERNEL_QUANT_METHODS = ("mxfp4", "eetq", "fbgemm_fp8", "metal")


def close_hub_gate() -> None:
    """Block the ``kernels`` package in this process and, when ``net_mode``
    is ``off``, put huggingface_hub in offline mode (``HF_HUB_OFFLINE=1``).

    Raises RuntimeError when ``kernels`` is already imported in this process.
    """
    existing = sys.modules.get(KERNELS_MODULE, False)
    if existing is not False and existing is not None:
        raise RuntimeError(
            "the HF worker could not block Hub kernels: the kernels package was "
            "imported before the gate was set up")
    # None in sys.modules blocks the import. See
    # test_closed_gate_hides_and_blocks_the_kernels_package.
    sys.modules[KERNELS_MODULE] = None  # pyright: ignore[reportArgumentType]
    logger.debug("hf worker: Hub kernels blocked")
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        set_hub_offline()


def set_hub_offline() -> None:
    """Make huggingface_hub refuse every network request in this process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    hub = sys.modules.get("huggingface_hub")
    if hub is not None:
        constants = getattr(hub, "constants", None)
        if constants is not None:
            constants.HF_HUB_OFFLINE = True
    logger.debug("hf worker: huggingface_hub offline (HF_HUB_OFFLINE=1)")


def hub_kernels_blocked() -> bool:
    """True while ``close_hub_gate`` is in force for ``kernels``."""
    return KERNELS_MODULE in sys.modules and sys.modules[KERNELS_MODULE] is None


def open_hub_kernels() -> None:
    """Make the ``kernels`` package importable again in this process. A no-op
    when the gate was never closed."""
    if hub_kernels_blocked():
        del sys.modules[KERNELS_MODULE]
        logger.debug("hf worker: Hub kernels allowed for this load")


def hub_fetch_refusal() -> str | None:
    """None when the network policy allows a request to the Hub endpoint
    (``HF_ENDPOINT``, default huggingface.co), else the policy's reason. An
    error while checking counts as a refusal."""
    from localm.netpolicy import NetworkPolicyError, check_url
    endpoint = os.environ.get("HF_ENDPOINT", "").strip() or HF_ENDPOINT_DEFAULT
    try:
        check_url(endpoint)
    except NetworkPolicyError as e:
        return str(e)
    except Exception as e:
        return f"the network policy could not be checked ({type(e).__name__}: {e})"
    return None


def record_kernel_version_ref(repo_id: str, version,
                              expected_commit: Optional[str] = None) -> Optional[Path]:
    """Write ``refs/v<version>`` (the commit of the loaded snapshot) into the
    Hub cache folder of *repo_id*'s kernel loaded in this process, which is
    what kernels reads to resolve a kernel version offline.

    A loaded kernel belongs to *repo_id* when its ``repo_info`` names it, or,
    when kernels recorded no ``repo_info`` (a kernel resolved from the local
    Hub cache), when its module sits in *repo_id*'s Hub cache folder. With
    *expected_commit*, only a kernel loaded from that commit's snapshot counts.

    Returns the ref written, or None when *version* is None, no kernel from
    *repo_id* is loaded, its module path is not inside a ``snapshots/<commit>``
    folder (for ``repo_info``'s commit, when there is one), or the ref already
    exists.
    """
    if version is None:
        return None
    from kernels import get_loaded_kernels
    cache_folder = "kernels--" + repo_id.replace("/", "--")
    for loaded in get_loaded_kernels():
        parts = Path(getattr(loaded.module, "__file__", "") or "").parts
        if "snapshots" not in parts:
            continue
        i = len(parts) - 1 - parts[::-1].index("snapshots")
        if i < 1 or i + 1 >= len(parts):
            continue
        commit = parts[i + 1]
        if expected_commit is not None and commit != expected_commit:
            continue
        info = loaded.repo_info
        if info is not None:
            if info.repo_id != repo_id or info.revision != commit:
                continue
        elif parts[i - 1] != cache_folder:
            continue
        ref = Path(*parts[:i]) / "refs" / f"v{version}"
        if ref.exists():
            return None
        ref.parent.mkdir(exist_ok=True)
        ref.write_text(commit, encoding="utf-8")
        logger.debug("hf worker: recorded %s -> %s for offline use", ref, commit)
        return ref
    return None
