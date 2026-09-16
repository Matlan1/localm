# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI model routes, discovery group: HuggingFace and CivitAI search and the
per-repo file listing with VRAM fit badges."""

from __future__ import annotations

import asyncio

from fastapi import Depends, FastAPI, HTTPException

from localm import scopes
from localm.inference.http_server import require_scope
from localm.executor import get_plugin_executor
from localm.plugins.gui.routes.models._context import ModelRouteContext


def register(app: FastAPI, context: ModelRouteContext) -> None:
    # Search HuggingFace or CivitAI and show per-quant/per-file "fits your VRAM"
    # (HF) or license/NSFW/scan-status (CivitAI) badges. net_mode=off blocks it.

    def _discover_status(e: Exception) -> int:
        msg = str(e)
        if "net_mode" in msg:
            return 403          # blocked by the network kill switch
        if "request failed" in msg:
            return 502          # HF/CivitAI unreachable
        return 422              # bad repo / no files / bad format token / bad type

    async def _run_discover(fn):
        """Run *fn* off the event loop; map DiscoverError/ModelSourceError to its
        HTTP status.

        A browser has no CLI to run, so a net_mode=off refusal (`e.off`) gets its
        own GUI-native remedy here rather than the message's own CLI-flavored one
        (`localm config net_mode ask`)."""
        from localm.discover import DiscoverError
        from localm.model_manager.sources import ModelSourceError
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(get_plugin_executor(), fn)
        except (DiscoverError, ModelSourceError) as e:
            if e.off:
                raise HTTPException(
                    _discover_status(e),
                    "Network access is off. Turn it on, or allow downloads "
                    "only, in Settings → Network.")
            raise HTTPException(_discover_status(e), str(e))

    async def _vram_total():
        """Off-thread vram_capacity() plus its extracted 'total' bytes, both
        of which discover_search/discover_files feed into fit_label().

        The returned dict's `free` is withheld unless the reading is BOTH fresh
        and device-global (sysstats._vram_reading_trusted, the same gate
        /api/vram-estimate and /api/stats apply). `total` is a static hardware
        fact and stands even under a stale or process-scoped probe."""
        from localm.discover import vram_capacity
        from localm.sysstats import _vram_reading_trusted
        loop = asyncio.get_running_loop()
        info, status = await loop.run_in_executor(
            get_plugin_executor(), lambda: vram_capacity(return_status=True))
        vram = {"total": info.get("total")}
        if _vram_reading_trusted(info, status):
            vram["free"] = info.get("free")
        return vram, vram.get("total")

    @app.get("/api/discover/search", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def discover_search(q: str = "", limit: int = 20, formats: str = "gguf",
                               types: str = "", source: str = "hf",
                               nsfw: bool = False):
        # `source` picks the provider; everything else keeps its existing meaning
        # for that provider. CivitAI results are returned close to CivitAI's own
        # shape (license flags, nsfw/nsfwLevel, modelVersions) rather than forced
        # through the HF result schema, which has no equivalent fields.
        if source == "civitai":
            from localm.model_manager.sources import civitai_search
            wanted_types = [t.strip() for t in types.split(",") if t.strip()]
            data = await _run_discover(
                lambda: civitai_search(q, limit=limit,
                                        types=wanted_types or None, nsfw=nsfw))
            return {"query": q, "source": "civitai", "results": data["items"],
                    "next_cursor": data.get("next_cursor")}
        # `formats` is a CSV of {gguf, hf} and `types` a CSV of MODEL_TYPES, both from
        # the search-page checkboxes. Empty tokens are dropped; hf_search raises
        # DiscoverError if none stay valid. An empty or absent `types` means an
        # untyped search (model_types=None). hf_backend_available lets the GUI warn
        # (not block) that a safetensors model needs the .[gpu] extra to run.
        from localm.discover import fit_label, hf_backend_available, hf_search
        wanted = [f.strip() for f in formats.split(",") if f.strip()]
        wanted_types = [t.strip().lower() for t in types.split(",") if t.strip()]
        model_types = wanted_types or None
        results = await _run_discover(
            lambda: hf_search(q, limit=limit, formats=wanted, model_types=model_types))
        # Attach a VRAM fit badge to results that carry a size estimate (HF results
        # with safetensors param metadata); GGUF results are sized per-file in the
        # /discover/files expander instead. fit_label yields "" when VRAM is unknown,
        # and a result with no size estimate keeps no fit.
        vram, total = await _vram_total()
        for r in results:
            if r.get("size_bytes"):
                r["fit"] = fit_label(r["size_bytes"], total)
        return {"query": q, "source": "hf", "results": results, "vram": vram,
                "hf_backend_available": hf_backend_available()}

    @app.get("/api/discover/files", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def discover_files(repo: str, source: str = "hf",
                              legacy_formats: bool = False):
        # For source=civitai, `repo` is a CivitAI model VERSION id, not a repo.
        if source == "civitai":
            from localm.model_manager.sources import civitai_list_files
            files = await _run_discover(
                lambda: civitai_list_files(repo, include_legacy_formats=legacy_formats))
            return {"repo": repo.strip(), "source": "civitai", "files": files}
        from localm.discover import fit_label, hf_gguf_files
        files = await _run_discover(lambda: hf_gguf_files(repo))
        vram, total = await _vram_total()
        models = []
        mmprojs = []
        for f in files:
            f["fit"] = fit_label(f["size_bytes"], total)
            if "mmproj" in f["file"].lower():
                mmprojs.append(f)
            else:
                models.append(f)
        return {"repo": repo.strip().strip("/"), "files": models, "mmprojs": mmprojs, "vram": vram}
