# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools over the model inventory: listing, HuggingFace discovery, pull,
embedding-model setup, and removal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from localm.pathsafe import is_unc_or_device_path

from ..server import EngineCache, _quiet_stdout, _text_result


# HuggingFace repos pull_model's MCP-client-supplied `repo` may target
# directly. An already-registered model's own source is accepted too - see
# _known_pull_repo().
KNOWN_PULL_REPOS = frozenset({
    "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "TheBloke/Mixtral-8x7B-v0.1-GGUF",
})


def _known_pull_repo(repo: str) -> bool:
    """True when *repo* is in KNOWN_PULL_REPOS, or matches an
    already-registered model's HuggingFace source."""
    if repo in KNOWN_PULL_REPOS:
        return True
    from localm.config import load_registry
    reg = load_registry()
    return any(info.get("source") == f"hf:{repo}" for info in reg.values())


def build(engines: EngineCache) -> Dict[str, dict]:
    """``list_models``, ``search_models``, ``list_model_files``, ``pull_model``,
    ``setup_embeddings`` and ``remove_model``. *engines* is this server's
    engine cache: ``pull_model`` loads into it and ``remove_model`` refuses to
    delete a file one of its residents holds."""
    def list_models(args: dict) -> dict:
        from localm.config import load_registry
        from localm.model_manager import _entry_path
        reg = load_registry()
        if not reg:
            return _text_result("No models registered.")
        lines = []
        for name, info in sorted(reg.items()):
            epath = _entry_path(info)
            if epath is None:
                # Match the CLI: a single malformed entry is shown corrupt, never
                # allowed to crash / blank the whole listing (removable via the
                # remove_model tool). Guards a hand-edited/half-written registry.
                lines.append(f"{name}  [corrupt]  (malformed registry entry)")
                continue
            # These stats run INLINE: this dispatcher has no event loop to
            # protect, since MCPStdioServer.run_stdio is a synchronous
            # `for line in stdin` loop and handle() is a plain def, so a thread
            # hop would only move the block. What keeps a pathological row (a
            # UNC path that blocks in the SMB redirector) out of this loop is
            # the REGISTRATION gate, not a probe here.
            p = Path(epath)
            if p.is_dir():
                size = "dir (HF format)"
            elif p.is_file():
                b = p.stat().st_size
                size = f"{b/1024**3:.2f} GB" if b >= 1024**3 else f"{b/1024**2:.0f} MB"
            else:
                size = "missing"
            lines.append(f"{name}  [{size}]  {info.get('source', 'local')}")
        return _text_result("\n".join(lines))

    def search_models(args: dict) -> dict:
        from localm.discover import DiscoverError, hf_search
        try:
            results = hf_search(args.get("query", ""), limit=args.get("limit", 20))
        except DiscoverError as e:
            return _text_result(str(e), is_error=True)
        return _text_result(json.dumps(results))

    def list_model_files(args: dict) -> dict:
        repo = args.get("repo", "")
        if not repo:
            return _text_result("'repo' is required (e.g. 'bartowski/Qwen2.5-7B-Instruct-GGUF')",
                                 is_error=True)
        from localm.discover import DiscoverError, fit_label, hf_gguf_files, vram_capacity
        try:
            files = hf_gguf_files(repo)
        except DiscoverError as e:
            return _text_result(str(e), is_error=True)
        total_vram = vram_capacity().get("total")
        for f in files:
            f["fit"] = fit_label(f["size_bytes"], total_vram)
        return _text_result(json.dumps(files))

    def pull_model(args: dict) -> dict:
        repo = args.get("repo", "")
        name = args.get("name", "")
        if not repo:
            return _text_result("'repo' is required", is_error=True)
        if not name:
            return _text_result(
                "'name' is required - pick a short registry name for this model",
                is_error=True)
        # `repo` is an MCP-CLIENT-supplied string, not a path the local user
        # picked, so is_unc_or_device_path's "remote value" contract applies:
        # refuse UNC/device syntax unconditionally, BEFORE the
        # Path(repo).exists() sink below ever runs. On Windows that sink dials
        # SMB and auto-authenticates for a UNC target, and can stall for minutes
        # inline in this handler.
        #
        # NOT gated on os.name, unlike reject_unsafe_path_string's `//`-form
        # check: no real HuggingFace repo id contains a backslash or starts with
        # `//`, on any platform.
        #
        # The message does not echo `repo` back, unlike the local-add message
        # below: this string never reached a safe-to-display check.
        if is_unc_or_device_path(repo):
            return _text_result(
                "'repo' looks like a filesystem path (UNC or device syntax), not "
                "a HuggingFace repo id. pull_model downloads a model by repo id "
                "(e.g. 'owner/name').", is_error=True)
        # A LOCAL PATH IS NOT A PULL. pull_model treats an existing path as a
        # local add, registering an arbitrary directory under a client-chosen
        # name; that name is then a registered model, so it passes the
        # membership check and resolves via the REGISTRY branch of
        # get_model_info rather than the direct-path gate. A refused add still
        # probes the path (config.json read, rglob, sha256). An MCP client pulls
        # from HuggingFace; registering something already on this disk is
        # `localm add`.
        try:
            if Path(repo).expanduser().exists():
                return _text_result(
                    f"{repo!r} is a path on this machine, not a HuggingFace repo. "
                    "pull_model downloads a model by repo id (e.g. "
                    "'owner/name'). To register a model already on this disk, run "
                    "'localm add <path>' on the host.", is_error=True)
        except OSError:
            pass          # unreadable/oversized path: not a local add, fall through
        if not _known_pull_repo(repo):
            return _text_result(
                f"Refusing to pull {repo!r} over MCP: it must be a known repo "
                f"{tuple(KNOWN_PULL_REPOS)} or match an already-registered "
                "model's source. Pull it with 'localm pull "
                f"{repo}' on the host first, then retry.", is_error=True)
        spec = f"{repo}:{args['file']}" if args.get("file") else repo

        from localm.model_manager.pull import pull_model as _pull

        # pull_model()'s progress bars/messages print via a rich Console (a
        # module-level singleton in model_manager/_shared.py, imported by value
        # into pull.py at load time). redirect_stdout catches it regardless of
        # which Console instance is in play.
        with _quiet_stdout():
            try:
                ok = _pull(spec, name=name)
            except Exception as e:
                return _text_result(f"pull failed: {e}", is_error=True)
        if not ok:
            return _text_result(f"pull failed for {spec!r} - see server stderr for detail",
                                 is_error=True)

        if not args.get("load", True):
            return _text_result(f"pulled and registered as {name!r} (not loaded)")

        try:
            # This load, like chat()/embed()'s, can print native sizing
            # diagnostics straight to stdout.
            #
            # engines.get() only constructs/registers the Engine and runs the
            # VRAM-eviction gate - it does NOT call Engine.load(), so the
            # backend stays unloaded until some later caller (normally
            # chat_stream()'s lazy-load path) touches it. The tool's own
            # description promises "load it - blocks until ready", so pull_model
            # calls .load() itself rather than leaving a resident-but-unloaded
            # engine parked in the cache.
            with _quiet_stdout():
                engine = engines.get(name)
                engine.load()
        except Exception as e:
            return _text_result(
                f"pulled and registered as {name!r}, but loading it failed: {e}",
                is_error=True)
        msg = f"pulled, registered, and loaded {name!r} - ready to use"
        # gpu_placement is None whenever the backend cannot report per-layer
        # placement for this engine - never fabricate a degraded warning without
        # evidence a load actually happened. When it IS known and partial/zero,
        # say so: a model too big to fully fit VRAM still loads, because the
        # backend's own sizing defers to a partial/zero GPU offload rather than
        # refusing.
        placement = getattr(engine, "gpu_placement", None)
        if placement and placement.get("degraded"):
            msg += (f" ({placement['gpu_layers_offloaded']}/"
                    f"{placement['gpu_layers_total']} layers on GPU, "
                    f"the rest on CPU - slower)")
        return _text_result(msg)

    def setup_embeddings(args: dict) -> dict:
        model = args.get("model")
        from localm.config import load_registry, update_config
        from localm.inference.embedder import (KNOWN_EMBEDDING_MODELS,
                                          resolve_embedding_model_path)
        # `model` is a free-form string chosen by the MCP CLIENT, and this
        # writes the admin_only `embedding_model` key. stdio gives no principal
        # to gate on, so the gate here is on the VALUE: a known key or a
        # registered model name only, never a raw path. Pointing the setting at
        # an arbitrary GGUF stays available to the owner through
        # `localm setup-embeddings` and the GUI. Refuse loudly rather than
        # silently ignoring the argument, so a caller is never told a selection
        # took effect when it did not.
        if model:
            if model not in KNOWN_EMBEDDING_MODELS and model not in load_registry():
                return _text_result(
                    f"Refusing to set the embedding model to {model!r}: over MCP it "
                    f"must be a known key {tuple(KNOWN_EMBEDDING_MODELS)} or an "
                    "already-registered model name, not a filesystem path. Use "
                    "'localm setup-embeddings <path>' or the GUI to point it at a "
                    "GGUF of your own.",
                    is_error=True)
            update_config(lambda c: c.update({"embedding_model": model}))

        with _quiet_stdout():
            try:
                path = resolve_embedding_model_path(allow_download=True)
            except Exception as e:
                return _text_result(f"Failed to setup embeddings: {e}", is_error=True)
        if not path:
            return _text_result(
                "Could not install the embedding model. It must be a known "
                f"key {tuple(KNOWN_EMBEDDING_MODELS)}, a registered model, or a GGUF "
                "path, and network must be enabled (net_mode is not 'off').",
                is_error=True
            )
        return _text_result(f"Embedding model ready: {path}. Memory and RAG will now use semantic search.")

    def _local_hold(model: str, reg: dict):
        """Whether an engine resident in THIS process is holding *model*'s file.

        The MCP server keeps its own residents (chat, embed and the coder tool
        all load through ``engines``), so this process can be the very thing
        holding the file open while it deletes it.
        """
        from localm.model_manager.registry import engine_holding_model_file
        candidates = [
            (name, getattr(engine, "model_path", None))
            for name, engine in list(getattr(engines, "_engines", {}).items())
            if getattr(engine, "loaded", False)
        ]
        return engine_holding_model_file(model, reg, candidates)

    def remove_model(args: dict) -> dict:
        model = args.get("model", "")
        if not model:
            return _text_result("'model' is required", is_error=True)
        from localm.model_manager import remove_model as _rm
        from localm.config import load_registry
        reg = load_registry()
        if model not in reg:
            return _text_result(f"Model not found: {model}", is_error=True)

        # Removing a registered model DELETES ITS FILE when that file lives in
        # the models dir, and nothing downstream of here asks whether anything
        # is still using it: model_manager.remove_model is the same code path
        # `localm rm` runs, with no server and no engine map in front of it.
        # Both holders are checked here - the engines resident in this process,
        # and any running server - and either one refuses.
        hold = _local_hold(model, reg)
        if hold is not None:
            if hold.reason is None:
                why = f"this MCP server still has its file loaded as {hold.key!r}"
            else:
                why = (f"this MCP server has {hold.key!r} loaded and "
                       f"{hold.reason}, so it cannot be ruled out as holding "
                       f"this file")
            return _text_result(
                f"Refusing to remove {model!r}: {why}. Removing it would "
                f"delete the model file while it is in use. Unload it first "
                f"(or restart this MCP server), then try again.",
                is_error=True)
        from localm.selfclient import remote_hold_reason
        remote = remote_hold_reason(model)
        if remote is not None:
            return _text_result(
                f"Refusing to remove {model!r}: {remote}. Removing it could "
                f"delete the model file while it is in use. Unload it there "
                f"(or stop that server), then try again.",
                is_error=True)

        with _quiet_stdout():
            try:
                _rm(model)
            except Exception as e:
                return _text_result(f"Failed to remove model: {e}", is_error=True)
        return _text_result(f"Model '{model}' successfully removed.")

    return {
        "list_models": {
            "description": "List locally registered models with size and source.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "title": "List models"},
            "handler": list_models,
        },
        "search_models": {
            "description": "Search HuggingFace for GGUF model repos (empty query = most downloaded).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text (optional)"},
                    "limit": {"type": "integer", "description": "Max results (default 20, max 50)"},
                },
            },
            "annotations": {"readOnlyHint": True, "title": "Search models"},
            "handler": search_models,
        },
        "list_model_files": {
            "description": (
                "List a HuggingFace repo's GGUF files (quant, size, and a fit "
                "badge - 'fits'/'tight'/'too-big' - against this machine's VRAM) "
                "so you can pick the right quant before pulling."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string",
                             "description": "HuggingFace repo id, e.g. 'bartowski/Qwen2.5-7B-Instruct-GGUF'"},
                },
                "required": ["repo"],
            },
            "annotations": {"readOnlyHint": True, "title": "List repo GGUF files"},
            "handler": list_model_files,
        },
        "pull_model": {
            "description": (
                "Download a GGUF file from HuggingFace, register it under 'name', "
                "and (by default) load it - blocks until ready. Use search_models "
                "+ list_model_files first to pick repo/file/quant."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "HuggingFace repo id"},
                    "file": {"type": "string",
                             "description": "Specific GGUF filename from list_model_files (omit for a full snapshot pull)"},
                    "name": {"type": "string", "description": "Registry name to give this model"},
                    "load": {"type": "boolean",
                             "description": "Load it into the engine after pulling (default true)"},
                },
                "required": ["repo", "name"],
            },
            "handler": pull_model,
        },
        "setup_embeddings": {
            "description": "Install the on-device embedding model for semantic search (memory + RAG).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Optional embedding model to set. Must be a known key (bge-small-en-v1.5, nomic-embed-text-v1.5) or an already-registered model name - a filesystem path is refused here; use 'localm setup-embeddings <path>' or the GUI for that."}
                }
            },
            "handler": setup_embeddings,
        },
        "remove_model": {
            "description": "Remove a model from the registry (and delete the file if it's in <data dir>/models/).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Registered model name to remove"}
                },
                "required": ["model"],
            },
            # Deletes the model file on disk - declare it so an MCP client can prompt
            # for confirmation before calling (confirmation belongs at the client).
            "annotations": {"destructiveHint": True, "title": "Remove model"},
            "handler": remove_model,
        },
    }
