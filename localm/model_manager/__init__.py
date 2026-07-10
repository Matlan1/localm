# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model management: registry/selection, the gguf helpers, and the download
backends, split into a package for navigability.

This module is a behavior-preserving split of the former single-file
``model_manager.py``. The public surface is unchanged: every name that was
importable as ``from localm.model_manager import X`` (or read as
``localm.model_manager.X``) is re-exported here, so importers and existing
monkeypatches keep working. The concerns live in:

  * ``gguf``     - gguf file-format helpers + SHA256 hashing (leaf utilities)
  * ``registry`` - registry/selection, info + vision, aliases, dedup, sync, add/remove
  * ``pull``     - download/transport routing (HF / URL / Ollama) + progress
  * ``_shared``  - the Rich console, the progress sentinel, the win32 stdout side effect

``sys`` and ``shutil`` are imported here so existing tests that patch
``localm.model_manager.sys.stdin`` / ``localm.model_manager.shutil.disk_usage``
keep resolving (both patch the global module singleton).
"""

import shutil  # noqa: F401  (re-exported so localm.model_manager.shutil resolves for tests)
import sys  # noqa: F401  (re-exported so localm.model_manager.sys resolves for tests)

# Config-derived runtime values are re-exported as package attributes so that
# tests which patch e.g. localm.model_manager.MODELS_DIR / load_registry still
# work; remove_model reads these through the package for the same reason.
from ..config import (
    HOME_DIR,
    MODELS_DIR,
    REGISTRY_FILE,
    ensure_dirs,
    load_config,
    load_registry,
    save_registry,
    update_registry,
)
from ._shared import PROGRESS_SENTINEL, console
from .gguf import (
    _GGUF_MIN_BYTES,
    _HASH_PROGRESS_MIN_BYTES,
    _gguf_first_parts,
    _has_gguf_magic,
    _hash_with_progress,
    _safe_models_filename,
    _sha256_file,
    _sha256_file_bytes,
    _SPLIT_GGUF_RE,
    first_split_part,
    missing_split_parts,
    split_gguf_parts,
)
from .registry import (
    _add_local_gguf_dir,
    _backup_registry,
    _hf_is_vision,
    _prompt_duplicate_action,
    _prompt_predownload_dup,
    _register,
    _register_with_dedup,
    _resolve_ollama_manifest,
    _sanitize_name,
    _SHORTCUT_SIZES,
    _store_into_models_dir,
    _store_loose_gguf_dir,
    _unique_registry_name,
    add_local,
    alias_model,
    find_aliases_by_path,
    find_by_sha256,
    find_by_size,
    find_sibling_mmproj,
    get_model_info,
    get_model_mmproj,
    get_model_path,
    is_auto_chat_eligible,
    is_llm,
    is_external_path,
    list_models,
    model_is_external,
    MODEL_SHORTCUTS,
    ModelSyncResult,
    relocate_model,
    remove_model,
    resolve_spec,
    set_model_type,
    show_shortcuts,
    sync_models_dir,
    vision_capable_models,
    vision_input_guidance,
    MODEL_TYPES,
)
from .pull import (
    _check_disk_space,
    _download_progress,
    _emit_progress,
    _hf_file_sha256,
    _progress_file_info,
    _pull_gguf_file,
    _pull_hf_snapshot,
    _pull_url,
    _snapshot_progress,
    _stem_from_url,
    pull_model,
)

__all__ = [
    # config re-exports
    "HOME_DIR", "MODELS_DIR", "REGISTRY_FILE", "ensure_dirs", "load_config",
    "load_registry", "save_registry", "update_registry",
    # shared
    "console", "PROGRESS_SENTINEL",
    # gguf
    "_SPLIT_GGUF_RE", "split_gguf_parts", "first_split_part", "missing_split_parts",
    "_sha256_file", "_sha256_file_bytes", "_safe_models_filename",
    "_HASH_PROGRESS_MIN_BYTES", "_hash_with_progress", "_has_gguf_magic",
    "_gguf_first_parts", "_GGUF_MIN_BYTES",
    # registry
    "MODEL_SHORTCUTS", "_SHORTCUT_SIZES", "resolve_spec", "get_model_path",
    "get_model_info", "find_sibling_mmproj", "get_model_mmproj", "_hf_is_vision",
    "vision_capable_models", "vision_input_guidance", "list_models",
    "relocate_model", "set_model_type", "is_auto_chat_eligible", "is_llm",
    "is_external_path", "model_is_external",
    "find_aliases_by_path", "find_by_sha256", "find_by_size", "alias_model",
    "_register", "_sanitize_name", "_unique_registry_name", "ModelSyncResult",
    "_backup_registry", "sync_models_dir", "_register_with_dedup", "remove_model",
    "_add_local_gguf_dir", "add_local", "show_shortcuts", "_prompt_duplicate_action",
    "_prompt_predownload_dup", "_resolve_ollama_manifest", "MODEL_TYPES",
    "_store_into_models_dir", "_store_loose_gguf_dir",
    # pull
    "_emit_progress", "_progress_file_info", "_download_progress",
    "_snapshot_progress", "pull_model", "_stem_from_url", "_check_disk_space",
    "_hf_file_sha256", "_pull_gguf_file", "_pull_hf_snapshot", "_pull_url",
]
