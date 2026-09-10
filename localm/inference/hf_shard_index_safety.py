# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load-time containment gate for a pulled HuggingFace model's shard index.

A sharded checkpoint ships an index file (``model.safetensors.index.json`` or
``pytorch_model.bin.index.json``) whose ``weight_map`` maps each weight name to
the shard FILENAME holding it. ``transformers.utils.hub`` builds the shard paths
with ``os.path.join(model_dir, subfolder, filename)`` for a local directory and
returns them unvalidated, so a ``..`` component escapes the model directory and
an absolute or drive-qualified value replaces it outright. Both forms are then
opened.

``pull_model`` accepts an unrestricted repo id and fetches the whole repo
verbatim, index file included, so those values are attacker-controlled whenever
a user pulls a hostile or compromised model.

This module validates every ``weight_map`` value against the model directory
before ``HFBackend.load()`` hands it to ``transformers``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from localm.pathsafe import confined_under

INDEX_GLOB = "*.index.json"


def _index_files(base: Path) -> List[Path]:
    """Index files ``transformers`` can reach for *base*: the directory itself
    and one level down, matching its ``subfolder`` argument."""
    return sorted(set(base.glob(INDEX_GLOB)) | set(base.glob(f"*/{INDEX_GLOB}")))


def validate_shard_index(model_path: str) -> None:
    """Raise ``RuntimeError`` if any shard index under *model_path* maps a
    weight to a shard that resolves outside the model directory.

    A no-op when there is nothing to check: no index file, an index that is not
    valid JSON, or one carrying no ``weight_map`` object. An unparseable index
    is left alone because ``transformers`` reads it with ``json.loads`` too and
    raises its own, clearer error.

    A ``weight_map`` value that is not a string raises, because it cannot be
    validated and must not be passed through unchecked.
    """
    base = Path(model_path)
    try:
        resolved_base = base.resolve()
    except OSError:
        return

    for index_file in _index_files(base):
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(index, dict):
            continue
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            continue

        for weight, shard in weight_map.items():
            if not isinstance(shard, str):
                raise RuntimeError(
                    f"'{base.name}' ships {index_file.name} with a non-string "
                    f"shard filename for weight '{weight}'; refusing to load.")
            try:
                confined_under(resolved_base, shard)
            except ValueError as exc:
                raise RuntimeError(
                    f"'{base.name}' ships {index_file.name} with a shard "
                    f"filename that points outside the model directory "
                    f"({exc}); refusing to load. A shard filename names a file "
                    f"inside the model, so this model is malformed or "
                    f"hostile.") from exc
