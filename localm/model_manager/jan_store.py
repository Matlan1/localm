# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read the model store of the Jan desktop app (``<data folder>/llamacpp/models``).

Each model is a directory holding ``model.yml``; its weights are ``model.gguf``
beside it, or the file ``model_path`` names (relative to the Jan data folder, or
absolute for a model Jan links in place). A vision projector is ``mmproj.gguf``
or the file ``mmproj_path`` names. A model id may contain ``/``
(``<author>/<file stem>``), so models sit at any depth below the store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

from .gguf import _has_gguf_magic, gguf_is_mmproj, gguf_unusable_reason

_MAX_DEPTH = 6
_YML_SCALAR_RE = re.compile(r"^(model_path|mmproj_path):[ \t]*(.*?)[ \t]*$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class JanModel:
    model_id: str
    weights: Path
    projector: Optional[Path]


def find_jan_models_root(p: Path) -> Optional[Path]:
    """The ``llamacpp/models`` folder *p* denotes: *p* itself, ``p/models`` for
    the ``llamacpp`` folder, or ``p/llamacpp/models`` for the Jan data folder;
    else None."""
    for candidate, ok in (
        (p, p.name == "models" and p.parent.name == "llamacpp"),
        (p / "models", p.name == "llamacpp"),
        (p / "llamacpp" / "models", True),
    ):
        try:
            if ok and candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


def _yml_paths(yml: Path) -> dict[str, str]:
    """``model_path`` and ``mmproj_path`` from the top level of *yml*, with
    surrounding quotes removed. Keys that are absent or unreadable are left out."""
    try:
        text = yml.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    found: dict[str, str] = {}
    for line in text.splitlines():
        m = _YML_SCALAR_RE.match(line)
        if not m:
            continue
        value = m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            found.setdefault(m.group(1), value)
    return found


def _resolve(value: str, data_folder: Path) -> Optional[Path]:
    """*value* as a path: absolute as written, else relative to *data_folder*.
    None when it climbs out with a ``..`` part."""
    if any(part == ".." for part in PureWindowsPath(value).parts) or \
            any(part == ".." for part in PurePosixPath(value).parts):
        return None
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute() \
            or _WINDOWS_DRIVE_RE.match(value):
        return Path(value)
    return data_folder / value


def _weights(unit: Path, paths: dict[str, str], data_folder: Path) -> tuple[Optional[Path], str]:
    """``(weights, "")`` for a model directory, or ``(None, reason)``."""
    named = paths.get("model_path")
    candidate = _resolve(named, data_folder) if named else None
    if candidate is None or not candidate.is_file():
        candidate = unit / "model.gguf"
    if not candidate.is_file():
        if (unit / "model.gguf.tmp").exists():
            return None, "its download has not finished"
        return None, f"the weights file is missing ({candidate.name})"
    if candidate.suffix.lower() != ".gguf":
        return None, f"{candidate.name} is not a GGUF file"
    if not _has_gguf_magic(candidate):
        return None, f"{candidate.name} is not a complete GGUF file"
    unusable = gguf_unusable_reason(candidate)
    if unusable is not None:
        return None, unusable
    return candidate, ""


def _projector(unit: Path, paths: dict[str, str], data_folder: Path) -> tuple[Optional[Path], str]:
    """``(projector, "")``, ``(None, "")`` when the model has none, or
    ``(None, reason)`` when it names one that cannot be used."""
    named = paths.get("mmproj_path")
    candidate = _resolve(named, data_folder) if named else None
    if candidate is None or not candidate.is_file():
        candidate = unit / "mmproj.gguf"
        if not candidate.is_file():
            return None, ("the projector file is missing" if named else "")
    if not (_has_gguf_magic(candidate) and gguf_is_mmproj(candidate)):
        return None, f"{candidate.name} is not a GGUF vision projector"
    return candidate, ""


def scan_jan_models(root: Path) -> tuple[list[JanModel], list[str]]:
    """``(models, problems)`` for the Jan model store *root*.

    A directory holding ``model.yml`` is one model and is not searched further.
    The id is its path below *root* with ``/`` as ``-``. *problems* lists one
    sentence per model that cannot be registered or whose projector is left off."""
    data_folder = root.parent.parent
    models: list[JanModel] = []
    problems: list[str] = []

    def walk(directory: Path, depth: int) -> None:
        try:
            children = sorted(c for c in directory.iterdir() if c.is_dir())
        except OSError:
            return
        for child in children:
            yml = child / "model.yml"
            if yml.is_file():
                model_id = "-".join(child.relative_to(root).parts)
                paths = _yml_paths(yml)
                weights, why = _weights(child, paths, data_folder)
                if weights is None:
                    problems.append(f"{model_id}: {why}")
                    continue
                projector, why = _projector(child, paths, data_folder)
                if why:
                    problems.append(f"{model_id}: no vision projector attached, {why}")
                models.append(JanModel(model_id, weights, projector))
            elif depth < _MAX_DEPTH:
                walk(child, depth + 1)

    walk(root, 1)
    return models, problems
