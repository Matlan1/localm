# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model files named by the native media settings: how a setting resolves to a
file on this machine and how a file is named to users."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ModelFileError(Exception):
    """A native model setting does not name a usable file. The message is
    user-facing and names the file, never its directory."""


def display_name(value: str) -> str:
    """*value* as shown to users: a registered model name unchanged, a file path
    reduced to its file name (either separator)."""
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def resolve_model_file(value: str, what: str) -> Path:
    """The file *value* names: a registered model, else a local path. *what*
    names the setting in messages (``"image model"``, ``"video vae"``). UNC and
    device paths are refused without touching the filesystem. Raises
    ``ModelFileError``."""
    from localm.model_manager import load_registry
    from localm.model_manager.registry import _entry_path
    from localm.pathsafe import is_unc_or_device_path
    shown = display_name(value)
    registered = _entry_path(load_registry().get(str(value)))
    raw = registered or str(value)
    if is_unc_or_device_path(raw):
        raise ModelFileError(f"The native {what} '{shown}' is a network or device path, "
                             "which is not allowed.")
    path = Path(raw).expanduser()
    try:
        exists, is_file = path.exists(), path.is_file()
    except OSError as e:
        raise ModelFileError(f"The native {what} '{shown}' cannot be read "
                             f"({type(e).__name__}).") from e
    if not exists and registered:
        raise ModelFileError(f"The native {what} '{shown}' is registered, but its file "
                             f"'{display_name(raw)}' is missing.")
    if not exists:
        raise ModelFileError(f"The native {what} '{shown}' is neither a registered model "
                             "nor a file on this machine.")
    if not is_file:
        raise ModelFileError(f"The native {what} '{shown}' is not a single model file.")
    return path


def registered_file(filename: str) -> Optional[tuple[str, Path]]:
    """The first registered model whose file is named *filename* and exists, as
    ``(name, path)``, or None. An entry whose path cannot be checked is
    skipped."""
    from localm.model_manager import load_registry
    from localm.pathsafe import is_unc_or_device_path
    for name, entry in load_registry().items():
        raw = entry.get("path") if isinstance(entry, dict) else None
        if not raw or is_unc_or_device_path(str(raw)):
            continue
        path = Path(raw)
        if path.name != filename:
            continue
        try:
            if path.is_file():
                return name, path
        except OSError:
            continue
    return None
