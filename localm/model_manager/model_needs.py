# SPDX-License-Identifier: AGPL-3.0-or-later
"""The reviewed list of Python packages each model type needs.

``model_needs.json`` maps a ``model_type`` to the packages, beyond what localm
ships, that a model of that type needs, each with the PyPI project that provides
the importable module, the version the list was built from, and the platforms
that have a Python 3.12 wheel for it (none when PyPI offers only source code to
build). Every entry is ``confirmed``: the project
was matched to the module by evidence (a requirements file, an install line in
the model's own code, transformers' own extras) and by the module's presence in
the project's files. A module without such evidence has no entry.

Nothing here installs a package. Adding a package name to this file widens what
localm may be asked to install, so the file changes only through review.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_FILE = Path(__file__).with_name("model_needs.json")

FORMAT_VERSION = 1
SCOPES = ("native", "custom_code")
PARTS = ("model", "tokenizer", "processor")
PLATFORMS = ("linux-x64", "win-x64", "macos-arm64")

_PROJECT = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r"^\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ROW_KEYS = frozenset({"scope", "part", "module", "project", "min_version", "platforms",
                       "status", "evidence"})
_ROW_OPTIONAL = frozenset({"repos", "of"})
_TOP_KEYS = frozenset({"version", "collected", "transformers", "types"})


class ModelNeedsError(ValueError):
    """The needs list is malformed or holds an entry that is not confirmed."""


@dataclass(frozen=True)
class Need:
    """One package a model type needs.

    ``scope`` is ``native`` (transformers' own code for the type) or
    ``custom_code`` (code bundled in model repositories of that type); ``part`` is
    the piece that needs it. ``repos`` and ``of`` are, for custom code, how many of
    the surveyed repositories of the type required the package."""
    scope: str
    part: str
    module: str
    project: str
    min_version: str
    platforms: tuple[str, ...]
    evidence: tuple[str, ...]
    repos: Optional[int] = None
    of: Optional[int] = None


def _fail(where: str, what: str) -> None:
    raise ModelNeedsError(f"{where}: {what}")


def _text(where: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(where, "must be a non-empty string")
    return value


def _count(where: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(where, "must be a positive integer")
    return value


def _need(where: str, row: object) -> Need:
    if not isinstance(row, dict):
        _fail(where, "must be an object")
    keys = set(row)
    if not _ROW_KEYS <= keys or keys - _ROW_KEYS - _ROW_OPTIONAL:
        _fail(where, f"keys must be {sorted(_ROW_KEYS)} plus optionally {sorted(_ROW_OPTIONAL)}")
    if row["status"] != "confirmed":
        _fail(where, f"status is {row['status']!r}; only confirmed entries may be listed")
    if row["scope"] not in SCOPES:
        _fail(where, f"scope must be one of {SCOPES}")
    if row["part"] not in PARTS:
        _fail(where, f"part must be one of {PARTS}")
    module = _text(f"{where}.module", row["module"])
    if not module.isidentifier():
        _fail(where, "module must be an importable name")
    project = _text(f"{where}.project", row["project"])
    if not _PROJECT.match(project):
        _fail(where, "project must be a normalised PyPI name (lower case letters, digits, hyphens)")
    version = _text(f"{where}.min_version", row["min_version"])
    if not _VERSION.match(version):
        _fail(where, "min_version must be a plain release version")
    platforms = row["platforms"]
    if (not isinstance(platforms, list)
            or len(set(platforms)) != len(platforms) or not set(platforms) <= set(PLATFORMS)):
        _fail(where, f"platforms must be a list without repeats from {PLATFORMS}")
    evidence = row["evidence"]
    if not isinstance(evidence, list) or not evidence:
        _fail(where, "a confirmed entry needs evidence")
    for i, item in enumerate(evidence):
        _text(f"{where}.evidence[{i}]", item)
    repos = of = None
    if ("repos" in row) != ("of" in row):
        _fail(where, "repos and of go together")
    if "repos" in row:
        repos, of = _count(f"{where}.repos", row["repos"]), _count(f"{where}.of", row["of"])
        if repos > of:
            _fail(where, "repos cannot exceed of")
    return Need(row["scope"], row["part"], module, project, version, tuple(platforms),
                tuple(evidence), repos, of)


def parse(doc: object) -> dict[str, tuple[Need, ...]]:
    """The needs of each model type in *doc*, a decoded ``model_needs.json``.

    Raises ModelNeedsError for any deviation from the format: an unknown or missing
    key, a malformed project, version or platform, an entry whose status is not
    ``confirmed`` or that has no evidence, or a repeated (scope, part, module)."""
    if not isinstance(doc, dict) or set(doc) != _TOP_KEYS:
        _fail("file", f"top-level keys must be {sorted(_TOP_KEYS)}")
    if doc["version"] != FORMAT_VERSION or isinstance(doc["version"], bool):
        _fail("file", f"version must be {FORMAT_VERSION}")
    if not isinstance(doc["collected"], str) or not _DATE.match(doc["collected"]):
        _fail("file", "collected must be a YYYY-MM-DD date")
    _text("file.transformers", doc["transformers"])
    types = doc["types"]
    if not isinstance(types, dict):
        _fail("file.types", "must be an object")
    out: dict[str, tuple[Need, ...]] = {}
    for model_type, rows in types.items():
        where = f"types[{model_type!r}]"
        _text(where, model_type)
        if not isinstance(rows, list) or not rows:
            _fail(where, "must be a non-empty list")
        needs = tuple(_need(f"{where}[{i}]", row) for i, row in enumerate(rows))
        seen = [(n.scope, n.part, n.module) for n in needs]
        if len(set(seen)) != len(seen):
            _fail(where, "a package is listed twice for the same scope and part")
        out[model_type] = needs
    return out


def load(path: Optional[Path] = None) -> dict[str, tuple[Need, ...]]:
    """The needs list read from *path* (the shipped ``model_needs.json`` when omitted),
    validated by :func:`parse`. Raises ModelNeedsError when the file is unreadable,
    is not JSON or is malformed."""
    target = DATA_FILE if path is None else Path(path)
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ModelNeedsError(f"{target.name}: cannot be read as JSON ({type(e).__name__})") from e
    return parse(doc)


@functools.lru_cache(maxsize=1)
def _shipped() -> dict[str, tuple[Need, ...]]:
    return load()


def needs_for(model_type: str, scope: Optional[str] = None) -> tuple[Need, ...]:
    """The packages the shipped list names for *model_type*, restricted to *scope*
    when given; empty for a type the list does not name."""
    rows = _shipped().get(model_type, ())
    return tuple(n for n in rows if scope is None or n.scope == scope)
