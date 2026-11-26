# SPDX-License-Identifier: AGPL-3.0-or-later
"""Code of the build that the update mechanism uses, loaded by file path from the
folder that holds this package, and the copy of it the applier runs from."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

#: The modules beside this package that it loads. Each imports only the standard
#: library at module level.
MODULES = ("_apply_update", "update_package", "updater_components")
#: Modules of the build copied beside them when it has them, loaded with :func:`also`.
ALSO = ("procs", "_home_cfg")

_PACKAGE = Path(__file__).resolve().parent


def _load(name: str):
    key = f"_localm_update_mechanism_{name}"
    path = _PACKAGE.parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


apply_update = _load("_apply_update")
update_package = _load("update_package")
updater_components = _load("updater_components")


def also(name: str):
    """The module *name* of :data:`ALSO`, None when this copy does not have it."""
    return _load(name) if (_PACKAGE.parent / f"{name}.py").is_file() else None


def copy_applier(localm_dir, dest) -> Path:
    """Copy the update mechanism of the build whose ``localm`` folder is
    *localm_dir* into the new folder *dest*, with the modules it loads and, where the
    build has them, the ones the applier loads, as that build's own copy of this module
    names them (its :data:`MODULES` and :data:`ALSO`; these when it names none), and
    return the folder to run. Raises OSError, also when *dest* exists or a module it
    loads is missing."""
    localm_dir, dest = Path(localm_dir), Path(dest)
    needed, wanted = _lists_of(localm_dir / _PACKAGE.name / "shared.py")
    dest.mkdir(parents=True)
    run = dest / _PACKAGE.name
    shutil.copytree(localm_dir / _PACKAGE.name, run,
                    ignore=shutil.ignore_patterns("__pycache__"))
    found = set()
    for entry in sorted(os.listdir(localm_dir)):
        if entry.endswith(".py") and entry[:-3] in needed | wanted:
            shutil.copyfile(localm_dir / entry, dest / entry)
            found.add(entry[:-3])
    missing = sorted(needed - found)
    if missing:
        raise OSError(f"the build has no {', '.join(n + '.py' for n in missing)}")
    return run


def _lists_of(path: Path) -> tuple:
    """``(MODULES, ALSO)`` as sets, as the module at *path* assigns them as literal
    tuples of names; the lists of this copy for each it does not."""
    import ast
    out = {"MODULES": set(MODULES), "ALSO": set(ALSO)}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, SyntaxError):
        return out["MODULES"], out["ALSO"]
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in out:
            try:
                names = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(names, tuple) and all(
                    isinstance(n, str) and n.isidentifier() for n in names):
                out[node.targets[0].id] = set(names)
    return out["MODULES"], out["ALSO"]
