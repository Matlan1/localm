# SPDX-License-Identifier: AGPL-3.0-or-later
"""Code of the build that the update mechanism uses, loaded by file path from the
folder that holds this package, and the copy of it the applier runs from."""

from __future__ import annotations

import importlib.util
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
    *localm_dir* into the new folder *dest*, with the modules it loads
    (:data:`MODULES`) and, where the build has them, the ones the applier loads
    (:data:`ALSO`), and return the folder to run. Raises OSError, also when *dest*
    exists."""
    localm_dir, dest = Path(localm_dir), Path(dest)
    dest.mkdir(parents=True)
    run = dest / _PACKAGE.name
    shutil.copytree(localm_dir / _PACKAGE.name, run,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in MODULES:
        shutil.copyfile(localm_dir / f"{name}.py", dest / f"{name}.py")
    for name in ALSO:
        if (localm_dir / f"{name}.py").is_file():
            shutil.copyfile(localm_dir / f"{name}.py", dest / f"{name}.py")
    return run
