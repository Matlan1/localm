# SPDX-License-Identifier: AGPL-3.0-or-later
"""This build installs an update only with the program the build ships
(``localm.update_mechanism``): the code of the old in-process swap is gone from
``localm/``. What an older localm (0.3.2 to 0.3.4b4) runs of this build after its own
updater swapped it in stays and works: its health watchdog runs this build's
``scripts/rollback_update.py``, which loads ``localm/_apply_update.py`` by file path and
puts the backup that updater made back."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._old_client_update import backup_and_swap

REPO = Path(__file__).resolve().parents[1]
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

#: Module -> the top-level names of the old swap that must not come back.
GONE = {
    "localm/updater.py": {
        "apply", "classify", "read_manifest", "class_summary", "_max_class", "_deps_set",
        "_requires_python", "_ORDER", "_new_run_dir", "_rename_folder", "_drop_run_dir",
        "_promote_backup", "_apply_components", "_check_package", "spawn_health_watchdog",
        "_watchdog_script", "_watchdog_takes_relaunch", "GIT_CLONE_NOTE",
    },
    "localm/_apply_update.py": {
        "swap_with_backup", "backup", "apply_files", "swap_entries", "check_space",
        "space_needed", "NoSpace", "BackupFailed",
    },
    "localm/updater_finish.py": {"run", "FinishFailed", "present"},
    "localm/cli/maintenance.py": {"_defer_update_steps", "_print_update_extras"},
}
#: Attribute names no module of localm may use: the calls of the old swap.
GONE_ATTRIBUTES = {"swap_with_backup", "apply_files", "_promote_backup",
                   "spawn_health_watchdog", "_restart_after_apply"}


def _top_level_names(tree: ast.Module) -> set:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_the_old_swap_is_gone_from_localm():
    found = []
    for rel, names in GONE.items():
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        found += [f"{rel}: {n}" for n in sorted(_top_level_names(tree) & names)]
    handoff = ast.parse((REPO / "localm" / "updater_handoff.py").read_text(encoding="utf-8"))
    prepared = next(n for n in handoff.body
                    if isinstance(n, ast.ClassDef) and n.name == "Prepared")
    if "legacy" in _top_level_names(ast.Module(body=prepared.body, type_ignores=[])):
        found.append("localm/updater_handoff.py: Prepared.legacy")
    for path in sorted((REPO / "localm").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and (
                    node.attr in GONE_ATTRIBUTES
                    or (node.attr == "apply" and isinstance(node.value, ast.Name)
                        and node.value.id in ("updater", "_updater"))):
                found.append(f"{path.relative_to(REPO).as_posix()}:{node.lineno} "
                             f".{node.attr}")
            elif isinstance(node, ast.keyword) and node.arg == "update_watchdog":
                found.append(f"{path.relative_to(REPO).as_posix()}:{node.value.lineno} "
                             "update_watchdog=")
    assert found == []


def test_the_scan_sees_a_name_that_comes_back():
    tree = ast.parse("def swap_with_backup():\n    pass\nNoSpace = 1\n")
    assert _top_level_names(tree) & GONE["localm/_apply_update.py"] == {
        "swap_with_backup", "NoSpace"}


# ------------------- what an older localm runs of this build --------------------

_LIB = "runtime/localm_llama_runtime/lib"


def _write(root: Path, files: dict) -> None:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")


@pytest.fixture
def updated(tmp_path):
    """An install at 0.3.2 with its native runtime that an older localm's updater
    replaced with this build's code: the backup in the data folder, this build's
    rollback script and the helper it loads by file path in the install."""
    inst, staged, home = tmp_path / "inst", tmp_path / "staged", tmp_path / "home"
    _write(inst, {"VERSION": "0.3.2", "localm/__init__.py": "# 0.3.2",
                  "localm/gone.py": "only in 0.3.2",
                  "runtime/localm_llama_runtime/__init__.py": "SCAFFOLD 0.3.2",
                  f"{_LIB}/llama.dll": "PROVISIONED"})
    _write(staged, {"VERSION": "0.3.4", "localm/__init__.py": "# 0.3.4",
                    "localm/added.py": "new in 0.3.4",
                    "runtime/localm_llama_runtime/__init__.py": "SCAFFOLD 0.3.4",
                    "runtime/localm_llama_runtime/new_helper.py": "new in 0.3.4"})
    for rel in ("scripts/rollback_update.py", "localm/_apply_update.py"):
        (staged / rel).parent.mkdir(parents=True, exist_ok=True)
        (staged / rel).write_bytes((REPO / rel).read_bytes())
    backup = home / "updates" / "backup"
    backup.parent.mkdir(parents=True)
    names = backup_and_swap(staged, inst, backup)
    (home / "updates" / "applied_names.json").write_text(json.dumps(names), encoding="utf-8")
    assert (inst / "VERSION").read_text(encoding="utf-8") == "0.3.4"
    return inst, home


def _files(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(root.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.relative_to(root).parts[0] != "scripts"}


def test_the_rollback_an_older_watchdog_runs_puts_its_backup_back(updated):
    """0.3.2's watchdog loads this build's ``scripts/rollback_update.py`` and calls
    ``main(["--yes"], install_root=...)``; the script loads this build's
    ``localm/_apply_update.py`` by file path (``rollback``, ``backup_names``,
    ``backup_problem``, and ``_prune`` and ``_copy_into`` for the runtime folder). Run
    here as its own process with ``-I -S``, so nothing of localm is importable."""
    inst, home = updated
    probe = ("import importlib.util, sys\n"
             "spec = importlib.util.spec_from_file_location('rb', sys.argv[1])\n"
             "rb = importlib.util.module_from_spec(spec)\n"
             "spec.loader.exec_module(rb)\n"
             "from pathlib import Path\n"
             "raise SystemExit(rb.main(['--yes'], install_root=Path(sys.argv[2])))\n")
    env = {"LOCALM_HOME": str(home), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    out = subprocess.run([sys.executable, "-I", "-S", "-c", probe,
                          str(inst / "scripts" / "rollback_update.py"), str(inst)],
                         cwd=str(inst.parent), env=env, capture_output=True, text=True,
                         timeout=120, **_NO_WINDOW)
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert "Traceback" not in out.stderr, out.stderr
    assert _files(inst) == {
        "VERSION": "0.3.2", "localm/__init__.py": "# 0.3.2",
        "localm/gone.py": "only in 0.3.2",
        "runtime/localm_llama_runtime/__init__.py": "SCAFFOLD 0.3.2",
        f"{_LIB}/llama.dll": "PROVISIONED"}
