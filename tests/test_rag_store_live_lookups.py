# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inside ``localm/rag/_store/``, a name read through ``localm.rag.store``
(``_st.NAME``) is never also read by its bare name inside a function, so a value
replaced on ``localm.rag.store`` reaches every call site; and every name the
facade re-exports is the object its module defines."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import localm.rag.store as store

PKG = Path(store.__file__).parent / "_store"

REPLACED_ON_THE_FACADE = (
    "_numpy", "_NUMPY_IS_STUB", "_NUMPY_DEGRADE_LOGGED", "_COLLECTION_CACHE_MAX_BYTES",
    "_collection_cache_fingerprint", "_storekit_atomic_write", "split_jsonl",
    "confine_index_path", "extract_bytes", "rag_dir", "collection_names", "Collection",
)


def _modules():
    return sorted(p for p in PKG.glob("*.py"))


def _reads() -> tuple[dict, dict]:
    live: dict = {}
    bare: dict = {}
    for f in _modules():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for stmt in fn.body:
                for node in ast.walk(stmt):
                    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                            and node.value.id == "_st"):
                        live.setdefault(node.attr, set()).add(f.name)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        bare.setdefault(node.id, set()).add(f"{f.name}:{node.lineno}")
    return live, bare


def test_live_names_are_never_read_by_bare_name():
    live, bare = _reads()
    assert live, "no module reads anything through localm.rag.store"
    assert {n: sorted(bare[n]) for n in live if n in bare} == {}


def test_every_name_replaced_on_the_facade_is_read_live():
    live, bare = _reads()
    assert [n for n in REPLACED_ON_THE_FACADE if n not in live] == []
    assert {n: sorted(bare[n]) for n in REPLACED_ON_THE_FACADE if n in bare} == {}


def test_every_live_name_is_an_attribute_of_the_facade():
    live, _ = _reads()
    assert sorted(n for n in live if not hasattr(store, n)) == []


def test_st_is_the_facade_module():
    for f in _modules():
        mod = importlib.import_module(f"localm.rag._store.{f.stem}"
                                      if f.stem != "__init__" else "localm.rag._store")
        if hasattr(mod, "_st"):
            assert mod._st is store, f.name


def test_reexported_names_are_the_defining_modules_objects():
    mismatched = []
    for f in _modules():
        if f.stem == "__init__":
            continue
        mod = importlib.import_module(f"localm.rag._store.{f.stem}")
        tree = ast.parse(f.read_text(encoding="utf-8"))
        defined = set()
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, ast.Assign):
                defined.update(t.id for t in n.targets if isinstance(t, ast.Name))
            elif isinstance(n, ast.AnnAssign):
                defined.add(n.target.id)
            elif isinstance(n, ast.Try):
                defined.add("_numpy")
        for name in sorted(defined):
            if hasattr(store, name) and getattr(store, name) is not getattr(mod, name):
                mismatched.append(f"{f.stem}.{name}")
    assert mismatched == []
