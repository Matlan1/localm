# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inside the ``localm.setup_llama`` package, a name that is read through the
package (``_sl.NAME``) is never also read by its bare name, so replacing that
name on ``localm.setup_llama`` reaches every call site."""

from __future__ import annotations

import ast
from pathlib import Path

import localm.setup_llama as sl

PKG = Path(sl.__file__).parent


def _reads() -> tuple[dict, dict]:
    live: dict = {}
    bare: dict = {}
    for f in sorted(PKG.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "_sl"):
                live.setdefault(node.attr, set()).add(f.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                bare.setdefault(node.id, set()).add(f"{f.name}:{node.lineno}")
    return live, bare


def test_live_names_are_never_read_by_bare_name():
    live, bare = _reads()
    assert live, "no module reads anything through the package"
    mixed = {name: sorted(bare[name]) for name in live if name in bare}
    assert mixed == {}


def test_every_live_name_is_an_attribute_of_the_package():
    live, _ = _reads()
    assert sorted(n for n in live if not hasattr(sl, n)) == []


def test_the_patched_names_are_read_live():
    live, _ = _reads()
    for name in ("_download", "_release_assets", "_native_loads_ok", "_repo_runtime_lib",
                 "verified_urlopen", "pinned_tag", "installed_build", "_LIBGOMP_DEB_SHA256"):
        assert name in live, name
