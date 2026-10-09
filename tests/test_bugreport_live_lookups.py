# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inside the ``localm.bugreport`` package, a name that is read through the
package (``_br.NAME``) is never also read or assigned by its bare name, and the
module state (the installed-handlers flag, the crash-trace handle and its
owner, the armed instance id) is defined once, on the package. Replacing any of
these on ``localm.bugreport`` therefore reaches every call site."""

from __future__ import annotations

import ast
from pathlib import Path

import localm.bugreport as br

PKG = Path(br.__file__).parent
STATE = ("_handlers_installed", "_crash_trace_fh", "_crash_trace_instance_id",
         "_armed_instance_id")


def _trees():
    for f in sorted(PKG.glob("*.py")):
        yield f.name, ast.parse(f.read_text(encoding="utf-8"))


def _uses() -> tuple[dict, dict]:
    """(names read through ``_br``, every other use of a bare name). A
    module-level assignment target is the name's definition, not a use."""
    live: dict = {}
    bare: dict = {}
    for name, tree in _trees():
        definitions = {id(t) for n in tree.body if isinstance(n, ast.Assign)
                       for t in n.targets}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "_br"):
                live.setdefault(node.attr, set()).add(name)
            elif isinstance(node, ast.Name) and id(node) not in definitions:
                bare.setdefault(node.id, set()).add(f"{name}:{node.lineno}")
    return live, bare


def test_live_names_are_never_used_by_bare_name():
    live, bare = _uses()
    assert live, "no module reads anything through the package"
    mixed = {name: sorted(bare[name]) for name in live if name in bare}
    assert mixed == {}


def test_every_live_name_is_an_attribute_of_the_package():
    live, _ = _uses()
    assert sorted(n for n in live if not hasattr(br, n)) == []


def test_the_patched_names_and_the_state_are_read_live():
    live, _ = _uses()
    for name in ("_scrub_secrets", "report_failure", "save_report", "upload_report",
                 "upload_config", "clear_crash_marker", "_crash_dir",
                 "_diagnostics_allowed", "_LOG_TAIL_READ_BYTES") + STATE:
        assert name in live, name


def test_the_state_is_defined_only_on_the_package_and_never_via_global():
    for name, tree in _trees():
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)], name
        defined = {t.id for n in tree.body if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)}
        if name == "__init__.py":
            assert set(STATE) <= defined
        else:
            assert not set(STATE) & defined, name
