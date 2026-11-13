# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every test file that lists a folder of the checkout is known to the
targeted-test selector.

The selector reaches a test through its imports and the file names it spells.
A test that walks a folder (every localm/*.py, every docs page, every workflow)
also depends on files it never names, so a file added there must select it
through scripts/affected_tests.py `_TREE_SCANNERS`. A test file that walks the
checkout, or loads a script or helper that does, is either listed there or
listed in `_NOT_TREE_SCANNERS` with how its result does not depend on the
folder. The walks are found by tests/_tree_walks.py.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

from tests._tree_walks import script_names, tree_walks

REPO_ROOT = Path(__file__).resolve().parents[1]
_SELECTOR = REPO_ROOT / "scripts" / "affected_tests.py"
_DELEGATION = "hands a checkout folder to "


def _selector():
    spec = importlib.util.spec_from_file_location("affected_tests_registry", _SELECTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def selector():
    return _selector()


@pytest.fixture(scope="module")
def found(selector) -> dict[str, str]:
    """{test file: how it walks the checkout} over the real tree."""
    graph = selector.Graph()

    def text(rel: str) -> str:
        return (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")

    walking: dict[str, str] = {}
    for mod, rel in graph.sources.items():
        if rel.startswith(("scripts/", "tests/")):
            direct = [w for w in tree_walks(text(rel)) if not w[1].startswith(_DELEGATION)]
            if direct:
                walking[mod] = f"{rel}:{direct[0][0]} ({direct[0][1]})"
    by_file = {Path(graph.sources[m]).name: m for m in walking if graph.sources[m].startswith("scripts/")}
    out: dict[str, str] = {}
    for rel in graph.test_files:
        source = text(rel)
        own = tree_walks(source)
        if own:
            out[rel] = f"line {own[0][0]}: {own[0][1]}"
            continue
        loaded = {m for m in graph.test_imports[rel] if m in walking}
        loaded |= {by_file[name] for name in script_names(source) if name in by_file}
        if loaded:
            mod = sorted(loaded)[0]
            out[rel] = f"loads {mod}, which walks at {walking[mod]}"
    return out


def test_every_test_file_that_walks_the_checkout_is_known_to_the_selector(selector, found):
    unknown = {t: why for t, why in found.items()
               if t not in selector._TREE_SCANNERS and t not in selector._NOT_TREE_SCANNERS}
    assert unknown == {}, (
        "these test files walk a folder of the checkout but are in neither "
        "_TREE_SCANNERS (with the patterns of what they read) nor _NOT_TREE_SCANNERS "
        f"(with how their result does not depend on it) in scripts/affected_tests.py: {unknown}")


def test_every_exempt_file_still_walks_the_checkout(selector, found):
    assert sorted(set(selector._NOT_TREE_SCANNERS) - set(found)) == []


def test_every_registered_scanner_is_found_by_the_detector(selector, found):
    assert sorted(set(selector._TREE_SCANNERS) - set(found)) == []


def test_no_test_file_is_both_a_scanner_and_exempt(selector):
    assert sorted(set(selector._TREE_SCANNERS) & set(selector._NOT_TREE_SCANNERS)) == []


def test_every_exempt_file_is_a_test_file(selector):
    graph = selector.Graph()
    assert sorted(set(selector._NOT_TREE_SCANNERS) - set(graph.test_files)) == []


def test_the_scans_that_went_unselected_are_found(found):
    for rel in ("tests/test_outbound_call_sites_contract.py", "tests/test_one_server_switch_path.py"):
        assert found[rel].endswith(": rglob"), found.get(rel)


def _walks(source: str) -> list[str]:
    return [kind for _line, kind in tree_walks(textwrap.dedent(source))]


@pytest.mark.parametrize("source", [
    """
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    def test_x():
        for p in (ROOT / "localm").rglob("*.py"):
            pass
    """,
    """
    import pathlib, localm
    PKG = pathlib.Path(localm.__file__).parent
    def test_x():
        assert sorted(PKG.glob("*.py"))
    """,
    """
    import os
    from pathlib import Path
    def _scan(root):
        return [n for _d, _s, n in os.walk(root)]
    def test_x():
        _scan(Path(__file__).parent.parent / "docs")
    """,
    """
    import subprocess
    from pathlib import Path
    REPO = Path(__file__).parents[1]
    def test_x():
        subprocess.run(["git", "ls-files", "localm"], cwd=REPO)
    """,
    """
    from pathlib import Path
    class Scanner:
        def __init__(self, root):
            self.root = root
        def files(self):
            return list(self.root.iterdir())
    def test_x():
        Scanner(Path(__file__).parents[1]).files()
    """,
    """
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[1]
    def test_x(mod):
        mod.Analyzer(REPO).findings()
    """,
], ids=["module-constant", "package-file", "helper-parameter", "git-ls-files", "self-attribute",
        "handed-to-a-script"])
def test_a_walk_of_the_checkout_is_found(source):
    assert _walks(source)


@pytest.mark.parametrize("source", [
    """
    def test_x(tmp_path):
        assert list(tmp_path.rglob("*")) == []
    """,
    """
    from pathlib import Path
    import requests
    def test_x():
        assert list(Path(requests.__file__).parent.rglob("*.py"))
    """,
    """
    import json
    from pathlib import Path
    ROOT = Path(__file__).parents[1]
    def test_x():
        data = json.loads((ROOT / "a.json").read_text())
        assert data.glob("x")
    """,
    """
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[1]
    def test_x(mod):
        assert mod.read_pins(REPO / "localm" / "pins.py")
    """,
    """
    from pathlib import Path
    ROOT = Path(__file__).parents[1]
    def test_named(root=ROOT):
        assert (root / "x.txt").exists()
    def test_tmp(tmp_path):
        root = tmp_path
        assert list(root.iterdir()) == []
    """,
], ids=["tmp-path", "third-party-package", "value-read-from-a-file", "one-named-file",
        "same-name-in-another-scope"])
def test_a_walk_of_something_else_is_not_reported(source):
    assert _walks(source) == []


def test_a_script_named_by_path_is_reported_as_loaded():
    assert script_names('P = ROOT / "scripts" / "check_hygiene.py"\nQ = "scripts/pin_weekly.py"\n') == {
        "check_hygiene.py", "pin_weekly.py"}


def test_unparsable_source_has_no_walks():
    assert tree_walks("def broken(:\n    ROOT.rglob('*')\n") == []
