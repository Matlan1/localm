# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py check 7: no module-level import cycles between the
top-level units under localm/.

Acyclicity rather than a declared tier map: localm has no declared layering, and
acyclicity needs no map and no allowlist. The two shapes a tier map would have to
carve out - an entry point, and unordered peers - are legal by construction,
because neither can form a cycle.

These tests pin the things that make the check worth having: it FIRES on a real
cycle, it does NOT fire on the entry-point and peer shapes, and it ignores
function-local imports, which are the standard way to break a cycle in Python.
They also pin the two things a cycle can hide behind that the check must still
SEE: a RELATIVE import (``from ..x import y``), resolved to its absolute target
rather than skipped, and an eager import inside a module-level ``try:``/``if:``
body, which runs during import exactly like a bare top-level statement despite
being indented - unlike a ``def``/``class`` body nested inside one of them, which
stays genuinely deferred.
"""

import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pkg(tmp_path, files: dict[str, str]) -> Path:
    """Build a throwaway localm/ package tree from {relative path: source}."""
    root = tmp_path / "localm"
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
#  NEGATIVE: it must actually fire                                             #
# --------------------------------------------------------------------------- #

def test_module_level_cycle_is_detected(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": "from localm.plugins.engine import PluginEngine\n",
        "plugins/engine.py": "from localm.inference.engine import Engine\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    problems = ch._import_cycle_violations()
    assert problems, "a real inference <-> plugins cycle must be reported"
    joined = "\n".join(problems)
    assert "import cycle" in joined
    assert "inference" in joined and "plugins" in joined
    # The report names both closing edges with a file:line, not merely that a
    # cycle exists.
    assert "inference/engine.py:1" in joined, joined
    assert "plugins/engine.py:1" in joined, joined


def test_three_unit_cycle_is_detected(tmp_path, monkeypatch):
    """A cycle does not have to be a mutual pair."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "a/x.py": "from localm.b.y import B\n",
        "b/y.py": "from localm.c.z import C\n",
        "c/z.py": "from localm.a.x import A\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    problems = ch._import_cycle_violations()
    assert problems
    joined = "\n".join(problems)
    for unit in ("a", "b", "c"):
        assert unit in joined, joined


# --------------------------------------------------------------------------- #
#  POSITIVE: the shapes a tier map would have to allowlist must be legal       #
# --------------------------------------------------------------------------- #

def test_entry_point_importing_downward_is_legal(tmp_path, monkeypatch):
    """localm/__main__.py -> localm.cli is a source node, never a cycle.

    This is the shape that would force an exception list into a hand-written
    tier map.
    """
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "__main__.py": "from localm.cli import main\n",
        "cli/_core.py": "from localm.config import load_config\n",
        "config.py": "X = 1\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._import_cycle_violations() == []


def test_peers_sharing_a_lower_unit_are_legal(tmp_path, monkeypatch):
    """image_gen / music_gen / video_gen all importing media is not an ordering
    violation - they are unordered peers, and no cycle exists."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "image_gen/comfy.py": "from localm.media.comfy_client import C\n",
        "music_gen/comfy.py": "from localm.media.comfy_client import C\n",
        "video_gen/comfy.py": "from localm.media.comfy_client import C\n",
        "media/comfy_client.py": "C = 1\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._import_cycle_violations() == []


def test_function_local_import_does_not_count(tmp_path, monkeypatch):
    """A deferred import is the standard, deliberate way to break a cycle and is
    NOT a violation. Only eager, module-level edges form the graph."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": (
            "def go():\n"
            "    from localm.plugins.engine import PluginEngine\n"
            "    return PluginEngine\n"
        ),
        "plugins/engine.py": "from localm.inference.engine import Engine\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._import_cycle_violations() == []


def test_self_import_within_a_unit_is_not_a_cycle(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": "from localm.inference.textnorm import scrub\n",
        "inference/textnorm.py": "def scrub():\n    pass\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._import_cycle_violations() == []


# --------------------------------------------------------------------------- #
#  Robustness                                                                  #
# --------------------------------------------------------------------------- #

def test_unparseable_file_is_reported_not_skipped(tmp_path, monkeypatch):
    """A file the gate cannot read is one it cannot vouch for. Silently treating
    it as edge-free would let a cycle hide behind a syntax error."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {"inference/broken.py": "def (\n"})
    monkeypatch.setattr(ch, "REPO", tmp_path)
    problems = ch._import_cycle_violations()
    assert any("could not parse" in p for p in problems), problems


def test_absent_localm_package_is_not_an_error(tmp_path, monkeypatch):
    """Not a localm checkout: other gates report that, this one stays quiet."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._import_cycle_violations() == []


# --------------------------------------------------------------------------- #
#  Relative imports: must resolve to their absolute target, not be skipped     #
# --------------------------------------------------------------------------- #

def test_relative_import_cycle_is_detected(tmp_path, monkeypatch):
    """A cycle built entirely from relative imports (``from ..x.y import z``) must
    be caught exactly like an absolute one. A graph that keys on ``node.level ==
    0`` skips every relative import, so a cycle built only out of them is
    invisible to it."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": "from ..plugins.engine import PluginEngine\n",
        "plugins/engine.py": "from ..inference.engine import Engine\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    problems = ch._import_cycle_violations()
    assert problems, "a cycle built from relative imports must be reported"
    joined = "\n".join(problems)
    assert "inference" in joined and "plugins" in joined


def test_relative_import_from_a_package_init_resolves_correctly(tmp_path, monkeypatch):
    """A relative import's anchor is ``__package__``, which for a package's
    ``__init__.py`` IS the package's own name - one level shallower than for a
    plain sibling module, whose ``__package__`` is its PARENT. Getting this wrong
    (e.g. always stripping a trailing component, as if every file were a plain
    module) would resolve a level-2 import from an ``__init__.py`` one package too
    high and silently miss this cycle."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "plugins/__init__.py": "from ..inference import get_engine\n",
        "inference/__init__.py": "from ..plugins import get_plugin\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    problems = ch._import_cycle_violations()
    assert problems, "the package-__init__ relative-import cycle must be reported"
    joined = "\n".join(problems)
    assert "inference" in joined and "plugins" in joined


def test_relative_self_import_within_a_unit_is_not_a_cycle(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": "from .textnorm import scrub\n",
        "inference/textnorm.py": "def scrub():\n    pass\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._import_cycle_violations() == []


def test_resolve_relative_import_is_package_vs_plain_module(tmp_path, monkeypatch):
    """Direct unit test of the resolver's arithmetic, asserting the relation
    rather than a literal. Same ``own_module`` string, same level-1 import, two
    different answers depending on ``is_package`` - because a package's
    ``__init__.py`` IS its own package, while a plain module of the same dotted
    name is a LEAF one level further down (its package is its parent)."""
    ch = _load_check_hygiene()
    node = ast.parse("from .x import y\n").body[0]
    # localm/plugins/__init__.py: __package__ == "localm.plugins" itself.
    assert ch._resolve_relative_import(node, "localm.plugins", is_package=True) \
        == "localm.plugins.x"
    # localm/plugins.py (a plain module, hypothetically): __package__ == "localm".
    assert ch._resolve_relative_import(node, "localm.plugins", is_package=False) \
        == "localm.x"


def test_resolve_relative_import_beyond_top_level_returns_none(tmp_path, monkeypatch):
    """A relative import whose dots walk past the top-level ``localm`` package is
    invalid Python (fails at runtime with "attempted relative import beyond
    top-level package") - the resolver must say "nothing to resolve", never guess
    a plausible-looking but wrong package."""
    ch = _load_check_hygiene()
    node = ast.parse("from ... import x\n").body[0]  # level == 3
    assert ch._resolve_relative_import(node, "localm.cli", is_package=False) is None


# --------------------------------------------------------------------------- #
#  Module-level try:/if: bodies are eager, not deferred like a function body   #
# --------------------------------------------------------------------------- #

def test_module_level_try_import_is_detected(tmp_path, monkeypatch):
    """``try: import X except ImportError: ...`` at module level runs eagerly
    during import - it is the standard optional-dependency idiom, not a deferred
    import. It must count exactly like a bare top-level import."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": (
            "try:\n"
            "    from localm.plugins.engine import PluginEngine\n"
            "except ImportError:\n"
            "    PluginEngine = None\n"
        ),
        "plugins/engine.py": "from localm.inference.engine import Engine\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    problems = ch._import_cycle_violations()
    assert problems, "an eager import inside a module-level try: body must count"
    joined = "\n".join(problems)
    assert "inference" in joined and "plugins" in joined


def test_module_level_try_except_handler_import_is_detected(tmp_path, monkeypatch):
    """The FALLBACK import in an ``except:`` handler is just as eager as the one
    in the ``try:`` body it replaces - both run during import, only one of them
    on any given run."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": (
            "try:\n"
            "    import does_not_exist_at_all\n"
            "except ImportError:\n"
            "    from localm.plugins.engine import PluginEngine\n"
        ),
        "plugins/engine.py": "from localm.inference.engine import Engine\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    problems = ch._import_cycle_violations()
    assert problems, "an eager import inside a module-level except: body must count"


def test_module_level_if_body_import_is_detected(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": (
            "import sys\n"
            "if sys.platform == 'win32':\n"
            "    from localm.plugins.engine import PluginEngine\n"
            "else:\n"
            "    PluginEngine = None\n"
        ),
        "plugins/engine.py": "from localm.inference.engine import Engine\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    problems = ch._import_cycle_violations()
    assert problems, "an eager import inside a module-level if: body must count"


def test_module_level_if_else_branch_import_is_detected(tmp_path, monkeypatch):
    """Either branch of an ``if``/``else`` can be the one that actually runs, so
    either must be able to create a real edge - not just the ``if`` body."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": (
            "import sys\n"
            "if sys.platform == 'win32':\n"
            "    PluginEngine = None\n"
            "else:\n"
            "    from localm.plugins.engine import PluginEngine\n"
        ),
        "plugins/engine.py": "from localm.inference.engine import Engine\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    problems = ch._import_cycle_violations()
    assert problems, "an eager import inside a module-level if/else body must count"


def test_type_checking_guarded_import_does_not_count(tmp_path, monkeypatch):
    """``if TYPE_CHECKING:`` is False at runtime by definition - an import inside
    it never actually executes, and is the standard way to add a type-only import
    WITHOUT creating a real cycle. Build a shape that WOULD be a cycle if this
    import counted, and confirm it is not reported: counting it would flag the
    exact idiom Python programs use to avoid a real cycle as if it created one."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from localm.plugins.engine import PluginEngine\n"
        ),
        "plugins/engine.py": "from localm.inference.engine import Engine\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._import_cycle_violations() == []


def test_def_inside_module_level_try_stays_deferred(tmp_path, monkeypatch):
    """A function DEFINED inside a module-level ``try:`` block is still a
    function: an import inside ITS body only runs when the function is later
    CALLED, exactly like a plain top-level ``def``. Recursing into ``try``/``if``
    bodies must not also start recursing into a ``def``/``class`` nested inside
    one of them."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/engine.py": (
            "try:\n"
            "    def go():\n"
            "        from localm.plugins.engine import PluginEngine\n"
            "        return PluginEngine\n"
            "except Exception:\n"
            "    pass\n"
        ),
        "plugins/engine.py": "from localm.inference.engine import Engine\n",
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._import_cycle_violations() == []


# --------------------------------------------------------------------------- #
#  The real tree                                                               #
# --------------------------------------------------------------------------- #

def test_the_shipped_tree_has_no_module_level_cycles():
    """The whole point: master is acyclic, so this check carries no allowlist.

    If this ever goes red, the fix is to move the shared code to a lower unit both
    sides can import - NOT to defer the import into a function, which hides the
    cycle rather than removing it.
    """
    ch = _load_check_hygiene()
    assert ch._import_cycle_violations() == []
