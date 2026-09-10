# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py check 9: a module-level import never goes UP the
layering declared in docs/layering.toml.

The map lists tiers top to bottom and places every top-level unit under localm/
in exactly one of them. A module-level import may target only a lower tier;
units that share a tier are peers and never import each other at module level;
the package root sits below every tier; function-local imports are not counted.

These tests pin what makes the check worth having. It FIRES on an upward
import, on a peer import, on the ``from localm import x`` shape, on the package
root importing a unit, and on a map that is incomplete, stale, duplicated or
malformed. It does NOT fire on a downward import, on the entry-point and
shared-lower-unit shapes, or on a function-local import. The map's schema has
no field that could hold an exception, and a tier carrying one is rejected. The
last section runs the check against the real tree and the real map.
"""

import importlib.util
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiers(*tiers: tuple[str, list[str]]) -> str:
    """A layering map from (tier name, units) pairs, top to bottom."""
    out = []
    for name, units in tiers:
        quoted = ", ".join(f'"{u}"' for u in units)
        out.append(f'[[tier]]\nname = "{name}"\nrole = "{name} tier"\n'
                   f'units = [{quoted}]\n')
    return "\n".join(out)


def _tree(tmp_path, files: dict[str, str], layering: str) -> Path:
    """A throwaway checkout: localm/ built from {relative path: source} (an
    empty __init__.py is added when the caller gives none) and
    docs/layering.toml from *layering*. Returns the localm/ root."""
    root = tmp_path / "localm"
    root.mkdir(exist_ok=True)
    if "__init__.py" not in files:
        (root / "__init__.py").write_text("", encoding="utf-8")
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "layering.toml").write_text(layering, encoding="utf-8")
    return root


_THREE_TIERS = _tiers(("top", ["app"]), ("mid", ["svc", "svc2"]), ("low", ["util"]))
_THREE_TIER_FILES = {"app.py": "", "svc.py": "", "svc2.py": "", "util.py": ""}


def _check(tmp_path, monkeypatch, files, layering=_THREE_TIERS, base=_THREE_TIER_FILES):
    ch = _load_check_hygiene()
    _tree(tmp_path, {**base, **files}, layering)
    monkeypatch.setattr(ch, "REPO", tmp_path)
    return ch, ch._import_direction_violations()


# --------------------------------------------------------------------------- #
#  NEGATIVE: it must actually fire                                             #
# --------------------------------------------------------------------------- #

def test_upward_module_level_import_fires(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch,
                         {"util.py": "from localm.app import run\n"})
    assert len(problems) == 1, problems
    msg = problems[0]
    assert "goes UP the declared layering" in msg
    assert "util -> app" in msg
    assert "localm/util.py:1" in msg
    assert "'low'" in msg and "'top'" in msg


def test_peer_import_fires(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch,
                         {"svc.py": "from localm.svc2 import thing\n"})
    assert len(problems) == 1, problems
    assert "between peers" in problems[0]
    assert "svc -> svc2" in problems[0]
    assert "'mid'" in problems[0]


def test_from_localm_import_shape_fires(tmp_path, monkeypatch):
    """``from localm import app`` names the unit app, not the package root."""
    _, problems = _check(tmp_path, monkeypatch,
                         {"util.py": "from localm import app\n"})
    assert len(problems) == 1, problems
    assert "goes UP" in problems[0] and "util -> app" in problems[0]


def test_module_level_try_import_fires(tmp_path, monkeypatch):
    """An import inside a module-level try: runs at import time and counts."""
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "try:\n    from localm.app import run\nexcept ImportError:\n"
                   "    run = None\n"})
    assert len(problems) == 1 and "util -> app" in problems[0], problems


def test_package_root_importing_a_unit_fires(tmp_path, monkeypatch):
    """localm/__init__.py sits below every tier."""
    _, problems = _check(tmp_path, monkeypatch,
                         {"__init__.py": "from localm import util\n"})
    assert len(problems) == 1, problems
    assert "the package root" in problems[0]
    assert "<root> -> util" in problems[0]


def test_unplaced_unit_fires(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {"extra.py": ""})
    assert len(problems) == 1, problems
    assert "localm/extra is not placed" in problems[0]


def test_unplaced_package_fires(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {"newpkg/__init__.py": ""})
    assert len(problems) == 1, problems
    assert "localm/newpkg is not placed" in problems[0]


def test_stale_map_entry_fires(tmp_path, monkeypatch):
    layering = _tiers(("top", ["app"]), ("mid", ["svc", "svc2", "ghost"]),
                      ("low", ["util"]))
    _, problems = _check(tmp_path, monkeypatch, {}, layering)
    assert len(problems) == 1, problems
    assert "places 'ghost'" in problems[0]


def test_unit_placed_twice_rejects_the_map(tmp_path, monkeypatch):
    layering = _tiers(("top", ["app", "util"]), ("mid", ["svc", "svc2"]),
                      ("low", ["util"]))
    _, problems = _check(tmp_path, monkeypatch,
                         {"util.py": "from localm.app import run\n"}, layering)
    # The map is rejected as a whole: no direction verdict is issued against it.
    assert len(problems) == 1, problems
    assert "placed twice" in problems[0]
    assert "goes UP" not in problems[0]


def test_unit_placed_twice_within_one_tier_rejects_the_map(tmp_path, monkeypatch):
    layering = _tiers(("top", ["app"]), ("mid", ["svc", "svc2", "svc"]), ("low", ["util"]))
    _, problems = _check(tmp_path, monkeypatch, {}, layering)
    assert len(problems) == 1 and "placed twice" in problems[0], problems


def test_unparseable_file_is_reported(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {"util.py": "from localm.app import (\n"})
    assert len(problems) == 1 and "could not parse" in problems[0], problems


def test_repeated_tier_name_rejects_the_map(tmp_path, monkeypatch):
    layering = _tiers(("top", ["app"]), ("top", ["svc", "svc2"]), ("low", ["util"]))
    _, problems = _check(tmp_path, monkeypatch, {}, layering)
    assert len(problems) == 1 and "declared twice" in problems[0], problems


def test_a_tier_carrying_an_extra_key_is_rejected(tmp_path, monkeypatch):
    """The schema is name, role, units and nothing else, so the map has nowhere
    to hold an allow-list."""
    layering = _THREE_TIERS.replace('name = "low"\n', 'name = "low"\nallow = ["app"]\n')
    _, problems = _check(tmp_path, monkeypatch,
                         {"util.py": "from localm.app import run\n"}, layering)
    assert len(problems) == 1, problems
    assert "must carry exactly the keys name, role, units" in problems[0]
    assert "allow" in problems[0]


def test_invalid_toml_fails_loud(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {}, "[[tier]\nname = ")
    assert len(problems) == 1 and "not valid TOML" in problems[0], problems


def test_a_map_without_tiers_fails_loud(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {}, 'title = "nothing"\n')
    assert len(problems) == 1 and "[[tier]]" in problems[0], problems


def test_missing_map_fails_loud(tmp_path, monkeypatch):
    ch, _ = _check(tmp_path, monkeypatch, {})
    (tmp_path / "docs" / "layering.toml").unlink()
    problems = ch._import_direction_violations()
    assert len(problems) == 1 and "could not be read" in problems[0], problems


def test_eager_import_of_an_untracked_module_fires(tmp_path, monkeypatch):
    """With a tracked-file inventory, an untracked module is not required in
    the map, but a module-level import of it from a placed unit is reported."""
    ch, _ = _check(tmp_path, monkeypatch,
                   {"gen.py": "", "util.py": "from localm.gen import VERSION\n"})
    tracked = [tmp_path / "localm" / f for f in _THREE_TIER_FILES]
    problems = ch._import_direction_violations(tracked)
    assert len(problems) == 1, problems
    assert "unplaced unit: util -> gen" in problems[0]


# --------------------------------------------------------------------------- #
#  NEGATIVE: every block that runs at import time is walked                    #
# --------------------------------------------------------------------------- #

def test_class_body_import_fires(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch,
                         {"util.py": "class C:\n    from localm.app import run\n"})
    assert len(problems) == 1 and "util -> app" in problems[0], problems


def test_with_block_import_fires(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "import contextlib\nwith contextlib.suppress(ImportError):\n"
                   "    from localm.app import run\n"})
    assert len(problems) == 1 and "util -> app" in problems[0], problems


def test_loop_body_and_loop_else_imports_fire(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "for _ in ():\n    pass\nelse:\n    from localm.app import run\n",
        "svc.py": "while False:\n    from localm.app import run\n"})
    assert len(problems) == 2, problems
    assert any("util -> app" in p for p in problems), problems
    assert any("svc -> app" in p for p in problems), problems


def test_match_case_import_fires(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "match 1:\n    case 1:\n        from localm.app import run\n"})
    assert len(problems) == 1 and "util -> app" in problems[0], problems


def test_type_checking_else_branch_import_fires(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\n"
                   "else:\n    from localm.app import run\n"})
    assert len(problems) == 1 and "util -> app" in problems[0], problems


def test_def_nested_in_a_class_body_stays_deferred(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "class C:\n    def m(self):\n        from localm.app import run\n"
                   "        return run\n"})
    assert problems == []


def test_namespace_directory_without_init_is_a_unit(tmp_path, monkeypatch):
    """A directory with no __init__.py still imports (a namespace package), so
    it must be placed like any other unit."""
    _, problems = _check(tmp_path, monkeypatch,
                         {"nspkg/mod.py": "from localm.app import run\n"})
    assert len(problems) == 1, problems
    assert "localm/nspkg is not placed" in problems[0]


def test_namespace_directory_placed_in_the_map_is_judged(tmp_path, monkeypatch):
    layering = _tiers(("top", ["app"]), ("mid", ["svc", "svc2"]),
                      ("low", ["util", "nspkg"]))
    _, problems = _check(tmp_path, monkeypatch,
                         {"nspkg/mod.py": "from localm.app import run\n"}, layering)
    assert len(problems) == 1, problems
    assert "goes UP" in problems[0] and "nspkg -> app" in problems[0]


def test_tracked_inventory_counts_a_directory_by_any_tracked_module(tmp_path, monkeypatch):
    """The tracked-mode inventory needs no __init__.py either, and a directory
    named like one of the hygiene scanner's skipped directories still counts."""
    ch, _ = _check(tmp_path, monkeypatch, {"lib/x.py": "from localm.app import run\n"})
    tracked = [tmp_path / "localm" / f for f in (*_THREE_TIER_FILES, "lib/x.py")]
    assert ch._layering_units(tmp_path / "localm", tracked) == {
        "app", "svc", "svc2", "util", "lib"}
    problems = ch._import_direction_violations(tracked)
    assert len(problems) == 1 and "localm/lib is not placed" in problems[0], problems


def test_default_inventory_is_the_unfiltered_git_list(tmp_path, monkeypatch):
    """main() passes no list: the inventory is `git ls-files -- localm` with
    nothing filtered out, so a unit named vendor or lib is required too."""
    ch, _ = _check(tmp_path, monkeypatch, {"vendor/x.py": ""})
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))

        class R:
            stdout = ("localm/app.py\0localm/svc.py\0localm/svc2.py\0"
                      "localm/util.py\0localm/vendor/x.py\0").encode("utf-8")
        return R()

    monkeypatch.setattr(ch.subprocess, "run", fake_run)
    problems = ch._import_direction_violations()
    assert seen and seen[0][:4] == ["git", "ls-files", "-z", "--"], seen
    assert len(problems) == 1 and "localm/vendor is not placed" in problems[0], problems


def test_git_inventory_keeps_a_non_ascii_unit_name(tmp_path, monkeypatch):
    """git quotes a non-ASCII path in its default output; the NUL-separated form
    is raw, so the unit is still required and still matches the map."""
    layering = _tiers(("top", ["app"]), ("mid", ["svc", "svc2"]), ("low", ["util", "café"]))
    ch, _ = _check(tmp_path, monkeypatch, {"café.py": ""}, layering)

    def fake_run(cmd, **kwargs):
        class R:
            stdout = ("localm/app.py\0localm/svc.py\0localm/svc2.py\0"
                      "localm/util.py\0localm/café.py\0").encode("utf-8")
        return R()

    monkeypatch.setattr(ch.subprocess, "run", fake_run)
    assert ch._import_direction_violations() == []


def test_default_inventory_falls_back_to_disk_without_git(tmp_path, monkeypatch):
    ch, _ = _check(tmp_path, monkeypatch, {"extra.py": ""})

    def no_git(cmd, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(ch.subprocess, "run", no_git)
    problems = ch._import_direction_violations()
    assert len(problems) == 1 and "localm/extra is not placed" in problems[0], problems


# --------------------------------------------------------------------------- #
#  POSITIVE: legal shapes stay legal, with no allowlist anywhere              #
# --------------------------------------------------------------------------- #

def test_downward_imports_are_clean(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "app.py": "from localm.svc import a\nfrom localm.svc2 import b\n"
                  "from localm.util import c\n",
        "svc.py": "from localm.util import c\n",
        "svc2.py": "import localm.util\n",
    })
    assert problems == []


def test_entry_point_and_shared_lower_unit_are_legal_by_structure(tmp_path, monkeypatch):
    """The two shapes the map must express without an exception list: an entry
    point reaching down, and peers that all import one shared lower unit."""
    layering = _tiers(("entry", ["__main__"]), ("cli", ["cli"]),
                      ("models", ["image_gen", "music_gen", "video_gen"]),
                      ("runtimes", ["media"]))
    _, problems = _check(tmp_path, monkeypatch, {
        "__main__.py": "from localm.cli import main\n",
        "cli/__init__.py": "from localm.media.comfy_client import x\n",
        "image_gen/comfy.py": "from localm.media.comfy_client import x\n",
        "image_gen/__init__.py": "",
        "music_gen/comfy.py": "from localm.media.comfy_client import x\n",
        "music_gen/__init__.py": "",
        "video_gen/comfy.py": "from localm.media.comfy_client import x\n",
        "video_gen/__init__.py": "",
        "media/__init__.py": "",
        "media/comfy_client.py": "x = 1\n",
    }, layering, base={})
    assert problems == []


def test_function_local_upward_import_does_not_count(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "def f():\n    from localm.app import run\n    return run\n"})
    assert problems == []


def test_type_checking_guarded_upward_import_does_not_count(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
                   "    from localm.app import App\n"})
    assert problems == []


def test_not_type_checking_body_fires_and_its_else_does_not(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "from typing import TYPE_CHECKING\nif not TYPE_CHECKING:\n"
                   "    from localm.app import run\nelse:\n    from localm.svc import x\n"})
    assert len(problems) == 1 and "util -> app" in problems[0], problems


def test_type_checking_and_clause_body_does_not_count(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "import os\nfrom typing import TYPE_CHECKING\n"
                   "if TYPE_CHECKING and os.name:\n    from localm.app import run\n"})
    assert problems == []


def test_type_checking_or_clause_body_counts(tmp_path, monkeypatch):
    """``TYPE_CHECKING or x`` can be True at runtime, so its body is walked."""
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "import os\nfrom typing import TYPE_CHECKING\n"
                   "if TYPE_CHECKING or os.name:\n    from localm.app import run\n"})
    assert len(problems) == 1 and "util -> app" in problems[0], problems


def test_reading_the_root_package_is_always_legal(tmp_path, monkeypatch):
    _, problems = _check(tmp_path, monkeypatch, {
        "util.py": "from localm import __version__\nimport localm\n"})
    assert problems == []


def test_foreign_package_sharing_the_prefix_is_not_an_edge(tmp_path, monkeypatch):
    ch, problems = _check(tmp_path, monkeypatch,
                          {"util.py": "import localm_llama_runtime\n"})
    assert problems == []
    assert ch._module_level_import_edges(tmp_path / "localm").get("util") is None


def test_untracked_module_is_not_required_in_the_map(tmp_path, monkeypatch):
    ch, _ = _check(tmp_path, monkeypatch, {"gen.py": ""})
    tracked = [tmp_path / "localm" / f for f in _THREE_TIER_FILES]
    assert ch._import_direction_violations(tracked) == []


def test_absent_localm_package_is_not_an_error(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._import_direction_violations() == []


# --------------------------------------------------------------------------- #
#  The real tree and the real map                                              #
# --------------------------------------------------------------------------- #

def test_the_shipped_tree_obeys_the_declared_layering():
    """No grandfathered exception: the real tree is clean against the real map.

    If this goes red, the fix is to move the shared code down to a unit both
    sides can import, or to re-tier deliberately in docs/layering.toml with the
    reason in the pull request. Deferring the import into a function hides it
    from this check and is only for breaking a genuine import cycle.
    """
    ch = _load_check_hygiene()
    assert ch._import_direction_violations() == []
    assert ch._import_direction_violations(ch._tracked_files()) == []


def test_the_shipped_map_places_every_unit_once_and_has_no_exception_field():
    ch = _load_check_hygiene()
    data = tomllib.loads((REPO_ROOT / "docs" / "layering.toml").read_text(encoding="utf-8"))
    assert set(data) == {"tier"}
    for tier in data["tier"]:
        assert set(tier) == {"name", "role", "units"}, tier
    placed = [u for tier in data["tier"] for u in tier["units"]]
    assert len(placed) == len(set(placed)), "a unit is placed twice"
    assert set(placed) == ch._units_on_disk(REPO_ROOT / "localm")


def test_the_design_test_edges_are_legal_by_structure():
    """The decision record's own test: the entry point and the three generator
    peers must be legal because of where the map puts them, not because of an
    exception. Each edge is asserted to EXIST in the real graph first, so this
    cannot pass by the import having moved out of module scope."""
    ch = _load_check_hygiene()
    tiers, problems = ch._layering_tiers(
        (REPO_ROOT / "docs" / "layering.toml").read_text(encoding="utf-8"))
    assert not problems
    index = {u: i for i, (_, units) in enumerate(tiers) for u in units}
    edges = ch._module_level_import_edges(REPO_ROOT / "localm")

    assert edges["__main__"]["cli"].startswith("localm/__main__.py:")
    assert index["__main__"] == 0, "the entry point is the top tier"
    assert index["cli"] > index["__main__"]

    for gen in ("image_gen", "music_gen", "video_gen"):
        assert edges[gen]["media"].startswith(f"localm/{gen}/comfy.py:")
        assert index[gen] == index["image_gen"], "the generators are peers"
        assert index["media"] > index[gen]

    assert edges["model_manager"]["media"] == "localm/model_manager/scan.py:9"
    assert index["media"] > index["model_manager"], "media sits below model_manager"
    assert "model_manager" not in edges.get("media", {})
