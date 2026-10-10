# SPDX-License-Identifier: AGPL-3.0-or-later
"""The groups in .github/dependabot.yml.

Every ecosystem groups its minor and patch updates into one pull request; a
major update never joins a group. The uv group is a set of direct
dependencies: the [project] requirement lists, minus the group's
exclude-patterns, minus what the ignore rules skip. A dependency whose
requirement carries a cap or a pin (an upper bound, ~= or ==) keeps its own
pull request, because a bump edits a boundary that was verified.
"""

import fnmatch
import importlib.util
import re
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = REPO_ROOT / ".github" / "dependabot.yml"
_SCRIPT = REPO_ROOT / "scripts" / "affected_tests.py"


def _load():
    spec = importlib.util.spec_from_file_location("affected_tests", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config() -> dict:
    return yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))


def _uv_members(mod) -> tuple[set[str], dict[str, list[tuple[str, str]]]]:
    """(the direct dependencies the uv groups take, every direct requirement)."""
    uv = next(u for u in _config()["updates"] if u["package-ecosystem"] == "uv")
    ignored = {mod.dist_key(i["dependency-name"]) for i in uv.get("ignore", [])
               if "update-types" not in i}
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = mod._requirements(pyproject)
    assert requirements is not None, "a [project] requirement does not start with a name"
    members: set[str] = set()
    for group in uv["groups"].values():
        include = [mod.dist_key(p) for p in group.get("patterns", [])]
        exclude = [mod.dist_key(p) for p in group.get("exclude-patterns", [])]
        for dist in requirements:
            if dist in ignored:
                continue
            if (any(fnmatch.fnmatchcase(dist, p) for p in include)
                    and not any(fnmatch.fnmatchcase(dist, p) for p in exclude)):
                members.add(dist)
    return members, requirements


def test_every_ecosystem_groups_only_its_minor_and_patch_updates():
    updates = _config()["updates"]
    assert {u["package-ecosystem"] for u in updates} >= {"uv", "github-actions", "npm"}
    for update in updates:
        ecosystem = update["package-ecosystem"]
        groups = update.get("groups")
        assert groups, f"{ecosystem} has no group, so each of its updates opens its own pull request"
        for name, group in groups.items():
            assert set(group["update-types"]) == {"minor", "patch"}, \
                f"group {name} must take minor and patch updates only, so a major update keeps its own pull request"
            assert "applies-to" not in group, \
                f"group {name} must apply to version updates only, so a security update keeps its own pull request"


def test_the_uv_group_takes_no_dependency_with_a_cap_or_a_pin():
    mod = _load()
    members, requirements = _uv_members(mod)
    assert members
    capped = {dist for dist, pairs in requirements.items()
              if any(re.search(r"<|~=|==", req.split(";")[0]) for _, req in pairs)}
    assert not capped & members, (
        f"{sorted(capped & members)} carry a cap or a pin: add them to the uv group's exclude-patterns")
