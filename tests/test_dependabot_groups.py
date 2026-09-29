# SPDX-License-Identifier: AGPL-3.0-or-later
"""The groups in .github/dependabot.yml.

Every ecosystem groups its minor and patch updates into one pull request; a
major update never joins a group. The uv group is a set of direct
dependencies: the [project] requirement lists, minus the group's
exclude-patterns, minus what the ignore rules skip. Two properties keep a
group pull request landing the way a single one would:

  - a dependency whose requirement carries a cap or a pin (an upper bound,
    ~= or ==) keeps its own pull request, because a bump edits a boundary
    that was verified;
  - bumping every member of the group at once still selects at most
    --max-share of the test files at --depth 0 in scripts/affected_tests.py,
    so the per-PR gate never reports a group as too wide for a targeted run.
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
_BUMPED_VERSION = "999.0.0"


def _load():
    spec = importlib.util.spec_from_file_location("affected_tests", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config() -> dict:
    return yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))


def _max_share() -> float:
    source = _SCRIPT.read_text(encoding="utf-8")
    return float(re.search(r'"--max-share",\s*type=float,\s*default=([0-9.]+)', source).group(1))


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


def _bump(lock: str, dists) -> str:
    for dist in sorted(dists):
        lock, count = re.subn(rf'(\[\[package\]\]\nname = "{re.escape(dist)}"\nversion = ")[^"]+(")',
                              rf"\g<1>{_BUMPED_VERSION}\g<2>", lock)
        assert count, f"{dist} has no entry in uv.lock"
    return lock


def _selection_share(monkeypatch, dists) -> float:
    """The share of test files the per-PR gate selects at depth 0 when every
    one of *dists* is bumped in uv.lock. Loads the script afresh, so no earlier
    call's patched readers leak into this one."""
    mod = _load()
    bumped = _bump((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"), dists)
    real_read = mod._read
    monkeypatch.setattr(mod, "_read", lambda rel: bumped if rel == "uv.lock" else real_read(rel))
    monkeypatch.setattr(mod, "_read_at", lambda ref, rel: real_read(rel))
    graph = mod.Graph()
    return len(mod.select(["uv.lock"], graph, depth=0)) / len(graph.test_files)


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


def test_bumping_every_member_of_the_uv_group_still_fits_the_per_pr_gate(monkeypatch):
    mod = _load()
    members, _ = _uv_members(mod)
    assert members
    limit = _max_share()
    share = _selection_share(monkeypatch, members)
    if share > limit:
        alone = sorted(((_selection_share(monkeypatch, {dist}), dist) for dist in sorted(members)),
                       reverse=True)
        wide = {dist: f"{s:.0%}" for s, dist in alone if s > limit}
        top = {dist: f"{s:.0%}" for s, dist in alone[:5]}
    assert share <= limit, (
        f"bumping all {len(members)} members selects {share:.0%} of the test files at depth 0 "
        f"(limit {limit:.0%}), so the gate would refuse the group pull request. "
        f"Too wide on their own, add to exclude-patterns: {wide}. Widest members: {top}")
