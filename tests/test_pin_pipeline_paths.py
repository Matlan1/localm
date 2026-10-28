# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every directory the pin pipeline creates (its git worktree, per-runtime scratch, temp
dirs) resolves inside the main checkout, never beside it in the checkout's parent."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "pin_weekly.py"
_spec = importlib.util.spec_from_file_location("pin_weekly_paths_under_test", _PATH)
pw = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pw
_spec.loader.exec_module(pw)
pp = pw.pp

_RESERVED_KEYS = ("comfy", "worktree-tmp", "weekly-tmp")


def _git_common_dir_parent(checkout: Path) -> Path:
    out = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                         cwd=checkout, capture_output=True, text=True, check=True).stdout.strip()
    return Path(out).resolve().parent


def _init_repo(root: Path) -> Path:
    repo = root / "main"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "master"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=T",
                    "commit", "-q", "--allow-empty", "-m", "init"], cwd=repo, check=True)
    return repo


@pytest.fixture(params=["this-checkout", "main-checkout", "linked-worktree-outside-main"])
def checkout(request, tmp_path, monkeypatch):
    """The checkout the pipeline scripts run from, set as REPO in both modules."""
    if request.param == "this-checkout":
        return pw.REPO
    repo = _init_repo(tmp_path)
    chosen = repo
    if request.param == "linked-worktree-outside-main":
        chosen = tmp_path / "elsewhere" / "linked"
        subprocess.run(["git", "worktree", "add", "-q", "--detach", str(chosen)], cwd=repo, check=True)
    monkeypatch.setattr(pw, "REPO", chosen)
    monkeypatch.setattr(pp, "REPO", chosen)
    return chosen


def _every_pipeline_path() -> dict[str, Path]:
    paths = {f"scratch_dir({a.key})": pw.scratch_dir(a.key) for a in pw.build_advancers()}
    worktree = pp.pipeline_worktree_path()
    paths["pipeline_worktree_path"] = worktree
    paths["worktree_temp_dir"] = pp.worktree_temp_dir(worktree)
    paths["comfyui_scratch_workdir"] = pp._comfyui_scratch_workdir()
    paths["run_temp_dir"] = pw.run_temp_dir()
    return paths


def test_every_pin_pipeline_path_resolves_inside_the_repo_root(checkout):
    root = _git_common_dir_parent(checkout)
    paths = _every_pipeline_path()
    outside = {name: str(p) for name, p in paths.items()
               if not p.resolve().is_relative_to(root) or p.resolve() == root}
    assert not outside, f"pin-pipeline paths outside {root}: {outside}"


def test_pipeline_paths_land_in_the_scratch_and_worktrees_dirs(checkout):
    root = _git_common_dir_parent(checkout)
    paths = _every_pipeline_path()
    assert paths.pop("pipeline_worktree_path").resolve() == root / ".claude" / "worktrees" / "pin-pipeline"
    for name, p in paths.items():
        assert p.parent.resolve() == root / "scratch" / "pin-pipeline", name


def test_the_scratch_and_worktrees_dirs_are_gitignored():
    root = _git_common_dir_parent(pw.REPO)
    for rel in ("scratch/pin-pipeline/x", ".claude/worktrees/pin-pipeline/x"):
        ignored = subprocess.run(["git", "check-ignore", "-q", rel], cwd=root)
        assert ignored.returncode == 0, rel


def test_scratch_keys_are_distinct_per_job():
    keys = [a.key for a in pw.build_advancers()]
    assert len(set(keys)) == len(keys)
    assert not set(keys) & set(_RESERVED_KEYS)


@pytest.mark.parametrize("key", ["", ".", "..", "../x", "a/b", "a\\b", "/abs", "C:x", ".hidden"])
def test_a_key_that_is_not_a_plain_name_is_refused(key, tmp_path):
    with pytest.raises(ValueError, match="not a plain scratch key"):
        pp.pipeline_scratch_dir(key, tmp_path)


def test_a_linked_worktree_resolves_to_its_main_checkout_like_git_does(tmp_path):
    repo = _init_repo(tmp_path)
    linked = tmp_path / "elsewhere" / "linked"
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(linked)], cwd=repo, check=True)

    assert pp.main_checkout_root(linked) == _git_common_dir_parent(linked) == repo.resolve()
    assert pp.main_checkout_root(repo) == repo
    assert pp.pipeline_worktree_path(linked) == repo.resolve() / ".claude" / "worktrees" / "pin-pipeline"
    assert pp.pipeline_scratch_dir("llama", linked) == repo.resolve() / "scratch" / "pin-pipeline" / "llama"
    assert pp.worktree_temp_dir(linked) == repo.resolve() / "scratch" / "pin-pipeline" / "worktree-tmp"


def test_the_worktree_temp_dir_is_in_the_main_checkout_before_and_after_the_worktree_exists(tmp_path):
    repo = _init_repo(tmp_path)
    worktree = pp.pipeline_worktree_path(repo)
    expected = repo / "scratch" / "pin-pipeline" / "worktree-tmp"
    assert not worktree.exists() and pp.worktree_temp_dir(worktree) == expected
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(worktree)], cwd=repo, check=True)
    assert (worktree / ".git").is_file() and pp.worktree_temp_dir(worktree).resolve() == expected.resolve()


def test_a_dot_git_file_that_is_not_a_gitdir_link_is_an_infra_error(tmp_path):
    (tmp_path / ".git").write_text("garbage\n", encoding="utf-8")
    with pytest.raises(pp.InfraError, match="not a git worktree link"):
        pp.main_checkout_root(tmp_path)
