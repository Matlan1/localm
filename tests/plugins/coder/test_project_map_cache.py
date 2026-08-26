# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-session project-map caching, wired into Agent._build_project_map
(persistence.py). The pure mechanism (save_cache/load_cached_and_reconcile) is
covered in test_indexer.py; this file covers the WIRING - a second Agent
constructed for the same project reuses the cache instead of doing a full
ProjectMap.build() - and that _project_map_path_for reuses the project-digest
scheme in checkpoint.py's _project_dir_for.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from localm.audit import SessionMode


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A controlled HOME_DIR, separate from the project directory each test
    constructs an Agent against - the cache lives under HOME/checkpoints/...,
    and it must never overlap with the project tree it describes (a cache
    file sitting INSIDE the scanned root would itself be indexed as a project
    file - see test_indexer.py's cross-contamination note)."""
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home_dir))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home_dir)
    monkeypatch.setattr(cfg, "MODELS_DIR", home_dir / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home_dir / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home_dir / "registry.json")
    return home_dir


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _agent(cwd, mode=SessionMode.LOG, **kw):
    from localm.plugins.coder.agent import Agent
    return Agent(_StubBackend(), cwd=cwd, mode=mode, **kw)


def _project(tmp_path) -> Path:
    p = tmp_path / "project"
    p.mkdir()
    (p / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    return p


def test_cache_path_reuses_the_1051_project_digest_scheme(home, tmp_path):
    from localm.plugins.coder.agent.checkpoint import (
        _project_dir_for, _project_map_path_for,
    )
    project = _project(tmp_path)
    assert (_project_map_path_for(project)
           == _project_dir_for(project) / "projectmap.json")


def test_first_agent_builds_full_and_writes_a_cache(home, tmp_path):
    project = _project(tmp_path)
    from localm.plugins.coder.agent.checkpoint import _project_map_path_for
    cache_path = _project_map_path_for(project)
    assert not cache_path.exists()

    agent = _agent(project)
    assert {f.path.name for f in agent._project_map.files} == {"a.py"}
    assert cache_path.is_file()   # left behind for the next session


def test_second_agent_in_the_same_project_loads_from_cache_not_a_full_build(
        home, tmp_path):
    """A second Agent for the SAME project must reconcile the cache rather than
    doing a full walk. ProjectMap.build() is still the one call persistence.py
    makes, so the WALK - not the build() call - is what must not happen on a
    cache hit. os.walk is the ground truth for that; ProjectMap.from_cache is
    the same signal persistence.py's own log message reads."""
    project = _project(tmp_path)
    _agent(project)   # first session: builds + caches

    import os
    with patch.object(os, "walk", wraps=os.walk) as walk_spy:
        agent2 = _agent(project)
        walk_spy.assert_not_called()
    assert agent2._project_map.from_cache is True
    assert {f.path.name for f in agent2._project_map.files} == {"a.py"}


def test_second_agent_sees_an_edit_made_between_sessions(home, tmp_path):
    project = _project(tmp_path)
    _agent(project)   # first session: builds + caches "def foo()"

    (project / "a.py").write_text("def bar():\n    pass\n", encoding="utf-8")
    agent2 = _agent(project)
    fs = next(f for f in agent2._project_map.files if f.path.name == "a.py")
    assert "bar" in fs.symbols and "foo" not in fs.symbols


def test_privacy_mode_agent_writes_no_project_map_cache(home, tmp_path):
    """The same promise save_checkpoint() makes for the conversation: the
    project map cache records this project's relative file paths and extracted
    symbol names - real project content - under HOME, so a privacy session must
    never leave it behind."""
    project = _project(tmp_path)
    from localm.plugins.coder.agent.checkpoint import _project_map_path_for
    cache_path = _project_map_path_for(project)
    assert not cache_path.exists()

    agent = _agent(project, mode=SessionMode.PRIVACY)
    # The feature itself must still work - only the disk artifact is gone.
    assert {f.path.name for f in agent._project_map.files} == {"a.py"}
    assert not cache_path.exists()


def test_privacy_mode_agent_ignores_an_existing_cache_too(home, tmp_path):
    """Not just "stops writing" - a privacy session must not even READ a
    cache an earlier non-privacy session left behind, so it always does a
    full in-memory walk rather than reconciling from a file on disk."""
    project = _project(tmp_path)
    _agent(project, mode=SessionMode.LOG)   # leaves a real cache behind
    from localm.plugins.coder.agent.checkpoint import _project_map_path_for
    cache_path = _project_map_path_for(project)
    assert cache_path.is_file()

    import os
    with patch.object(os, "walk", wraps=os.walk) as walk_spy:
        agent = _agent(project, mode=SessionMode.PRIVACY)
        walk_spy.assert_called()   # full walk, never reconciled from cache
    assert agent._project_map.from_cache is False
    assert {f.path.name for f in agent._project_map.files} == {"a.py"}


def test_two_different_projects_never_share_a_cache(home, tmp_path):
    project_a = _project(tmp_path)
    project_b = tmp_path / "project_b"
    project_b.mkdir()
    (project_b / "z.py").write_text("z = 1\n", encoding="utf-8")

    agent_a = _agent(project_a)
    agent_b = _agent(project_b)
    assert {f.path.name for f in agent_a._project_map.files} == {"a.py"}
    assert {f.path.name for f in agent_b._project_map.files} == {"z.py"}

    from localm.plugins.coder.agent.checkpoint import _project_map_path_for
    assert (_project_map_path_for(project_a)
           != _project_map_path_for(project_b))
