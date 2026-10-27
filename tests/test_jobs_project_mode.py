# SPDX-License-Identifier: AGPL-3.0-or-later
"""A scheduled coder job refuses an invalid project ``mode`` instead of
silently running with the global mode, as the one-shot coder runner does."""

from __future__ import annotations

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


@pytest.fixture
def agent_spy(monkeypatch):
    from localm.plugins.builtin.jobs import runner
    built: list = []

    class _FakeAgent:
        def __init__(self, backend, cwd, **kw):
            built.append(kw)

        def run_task(self, prompt):
            return "ran"

        def close(self):
            return None

    monkeypatch.setattr(runner, "_coder_backend", lambda job: object())
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", _FakeAgent)
    return runner, built


def _project(tmp_path, mode_line: str | None):
    proj = tmp_path / "proj"
    (proj / ".localcoder").mkdir(parents=True)
    if mode_line is not None:
        (proj / ".localcoder" / "config.toml").write_text(mode_line + "\n",
                                                          encoding="utf-8")
    return proj


def _job(cwd):
    from localm.plugins.builtin.jobs.store import Job
    return Job(name="j", task_kind="coder", prompt="do x", cwd=str(cwd),
               schedule_kind="interval", schedule=60)


@pytest.mark.parametrize("line", ['mode = "bogus"', 'mode = "privcy"', "mode = 3"])
def test_an_invalid_project_mode_is_refused(home, tmp_path, agent_spy, line):
    runner, built = agent_spy
    proj = _project(tmp_path, line)
    with pytest.raises(RuntimeError) as exc:
        runner._run_coder(_job(proj))
    assert "invalid mode" in str(exc.value), exc.value
    assert "config.toml" in str(exc.value)
    assert built == [], "the coder ran despite the invalid mode"


def test_the_refusal_is_recorded_in_the_job_result(home, tmp_path, agent_spy):
    runner, built = agent_spy
    record = runner.run_job(_job(_project(tmp_path, 'mode = "bogus"')))
    assert built == []
    assert "invalid mode" in str(record), record


@pytest.mark.parametrize("line, expected", [
    ('mode = "privacy"', "privacy"),
    ('mode = "Log"', "privacy"),
    (None, None),
])
def test_a_valid_or_absent_project_mode_still_runs(home, tmp_path, agent_spy, line, expected):
    runner, built = agent_spy
    runner._run_coder(_job(_project(tmp_path, line)))
    assert len(built) == 1
    if expected is not None:
        assert built[0]["mode"].value == expected
