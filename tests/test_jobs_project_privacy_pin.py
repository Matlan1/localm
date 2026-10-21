# SPDX-License-Identifier: AGPL-3.0-or-later
"""A scheduled coder job resolves its session mode with the job's project
directory, so a project's ``.localcoder/config.toml`` ``mode`` applies to the
run when it persists less than the configured mode (precedence: LOCALM_MODE >
project config > coder_mode > mode > privacy); a project mode that would persist
more is ignored."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localm.audit import SessionMode


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _write_config(home: Path, **values) -> None:
    (home / "config.json").write_text(json.dumps(values), encoding="utf-8")


def _project(tmp_path: Path, toml: str | None) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    if toml is not None:
        (proj / ".localcoder").mkdir()
        (proj / ".localcoder" / "config.toml").write_text(toml, encoding="utf-8")
    return proj


def _capture_agent(monkeypatch) -> tuple:
    from localm.plugins.builtin.jobs import runner
    captured: dict = {}

    class _FakeAgent:
        def __init__(self, backend, cwd, **kw):
            captured["cwd"] = Path(cwd)
            captured.update(kw)

        def run_task(self, prompt):
            return "ran"

        def close(self):
            return None

    monkeypatch.setattr(runner, "_coder_backend", lambda job: object())
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", _FakeAgent)
    return runner, captured


def _run(runner, proj: Path) -> dict:
    from localm.plugins.builtin.jobs.store import Job
    job = Job(name="x", task_kind="coder", prompt="p", cwd=str(proj),
              schedule_kind="interval", schedule=60, owner=None)
    return runner.run_job(job, engine=None)


def test_a_project_privacy_pin_beats_a_logging_coder_mode(home, tmp_path, monkeypatch):
    _write_config(home, coder_mode="full")
    proj = _project(tmp_path, 'mode = "privacy"\n')
    runner, captured = _capture_agent(monkeypatch)

    assert _run(runner, proj)["status"] == "ok"

    assert captured["cwd"] == proj.resolve()
    assert captured["mode"] == SessionMode.PRIVACY


def test_without_a_project_pin_the_coder_mode_applies(home, tmp_path, monkeypatch):
    _write_config(home, coder_mode="full")
    proj = _project(tmp_path, None)
    runner, captured = _capture_agent(monkeypatch)

    assert _run(runner, proj)["status"] == "ok"

    assert captured["mode"] == SessionMode.FULL


def test_an_unreadable_project_config_runs_the_job_in_privacy(home, tmp_path,
                                                              monkeypatch):
    _write_config(home, coder_mode="full")
    proj = _project(tmp_path, 'mode = "privacy\n')
    runner, captured = _capture_agent(monkeypatch)

    assert _run(runner, proj)["status"] == "ok"

    assert captured["mode"] == SessionMode.PRIVACY


def test_localm_mode_still_outranks_the_project_pin(home, tmp_path, monkeypatch):
    _write_config(home, coder_mode="full")
    proj = _project(tmp_path, 'mode = "log"\n')
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    runner, captured = _capture_agent(monkeypatch)

    assert _run(runner, proj)["status"] == "ok"

    assert captured["mode"] == SessionMode.PRIVACY


def test_a_project_dir_that_cannot_be_searched_runs_the_job_in_privacy(
        home, tmp_path, monkeypatch, caplog):
    _write_config(home, coder_mode="full")
    proj = _project(tmp_path, 'mode = "log"\n')
    real_is_file = Path.is_file

    def _is_file(self):
        if self.name == "config.toml" and self.parent.name == ".localcoder":
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", _is_file)
    runner, captured = _capture_agent(monkeypatch)

    with caplog.at_level("WARNING"):
        assert _run(runner, proj)["status"] == "ok"

    assert captured["mode"] == SessionMode.PRIVACY
    assert any("privacy mode" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("global_cfg, pin, expected", [
    ({}, "full", SessionMode.PRIVACY),
    ({}, "log", SessionMode.PRIVACY),
    ({"coder_mode": "log"}, "full", SessionMode.LOG),
    ({"mode": "log"}, "full", SessionMode.LOG),
    ({"coder_mode": "full"}, "log", SessionMode.LOG),
    ({"coder_mode": "full"}, "privacy", SessionMode.PRIVACY),
    ({"coder_mode": "log"}, "log", SessionMode.LOG),
])
def test_a_project_pin_can_only_tighten_a_scheduled_job(
        home, tmp_path, monkeypatch, global_cfg, pin, expected):
    _write_config(home, **global_cfg)
    proj = _project(tmp_path, f'mode = "{pin}"\n')
    runner, captured = _capture_agent(monkeypatch)

    assert _run(runner, proj)["status"] == "ok"

    assert captured["mode"] == expected


def test_a_project_pin_that_would_loosen_a_job_is_logged(
        home, tmp_path, monkeypatch, caplog):
    _write_config(home, coder_mode="log")
    proj = _project(tmp_path, 'mode = "full"\n')
    runner, captured = _capture_agent(monkeypatch)

    with caplog.at_level("WARNING"):
        assert _run(runner, proj)["status"] == "ok"

    assert captured["mode"] == SessionMode.LOG
    assert any("project config" in r.getMessage() and "full" in r.getMessage()
               for r in caplog.records)


def test_a_confined_job_uses_the_pin_of_the_directory_it_runs_in(
        home, tmp_path, monkeypatch):
    from localm import auth
    from localm.plugins.builtin.jobs.store import Job, JobStore
    _write_config(home, coder_mode="full")
    root = tmp_path / "root"
    (root / ".localcoder").mkdir(parents=True)
    (root / ".localcoder" / "config.toml").write_text('mode = "privacy"\n',
                                                     encoding="utf-8")
    elsewhere = _project(tmp_path, 'mode = "full"\n')
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: str(root))
    auth.set_api_key("owner-key-privacy-pin-0123456789")
    created = auth.create_key("bot", ["jobs"], allow_privileged=False)
    job = Job(name="x", task_kind="coder", prompt="p", cwd=str(elsewhere),
              schedule_kind="interval", schedule=60,
              owner=auth._hash_key(created["key"]))
    JobStore().add(job)
    runner, captured = _capture_agent(monkeypatch)

    assert runner.run_job(job, engine=None)["status"] == "ok"

    assert captured["cwd"] == root.resolve()
    assert captured["mode"] == SessionMode.PRIVACY
