# SPDX-License-Identifier: AGPL-3.0-or-later
"""``_coder_backend`` presents the owner's actual credential, never the open-mode
placeholder, when one is configured."""

from __future__ import annotations

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _make_job(**kw):
    from localm.plugins.builtin.jobs.store import Job
    base = dict(name="test", task_kind="chat", prompt="hi",
                schedule_kind="interval", schedule=60)
    base.update(kw)
    return Job(**base)


def test_persisted_owner_key_is_presented_not_the_placeholder(home, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_SELF_URL", raising=False)
    import localm.auth as auth
    auth.set_api_key("PersistedOwnerKey123456")          # writes a real <home>/auth.key
    assert auth.verify("localm") is None, (
        "precondition: this server must reject the placeholder, or the test proves nothing")

    captured = {}

    class _FakeBackend:
        def __init__(self, url, model=None, api_key=None, **kw):
            captured["api_key"] = api_key

    monkeypatch.setattr("localm.plugins.coder.backends.http.HTTPBackend", _FakeBackend)

    from localm.plugins.builtin.jobs import runner
    runner._coder_backend(_make_job(task_kind="coder", cwd=str(home)))

    # THE PROPERTY: the server itself must accept what the job presents.
    assert auth.verify(captured["api_key"]) is not None, (
        f"the scheduled job presents {captured['api_key']!r}, which this server "
        f"rejects: every self-call from a scheduled coder job 401s")
    assert captured["api_key"] == "PersistedOwnerKey123456"


def test_env_var_still_outranks_the_persisted_key(home, monkeypatch):
    monkeypatch.delenv("LOCALM_SELF_URL", raising=False)
    import localm.auth as auth
    auth.set_api_key("PersistedOwnerKey123456")
    monkeypatch.setenv("LOCALM_API_KEY", "EnvKey987654321")

    captured = {}

    class _FakeBackend:
        def __init__(self, url, model=None, api_key=None, **kw):
            captured["api_key"] = api_key

    monkeypatch.setattr("localm.plugins.coder.backends.http.HTTPBackend", _FakeBackend)

    from localm.plugins.builtin.jobs import runner
    runner._coder_backend(_make_job(task_kind="coder", cwd=str(home)))

    assert captured["api_key"] == "EnvKey987654321"


def test_open_mode_presents_the_instance_attach_token_not_the_placeholder(home, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_SELF_URL", raising=False)
    import localm.auth as auth
    assert auth.get_api_key() is None, (
        "precondition: no owner key configured, or this test proves nothing")

    fake_entry = {"base_url": "http://127.0.0.1:8642/v1", "token": "instTok",
                  "port": 8642, "scheme": "http", "mode": "gui"}
    from localm import instances
    monkeypatch.setattr(instances, "attach_target", lambda *a, **kw: fake_entry)

    captured = {}

    class _FakeBackend:
        def __init__(self, url, model=None, api_key=None, **kw):
            captured["api_key"] = api_key

    monkeypatch.setattr("localm.plugins.coder.backends.http.HTTPBackend", _FakeBackend)

    from localm.plugins.builtin.jobs import runner
    runner._coder_backend(_make_job(task_kind="coder", cwd=str(home)))

    assert captured["api_key"] == "instTok"
