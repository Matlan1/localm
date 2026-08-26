# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.audit - mode resolution and per-surface enforcement."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localm.audit import (
    AuditLog,
    MarkdownTranscript,
    NullAuditLog,
    SessionMode,
    effective_mode,
    make_audit_log,
    make_transcript,
)


# ------------------------------------------------------------------ #
#  effective_mode precedence                                          #
# ------------------------------------------------------------------ #

def _cfg(**kw):
    base = {"mode": "privacy", "chat_mode": None, "coder_mode": None}
    base.update(kw)
    return base


class TestEffectiveMode:
    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("LOCALM_MODE", raising=False)

    @pytest.mark.parametrize("mode,expected", [
        ("privacy", SessionMode.PRIVACY),
        ("log", SessionMode.LOG),
    ])
    def test_global_mode_applies_to_all_surfaces(self, mode, expected):
        with patch("localm.config.load_config", return_value=_cfg(mode=mode)):
            assert effective_mode("chat") == expected
            assert effective_mode("coder") == expected
            assert effective_mode("server") == expected

    def test_per_surface_overrides_global(self):
        cfg = _cfg(mode="log", chat_mode="full", coder_mode="privacy")
        with patch("localm.config.load_config", return_value=cfg):
            assert effective_mode("chat") == SessionMode.FULL
            assert effective_mode("coder") == SessionMode.PRIVACY
            assert effective_mode("server") == SessionMode.LOG  # no server override

    def test_env_beats_everything(self, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "full")
        cfg = _cfg(mode="log", chat_mode="privacy")
        with patch("localm.config.load_config", return_value=cfg):
            assert effective_mode("chat") == SessionMode.FULL
            assert effective_mode("server") == SessionMode.FULL

    def test_invalid_values_fall_through(self, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "bogus")
        cfg = _cfg(mode="nonsense", chat_mode="alsobad")
        with patch("localm.config.load_config", return_value=cfg):
            assert effective_mode("chat") == SessionMode.PRIVACY

    def test_config_failure_means_privacy(self):
        with patch("localm.config.load_config", side_effect=OSError("boom")):
            assert effective_mode("chat") == SessionMode.PRIVACY

    def test_unreadable_project_config_means_privacy_not_the_global_mode(
            self, tmp_path):
        """An UNPARSEABLE .localcoder/config.toml must fail SAFE, not fall
        through to a more permissive global mode.

        The project file is the only place a user can say "this project is
        private". When it does not parse we cannot know whether it said so, and
        resolution used to continue to the global coder_mode - so a user whose
        global mode is "log" gets a full transcript written for a session they
        had marked private, with nothing said. Same direction as
        test_config_failure_means_privacy above: a mode we cannot establish
        resolves to the safest one.
        """
        cfg_dir = tmp_path / ".localcoder"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text(
            'mode = "privacy"\nthis is NOT [valid toml [\n', encoding="utf-8")
        cfg = _cfg(mode="log", coder_mode="log")
        with patch("localm.config.load_config", return_value=cfg):
            assert effective_mode("coder", cwd=tmp_path) == SessionMode.PRIVACY

            # The control: with NO config file the global mode must still be
            # honoured, or this test would pass on a resolver that returned
            # PRIVACY unconditionally. Remove the file rather than pointing at a
            # sibling directory - find_project_config walks UP, so a subdir of
            # tmp_path still finds tmp_path's own file (that cost a red here).
            (cfg_dir / "config.toml").unlink()
            # Self-check, so an ancestor .localcoder anywhere above the tmp tree
            # fails loudly instead of quietly changing what this control means.
            from localm.plugins.coder.project_config import find_project_config
            assert find_project_config(tmp_path) is None
            assert effective_mode("coder", cwd=tmp_path) == SessionMode.LOG


# ------------------------------------------------------------------ #
#  Factories + transcript                                             #
# ------------------------------------------------------------------ #

class TestFactories:
    def test_privacy_gets_null_log_and_no_transcript(self):
        assert isinstance(make_audit_log(SessionMode.PRIVACY), NullAuditLog)
        assert make_transcript(SessionMode.PRIVACY) is None
        assert make_transcript(SessionMode.LOG) is None

    def test_log_mode_writes_jsonl(self, tmp_path):
        with patch("localm.audit._SESSIONS_DIR", tmp_path):
            log = make_audit_log(SessionMode.LOG, label="chat")
            assert isinstance(log, AuditLog)
            log.user("hello")
            log.llm("world")
            log.close()
            files = list(tmp_path.glob("*_chat.jsonl"))
            assert len(files) == 1
            events = [json.loads(l) for l in
                      files[0].read_text(encoding="utf-8").splitlines()]
            types = [e["type"] for e in events]
            assert types == ["system", "user", "llm", "system"]
            assert events[1]["data"]["content"] == "hello"

    def test_full_mode_transcript(self, tmp_path):
        with patch("localm.audit._SESSIONS_DIR", tmp_path):
            t = make_transcript(SessionMode.FULL, label="chat")
            assert isinstance(t, MarkdownTranscript)
            t.exchange("question?", "answer.")
            text = t.path.read_text(encoding="utf-8")
            assert "## You" in text and "question?" in text
            assert "## Assistant" in text and "answer." in text


# ------------------------------------------------------------------ #
#  Back-compat shim                                                   #
# ------------------------------------------------------------------ #

class TestShim:
    def test_coder_audit_module_is_aliased(self):
        import localm.audit as core
        from localm.plugins.coder import audit as shim
        assert shim is core
        assert shim.SessionMode is SessionMode

    def test_patching_shim_sessions_dir_affects_core(self, tmp_path):
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="x")
            log.close()
            assert list(tmp_path.glob("*_x.jsonl"))


# ------------------------------------------------------------------ #
#  HTTP server enforcement                                            #
# ------------------------------------------------------------------ #

def _engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.chat_stream.side_effect = lambda messages, **kw: iter(["Hi", " there"])
    engine.count_tokens.side_effect = lambda text: max(1, len(text.split()))
    return engine


def _make_app(tmp_path, mode):
    from localm.inference.http_server import create_app
    with patch("localm.audit._SESSIONS_DIR", tmp_path), \
         patch("localm.config.load_config",
               return_value=_cfg(mode=mode, n_ctx_max=16384)):
        return create_app(_engine())


_CHAT_BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "what is up"}],
}


class TestServerModes:
    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("LOCALM_MODE", raising=False)

    def test_privacy_writes_nothing(self, tmp_path):
        app = _make_app(tmp_path, "privacy")
        with patch("localm.audit._SESSIONS_DIR", tmp_path):
            with TestClient(app) as client:
                r = client.post("/v1/chat/completions", json=_CHAT_BODY)
        assert r.status_code == 200
        assert list(tmp_path.iterdir()) == []

    def test_log_mode_audits_exchange(self, tmp_path):
        app = _make_app(tmp_path, "log")
        with patch("localm.audit._SESSIONS_DIR", tmp_path):
            with TestClient(app) as client:
                r = client.post("/v1/chat/completions", json=_CHAT_BODY)
        assert r.status_code == 200
        files = list(tmp_path.glob("*_server.jsonl"))
        assert len(files) == 1
        text = files[0].read_text(encoding="utf-8")
        assert "what is up" in text
        assert "Hi there" in text
        # no markdown transcript in log mode
        assert not list(tmp_path.glob("*.md"))

    def test_log_mode_audits_streamed_exchange(self, tmp_path):
        app = _make_app(tmp_path, "log")
        with patch("localm.audit._SESSIONS_DIR", tmp_path):
            with TestClient(app) as client:
                with client.stream("POST", "/v1/chat/completions",
                                   json={**_CHAT_BODY, "stream": True}) as r:
                    assert r.status_code == 200
                    for _ in r.iter_lines():
                        pass
        files = list(tmp_path.glob("*_server.jsonl"))
        assert len(files) == 1
        text = files[0].read_text(encoding="utf-8")
        assert "what is up" in text and "Hi there" in text

    def test_full_mode_also_writes_markdown(self, tmp_path):
        app = _make_app(tmp_path, "full")
        with patch("localm.audit._SESSIONS_DIR", tmp_path):
            with TestClient(app) as client:
                r = client.post("/v1/chat/completions", json=_CHAT_BODY)
        assert r.status_code == 200
        md = list(tmp_path.glob("*_server.md"))
        assert len(md) == 1
        text = md[0].read_text(encoding="utf-8")
        assert "what is up" in text and "Hi there" in text

    def test_config_reports_effective_mode(self, tmp_path):
        app = _make_app(tmp_path, "log")
        with TestClient(app) as client:
            with patch("localm.config.load_config",
                       return_value=_cfg(mode="log")):
                data = client.get("/v1/config", headers={"Authorization": f"Bearer {app.state.shell_token}"}).json()
        assert data["effective_mode"] == "log"
        # AUD-INSTANCEID: /v1/config always carries a stable per-data-directory
        # id so the GUI can tell a normal restart of THIS install apart from a
        # different install sharing the same browser origin.
        assert data["instance_id"], "instance_id must be present and non-empty"


# ------------------------------------------------------------------ #
#  Coder checkpoint privacy gate                                      #
# ------------------------------------------------------------------ #

class TestCheckpointPrivacyGate:
    def _agent(self, tmp_path, mode):
        # A REAL Agent, not a Agent.__new__(Agent) stub with hand-picked
        # attributes: that shape broke twice (once for _todos/B2, again for
        # _changed_files) because save_checkpoint's serialised state keeps
        # growing and nothing forces a hand-built stub to grow with it - each
        # gap surfaces as an AttributeError that reads exactly like a
        # production bug, not a stale fixture (it did, here, for several
        # minutes). A real __init__ can never go stale relative to itself, so
        # this is immune to the NEXT field save_checkpoint learns to read.
        # Same construction pattern as tests/plugins/coder/test_session_mode.py's
        # _make_agent - a MagicMock backend plus patched make_audit_log/
        # load_memory/ProjectMap keeps this as cheap as the old stub.
        from localm.plugins.coder.agent import Agent
        backend = MagicMock()
        backend.model_id = "test-model"
        backend.last_usage = {}
        with patch("localm.plugins.coder.agent.make_audit_log") as mock_factory, \
             patch("localm.plugins.coder.agent.load_memory", return_value=""), \
             patch("localm.plugins.coder.agent.ProjectMap") as mock_pm:
            mock_pm.build.return_value.file_count.return_value = 0
            mock_factory.return_value = NullAuditLog()
            agent = Agent(backend=backend, cwd=tmp_path, mode=mode)
        agent._turns = 1
        agent._total_tokens = 10
        agent._messages = [{"role": "user", "content": "secret stuff"}]
        # The checkpoint also carries the model's task list (B2).
        agent._todos = [{"text": "secret plan step", "status": "in_progress"}]
        return agent

    def test_privacy_mode_writes_no_checkpoint(self, tmp_path, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
        agent = self._agent(tmp_path, SessionMode.PRIVACY)
        agent.save_checkpoint()
        # Nothing in the project tree, nothing under HOME (CODER-4). The task
        # list rides in the same file, so it is covered by the same promise (B2).
        assert not (tmp_path / ".localcoder" / "checkpoint.json").exists()
        assert not (tmp_path / "home" / "checkpoints").exists()

    def test_log_mode_writes_checkpoint_under_home(self, tmp_path, monkeypatch):
        # CODER-4: the checkpoint is session DATA - it lives under HOME, not in
        # the project tree (which used to gain a stray .localcoder/ folder).
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
        agent = self._agent(tmp_path, SessionMode.LOG)
        agent.save_checkpoint()
        assert not (tmp_path / ".localcoder").exists()       # nothing in the repo
        path = agent._checkpoint_path
        assert (tmp_path / "home") in path.parents           # under HOME/checkpoints
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "secret stuff" in text
        assert "secret plan step" in text          # the task list rides along (B2)


# ------------------------------------------------------------------ #
#  Sidecar privacy gate (generators)                                  #
# ------------------------------------------------------------------ #

class TestSidecarGate:
    def test_generate_image_skips_sidecar_when_disabled(self, tmp_path):
        """write_sidecar=False must leave only the image file on disk."""
        from localm.image_gen import comfy as ic
        from localm.media import comfy_client
        out = tmp_path / "img.png"

        fake_history = {"p1": {"outputs": {"9": {"images": [
            {"filename": "x.png", "subfolder": "", "type": "output"}]}}}}

        class _Resp:
            def __init__(self, payload): self._p = payload
            def read(self): return self._p
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            if "/prompt" in url:
                return _Resp(json.dumps({"prompt_id": "p1"}).encode())
            if "/history/" in url:
                return _Resp(json.dumps(fake_history).encode())
            if "/view" in url:
                return _Resp(b"\x89PNG\r\n\x1a\nfake")
            if "/system_stats" in url:
                return _Resp(b"{}")
            raise AssertionError(url)

        with patch.object(comfy_client, "_comfy_urlopen", side_effect=fake_urlopen), \
             patch.object(ic, "_localm_unload"), \
             patch.dict(ic.os.environ, {"COMFY_OUTPUT_DIR": str(tmp_path / "comfy")}):
            ok, msg = ic.generate_image("a fox", out, write_sidecar=False)

        assert ok, msg
        assert out.is_file()
        assert not out.with_suffix(out.suffix + ".json").exists()


# ------------------------------------------------------------------ #
#  Self-contained data dir resolution (localm.config._detect_home)    #
# ------------------------------------------------------------------ #

class TestDetectHome:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        from localm.config import _detect_home
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "custom"))
        assert _detect_home() == tmp_path / "custom"

    def test_default_is_contained_never_user_localm(self, tmp_path, monkeypatch, capsys):
        # No LOCALM_HOME, no marker, no ./home: the default is a CONTAINED ./home
        # inside the install, surfaced on stderr - NEVER a shared ~/.localm outside it.
        from localm import config as cfg
        monkeypatch.delenv("LOCALM_HOME", raising=False)
        # Path.home() must not decide the default now; point it at an obvious dir so
        # any regression back to ~/.localm would show up as this path.
        monkeypatch.setattr(cfg.Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(cfg, "_warned_unconfigured_home", False)
        result = cfg._detect_home()
        repo_root = cfg.Path(cfg.__file__).resolve().parents[1]
        assert result != tmp_path / ".localm"          # never the shared user dir
        if not (repo_root / "localm-home.cfg").is_file():
            assert result == repo_root / "home"        # contained fallback in the install
            if not (repo_root / "home").is_dir():
                assert "no data directory is configured" in capsys.readouterr().err

    def test_portable_marker_file(self, tmp_path, monkeypatch):
        from localm import config as cfg
        monkeypatch.delenv("LOCALM_HOME", raising=False)
        # simulate a checkout: pyproject + marker pointing at a custom dir
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "localm-home.cfg").write_text(str(tmp_path / "data"))
        fake_file = tmp_path / "localm" / "config.py"
        fake_file.parent.mkdir()
        fake_file.write_text("# stub")
        monkeypatch.setattr(cfg, "__file__", str(fake_file))
        assert cfg._detect_home() == tmp_path / "data"

    def test_portable_home_dir(self, tmp_path, monkeypatch):
        from localm import config as cfg
        monkeypatch.delenv("LOCALM_HOME", raising=False)
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "home").mkdir()
        fake_file = tmp_path / "localm" / "config.py"
        fake_file.parent.mkdir()
        fake_file.write_text("# stub")
        monkeypatch.setattr(cfg, "__file__", str(fake_file))
        assert cfg._detect_home() == tmp_path / "home"
