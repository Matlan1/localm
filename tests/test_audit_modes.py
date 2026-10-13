# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.audit - mode resolution and per-surface enforcement."""

import json
import sys
from contextlib import contextmanager
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
        private". When it does not parse we cannot know whether it said so, so
        continuing to the global coder_mode gives a user whose global mode is
        "log" a full transcript written for a session they had marked private,
        with nothing said. Same direction as test_config_failure_means_privacy
        above: a mode we cannot establish resolves to the safest one.
        """
        cfg_dir = tmp_path / ".localcoder"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text(
            'mode = "privacy"\nthis is NOT [valid toml [\n', encoding="utf-8")
        cfg = _cfg(mode="log", coder_mode="log")
        with patch("localm.config.load_config", return_value=cfg):
            assert effective_mode("coder", cwd=tmp_path) == SessionMode.PRIVACY

            # With NO config file the global mode is still honoured. The file is
            # removed rather than pointing at a sibling directory:
            # find_project_config walks UP, so a subdir of tmp_path would still
            # find tmp_path's own file.
            (cfg_dir / "config.toml").unlink()
            # Self-check: an ancestor .localcoder above the tmp tree fails loudly.
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
#  Session identity: two logs opened in the same second by the same   #
#  process must never collide (REG: coder session audit log merge)    #
# ------------------------------------------------------------------ #

class TestSessionIdentity:
    def test_distinct_session_ids_give_distinct_files_same_second_same_pid(
            self, tmp_path):
        with patch("localm.audit._SESSIONS_DIR", tmp_path), \
             patch("time.strftime", return_value="2026-01-01_000000"):
            log1 = AuditLog(label="localcoder", session_id="aaaaaaaa")
            log2 = AuditLog(label="localcoder", session_id="bbbbbbbb")
        try:
            assert log1.path != log2.path
            assert "aaaaaaaa" in log1.path.name
            assert "bbbbbbbb" in log2.path.name
        finally:
            log1.close()
            log2.close()

    def test_no_session_id_still_gives_distinct_files_same_second_same_pid(
            self, tmp_path):
        """A caller that passes no session_id (a future call site, or one of
        the existing non-coder ones) must not be able to reintroduce the
        collision: AuditLog mints its own id when none is given."""
        with patch("localm.audit._SESSIONS_DIR", tmp_path), \
             patch("time.strftime", return_value="2026-01-01_000000"):
            log1 = AuditLog(label="chat")
            log2 = AuditLog(label="chat")
        try:
            assert log1.path != log2.path
        finally:
            log1.close()
            log2.close()

    def test_every_record_is_tagged_with_the_session_id(self, tmp_path):
        with patch("localm.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="localcoder", session_id="deadbeef")
            log.user("hello")
            log.llm("hi there")
            log.close()
        events = [json.loads(l) for l in
                  log.path.read_text(encoding="utf-8").splitlines()]
        assert len(events) == 4  # started, user, llm, ended
        assert all(e["session"] == "deadbeef" for e in events)


# ------------------------------------------------------------------ #
#  Transcript identity: two transcripts opened in the same second by  #
#  the same process must never collide (REG: sibling of              #
#  TestSessionIdentity above, same shape as the coder markdown        #
#  transcript's own filename-collision fix)                           #
# ------------------------------------------------------------------ #

class TestTranscriptIdentity:
    def test_two_transcripts_same_second_same_pid_do_not_collide(self, tmp_path):
        with patch("localm.audit._SESSIONS_DIR", tmp_path), \
             patch("time.strftime", return_value="2026-01-01_000000"):
            t1 = MarkdownTranscript(label="server")
            t2 = MarkdownTranscript(label="server")
        assert t1.path != t2.path

    def test_colliding_transcripts_do_not_interleave_exchanges(self, tmp_path):
        """The append-only append mode means a collision does not overwrite -
        it interleaves two unrelated conversations into what reads as one
        transcript. Confirm each instance's exchange lands only in its own
        file."""
        with patch("localm.audit._SESSIONS_DIR", tmp_path), \
             patch("time.strftime", return_value="2026-01-01_000000"):
            t1 = MarkdownTranscript(label="server")
            t2 = MarkdownTranscript(label="server")
        t1.exchange("first question", "first answer")
        t2.exchange("second question", "second answer")
        text1 = t1.path.read_text(encoding="utf-8")
        text2 = t2.path.read_text(encoding="utf-8") if t2.path != t1.path else text1
        assert "first question" in text1 and "second question" not in text1
        assert "second question" in text2 and "first question" not in text2


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
    engine.chat_stream.side_effect = lambda messages, **kw: (t for t in ["Hi", " there"])
    engine.count_tokens.side_effect = lambda text: max(1, len(text.split()))
    return engine


class _Config:
    """A config whose session mode a test can change while the server runs."""

    def __init__(self, mode):
        self.cfg = _cfg(mode=mode, n_ctx_max=16384)

    def set_mode(self, mode):
        self.cfg = {**self.cfg, "mode": mode}

    def set(self, **kw):
        self.cfg = {**self.cfg, **kw}

    def load(self):
        return dict(self.cfg)


@contextmanager
def _server(tmp_path, mode, engine=None):
    """A TestClient for a server whose config starts in *mode*, with its session
    files in *tmp_path*. Yields ``(client, config)``; ``config.set_mode`` changes
    the configured mode for every later read."""
    from localm.inference.http_server import create_app
    config = _Config(mode)
    with patch("localm.audit._SESSIONS_DIR", tmp_path), \
         patch("localm.config.load_config", side_effect=config.load):
        app = create_app(engine or _engine())
        with TestClient(app) as client:
            client.headers["Authorization"] = f"Bearer {app.state.shell_token}"
            yield client, config


_CHAT_BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "what is up"}],
}


def _chat(client, text):
    r = client.post("/v1/chat/completions", json={
        "model": "test-model", "messages": [{"role": "user", "content": text}]})
    assert r.status_code == 200
    return r


def _session_text(tmp_path, pattern):
    return "".join(p.read_text(encoding="utf-8") for p in sorted(tmp_path.glob(pattern)))


class TestServerModes:
    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("LOCALM_MODE", raising=False)

    def test_privacy_writes_nothing(self, tmp_path):
        with _server(tmp_path, "privacy") as (client, _config):
            _chat(client, "what is up")
        entries = list(tmp_path.iterdir())
        assert [p.name for p in entries if p.is_file()] == []
        assert [p.name for p in entries if p.is_dir()] == [".localm"]

    def test_log_mode_audits_exchange(self, tmp_path):
        with _server(tmp_path, "log") as (client, _config):
            _chat(client, "what is up")
        files = list(tmp_path.glob("*_server.jsonl"))
        assert len(files) == 1
        text = files[0].read_text(encoding="utf-8")
        assert "what is up" in text
        assert "Hi there" in text
        # no markdown transcript in log mode
        assert not list(tmp_path.glob("*.md"))

    def test_log_mode_audits_streamed_exchange(self, tmp_path):
        with _server(tmp_path, "log") as (client, _config):
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
        with _server(tmp_path, "full") as (client, _config):
            _chat(client, "what is up")
        md = list(tmp_path.glob("*_server.md"))
        assert len(md) == 1
        text = md[0].read_text(encoding="utf-8")
        assert "what is up" in text and "Hi there" in text

    def test_config_reports_effective_mode(self, tmp_path):
        with _server(tmp_path, "log") as (client, _config):
            data = client.get("/v1/config").json()
        assert data["effective_mode"] == "log"
        # /v1/config always carries a stable per-data-directory id so the GUI can
        # tell a restart of THIS install apart from a different install on the
        # same browser origin.
        assert data["instance_id"], "instance_id must be present and non-empty"


class TestServerModeFollowsTheConfig:
    """A change of the configured mode applies to the running server at once."""

    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("LOCALM_MODE", raising=False)

    def test_switching_privacy_to_full_records_the_next_exchange(self, tmp_path):
        with _server(tmp_path, "privacy") as (client, config):
            _chat(client, "said in privacy")
            config.set_mode("full")
            assert client.get("/v1/config").json()["effective_mode"] == "full"
            _chat(client, "said in full")
        log = _session_text(tmp_path, "*_server.jsonl")
        md = _session_text(tmp_path, "*_server.md")
        assert "said in full" in log and "said in full" in md
        assert "said in privacy" not in log + md

    def test_switching_full_to_privacy_stops_recording_at_once(self, tmp_path):
        with _server(tmp_path, "full") as (client, config):
            _chat(client, "said in full")
            config.set_mode("privacy")
            assert client.get("/v1/config").json()["effective_mode"] == "privacy"
            _chat(client, "said in privacy")
        log = _session_text(tmp_path, "*_server.jsonl")
        md = _session_text(tmp_path, "*_server.md")
        assert "said in full" in log and "said in full" in md
        assert "said in privacy" not in log + md
        assert log.rstrip().endswith('"data": {"msg": "session ended"}}')

    def test_switching_full_to_privacy_mid_stream_does_not_record_that_reply(self, tmp_path):
        engine = _engine()
        holder = {}

        def _tokens(messages, **kw):
            yield "Hi"
            holder["config"].set_mode("privacy")
            yield " there"

        engine.chat_stream.side_effect = _tokens
        with _server(tmp_path, "full", engine=engine) as (client, config):
            holder["config"] = config
            with client.stream("POST", "/v1/chat/completions", json={
                    "model": "test-model", "stream": True,
                    "messages": [{"role": "user", "content": "streamed secret"}]}) as r:
                assert r.status_code == 200
                body = "".join(r.iter_lines())
        assert "there" in body
        assert config.load()["mode"] == "privacy"
        assert "streamed secret" not in _session_text(tmp_path, "*_server.*")

    def test_a_settings_save_of_the_mode_applies_without_a_restart(self, tmp_path):
        """The real PATCH /v1/config path the Settings page uses, against the
        real config file, not a patched load_config."""
        from localm.inference.http_server import create_app
        with patch("localm.audit._SESSIONS_DIR", tmp_path):
            app = create_app(_engine())
            with TestClient(app) as client:
                client.headers["Authorization"] = f"Bearer {app.state.shell_token}"
                assert client.get("/v1/config").json()["effective_mode"] == "privacy"
                r = client.patch("/v1/config", json={"mode": "full"})
                assert r.status_code == 200, r.text
                cfg = client.get("/v1/config").json()
                assert (cfg["mode"], cfg["effective_mode"]) == ("full", "full")
                _chat(client, "after the save")
                r = client.patch("/v1/config", json={"mode": "privacy"})
                assert r.status_code == 200, r.text
        assert "after the save" in _session_text(tmp_path, "*_server.md")


class TestSessionRecorder:
    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("LOCALM_MODE", raising=False)

    def test_a_mode_whose_files_cannot_open_keeps_the_previous_mode_and_warns_once(
            self, tmp_path, caplog):
        from localm.audit import SessionRecorder
        config = _Config("privacy")
        with patch("localm.audit._SESSIONS_DIR", tmp_path), \
             patch("localm.config.load_config", side_effect=config.load):
            recorder = SessionRecorder("server", label="server")
            config.set_mode("log")
            with patch("localm.audit.AuditLog", side_effect=PermissionError("denied")), \
                 caplog.at_level("WARNING", logger="localm"):
                first = recorder.current()
                second = recorder.current()
            warnings = [r for r in caplog.records
                        if "could not take effect" in r.getMessage()]
            assert len(warnings) == 1
            assert "denied" in warnings[0].getMessage()
            assert first[2] == second[2] == SessionMode.PRIVACY
            assert isinstance(second[0], NullAuditLog)
            audit, _transcript, mode = recorder.current()
            assert mode == SessionMode.LOG and isinstance(audit, AuditLog)
            recorder.close()

    def test_full_to_log_keeps_the_open_log_and_drops_the_transcript(self, tmp_path):
        from localm.audit import SessionRecorder
        config = _Config("full")
        with patch("localm.audit._SESSIONS_DIR", tmp_path), \
             patch("localm.config.load_config", side_effect=config.load):
            recorder = SessionRecorder("server", label="server")
            audit_full, transcript_full, _ = recorder.current()
            assert transcript_full is not None
            config.set_mode("log")
            with patch("localm.audit.AuditLog", side_effect=PermissionError("denied")):
                audit, transcript, mode = recorder.current()
            assert mode == SessionMode.LOG
            assert audit is audit_full and transcript is None
            config.set_mode("full")
            audit, transcript, mode = recorder.current()
            assert mode == SessionMode.FULL and audit is audit_full
            assert transcript is not None and transcript is not transcript_full
            recorder.close()
        assert len(list(tmp_path.glob("*.jsonl"))) == 1

    def test_close_ends_the_log_and_reports_privacy_after(self, tmp_path, caplog):
        from localm.audit import SessionRecorder
        config = _Config("log")
        with patch("localm.audit._SESSIONS_DIR", tmp_path), \
             patch("localm.config.load_config", side_effect=config.load):
            recorder = SessionRecorder("server", label="server")
            path = recorder.current()[0].path
            recorder.close()
            recorder.close()
            with caplog.at_level("WARNING", logger="localm"):
                audit, transcript, mode = recorder.current()
                config.set_mode("full")
                again = recorder.current()
            assert isinstance(audit, NullAuditLog) and transcript is None
            assert mode == again[2] == SessionMode.PRIVACY
            assert isinstance(again[0], NullAuditLog) and again[1] is None
            closed = [r for r in caplog.records if "recording is closed" in r.getMessage()]
            assert len(closed) == 1
        assert path.read_text(encoding="utf-8").count("session ended") == 1
        assert list(tmp_path.glob("*.jsonl")) == [path]


class TestSurfaceOverridesOnServerTraffic:
    """A request to the server's chat routes from the GUI chat (X-Client-Id:
    chat) or the coder (X-Client-Id: coder, plus its session mode in
    X-Localm-Session-Mode) is recorded no less privately than that surface's
    persistence override, and never less privately than the server's mode."""

    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("LOCALM_MODE", raising=False)

    @staticmethod
    def _post(client, text, headers=None, stream=False, route="chat"):
        if route == "chat":
            url, body = "/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": text}]}
        else:
            url, body = "/v1/completions", {"model": "test-model", "prompt": text}
        if stream:
            with client.stream("POST", url, json={**body, "stream": True},
                               headers=headers or {}) as r:
                assert r.status_code == 200
                for _ in r.iter_lines():
                    pass
        else:
            r = client.post(url, json=body, headers=headers or {})
            assert r.status_code == 200, r.text

    @pytest.mark.parametrize("route,stream", [
        ("chat", False), ("chat", True), ("completions", False), ("completions", True)])
    def test_chat_privacy_override_keeps_gui_chat_out_of_a_full_server(
            self, tmp_path, route, stream):
        with _server(tmp_path, "full") as (client, config):
            config.set(chat_mode="privacy")
            self._post(client, "gui chat secret", {"X-Client-Id": "chat"},
                       stream=stream, route=route)
            self._post(client, "plain api call", stream=stream, route=route)
        assert "plain api call" in _session_text(tmp_path, "*_server.jsonl")
        assert "plain api call" in _session_text(tmp_path, "*_server.md")
        assert "gui chat secret" not in _session_text(tmp_path, "*_server.*")

    def test_chat_log_override_drops_the_transcript_only(self, tmp_path):
        with _server(tmp_path, "full") as (client, config):
            config.set(chat_mode="log")
            self._post(client, "gui chat line", {"X-Client-Id": "chat"})
        assert "gui chat line" in _session_text(tmp_path, "*_server.jsonl")
        assert "gui chat line" not in _session_text(tmp_path, "*_server.md")

    def test_chat_full_override_never_records_above_a_privacy_server(self, tmp_path):
        with _server(tmp_path, "privacy") as (client, config):
            config.set(chat_mode="full")
            self._post(client, "gui chat line", {"X-Client-Id": "chat"})
            data = client.get("/v1/config").json()
        assert (data["effective_mode"], data["effective_chat_mode"]) == (
            "privacy", "full")
        assert "gui chat line" not in _session_text(tmp_path, "*_server.*")

    def test_config_reports_the_chat_mode_for_the_gui(self, tmp_path):
        with _server(tmp_path, "full") as (client, config):
            config.set(chat_mode="privacy")
            data = client.get("/v1/config").json()
            assert (data["effective_mode"], data["effective_chat_mode"]) == (
                "full", "privacy")
            config.set(chat_mode=None)
            assert client.get("/v1/config").json()["effective_chat_mode"] == "full"

    def test_a_chat_mode_change_mid_stream_keeps_that_reply_out(self, tmp_path):
        engine = _engine()
        holder = {}

        def _tokens(messages, **kw):
            yield "Hi"
            holder["config"].set(chat_mode="privacy")
            yield " there"

        engine.chat_stream.side_effect = _tokens
        with _server(tmp_path, "full", engine=engine) as (client, config):
            holder["config"] = config
            self._post(client, "streamed gui secret", {"X-Client-Id": "chat"},
                       stream=True)
        assert "streamed gui secret" not in _session_text(tmp_path, "*_server.*")

    @pytest.mark.parametrize("stream", [False, True])
    def test_coder_privacy_override_keeps_coder_calls_out_of_a_full_server(
            self, tmp_path, stream):
        with _server(tmp_path, "full") as (client, config):
            config.set(coder_mode="privacy")
            self._post(client, "contents of secret.py", {"X-Client-Id": "coder"},
                       stream=stream)
        assert "contents of secret.py" not in _session_text(tmp_path, "*_server.*")

    @pytest.mark.parametrize("asserted,in_log,in_md", [
        ("privacy", False, False),
        ("log", True, False),
        ("full", True, True),
        ("PRIVACY ", False, False),
        ("not-a-mode", False, False),
        ("", True, True),
    ])
    def test_the_coder_session_mode_header_caps_recording(
            self, tmp_path, asserted, in_log, in_md):
        with _server(tmp_path, "full") as (client, config):
            config.set(coder_mode="full")
            self._post(client, "coder prompt", {
                "X-Client-Id": "coder", "X-Localm-Session-Mode": asserted})
        assert ("coder prompt" in _session_text(tmp_path, "*_server.jsonl")) is in_log
        assert ("coder prompt" in _session_text(tmp_path, "*_server.md")) is in_md

    def test_a_caller_mode_header_never_records_above_the_server(self, tmp_path):
        with _server(tmp_path, "log") as (client, config):
            config.set(coder_mode="full")
            self._post(client, "asks for full", {
                "X-Client-Id": "coder", "X-Localm-Session-Mode": "full"})
            self._post(client, "untagged asks for full",
                       {"X-Localm-Session-Mode": "full"})
        log = _session_text(tmp_path, "*_server.jsonl")
        assert "asks for full" in log and "untagged asks for full" in log
        assert not list(tmp_path.glob("*_server.md"))

    def test_an_unknown_client_id_stays_on_the_server_mode(self, tmp_path):
        with _server(tmp_path, "full") as (client, config):
            config.set(chat_mode="privacy", coder_mode="privacy")
            self._post(client, "other client", {"X-Client-Id": "my-script"})
        assert "other client" in _session_text(tmp_path, "*_server.md")

    def test_the_coder_backend_headers_carry_the_session_mode(self, tmp_path):
        """The headers HTTPBackend sends to a localm server, replayed through
        the real route: a project-pinned privacy session records nothing."""
        from localm.plugins.coder.backends.http import HTTPBackend
        backend = HTTPBackend("http://127.0.0.1:1/v1", "test-model",
                              localm_server=True)
        backend.session_mode = SessionMode.PRIVACY
        headers = backend._headers()
        assert headers["X-Localm-Session-Mode"] == "privacy"
        with _server(tmp_path, "full") as (client, config):
            config.set(coder_mode="full")
            headers["Authorization"] = client.headers["Authorization"]
            self._post(client, "pinned project file", headers)
        assert "pinned project file" not in _session_text(tmp_path, "*_server.*")

    @pytest.mark.parametrize("held,recorded", [
        ({"chat"}, True),
        ({"admin"}, False),
    ])
    def test_only_an_owner_caller_can_narrow_recording(self, tmp_path, held, recorded):
        """A scoped (non-ADMIN) key's chat- or coder-tagged traffic is recorded
        under the server's mode; the owner's is capped."""
        from localm import scopes
        held = {scopes.ADMIN} if held == {"admin"} else held
        with _server(tmp_path, "full") as (client, config), \
             patch("localm.inference.http_server.caller_scopes", return_value=held):
            config.set(chat_mode="privacy", coder_mode="privacy")
            self._post(client, "tagged chat", {"X-Client-Id": "chat"})
            self._post(client, "tagged coder", {
                "X-Client-Id": "coder", "X-Localm-Session-Mode": "privacy"})
        text = _session_text(tmp_path, "*_server.*")
        assert ("tagged chat" in text) is recorded
        assert ("tagged coder" in text) is recorded

    @pytest.mark.parametrize("owner,chat_mode", [(True, "privacy"), (False, "full")])
    def test_config_reports_the_chat_mode_the_server_applies_to_the_caller(
            self, tmp_path, owner, chat_mode):
        """A non-owner caller's chat is recorded under the server's mode, so
        /v1/config gives it that mode as its chat mode."""
        from localm import scopes
        held = {scopes.ADMIN} if owner else {"chat"}
        with _server(tmp_path, "full") as (client, config), \
             patch("localm.inference.http_server.caller_scopes", return_value=held):
            config.set(chat_mode="privacy")
            data = client.get("/v1/config").json()
            self._post(client, "chat line", {"X-Client-Id": "chat"})
        assert data["effective_chat_mode"] == chat_mode
        recorded = "chat line" in _session_text(tmp_path, "*_server.*")
        assert recorded is (chat_mode != "privacy")

    @pytest.mark.parametrize("headers,held,expected", [
        ({}, None, SessionMode.FULL),
        ({"X-Client-Id": "coder"}, None, SessionMode.PRIVACY),
        ({"X-Localm-Session-Mode": "log"}, None, SessionMode.LOG),
        ({"X-Client-Id": "coder"}, {"chat"}, SessionMode.FULL),
    ])
    def test_request_session_mode_answers_what_the_server_records_under(
            self, headers, held, expected):
        from starlette.requests import Request

        from localm.inference.http_server import request_session_mode
        config = _Config("full")
        config.set(coder_mode="privacy")
        request = Request({"type": "http", "method": "POST", "path": "/",
                           "headers": [(k.lower().encode(), v.encode())
                                       for k, v in headers.items()]})
        with patch("localm.config.load_config", side_effect=config.load),              patch("localm.inference.http_server.caller_scopes", return_value=held):
            assert request_session_mode(request) == expected

    def test_localm_run_attached_to_the_server_records_by_the_chat_mode(self, tmp_path):
        """The headers `localm run`'s HttpEngine sends when attached to a
        server, replayed through the real route."""
        from localm.inference.http_engine import HttpEngine
        engine = HttpEngine("http://127.0.0.1:1/v1", token="t", model="test-model",
                            client_id="chat", session_mode=SessionMode.PRIVACY)
        headers = engine._headers()
        assert (headers["X-Client-Id"], headers["X-Localm-Session-Mode"]) == (
            "chat", "privacy")
        assert "X-Client-Id" not in HttpEngine("http://127.0.0.1:1/v1")._headers()
        with _server(tmp_path, "full") as (client, config):
            config.set(chat_mode="full")
            headers["Authorization"] = client.headers["Authorization"]
            self._post(client, "run --mode privacy line", headers)
        assert "run --mode privacy line" not in _session_text(tmp_path, "*_server.*")

    def test_the_coder_backend_sends_no_session_mode_off_localm(self):
        from localm.plugins.coder.backends.http import HTTPBackend
        backend = HTTPBackend("http://127.0.0.1:1/v1", "m")
        backend.session_mode = "privacy"
        assert "X-Localm-Session-Mode" not in backend._headers()
        local = HTTPBackend("http://127.0.0.1:1/v1", "m", localm_server=True)
        assert "X-Localm-Session-Mode" not in local._headers()


# ------------------------------------------------------------------ #
#  Coder checkpoint privacy gate                                      #
# ------------------------------------------------------------------ #

class TestCheckpointPrivacyGate:
    def _agent(self, tmp_path, mode):
        # A REAL Agent, not an Agent.__new__(Agent) stub with hand-picked
        # attributes: a MagicMock backend plus patched make_audit_log,
        # load_memory and ProjectMap.
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
        # The checkpoint also carries the model's task list.
        agent._todos = [{"text": "secret plan step", "status": "in_progress"}]
        return agent

    def test_privacy_mode_writes_no_checkpoint(self, tmp_path, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
        agent = self._agent(tmp_path, SessionMode.PRIVACY)
        agent.save_checkpoint()
        # Nothing in the project tree, nothing under HOME. The task list rides in
        # the same file.
        assert not (tmp_path / ".localcoder" / "checkpoint.json").exists()
        assert not (tmp_path / "home" / "checkpoints").exists()

    def test_log_mode_writes_checkpoint_under_home(self, tmp_path, monkeypatch):
        # The checkpoint is session DATA: it lives under HOME, not in the project
        # tree.
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
        assert "secret plan step" in text          # the task list rides along


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
        # inside the install, surfaced on stderr, never a shared ~/.localm.
        from localm import config as cfg
        monkeypatch.delenv("LOCALM_HOME", raising=False)
        monkeypatch.delenv("LOCALM_SETUP", raising=False)
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        fake_file = tmp_path / "localm" / "config.py"
        fake_file.parent.mkdir()
        fake_file.write_text("# stub")
        monkeypatch.setattr(cfg, "__file__", str(fake_file))
        # Point Path.home() at an obvious dir, so a fallback to ~/.localm would
        # show up as this path.
        fake_home = tmp_path / "fakehome"
        monkeypatch.setattr(cfg.Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setattr(cfg, "_warned_unconfigured_home", False)
        result = cfg._detect_home()
        assert result == tmp_path / "home"              # contained fallback in the install
        assert result != fake_home / ".localm"           # never the shared user dir
        err = capsys.readouterr().err
        assert "no data directory is configured" in err
        assert str(result) in err

    def test_default_is_contained_never_user_localm_outside_a_checkout(
        self, tmp_path, monkeypatch, capsys
    ):
        # No pyproject.toml at all: the fallback still applies, since
        # config.py only gates the portable lookups on being in a checkout.
        from localm import config as cfg
        monkeypatch.delenv("LOCALM_HOME", raising=False)
        monkeypatch.delenv("LOCALM_SETUP", raising=False)
        fake_file = tmp_path / "localm" / "config.py"
        fake_file.parent.mkdir()
        fake_file.write_text("# stub")
        monkeypatch.setattr(cfg, "__file__", str(fake_file))
        fake_home = tmp_path / "fakehome"
        monkeypatch.setattr(cfg.Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setattr(cfg, "_warned_unconfigured_home", False)
        result = cfg._detect_home()
        assert result == tmp_path / "home"
        assert result != fake_home / ".localm"
        err = capsys.readouterr().err
        assert "no data directory is configured" in err
        assert str(result) in err

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

    def _checkout(self, tmp_path, monkeypatch, cfg_bytes):
        from localm import config as cfg
        monkeypatch.delenv("LOCALM_HOME", raising=False)
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "localm-home.cfg").write_bytes(cfg_bytes)
        fake_file = tmp_path / "localm" / "config.py"
        fake_file.parent.mkdir()
        fake_file.write_text("# stub")
        monkeypatch.setattr(cfg, "__file__", str(fake_file))
        return cfg

    def test_marker_with_a_non_ascii_path_in_utf8_and_a_bom(self, tmp_path, monkeypatch):
        target = tmp_path / "Données"
        cfg = self._checkout(tmp_path, monkeypatch,
                             b"\xef\xbb\xbf" + str(target).encode("utf-8") + b"\r\n")
        assert cfg._detect_home() == target

    @pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe writes the OEM code page")
    def test_marker_written_by_cmd_echo_in_the_oem_code_page(self, tmp_path, monkeypatch):
        import ctypes
        oem = f"cp{ctypes.windll.kernel32.GetOEMCP()}"
        target = tmp_path / "Données"
        try:
            raw = str(target).encode(oem) + b"\r\n"
        except UnicodeEncodeError:
            pytest.skip(f"{oem} has no e-acute")
        if raw.decode("utf-8", errors="replace") == raw.decode("utf-8", errors="ignore"):
            pytest.skip(f"{oem} bytes happen to be valid UTF-8 here")
        cfg = self._checkout(tmp_path, monkeypatch, raw)
        assert cfg._detect_home() == target

    def test_an_undecodable_marker_warns_instead_of_crashing(self, tmp_path, monkeypatch, capsys):
        cfg = self._checkout(tmp_path, monkeypatch, b"\xff\xfe\xfd not text")
        monkeypatch.setattr(cfg.sys, "platform", "linux")
        cfg._detect_home()
        assert "cannot read" in capsys.readouterr().err

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
