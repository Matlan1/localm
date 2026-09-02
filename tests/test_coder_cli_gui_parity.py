# SPDX-License-Identifier: AGPL-3.0-or-later
"""The six coder options that exist on the CLI each have a web form, per the
standing rule that anything available in the CLI must be available in SOME form
in GUI mode.

  --estimate       POST /api/coder/sessions/{id}/estimate
  --patch-mode     patch_mode on create + GET .../patch and .../patch/download
  --native-tools   native_tools on create, with the EFFECTIVE value reported
  --output-format  GET .../result (the CLI's json payload, no SSE needed)
  --episodes       GET /api/coder/episodes?cwd=
  --until          unified onto the existing verify oracle; its retry cap
                   (the CLI's --goal-max-iters) is settable as
                   verify_max_retries

The two properties these tests pin, because both fail SILENTLY: reading a patch
must not consume it, and an option the server cannot honour must be reported as
not applied rather than echoed back as if it were.
"""

import inspect
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Non-routable RFC5737 documentation address: guaranteed never to route anywhere,
# so even a total guard failure cannot dial a real host.
_UNC = "\\\\192.0.2.1\\share"
_UNC_FWD = "//192.0.2.1/share"
_DEVICE = "\\\\.\\PhysicalDrive0"


# --------------------------------------------------------------------------- #
#  Harness                                                                     #
# --------------------------------------------------------------------------- #

class _StubBackend:
    """Enough backend for a session. ``chat`` returns a canned plan so the
    estimate route can be driven without a model."""
    model_id = "stub-model"
    native_tools = False
    supports_native_tools = True
    supports_grammar = False

    def __init__(self):
        self.calls: list = []
        self.last_usage = {"prompt_tokens": 11, "total_tokens": 33}

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return "PLAN: read a.py, change b.py. About 3 turns."


def _coder_app(tmp_path, monkeypatch, *, api_key):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", api_key)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


def _owner(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    return app, proj, {"Authorization": "Bearer ownersecret"}


def _stub_session_backend(app, sid):
    """Swap the session's real HTTPBackend (which would dial 127.0.0.1:9) for a
    stub, so an estimate can be driven without a model or a server."""
    sess = app.state.coder_sessions.get(sid)
    stub = _StubBackend()
    sess.agent.backend = stub
    return sess, stub


# --------------------------------------------------------------------------- #
#  --estimate                                                                  #
# --------------------------------------------------------------------------- #

def test_estimate_plans_without_touching_the_conversation(tmp_path, monkeypatch):
    """One planning turn, zero execution, and the session is left exactly as it
    was: an estimate is a question ABOUT a task, not a turn OF one. In the CLI
    the process exits straight afterwards so a polluted history costs nothing;
    a GUI session lives on and would carry it into every later turn."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions", headers=owner,
                          json={"cwd": str(proj)}).json()["id"]
        sess, stub = _stub_session_backend(app, sid)
        before_msgs = list(sess.agent._messages)
        before_turns = sess.agent.turns

        r = client.post("/api/coder/sessions/%s/estimate" % sid, headers=owner,
                        json={"text": "add a --foo flag"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["estimate"].startswith("PLAN:")
        assert body["prompt_tokens"] == 11 and body["total_tokens"] == 33

        # The prompt carried the no-tools instruction and the task.
        sent = stub.calls[0]
        assert sent[0]["role"] == "system"
        assert "ESTIMATE ONLY" in sent[1]["content"]
        assert "add a --foo flag" in sent[1]["content"]

        # Nothing was appended to the conversation and no turn was billed.
        assert sess.agent._messages == before_msgs
        assert sess.agent.turns == before_turns


def test_estimate_reaches_the_feed_so_every_tab_sees_it(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions", headers=owner,
                          json={"cwd": str(proj)}).json()["id"]
        sess, _ = _stub_session_backend(app, sid)
        client.post("/api/coder/sessions/%s/estimate" % sid, headers=owner,
                    json={"text": "add a --foo flag"})
        ests = [e for e in sess.history if e.get("type") == "estimate"]
        assert len(ests) == 1
        assert ests[0]["task"] == "add a --foo flag"
        assert ests[0]["text"].startswith("PLAN:")


def test_estimate_refuses_an_empty_task_and_a_busy_session(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions", headers=owner,
                          json={"cwd": str(proj)}).json()["id"]
        sess, _ = _stub_session_backend(app, sid)
        assert client.post("/api/coder/sessions/%s/estimate" % sid, headers=owner,
                           json={"text": "   "}).status_code == 400
        sess.busy = True
        busy = client.post("/api/coder/sessions/%s/estimate" % sid, headers=owner,
                           json={"text": "real task"})
        assert busy.status_code == 409
        # "busy" and "closed" are both 409 but carry different messages: come
        # back in a moment, versus this session is gone.
        assert "busy" in busy.json()["detail"]
        sess.busy = False
        sess.closed = True
        r = client.post("/api/coder/sessions/%s/estimate" % sid, headers=owner,
                        json={"text": "real task"})
        assert r.status_code == 409 and "closed" in r.json()["detail"]


def test_estimate_claims_the_session_and_releases_it(tmp_path, monkeypatch):
    """An estimate is a SECOND trigger for a backend that had exactly one. It
    takes `busy` under the session's own lock for the duration - so a task
    cannot start underneath it and clobber the token numbers it is about to
    read - and gives it back afterwards, including when the model call raises."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions", headers=owner,
                          json={"cwd": str(proj)}).json()["id"]
        sess, stub = _stub_session_backend(app, sid)
        seen = []

        def _chat(messages, **kw):
            seen.append(sess.busy)
            return "PLAN: ok"

        stub.chat = _chat
        client.post("/api/coder/sessions/%s/estimate" % sid, headers=owner,
                    json={"text": "plan it"})
        assert seen == [True], "the session must be claimed while the turn runs"
        assert sess.busy is False, "and released afterwards"

        def _boom(messages, **kw):
            raise RuntimeError("model exploded")

        stub.chat = _boom
        r = client.post("/api/coder/sessions/%s/estimate" % sid, headers=owner,
                        json={"text": "plan it"})
        assert r.status_code == 502
        assert sess.busy is False, "a failed estimate must not wedge the session"


def test_a_message_typed_during_an_estimate_is_not_stranded(tmp_path, monkeypatch):
    """While the estimate holds `busy`, send_message QUEUES rather than starts,
    and the only code that ever re-runs a queued message is the task thread's
    own finally - which never runs here. Without the drain the message would sit
    unsent until the user typed again, with nothing on screen saying so."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions", headers=owner,
                          json={"cwd": str(proj)}).json()["id"]
        sess, stub = _stub_session_backend(app, sid)
        started = []
        sess.agent.run_task = lambda text: started.append(text) or "done"

        def _chat(messages, **kw):
            # Arrives mid-estimate, exactly as a user typing would.
            assert sess.send_message("do the thing") == "queued"
            return "PLAN: ok"

        stub.chat = _chat
        client.post("/api/coder/sessions/%s/estimate" % sid, headers=owner,
                    json={"text": "plan it"})
        if sess._thread is not None:
            sess._thread.join(timeout=10)
        assert started == ["do the thing"], (
            "the queued message must be run as a follow-up, not stranded")


# --------------------------------------------------------------------------- #
#  --patch-mode                                                                #
# --------------------------------------------------------------------------- #

def test_patch_mode_is_wired_from_the_request_to_the_agent(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "patch_mode": True})
        assert r.status_code == 200, r.text
        assert r.json()["patch_mode"] is True
        sess = app.state.coder_sessions.get(r.json()["id"])
        assert sess.patch_mode is True
        assert sess.agent.patch_mode is True

        # And the default is still off.
        d = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj)})
        assert d.json()["patch_mode"] is False
        assert app.state.coder_sessions.get(d.json()["id"]).agent.patch_mode is False


def test_reading_the_patch_does_not_consume_it(tmp_path, monkeypatch):
    """The defect this pins is silent: ``flush_patch()`` CLEARS the buffer, so a
    read built on it looks perfect the first time and returns an empty patch to
    every reader after - a reloaded tab, a retry, the download after the
    preview. Assert on the SECOND read, and on the buffer itself."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        created = client.post("/api/coder/sessions", headers=owner,
                              json={"cwd": str(proj), "patch_mode": True}).json()
        sid = created["id"]
        # has_patch is about CONTENT, not about the mode: a patch-mode session
        # that has captured nothing reports false.
        assert created["has_patch"] is False
        sess = app.state.coder_sessions.get(sid)
        sess.agent._patch_chunks.append(
            "--- a/x.py\n+++ b/x.py\n+print(1)\n")
        assert client.get("/api/coder/sessions", headers=owner).json(
            )["sessions"][0]["has_patch"] is True

        first = client.get("/api/coder/sessions/%s/patch" % sid, headers=owner)
        assert first.status_code == 200
        assert "print(1)" in first.json()["patch"]
        assert first.json()["empty"] is False

        second = client.get("/api/coder/sessions/%s/patch" % sid, headers=owner)
        assert second.json()["patch"] == first.json()["patch"], (
            "the patch was consumed by reading it")
        assert sess.agent._patch_chunks, "the agent's patch buffer was drained"

        # The download is the web form of --patch-mode FILE and carries the same
        # bytes, checked AFTER two reads so a drain anywhere shows up here.
        dl = client.get("/api/coder/sessions/%s/patch/download" % sid, headers=owner)
        assert dl.status_code == 200
        assert "print(1)" in dl.text
        assert ".patch" in dl.headers["content-disposition"]


def test_patch_routes_refuse_a_session_that_is_not_in_patch_mode(tmp_path,
                                                                monkeypatch):
    """An empty diff from a normal session would mean "everything was written to
    disk", the opposite of what this endpoint reports - so it must not answer 200
    with an empty body."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions", headers=owner,
                          json={"cwd": str(proj)}).json()["id"]
        assert client.get("/api/coder/sessions/%s/patch" % sid,
                          headers=owner).status_code == 409
        assert client.get("/api/coder/sessions/%s/patch/download" % sid,
                          headers=owner).status_code == 409


def test_patch_download_404s_when_nothing_was_captured(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions", headers=owner,
                          json={"cwd": str(proj),
                                "patch_mode": True}).json()["id"]
        assert client.get("/api/coder/sessions/%s/patch/download" % sid,
                          headers=owner).status_code == 404


# --------------------------------------------------------------------------- #
#  --native-tools                                                              #
# --------------------------------------------------------------------------- #

def test_native_tools_is_reported_as_not_applied_against_localms_own_server(
        tmp_path, monkeypatch):
    """localm's /v1/chat/completions declares no tools/tool_choice, so the
    fields are dropped and the run proceeds exactly as if the option had never
    been passed. Nothing breaks, so silence is the problem. The
    response must say it did not take effect."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "native_tools": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["native_tools_requested"] is True
        assert body["native_tools"] is False, (
            "the EFFECTIVE value must not echo the request")
        assert any("native_tools was not applied" in n for n in body["notes"])

        # The request really did reach the backend.
        sess = app.state.coder_sessions.get(body["id"])
        assert sess.agent.backend.native_tools is True
        assert sess.agent.backend.supports_native_tools is False


def test_not_asking_for_native_tools_produces_no_note(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        body = client.post("/api/coder/sessions", headers=owner,
                           json={"cwd": str(proj)}).json()
        assert body["native_tools"] is False
        assert body["native_tools_requested"] is False
        assert body["notes"] == []


def test_localm_chat_request_really_has_no_tools_field():
    """The premise the whole native_tools decision rests on. If localm ever DOES
    implement the tools API, this test fails and the "not applied" note above
    becomes a lie that needs removing."""
    from localm.inference.protocol import ChatRequest
    req = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}],
                      tools=[{"type": "function"}], tool_choice="auto")
    assert not hasattr(req, "tools")
    assert "tools" not in req.model_dump()


def test_supports_native_tools_is_true_for_a_real_openai_style_endpoint():
    """The capability answer must not be a blanket False, or the warning would
    fire for the very backends the option exists for."""
    from localm.plugins.coder.backends.http import HTTPBackend
    remote = HTTPBackend("https://api.openai.com/v1", "gpt-4o",
                         api_key="k", native_tools=True)
    assert remote.supports_native_tools is True
    local = HTTPBackend("http://127.0.0.1:9/v1", "m", api_key="k",
                        localm_server=True, native_tools=True)
    assert local.supports_native_tools is False


# --------------------------------------------------------------------------- #
#  --output-format json                                                        #
# --------------------------------------------------------------------------- #

def test_result_route_carries_the_clis_json_payload(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions", headers=owner,
                          json={"cwd": str(proj)}).json()["id"]
        # Before any task there is no result: a 404 rather than an empty body.
        assert client.get("/api/coder/sessions/%s/result" % sid,
                          headers=owner).status_code == 404

        sess = app.state.coder_sessions.get(sid)
        sess.last_result = {"text": "done", "response": "done", "ok": True,
                            "verify_state": "passed", "turns": 4,
                            "total_tokens": 900, "changed_files": ["a.py"]}
        got = client.get("/api/coder/sessions/%s/result" % sid,
                         headers=owner).json()
        # The CLI's --output-format json keys, all present.
        for key in ("response", "turns", "total_tokens"):
            assert key in got, key
        assert got["ok"] is True and got["response"] == "done"
        assert got["verify_state"] == "passed"


def test_a_finished_task_latches_its_result(tmp_path, monkeypatch):
    """The result is recorded from the SAME payload the final event carries, so
    a polling client and an SSE client cannot disagree."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions", headers=owner,
                          json={"cwd": str(proj)}).json()["id"]
        sess = app.state.coder_sessions.get(sid)

        def _fake_run_task(text):
            sess.agent._turns = 2
            return "all done"

        sess.agent.run_task = _fake_run_task
        client.post("/api/coder/sessions/%s/message" % sid, headers=owner,
                    json={"text": "do it"})
        if sess._thread is not None:
            sess._thread.join(timeout=10)

        final = [e for e in sess.history if e.get("type") == "final"]
        assert final, "the task never finished"
        got = client.get("/api/coder/sessions/%s/result" % sid,
                         headers=owner).json()
        assert got["response"] == "all done"
        assert got["turns"] == final[0]["turns"]
        assert got["ok"] == final[0]["ok"]


# --------------------------------------------------------------------------- #
#  --episodes                                                                  #
# --------------------------------------------------------------------------- #

def _seed_episode(proj, lesson="always run the tests"):
    from localm.plugins.coder.episodes import Episode, EpisodeStore
    store = EpisodeStore(Path(proj))
    return store.add(Episode(task="fix the parser", outcome="ok",
                             summary="parser fixed", lesson=lesson, turns=3))


def test_episodes_lists_the_projects_stored_lessons(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        ep = _seed_episode(proj)
        r = client.get("/api/coder/episodes", headers=owner,
                       params={"cwd": str(proj)})
        assert r.status_code == 200, r.text
        rows = r.json()["episodes"]
        assert len(rows) == 1
        # The id is what --forget-episode / --restore-episode address, so it has
        # to travel.
        assert rows[0]["id"] == ep.id
        assert rows[0]["lesson"] == "always run the tests"
        assert rows[0]["outcome"] == "ok"


def test_episodes_is_owner_only_and_validates_cwd(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    from localm import auth
    with TestClient(app) as client:
        _seed_episode(proj, lesson="the owner's private lesson")
        scoped = auth.create_key("phone", ["coder"])
        sh = {"Authorization": "Bearer %s" % scoped["key"]}
        # A scoped key is never shown the owner's lessons: an empty list, not a 403.
        assert client.get("/api/coder/episodes", headers=sh,
                          params={"cwd": str(proj)}).json()["episodes"] == []
        assert client.get("/api/coder/episodes", headers=owner).status_code == 400
        # A directory that does not exist is NOT an error: lessons live under the
        # localm data dir keyed by the resolved project path, so a project you
        # moved or deleted still has an entry.
        gone = client.get("/api/coder/episodes", headers=owner,
                          params={"cwd": str(tmp_path / "nope")})
        assert gone.status_code == 200
        assert gone.json()["episodes"] == []


@pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE])
def test_episodes_refuses_unc_and_device_cwd(tmp_path, monkeypatch, bad):
    """Same unconditional lexical refusal every other cwd-taking coder route
    carries: a UNC string reaching the filesystem is the SMB dial (and the
    net-NTLMv2 leak), which happens before any status code is chosen.

    The spy covers ``resolve`` AND ``is_dir``, not merely whichever one this
    route calls today. A spy pointed at a single method goes structurally DEAD
    the moment the code reaches for the other one, and a dead fault injector is
    indistinguishable from a guard that correctly found nothing to refuse, since
    both produce a clean green."""
    real = {"resolve": Path.resolve, "is_dir": Path.is_dir}

    def make_spy(name):
        def spy(self, *a, **kw):
            s = str(self)
            if s[:2] in ("\\\\", "//", "\\/", "/\\"):
                raise AssertionError(
                    "Path.%s() reached the filesystem with %r" % (name, s))
            return real[name](self, *a, **kw)
        return spy

    for name in real:
        monkeypatch.setattr(Path, name, make_spy(name))
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/api/coder/episodes", headers=owner, params={"cwd": bad})
        assert r.status_code == 400
        assert "UNC or device" in r.json()["detail"]


# --------------------------------------------------------------------------- #
#  --until                                                                     #
# --------------------------------------------------------------------------- #

def test_verify_max_retries_is_settable_from_the_web(tmp_path, monkeypatch):
    """--goal-max-iters needs a web equivalent, or the GUI gets the Agent's
    hardcoded default with no way to change it. Bounded 1..50, matching the
    CLI's own IntRange, so a request cannot pin the shared engine on an endless
    loop."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "verify": "pytest -x",
                              "verify_max_retries": 7})
        assert r.status_code == 200, r.text
        assert r.json()["verify_max_retries"] == 7
        assert r.json()["verify"] == "pytest -x"
        assert app.state.coder_sessions.get(
            r.json()["id"]).agent.verify_max_retries == 7

        # Omitted keeps the Agent's own default rather than duplicating it here.
        d = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "auto_verify": False})
        from localm.plugins.coder.agent import Agent
        default = inspect.signature(Agent.__init__).parameters[
            "verify_max_retries"].default
        assert d.json()["verify_max_retries"] == default


@pytest.mark.parametrize("bad", [0, -1, 51])
def test_verify_max_retries_is_bounded(tmp_path, monkeypatch, bad):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "verify_max_retries": bad})
        assert r.status_code == 422


def test_the_web_oracle_is_the_same_one_until_uses():
    """--until is unified rather than rebuilt: the agent's pre-done gate runs
    verify.py's own primitives, so a web session and ``--until`` judge a task by
    the same exit code and feed back the same anti-gaming text."""
    from localm.plugins.coder import verify as v
    from localm.plugins.coder.cli import goal as g
    assert g._run_verify is v.run_verify
    assert g._goal_feedback is v.verify_feedback
    assert "Do not modify the check itself" in v.verify_feedback("pytest", 1,
                                                                "boom")

# --------------------------------------------------------------------------- #
#  The reverse direction: a GUI capability the CLI could not reach at all.     #
# --------------------------------------------------------------------------- #

def test_terminal_coder_honours_the_browser_setting(monkeypatch):
    """browser_enabled had no effect on a terminal session.

    Agent(browser_enabled=...) defaults to False and the CLI never passed it, so
    `localm coder` could never use the browser tools the GUI coder gets, however
    the setting was set.
    """
    from localm.plugins.coder.cli import _main

    monkeypatch.setattr("localm.config.load_config", lambda: {"browser_enabled": True})
    assert _main._browser_enabled() is True
    monkeypatch.setattr("localm.config.load_config", lambda: {"browser_enabled": False})
    assert _main._browser_enabled() is False
    # An unreadable config answers False, matching the GUI gate.
    def _boom():
        raise OSError("no config")
    monkeypatch.setattr("localm.config.load_config", _boom)
    assert _main._browser_enabled() is False


def test_terminal_coder_actually_passes_browser_enabled_to_the_agent():
    """The helper above is only worth anything if the Agent is built with it."""
    from localm.plugins.coder.cli import _main

    # click wraps the command, so the function is on .callback.    src = inspect.getsource(_main.main.callback)
    assert "browser_enabled=_browser_enabled()" in src, (
        "the terminal coder builds its Agent without browser_enabled, "
        "so the setting cannot reach it")
