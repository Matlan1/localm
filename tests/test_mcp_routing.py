# SPDX-License-Identifier: AGPL-3.0-or-later
"""The MCP server's chat and coder tools: a call that names no model and needs
something the default model does not provide is answered by an installed model
that has it, a named model is always used, and with --share-loaded-models a
model another localm instance on this machine already has loaded is used
through that instance instead of being loaded a second time.

Capability answers come from the real registry readers over a real registry
shape (a recorded projector on disk for the vision model)."""

from __future__ import annotations

import http.server
import json
import os
import threading
from unittest.mock import patch

import pytest

from localm import gpu_registry
from localm.plugins.mcpserver.server import EngineCache, MCPStdioServer, build_tools


class _Engine:
    def __init__(self, name, images=False):
        self.display_name = name
        self.loaded = True
        self.supports_images = images
        self.active_requests = 0
        self.unloading = False
        self.answered = 0

    def chat_stream(self, messages, **kw):
        from localm.inference.backends.base import (UnsupportedInputError,
                                                    messages_contain_image)
        if messages_contain_image(messages) and not self.supports_images:
            raise UnsupportedInputError("cannot accept image input")
        self.answered += 1
        yield f"reply-from-{self.display_name}"

    def unload(self):
        self.loaded = False


class _LazyEngine(_Engine):
    """Built unloaded, as a real Engine is: load() brings the model up, or
    raises when it cannot, and chat_stream loads it first when it is not
    loaded. Takes a grammar, as a GGUF engine does."""

    supports_grammar = True

    def __init__(self, name, images=False, fails_to_load=False):
        super().__init__(name, images=images)
        self.loaded = False
        self.fails_to_load = fails_to_load
        self.released = False

    def validate_grammar(self, grammar, lazy=False):
        pass

    def load(self):
        if self.fails_to_load:
            raise RuntimeError(f"{self.display_name} could not be loaded")
        self.loaded = True

    def chat_stream(self, messages, **kw):
        if not self.loaded:
            self.load()
        return super().chat_stream(messages, **kw)

    def context_capacity(self):
        return 32768 if self.loaded else None

    def unload(self):
        self.released = True
        self.loaded = False


@pytest.fixture
def reg(tmp_path, monkeypatch):
    for n in ("plain", "seer", "tooly"):
        (tmp_path / n).mkdir()
        (tmp_path / n / f"{n}.gguf").write_bytes(b"GGUF" + n.encode() * 8)
    proj = tmp_path / "seer" / "seer-mmproj.gguf"
    proj.write_bytes(b"GGUF")
    registry = {
        "plain": {"path": str(tmp_path / "plain" / "plain.gguf"), "source": "local",
                  "model_type": "llm", "tool_use": False},
        "seer": {"path": str(tmp_path / "seer" / "seer.gguf"), "source": "local",
                 "model_type": "llm", "mmproj": str(proj)},
        "tooly": {"path": str(tmp_path / "tooly" / "tooly.gguf"), "source": "local",
                  "model_type": "llm", "tool_use": True},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    return registry, img, tmp_path


def _add_seer2(reg, **extra):
    """A second vision model, seer2, with its own recorded projector."""
    registry, _, tmp_path = reg
    (tmp_path / "seer2").mkdir()
    (tmp_path / "seer2" / "seer2.gguf").write_bytes(b"GGUF" + b"seer2" * 8)
    proj = tmp_path / "seer2" / "seer2-mmproj.gguf"
    proj.write_bytes(b"GGUF")
    registry["seer2"] = {"path": str(tmp_path / "seer2" / "seer2.gguf"),
                         "source": "local", "model_type": "llm", "mmproj": str(proj),
                         **extra}


def _cache(lazy=False, **kw):
    made = {}
    kind = _LazyEngine if lazy else _Engine

    def factory(name):
        return made.setdefault(name, kind(name, images=(name == "seer")))

    cache = EngineCache("plain", engine_factory=factory, **kw)
    cache.made = made
    return cache


def _call(engines, tool, args):
    # build_tools probes a custom factory's engine for embedding support, which
    # would load the default model before the call under test.
    with patch("localm.plugins.mcpserver.server._backend_can_embed", return_value=True):
        server = MCPStdioServer(build_tools(engines, enable_images=False,
                                            enable_coder=False, enable_memory=False))
    with patch.object(EngineCache, "_make_room_for", lambda self, name: None):
        resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": tool, "arguments": args}})
    return resp["result"]


class TestChatRouting:
    def test_an_image_without_a_named_model_is_read_by_the_vision_model(self, reg):
        _, img, _ = reg
        engines = _cache()
        res = _call(engines, "chat", {"prompt": "what is this?", "images": [str(img)]})
        assert res["isError"] is False
        assert res["content"][0]["text"] == "reply-from-seer"
        assert "answered by seer" in res["content"][1]["text"]
        assert "reading images" in res["content"][1]["text"]

    def test_a_named_model_is_always_used(self, reg):
        _, img, _ = reg
        engines = _cache()
        res = _call(engines, "chat", {"prompt": "what is this?", "images": [str(img)],
                                      "model": "plain"})
        assert "seer" not in engines.made
        assert res["isError"] is True

    def test_a_text_prompt_is_answered_by_the_default_model(self, reg):
        engines = _cache()
        res = _call(engines, "chat", {"prompt": "hi"})
        assert res["content"] == [{"type": "text", "text": "reply-from-plain"}]

    @pytest.mark.parametrize("ref,why", [
        ("\\\\server\\share\\x.png", "UNC"),
        ("does-not-exist.png", "not found"),
        ("data:text/plain;base64,aGk=", "data:image"),
    ])
    def test_bad_images_are_refused_before_anything_loads(self, reg, ref, why):
        engines = _cache()
        res = _call(engines, "chat", {"prompt": "x", "images": [ref]})
        assert res["isError"] is True
        assert why in res["content"][0]["text"]
        assert engines.made == {}

    def test_a_non_image_file_is_refused(self, reg):
        _, _, tmp = reg
        f = tmp / "notes.txt"
        f.write_text("hi")
        res = _call(_cache(), "chat", {"prompt": "x", "images": [str(f)]})
        assert res["isError"] is True

    def test_a_candidate_that_fails_to_load_is_not_kept_as_resident(self, reg):
        _, img, _ = reg
        _add_seer2(reg)
        loads = []

        class Counted(_LazyEngine):
            def load(self):
                loads.append(self.display_name)
                super().load()

        made = {}

        def factory(name):
            return made.setdefault(name, Counted(
                name, images=name.startswith("seer"), fails_to_load=(name == "seer")))
        engines = EngineCache("plain", engine_factory=factory)
        for _ in range(3):
            res = _call(engines, "chat", {"prompt": "what is this?", "images": [str(img)]})
            assert res["content"][0]["text"] == "reply-from-seer2"
        assert engines.resident == ["seer2"], "a model that failed to load is listed as resident"
        assert loads == ["seer", "seer2"], "the model that failed to load was loaded again"


class TestCoderRouting:
    def test_without_a_named_model_the_coder_gets_a_tool_capable_one(self, reg):
        engines = _cache()
        d = engines.route(None, [], required=("tool_use",), pinned=False)
        assert d.resolved == "tooly"

    def test_a_named_model_is_kept(self, reg):
        engines = _cache()
        d = engines.route("plain", [], required=("tool_use",), pinned=True)
        assert d.resolved == "plain"

    @staticmethod
    def _add_tooly2(reg):
        registry, _, tmp_path = reg
        (tmp_path / "tooly2").mkdir(exist_ok=True)
        (tmp_path / "tooly2" / "tooly2.gguf").write_bytes(b"GGUF" + b"tooly2" * 8)
        registry["tooly2"] = {"path": str(tmp_path / "tooly2" / "tooly2.gguf"),
                              "source": "local", "model_type": "llm", "tool_use": True}

    @staticmethod
    def _two_tool_models(reg, failing, exc):
        TestCoderRouting._add_tooly2(reg)
        made = {}

        def factory(name):
            if name in failing:
                raise exc(f"{name} is unavailable")
            return made.setdefault(name, _Engine(name))
        cache = EngineCache("plain", engine_factory=factory)
        cache.made = made
        return cache

    @staticmethod
    def _lazy_tool_models(reg, fails_to_load):
        """A cache over plain, tooly and tooly2 whose engines build unloaded;
        the ones named in *fails_to_load* build fine and fail on load()."""
        TestCoderRouting._add_tooly2(reg)
        made = {}

        def factory(name):
            return made.setdefault(
                name, _LazyEngine(name, fails_to_load=name in fails_to_load))
        cache = EngineCache("plain", engine_factory=factory)
        cache.made = made
        return cache

    @pytest.mark.parametrize("exc", [RuntimeError, ValueError])
    def test_a_candidate_that_fails_gives_way_to_the_next_one(self, reg, exc):
        from localm.plugins.mcpserver.tools.media_coder import coder_engine
        d = TestCoderRouting._two_tool_models(reg, set(), exc)
        decision = d.route(None, [], required=("tool_use",), pinned=False)
        first, second = decision.candidates
        engines = TestCoderRouting._two_tool_models(reg, {first}, exc)
        with patch.object(EngineCache, "_make_room_for", lambda self, name: None):
            engine, name, got = coder_engine(engines, decision)
        assert name == second and engine is engines.made[second]
        assert got.routed and got.resolved == second, \
            "the decision names the candidate that is used, so the reply says so"

    @pytest.mark.parametrize("exc", [RuntimeError, ValueError])
    def test_when_every_candidate_fails_the_default_is_used(self, reg, exc):
        from localm.plugins.mcpserver.tools.media_coder import coder_engine
        engines = TestCoderRouting._two_tool_models(reg, {"tooly", "tooly2"}, exc)
        decision = engines.route(None, [], required=("tool_use",), pinned=False)
        with patch.object(EngineCache, "_make_room_for", lambda self, name: None):
            engine, name, got = coder_engine(engines, decision)
        assert name == "plain" and not got.routed
        assert len(got.load_errors) == 2

    def test_a_candidate_that_fails_to_load_gives_way_to_the_next_one(self, reg):
        from localm.plugins.mcpserver.tools.media_coder import coder_engine
        engines = TestCoderRouting._lazy_tool_models(reg, {"tooly"})
        decision = engines.route(None, [], required=("tool_use",), pinned=False)
        assert decision.candidates == ("tooly", "tooly2")
        with patch.object(EngineCache, "_make_room_for", lambda self, name: None):
            engine, name, got = coder_engine(engines, decision)
        assert name == "tooly2" and engine is engines.made["tooly2"]
        assert engine.loaded, "the engine is loaded before the task gets it"
        assert got.routed and got.resolved == "tooly2"
        assert engines.resident == ["tooly2"], "a model that failed to load is not resident"
        assert engines.made["tooly"].released

    def test_when_every_candidate_fails_to_load_the_default_is_used(self, reg):
        from localm.plugins.mcpserver.tools.media_coder import coder_engine
        engines = TestCoderRouting._lazy_tool_models(reg, {"tooly", "tooly2"})
        decision = engines.route(None, [], required=("tool_use",), pinned=False)
        with patch.object(EngineCache, "_make_room_for", lambda self, name: None):
            engine, name, got = coder_engine(engines, decision)
        assert name == "plain" and engine.loaded and not got.routed
        assert [e.split(":")[0] for e in got.load_errors] == ["tooly", "tooly2"]
        assert "could not be loaded" in got.load_errors[0]
        assert engines.resident == ["plain"]

    def test_a_fallback_names_what_the_candidate_that_answers_lacks(self, reg):
        from localm.plugins.mcpserver.tools.media_coder import coder_engine
        _add_seer2(reg, tool_use=True)
        made = {}

        def factory(name):
            return made.setdefault(name, _LazyEngine(
                name, images=name.startswith("seer"), fails_to_load=(name == "seer2")))
        engines = EngineCache("plain", engine_factory=factory)
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        messages = [{"role": "user", "content": [image, {"type": "text", "text": "hi"}]}]
        decision = engines.route(None, messages, required=("tool_use", "reasoning"))
        assert decision.candidates == ("seer2", "seer")
        assert decision.unmet == ("reasoning",)
        with patch.object(EngineCache, "_make_room_for", lambda self, name: None):
            engine, name, got = coder_engine(engines, decision)
        assert name == "seer" and got.resolved == "seer"
        assert got.unmet == ("reasoning", "tool_use"), \
            "unmet names what the model that answers lacks, not what the first candidate lacked"

    def test_the_default_failing_to_load_fails_the_call(self, reg):
        from localm.plugins.mcpserver.tools.media_coder import coder_engine
        engines = TestCoderRouting._lazy_tool_models(reg, {"plain"})
        decision = engines.route("plain", [], required=("tool_use",), pinned=True)
        with patch.object(EngineCache, "_make_room_for", lambda self, name: None), \
                pytest.raises(RuntimeError, match="plain could not be loaded"):
            coder_engine(engines, decision)
        assert engines.resident == []


@pytest.fixture
def coder_project(tmp_path, monkeypatch):
    """A project directory for run_coder_task, with the coder plugin active and
    an isolated data dir."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr("localm.plugins.engine.PluginManager.is_active",
                        lambda self, name: True)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    return project


def _run_coder_task(engines, project, **args):
    """The run_coder_task tool's result for *args* in *project*."""
    with patch("localm.plugins.mcpserver.server._backend_can_embed", return_value=True):
        server = MCPStdioServer(build_tools(engines, enable_images=False,
                                            enable_memory=False))
    call = {"task": "do the thing", "cwd": str(project), **args}
    with patch.object(EngineCache, "_make_room_for", lambda self, name: None), \
            patch("localm.plugins.coder.agent.ProjectMap") as project_map, \
            patch("localm.plugins.coder.agent.make_audit_log"), \
            patch("localm.plugins.coder.agent.load_memory", return_value=""):
        project_map.build.return_value.file_count.return_value = 0
        resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "run_coder_task", "arguments": call}})
    return resp["result"]


class TestCoderTaskRuns:
    def test_a_candidate_that_fails_to_load_is_skipped_for_the_run(
            self, reg, coder_project):
        engines = TestCoderRouting._lazy_tool_models(reg, {"tooly"})
        res = _run_coder_task(engines, coder_project)
        text = res["content"][0]["text"]
        assert res["isError"] is False, text
        assert "reply-from-tooly2" in text
        assert "[answered by tooly2: plain lacks structured tool calls]" in text

    @staticmethod
    def _small_default(reg):
        """plain emits tool calls but was trained on 4096 tokens; tooly was
        trained on 131072."""
        registry, _, _ = reg
        registry["plain"].update(tool_use=True, context_length=4096)
        registry["tooly"]["context_length"] = 131072

    def test_a_task_longer_than_the_default_models_window_goes_to_a_roomier_one(
            self, reg, coder_project):
        TestCoderTaskRuns._small_default(reg)
        engines = _cache(lazy=True)
        task = "Refactor the parser as this log shows. " + "trace line " * 2400
        res = _run_coder_task(engines, coder_project, task=task)
        text = res["content"][0]["text"]
        assert "tooly" in engines.made and "plain" not in engines.made, \
            "the task was run on the model whose window is too small for it"
        assert res["isError"] is False, text
        assert "reply-from-tooly" in text
        assert "[answered by tooly: plain lacks a longer conversation]" in text

    def test_a_short_task_stays_on_the_default_model(self, reg, coder_project):
        TestCoderTaskRuns._small_default(reg)
        engines = _cache(lazy=True)
        res = _run_coder_task(engines, coder_project)
        text = res["content"][0]["text"]
        assert list(engines.made) == ["plain"]
        assert res["isError"] is False, text
        assert "reply-from-plain" in text and "answered by" not in text


# --------------------------------------------------------------------------- #
#  --share-loaded-models                                                       #
# --------------------------------------------------------------------------- #

class _Peer:
    """A real HTTP server standing in for another localm instance with a
    model loaded: GET /v1/models and POST /v1/chat/completions, streamed or
    not, with no API key (open mode), listening on *host*. *replies* are its
    answers in order, the last one repeating. Every POST after the first
    *answers* fails as *then*: "drop" closes the connection without a
    response, a number answers with that HTTP status."""

    def __init__(self, replies=("reply-from-peer",), answers=None, then="drop",
                 host="127.0.0.1", port=0):
        outer = self
        self.bodies = []
        self.auth = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, content_type, data):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.auth.append(self.headers.get("Authorization"))
                self._send(200, "application/json", b'{"object": "list", "data": []}')

            def do_POST(self):
                outer.auth.append(self.headers.get("Authorization"))
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                outer.bodies.append(body)
                if answers is not None and len(outer.bodies) > answers:
                    if then == "drop":
                        self.close_connection = True
                        return
                    self._send(then, "application/json",
                               json.dumps({"detail": f"peer failure {then}"}).encode())
                    return
                reply = replies[min(len(outer.bodies), len(replies)) - 1]
                if body.get("stream"):
                    chunk = {"model": body["model"],
                             "choices": [{"delta": {"content": reply}}]}
                    data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                    self._send(200, "text/event-stream", data)
                else:
                    message = {"role": "assistant", "content": reply}
                    data = json.dumps({"model": body["model"],
                                       "choices": [{"message": message}]}).encode()
                    self._send(200, "application/json", data)

        self.srv = http.server.ThreadingHTTPServer((host, port), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def _advertise(reg, tmp_path, monkeypatch, p, *, instance_id="other", host="127.0.0.1"):
    """A machine-wide registry entry for instance *instance_id* at *host* and
    *p*'s port, with plain's model file loaded."""
    registry, _, _ = reg
    d = tmp_path / "gpu"
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: True)
    monkeypatch.setattr(gpu_registry, "_try_whoami", lambda scheme, port, iid, timeout: True)
    plain = os.path.realpath(registry["plain"]["path"])
    gpu_registry.write_entry(
        d, instance_id=instance_id, pid=os.getpid() + 1, port=p.port, host=host,
        scheme="http", model="their-plain", vram_estimate_bytes=None, gpu_index=0,
        coordination_token="t",
        models=[{"name": "their-plain", "path": plain,
                 "size": os.path.getsize(plain), "sha256": None}])


@pytest.fixture
def peer(reg, tmp_path, monkeypatch):
    p = _Peer()
    _advertise(reg, tmp_path, monkeypatch, p)
    return p


class TestShareLoadedModels:
    def test_the_peers_copy_answers_and_nothing_loads_here(self, peer):
        engines = _cache(share_loaded=True)
        res = _call(engines, "chat", {"prompt": "hi"})
        assert res["content"][0]["text"] == "reply-from-peer"
        assert engines.made == {}, "no second copy loaded"
        assert peer.bodies[-1]["model"] == "their-plain"

    def test_without_the_flag_it_loads_here(self, peer):
        engines = _cache()
        res = _call(engines, "chat", {"prompt": "hi"})
        assert res["content"][0]["text"] == "reply-from-plain"
        assert peer.bodies == []

    def test_a_peer_that_stops_answering_is_replaced_by_a_local_load(self, peer):
        engines = _cache(share_loaded=True)
        _call(engines, "chat", {"prompt": "hi"})
        peer.srv.shutdown()
        peer.srv.server_close()
        res = _call(engines, "chat", {"prompt": "again"})
        assert res["content"][0]["text"] == "reply-from-plain"
        assert "plain" in engines.made


def _tool_call(name, **args):
    return "<tool_call>" + json.dumps({"name": name, "args": args}) + "</tool_call>"


class TestCoderTaskOnAPeer:
    def test_a_peer_that_stopped_answering_is_not_handed_to_the_next_task(self, peer):
        from localm.plugins.mcpserver.tools.media_coder import coder_engine
        engines = _cache(lazy=True, share_loaded=True)
        decision = engines.route("plain", [], required=("tool_use",), pinned=True)
        with patch.object(EngineCache, "_make_room_for", lambda self, name: None):
            first, _, _ = coder_engine(engines, decision)
            assert engines.is_peer(first)
            peer.srv.shutdown()
            peer.srv.server_close()
            second, name, _ = coder_engine(engines, decision)
        assert not engines.is_peer(second), "the instance that stopped answering was used again"
        assert name == "plain" and second is engines.made["plain"] and second.loaded
        assert not engines.is_peer(first)

    def test_a_peer_that_still_answers_is_kept(self, peer):
        from localm.plugins.mcpserver.tools.media_coder import coder_engine
        engines = _cache(lazy=True, share_loaded=True)
        decision = engines.route("plain", [], required=("tool_use",), pinned=True)
        with patch.object(EngineCache, "_make_room_for", lambda self, name: None):
            first, _, _ = coder_engine(engines, decision)
            second, _, _ = coder_engine(engines, decision)
        assert second is first and engines.is_peer(second)
        assert engines.made == {}

    @pytest.mark.parametrize("then", ["drop", 401])
    def test_a_task_moves_to_a_copy_loaded_here_when_the_peer_fails_mid_run(
            self, reg, tmp_path, monkeypatch, coder_project, then):
        p = _Peer(replies=[_tool_call("read_file", path="a.py")], answers=1, then=then)
        _advertise(reg, tmp_path, monkeypatch, p)
        engines = _cache(lazy=True, share_loaded=True)
        res = _run_coder_task(engines, coder_project, model="plain")
        text = res["content"][0]["text"]
        assert res["isError"] is False, text
        assert "reply-from-plain" in text
        assert len(p.bodies) == 2, "one answer from the peer, then the call that failed there"
        local = engines.made["plain"]
        assert local.answered >= 1
        assert local.active_requests == 0, "the copy loaded here is unpinned when the run ends"
        assert engines.resident == ["plain"]

    def test_an_error_the_peer_answers_with_does_not_load_a_copy_here(
            self, reg, tmp_path, monkeypatch, coder_project):
        p = _Peer(answers=0, then=503)
        _advertise(reg, tmp_path, monkeypatch, p)
        engines = _cache(lazy=True, share_loaded=True)
        res = _run_coder_task(engines, coder_project, model="plain")
        assert engines.made == {}, "an answer from a live peer is not a reason to load here"
        assert res["isError"] is True
        assert "peer failure 503" in res["content"][0]["text"]

    @pytest.mark.parametrize("stop", ["cancel", "release"])
    def test_nothing_is_loaded_here_once_the_run_is_over(
            self, reg, tmp_path, monkeypatch, stop):
        import requests

        from localm.plugins.coder.backends.http import HTTPBackend
        from localm.plugins.mcpserver.tools.media_coder import PeerCoderBackend
        p = _Peer(answers=0, then="drop")
        _advertise(reg, tmp_path, monkeypatch, p)
        engines = _cache(lazy=True, share_loaded=True)
        peer_engine = engines.get_chat("plain")
        assert engines.is_peer(peer_engine)
        backend = PeerCoderBackend(engines, "plain", peer_engine, HTTPBackend(
            peer_engine._base, model="their-plain", localm_server=True))
        getattr(backend, stop)()
        with pytest.raises(requests.ConnectionError):
            backend.chat([{"role": "user", "content": "hi"}])
        assert engines.made == {} and backend.local_engine is None

    def test_a_stream_that_cannot_reach_the_peer_is_answered_here(
            self, reg, tmp_path, monkeypatch):
        from localm.plugins.coder.backends.http import HTTPBackend
        from localm.plugins.mcpserver.tools.media_coder import PeerCoderBackend
        p = _Peer(answers=0, then="drop")
        _advertise(reg, tmp_path, monkeypatch, p)
        engines = _cache(lazy=True, share_loaded=True)
        peer_engine = engines.get_chat("plain")
        backend = PeerCoderBackend(engines, "plain", peer_engine, HTTPBackend(
            peer_engine._base, model="their-plain", localm_server=True))
        with patch.object(EngineCache, "_make_room_for", lambda self, name: None):
            pieces = list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        assert pieces == ["reply-from-plain"]
        assert backend.local_engine is engines.made["plain"]
        assert backend.local_engine.active_requests == 1
        backend.release()
        assert backend.local_engine.active_requests == 0
        assert not engines.is_peer(peer_engine)


def _claimed_as_this_install(reg, tmp_path, monkeypatch, advertised_port, served_port,
                             advertised_host="127.0.0.1"):
    """A machine-wide registry entry naming this install's instance "mine" at
    *advertised_host*:*advertised_port* while this install's own instance file
    says it serves 127.0.0.1:*served_port*. The owner key is OWNER-KEY."""
    registry, _, _ = reg
    d = tmp_path / "gpu"
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: True)
    monkeypatch.setattr(gpu_registry, "_try_whoami", lambda scheme, port, iid, timeout: True)
    plain = os.path.realpath(registry["plain"]["path"])
    gpu_registry.write_entry(
        d, instance_id="mine", pid=os.getpid() + 1, port=advertised_port,
        host=advertised_host, scheme="http", model="their-plain",
        vram_estimate_bytes=None, gpu_index=0, coordination_token="t",
        models=[{"name": "their-plain", "path": plain,
                 "size": os.path.getsize(plain), "sha256": None}])
    monkeypatch.setattr("localm.instances.list_entries", lambda home: [
        {"instance_id": "mine", "port": served_port, "host": "127.0.0.1",
         "scheme": "http", "token": "instance-token"}])
    monkeypatch.setattr("localm.auth.get_api_key", lambda: "OWNER-KEY")


class TestThisInstallsCredential:
    def test_it_never_reaches_a_port_this_install_does_not_serve(
            self, reg, tmp_path, monkeypatch):
        genuine, forged = _Peer(), _Peer()
        _claimed_as_this_install(reg, tmp_path, monkeypatch,
                                 advertised_port=forged.port, served_port=genuine.port)
        engines = _cache(share_loaded=True)
        res = _call(engines, "chat", {"prompt": "hi"})
        assert forged.auth == [], "nothing, credential or not, is sent to the claimed port"
        assert genuine.auth == []
        assert res["content"][0]["text"] == "reply-from-plain"
        assert "plain" in engines.made

    def test_this_installs_own_instance_is_used_with_its_credential(
            self, reg, tmp_path, monkeypatch):
        genuine = _Peer()
        _claimed_as_this_install(reg, tmp_path, monkeypatch,
                                 advertised_port=genuine.port, served_port=genuine.port)
        engines = _cache(share_loaded=True)
        res = _call(engines, "chat", {"prompt": "hi"})
        assert res["content"][0]["text"] == "reply-from-peer"
        assert genuine.auth and set(genuine.auth) == {"Bearer OWNER-KEY"}
        assert engines.made == {}

    def test_it_never_reaches_another_loopback_host_at_this_installs_port(
            self, reg, tmp_path, monkeypatch):
        genuine = _Peer()
        try:
            forged = _Peer(host="127.0.0.2", port=genuine.port)
        except OSError as e:
            pytest.skip(f"cannot listen on 127.0.0.2 on this machine: {e}")
        _claimed_as_this_install(reg, tmp_path, monkeypatch,
                                 advertised_port=genuine.port, served_port=genuine.port,
                                 advertised_host="127.0.0.2")
        engines = _cache(share_loaded=True)
        res = _call(engines, "chat", {"prompt": "hi"})
        assert forged.auth == [], \
            "a request went to the host the machine-wide entry claimed, not the instance file's"
        assert genuine.auth and set(genuine.auth) == {"Bearer OWNER-KEY"}
        assert res["content"][0]["text"] == "reply-from-peer"
        assert engines.made == {}

