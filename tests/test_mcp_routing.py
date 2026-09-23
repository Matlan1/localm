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


def _cache(**kw):
    made = {}

    def factory(name):
        return made.setdefault(name, _Engine(name, images=(name == "seer")))

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
    def _two_tool_models(reg, failing, exc):
        registry, _, tmp_path = reg
        (tmp_path / "tooly2").mkdir(exist_ok=True)
        (tmp_path / "tooly2" / "tooly2.gguf").write_bytes(b"GGUF" + b"tooly2" * 8)
        registry["tooly2"] = {"path": str(tmp_path / "tooly2" / "tooly2.gguf"),
                              "source": "local", "model_type": "llm", "tool_use": True}
        made = {}

        def factory(name):
            if name in failing:
                raise exc(f"{name} is unavailable")
            return made.setdefault(name, _Engine(name))
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


# --------------------------------------------------------------------------- #
#  --share-loaded-models                                                       #
# --------------------------------------------------------------------------- #

class _Peer:
    """A real HTTP server standing in for another localm instance with a
    model loaded: GET /v1/models and a streaming POST /v1/chat/completions,
    with no API key (open mode)."""

    def __init__(self):
        outer = self
        self.bodies = []
        self.auth = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.auth.append(self.headers.get("Authorization"))
                data = b'{"object": "list", "data": []}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                outer.auth.append(self.headers.get("Authorization"))
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                outer.bodies.append(body)
                chunk = {"model": body["model"],
                         "choices": [{"delta": {"content": "reply-from-peer"}}]}
                data = (f"data: {json.dumps(chunk)}\n\n" + "data: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


@pytest.fixture
def peer(reg, tmp_path, monkeypatch):
    registry, _, _ = reg
    d = tmp_path / "gpu"
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: True)
    monkeypatch.setattr(gpu_registry, "_try_whoami", lambda scheme, port, iid, timeout: True)
    p = _Peer()
    plain = os.path.realpath(registry["plain"]["path"])
    gpu_registry.write_entry(
        d, instance_id="other", pid=os.getpid() + 1, port=p.port, host="127.0.0.1",
        scheme="http", model="their-plain", vram_estimate_bytes=None, gpu_index=0,
        coordination_token="t",
        models=[{"name": "their-plain", "path": plain,
                 "size": os.path.getsize(plain), "sha256": None}])
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


def _claimed_as_this_install(reg, tmp_path, monkeypatch, advertised_port, served_port):
    """A machine-wide registry entry naming this install's instance "mine" at
    *advertised_port* while this install's own instance file says it serves
    *served_port*. The owner key is OWNER-KEY."""
    registry, _, _ = reg
    d = tmp_path / "gpu"
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: True)
    monkeypatch.setattr(gpu_registry, "_try_whoami", lambda scheme, port, iid, timeout: True)
    plain = os.path.realpath(registry["plain"]["path"])
    gpu_registry.write_entry(
        d, instance_id="mine", pid=os.getpid() + 1, port=advertised_port,
        host="127.0.0.1", scheme="http", model="their-plain", vram_estimate_bytes=None,
        gpu_index=0, coordination_token="t",
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

