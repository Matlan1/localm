# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capability routing reached from each client surface, against a REAL uvicorn
server: the coder's HTTP backend, the terminal chat's attach engine, and the
Knowledge plugin's image description. Each asserts on the engine that actually
generated the reply, never on a status code alone.

The loaded model ("plain") cannot emit structured tool calls or read images;
"tooly" can emit tool calls and "seer" can read images. A request that is not
pinned and needs one of those is answered by the model that has it; a pinned
one is answered by the model it names."""

from __future__ import annotations

import asyncio
import socket as _socket
import threading
import time

import pytest
import uvicorn

import localm.inference.http_server as hs


class _Engine:
    def __init__(self, name, *, images=False):
        self.display_name = name
        self.loaded = False
        self.supports_images = images
        self.can_be_multimodal = images
        self.last_finish_reason = "stop"
        self.unloading = False
        self.answered = 0

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def chat_stream(self, messages, **kw):
        self.answered += 1
        yield f"answered-by-{self.display_name}"

    def count_tokens(self, text):
        return 3

    def count_messages_tokens(self, messages):
        return 5

    def context_capacity(self):
        return {"plain": 4096, "tooly": 32768, "seer": 8192}.get(self.display_name, 4096)

    def validate_grammar(self, grammar, lazy=False):
        return None


def _wait(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A real server with "plain" loaded and "tooly"/"seer" installed.

    "seer" is vision-capable for real: a model file plus a recorded projector
    that exists on disk, which is what the shipped vision probe reads."""
    for n in ("plain", "tooly", "seer"):
        (tmp_path / n).mkdir()
        (tmp_path / n / f"{n}.gguf").write_bytes(b"GGUF")
    proj = tmp_path / "seer" / "seer-mmproj.gguf"
    proj.write_bytes(b"GGUF")
    registry = {
        "plain": {"path": str(tmp_path / "plain" / "plain.gguf"), "source": "local",
                  "model_type": "llm", "tool_use": False, "context_length": 4096},
        "tooly": {"path": str(tmp_path / "tooly" / "tooly.gguf"), "source": "local",
                  "model_type": "llm", "tool_use": True, "context_length": 32768},
        "seer": {"path": str(tmp_path / "seer" / "seer.gguf"), "source": "local",
                 "model_type": "llm", "tool_use": False, "context_length": 8192,
                 "mmproj": str(proj)},
    }
    engines: dict = {}

    def factory(name):
        return engines.setdefault(name, _Engine(name, images=(name == "seer")))

    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name, **kw: (registry[name]["path"], "hint")
                        if name in registry else None)
    monkeypatch.setattr("localm.model_manager.get_model_mmproj",
                        lambda name, **kw: str(proj) if name == "seer" else None)
    monkeypatch.setattr(hs, "_engine_factory", factory)
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._last_active_model_name = None
    hs._engine = None
    hs._inference_sem = None

    startup = factory("plain")
    startup.load()
    app = hs.create_app(startup)
    lsock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    lsock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    th = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[lsock])),
                          daemon=True)
    th.start()
    assert _wait(lambda: server.started), "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{port}/v1", engines, app
    finally:
        server.should_exit = True
        th.join(timeout=5.0)


def _answering(engines):
    return sorted(n for n, e in engines.items() if e.answered)


# --------------------------------------------------------------------------- #
#  The coder's HTTP backend                                                    #
# --------------------------------------------------------------------------- #

class TestCoderBackend:
    def _backend(self, base, *, pinned):
        from localm.plugins.coder.backends.http import HTTPBackend
        return HTTPBackend(base, model="plain", api_key="localm", localm_server=True,
                           model_pinned=pinned, required_capabilities=("tool_use",))

    def test_an_unpinned_session_is_answered_by_a_model_with_tool_calls(self, live):
        base, engines, _ = live
        be = self._backend(base, pinned=False)
        text = be.chat([{"role": "user", "content": "list files"}])
        assert _answering(engines) == ["tooly"]
        assert text == "answered-by-tooly"
        assert be.answered_model == "tooly"

    def test_streaming_is_routed_the_same_way(self, live):
        base, engines, _ = live
        be = self._backend(base, pinned=False)
        text = "".join(be.chat_stream([{"role": "user", "content": "list files"}]))
        assert text == "answered-by-tooly"
        assert _answering(engines) == ["tooly"]

    def test_a_pinned_session_is_answered_by_its_model(self, live):
        base, engines, _ = live
        be = self._backend(base, pinned=True)
        be.chat([{"role": "user", "content": "list files"}])
        assert _answering(engines) == ["plain"]
        assert be.answered_model == "plain"

    def test_the_context_budget_follows_the_model_that_answered(self, live):
        base, engines, _ = live
        be = self._backend(base, pinned=False)
        be.chat([{"role": "user", "content": "list files"}])
        assert be.context_capacity() == 32768

    def test_routing_leaves_the_loaded_model_loaded_for_everyone_else(self, live):
        base, engines, _ = live
        self._backend(base, pinned=False).chat([{"role": "user", "content": "x"}])
        assert hs._resolve_unnamed_model_name() == "plain"
