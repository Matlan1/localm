# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory recall must be OBSERVABLE:
- the store reports WHY semantic recall degraded to lexical (a degrade reason,
  mirroring RAG's lexical fallback), plus record/vector/recall counts;
- the chat inlet stashes the injected records + degrade reason in ctx.state so a
  response header can surface a "used N memories" affordance;
- the http header helper renders that ctx.state into an X-Localm-Memory header.
"""

from __future__ import annotations

import json

import pytest

from localm.inference.chat_pipeline import ChatHookContext
from localm.memory import MemoryRecord, MemoryStore
from localm.memory.store import TINY_CORPUS, VEC_COVERAGE

_MEASURE = ("metric", "unit", "measure", "measurement", "system")


def _fake_embed(texts):
    """Deterministic topic vectors (measurement words load axis 0)."""
    out = []
    for t in texts:
        lo = t.lower()
        out.append([1.0 if any(w in lo for w in _MEASURE) else 0.05, 0.3])
    return out


# --------------------------------------------------- store diagnostics ------ #

def test_recall_diagnostics_no_embedder(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(TINY_CORPUS + 2):
        s.add(MemoryRecord(text=f"fact {i} about metric units", source="user"),
              embed_fn=None)
    diag: dict = {}
    hits = s.recall("units", k=3, embed_fn=None, diagnostics=diag)
    assert diag["degrade_reason"] == "no_embedder"
    assert diag["n_records"] == TINY_CORPUS + 2
    assert diag["n_vectors"] == 0
    assert diag["n_recalled"] == len(hits) >= 1


def test_recall_diagnostics_no_vectors(tmp_path):
    # An embedder is available at query time, but the records were stored before
    # 'setup-embeddings' (no vectors yet) -> semantic recall still degraded.
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(TINY_CORPUS + 2):
        s.add(MemoryRecord(text=f"fact {i} about metric units", source="user"),
              embed_fn=None)
    diag: dict = {}
    s.recall("units", k=3, embed_fn=_fake_embed, diagnostics=diag)
    assert diag["degrade_reason"] == "no_vectors"


def test_recall_diagnostics_vector_used(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(TINY_CORPUS + 2):
        s.add(MemoryRecord(text=f"fact {i} about metric measurement units",
                           source="user"), embed_fn=_fake_embed)
    diag: dict = {}
    s.recall("what measurement system", k=3, embed_fn=_fake_embed, diagnostics=diag)
    assert diag["degrade_reason"] is None            # semantic active
    assert diag["n_vectors"] == TINY_CORPUS + 2


def test_recall_diagnostics_low_coverage(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    n = TINY_CORPUS + 2
    for i in range(n):
        # Embed only the first record so coverage stays well below VEC_COVERAGE.
        s.add(MemoryRecord(text=f"fact {i} about metric units", source="user"),
              embed_fn=(_fake_embed if i == 0 else None))
    assert len(s._vectors) / n < VEC_COVERAGE
    diag: dict = {}
    s.recall("units", k=3, embed_fn=_fake_embed, diagnostics=diag)
    assert diag["degrade_reason"] == "low_coverage"


def test_recall_diagnostics_optional(tmp_path):
    # diagnostics defaults to None: recall behaviour is unchanged when omitted.
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User prefers metric units", source="user"))
    hits = s.recall("units", k=3, embed_fn=None)     # no diagnostics arg
    assert hits and hits[0].text == "User prefers metric units"


# --------------------------------------------------- inlet stashing --------- #

def test_inlet_stashes_memory_used(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")
    from localm.plugins.builtin.memory import plug
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)      # no embedder

    store = plug._chat_store(None)
    rec = store.add(MemoryRecord(text="User prefers metric units", source="user"))

    ctx = ChatHookContext(model_id="m", stream=False, request_id="r")
    messages = [{"role": "user", "content": "what units do I prefer"}]
    out = plug._memory_inlet(messages, ctx)

    assert out is not None                                    # something injected
    used = ctx.state.get("memory_used")
    assert used, "inlet did not stash memory_used"
    assert used[0]["id"] == rec.id
    assert used[0]["text"].startswith("User prefers metric")
    assert used[0]["source"] == "user"
    assert ctx.state.get("memory_degrade_reason") == "no_embedder"


def test_inlet_no_stash_when_recall_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")
    from localm.plugins.builtin.memory import plug
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_recall_enabled", lambda: False)
    ctx = ChatHookContext(model_id="m", stream=False, request_id="r")
    plug._memory_inlet([{"role": "user", "content": "hi"}], ctx)
    assert "memory_used" not in ctx.state


# --------------------------------------------------- header helper ---------- #

def test_memory_used_header_from_ctx():
    from localm.inference.http_server import _memory_used_header

    ctx = ChatHookContext(model_id="m", stream=False, request_id="r")
    assert _memory_used_header(ctx) == {}                     # nothing stashed

    ctx.state["memory_used"] = [
        {"id": "abc123", "text": "User prefers metric units",
         "source": "user", "kind": "semantic"}]
    ctx.state["memory_degrade_reason"] = "no_embedder"
    h = _memory_used_header(ctx)
    payload = json.loads(h["X-Localm-Memory"])
    assert payload["n"] == 1
    assert payload["degrade"] == "no_embedder"
    assert payload["items"][0]["id"] == "abc123"


def test_memory_used_header_empty_list_still_reports():
    # Recall ran but injected nothing (empty store / no match): the header still
    # reports n=0 plus the degrade reason.
    from localm.inference.http_server import _memory_used_header

    ctx = ChatHookContext(model_id="m", stream=False, request_id="r")
    ctx.state["memory_used"] = []
    ctx.state["memory_degrade_reason"] = "no_embedder"
    h = _memory_used_header(ctx)
    payload = json.loads(h["X-Localm-Memory"])
    assert payload["n"] == 0
    assert payload["degrade"] == "no_embedder"


# --------------------------------------------------- route wiring ---------- #
# The real /v1/chat/completions handler attaches the X-Localm-Memory header from
# ctx.state (built by create_app plus the memory inlet). Drives the real route
# with a stub engine and a synthetic inlet that stashes memory_used exactly like
# the memory plugin does, streaming and non-streaming.

import textwrap  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _echo_engine():
    from unittest.mock import MagicMock
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5

    def _stream(messages, **kw):
        last = messages[-1].get("content", "") if messages else ""
        return iter([last if isinstance(last, str) else ""])

    engine.chat_stream.side_effect = _stream
    type(engine).loaded = property(lambda self: True)
    engine.supports_images = True
    return engine


def _make_plugin(root, name, body):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


def _install(app, plugins_root, name):
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(app, external_root=plugins_root, builtin_root=None)
    mgr.discover()
    mgr.install(name)
    return mgr


MEMSTASH_PLUGIN = '''
    def _inlet(messages, ctx):
        ctx.state["memory_used"] = [
            {"id": "rec1", "text": "User prefers metric units",
             "source": "user", "kind": "semantic"}]
        ctx.state["memory_degrade_reason"] = "no_embedder"
        return messages

    def register(host):
        host.register_chat_hook("inlet", _inlet)

    def unregister():
        pass
'''


def test_chat_route_attaches_memory_header_non_streaming(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    _make_plugin(env / "plugins", "memstash", MEMSTASH_PLUGIN)
    _install(app, env / "plugins", "memstash")
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions",
                   json={"model": "m",
                         "messages": [{"role": "user", "content": "units?"}]})
    assert r.status_code == 200
    blob = r.headers.get("X-Localm-Memory")
    assert blob, "route did not attach the X-Localm-Memory header"
    payload = json.loads(blob)
    assert payload["n"] == 1
    assert payload["degrade"] == "no_embedder"
    assert payload["items"][0]["id"] == "rec1"


def test_chat_route_attaches_memory_header_streaming(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    _make_plugin(env / "plugins", "memstash", MEMSTASH_PLUGIN)
    _install(app, env / "plugins", "memstash")
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions",
                   json={"model": "m", "stream": True,
                         "messages": [{"role": "user", "content": "units?"}]})
    assert r.status_code == 200
    blob = r.headers.get("X-Localm-Memory")
    assert blob, "streaming route did not attach the X-Localm-Memory header"
    assert json.loads(blob)["n"] == 1


def test_chat_route_no_header_without_memory(env):
    # No memory plugin installed -> nothing stashed -> no header (not an empty one).
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions",
                   json={"model": "m",
                         "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.headers.get("X-Localm-Memory") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
