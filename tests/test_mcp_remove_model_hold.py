# SPDX-License-Identifier: AGPL-3.0-or-later
"""The MCP `remove_model` tool must not delete a model file something is using.

`model_manager.remove_model` is the same code path `localm rm` runs: it deletes
the file outright when that file lives in the models dir.

TWO HOLDERS, and each gets its own arm here:

* THIS process. The MCP server keeps its own ``EngineCache``, so `chat` and
  `embed` leave a model resident right here.
* A running localm SERVER. A separate process sharing no memory with this one,
  so the only way to find out is to ask it over HTTP.

Every arm asserts on the FILE before it asserts on the refusal message, and
asserts that its own injected state is in effect before it asserts anything
about the outcome.
"""

from __future__ import annotations

import json

import pytest

import localm.config as config
import localm.model_manager as model_manager
from localm.plugins.mcpserver.server import EngineCache, build_tools


class _FileEngine:
    """An engine holding a real file, the way a loaded one does."""

    def __init__(self, name, path, loaded=True):
        self.display_name = name
        self.model_path = str(path) if path is not None else path
        self.loaded = loaded


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway data dir every call site resolves through.

    model_manager.MODELS_DIR is what is_owned_model_path (and therefore
    resolve_deletion_target) reads; config.MODELS_DIR is pinned to the same
    directory so nothing resolves against the session's real home.
    """
    h = tmp_path / ".localm"
    models = h / "models"
    models.mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(model_manager, "MODELS_DIR", models)
    monkeypatch.setattr(config, "HOME_DIR", h)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(config, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(config, "home_dir", lambda: h, raising=False)
    return h


def _model_file(home, filename="m.gguf"):
    p = home / "models" / filename
    p.write_bytes(b"GGUF" + b"\0" * 64)
    return p


def _register(home, entries):
    (home / "registry.json").write_text(json.dumps(entries), encoding="utf-8")


def _no_servers(monkeypatch):
    """No localm server running: the cross-process arm has nothing to ask."""
    from localm import instances
    monkeypatch.setattr(instances, "snapshot", lambda *a, **kw: [])


def _servers(monkeypatch, rows):
    from localm import instances
    monkeypatch.setattr(instances, "snapshot", lambda *a, **kw: rows)


def _reader(monkeypatch, result):
    """Pin what read_model_file_hold answers, and count that it was consulted.

    Patched on localm.selfclient, which is where _remote_hold imports it from
    at call time.
    """
    calls = []

    def fake(scheme, port, model, token=None, bind_host=None):
        calls.append((scheme, port, model))
        return result

    import localm.selfclient as sc
    monkeypatch.setattr(sc, "read_model_file_hold", fake)
    return calls


def _tools(engines=None):
    return build_tools(engines or EngineCache(default_model=None),
                       enable_images=False, enable_coder=False)


def _remove(tools, model):
    out = tools["remove_model"]["handler"]({"model": model})
    return out["content"][0]["text"], bool(out.get("isError"))


ROW = {"scheme": "http", "host": "127.0.0.1", "port": 8123,
       "instance_id": "abc", "pid": 4242, "token": "t", "alive": True}


# --------------------------------------------------------------------------
#  Holder 1: an engine resident in THIS process
# --------------------------------------------------------------------------

def test_refuses_when_this_process_holds_the_file(home, monkeypatch):
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _no_servers(monkeypatch)

    engines = EngineCache(default_model=None)
    engines._engines["victim"] = _FileEngine("victim", path)
    engines._lru.append("victim")
    # The injection took: a loaded engine whose recorded path IS this file.
    assert engines._engines["victim"].loaded
    assert engines._engines["victim"].model_path == str(path)

    text, is_error = _remove(_tools(engines), "victim")

    assert path.exists(), (
        "the model file was DELETED while an engine in this very process had "
        "it loaded - the download is gone")
    assert is_error
    assert "Refusing to remove" in text
    assert "victim" in text


def test_refuses_after_a_cli_rename_left_the_engine_keyed_on_the_old_name(
        home, monkeypatch):
    """Identity is by FILE PATH, not by name.

    `localm rename` runs in a separate process and cannot re-key this one's
    engine map, so the engine stays keyed under the OLD name while the registry
    carries only the new one.
    """
    path = _model_file(home)
    _register(home, {"new-name": {"path": str(path), "source": "test"}})
    _no_servers(monkeypatch)

    engines = EngineCache(default_model=None)
    engines._engines["old-name"] = _FileEngine("old-name", path)
    engines._lru.append("old-name")
    # The injection took: the engine's key is absent from the registry the
    # tool reads.
    assert "old-name" not in json.loads(
        (home / "registry.json").read_text(encoding="utf-8"))
    assert engines._engines["old-name"].model_path == str(path)

    text, is_error = _remove(_tools(engines), "new-name")

    assert path.exists(), (
        "a CLI rename left the engine keyed on the old name, a name-keyed "
        "guard missed it, and the live model's file was deleted")
    assert is_error
    assert "old-name" in text


def test_refuses_when_a_resident_engine_has_no_recorded_path(home, monkeypatch):
    """An engine that cannot say what it holds is an UNKNOWN, and unknown
    refuses."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _no_servers(monkeypatch)

    engines = EngineCache(default_model=None)
    engines._engines["mystery"] = _FileEngine("mystery", None)
    engines._lru.append("mystery")
    assert engines._engines["mystery"].loaded
    assert not engines._engines["mystery"].model_path

    text, is_error = _remove(_tools(engines), "victim")

    assert path.exists(), (
        "an engine that could not say which file it holds was treated as "
        "holding nothing, and the model file was deleted")
    assert is_error
    # The two refusals read differently: unreadable path, not in use.
    assert "cannot be ruled out" in text


def test_an_unloaded_engine_does_not_block_removal(home, monkeypatch):
    """The refusal is scoped, not blanket: a cached-but-unloaded engine holds
    no file, so the removal goes ahead."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _no_servers(monkeypatch)

    engines = EngineCache(default_model=None)
    engines._engines["victim"] = _FileEngine("victim", path, loaded=False)
    engines._lru.append("victim")
    assert not engines._engines["victim"].loaded

    text, is_error = _remove(_tools(engines), "victim")

    assert not is_error, text
    assert not path.exists(), "the removal was reported but the file survived"


def test_an_alias_keeps_the_bytes_so_there_is_nothing_to_refuse(home, monkeypatch):
    """While another registered name still points at the file, removal drops
    the NAME and keeps the bytes, so a loaded engine is not a reason to
    refuse."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"},
                     "other": {"path": str(path), "source": "test"}})
    _no_servers(monkeypatch)

    engines = EngineCache(default_model=None)
    engines._engines["victim"] = _FileEngine("victim", path)
    engines._lru.append("victim")
    assert engines._engines["victim"].loaded

    text, is_error = _remove(_tools(engines), "victim")

    assert not is_error, text
    assert path.exists(), "an aliased file must survive the removal of one name"


# --------------------------------------------------------------------------
#  Holder 2: a running localm server, in another process
# --------------------------------------------------------------------------

def test_refuses_when_a_running_server_reports_a_hold(home, monkeypatch):
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, ("ok", {"held": True, "key": "served",
                                         "reason": None}))

    text, is_error = _remove(_tools(), "victim")

    assert path.exists(), (
        "a running server reported it had this model's file loaded and the "
        "file was deleted anyway")
    # The injection took: the server really was asked about THIS model.
    assert calls == [("http", 8123, "victim")], calls
    assert is_error
    assert "served" in text


@pytest.mark.parametrize("state,payload,expected_phrase", [
    ("unreachable", "ConnectionError", "could not be reached"),
    ("unauthorized", 401, "requires an API key"),
    ("unsupported", 404, "older localm"),
    ("http", 500, "answered HTTP 500"),
])
def test_refuses_when_a_server_cannot_be_asked(home, monkeypatch, state,
                                               payload, expected_phrase):
    """EVERY outcome that is not an answer is a refusal, and the message names
    which one it was."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, (state, payload))

    text, is_error = _remove(_tools(), "victim")

    assert path.exists(), (
        f"a server that could not be asked ({state}) was treated as having "
        f"ruled itself out, and the model file was deleted")
    assert calls, "the server was never actually asked"
    assert is_error
    # The message names WHICH state, not merely that it refused.
    assert expected_phrase in text, text


def test_refuses_when_a_registered_server_did_not_answer_its_identity_check(
        home, monkeypatch):
    """alive=False is a failed /whoami, NOT proof the process is gone.

    snapshot() reaps entries whose pid has died by the time this runs, so a
    listed instance that did not answer is a live process of unknown state.
    """
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW, alive=False)])
    calls = _reader(monkeypatch, ("ok", {"held": False}))

    text, is_error = _remove(_tools(), "victim")

    assert path.exists(), (
        "an instance that failed its identity probe was treated as dead, and "
        "the model file was deleted while it may still have been serving")
    assert calls == [], (
        "a non-answering instance must not be asked and then believed - the "
        "refusal comes from it not answering at all")
    assert is_error
    assert "did not answer an identity check" in text


def test_removes_when_the_only_server_positively_rules_itself_out(home, monkeypatch):
    """The permissive arm: a server that positively rules itself out lets the
    removal through."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, ("ok", {"held": False}))

    text, is_error = _remove(_tools(), "victim")

    assert not is_error, text
    assert calls, "the server was never asked"
    assert not path.exists(), "the removal was reported but the file survived"


def test_a_server_serving_a_different_library_is_not_a_holder(home, monkeypatch):
    """A 404 with a model-not-registered body means that instance does not
    carry this model at all, which is separate from residency."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, ("absent", 404))

    text, is_error = _remove(_tools(), "victim")

    assert not is_error, text
    assert calls, "the server was never asked"
    assert not path.exists()


def test_one_holding_server_outranks_another_that_ruled_itself_out(home, monkeypatch):
    """Two instances, one clean and one holding: the clean answer does not
    short-circuit the loop into an all-clear."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW, port=8123), dict(ROW, port=8124)])

    seen = []

    def fake(scheme, port, model, token=None, bind_host=None):
        seen.append(port)
        if port == 8123:
            return "ok", {"held": False}
        return "ok", {"held": True, "key": "served", "reason": None}

    import localm.selfclient as sc
    monkeypatch.setattr(sc, "read_model_file_hold", fake)

    text, is_error = _remove(_tools(), "victim")

    assert path.exists(), (
        "a second instance held the file and the first one's all-clear was "
        "taken for the whole answer")
    assert seen == [8123, 8124], seen
    assert is_error
