# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm rm` must not delete a model file a running server still has open.

`model_manager.remove_model` deletes the file outright when it lives in the
models dir. The MCP `remove_model` tool already guards this (see
`test_mcp_remove_model_hold.py`); `localm rm` did not, and it is the more
exposed gap - it is what the MCP tool itself calls, it is what the GUI's
remove route spawns as a child process, and it is what a user runs directly
from a terminal.

`localm rm` is a fresh, ONE-SHOT process. It never has an engine of its own to
check, so the only possible holder is a SEPARATE running localm server - a
different process this one shares no memory with. `remote_hold_reason`
(`localm/selfclient.py`) is the shared function that asks every discovered
instance over HTTP; this file proves `localm rm` actually calls it, in every
branch, and - the part a mock cannot prove - that it is asking a REAL server
over a REAL socket rather than trusting a candidate list that would always be
empty in this process.

WHY THESE TESTS ASSERT ON THE FILE FIRST. The property is "the bytes
survive", and the refusal message is a proxy for it. A runner stops at the
first failing assertion, so whichever comes first is the one that speaks:
lead with the file and a regression reports that a model file was deleted
while a server that could not be ruled out was running, which you cannot
talk yourself out of.

AND EVERY ARM PROVES ITS OWN FAULT INJECTION FIRED, same discipline as
test_mcp_remove_model_hold.py: a snapshot patch that did not take looks
exactly like a guard that checked and correctly found nothing to refuse.
"""

from __future__ import annotations

import asyncio
import json
import socket as _socket
import threading
import time

import pytest
import uvicorn
from click.testing import CliRunner

import localm.config as config
import localm.model_manager as model_manager
from localm.cli import main
from localm.inference.http_server import create_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway data dir every call site (load_registry, home_dir) resolves
    through. Mirrors test_mcp_remove_model_hold.py's fixture exactly: the two
    tests exercise the same underlying delete path and must agree on what
    "the data dir" means, or a passing test here would prove nothing about
    the real command.
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
    from localm import instances
    monkeypatch.setattr(instances, "snapshot", lambda *a, **kw: [])


def _servers(monkeypatch, rows):
    from localm import instances
    monkeypatch.setattr(instances, "snapshot", lambda *a, **kw: rows)


def _reader(monkeypatch, result):
    """Pin what read_model_file_hold answers, and count that it was consulted.

    Patched on localm.selfclient, which is where remote_hold_reason imports
    it from at call time.
    """
    calls = []

    def fake(scheme, port, model, token=None, bind_host=None):
        calls.append((scheme, port, model))
        return result

    import localm.selfclient as sc
    monkeypatch.setattr(sc, "read_model_file_hold", fake)
    return calls


def _rm(model, yes=True):
    args = ["rm", model] + (["--yes"] if yes else [])
    res = CliRunner().invoke(main, args)
    # rich console word-wraps long lines, so a substring split across the
    # wrap point would never match - collapse whitespace before returning.
    output = " ".join(res.output.split())
    return output, res.exit_code


ROW = {"scheme": "http", "host": "127.0.0.1", "port": 8123,
       "instance_id": "abc", "pid": 4242, "token": "t", "alive": True}


# --------------------------------------------------------------------------
#  Branch coverage (mocked read_model_file_hold / instances.snapshot)
# --------------------------------------------------------------------------

def test_no_server_running_is_not_a_refusal(home, monkeypatch):
    """The design constraint stated in the brief: absence of a server is a
    positive all-clear, not a refusal. Without this, `localm rm` would be
    useless on the common case of nothing running."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _no_servers(monkeypatch)

    output, code = _rm("victim")

    assert code == 0, output
    assert not path.exists(), "the removal was reported but the file survived"


def test_refuses_when_a_running_server_reports_a_hold(home, monkeypatch):
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, ("ok", {"held": True, "key": "served",
                                         "reason": None}))

    output, code = _rm("victim")

    assert path.exists(), (
        "a running server reported it had this model's file loaded and the "
        "file was deleted anyway")
    assert calls == [("http", 8123, "victim")], calls
    assert code != 0
    assert "Refusing to remove" in output
    assert "served" in output


@pytest.mark.parametrize("state,payload,expected_phrase", [
    ("unreachable", "ConnectionError", "could not be reached"),
    ("unauthorized", 401, "requires an API key"),
    ("unsupported", 404, "older localm"),
    ("http", 500, "answered HTTP 500"),
])
def test_refuses_when_a_server_cannot_be_asked(home, monkeypatch, state,
                                               payload, expected_phrase):
    """EVERY outcome that is not an answer is a refusal. A test that only
    covers the reachable case cannot tell a fail-closed design from a
    fail-open one."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, (state, payload))

    output, code = _rm("victim")

    assert path.exists(), (
        f"a server that could not be asked ({state}) was treated as having "
        f"ruled itself out, and the model file was deleted")
    assert calls, "the server was never actually asked"
    assert code != 0
    assert expected_phrase in output, output


def test_refuses_when_a_registered_server_did_not_answer_its_identity_check(
        home, monkeypatch):
    """alive=False is a failed /whoami, NOT proof the process is gone."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW, alive=False)])
    calls = _reader(monkeypatch, ("ok", {"held": False}))

    output, code = _rm("victim")

    assert path.exists(), (
        "an instance that failed its identity probe was treated as dead, "
        "and the model file was deleted while it may still have been "
        "serving")
    assert calls == [], (
        "a non-answering instance must not be asked and then believed - "
        "the refusal comes from it not answering at all")
    assert code != 0
    assert "did not answer an identity check" in output


def test_removes_when_the_only_server_positively_rules_itself_out(
        home, monkeypatch):
    """The permissive arm. Without it, a fail-closed guard that refuses
    everything would pass every test above while making the command
    useless."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, ("ok", {"held": False}))

    output, code = _rm("victim")

    assert code == 0, output
    assert calls, "the server was never asked"
    assert not path.exists(), "the removal was reported but the file survived"


def test_a_server_serving_a_different_library_is_not_a_holder(home, monkeypatch):
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, ("absent", 404))

    output, code = _rm("victim")

    assert code == 0, output
    assert calls, "the server was never asked"
    assert not path.exists()


def test_one_holding_server_outranks_another_that_ruled_itself_out(
        home, monkeypatch):
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

    output, code = _rm("victim")

    assert path.exists(), (
        "a second instance held the file and the first one's all-clear was "
        "taken for the whole answer")
    assert seen == [8123, 8124], seen
    assert code != 0


def test_unregistered_model_never_asks_a_server(home, monkeypatch):
    """The gate is `if model in reg`: an unregistered name deletes nothing
    either way (remove_model just reports "Not found"), so there is nothing
    to check and no round trip to make."""
    _register(home, {"other": {"path": str(_model_file(home)), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, ("ok", {"held": True, "key": "x", "reason": None}))

    output, code = _rm("nope")

    assert calls == [], (
        "an unregistered model has nothing to delete, and the remote check "
        "was consulted anyway")


def test_check_runs_even_without_yes_and_precedes_confirmation(home, monkeypatch):
    """The check must fire regardless of --yes: that flag is what the GUI's
    remove route passes to the child it spawns, and that child is exactly
    the process that must not skip this. It must also run BEFORE the
    interactive confirmation, or a held file would still prompt the user for
    something about to be refused anyway."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    _servers(monkeypatch, [dict(ROW)])
    calls = _reader(monkeypatch, ("ok", {"held": True, "key": "served",
                                         "reason": None}))

    output, code = _rm("victim", yes=False)

    assert path.exists()
    assert calls == [("http", 8123, "victim")], calls
    assert code != 0
    assert "Refusing to remove" in output
    # The confirmation prompt never rendered - the refusal preempted it.
    assert "Continue?" not in output


# --------------------------------------------------------------------------
#  Real HTTP: a real server, no liveness-check mock anywhere in the test.
#
# A unit test that monkeypatches read_model_file_hold (all of the above)
# proves the CLI's own branching is correct, but it cannot prove the CLI
# actually reaches across the process boundary rather than, say, silently
# using an empty local candidate list (always empty in a one-shot CLI
# process) and passing every time. These tests hit a REAL uvicorn server on
# a REAL socket, running the REAL /v1/models/{id}/hold route, computing its
# answer from that server's OWN in-process engine map - the two-process
# shape the brief calls for, with process discovery mocked (as
# test_cli_unload_stop_open_mode_auth.py's own RealHttp tier already does)
# but the wire hop and the server-side computation both real.
# --------------------------------------------------------------------------

class _FileEngine:
    def __init__(self, name, path, loaded=True):
        self.display_name = name
        self.model_path = str(path)
        self.loaded = loaded


def _wait_sync(cond, want=True, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(cond()) == want:
            return True
        time.sleep(0.02)
    return bool(cond()) == want


class _RealServer:
    def __init__(self, app, port, server, thread):
        self.app = app
        self.port = port
        self.server = server
        self.thread = thread

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10.0)


def _start_real_server() -> _RealServer:
    app = create_app(None)
    app.state.instance_token = "real-instance-token-0123456789"

    lsock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    lsock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]

    cfg = uvicorn.Config(app, log_level="warning", lifespan="on")
    server = uvicorn.Server(cfg)

    def _serve():
        asyncio.run(server.serve(sockets=[lsock]))

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    assert _wait_sync(lambda: server.started, True, 10.0), "uvicorn did not start"
    return _RealServer(app, port, server, thread)


def _put_resident(monkeypatch, name, path):
    """Put an engine in the real server's OWN map, the way a load does.

    Call AFTER the real server thread has started (its lifespan clears
    _engines on startup): an engine injected earlier is wiped by app
    construction, leaving exactly the state of a server with nothing
    loaded - a confident all-clear from the guard this test means to defeat.
    """
    import localm.inference.http_server as hs
    eng = _FileEngine(name, path)
    monkeypatch.setattr(hs, "_engines", {name: eng})
    monkeypatch.setattr(hs, "_engine", eng)
    assert hs._engines[name].loaded
    assert hs._engines[name].model_path == str(path)
    return eng


class TestRmRealHttp:
    def test_refuses_when_a_real_running_server_holds_the_file(
            self, home, monkeypatch):
        path = _model_file(home)
        _register(home, {"victim": {"path": str(path), "source": "test"}})
        rs = _start_real_server()
        try:
            _put_resident(monkeypatch, "victim", path)
            row = dict(ROW, port=rs.port, token=rs.app.state.instance_token)
            _servers(monkeypatch, [row])

            output, code = _rm("victim")
        finally:
            rs.stop()

        assert path.exists(), (
            "a REAL running server had this model's file loaded over a "
            "REAL socket, and the file was deleted anyway")
        assert code != 0
        assert "Refusing to remove" in output
        assert "victim" in output

    def test_removes_when_a_real_running_server_positively_rules_itself_out(
            self, home, monkeypatch):
        """The permissive arm on the wire. Without it, a client that read
        every real response as "could not be established" would satisfy the
        refusal test above while making `localm rm` unable to delete
        anything while any server is running.

        "The file is gone and the command exited 0" is ALSO what a bug that
        never contacted the server at all would look like (a silently broken
        discovery mock defaults to the same outcome as a real all-clear), so
        this test additionally spies on read_model_file_hold - forwarding to
        the real implementation, never replacing it - to prove the real
        socket was the thing that answered."""
        path = _model_file(home)
        _register(home, {"victim": {"path": str(path), "source": "test"}})
        rs = _start_real_server()
        real_reader = None
        calls = []

        def _spy(*a, **kw):
            calls.append((a, kw))
            return real_reader(*a, **kw)

        try:
            # Nothing resident: the real route's own guard finds no holder.
            row = dict(ROW, port=rs.port, token=rs.app.state.instance_token)
            _servers(monkeypatch, [row])
            import localm.selfclient as sc
            real_reader = sc.read_model_file_hold
            monkeypatch.setattr(sc, "read_model_file_hold", _spy)

            output, code = _rm("victim")
        finally:
            rs.stop()

        assert calls and calls[0][0][:2] == ("http", rs.port), (
            "the real server was never actually asked - a silently broken "
            "discovery mock would produce this exact same all-clear")
        assert code == 0, output
        assert not path.exists(), (
            "the removal was reported but the file survived")

    def test_refuses_when_the_registered_port_is_dead(self, home, monkeypatch):
        """The fail-closed arm on a REAL socket, exercised through the real
        CLI command: a registered instance whose port nothing is listening
        on any more must read as "could not be established", never as an
        all-clear. Bind a socket to claim a port, close it, then point the
        CLI at it - so the port is genuinely one nothing answers on, not
        one guessed at."""
        path = _model_file(home)
        _register(home, {"victim": {"path": str(path), "source": "test"}})
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
        s.close()
        row = dict(ROW, port=dead_port, token="whatever")
        _servers(monkeypatch, [row])

        output, code = _rm("victim")

        assert path.exists(), (
            "a registered instance on a dead port was treated as having "
            "ruled itself out, and the model file was deleted")
        assert code != 0
        assert "could not be reached" in output, output
