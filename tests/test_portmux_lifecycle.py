# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-process tests for portmux's server-lifecycle functions: ``run_server``,
``_serve_async`` and ``_serve_async_plain``, plus the real (non-faked) ``_relay``
round trip.

tests/test_portmux.py drives these in a SUBPROCESS (real uvicorn, real TLS),
which is valuable black-box evidence but registers no coverage in this process.
tests/test_portmux_redirect.py covers the pure/fast pieces with fake streams.
This module fills the remaining gap: the async lifecycle functions themselves,
started as real asyncio tasks against real ephemeral (port 0) sockets in THIS
process, so both the happy path and the "internal server never came up"
failure path are exercised for real.

The one deliberate substitution: on the PLAIN (no-TLS) path the internal
uvicorn always binds to a hardcoded ``127.0.0.1:0`` (an ephemeral loopback
port), which cannot practically be made to fail in a test - there is no
reachable way to make that bind raise without mocking something. So the
"internal server startup failed" branch is exercised via a minimal fake
``uvicorn.Server`` there. The TLS variant gets the SAME branch exercised for
real instead, via a genuinely bad certificate path - a real, reachable trust-
relevant failure mode (a corrupt/missing cert must not hang or silently fall
back to plaintext).
"""
from __future__ import annotations

import asyncio
import contextlib
import socket
import ssl

import pytest

from localm import bugreport as bugreport_mod
from localm import portmux, tls


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #

async def _tiny_asgi_app(scope, receive, send):
    """A minimal real ASGI app: implements lifespan (so uvicorn's startup
    completes cleanly) and answers every HTTP request with a fixed body and an
    explicit Content-Length (avoids chunked encoding, keeping response
    assertions simple)."""
    if scope["type"] == "lifespan":
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    elif scope["type"] == "http":
        body = b"lifecycle-ok"
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/plain"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _wait_connectable(port: int, timeout: float = 10.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    last = None
    while loop.time() < deadline:
        try:
            _, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            with contextlib.suppress(Exception):
                await w.wait_closed()
            return
        except OSError as e:
            last = e
            await asyncio.sleep(0.05)
    raise AssertionError(f"port {port} never became connectable: {last}")


async def _shutdown(task: asyncio.Task, timeout: float = 10.0) -> None:
    """Cancel a running server-lifecycle task and wait for its finally-block
    teardown (demux.close()/wait_closed(), wakeup_task cancel) to finish."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=timeout)


# --------------------------------------------------------------------------- #
#  _relay: real byte-for-byte round trip (both directions, through _pump)
# --------------------------------------------------------------------------- #

async def _make_echo_server():
    async def handle(reader, writer):
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()
    server = await asyncio.start_server(handle, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def test_relay_pumps_bytes_both_directions_byte_for_byte():
    # Proves the module docstring's trust claim - "TLS still terminates at
    # uvicorn; we only shuffle bytes" - by round-tripping arbitrary bytes
    # through a real internal listener and asserting the reply is byte-for-
    # byte identical: no header rewriting, no injected/dropped bytes.
    async def go():
        echo_server, internal_port = await _make_echo_server()

        async def on_client(reader, writer):
            first = await reader.readexactly(1)
            await portmux._relay(first, reader, writer, internal_port)
            writer.close()

        front = await asyncio.start_server(on_client, host="127.0.0.1", port=0)
        front_port = front.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", front_port)
            writer.write(b"hello-relay-roundtrip")
            await writer.drain()
            writer.write_eof()
            reply = await asyncio.wait_for(reader.read(-1), timeout=10)
            assert reply == b"hello-relay-roundtrip"
        finally:
            writer.close()
            front.close()
            await front.wait_closed()
            echo_server.close()
            await echo_server.wait_closed()
    asyncio.run(go())


# --------------------------------------------------------------------------- #
#  _serve_async_plain / _serve_async: real end-to-end lifecycle
# --------------------------------------------------------------------------- #

def test_serve_async_plain_relays_real_http_and_shuts_down_cleanly():
    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async_plain(_tiny_asgi_app, "127.0.0.1", port, "warning"))
        try:
            await _wait_connectable(port)
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.read(-1), timeout=10)
            assert data.startswith(b"HTTP/1.1 200")
            assert data.endswith(b"lifecycle-ok")
            writer.close()
        finally:
            await _shutdown(task)
        # Teardown must actually release the public port, not just return.
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
    asyncio.run(go())


def test_serve_async_tls_relays_a_real_handshake_and_shuts_down_cleanly(tmp_path):
    cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])
    ca = str(tls.ca_cert_path(tmp_path))

    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async(_tiny_asgi_app, "127.0.0.1", port, cert, key, "warning"))
        try:
            await _wait_connectable(port)
            ctx = ssl.create_default_context(cafile=ca)
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", port, ssl=ctx, server_hostname="127.0.0.1")
            writer.write(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.read(-1), timeout=10)
            assert data.startswith(b"HTTP/1.1 200")
            assert data.endswith(b"lifecycle-ok")
            writer.close()
        finally:
            await _shutdown(task)
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
    asyncio.run(go())


def test_serve_async_plain_cancels_an_inflight_connection_on_shutdown():
    """Regression for issue #963 / F2: asyncio.start_server()'s
    client_connected_cb creates a Task for each connection that nothing keeps
    a reference to, so a connection still blocked in _relay's pumps at
    shutdown was invisible to demux.wait_closed() (which only waits for the
    LISTENING socket, never for handler tasks already running) and was
    silently destroyed mid-flight instead of being closed ("Task was
    destroyed but it is pending!"). With the fix, shutdown cancels and awaits
    the tracked task, whose finally closes the writer - so the client sees a
    clean EOF as PART of shutdown, not eventually or never."""
    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async_plain(_tiny_asgi_app, "127.0.0.1", port, "warning"))
        await _wait_connectable(port)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            # One byte only, never the rest of a request line: the internal
            # uvicorn never gets a full HTTP request, so both pump directions
            # block on read() and the connection's task stays pending - the
            # exact state the bug describes.
            writer.write(b"G")
            await writer.drain()
            await asyncio.sleep(0.1)   # let accept + relay-connect land

            await _shutdown(task)

            # The server must have closed THIS connection as part of its OWN
            # shutdown - not left it for the OS to reap whenever the process
            # exits. A short timeout distinguishes "closed promptly by the
            # fix" from "never closed" (hangs until wait_for's own timeout -
            # a clear failure, not a slow pass).
            data = await asyncio.wait_for(reader.read(-1), timeout=2)
            assert data == b""
        finally:
            writer.close()
    asyncio.run(go())


def test_serve_async_tls_cancels_an_inflight_connection_on_shutdown(tmp_path):
    """Same regression, TLS variant - the bug report names only the plain
    path's _on_conn, but _serve_async wires an identical untracked callback
    and must not be the one left unfixed."""
    cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])
    ca = str(tls.ca_cert_path(tmp_path))

    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async(_tiny_asgi_app, "127.0.0.1", port, cert, key, "warning"))
        await _wait_connectable(port)
        ctx = ssl.create_default_context(cafile=ca)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", port, ssl=ctx, server_hostname="127.0.0.1")
        try:
            # A real completed TLS handshake, then a partial request line only
            # - the internal uvicorn never gets a full request, so the pumps
            # block and the connection task stays pending through shutdown.
            writer.write(b"G")
            await writer.drain()
            await asyncio.sleep(0.1)

            await _shutdown(task)

            data = await asyncio.wait_for(reader.read(-1), timeout=2)
            assert data == b""
        finally:
            writer.close()
    asyncio.run(go())


def test_serve_async_propagates_a_bad_tls_cert_instead_of_hanging(tmp_path):
    # Trust-relevant: a corrupt/missing certificate must surface loudly (the
    # server never comes up, the caller finds out immediately) rather than
    # hang forever or silently fall through to an unprotected bind.
    #
    # The outer wait_for(timeout=10) is a safety net so a regression can't hang
    # the suite, NOT the behavioural assertion: asyncio.TimeoutError (raised by
    # wait_for on expiry) is itself an Exception subclass, so a bare
    # `pytest.raises(Exception)` here would pass identically whether the cert
    # error propagated in milliseconds OR the code hung for the full 10s and
    # wait_for's own timeout fired instead - exactly the "hang forever"
    # regression this test exists to catch, silently swallowed. The isinstance
    # check below is what actually pins "failed fast", and it's a type check,
    # not a wall-clock one, so it stays meaningful under shared-box load.
    async def go():
        port = _free_port()
        missing_cert = str(tmp_path / "does-not-exist.crt")
        missing_key = str(tmp_path / "does-not-exist.key")
        with pytest.raises(Exception) as exc_info:
            await asyncio.wait_for(
                portmux._serve_async(_tiny_asgi_app, "127.0.0.1", port,
                                     missing_cert, missing_key, "warning"),
                timeout=10)
        assert not isinstance(exc_info.value, asyncio.TimeoutError), (
            "startup hung instead of failing fast on the bad cert "
            "(wait_for's own timeout fired, not a real startup error)")
    asyncio.run(go())


class _FailFastServer:
    """Stand-in for uvicorn.Server whose serve() fails before startup
    completes - simulates the internal loopback uvicorn never coming up. A
    REAL bind failure on 127.0.0.1:0 is not practically reproducible (an
    ephemeral loopback port essentially never collides or exhausts in a test
    run), so this narrow substitution is the pragmatic way to exercise
    portmux's OWN response to that failure: it must propagate the error, not
    hang or silently continue with no backend listening. See the module
    docstring for why the TLS variant gets a REAL failure instead."""
    def __init__(self, config):
        self.config = config
        self.started = False
        self.should_exit = False
        self.servers = []

    async def serve(self, sockets=None):
        raise RuntimeError("simulated internal-server startup failure")


def test_serve_async_plain_propagates_internal_server_startup_failure(monkeypatch):
    import uvicorn as uvicorn_mod
    monkeypatch.setattr(uvicorn_mod, "Server", _FailFastServer)

    async def go():
        port = _free_port()
        with pytest.raises(RuntimeError, match="simulated internal-server startup failure"):
            await portmux._serve_async_plain(_tiny_asgi_app, "127.0.0.1", port, "warning")
        # No demux must have been created off the back of a failed backend.
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
    asyncio.run(go())


# --------------------------------------------------------------------------- #
#  run_server: crash-guard wiring, instance_id extraction, failure handling
# --------------------------------------------------------------------------- #

def _patch_bugreport(monkeypatch):
    """Record calls to the crash-guard hooks without touching disk: run_server
    is tested here for its OWN wiring/ordering, not bugreport's own (already
    tested, hermetic-per-LOCALM_HOME) behaviour."""
    calls = []
    monkeypatch.setattr(bugreport_mod, "check_and_report_prior_crash",
                        lambda *a, **k: calls.append(("checked",)))
    monkeypatch.setattr(
        bugreport_mod, "arm_crash_guard",
        lambda context=None, home=None, instance_id=None:
            calls.append(("armed", context, instance_id)))
    monkeypatch.setattr(
        bugreport_mod, "disarm_crash_guard",
        lambda home=None, instance_id=None: calls.append(("disarmed", instance_id)))
    return calls


async def _bare_app(scope, receive, send):
    pass


def test_run_server_plain_wires_crash_guard_and_extracts_instance_id(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, log_level):
        return
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    class State:
        instance_id = "inst-abc"

    class App:
        state = State()

    portmux.run_server(App(), "0.0.0.0", 9999)

    assert calls[0] == ("checked",)
    assert calls[1] == ("armed", {"host": "0.0.0.0", "port": 9999, "tls": False}, "inst-abc")
    assert calls[2] == ("disarmed", "inst-abc")


def test_run_server_handles_a_bare_asgi_callable_with_no_state(monkeypatch):
    # A bare ASGI function (no .state) must degrade to instance_id=None
    # instead of raising AttributeError before the server ever binds.
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, log_level):
        return
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    portmux.run_server(_bare_app, "127.0.0.1", 8000)

    assert calls[1] == ("armed", {"host": "127.0.0.1", "port": 8000, "tls": False}, None)
    assert calls[2] == ("disarmed", None)


def test_run_server_plain_swallows_keyboard_interrupt(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, log_level):
        raise KeyboardInterrupt()
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    portmux.run_server(_bare_app, "127.0.0.1", 8001)   # must not raise
    assert calls[-1] == ("disarmed", None), "crash guard must still be disarmed"


def test_run_server_plain_falls_back_to_uvicorn_run_on_unexpected_error(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, log_level):
        raise RuntimeError("peek layer exploded")
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    def fail_socket(host, port):   # simulates create_listen_socket failing (#1517)
        raise OSError("simulated: cannot build the listening socket")
    monkeypatch.setattr(portmux, "create_listen_socket", fail_socket)

    import uvicorn as uvicorn_mod
    fallback_calls = []
    monkeypatch.setattr(uvicorn_mod, "run", lambda app, **kw: fallback_calls.append(kw))

    portmux.run_server(_bare_app, "127.0.0.1", 8002)   # must not raise
    assert fallback_calls == [{"host": "127.0.0.1", "port": 8002, "log_level": "warning"}]
    assert calls[-1] == ("disarmed", None)


def test_run_server_tls_swallows_keyboard_interrupt(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, ssl_certfile, ssl_keyfile, log_level):
        raise KeyboardInterrupt()
    monkeypatch.setattr(portmux, "_serve_async", fake_serve)

    portmux.run_server(_bare_app, "0.0.0.0", 8443,
                       ssl_certfile="cert.pem", ssl_keyfile="key.pem")   # must not raise
    assert calls[1][1]["tls"] is True
    assert calls[-1] == ("disarmed", None)


def test_run_server_tls_falls_back_to_uvicorn_run_on_unexpected_error(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, ssl_certfile, ssl_keyfile, log_level):
        raise RuntimeError("demux exploded")
    monkeypatch.setattr(portmux, "_serve_async", fake_serve)

    def fail_socket(host, port):   # simulates create_listen_socket failing (#1517)
        raise OSError("simulated: cannot build the listening socket")
    monkeypatch.setattr(portmux, "create_listen_socket", fail_socket)

    import uvicorn as uvicorn_mod
    fallback_calls = []
    monkeypatch.setattr(uvicorn_mod, "run", lambda app, **kw: fallback_calls.append(kw))

    portmux.run_server(_bare_app, "0.0.0.0", 8443,
                       ssl_certfile="cert.pem", ssl_keyfile="key.pem")   # must not raise
    assert fallback_calls == [{
        "host": "0.0.0.0", "port": 8443, "log_level": "warning",
        "ssl_certfile": "cert.pem", "ssl_keyfile": "key.pem",
    }]
    assert calls[-1] == ("disarmed", None)
